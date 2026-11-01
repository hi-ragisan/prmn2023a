import numpy as np
import pandas as pd
import pytest

from banei_ai import backtest, features, market, schema
from banei_ai.clogit import ConditionalLogit, harville_top3
from banei_ai.simulate import SimConfig, simulate


@pytest.fixture(scope="module")
def sim():
    return simulate(SimConfig(start="2023-04-01", end="2024-06-30", seed=1))


def test_parse_time():
    assert schema.parse_time_to_sec("1:45.3") == pytest.approx(105.3)
    assert schema.parse_time_to_sec("2分01秒5") == pytest.approx(121.5)
    assert schema.parse_time_to_sec("98.7") == pytest.approx(98.7)
    assert np.isnan(schema.parse_time_to_sec("-"))


def test_clogit_recovers_coefficients():
    rng = np.random.default_rng(0)
    beta = np.array([1.0, -0.5])
    rows = []
    for r in range(3000):
        n = rng.integers(6, 11)
        X = rng.normal(size=(n, 2))
        u = X @ beta
        p = np.exp(u) / np.exp(u).sum()
        w = rng.choice(n, p=p)
        for i in range(n):
            rows.append((r, X[i, 0], X[i, 1], int(i == w)))
    d = pd.DataFrame(rows, columns=["race", "a", "b", "win"])
    m = ConditionalLogit(l2=0).fit(d[["a", "b"]], d["win"], d["race"])
    est = m.coef_ / m.scale_
    assert est == pytest.approx(beta, abs=0.1)
    p = m.predict_proba(d[["a", "b"]], d["race"])
    sums = pd.Series(p).groupby(d["race"]).sum()
    assert np.allclose(sums, 1.0)


def test_harville_sums_to_three():
    p = np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
    t3 = harville_top3(p)
    assert t3.sum() == pytest.approx(3.0)
    assert np.all(np.diff(t3) <= 1e-12)  # 勝率順に並ぶ
    assert np.all(harville_top3(np.array([0.5, 0.3, 0.2])) == 1.0)


def test_market_probs_sum_to_one(sim):
    d = market.add_market_probs(schema.normalize(sim))
    s = d.groupby("race_id")["p_mkt"].sum()
    assert np.allclose(s, 1.0)


def test_no_future_leakage(sim):
    """ある日の結果を書き換えても、その日以前の特徴量は変わらない."""
    days = sorted(sim["date"].unique())
    cut = days[len(days) // 2]
    base = features.build_features(sim)
    mod = sim.copy()
    m = mod["date"] >= cut
    # 当日以降の着順を逆順にする
    mod.loc[m, "finish"] = mod[m].groupby("race_id")["finish"].transform(lambda s: s.max() + 1 - s)
    mod.loc[m, "odds_win"] = mod.loc[m, "odds_win"][::-1].to_numpy()
    alt = features.build_features(mod)
    cols = [c for c in features.FUNDAMENTAL_FEATURES if c in base]
    key = ["race_id", "post"]
    a = base[base["date"] <= cut].set_index(key)[cols].sort_index()
    b = alt[alt["date"] <= cut].set_index(key)[cols].sort_index()
    pd.testing.assert_frame_equal(a, b)
    # 翌開催日以降は変わる (状態更新が効いている)。騎手は毎開催乗るので翌日で確認
    nxt = days[len(days) // 2 + 1]
    a2 = base[base["date"] == nxt].set_index(key)["j_win_ae"].sort_index()
    b2 = alt[alt["date"] == nxt].set_index(key)["j_win_ae"].sort_index()
    assert not np.allclose(a2, b2)
    later = days[len(days) // 2 + 12]
    a3 = base[base["date"] == later].set_index(key)["h_rating"].sort_index()
    b3 = alt[alt["date"] == later].set_index(key)["h_rating"].sort_index()
    assert not np.allclose(a3, b3)


def test_entries_do_not_update_state(sim):
    """出馬表 (status=entry) は結果として扱われない."""
    last = sim["date"].max()
    card = sim[sim["date"] == last].copy()
    card["status"] = "entry"
    card["finish"] = np.nan
    hist = sim[sim["date"] < last]
    X = features.build_features(pd.concat([hist, card]))
    e = X[X["status"] == "entry"]
    assert len(e) == len(card)
    assert e["finish_eff"].isna().all()
    assert e["h_rating"].notna().all()


def test_bet_rule_selection():
    P = pd.DataFrame({
        "race_id": ["a"] * 3 + ["b"] * 3,
        "date": pd.Timestamp("2024-01-01"),
        "odds_win": [2.0, 5.0, 30.0, 3.0, 4.0, 100.0],
        "p_mkt": [0.4, 0.16, 0.03, 0.27, 0.2, 0.01],
        "p_model": [0.45, 0.25, 0.02, 0.2, 0.3, 0.02],
        "win": [0, 1, 0, 0, 1, 0],
    })
    b = backtest.BetRule(min_ev=1.1, max_odds=50).select(P)
    # a: 5.0*0.25=1.25 ✓, 2.0*0.45=0.9 ×; b: 4.0*0.3=1.2 ✓, 100倍は上限超え
    assert sorted(zip(b["race_id"], b["odds_win"])) == [("a", 5.0), ("b", 4.0)]
    r = backtest.flat_bet(b)
    assert r["roi"] == pytest.approx((5.0 + 4.0) / 2)


def test_partial_odds_race_sums_to_one():
    """一部の馬にオッズが無いレースでも確率の合計は 1."""
    from banei_ai.model import Stage2

    rng = np.random.default_rng(0)
    rows = []
    for r in range(300):
        n = 8
        s1 = rng.normal(size=n)
        p = np.exp(s1) / np.exp(s1).sum()
        w = rng.choice(n, p=p)
        for i in range(n):
            rows.append({"race_id": f"r{r}", "s1": s1[i], "p_mkt": p[i], "log_p_mkt": np.log(p[i]),
                         "win": int(i == w), "j_popularity": 1.0, "h_last_pmkt_share": 1.0})
    d = pd.DataFrame(rows)
    s2 = Stage2().fit(d)
    test = d[d["race_id"] == "r0"].copy()
    test.loc[test.index[0], ["p_mkt", "log_p_mkt"]] = np.nan
    out = s2.predict(test)
    assert out["p_model"].sum() == pytest.approx(1.0)
