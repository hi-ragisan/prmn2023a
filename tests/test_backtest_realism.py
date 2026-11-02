"""バックテストの現実性 (自分の購入によるオッズ低下・締切前オッズのズレ) のテスト."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from banei_ai import backtest as B
from banei_ai.model import Stage2


# ---------------------------------------------------------------------------
# 従来 (変更前) の実装。pool=0 / σ=0 でこれと完全に一致することを確かめる
# ---------------------------------------------------------------------------

def _flat_bet_ref(bets, stake=100.0):
    b = bets.sort_values(["date", "race_id"])
    ret = b["win"] * b["odds_win"] * stake
    pnl = ret - stake
    cum = pnl.cumsum().to_numpy()
    return {"bets": len(b), "hits": int(b["win"].sum()), "hit_rate": float(b["win"].mean()),
            "roi": float(ret.sum() / (stake * len(b))), "profit": float(pnl.sum()),
            "max_dd": B._max_drawdown(cum), "avg_odds": float(b["odds_win"].mean())}


def _kelly_bet_ref(bets, p_col="p_model", bankroll=100_000.0, fraction=0.25, cap=0.03):
    b = bets.sort_values(["date", "race_id"])
    bk = bankroll
    peak = bk
    max_dd = 0.0
    for _, race in b.groupby(["date", "race_id"], sort=False):
        base = bk
        pnl = 0.0
        for r in race.itertuples(index=False):
            p, o = getattr(r, p_col), r.odds_win
            f = fraction * (p * o - 1) / (o - 1) if o > 1 else 0.0
            stake = np.floor(min(max(f, 0), cap) * base / 100) * 100
            pnl += stake * (o * r.win - 1)
        bk += pnl
        peak = max(peak, bk)
        max_dd = max(max_dd, (peak - bk) / peak)
    return {"final_bankroll": bk, "growth": bk / bankroll, "max_dd_pct": max_dd}


# ---------------------------------------------------------------------------
# 合成データ: 本物の Stage2 を学習させ、ウォークフォワードの出力と同じ形の表を作る
# ---------------------------------------------------------------------------

def _synthetic(n_races=500, n=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    days = pd.date_range("2024-01-06", periods=n_races // 10 + 1, freq="7D")
    for r in range(n_races):
        talent = rng.normal(0, 1, n)
        q = np.exp(0.9 * talent + rng.normal(0, 0.5, n))
        q /= q.sum()
        odds = np.maximum(1.0, np.floor(0.8 / q * 10) / 10)
        true = np.exp(1.0 * talent)
        true /= true.sum()
        w = np.zeros(n)
        w[rng.choice(n, p=true)] = 1
        last = rng.uniform(0.02, 0.4, n)
        last[rng.random(n) < 0.15] = np.nan
        for i in range(n):
            rows.append({"race_id": f"R{r:04d}", "date": days[r // 10], "post": i + 1, "odds_win": odds[i],
                         "win": w[i], "s1": talent[i] + rng.normal(0, 0.6),
                         "j_popularity": rng.normal(1, 0.2), "h_last_pmkt_share": last[i]})
    d = pd.DataFrame(rows)
    inv = 1 / d["odds_win"]
    d["p_mkt"] = inv / inv.groupby(d["race_id"]).transform("sum")
    d["log_p_mkt"] = np.log(d["p_mkt"].clip(lower=1e-4))
    return d


@pytest.fixture(scope="module")
def fitted():
    d = _synthetic()
    tr = d[d["race_id"] < "R0300"]
    te = d[d["race_id"] >= "R0300"].copy()
    s2 = Stage2().fit(tr)
    te["block"] = np.where(te["race_id"] < "R0400", 1, 2)
    P = pd.concat([te, s2.predict(te)], axis=1).reset_index(drop=True)
    return s2, P


# ---------------------------------------------------------------------------
# 自分の購入によるオッズ低下
# ---------------------------------------------------------------------------

def test_impact_formula_matches_hand_computation():
    T, R = 1_000_000.0, 0.8
    # 2 頭に 1 万円ずつ (同じレース → どちらも総売上に入る)
    eff = B.impact_odds([4.0, 10.0], [10_000, 10_000], ["a", "a"], T, R)
    W1, W2 = R * T / 4.05, R * T / 10.05
    o1 = R * (T + 20_000) / (W1 + 10_000)   # 3.93...
    o2 = R * (T + 20_000) / (W2 + 10_000)   # 9.10...
    assert np.allclose(eff, [np.floor(o1 * 10) / 10, np.floor(o2 * 10) / 10])
    assert np.allclose(eff, [3.9, 9.1])
    # 別レースなら総売上には自分の 1 点分しか入らない
    eff2 = B.impact_odds([10.0, 10.0], [10_000, 10_000], ["a", "b"], T, R)
    o = R * (T + 10_000) / (R * T / 10.05 + 10_000)
    assert np.allclose(eff2, np.floor(o * 10) / 10)
    # 元返し (1.0) を下回らない
    assert B.impact_odds([1.1], [5_000_000], ["a"], T, R)[0] == 1.0
    # プール 0 = 影響なし
    assert np.array_equal(B.impact_odds([4.0, 10.0], 10_000, ["a", "a"], 0.0, R), [4.0, 10.0])


def test_impact_zero_stake_reproduces_displayed_odds():
    odds = np.round(np.arange(1.0, 300.0, 0.1), 1)
    for T in (300_000.0, 2_500_000.0, 1e8):
        eff = B.impact_odds(odds, 0.0, np.arange(len(odds)), T)
        assert np.array_equal(eff, odds)


def test_bigger_stake_lowers_effective_odds():
    stakes = np.array([0, 100, 1_000, 5_000, 10_000, 50_000, 200_000, 1_000_000], dtype=float)
    for o in (1.5, 3.0, 8.0, 25.0, 80.0):
        eff = np.array([B.impact_odds([o], s, ["a"], 1_000_000.0)[0] for s in stakes])
        assert np.all(np.diff(eff) <= 0)
        assert eff[-1] < o
    # 同じ賭け金なら、プールが小さいほど低下が大きい
    small = B.impact_odds([10.0], 10_000, ["a"], 500_000.0)[0]
    large = B.impact_odds([10.0], 10_000, ["a"], 5_000_000.0)[0]
    assert small < large <= 10.0


def test_flat_and_kelly_roi_fall_with_stake_and_bankroll(fitted):
    _, P = fitted
    bets = B.BetRule(min_ev=1.0).select(P)
    assert len(bets) > 20
    rois = [B.flat_bet(bets, s, pool_yen=1_000_000.0)["roi"] for s in (100, 1_000, 10_000, 100_000)]
    assert np.all(np.diff(rois) <= 1e-12) and rois[-1] < rois[0]
    small = B.kelly_bet(bets, bankroll=100_000.0, pool_yen=1_000_000.0, detail=True)
    big = B.kelly_bet(bets, bankroll=10_000_000.0, pool_yen=1_000_000.0, detail=True)
    assert big["paid_odds_ratio"] < small["paid_odds_ratio"] <= 1.0
    capped = B.kelly_bet(bets, bankroll=10_000_000.0, pool_yen=1_000_000.0, max_stake=10_000, detail=True)
    assert capped["max_stake"] <= 10_000
    # レースごとの売上を列で渡せる (一定値なら定数と同じ、0 のレースは影響なし)
    col = bets.assign(win_pool_yen=1_000_000.0)
    assert B.flat_bet(col, 10_000, pool_yen="win_pool_yen") == B.flat_bet(bets, 10_000, pool_yen=1_000_000.0)
    assert B.kelly_bet(col, pool_yen="win_pool_yen") == B.kelly_bet(bets, pool_yen=1_000_000.0)
    zero = bets.assign(win_pool_yen=0.0)
    assert B.flat_bet(zero, 10_000, pool_yen="win_pool_yen")["roi"] == _flat_bet_ref(bets, 10_000)["roi"]


def test_pool0_sigma0_reproduce_current_results(fitted):
    _, P = fitted
    for ev in (1.0, 1.1):
        bets = B.BetRule(min_ev=ev).select(P)
        for stake in (100.0, 1_000.0):
            assert B.flat_bet(bets, stake) == _flat_bet_ref(bets, stake)
            assert B.flat_bet(bets, stake, pool_yen=0.0) == _flat_bet_ref(bets, stake)
        for bk in (100_000.0, 3_000_000.0):
            assert B.kelly_bet(bets, bankroll=bk) == _kelly_bet_ref(bets, bankroll=bk)
        # 締切前ビュー (σ=0) を通して確定オッズで払い戻しても同じ
        v = B.preclose_view(P, 0.0)
        assert np.array_equal(v["odds_final"], P["odds_win"])
        vb = B.BetRule(min_ev=ev).select(v)
        ref = _flat_bet_ref(bets)
        got = B.flat_bet(vb, pay_col="odds_final")
        assert {k: got[k] for k in ref} == ref
        kb = B.kelly_bet(vb, pay_col="odds_final", pool_yen=0.0)
        assert kb == _kelly_bet_ref(bets)
    # 感度分析の表の (プール 0, σ 0, 100 円) の行は従来の均等買い・ブートストラップと一致
    rule = B.BetRule(min_ev=1.1)
    bets = rule.select(P)
    rt = B.robustness_table(P, rule, pools=(0.0,), sigmas=(0.0,), stakes=(100.0,))
    row = rt.iloc[0]
    ref = _flat_bet_ref(bets)
    assert row["bets"] == ref["bets"] and row["roi"] == pytest.approx(ref["roi"], abs=1e-12)
    assert row["profit"] == pytest.approx(ref["profit"], abs=1e-6)
    lo, hi = B.bootstrap_roi(bets)
    assert row["roi_lo90"] == pytest.approx(lo, abs=1e-12) and row["roi_hi90"] == pytest.approx(hi, abs=1e-12)
    kr = B.kelly_robustness(P, rule, pools=(0.0,), sigmas=(0.0,), bankrolls=(100_000.0,))
    assert kr["growth"].iloc[0] == pytest.approx(_kelly_bet_ref(bets)["growth"], abs=1e-12)


# ---------------------------------------------------------------------------
# 締切前オッズのズレ
# ---------------------------------------------------------------------------

def test_stage2_market_coef_recovers_combo_coefficient(fitted):
    s2, P = fitted
    coef = B.stage2_market_coef(P)
    t = s2.combo_.coef_table().set_index("feature")
    assert set(coef["block"]) == {1, 2}
    assert np.allclose(coef["b"], t.loc["log_p_mkt", "coef_raw"], atol=1e-8)
    assert coef["max_abs_err"].max() < 1e-9


def test_reprice_matches_stage2_refit_prediction(fitted):
    s2, P = fitted
    coef = B.stage2_market_coef(P)
    # σ=0: p_model をそのまま再現
    assert np.abs(B.reprice_model(P, P["log_p_mkt"], coef) - P["p_model"]).max() < 1e-9
    # 市場確率を入れ替えた結果は、Stage2 にその市場確率を入れて予測し直した値と一致する
    rng = np.random.default_rng(1)
    new_log = P["log_p_mkt"] + rng.normal(0, 0.2, len(P))
    P2 = P.copy()
    P2["log_p_mkt"] = new_log
    direct = s2.predict(P2)["p_model"].to_numpy()
    assert np.abs(B.reprice_model(P, new_log, coef) - direct).max() < 1e-9


def test_noise_keeps_overround_and_probabilities(fitted):
    _, P = fitted
    v = B.preclose_view(P, 0.2, seed=3)
    rid = P["race_id"]
    over0 = (1 / P["odds_win"]).groupby(rid).sum()
    over1 = (1 / v["odds_win"]).groupby(rid).sum()
    assert np.allclose(over0, over1, rtol=1e-12)
    assert np.allclose(v.groupby("race_id")["p_mkt"].sum(), 1.0)
    assert np.allclose(v.groupby("race_id")["p_model"].sum(), 1.0)
    assert np.array_equal(v["odds_final"], P["odds_win"])
    assert not np.allclose(v["odds_win"], P["odds_win"])
    # ズレの大きさ ≈ σ (正規化で少し縮む)
    sd = np.log(v["odds_win"] / P["odds_win"]).std()
    assert 0.15 < sd < 0.22
    # 同じ seed なら同じ結果
    v2 = B.preclose_view(P, 0.2, seed=3)
    assert np.array_equal(v["odds_win"], v2["odds_win"])


def test_noise_with_partial_odds_race_keeps_model():
    """オッズが欠けたレース (p_model = p_fund) は締切前ビューでも p_model を変えない."""
    d = _synthetic(n_races=40, seed=5)
    s2 = Stage2().fit(d)
    d.loc[d.index[:3], ["odds_win", "p_mkt", "log_p_mkt"]] = np.nan  # R0000 の 3 頭だけ欠損
    d["block"] = 0
    P = pd.concat([d, s2.predict(d)], axis=1)
    v = B.preclose_view(P, 0.1, seed=0)
    first = P["race_id"] == "R0000"
    assert np.array_equal(v.loc[first, "p_model"], P.loc[first, "p_model"])
    assert v.loc[first, "odds_win"].isna().sum() == 3
    assert np.allclose(v.groupby("race_id")["p_model"].sum(), 1.0)


# ---------------------------------------------------------------------------
# 損益分岐の賭け金・レポート
# ---------------------------------------------------------------------------

def test_breakeven_stake_matches_closed_form():
    # 全 100 点が単勝 5.0 倍・的中率 25% → 回収率 = 0.25 × 実効オッズ。
    # 実効オッズが 4.0 を下回る賭け金: 0.8(T+s) < 4(0.8T/5.05 + s)  ⇔ s > T(0.8 − 3.2/5.05)/3.2
    n = 100
    bets = pd.DataFrame({"race_id": [f"R{i:03d}" for i in range(n)],
                         "date": pd.Timestamp("2025-01-01"), "odds_win": 5.0, "odds_final": 5.0,
                         "win": (np.arange(n) % 4 == 0).astype(float), "p_model": 0.3, "p_mkt": 0.16})
    T = 1_000_000.0
    s_star = T * (0.8 - 3.2 / 5.05) / 3.2
    be = B.breakeven_table(bets, pools=(T,), sigmas=(0.0,), decisions={0.0: [bets]})
    row = be.iloc[0]
    assert row["roi_100"] == pytest.approx(1.25)
    assert row["breakeven_stake"] == np.ceil(s_star / 100) * 100
    assert 0 < row["best_stake"] < row["breakeven_stake"]


def test_optimal_stake_closed_form():
    T, R = 2_500_000.0, 0.8
    for p, o in [(0.22, 5.0), (0.11, 10.0), (0.06, 20.0), (0.6, 2.0)]:
        opt, be = B.optimal_stake(p, o, T, R)
        s = np.linspace(1, 3 * max(float(be), 1.0), 200_001)
        W = R * T / o
        prof = s * (p * R * (T + s) / (W + s) - 1)
        assert float(opt) == pytest.approx(s[np.argmax(prof)], rel=1e-3)
        assert p * R * (T + be) / (W + be) == pytest.approx(1.0)
        assert 0 < opt < be
    opt, be = B.optimal_stake(0.1, 9.0, T, R)  # EV 0.9 → 買わない
    assert float(opt) == 0.0 and float(be) == 0.0


def test_realism_report_section(fitted):
    from banei_ai.cli import _realism_lines

    _, P = fitted
    a = SimpleNamespace(min_ev=1.05, pool_yen=1_000_000.0, odds_noise=0.1, payout_rate=0.8, noise_reps=2)
    text = "\n".join(_realism_lines(P, a))
    assert "## 現実性" in text and "損益分岐の賭け金" in text and "1/4ケリー" in text
    assert "100万円" in text and "影響なし" in text
