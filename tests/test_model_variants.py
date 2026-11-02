"""Stage1 / Stage2 の拡張 (アンサンブル・学習回数の自動選択・Stage2 の正則化選択など) のテスト."""
import numpy as np
import pandas as pd
import pytest

from banei_ai import model as M
from banei_ai.clogit import ConditionalLogit
from banei_ai.features import FUNDAMENTAL_FEATURES, build_features
from banei_ai.simulate import SimConfig, simulate


@pytest.fixture(scope="module")
def X():
    return build_features(simulate(SimConfig(start="2023-04-01", end="2024-03-31", seed=11)))


def _toy_stage2_data(n_races=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_races // 4, freq="D")
    for r in range(n_races):
        n = 8
        s1 = rng.normal(size=n)
        mk = s1 + rng.normal(0, 0.5, n)
        p = np.exp(mk) / np.exp(mk).sum()
        pt = np.exp(s1) / np.exp(s1).sum()
        w = rng.choice(n, p=pt)
        for i in range(n):
            rows.append({"race_id": f"r{r:04d}", "date": dates[r // 4], "s1": s1[i], "p_mkt": p[i],
                         "log_p_mkt": np.log(p[i]), "win": int(i == w),
                         "j_popularity": 1.0 + rng.normal(0, 0.1), "h_last_pmkt_share": 1.0})
    return pd.DataFrame(rows)


def test_stage1_ensemble_scores_and_contributions(X):
    tr = X[X["date"] < "2024-01-01"]
    te = X[X["date"] >= "2024-01-01"]
    s = M.Stage1(n_rounds=60, n_bags=2, extra_objectives=("lambdarank",)).fit(tr)
    assert len(s.boosters_) == 4 and len(s.scales_) == 4 and all(v > 0 for v in s.scales_)
    sc = s.score(te)
    assert sc.shape == (len(te),) and np.isfinite(sc).all()
    # SHAP 寄与の合計はレース内の相対スコアを再現する (定数項はレース内で打ち消す)
    C = s.contributions(te)
    rid = te["race_id"].to_numpy()
    a = pd.Series(sc, index=te.index)
    b = C.sum(axis=1)
    da = a - a.groupby(rid).transform("mean")
    db = b - b.groupby(rid).transform("mean")
    assert np.allclose(da, db, atol=1e-6)
    imp = s.importance()
    assert imp.sum() == pytest.approx(1.0) and (imp >= 0).all()


def test_stage1_single_model_unchanged(X):
    """既定 (単一モデル) は従来どおり生のスコアを返す."""
    tr = X[X["date"] < "2024-01-01"]
    s = M.Stage1(n_rounds=30).fit(tr)
    assert s.boosters_ is None
    Xf = tr.reindex(columns=s.features).astype(float)
    assert np.allclose(s.score(tr), s.booster.predict(Xf))


def test_stage1_early_stopping_chooses_rounds(X):
    tr = X[X["date"] < "2024-03-01"]
    s = M.Stage1(early_stopping=True, es_max_rounds=150, es_step=50).fit(tr)
    assert s.best_rounds_ is not None and 50 <= s.best_rounds_ <= 150 * 2
    assert s.booster.current_iteration() == s.best_rounds_


@pytest.mark.parametrize("kw", [dict(market_sq=True), dict(s1_race_z=True),
                                dict(l2_grid=(1e-4, 1e-2, 1.0))])
def test_stage2_variants_are_probabilities(kw):
    d = _toy_stage2_data()
    s2 = M.Stage2(**kw).fit(d)
    out = s2.predict(d)
    assert list(out.columns) == ["p_fund", "p_mkt_cal", "p_model"]
    assert np.allclose(out["p_model"].groupby(d["race_id"]).sum(), 1.0)
    assert "log_p_mkt_sq" not in d  # 入力は書き換えない
    if "l2_grid" in kw:
        assert s2.l2_ in kw["l2_grid"]
    if kw.get("market_sq"):
        assert "log_p_mkt_sq" in s2.cols_


def test_stage2_l2_grid_prefers_regularisation_for_useless_features():
    """役に立たない特徴量だけのときは強い正則化 (係数 0 = 一様) が選ばれる."""
    d = _toy_stage2_data()
    rng = np.random.default_rng(1)
    d["s1"] = rng.normal(size=len(d))
    d["log_p_mkt"] = rng.normal(size=len(d))
    d["p_mkt"] = np.exp(d["log_p_mkt"])
    s2 = M.Stage2(l2_grid=(1e-4, 1e3)).fit(d)
    assert s2.l2_ == 1e3


def test_old_pickled_stage2_without_new_fields_still_predicts():
    """新しい属性を持たない (旧バージョンで保存した) Stage2 / Stage1 でも予測できる."""
    d = _toy_stage2_data(120)
    s2 = M.Stage2().fit(d)
    for k in ("l2", "l2_grid", "l2_valid_frac", "market_sq", "s1_race_z", "recency_half_life", "l2_"):
        s2.__dict__.pop(k, None)
    out = s2.predict(d)
    assert np.allclose(out["p_model"].groupby(d["race_id"]).sum(), 1.0)


def test_walk_forward_factories_keep_output_columns(X):
    base_cols = None
    for f1, f2 in [(None, None),
                   (lambda: M.Stage1(n_rounds=30), lambda: M.Stage2(market_sq=True))]:
        P = M.walk_forward(X, min_train_days=180, min_stage2_blocks=1, verbose=False,
                           stage1_factory=f1 or (lambda: M.Stage1(n_rounds=30)), stage2_factory=f2)
        cols = list(P.columns)
        base_cols = base_cols or cols
        assert cols == base_cols
        assert {"s1", "block", "p_fund", "p_mkt_cal", "p_model"} <= set(cols)


def test_oos_stage1_models_out(X):
    mo = {}
    O = M.oos_stage1(X, min_train_days=180, stage1_factory=lambda: M.Stage1(n_rounds=20), models_out=mo)
    assert set(mo) == set(O.loc[O["block"] >= 0, "block"].unique())
    k = max(mo)
    start, s1 = mo[k]
    te = O[O["block"] == k]
    assert te["date"].min() >= start
    assert np.allclose(s1.score(te), te["s1"])


def test_weighted_clogit():
    """重み 1 は重みなしと同じ。重み 0 のレースは学習に効かない."""
    d = _toy_stage2_data(300)
    cols = ["s1", "log_p_mkt"]
    a = ConditionalLogit(l2=1e-3).fit(d[cols], d["win"], d["race_id"], cols)
    b = ConditionalLogit(l2=1e-3).fit(d[cols], d["win"], d["race_id"], cols, weights=np.ones(len(d)))
    assert np.allclose(a.coef_, b.coef_, atol=1e-6)
    # 後半のレースだけ重み 1 → 後半だけで学習した場合と同じ係数 (標準化の基準は全体なので係数で比較)
    late = d["race_id"] >= "r0150"
    w = late.astype(float).to_numpy()
    c = ConditionalLogit(l2=0).fit(d[cols], d["win"], d["race_id"], cols, weights=w)
    e = ConditionalLogit(l2=0).fit(d.loc[late, cols], d.loc[late, "win"], d.loc[late, "race_id"], cols)
    assert np.allclose(c.coef_ / c.scale_, e.coef_ / e.scale_, atol=1e-3)


def test_stage2_recency_weights():
    d = _toy_stage2_data()
    s2 = M.Stage2(recency_half_life=30).fit(d)
    w = s2._weights(d)
    assert w.max() == pytest.approx(1.0) and w.min() < 0.5
    assert np.allclose(s2.predict(d)["p_model"].groupby(d["race_id"]).sum(), 1.0)


def test_cw_rel_change_feature(X):
    """季節に頑健な重量の前走比: 今回と前走の「レース平均との差」の差."""
    assert "h_cw_rel_change" in FUNDAMENTAL_FEATURES
    d = X.sort_values(["date", "race_id"])
    prev = d.groupby("horse_id")["cw_rel"].shift(1)
    ok = d["h_n_runs"] > 0
    assert np.allclose(d.loc[ok, "h_cw_rel_change"], (d["cw_rel"] - prev.fillna(0.0))[ok], atol=1e-9)
    assert d.loc[~ok, "h_cw_rel_change"].isna().all()
