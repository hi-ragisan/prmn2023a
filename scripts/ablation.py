"""アブレーション (何が効いているか) と改善候補の比較実験.

  # 特徴量グループを 1 つずつ外したときの悪化量 (シミュレーション 3 シード)
  OMP_NUM_THREADS=1 python scripts/ablation.py --part ablation --cache /tmp/banei_cache
  # 改善候補 (アンサンブル / Stage2 正則化 / 市場項の非線形 / 学習回数の自動選択)
  OMP_NUM_THREADS=1 python scripts/ablation.py --part candidates --cache /tmp/banei_cache
  # 本番モデルの「Stage1 スコアの尺度ずれ」の検証 (期間を区切ったホールドアウト)
  OMP_NUM_THREADS=1 python scripts/ablation.py --part scale --cache /tmp/banei_cache
  # 実データで
  python scripts/ablation.py --part ablation --data data/interim/races.csv

指標 (すべてウォークフォワード・完全サンプル外):
  合成 vs市場 : 市場確率 (オッズそのまま) に対する合成モデルの対数損失の改善 (正 = 市場より良い)
  能力 LL     : Stage1 だけ (オッズ不使用) の対数損失
  寄与        : 外したときの悪化量 (正 = そのグループが役に立っている)
  ±SE         : レース単位の対応のある差の標準誤差
  点数・回収率 : 既定の購入ルール (期待値 1.1 以上など, backtest.BetRule) の単勝均等買い

⚠ シミュレーションの市場には意図的な歪み (大穴過剰人気・人気騎手・前走着順への過剰反応・
水分適性の無視・馬体重減の軽視) が埋め込まれている。その歪みに直接対応するグループの
寄与はシミュレーションの「作り」を反映しているだけで、実データでの大きさは分からない。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from banei_ai import backtest as B  # noqa: E402
from banei_ai import model as M  # noqa: E402
from banei_ai import schema  # noqa: E402
from banei_ai.features import FUNDAMENTAL_FEATURES, REL_COLS, build_features  # noqa: E402
from banei_ai.simulate import SimConfig, simulate  # noqa: E402


def _rel(*cols: str) -> list[str]:
    return [f"r_{c}_{k}" for c in cols for k in ("diff", "rank")]


# Stage1 の特徴量グループ。各グループはその特徴量の「レース内相対値 r_*」も含む
# (絶対値だけ外しても相対値から情報が漏れ戻るため)。r_* グループは表現方法の検証。
FEATURE_GROUPS: dict[str, list[str]] = {
    "馬: 実績・近走・着差": [
        "h_n_runs", "h_rating", "h_days_since", "h_last1_pct", "h_last2_pct", "h_last3_pct",
        "h_avg5_pct", "h_ewm_pct", "h_last_gap", "h_avg3_gap", "h_best5_gap",
        "h_win_ae", "h_top3_rate", "h_cls_change",
    ] + _rel("h_rating", "h_ewm_pct", "h_avg3_gap", "h_win_ae", "h_best5_gap"),
    "馬: スピード指数": ["h_spd_last", "h_spd_avg3", "h_spd_best5", "h_spd_ewm"]
    + _rel("h_spd_ewm", "h_spd_best5"),
    "馬: 馬場水分適性": ["h_moist_apt", "h_moist_n"] + _rel("h_moist_apt"),
    "馬: 馬体重": ["horse_weight", "horse_weight_diff"],
    "負担重量": ["carried_weight", "cw_rel", "cw_ratio", "r_cw_rank", "h_cw_change", "h_cw_rel_change",
             "h_heavy_perf"],
    "騎手: 腕 (勝率・複勝率)": ["j_rides", "j_win_ae", "j_top3_ae"] + _rel("j_win_ae", "j_top3_ae"),
    "騎手×馬場水分": ["j_moist_rel"],
    "騎手×馬コンビ・乗り替わり": ["jh_rides", "jh_win_ae", "h_jockey_change"],
    "調教師": ["t_win_ae"] + _rel("t_win_ae"),
    "レース条件・馬齢": ["n_runners", "moisture", "class_rank", "age", "age_rel"],
    "レース内相対値 r_* (全部)": [f for f in FUNDAMENTAL_FEATURES if f.startswith("r_")],
}
# まとめて外す (馬の良し悪し / 騎手の良し悪し の総量)
AGGREGATES = {
    "【馬の情報すべて】": ["馬: 実績・近走・着差", "馬: スピード指数", "馬: 馬場水分適性", "馬: 馬体重"],
    "【騎手の情報すべて】": ["騎手: 腕 (勝率・複勝率)", "騎手×馬場水分", "騎手×馬コンビ・乗り替わり"],
}
# 歪みが埋め込まれているグループ (シミュレーションの結果を割り引いて読む)
PLANTED = {"馬: 馬場水分適性", "馬: 馬体重"}


def _check_groups() -> None:
    allg = {f for fs in FEATURE_GROUPS.values() for f in fs}
    missing = set(FUNDAMENTAL_FEATURES) - allg
    assert not missing, f"どのグループにも入っていない特徴量: {missing}"
    assert {f"r_{c}_diff" for c in REL_COLS} <= set(FUNDAMENTAL_FEATURES)


def _h(s: str) -> str:
    """キャッシュ用の安定したハッシュ (hash() はプロセスごとに変わる)."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# データとキャッシュ
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, cache: Path | None):
        self.cache = cache
        if cache:
            cache.mkdir(parents=True, exist_ok=True)

    def _cached(self, key: str, make: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        if self.cache:
            p = self.cache / f"{key}.pkl"
            if p.exists():
                return pd.read_pickle(p)
        t = time.time()
        out = make()
        print(f"    [{key}] {time.time() - t:.0f}s", flush=True)
        if self.cache:
            out.to_pickle(self.cache / f"{key}.pkl")
        return out

    def features(self, name: str, make: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        return self._cached(f"feat_{name}", make)

    def s1(self, name: str, tag: str, X: pd.DataFrame,
           factory: Callable[[], M.Stage1] | None) -> pd.DataFrame:
        def make():
            O = M.oos_stage1(X, stage1_factory=factory)
            return O[["race_id", "post", "s1", "block"]]
        # キャッシュのキーには Stage1 の設定 (特徴量・パラメータ) を含める (既定が変わったら再計算)
        S = self._cached(f"s1_{name}_{tag}_{_h(repr((factory or M.Stage1)()))}", make)
        base = X.sort_values(["date", "race_id", "post"]).reset_index(drop=True)
        return base.drop(columns=["s1", "block"], errors="ignore").merge(S, on=["race_id", "post"], how="left")


def datasets(a, runner: Runner) -> dict[str, pd.DataFrame]:
    if a.data:
        return {"real": runner.features("real", lambda: build_features(schema.normalize(schema.read_csv(a.data))))}
    out = {}
    for s in a.seeds.split(","):
        s = int(s)
        if a.sim_preset:  # 例: efficient (歪みなしの市場 = 帰無仮説の検証用)
            if not hasattr(SimConfig, "preset"):
                raise SystemExit("この simulate.py には SimConfig.preset がありません")
            cfg = SimConfig.preset(a.sim_preset, end=a.sim_end, seed=s)
            name = f"sim{s}_{a.sim_preset}"
        else:
            cfg, name = SimConfig(end=a.sim_end, seed=s), f"sim{s}"
        out[name] = runner.features(name, lambda cfg=cfg: build_features(simulate(cfg)))
    return out


# ---------------------------------------------------------------------------
# 指標
# ---------------------------------------------------------------------------
def race_ll(P: pd.DataFrame, col: str) -> pd.Series:
    """レースごとの -log(勝ち馬の確率) (同着は平均)."""
    w = P[(P["win"] == 1) & P["p_mkt"].notna()]
    return (-np.log(w[col].clip(1e-9, 1))).groupby(w["race_id"]).mean()


def metrics(P: pd.DataFrame) -> dict:
    ev = M.evaluate_probs(P).set_index("model")
    sel = B.BetRule(min_ev=1.1).select(P)
    bets = B.flat_bet(sel)
    out = {
        "合成LL": ev.loc["合成モデル", "logloss"],
        "合成vs市場": ev.loc["合成モデル", "vs_market"],
        "能力LL": ev.loc["ファンダメンタル", "logloss"],
        "点数": bets["bets"], "回収率": bets["roi"],
    }
    if "_p_true" in sel:  # シミュレーションのみ: 買った馬券の「真の」期待値 (運に左右されない)
        out["真のEV"] = float((_true_p(P).loc[sel.index] * sel["odds_win"]).mean()) if len(sel) else np.nan
    return out


def _true_p(P: pd.DataFrame) -> pd.Series:
    """真の勝率。シミュレーターに true_win_probs (第2障害での停止まで含めた厳密な勝率) があればそれを使う."""
    try:
        from banei_ai.simulate import true_win_probs
    except ImportError:
        return P["_p_true"] / P.groupby("race_id")["_p_true"].transform("sum")
    return true_win_probs(P)


def paired(P: pd.DataFrame, base: pd.DataFrame, col: str = "p_model") -> tuple[float, float, int]:
    """(variant − base) のレース単位の対数損失差の平均と標準誤差."""
    a, b = race_ll(P, col), race_ll(base, col)
    d = (a - b).dropna()
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d)


def _fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4f")


# ---------------------------------------------------------------------------
# 1) アブレーション
# ---------------------------------------------------------------------------
def run_ablation(data: dict[str, pd.DataFrame], runner: Runner) -> str:
    _check_groups()
    s1_variants: dict[str, list[str]] = {g: fs for g, fs in FEATURE_GROUPS.items()}
    for agg, gs in AGGREGATES.items():
        s1_variants[agg] = sorted({f for g in gs for f in FEATURE_GROUPS[g]})
    rows, coefs = [], []
    for name, X in data.items():
        print(f"== {name}", flush=True)
        Xb = runner.s1(name, "base", X, None)
        Pb = M.oos_stage2(Xb)
        mb = metrics(Pb)
        rows.append({"data": name, "variant": "ベースライン (全部入り)", **mb,
                     "寄与(合成)": 0.0, "SE": 0.0, "寄与(能力)": 0.0})

        def add(label: str, P: pd.DataFrame):
            m = metrics(P)
            dm, se, _ = paired(P, Pb)
            rows.append({"data": name, "variant": label, **m, "寄与(合成)": dm, "SE": se,
                         "寄与(能力)": m["能力LL"] - mb["能力LL"]})
            print(f"  {label}: 合成vs市場 {m['合成vs市場']:+.4f} 寄与 {dm:+.4f}±{se:.4f}", flush=True)

        # ノイズの目安: 同じ設定で LightGBM の乱数シードだけ変える
        seed2 = lambda: M.Stage1(params={**M.Stage1().params, "seed": 4242})  # noqa: E731
        add("(参考) LightGBM 乱数シード違い", M.oos_stage2(runner.s1(name, "seed4242", X, seed2)))
        for label, drop in s1_variants.items():
            keep = [f for f in FUNDAMENTAL_FEATURES if f not in set(drop)]
            tag = "drop_" + _h(",".join(sorted(drop)))
            Xv = runner.s1(name, tag, X, lambda keep=keep: M.Stage1(features=list(keep)))
            add(f"Stage1 − {label}", M.oos_stage2(Xv))
        # Stage2 の市場系の追加項
        for label, extra in [("Stage2 − 騎手人気度 j_popularity", ["h_last_pmkt_share"]),
                             ("Stage2 − 前走の市場シェア h_last_pmkt_share", ["j_popularity"]),
                             ("Stage2 − 市場系の追加項 両方", [])]:
            orig = list(M.STAGE2_MARKET_EXTRA)
            M.STAGE2_MARKET_EXTRA[:] = extra
            try:
                add(label, M.oos_stage2(Xb))
            finally:
                M.STAGE2_MARKET_EXTRA[:] = orig
        # Stage1 を丸ごと使わない (市場 + 市場系の追加項だけ)
        add("Stage2 − Stage1 スコア (市場情報だけ)", M.oos_stage2(Xb.assign(s1=0.0)))
        # Stage2 の係数 (サンプル外 s1 の全ブロックで学習, 標準化係数)
        s2 = M.Stage2().fit(Xb[Xb["block"] >= 0])
        for r in s2.combo_.coef_table().itertuples(index=False):
            coefs.append({"data": name, "feature": r.feature, "coef_std": r.coef_std})
    R = pd.DataFrame(rows)
    C = pd.DataFrame(coefs).pivot(index="feature", columns="data", values="coef_std")
    return (_summarise(R, "variant")
            + "\n\n### Stage2 (合成モデル) の標準化係数\n\n" + C.to_markdown(floatfmt=".3f"))


def _mean_table(R: pd.DataFrame, key: str, delta: str, flag: str, all_positive: bool) -> pd.DataFrame:
    """シード平均。回収率・真の EV は点数で重み付けした合算値 (1 シードの運に左右されにくい)."""
    g = R.groupby(key, sort=False)
    cols = [c for c in ["合成vs市場", delta, "寄与(能力)", "能力LL", "点数"] if c in R]
    t = g[cols].mean()
    t["SE"] = g["SE"].apply(lambda s: float(np.sqrt((s ** 2).sum())) / len(s))  # 平均の SE
    for c in ("回収率", "真のEV"):
        if c in R:
            t[c + "(合算)"] = g.apply(lambda x, c=c: float(np.nansum(x[c] * x["点数"]) / max(x["点数"].sum(), 1)),
                                    include_groups=False)
    if all_positive:
        t[flag] = g[delta].apply(lambda s: bool((s > 0).all()))
    else:
        t[flag] = g[delta].apply(lambda s: bool((s > 0).all() or (s < 0).all()))
    return t.reset_index()


def _summarise(R: pd.DataFrame, key: str) -> str:
    mean = _mean_table(R, key, "寄与(合成)", "全シードで同符号", all_positive=False)
    return ("### シード平均\n\n" + mean.to_markdown(index=False, floatfmt=".4f")
            + "\n\n### シード別\n\n" + _fmt(R))


# ---------------------------------------------------------------------------
# 2) 改善候補
# ---------------------------------------------------------------------------
STAGE1_CANDIDATES: dict[str, Callable[[], M.Stage1]] = {
    "(a) Stage1 バギング 3 本 (乱数シード違い)": lambda: M.Stage1(n_bags=3),
    "(a) Stage1 rank_xendcg + lambdarank 平均": lambda: M.Stage1(extra_objectives=("lambdarank",)),
    "(a) Stage1 (xendcg + lambdarank) × 2 シード": lambda: M.Stage1(n_bags=2, extra_objectives=("lambdarank",)),
    "(d) Stage1 学習回数を内側の時系列分割で選択": lambda: M.Stage1(early_stopping=True),
    "(d') 参考: 学習回数 200 固定": lambda: M.Stage1(n_rounds=200),
    "(d') 参考: 学習回数 800 固定": lambda: M.Stage1(n_rounds=800),
    # 実際のばんえいは基礎重量が開催回ごとに上がる (年度初めに戻る) → 季節に頑健な重量の前走比
    # h_cw_rel_change は採用済み (既定の特徴量の末尾)。ここでは外した場合・置き換えた場合と比べる
    "(f) 参考: h_cw_rel_change なし (採用前)": lambda: M.Stage1(
        features=[f for f in FUNDAMENTAL_FEATURES if f != "h_cw_rel_change"]),
    "(f') h_cw_change を外し h_cw_rel_change だけにする": lambda: M.Stage1(
        features=[f for f in FUNDAMENTAL_FEATURES if f != "h_cw_change"]),
}
STAGE2_CANDIDATES: dict[str, Callable[[], M.Stage2]] = {
    "(b) Stage2 L2 を内側の時系列分割で選択": lambda: M.Stage2(l2_grid=(1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1)),
    "(b') 参考: Stage2 L2 = 0.1 固定": lambda: M.Stage2(l2=0.1),
    "(c) Stage2 + (log 市場確率)^2": lambda: M.Stage2(market_sq=True),
    "(e) Stage2 で s1 をレース内標準化": lambda: M.Stage2(s1_race_z=True),
    "(e') Stage2 を新しいレースほど重く (半減期 180 日)": lambda: M.Stage2(recency_half_life=180),
    "(e') Stage2 を新しいレースほど重く (半減期 365 日)": lambda: M.Stage2(recency_half_life=365),
}


def _tag(label: str) -> str:
    return "cand_" + _h(label)


def run_candidates(data: dict[str, pd.DataFrame], runner: Runner, only: str | None = None) -> str:
    rows = []
    for name, X in data.items():
        print(f"== {name}", flush=True)
        Xb = runner.s1(name, "base", X, None)
        Pb = M.oos_stage2(Xb)
        mb = metrics(Pb)
        rows.append({"data": name, "candidate": "ベースライン (現行)", **mb, "改善": 0.0, "SE": 0.0})

        def add(label: str, P: pd.DataFrame):
            m = metrics(P)
            dm, se, _ = paired(P, Pb)
            rows.append({"data": name, "candidate": label, **m, "改善": -dm, "SE": se})
            print(f"  {label}: 合成vs市場 {m['合成vs市場']:+.4f} 改善 {-dm:+.4f}±{se:.4f}", flush=True)

        for label, fac in STAGE1_CANDIDATES.items():
            if only and only not in label:
                continue
            # 学習回数はタグに含める (同じラベルでも設定が変わったら再計算)
            add(label, M.oos_stage2(runner.s1(name, _tag(label + repr(fac())), X, fac)))
        for label, fac in STAGE2_CANDIDATES.items():
            if only and only not in label:
                continue
            add(label, M.oos_stage2(Xb, stage2_factory=fac))
    R = pd.DataFrame(rows)
    mean = _mean_table(R, "candidate", "改善", "全シードで改善", all_positive=True)
    return ("### シード平均 (改善 = 合成モデルの対数損失の減少, 正が良い)\n\n"
            + mean.to_markdown(index=False, floatfmt=".4f")
            + "\n\n### シード別\n\n" + _fmt(R))


# ---------------------------------------------------------------------------
# 3) 本番モデルの尺度ずれ (期間を区切ったホールドアウト)
# ---------------------------------------------------------------------------
def _best_scale(s2: M.Stage2, H: pd.DataFrame) -> float:
    """後知恵で最適な s1 の倍率 (1 なら尺度ずれ無し)."""
    best, best_ll = 1.0, np.inf
    for c in np.round(np.arange(0.5, 1.81, 0.05), 2):
        P = s2.predict(H.assign(s1=H["s1"] * c))
        ll = race_ll(pd.concat([H, P], axis=1), "p_model").mean()
        if ll < best_ll:
            best, best_ll = c, ll
    return float(best)


def run_scale(data: dict[str, pd.DataFrame], runner: Runner, months: int = 2,
              cutoffs: tuple[str, ...] = ("2025-02-01", "2025-04-01", "2025-06-01", "2025-08-01")) -> str:
    """BaneiModel.fit と同じ手順で cut より前のデータだけから学習し、cut から months ヶ月を予測.

    比較:
      現行             : Stage2 = サンプル外 s1 で学習 / Stage1 = cut までの全データで学習
      最後の OOS モデル : Stage1 を、Stage2 の学習に使った最後のブロックのモデルにする (尺度は揃うが 2 ヶ月古い)
      s1 レース内標準化 : Stage2 の前に s1 をレース内で標準化
    """
    rows, diag = [], []
    for name, X in data.items():
        print(f"== {name}", flush=True)
        preds: dict[str, list[pd.DataFrame]] = {"現行": [], "最後の OOS モデル": [], "s1 レース内標準化": []}
        for cut in cutoffs:
            cut = pd.Timestamp(cut)
            if cut >= X["date"].max():
                continue
            hist = X[X["date"] < cut]
            hold = X[(X["date"] >= cut) & (X["date"] < cut + pd.DateOffset(months=months))].copy()
            mo: dict = {}
            oos = M.oos_stage1(hist, models_out=mo)
            oos = oos[oos["block"] >= 0]
            s2 = M.Stage2().fit(oos)
            s2z = M.Stage2(s1_race_z=True).fit(oos)
            t0 = hist["date"].min() + pd.Timedelta(days=90)
            full = M.Stage1().fit(hist[hist["date"] >= t0])
            last = mo[max(mo)][1]
            Hf, Hl = hold.assign(s1=full.score(hold)), hold.assign(s1=last.score(hold))
            preds["現行"].append(pd.concat([Hf, s2.predict(Hf)], axis=1))
            preds["最後の OOS モデル"].append(pd.concat([Hl, s2.predict(Hl)], axis=1))
            preds["s1 レース内標準化"].append(pd.concat([Hf, s2z.predict(Hf)], axis=1))
            rid = hold["race_id"].to_numpy()
            oos_last = oos[oos["block"] == max(mo)]
            diag.append({
                "data": name, "cut": f"{cut:%Y-%m}",
                "s1尺度 OOS(学習に使った最後のブロック)": M._race_demeaned_sd(oos_last["s1"], oos_last["race_id"]),
                "s1尺度 現行(全データ)": M._race_demeaned_sd(Hf["s1"], rid),
                "s1尺度 最後のOOSモデル": M._race_demeaned_sd(Hl["s1"], rid),
                "後知恵の最適倍率 現行": _best_scale(s2, Hf),
                "後知恵の最適倍率 最後のOOS": _best_scale(s2, Hl),
                "Stage2 s1係数(raw)": float(s2.combo_.coef_table().set_index("feature").loc["s1", "coef_raw"]),
            })
            print(f"  cut {cut:%Y-%m} done", flush=True)
        base = pd.concat(preds["現行"])
        # 参考: 同じ期間のウォークフォワード予測 (ブロック境界が cut と一致していれば本番手順と同じ)
        Pw = M.oos_stage2(runner.s1(name, "base", X, None))
        preds["(参考) ウォークフォワードの同期間"] = [Pw[Pw["race_id"].isin(base["race_id"])]]
        for k, ps in preds.items():
            P = pd.concat(ps)
            m = metrics(P)
            dm, se, n = paired(P, base)
            rows.append({"data": name, "variant": k, **m, "改善": -dm, "SE": se, "レース数": n})
    R, D = pd.DataFrame(rows), pd.DataFrame(diag)
    mean = _mean_table(R, "variant", "改善", "全シードで改善", all_positive=True)
    return ("### ホールドアウトの成績 (シード平均)\n\n" + mean.to_markdown(index=False, floatfmt=".4f")
            + "\n\n### シード別\n\n" + _fmt(R)
            + "\n\n### 尺度の診断 (レース内標準偏差 / 後知恵で最適な s1 倍率)\n\n" + _fmt(D))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["ablation", "candidates", "scale"], default="ablation")
    ap.add_argument("--data", default=None, help="実データ CSV (省略時はシミュレーション)")
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--sim-end", default="2025-09-30")
    ap.add_argument("--sim-preset", default=None, help="SimConfig.preset の名前 (例: efficient)")
    ap.add_argument("--cache", default=None, help="特徴量・サンプル外 s1 のキャッシュ置き場")
    ap.add_argument("--only", default=None, help="candidates: ラベルにこの文字列を含む候補だけ")
    ap.add_argument("--out", default=None, help="結果を Markdown で保存")
    a = ap.parse_args()
    runner = Runner(Path(a.cache) if a.cache else None)
    data = datasets(a, runner)
    if a.part == "ablation":
        md = run_ablation(data, runner)
    elif a.part == "candidates":
        md = run_candidates(data, runner, a.only)
    else:
        md = run_scale(data, runner)
    print(md)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
