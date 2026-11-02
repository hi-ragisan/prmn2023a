"""二段階モデル (Benter 方式).

Stage1  ファンダメンタル: LightGBM ランキング学習 (rank_xendcg, オッズは使わない)
        → 馬・騎手・負担重量・馬場水分などから「強さスコア」s1 を出す
Stage2  合成: 条件付きロジット  u = a·s1 + b·log(p_mkt) + c·(市場系特徴量)
        → 市場の情報とモデルの情報を最適に混ぜた勝率 p_model

「オッズの歪み」= p_model / p_mkt。これが大きく、期待値 p_model × オッズ > 1 の馬が狙い目。

Stage2 の学習には Stage1 の *サンプル外* スコアだけを使う (in-sample スコアは過信を生む)。
評価は時系列のウォークフォワードで行い、未来の情報は一切使わない。

設計比較 (アブレーション・改善候補) の結果は docs/model_experiments.md を参照。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import config
from .clogit import ConditionalLogit, race_logloss
from .features import FUNDAMENTAL_FEATURES

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

STAGE2_MARKET_EXTRA = ["j_popularity", "h_last_pmkt_share"]


def _relevance(d: pd.DataFrame) -> np.ndarray:
    f = d["finish_eff"].to_numpy()
    return np.clip(4 - f, 0, 3).astype(int)  # 1着=3, 2着=2, 3着=1


def _race_demeaned_sd(s: np.ndarray, race_ids: np.ndarray) -> float:
    """レース内平均を引いたスコアの標準偏差 (条件付きロジットにとって意味のある「尺度」)."""
    s = pd.Series(np.asarray(s, dtype=float))
    dm = s - s.groupby(np.asarray(race_ids)).transform("mean")
    sd = float(dm.std())
    return sd if np.isfinite(sd) and sd > 0 else 1.0


def _inner_split(dates: pd.Series, valid_frac: float) -> pd.Series:
    """時系列の内側分割: 最後の valid_frac (開催日ベース) を検証用に (True = 検証)."""
    days = np.sort(dates.unique())
    if len(days) < 10:
        return pd.Series(False, index=dates.index)
    cut = days[int(len(days) * (1 - valid_frac))]
    return dates >= cut


@dataclass
class Stage1:
    features: list[str] = field(default_factory=lambda: list(FUNDAMENTAL_FEATURES))
    params: dict = field(default_factory=lambda: dict(
        # rank_xendcg: シミュレーション 3 シードで lambdarank より一貫して良かった
        objective="rank_xendcg",
        learning_rate=0.03,
        num_leaves=15,
        min_data_in_leaf=60,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=5.0,
        verbose=-1,
        seed=config.RANDOM_SEED,
    ))
    n_rounds: int = 400
    booster: object = None
    fallback: ConditionalLogit | None = None
    # --- アンサンブル (既定は従来どおり単一モデル) ---
    # n_bags: 乱数シード違いのモデル数。extra_objectives: 追加で学習する目的関数 (例 "lambdarank")。
    # 2 本以上のときは各モデルのスコアを学習データ上の尺度 (レース内標準偏差) で割って平均する。
    n_bags: int = 1
    extra_objectives: tuple[str, ...] = ()
    # --- 内側の時系列分割による学習回数の自動選択 (既定は無効 = n_rounds 固定) ---
    early_stopping: bool = False
    es_valid_frac: float = 0.2
    es_max_rounds: int = 1200
    es_step: int = 50
    boosters_: list | None = None
    scales_: list | None = None
    best_rounds_: int | None = None

    # ------------------------------------------------------------------
    def _specs(self) -> list[dict]:
        base = dict(self.params)
        specs = []
        for obj in (base.get("objective", "rank_xendcg"), *getattr(self, "extra_objectives", ())):
            for b in range(max(1, getattr(self, "n_bags", 1))):
                p = {**base, "objective": obj}
                p["seed"] = int(base.get("seed", config.RANDOM_SEED)) + 1000 * b
                specs.append(p)
        return specs

    def _choose_rounds(self, d: pd.DataFrame, X: pd.DataFrame, params: dict) -> int:
        """学習期間の最後 es_valid_frac を検証に取り、学習回数を選ぶ.

        指標は検証期間での条件付きロジットの対数損失 (オッズがあれば s1 + log 市場確率、
        無ければ s1 だけ)。Stage2 が実際に使う形でスコアの良し悪しを測る。
        """
        va = _inner_split(d["date"], self.es_valid_frac).to_numpy()
        if va.sum() < 200 or (~va).sum() < 1000:
            return self.n_rounds
        dt, dv = d[~va], d[va]
        grp = dt.groupby("race_id", sort=False).size().to_numpy()
        ds = lgb.Dataset(X[~va], label=_relevance(dt), group=grp, free_raw_data=False)
        bst = lgb.train(params, ds, num_boost_round=self.es_max_rounds)
        Xv = X[va]
        rid = dv["race_id"].to_numpy()
        use_mkt = "log_p_mkt" in dv and dv["log_p_mkt"].notna().all()
        best, best_ll = self.n_rounds, np.inf
        for k in range(self.es_step, self.es_max_rounds + 1, self.es_step):
            Z = pd.DataFrame({"s1": bst.predict(Xv, num_iteration=k)}, index=dv.index)
            cols = ["s1"]
            if use_mkt:
                Z["log_p_mkt"] = dv["log_p_mkt"]
                cols.append("log_p_mkt")
            cl = ConditionalLogit(l2=1e-4).fit(Z[cols], dv["win"], rid, cols)
            ll = race_logloss(cl.predict_proba(Z[cols], rid), dv["win"].to_numpy(), rid)
            if ll < best_ll - 1e-6:
                best, best_ll = k, ll
        # 全期間で学習し直すときはデータ量に比例して少し回数を増やす
        return int(round(best * len(d) / max(len(dt), 1)))

    def fit(self, d: pd.DataFrame) -> "Stage1":
        d = d.sort_values(["date", "race_id"])
        # データに無い列 (例: タイムが無いデータのスピード指数) は使わない
        self.features = [c for c in self.features if c in d.columns and d[c].notna().any()]
        X = d[self.features].astype(float)
        if lgb is None:
            self.fallback = ConditionalLogit(l2=1e-2).fit(X, d["win"], d["race_id"], self.features)
            return self
        grp = d.groupby("race_id", sort=False).size().to_numpy()
        ds = lgb.Dataset(X, label=_relevance(d), group=grp, free_raw_data=False)
        specs = self._specs()
        rounds = self.n_rounds
        if getattr(self, "early_stopping", False):
            rounds = self._choose_rounds(d, X, specs[0])
            self.best_rounds_ = rounds
        boosters = [lgb.train(p, ds, num_boost_round=rounds) for p in specs]
        self.booster = boosters[0]
        if len(boosters) > 1:
            rid = d["race_id"].to_numpy()
            self.boosters_ = boosters
            self.scales_ = [_race_demeaned_sd(b.predict(X), rid) for b in boosters]
        else:
            self.boosters_, self.scales_ = None, None
        return self

    def _members(self) -> list[tuple[object, float]]:
        bs = getattr(self, "boosters_", None)
        if bs:
            return list(zip(bs, self.scales_))
        return [(self.booster, 1.0)]

    def score(self, d: pd.DataFrame) -> np.ndarray:
        X = d.reindex(columns=self.features).astype(float)
        if self.booster is None:
            return self.fallback.utility(X)
        m = self._members()
        return sum(b.predict(X) / s for b, s in m) / len(m)

    def contributions(self, d: pd.DataFrame) -> pd.DataFrame:
        """各馬のスコアへの特徴量ごとの寄与 (LightGBM の SHAP 値)."""
        X = d.reindex(columns=self.features).astype(float)
        if self.booster is None:
            c = self.fallback.coef_ / self.fallback.scale_
            Z = (X - self.fallback.mean_).fillna(0.0)
            return pd.DataFrame(Z.to_numpy() * c, index=d.index, columns=self.features)
        m = self._members()
        v = sum(b.predict(X, pred_contrib=True)[:, :-1] / s for b, s in m) / len(m)
        return pd.DataFrame(v, index=d.index, columns=self.features)

    def importance(self) -> pd.Series:
        if self.booster is None:
            t = self.fallback.coef_table().set_index("feature")["coef_std"].abs()
            return t.sort_values(ascending=False)
        m = self._members()
        if len(m) == 1:
            imp = self.booster.feature_importance("gain")
        else:  # 各モデルの gain を割合にして平均
            gains = [b.feature_importance("gain") for b, _ in m]
            imp = sum(g / max(g.sum(), 1e-12) for g in gains) / len(gains)
        return pd.Series(imp, index=self.features).sort_values(ascending=False)


@dataclass
class Stage2:
    use_market_extra: bool = True
    cols_: list[str] | None = None
    fund_: ConditionalLogit | None = None
    combo_: ConditionalLogit | None = None
    mkt_: ConditionalLogit | None = None
    # --- 改善候補 (既定は従来どおり) ---
    l2: float = 1e-3                    # 合成モデルの L2 正則化 (特徴量は標準化済み)
    l2_grid: tuple[float, ...] | None = None  # 指定すると内側の時系列分割で l2 を選ぶ
    l2_valid_frac: float = 0.25
    market_sq: bool = False             # (log p_mkt)^2 を加え、非線形な本命-大穴バイアスを表現
    s1_race_z: bool = False             # s1 をレース内で標準化してから使う (尺度ずれ対策の候補)
    # 新しいレースほど重く学習する (半減期・日)。後のブロックほど Stage1 の学習データが多く
    # s1 の情報量が増えるので、古いブロックで決めた s1 の重みは小さすぎる可能性がある
    recency_half_life: float | None = None
    l2_: float | None = None

    def _cols(self, d: pd.DataFrame) -> list[str]:
        cols = ["s1", "log_p_mkt"]
        if getattr(self, "market_sq", False):
            cols.append("log_p_mkt_sq")
        if self.use_market_extra:
            cols += [c for c in STAGE2_MARKET_EXTRA if c in d]
        return cols

    def _prep(self, d: pd.DataFrame) -> pd.DataFrame:
        add = {}
        if getattr(self, "market_sq", False):
            add["log_p_mkt_sq"] = d["log_p_mkt"] ** 2
        if getattr(self, "s1_race_z", False):
            g = d.groupby("race_id")["s1"]
            add["s1"] = (d["s1"] - g.transform("mean")) / g.transform("std").replace(0, 1).fillna(1)
        return d.assign(**add) if add else d

    def _choose_l2(self, d: pd.DataFrame) -> float:
        va = _inner_split(d["date"], self.l2_valid_frac) if "date" in d else None
        if va is None or va.sum() < 200 or (~va).sum() < 1000:
            return self.l2
        tr, te = d[~va], d[va]
        best, best_ll = self.l2, np.inf
        for l2 in self.l2_grid:
            m = ConditionalLogit(l2=l2).fit(tr[self.cols_], tr["win"], tr["race_id"], self.cols_)
            rid = te["race_id"].to_numpy()
            ll = race_logloss(m.predict_proba(te[self.cols_], rid), te["win"].to_numpy(), rid)
            if ll < best_ll - 1e-7:
                best, best_ll = l2, ll
        return best

    def _weights(self, d: pd.DataFrame) -> np.ndarray | None:
        hl = getattr(self, "recency_half_life", None)
        if not hl or "date" not in d:
            return None
        age = (d["date"].max() - d["date"]).dt.days.to_numpy(dtype=float)
        return 0.5 ** (age / hl)

    def fit(self, d: pd.DataFrame) -> "Stage2":
        d = self._prep(d[d["p_mkt"].notna()])
        self.cols_ = self._cols(d)
        w = self._weights(d)
        self.fund_ = ConditionalLogit(l2=1e-4).fit(d[["s1"]], d["win"], d["race_id"], ["s1"], weights=w)
        self.mkt_ = ConditionalLogit(l2=1e-4).fit(d[["log_p_mkt"]], d["win"], d["race_id"], ["log_p_mkt"],
                                                 weights=w)
        l2 = self.l2
        if getattr(self, "l2_grid", None):
            l2 = self._choose_l2(d)
        self.l2_ = l2
        self.combo_ = ConditionalLogit(l2=l2).fit(d[self.cols_], d["win"], d["race_id"], self.cols_, weights=w)
        return self

    def predict(self, d: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=d.index)
        d = self._prep(d)
        rid = d["race_id"].to_numpy()
        out["p_fund"] = self.fund_.predict_proba(d[["s1"]], rid)
        # レース内の全馬にオッズがある場合だけ市場と合成 (一部欠けると確率の合計が崩れる)
        has = d["p_mkt"].notna().groupby(d["race_id"]).transform("all")
        out["p_mkt_cal"] = np.nan
        out["p_model"] = out["p_fund"]
        if has.any():
            dh = d[has]
            out.loc[has, "p_mkt_cal"] = self.mkt_.predict_proba(dh[["log_p_mkt"]], dh["race_id"].to_numpy())
            out.loc[has, "p_model"] = self.combo_.predict_proba(dh[self.cols_], dh["race_id"].to_numpy())
        return out


def _blocks(dates: pd.Series, months: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = dates.min().to_period("M").to_timestamp()
    end = dates.max() + pd.Timedelta(days=1)
    edges = list(pd.date_range(start, end + pd.offsets.MonthBegin(months), freq=f"{months}MS"))
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if a <= dates.max()]


def walk_forward(
    X: pd.DataFrame,
    block_months: int = 2,
    warmup_days: int = 90,
    min_train_days: int = 270,
    min_stage2_blocks: int = 2,
    verbose: bool = True,
    stage1_factory: Callable[[], Stage1] | None = None,
    stage2_factory: Callable[[], Stage2] | None = None,
) -> pd.DataFrame:
    """完全サンプル外の予測を作る.

    1) ブロック (既定 2 ヶ月) ごとに、それ以前の全データで Stage1 を学習 → ブロックを予測 (s1)
    2) s1 がサンプル外で揃ったブロックについて、それ以前の s1 で Stage2 を学習 → 予測

    stage1_factory / stage2_factory: 比較実験用 (既定は Stage1() / Stage2())。
    """
    X = oos_stage1(X, block_months, warmup_days, min_train_days, verbose, stage1_factory=stage1_factory)
    return oos_stage2(X, min_stage2_blocks, stage2_factory=stage2_factory)


def oos_stage2(X: pd.DataFrame, min_stage2_blocks: int = 2,
               stage2_factory: Callable[[], Stage2] | None = None) -> pd.DataFrame:
    """oos_stage1 の結果 (s1, block 付き) から、各ブロックをそれ以前の s1 だけで学習した Stage2 で予測."""
    make = stage2_factory or Stage2
    preds = []
    oos_blocks = sorted(X.loc[X["block"] >= 0, "block"].unique())
    for i, k in enumerate(oos_blocks):
        if i < min_stage2_blocks:
            continue
        tr = X[(X["block"] >= 0) & (X["block"] < k)]
        te = X[X["block"] == k]
        s2 = make().fit(tr)
        p = s2.predict(te)
        preds.append(pd.concat([te, p], axis=1))
    if not preds:
        raise ValueError("ウォークフォワードに十分な期間がありません (データ期間を延ばしてください)")
    return pd.concat(preds).reset_index(drop=True)


def oos_stage1(X: pd.DataFrame, block_months: int = 2, warmup_days: int = 90,
               min_train_days: int = 270, verbose: bool = False,
               stage1_factory: Callable[[], Stage1] | None = None,
               models_out: dict | None = None) -> pd.DataFrame:
    """ブロックごとに「それ以前のデータだけ」で Stage1 を学習し、サンプル外スコア s1 を付ける.

    models_out に dict を渡すと {ブロック番号: (ブロック開始日, 学習済み Stage1)} を格納する。
    """
    make = stage1_factory or Stage1
    X = X.sort_values(["date", "race_id", "post"]).reset_index(drop=True)
    t0 = X["date"].min()
    usable = X["date"] >= t0 + pd.Timedelta(days=warmup_days)  # 特徴量が育つまでは学習に使わない
    blocks = _blocks(X["date"], block_months)
    X["s1"] = np.nan
    X["block"] = -1
    for k, (a, b) in enumerate(blocks):
        if a < t0 + pd.Timedelta(days=min_train_days):
            continue
        tr = X[usable & (X["date"] < a)]
        te = (X["date"] >= a) & (X["date"] < b)
        if not te.any():
            continue
        s1 = make().fit(tr)
        X.loc[te, "s1"] = s1.score(X[te])
        X.loc[te, "block"] = k
        if models_out is not None:
            models_out[k] = (a, s1)
        if verbose:
            print(f"  stage1 block {a:%Y-%m}〜: train={len(tr)} test={int(te.sum())}")
    return X


@dataclass
class BaneiModel:
    """本番用: 全データで学習した Stage1 + サンプル外 s1 で学習した Stage2."""

    stage1: Stage1 | None = None
    stage2: Stage2 | None = None
    trained_until: pd.Timestamp | None = None

    def fit(self, X: pd.DataFrame, block_months: int = 2, warmup_days: int = 90) -> "BaneiModel":
        oos = oos_stage1(X, block_months=block_months, warmup_days=warmup_days)
        oos = oos[oos["block"] >= 0]
        if oos.empty:
            raise ValueError("学習期間が短すぎます (目安: 1 年以上)")
        self.stage2 = Stage2().fit(oos)
        t0 = X["date"].min() + pd.Timedelta(days=warmup_days)
        self.stage1 = Stage1().fit(X[X["date"] >= t0])
        self.trained_until = X["date"].max()
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["s1"] = self.stage1.score(X)
        p = self.stage2.predict(X)
        return pd.concat([X, p], axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "BaneiModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def evaluate_probs(P: pd.DataFrame) -> pd.DataFrame:
    """各確率の 1 レースあたり対数損失 (小さいほど良い)."""
    d = P[P["p_mkt"].notna()]
    rows = []
    n = d.groupby("race_id")["post"].transform("size")
    for name, col in [("一様 (1/頭数)", None), ("市場 (オッズそのまま)", "p_mkt"),
                      ("市場 (較正後)", "p_mkt_cal"), ("ファンダメンタル", "p_fund"),
                      ("合成モデル", "p_model")]:
        p = (1.0 / n) if col is None else d[col]
        rows.append({"model": name, "logloss": race_logloss(p, d["win"], d["race_id"])})
    if "_p_true" in d:  # シミュレーションのみ
        rows.append({"model": "(参考) 真の確率", "logloss": race_logloss(d["_p_true"], d["win"], d["race_id"])})
    t = pd.DataFrame(rows)
    base = t.loc[t["model"] == "市場 (オッズそのまま)", "logloss"].iloc[0]
    t["vs_market"] = base - t["logloss"]  # 正 = 市場より良い
    return t
