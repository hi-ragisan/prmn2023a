"""二段階モデル (Benter 方式).

Stage1  ファンダメンタル: LightGBM LambdaRank (オッズは使わない)
        → 馬・騎手・負担重量・馬場水分などから「強さスコア」s1 を出す
Stage2  合成: 条件付きロジット  u = a·s1 + b·log(p_mkt) + c·(市場系特徴量)
        → 市場の情報とモデルの情報を最適に混ぜた勝率 p_model

「オッズの歪み」= p_model / p_mkt。これが大きく、期待値 p_model × オッズ > 1 の馬が狙い目。

Stage2 の学習には Stage1 の *サンプル外* スコアだけを使う (in-sample スコアは過信を生む)。
評価は時系列のウォークフォワードで行い、未来の情報は一切使わない。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class Stage1:
    features: list[str] = field(default_factory=lambda: list(FUNDAMENTAL_FEATURES))
    params: dict = field(default_factory=lambda: dict(
        objective="lambdarank",
        learning_rate=0.03,
        num_leaves=15,
        min_data_in_leaf=60,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=5.0,
        lambdarank_truncation_level=5,
        verbose=-1,
        seed=config.RANDOM_SEED,
    ))
    n_rounds: int = 400
    booster: object = None
    fallback: ConditionalLogit | None = None

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
        self.booster = lgb.train(self.params, ds, num_boost_round=self.n_rounds)
        return self

    def score(self, d: pd.DataFrame) -> np.ndarray:
        X = d.reindex(columns=self.features).astype(float)
        if self.booster is None:
            return self.fallback.utility(X)
        return self.booster.predict(X)

    def importance(self) -> pd.Series:
        if self.booster is None:
            t = self.fallback.coef_table().set_index("feature")["coef_std"].abs()
            return t.sort_values(ascending=False)
        imp = self.booster.feature_importance("gain")
        return pd.Series(imp, index=self.features).sort_values(ascending=False)


@dataclass
class Stage2:
    use_market_extra: bool = True
    cols_: list[str] | None = None
    fund_: ConditionalLogit | None = None
    combo_: ConditionalLogit | None = None
    mkt_: ConditionalLogit | None = None

    def _cols(self, d: pd.DataFrame) -> list[str]:
        cols = ["s1", "log_p_mkt"]
        if self.use_market_extra:
            cols += [c for c in STAGE2_MARKET_EXTRA if c in d]
        return cols

    def fit(self, d: pd.DataFrame) -> "Stage2":
        d = d[d["p_mkt"].notna()]
        self.cols_ = self._cols(d)
        self.fund_ = ConditionalLogit(l2=1e-4).fit(d[["s1"]], d["win"], d["race_id"], ["s1"])
        self.mkt_ = ConditionalLogit(l2=1e-4).fit(d[["log_p_mkt"]], d["win"], d["race_id"], ["log_p_mkt"])
        self.combo_ = ConditionalLogit(l2=1e-3).fit(d[self.cols_], d["win"], d["race_id"], self.cols_)
        return self

    def predict(self, d: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=d.index)
        rid = d["race_id"].to_numpy()
        out["p_fund"] = self.fund_.predict_proba(d[["s1"]], rid)
        has = d["p_mkt"].notna()
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
) -> pd.DataFrame:
    """完全サンプル外の予測を作る.

    1) ブロック (既定 2 ヶ月) ごとに、それ以前の全データで Stage1 を学習 → ブロックを予測 (s1)
    2) s1 がサンプル外で揃ったブロックについて、それ以前の s1 で Stage2 を学習 → 予測
    """
    X = oos_stage1(X, block_months, warmup_days, min_train_days, verbose)
    preds = []
    oos_blocks = sorted(X.loc[X["block"] >= 0, "block"].unique())
    for i, k in enumerate(oos_blocks):
        if i < min_stage2_blocks:
            continue
        tr = X[(X["block"] >= 0) & (X["block"] < k)]
        te = X[X["block"] == k]
        s2 = Stage2().fit(tr)
        p = s2.predict(te)
        preds.append(pd.concat([te, p], axis=1))
    if not preds:
        raise ValueError("ウォークフォワードに十分な期間がありません (データ期間を延ばしてください)")
    return pd.concat(preds).reset_index(drop=True)


def oos_stage1(X: pd.DataFrame, block_months: int = 2, warmup_days: int = 90,
               min_train_days: int = 270, verbose: bool = False) -> pd.DataFrame:
    """ブロックごとに「それ以前のデータだけ」で Stage1 を学習し、サンプル外スコア s1 を付ける."""
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
        s1 = Stage1().fit(tr)
        X.loc[te, "s1"] = s1.score(X[te])
        X.loc[te, "block"] = k
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
