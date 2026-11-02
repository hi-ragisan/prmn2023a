"""時点整合 (point-in-time) な特徴量生成.

開催日ごとに時系列順に処理し、その日のレースの特徴量は「前日までの結果だけ」から作る。
(当日結果 → 当日特徴量 のリークを構造的に防ぐ)

特徴量の 3 つの柱:
  h_* : 馬 (レーティング・近走・着差・負担重量・馬体重・馬場水分適性・市場超過)
  j_* : 騎手 (経験ベイズで縮小した勝率、市場期待に対する超過勝利、人気度)
  t_* : 調教師
  jh_*: 騎手×馬コンビ
  r_* : レース内での相対値 (ランキング学習に効く)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, market, schema

CLASS_RANK = {
    "オープン": 10, "OP": 10, "重賞": 10,
    "A1": 9, "A2": 8, "B1": 7, "B2": 6, "B3": 5, "B4": 4, "C1": 3, "C2": 2,
}


def class_rank(s: pd.Series) -> pd.Series:
    def f(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
            return np.nan
        v = str(v)
        for k, r in CLASS_RANK.items():
            if k in v:
                return r
        if "歳" in v:
            return 3
        return np.nan

    return s.map(f).astype(float)


class Decayed:
    """時間減衰付きの累積カウンタ (ベクトル)."""

    __slots__ = ("hl", "t", "v")

    def __init__(self, k: int, half_life_days: float):
        self.hl = half_life_days
        self.t: pd.Timestamp | None = None
        self.v = np.zeros(k)

    def _decay(self, t: pd.Timestamp) -> None:
        if self.t is not None:
            dt = (t - self.t).days
            if dt > 0:
                self.v *= 0.5 ** (dt / self.hl)
        self.t = t

    def get(self, t: pd.Timestamp) -> np.ndarray:
        self._decay(t)
        return self.v

    def add(self, t: pd.Timestamp, x: np.ndarray) -> None:
        self._decay(t)
        self.v += x


# 騎手・調教師カウンタのベクトル要素
#   0 騎乗数, 1 勝利, 2 3着内, 3 期待勝利(1/頭数), 4 期待3着内(3/頭数),
#   5 市場期待勝利(Σp_mkt), 6 人気度(Σ p_mkt*頭数), 7 オッズ有り騎乗数, 8 単勝回収額
NV = 9


@dataclass
class HorseRun:
    date: pd.Timestamp
    pct: float          # 着順パーセンタイル (0=1着, 1=最下位)
    win: int
    top3: int
    gap: float          # 1着とのタイム差 (秒)
    cw: float           # 負担重量
    cw_rel: float       # レース平均との差
    moist_bin: int
    p_mkt: float
    n: int
    cls: float
    jockey: str
    perf: float         # 0.5 - pct (大きいほど良い)
    time: float = np.nan    # 走破タイム (秒)
    moist: float = np.nan   # 馬場水分


@dataclass
class HorseState:
    runs: list[HorseRun] = field(default_factory=list)
    rating: float = 1500.0


def _moist_bin(m: float) -> int:
    if m is None or np.isnan(m):
        return -1
    return int(np.searchsorted(config.MOISTURE_BINS[1:-1], m, side="right"))


def _beta_shrink(succ: float, trials_exp: float, prior: float) -> float:
    """A/E 比を事前 A/E=1 に向けて縮小: (succ + prior) / (exp + prior)."""
    return (succ + prior) / (trials_exp + prior)


def _elo_update(ratings: np.ndarray, finish: np.ndarray, k: float = 24.0) -> np.ndarray:
    n = len(ratings)
    if n < 2:
        return np.zeros(n)
    diff = ratings[None, :] - ratings[:, None]          # R_j - R_i
    e = 1.0 / (1.0 + 10 ** (diff / 400.0))              # i が j に勝つ期待
    s = (finish[:, None] < finish[None, :]).astype(float)
    s += 0.5 * (finish[:, None] == finish[None, :])
    np.fill_diagonal(e, 0)
    np.fill_diagonal(s, 0)
    return k * (s - e).sum(axis=1) / (n - 1)


class TimeBaseline:
    """走破タイムの基準 = f(馬場水分, 負担重量) を過去データだけで逐次推定 (リッジ回帰).

    スピード指数 = 基準タイム - 実タイム (秒)。正なら基準より速い。
    クラスは説明変数に入れない (上級馬が速いことを指数に残すため)。
    """

    def __init__(self, lam: float = 1.0):
        self.k = 6
        self.xtx = np.eye(self.k) * lam
        self.xty = np.zeros(self.k)
        self.beta = np.zeros(self.k)
        self.n = 0

    @staticmethod
    def design(moist, cw) -> np.ndarray:
        m = np.nan_to_num(np.asarray(moist, dtype=float), nan=2.0)
        m = np.clip(m, 0, 12)
        w = (np.asarray(cw, dtype=float) - 650.0) / 100.0
        w = np.nan_to_num(w, nan=0.0)
        return np.column_stack([np.ones_like(m), m, m ** 2, np.log1p(m), w, w * m])

    def add(self, moist, cw, time) -> None:
        t = np.asarray(time, dtype=float)
        ok = np.isfinite(t)
        if not ok.any():
            return
        X = self.design(np.asarray(moist)[ok], np.asarray(cw)[ok])
        # 障害で止まった極端な遅いタイムは基準推定から外す
        self.xtx += X.T @ X
        self.xty += X.T @ t[ok]
        self.n += int(ok.sum())

    def refit(self) -> None:
        if self.n >= 50:
            self.beta = np.linalg.solve(self.xtx, self.xty)

    def speed(self, moist, cw, time) -> np.ndarray:
        if self.n < 50:
            return np.full(np.shape(np.atleast_1d(time)), np.nan)
        pred = self.design(np.atleast_1d(moist), np.atleast_1d(cw)) @ self.beta
        return np.clip(pred - np.atleast_1d(np.asarray(time, dtype=float)), -40, 40)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """スキーマ正規化 → 出走馬のみ → 市場確率 → 目的変数."""
    d = schema.runners_only(schema.normalize(df))
    d = market.add_market_probs(d)
    d["class_rank"] = class_rank(d["race_class"])
    n = d["n_runners"]
    fin = d["finish"].where(d["status"] == "ok")
    # 中止は最下位扱い。出馬表 (status=entry, 結果未確定) は NaN のまま
    d["finish_eff"] = fin.fillna(n).where(d["status"] != "entry")
    d["win"] = (fin == 1).astype(int)
    d["top3"] = (fin <= 3).astype(int)
    d["finish_pct"] = (d["finish_eff"] - 1) / (n - 1).clip(lower=1)
    win_time = d["time_sec"].where(d["finish"] == 1).groupby(d["race_id"]).transform("min")
    gap = d["time_sec"] - win_time
    max_gap = gap.groupby(d["race_id"]).transform("max")
    d["time_gap"] = gap.fillna(max_gap + 10).fillna(30.0)
    d["cw_rel"] = d["carried_weight"] - d.groupby("race_id")["carried_weight"].transform("mean")
    d["cw_ratio"] = d["carried_weight"] / d["horse_weight"]
    d["moist_bin"] = d["moisture"].map(_moist_bin)
    return d


def build_features(df: pd.DataFrame, upto_results: bool = True) -> pd.DataFrame:
    """全レースの特徴量を作る.

    upto_results=True の場合、各日の特徴量計算後にその日の結果で状態を更新する。
    予測用の出馬表 (結果なし) を末尾に連結して渡せば、その分は更新されず特徴量だけ出る。
    """
    d = prepare(df)
    horses: dict[str, HorseState] = defaultdict(HorseState)
    jockeys: dict[str, Decayed] = {}
    trainers: dict[str, Decayed] = {}
    combo: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(3))
    tb = TimeBaseline()
    jmoist: dict[tuple[str, int], Decayed] = {}  # 騎手×水分帯: [勝利, 期待勝利]

    def jget(key: str, t) -> np.ndarray:
        if key not in jockeys:
            jockeys[key] = Decayed(NV, config.JOCKEY_HALF_LIFE_DAYS)
        return jockeys[key].get(t)

    def tget(key: str, t) -> np.ndarray:
        if key not in trainers:
            trainers[key] = Decayed(NV, config.TRAINER_HALF_LIFE_DAYS)
        return trainers[key].get(t)

    feats: list[dict] = []
    for date, day in d.groupby("date", sort=True):
        # ---------- 特徴量 (前日までの状態から) ----------
        for r in day.itertuples(index=False):
            hs = horses[r.horse_id]
            runs = hs.runs
            f: dict = {"_idx": r.Index if hasattr(r, "Index") else None}
            f["race_id"] = r.race_id
            f["post"] = r.post
            nr = len(runs)
            f["h_n_runs"] = nr
            f["h_rating"] = hs.rating
            if nr:
                last = runs[-1]
                f["h_days_since"] = (date - last.date).days
                pcts = np.array([x.pct for x in runs])
                gaps = np.array([x.gap for x in runs])
                ages = np.array([(date - x.date).days for x in runs], dtype=float)
                w = 0.5 ** (ages / config.HORSE_FORM_HALF_LIFE_DAYS)
                f["h_last1_pct"] = pcts[-1]
                f["h_last2_pct"] = pcts[-2] if nr >= 2 else np.nan
                f["h_last3_pct"] = pcts[-3] if nr >= 3 else np.nan
                f["h_avg5_pct"] = pcts[-5:].mean()
                f["h_ewm_pct"] = float((pcts * w).sum() / w.sum())
                f["h_last_gap"] = gaps[-1]
                f["h_avg3_gap"] = gaps[-3:].mean()
                f["h_best5_gap"] = gaps[-5:].min()
                wins = sum(x.win for x in runs)
                top3 = sum(x.top3 for x in runs)
                expw = sum(1.0 / x.n for x in runs)
                f["h_win_ae"] = _beta_shrink(wins, expw, 2.0)
                f["h_top3_rate"] = (top3 + 0.3 * 3) / (nr + 3)
                pm = [(x.win, x.p_mkt) for x in runs[-10:] if not np.isnan(x.p_mkt)]
                if pm:
                    f["h_mkt_ae"] = _beta_shrink(sum(a for a, _ in pm), sum(b for _, b in pm), 1.5)
                    f["h_last_pmkt_share"] = runs[-1].p_mkt * runs[-1].n
                # スピード指数 (水分・負担重量補正タイム)。現時点の基準で過去走を再評価
                tms = np.array([x.time for x in runs[-8:]])
                if np.isfinite(tms).any():
                    spd = tb.speed([x.moist for x in runs[-8:]], [x.cw for x in runs[-8:]], tms)
                    ok = np.isfinite(tms)
                    sp = spd[ok]
                    ww = w[-8:][ok]
                    f["h_spd_last"] = sp[-1]
                    f["h_spd_avg3"] = sp[-3:].mean()
                    f["h_spd_best5"] = sp[-5:].max()
                    f["h_spd_ewm"] = float((sp * ww).sum() / ww.sum())
                f["h_cw_change"] = r.carried_weight - last.cw
                # レース内相対の重量の前走比。実際のばんえいは基礎重量が開催回ごとに約 10kg ずつ
                # 上がる (年度初めに戻る) ので、生の重量差 h_cw_change には季節の要因が混ざる。
                # レース平均との差どうしを比べれば季節の上下が打ち消され、ハンデの見直しだけが残る。
                f["h_cw_rel_change"] = r.cw_rel - last.cw_rel
                f["h_cls_change"] = r.class_rank - last.cls if not np.isnan(last.cls) else np.nan
                f["h_jockey_change"] = float(str(r.jockey_id) != last.jockey)
                # 馬場水分適性: 同じ水分帯での成績 - 全体平均 (縮小)
                perf_all = np.array([x.perf for x in runs])
                mb = np.array([x.moist_bin for x in runs])
                same = perf_all[mb == r.moist_bin]
                mu = perf_all.mean()
                f["h_moist_apt"] = (same.sum() + 3 * mu) / (len(same) + 3) - mu
                f["h_moist_n"] = len(same)
                # 重い斤量での成績 (斤量差を背負った時の着順)
                cwr = np.array([x.cw_rel for x in runs])
                heavy = perf_all[cwr > 5]
                f["h_heavy_perf"] = (heavy.sum() + 3 * mu) / (len(heavy) + 3) - mu
            # 騎手
            jv = jget(r.jockey_id, date)
            f["j_rides"] = jv[0]
            f["j_win_ae"] = _beta_shrink(jv[1], jv[3], config.JOCKEY_PRIOR_STRENGTH / 10)
            f["j_top3_ae"] = _beta_shrink(jv[2], jv[4], config.JOCKEY_PRIOR_STRENGTH / 4)
            f["j_mkt_ae"] = _beta_shrink(jv[1], jv[5], config.JOCKEY_PRIOR_STRENGTH / 10)
            f["j_popularity"] = (jv[6] + 10) / (jv[7] + 10)  # 1 = 平均的な人気
            f["j_roi"] = (jv[8] + 0.8 * 30) / (jv[7] + 30)
            # 騎手×馬場水分: 今回の水分帯での A/E を、騎手全体の A/E に対する比で
            jm = jmoist.get((r.jockey_id, r.moist_bin))
            if jm is not None:
                v = jm.get(date)
                k = 4.0
                f["j_moist_rel"] = ((v[0] + k * f["j_win_ae"]) / (v[1] + k)) / f["j_win_ae"]
            else:
                f["j_moist_rel"] = 1.0
            # 調教師
            tv = tget(str(r.trainer_id), date)
            f["t_win_ae"] = _beta_shrink(tv[1], tv[3], config.TRAINER_PRIOR_STRENGTH / 10)
            f["t_mkt_ae"] = _beta_shrink(tv[1], tv[5], config.TRAINER_PRIOR_STRENGTH / 10)
            # コンビ
            cv = combo[(r.jockey_id, r.horse_id)]
            f["jh_rides"] = cv[0]
            f["jh_win_ae"] = _beta_shrink(cv[1], cv[2], config.COMBO_PRIOR_STRENGTH / 4)
            feats.append(f)

        if not upto_results:
            continue
        done = day[day["finish_eff"].notna() & (day["status"] != "entry")]
        if done.empty:
            continue
        # ---------- 状態更新 (当日結果) ----------
        for rid, race in done.groupby("race_id", sort=False):
            hid = race["horse_id"].to_numpy()
            fin = race["finish_eff"].to_numpy(dtype=float)
            ratings = np.array([horses[h].rating for h in hid])
            delta = _elo_update(ratings, fin)
            for i, row in enumerate(race.itertuples(index=False)):
                hs = horses[row.horse_id]
                hs.rating += delta[i]
                pct = float(row.finish_pct)
                hs.runs.append(HorseRun(
                    date=date, pct=pct, win=int(row.win), top3=int(row.top3),
                    gap=float(row.time_gap), cw=float(row.carried_weight),
                    cw_rel=float(row.cw_rel) if not np.isnan(row.cw_rel) else 0.0,
                    moist_bin=int(row.moist_bin), p_mkt=float(row.p_mkt),
                    n=int(row.n_runners), cls=float(row.class_rank),
                    jockey=str(row.jockey_id), perf=0.5 - pct,
                    time=float(row.time_sec) if row.status == "ok" else np.nan,
                    moist=float(row.moisture),
                ))
                n = float(row.n_runners)
                has_odds = not np.isnan(row.p_mkt)
                vec = np.array([
                    1.0, row.win, row.top3, 1.0 / n, min(3.0, n) / n,
                    row.p_mkt if has_odds else 1.0 / n,
                    row.p_mkt * n if has_odds else 0.0,
                    float(has_odds),
                    row.win * row.odds_win if has_odds else 0.0,
                ])
                jget(row.jockey_id, date)
                jockeys[row.jockey_id].add(date, vec)
                tget(str(row.trainer_id), date)
                trainers[str(row.trainer_id)].add(date, vec)
                cv = combo[(row.jockey_id, row.horse_id)]
                cv += np.array([1.0, row.win, 1.0 / n])
                jk = (row.jockey_id, int(row.moist_bin))
                if jk not in jmoist:
                    jmoist[jk] = Decayed(2, config.JOCKEY_HALF_LIFE_DAYS * 2)
                jmoist[jk].add(date, np.array([row.win, 1.0 / n]))
            ok = race["status"].eq("ok").to_numpy()
            tb.add(race["moisture"].to_numpy()[ok], race["carried_weight"].to_numpy()[ok],
                   race["time_sec"].to_numpy()[ok])
        tb.refit()

    F = pd.DataFrame(feats)
    F = F.drop(columns=["_idx"])
    out = d.merge(F, on=["race_id", "post"], how="left")
    out = add_race_relative(out)
    return out


REL_COLS = [
    "h_rating", "h_ewm_pct", "h_avg3_gap", "h_win_ae", "j_win_ae", "j_top3_ae",
    "h_moist_apt", "t_win_ae", "h_best5_gap", "h_spd_ewm", "h_spd_best5",
]


def add_race_relative(d: pd.DataFrame) -> pd.DataFrame:
    """レース内での相対値 (平均との差・順位)."""
    g = d.groupby("race_id")
    for c in REL_COLS:
        if c not in d:
            continue
        d[f"r_{c}_diff"] = d[c] - g[c].transform("mean")
        asc = c.endswith("pct") or c.endswith("gap")  # 小さいほど良い指標
        d[f"r_{c}_rank"] = g[c].rank(ascending=asc, method="average")
    d["r_cw_rank"] = g["carried_weight"].rank(ascending=False, method="average")
    d["age_rel"] = d["age"] - g["age"].transform("mean")
    return d


# Stage1 (オッズを使わない) で使う特徴量
FUNDAMENTAL_FEATURES = [
    "n_runners", "moisture", "class_rank", "age", "age_rel",
    "carried_weight", "cw_rel", "cw_ratio", "horse_weight", "horse_weight_diff",
    "h_n_runs", "h_rating", "h_days_since", "h_last1_pct", "h_last2_pct", "h_last3_pct",
    "h_avg5_pct", "h_ewm_pct", "h_last_gap", "h_avg3_gap", "h_best5_gap",
    "h_win_ae", "h_top3_rate", "h_cw_change", "h_cls_change", "h_jockey_change",
    "h_moist_apt", "h_moist_n", "h_heavy_perf",
    "h_spd_last", "h_spd_avg3", "h_spd_best5", "h_spd_ewm",
    "j_rides", "j_win_ae", "j_top3_ae", "j_moist_rel", "t_win_ae",
    "jh_rides", "jh_win_ae",
    "r_cw_rank",
] + [f"r_{c}_{k}" for c in REL_COLS for k in ("diff", "rank")] + [
    # 季節に頑健な重量の前走比 (docs/model_experiments.md の (f))。
    # 評価したときと同じ列順にするため末尾に置く (LightGBM の feature_fraction は列順に依存する)
    "h_cw_rel_change",
]

# 市場との関係を表す特徴量 (Stage2 / 歪み分析で使用。Stage1 には入れない)
MARKET_FEATURES = [
    "log_p_mkt", "h_mkt_ae", "h_last_pmkt_share", "j_mkt_ae", "j_popularity", "j_roi", "t_mkt_ae",
]


# 要因説明用の日本語ラベル
FEATURE_LABELS = {
    "n_runners": "頭数", "moisture": "馬場水分", "class_rank": "クラス", "age": "馬齢",
    "age_rel": "馬齢(相対)", "carried_weight": "負担重量", "cw_rel": "負担重量(レース内差)",
    "cw_ratio": "負担重量/馬体重", "horse_weight": "馬体重", "horse_weight_diff": "馬体重増減",
    "h_n_runs": "出走数", "h_rating": "馬レーティング", "h_days_since": "休養明け日数",
    "h_last1_pct": "前走着順", "h_last2_pct": "2走前着順", "h_last3_pct": "3走前着順",
    "h_avg5_pct": "近5走着順", "h_ewm_pct": "近走着順(加重)", "h_last_gap": "前走着差",
    "h_avg3_gap": "近3走着差", "h_best5_gap": "近5走最小着差", "h_win_ae": "馬の勝率",
    "h_top3_rate": "馬の複勝率", "h_cw_change": "負担重量の前走比",
    "h_cw_rel_change": "負担重量の前走比(レース内相対)", "h_cls_change": "クラス昇降",
    "h_jockey_change": "乗り替わり", "h_moist_apt": "馬場水分適性", "h_moist_n": "同水分帯経験",
    "h_heavy_perf": "重い斤量での成績", "h_spd_last": "前走スピード指数",
    "h_spd_avg3": "近3走スピード指数", "h_spd_best5": "近5走最高スピード指数",
    "h_spd_ewm": "スピード指数(加重)", "j_rides": "騎手騎乗数", "j_win_ae": "騎手の勝率",
    "j_top3_ae": "騎手の複勝率", "t_win_ae": "調教師の勝率", "jh_rides": "コンビ騎乗数",
    "jh_win_ae": "コンビ実績", "r_cw_rank": "負担重量順位", "j_moist_rel": "騎手の水分帯相性",
}


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("r_"):
        for suf, tag in (("_diff", "レース内差"), ("_rank", "レース内順位")):
            if name.endswith(suf):
                base = name[2: -len(suf)]
                return f"{FEATURE_LABELS.get(base, base)}({tag})"
    return name


# 要因説明のグループ (関連する特徴量をまとめて「何が効いているか」を読みやすくする)
EXPLAIN_GROUPS = [
    ("負担重量", ["carried_weight", "cw_rel", "cw_ratio", "cw"]),
    ("斤量の前走比", ["h_cw_change", "h_cw_rel_change"]),
    ("重い斤量での実績", ["h_heavy_perf"]),
    ("馬齢", ["age", "age_rel"]),
    ("騎手の腕", ["j_win_ae", "j_top3_ae", "j_rides"]),
    ("騎手×馬場水分", ["j_moist_rel"]),
    ("騎手×馬コンビ", ["jh_rides", "jh_win_ae"]),
    ("調教師", ["t_win_ae"]),
    ("馬の実績・格", ["h_rating", "h_win_ae", "h_top3_rate", "h_n_runs"]),
    ("近走着順", ["h_last1_pct", "h_last2_pct", "h_last3_pct", "h_avg5_pct", "h_ewm_pct"]),
    ("タイム・スピード指数", ["h_last_gap", "h_avg3_gap", "h_best5_gap",
                     "h_spd_last", "h_spd_avg3", "h_spd_best5", "h_spd_ewm"]),
    ("馬場水分適性", ["h_moist_apt", "h_moist_n", "moisture"]),
    ("馬体重", ["horse_weight", "horse_weight_diff"]),
    ("休養明け", ["h_days_since"]),
    ("クラス昇降", ["class_rank", "h_cls_change"]),
    ("乗り替わり", ["h_jockey_change"]),
    ("頭数", ["n_runners"]),
]
_GROUP_OF = {f: g for g, fs in EXPLAIN_GROUPS for f in fs}


def explain_group(name: str) -> str:
    base = name
    if name.startswith("r_") and (name.endswith("_diff") or name.endswith("_rank")):
        base = name[2:-5]
    return _GROUP_OF.get(base, feature_label(base))
