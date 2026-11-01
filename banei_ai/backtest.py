"""馬券シミュレーション (単勝).

注意:
  * 学習データのオッズは「確定オッズ」。実際に買う時点 (締切前) のオッズとはズレる。
  * ばんえいは売上 (プール) が小さく、自分の購入でオッズが下がる。大口購入ほど期待値は目減りする。
  * よってバックテストの回収率は「上限の目安」と考えること。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BetRule:
    min_ev: float = 1.10        # 期待値 p × オッズ の下限
    min_prob: float = 0.03      # 勝率の下限 (超大穴の過信を避ける)
    max_odds: float = 50.0      # オッズの上限
    min_edge: float = 0.0       # p_model / p_mkt - 1 の下限 (歪みの大きさ)
    max_per_race: int = 2       # 1 レースで買う最大頭数 (EV 上位から)

    def select(self, P: pd.DataFrame, p_col: str = "p_model") -> pd.DataFrame:
        d = P[P["odds_win"].notna()].copy()
        d["ev"] = d[p_col] * d["odds_win"]
        d["edge"] = d[p_col] / d["p_mkt"] - 1
        m = (
            (d["ev"] >= self.min_ev)
            & (d[p_col] >= self.min_prob)
            & (d["odds_win"] <= self.max_odds)
            & (d["edge"] >= self.min_edge)
        )
        d = d[m].sort_values(["race_id", "ev"], ascending=[True, False])
        d = d.groupby("race_id").head(self.max_per_race)
        return d


def _max_drawdown(cum: np.ndarray) -> float:
    if len(cum) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.r_[0.0, cum])
    return float((peak[1:] - cum).max())


def flat_bet(bets: pd.DataFrame, stake: float = 100.0) -> dict:
    if bets.empty:
        return {"bets": 0, "hits": 0, "hit_rate": np.nan, "roi": np.nan,
                "profit": 0.0, "max_dd": 0.0, "avg_odds": np.nan}
    b = bets.sort_values(["date", "race_id"])
    ret = b["win"] * b["odds_win"] * stake
    pnl = ret - stake
    cum = pnl.cumsum().to_numpy()
    return {
        "bets": len(b),
        "hits": int(b["win"].sum()),
        "hit_rate": float(b["win"].mean()),
        "roi": float(ret.sum() / (stake * len(b))),
        "profit": float(pnl.sum()),
        "max_dd": _max_drawdown(cum),
        "avg_odds": float(b["odds_win"].mean()),
    }


def kelly_bet(bets: pd.DataFrame, p_col: str = "p_model", bankroll: float = 100_000.0,
              fraction: float = 0.25, cap: float = 0.03) -> dict:
    """分数ケリー: 賭け率 f = fraction × (p·o − 1)/(o − 1), 1 点あたり資金の cap まで."""
    if bets.empty:
        return {"final_bankroll": bankroll, "growth": 1.0, "max_dd_pct": 0.0}
    b = bets.sort_values(["date", "race_id"])
    bk = bankroll
    peak = bk
    max_dd = 0.0
    for _, race in b.groupby(["date", "race_id"], sort=False):
        base = bk  # 同一レース内は同じ資金を基準
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


def threshold_grid(P: pd.DataFrame, p_col: str = "p_model",
                   evs=(1.0, 1.05, 1.1, 1.2, 1.3, 1.5)) -> pd.DataFrame:
    rows = []
    for ev in evs:
        rule = BetRule(min_ev=ev)
        b = rule.select(P, p_col)
        r = {"min_ev": ev, **flat_bet(b)}
        r.update(kelly_bet(b, p_col))
        rows.append(r)
    return pd.DataFrame(rows)


def split_half_check(P: pd.DataFrame, p_col: str = "p_model",
                     evs=(1.0, 1.05, 1.1, 1.2, 1.3), min_bets: int = 30) -> dict:
    """前半でしきい値を選び、後半にそのまま適用 (しきい値選びの楽観バイアスを除いた成績)."""
    races = P[["date", "race_id"]].drop_duplicates().sort_values("date")
    cut = races["date"].iloc[len(races) // 2]
    first, second = P[P["date"] < cut], P[P["date"] >= cut]
    best, best_roi = None, -np.inf
    for ev in evs:
        r = flat_bet(BetRule(min_ev=ev).select(first, p_col))
        if r["bets"] >= min_bets and r["roi"] > best_roi:
            best, best_roi = ev, r["roi"]
    if best is None:
        return {"cut": cut, "chosen_min_ev": None}
    b2 = BetRule(min_ev=best).select(second, p_col)
    r2 = flat_bet(b2)
    lo, hi = bootstrap_roi(b2)
    return {"cut": cut, "chosen_min_ev": best, "first_half_roi": best_roi,
            "second_half_bets": r2["bets"], "second_half_roi": r2["roi"],
            "second_half_roi_lo90": lo, "second_half_roi_hi90": hi}


def by_period(P: pd.DataFrame, rule: BetRule, p_col: str = "p_model", freq: str = "QS") -> pd.DataFrame:
    """期間別の安定性 (どの時期でも勝てているか)."""
    b = rule.select(P, p_col)
    if b.empty:
        return pd.DataFrame()
    b["period"] = b["date"].dt.to_period(freq[0])
    rows = []
    for k, g in b.groupby("period"):
        rows.append({"period": str(k), **flat_bet(g)})
    return pd.DataFrame(rows)


def calibration(P: pd.DataFrame, p_col: str = "p_model", bins=(0, .02, .05, .1, .15, .2, .3, .45, 1.0)) -> pd.DataFrame:
    d = P[P[p_col].notna()].copy()
    d["bin"] = pd.cut(d[p_col], bins)
    g = d.groupby("bin", observed=True)
    return pd.DataFrame({"n": g.size(), "pred": g[p_col].mean(), "actual": g["win"].mean()})


def bootstrap_roi(bets: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """レース単位ブートストラップによる回収率の 90% 区間."""
    if bets.empty:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    g = bets.assign(ret=bets["win"] * bets["odds_win"]).groupby("race_id").agg(
        ret=("ret", "sum"), n=("ret", "size"))
    r, n = g["ret"].to_numpy(), g["n"].to_numpy()
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    rois = r[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(np.quantile(rois, 0.05)), float(np.quantile(rois, 0.95))
