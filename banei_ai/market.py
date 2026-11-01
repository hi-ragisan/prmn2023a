"""オッズ → 市場確率、および「オッズの歪み」の分析.

歪み (distortion) の測り方:
  A/E 比 = 実際の勝利数 / 市場が予想した勝利数 (Σ p_mkt)
    > 1 … 市場がその集団を過小評価 (買い材料)
    < 1 … 市場がその集団を過大評価 (消し材料)
  回収率 = Σ(的中時のオッズ) / ベット数 (単勝を全頭 100 円ずつ買った場合)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def add_market_probs(df: pd.DataFrame) -> pd.DataFrame:
    """単勝オッズから市場確率 p_mkt (レース内で和が 1) を計算して付与."""
    out = df.copy()
    inv = 1.0 / out["odds_win"].where(out["odds_win"] > 0)
    s = inv.groupby(out["race_id"]).transform("sum")
    out["p_mkt"] = inv / s
    out["overround"] = s  # 1 / 払戻率 に相当
    out["log_p_mkt"] = np.log(out["p_mkt"].clip(lower=1e-4))
    if "popularity" in out:
        miss = out["popularity"].isna() & out["odds_win"].notna()
        if miss.any():
            full_rank = out.groupby("race_id")["odds_win"].rank(method="min")
            out.loc[miss, "popularity"] = full_rank[miss]
    return out


def _ae_interval(wins: float, expected: float, level: float = 0.9) -> tuple[float, float]:
    """A/E 比のポアソン信頼区間 (Garwood)."""
    if expected <= 0:
        return (np.nan, np.nan)
    a = (1 - level) / 2
    lo = stats.chi2.ppf(a, 2 * wins) / 2 if wins > 0 else 0.0
    hi = stats.chi2.ppf(1 - a, 2 * (wins + 1)) / 2
    return lo / expected, hi / expected


def distortion_table(
    df: pd.DataFrame,
    by: str | list[str],
    min_bets: int = 30,
    win_col: str = "win",
) -> pd.DataFrame:
    """グループごとの A/E 比・回収率・信頼区間."""
    d = df[df["p_mkt"].notna()].copy()
    d["_ret"] = d[win_col] * d["odds_win"]
    g = d.groupby(by, observed=True)
    t = pd.DataFrame({
        "bets": g.size(),
        "wins": g[win_col].sum(),
        "exp_wins": g["p_mkt"].sum(),
        "roi": g["_ret"].mean(),
        "avg_odds": g["odds_win"].mean(),
    })
    t = t[t["bets"] >= min_bets]
    t["ae"] = t["wins"] / t["exp_wins"]
    ci = [_ae_interval(w, e) for w, e in zip(t["wins"], t["exp_wins"])]
    t["ae_lo90"] = [c[0] for c in ci]
    t["ae_hi90"] = [c[1] for c in ci]
    # 有意性: 区間が 1 をまたがなければ「歪みあり」
    t["signal"] = np.where(t["ae_lo90"] > 1, "過小評価", np.where(t["ae_hi90"] < 1, "過大評価", ""))
    return t.sort_values("ae", ascending=False)


ODDS_BINS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0, 10000.0]


def favorite_longshot_table(df: pd.DataFrame) -> pd.DataFrame:
    """オッズ帯別の A/E と回収率 (大穴の過剰人気 = favorite-longshot bias の確認)."""
    d = df.copy()
    d["odds_band"] = pd.cut(d["odds_win"], ODDS_BINS, right=False)
    return distortion_table(d, "odds_band", min_bets=1).sort_index()


def popularity_table(df: pd.DataFrame) -> pd.DataFrame:
    return distortion_table(df, "popularity", min_bets=1).sort_index()


def fit_power_calibration(df: pd.DataFrame) -> float:
    """p ∝ p_mkt^α の α を最尤推定.

    α > 1: 本命が過小評価・大穴が過大評価 (典型的な favorite-longshot bias)
    α < 1: 逆
    """
    d = df[df["p_mkt"].notna()]
    d = d[d.groupby("race_id")["win"].transform("sum") > 0].sort_values("race_id")
    lp = d["log_p_mkt"].to_numpy()
    w = d["win"].to_numpy()
    _, starts = np.unique(d["race_id"].to_numpy(), return_index=True)

    def nll(a: float) -> float:
        u = a * lp
        m = np.maximum.reduceat(u, starts)
        mm = np.repeat(m, np.diff(np.append(starts, len(u))))
        lse = np.log(np.add.reduceat(np.exp(u - mm), starts)) + m
        return -(np.sum(u * w) - np.sum(lse))

    from scipy.optimize import minimize_scalar

    r = minimize_scalar(nll, bounds=(0.3, 3.0), method="bounded")
    return float(r.x)
