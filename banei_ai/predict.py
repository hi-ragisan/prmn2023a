"""当日予測: 過去データ + 出馬表 (+現在オッズ) → 勝率・歪み・期待値・推奨買い目."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BetRule
from .clogit import harville_top3
from .features import build_features
from .model import BaneiModel


def predict_card(history: pd.DataFrame, card: pd.DataFrame, model: BaneiModel,
                 rule: BetRule | None = None, bankroll: float = 100_000.0,
                 kelly_fraction: float = 0.25, kelly_cap: float = 0.03) -> pd.DataFrame:
    """card: status='entry' の出馬表 (odds_win は締切直前の値を入れるほど良い)."""
    rule = rule or BetRule()
    card = card.copy()
    card["status"] = "entry"
    hist = history[history["date"] < card["date"].min()]
    allx = pd.concat([hist, card], ignore_index=True)
    X = build_features(allx)
    X = X[X["status"] == "entry"]
    P = model.predict(X)
    P["ev"] = P["p_model"] * P["odds_win"]
    P["edge"] = P["p_model"] / P["p_mkt"] - 1
    P["p_top3"] = np.nan
    for rid, g in P.groupby("race_id"):
        P.loc[g.index, "p_top3"] = harville_top3(g["p_model"].to_numpy())
    P["fair_odds"] = 1.0 / P["p_model"]
    chosen = rule.select(P) if P["odds_win"].notna().any() else P.iloc[0:0]
    P["bet"] = P.index.isin(chosen.index)
    f = kelly_fraction * (P["ev"] - 1) / (P["odds_win"] - 1)
    P["kelly_stake"] = np.where(P["bet"], np.floor(np.clip(f, 0, kelly_cap) * bankroll / 100) * 100, 0)
    P["rank_model"] = P.groupby("race_id")["p_model"].rank(ascending=False, method="min")
    return P.sort_values(["race_no", "post"])


DISPLAY = [
    ("race_no", "R"), ("post", "馬番"), ("horse_name", "馬名"), ("jockey_name", "騎手"),
    ("carried_weight", "負担重量"), ("odds_win", "単勝"), ("p_mkt", "市場確率"),
    ("p_fund", "能力確率"), ("p_model", "AI勝率"), ("p_top3", "AI複勝率"),
    ("edge", "歪み"), ("ev", "期待値"), ("fair_odds", "適正オッズ"),
    ("h_rating", "馬レート"), ("j_win_ae", "騎手力"), ("h_moist_apt", "水分適性"),
    ("bet", "推奨"), ("kelly_stake", "ケリー額"),
]


def format_table(P: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c, _ in DISPLAY if c in P]
    t = P[cols].rename(columns=dict(DISPLAY)).copy()
    for c in ["市場確率", "能力確率", "AI勝率", "AI複勝率", "歪み"]:
        if c in t:
            t[c] = (t[c] * 100).round(1)
    for c in ["期待値", "騎手力", "水分適性"]:
        if c in t:
            t[c] = t[c].round(2)
    for c in ["適正オッズ"]:
        t[c] = t[c].round(1)
    if "馬レート" in t:
        t["馬レート"] = t["馬レート"].round(0)
    t["推奨"] = np.where(t["推奨"], "◎", "")
    return t
