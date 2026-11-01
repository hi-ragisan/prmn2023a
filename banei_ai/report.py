"""騎手・馬の「良し悪し」と「オッズの歪み」のレポート.

騎手:
  win_ae   = 勝利数 / 頭数から見た期待勝利 (Σ 1/頭数)   … 純粋な腕 + 騎乗馬の質
  mkt_ae   = 勝利数 / 市場の期待勝利 (Σ p_mkt)         … 市場より勝っているか (歪み)
  popularity = 平均の市場シェア × 頭数                  … 1 より大きいと人気になりやすい騎手
  model_skill = 条件付きロジットで馬の強さを統制した騎手固有効果
馬:
  rating (Elo), 近走の着順パーセンタイル, 市場超過 (h_mkt_ae)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import market
from .clogit import ConditionalLogit


def jockey_table(X: pd.DataFrame, since: pd.Timestamp | None = None, min_rides: int = 50) -> pd.DataFrame:
    d = X if since is None else X[X["date"] >= since]
    d = d.assign(exp_field=1.0 / d["n_runners"], share=d["p_mkt"] * d["n_runners"],
                 ret=d["win"] * d["odds_win"])
    g = d.groupby("jockey_id")
    t = pd.DataFrame({
        "name": g["jockey_name"].agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else ""),
        "rides": g.size(),
        "wins": g["win"].sum(),
        "top3": g["top3"].sum(),
        "exp_field": g["exp_field"].sum(),
        "exp_mkt": g["p_mkt"].sum(),
        "popularity": g["share"].mean(),
        "roi": g["ret"].mean(),
    })
    t = t[t["rides"] >= min_rides]
    t["win_rate"] = t["wins"] / t["rides"]
    t["top3_rate"] = t["top3"] / t["rides"]
    t["win_ae"] = t["wins"] / t["exp_field"]
    t["mkt_ae"] = t["wins"] / t["exp_mkt"]
    ci = [market._ae_interval(w, e) for w, e in zip(t["wins"], t["exp_mkt"])]
    t["mkt_ae_lo90"] = [c[0] for c in ci]
    t["mkt_ae_hi90"] = [c[1] for c in ci]
    t["verdict"] = np.select(
        [t["mkt_ae_lo90"] > 1, t["mkt_ae_hi90"] < 1],
        ["市場が過小評価 (狙い)", "市場が過大評価 (割引)"], default="")
    return t.sort_values("mkt_ae", ascending=False)


def jockey_skill_clogit(X: pd.DataFrame, min_rides: int = 100, l2: float = 3e-3) -> pd.DataFrame:
    """馬の強さ (s1 または能力系特徴量) を統制した上での騎手固有効果.

    騎手ダミー + 馬の能力指標で条件付きロジットを当て、騎手ダミーの係数を「腕」とみなす。
    L2 正則化で騎乗数の少ない騎手は 0 (平均) に縮む。
    """
    d = X[X["h_n_runs"] > 0].copy()
    counts = d["jockey_id"].value_counts()
    keep = counts[counts >= min_rides].index
    d["_j"] = d["jockey_id"].where(d["jockey_id"].isin(keep), "_other")
    ctrl = [c for c in ["r_h_rating_diff", "r_h_ewm_pct_diff", "r_h_avg3_gap_diff", "cw_rel",
                        "h_moist_apt", "r_t_win_ae_diff"] if c in d]
    dummies = pd.get_dummies(d["_j"], dtype=float).drop(columns=["_other"], errors="ignore")
    Z = pd.concat([d[ctrl].astype(float), dummies], axis=1)
    m = ConditionalLogit(l2=l2).fit(Z, d["win"], d["race_id"], list(Z.columns))
    ct = m.coef_table().set_index("feature")
    skill = ct.loc[dummies.columns, "coef_raw"]
    names = d.groupby("jockey_id")["jockey_name"].last()
    out = pd.DataFrame({"skill": skill, "rides": counts.reindex(skill.index)})
    out["name"] = names.reindex(out.index)
    out["skill"] -= out["skill"].mean()
    # skill 0.1 ≒ 勝率を約 10% 押し上げ (相対)
    return out.sort_values("skill", ascending=False)


def horse_table(X: pd.DataFrame, asof: pd.Timestamp | None = None, top: int = 50) -> pd.DataFrame:
    """各馬の最新の状態 (最後の出走時点の特徴量 + そのレース結果)."""
    d = X if asof is None else X[X["date"] <= asof]
    last = d.sort_values("date").groupby("horse_id").tail(1)
    cols = ["horse_id", "horse_name", "date", "race_class", "h_rating", "h_n_runs",
            "h_ewm_pct", "h_win_ae", "h_mkt_ae", "h_moist_apt", "finish", "odds_win"]
    cols = [c for c in cols if c in last]
    return last[cols].sort_values("h_rating", ascending=False).head(top)


def distortion_report(X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """市場の歪みの棚卸し (どの切り口で市場が間違えやすいか)."""
    d = X[X["p_mkt"].notna()].copy()
    out: dict[str, pd.DataFrame] = {}
    out["オッズ帯"] = market.favorite_longshot_table(d)
    out["人気順"] = market.popularity_table(d)
    d["前走着順"] = pd.cut(d["h_last1_pct"], [-0.01, 0.0, 0.34, 0.67, 1.0],
                       labels=["1着", "上位", "中位", "下位"])
    out["前走着順"] = market.distortion_table(d, "前走着順", min_bets=1).sort_index()
    d["水分適性"] = pd.qcut(d["h_moist_apt"], 5, duplicates="drop")
    out["馬場水分適性"] = market.distortion_table(d, "水分適性", min_bets=1).sort_index()
    d["斤量増減"] = pd.cut(d["h_cw_change"], [-1000, -11, -1, 0, 10, 1000],
                       labels=["-11kg以下", "-10〜-1", "増減なし", "+1〜+10", "+11kg以上"])
    out["負担重量の増減"] = market.distortion_table(d, "斤量増減", min_bets=1).sort_index()
    d["馬体重増減"] = pd.cut(d["horse_weight_diff"], [-1000, -20, -6, 5, 19, 1000],
                        labels=["-20以下", "-19〜-6", "±5", "+6〜+19", "+20以上"])
    out["馬体重の増減"] = market.distortion_table(d, "馬体重増減", min_bets=1).sort_index()
    d["騎手人気度"] = pd.qcut(d["j_popularity"], 4, labels=["不人気騎手", "やや不人気", "やや人気", "人気騎手"],
                         duplicates="drop")
    out["騎手の人気度"] = market.distortion_table(d, "騎手人気度", min_bets=1).sort_index()
    out["乗り替わり"] = market.distortion_table(d, "h_jockey_change", min_bets=1)
    return out


def to_markdown(tables: dict[str, pd.DataFrame], float_fmt: str = ".3f") -> str:
    parts = []
    for k, t in tables.items():
        parts.append(f"### {k}\n")
        parts.append(t.to_markdown(floatfmt=float_fmt) if hasattr(t, "to_markdown") else t.to_string())
        parts.append("")
    return "\n".join(parts)
