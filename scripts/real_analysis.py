"""実データ (netkeiba から取得した成績) の総合分析レポート.

  python scripts/real_analysis.py --data data/interim/races.csv [--preds data/reports/walkforward_preds.pkl]
  → data/reports/real_analysis.md

出力は取得データからの集計なので、公開リポジトリにはコミットしないこと (利用規約)。
内容:
  1. データの概要と品質 (欠損率・1番人気の成績・払戻率との整合)
  2. 予測精度: 市場 / 較正後の市場 / 合成モデル、月ごとの一貫性 (較正後の市場との差)
  3. 改善の内訳: 前半で学習 → 後半で評価 (市場の較正・前走の市場シェア・能力モデル)
  4. 期待値の分布と購入候補の数
  5. オッズの歪みの棚卸し (A/E)
  6. 騎手: 市場比 A/E と、馬の強さを統制した腕
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from banei_ai import config, market, report, schema  # noqa: E402
from banei_ai.clogit import ConditionalLogit, race_logloss  # noqa: E402
from banei_ai.features import build_features  # noqa: E402
from banei_ai.model import evaluate_probs, walk_forward  # noqa: E402


def md(t: pd.DataFrame, fmt: str = ".3f", index: bool = True) -> str:
    return t.to_markdown(floatfmt=fmt, index=index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.INTERIM_DIR / "races.csv"))
    ap.add_argument("--preds", default=None, help="保存済みのウォークフォワード予測 (無ければ計算)")
    ap.add_argument("--out", default=str(config.REPORT_DIR / "real_analysis.md"))
    a = ap.parse_args()

    raw = schema.normalize(schema.read_csv(a.data))
    L: list[str] = ["# ばんえい 実データ分析\n", f"- データ: {a.data}"]

    # 1. 概要
    ok = raw[raw["status"] == "ok"]
    fav = ok[ok["popularity"] == 1]
    inv = (1 / ok["odds_win"]).groupby(ok["race_id"]).sum()
    L += [
        f"- 期間: {raw['date'].min():%Y-%m-%d} 〜 {raw['date'].max():%Y-%m-%d} / "
        f"{raw['date'].nunique()} 開催日 / {raw['race_id'].nunique()} レース / {len(raw)} 頭",
        f"- 平均頭数 {raw.groupby('race_id').size().mean():.2f} / 取消 {int((raw['status'] == 'scratch').sum())} / "
        f"中止 {int((raw['status'] == 'dnf').sum())}",
        f"- 1番人気: 勝率 {(fav['finish'] == 1).mean():.3f} / 単勝回収率 {((fav['finish'] == 1) * fav['odds_win']).mean():.3f}",
        f"- 全馬の単勝回収率 {((ok['finish'] == 1) * ok['odds_win']).mean():.3f} / オッズ逆数の合計 (平均) {inv.mean():.3f}",
        "",
        "欠損率: " + ", ".join(f"{c} {raw[c].isna().mean():.1%}" for c in
                             ["odds_win", "moisture", "carried_weight", "horse_weight", "time_sec", "prize1"]
                             if c in raw),
        "",
    ]

    X = build_features(raw)
    if a.preds and Path(a.preds).exists():
        P = pd.read_pickle(a.preds)
    else:
        P = walk_forward(X, verbose=False)
    d = P[P["p_mkt"].notna() & P["p_mkt_cal"].notna()]

    # 2. 予測精度と一貫性
    L += ["## 予測精度 (完全サンプル外)\n",
          f"評価期間 {P['date'].min():%Y-%m-%d} 〜 {P['date'].max():%Y-%m-%d} / {P['race_id'].nunique()} レース\n",
          md(evaluate_probs(P), ".4f", index=False), ""]
    rows = []
    for k, g in d.groupby(d["date"].dt.to_period("M")):
        rows.append({"月": str(k), "レース": g["race_id"].nunique(),
                     "較正後の市場との差": race_logloss(g["p_mkt_cal"], g["win"], g["race_id"])
                     - race_logloss(g["p_model"], g["win"], g["race_id"])})
    mt = pd.DataFrame(rows)
    L += ["### 月ごとの一貫性 (正 = 合成モデルが較正後の市場より正確)\n", md(mt, "+.4f", index=False),
          f"\n→ {int((mt['較正後の市場との差'] > 0).sum())}/{len(mt)} か月でプラス、平均 {mt['較正後の市場との差'].mean():+.4f}\n"]

    # 3. 改善の内訳
    cut = d["date"].quantile(0.5)
    tr, te = d[d["date"] < cut], d[d["date"] >= cut]
    base = race_logloss(te["p_mkt"], te["win"], te["race_id"])
    rows = []
    for name, cols in [("市場の較正のみ", ["log_p_mkt"]),
                       ("+ 前走の市場シェア", ["log_p_mkt", "h_last_pmkt_share"]),
                       ("+ 騎手の人気度", ["log_p_mkt", "j_popularity"]),
                       ("+ 能力モデル (馬・騎手)", ["log_p_mkt", "s1"]),
                       ("全部 (既定の構成)", ["log_p_mkt", "s1", "j_popularity", "h_last_pmkt_share"])]:
        cols = [c for c in cols if c in tr]
        m = ConditionalLogit(l2=1e-3).fit(tr[cols], tr["win"], tr["race_id"], cols)
        p = m.predict_proba(te[cols], te["race_id"].to_numpy())
        coefs = ", ".join(f"{c}={v:+.3f}" for c, v in zip(cols, m.coef_ / m.scale_))
        rows.append({"構成": name, "市場との差": base - race_logloss(p, te["win"], te["race_id"]), "係数": coefs})
    L += [f"## 改善の内訳 (前半で学習 → 後半 {te['race_id'].nunique()} レースで評価)\n",
          md(pd.DataFrame(rows), "+.4f", index=False), ""]

    # 4. 期待値
    ev = (P["p_model"] * P["odds_win"]).dropna()
    L += ["## 期待値 (AI勝率 × 単勝オッズ) の分布\n",
          f"- 中央値 {ev.median():.3f} / 90% 点 {ev.quantile(.9):.3f} / 99% 点 {ev.quantile(.99):.3f} / 最大 {ev.max():.3f}",
          "- 期待値 ≥ 1.00: " + str(int((ev >= 1.0).sum())) + " 頭 / ≥ 1.05: " + str(int((ev >= 1.05).sum()))
          + " 頭 / ≥ 1.10: " + str(int((ev >= 1.10).sum())) + " 頭", ""]
    hit = P[(P["p_model"] * P["odds_win"]) >= 1.0]
    if len(hit):
        L.append(f"- 期待値 ≥ 1.00 の馬の実績: {len(hit)} 点, 的中 {int(hit['win'].sum())}, "
                 f"回収率 {(hit['win'] * hit['odds_win']).mean():.3f} (点数が少なく運の影響大)\n")

    # 5. 歪み
    dist = report.distortion_report(X)
    keep = ["bets", "wins", "exp_wins", "ae", "ae_lo90", "ae_hi90", "roi", "signal"]
    L += ["## オッズの歪みの棚卸し (A/E = 実勝利 ÷ 市場の期待勝利、全期間)\n",
          report.to_markdown({k: v[[c for c in keep if c in v]] for k, v in dist.items()})]
    Xm = market.add_market_probs(X) if "p_mkt" not in X else X
    Xm = Xm.assign(前走市場シェア=pd.qcut(Xm["h_last_pmkt_share"], 5, duplicates="drop"))
    L += ["### 前走の市場シェア (前走でどれだけ人気を集めたか) 別\n",
          md(market.distortion_table(Xm, "前走市場シェア", min_bets=1).sort_index()[keep]), ""]

    # 6. 騎手
    since = X["date"].max() - pd.Timedelta(days=365)
    jt = report.jockey_table(X, since=since, min_rides=100)
    sk = report.jockey_skill_clogit(X[X["date"] >= since], min_rides=100)
    show = ["name", "rides", "win_rate", "top3_rate", "win_ae", "mkt_ae", "mkt_ae_lo90", "mkt_ae_hi90",
            "popularity", "roi", "verdict"]
    L += ["## 騎手 (直近 1 年, 100 騎乗以上)\n",
          "- win_ae: 頭数から見た期待に対する実勝利 / mkt_ae: **オッズが示す期待に対する実勝利** (>1 なら市場が過小評価)\n",
          md(jt[show]), "",
          "### 馬の強さを統制した騎手固有の腕 (条件付きロジット係数, 0 = 平均)\n", md(sk), ""]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
