"""頑健性の検証: 市場の歪みを弱めていくと、パイプラインが見つける「エッジ」はどう縮むか.

  python scripts/robustness.py                                  # 歪み {0, .25, .5, 1} × seed {42, 3, 7} + 効率的市場
  python scripts/robustness.py --scales 0,1 --seeds 42 --out docs/robustness_table.md

!!! すべて人工データ (banei_ai/simulate.py) 上の数値。実際のばんえい競馬の成績ではない !!!

各 (歪みの強さ s, seed) について:
  1. simulate(SimConfig(distortion=s, seed=...))  … s=1 が従来の既定 (README の数値)
     同じ seed ならレース結果は s によらず完全に同じで、オッズだけが変わる (共通乱数)。
     ※ s=0 でも市場の情報誤差 (market_info_sd=0.2) は残る。この誤差は公開情報で補正されて
       いないので、モデルが拾える小さな情報が残る (下の "efficient" 行と比較)。
  2. 行 "efficient" = SimConfig.preset("efficient"): 歪みなし + 市場が真の強さを完全に知り、
     停止の偶然まで織り込んだ真の勝率で値付け (帰無仮説。モデルが市場に足せる情報は原理的にゼロ)
     行 "realistic" = SimConfig.preset("realistic"): 実データの統計に合わせた設定 (頭数などが違うので
     レース結果は他の行と共通ではない)
  3. build_features → walk_forward (完全サンプル外) → evaluate_probs / BetRule(min_ev=1.1) /
     bootstrap_roi / split_half_check
  4. シミュレーションだけで分かる「真の期待値」 Σ 真の勝率·オッズ / 点数 も出す
     (真の勝率は停止の偶然まで含めたモンテカルロ: simulate.true_win_probs)。
     実現回収率は運に大きく左右される (数百点では ±30% 以上ぶれる) ので、エッジが本物かは
     こちらで判断する。「モデルの見積もり」は p_model·オッズ の平均 (買った理由)。

特徴量・予測は --cache (既定 data/interim/robustness/) に pickle で保存し、再実行時は再利用する。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from banei_ai import backtest as B  # noqa: E402
from banei_ai import config  # noqa: E402
from banei_ai.features import build_features  # noqa: E402
from banei_ai.model import evaluate_probs, walk_forward  # noqa: E402
from banei_ai.simulate import PRESETS, SimConfig, simulate, true_win_probs  # noqa: E402


def make_config(scale: str, seed: int, start: str, end: str) -> SimConfig:
    """scale は歪みの強さ (数値) かプリセット名."""
    if scale in PRESETS:
        return SimConfig.preset(scale, seed=seed, start=start, end=end)
    return SimConfig(distortion=float(scale), seed=seed, start=start, end=end)


def predictions(scale: str, seed: int, start: str, end: str, cache: Path) -> pd.DataFrame:
    f = cache / f"P_{scale}_s{seed}_{start}_{end}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    t = time.time()
    X = build_features(simulate(make_config(scale, seed, start, end)))
    P = walk_forward(X, verbose=False)
    P.to_pickle(f)
    print(f"  [{scale} seed {seed}] {time.time() - t:.0f}s", file=sys.stderr, flush=True)
    return P


def bet_true_p(P: pd.DataFrame, bets: pd.DataFrame, cfg: SimConfig) -> pd.Series:
    """買った馬の真の勝率 (停止込みのモンテカルロ)。計算量を抑えるため買ったレースだけ計算する."""
    if bets.empty:
        return pd.Series(dtype=float)
    races = P[P["race_id"].isin(bets["race_id"])]
    return true_win_probs(races, cfg).loc[bets.index]


def summarize(P: pd.DataFrame, cfg: SimConfig, min_ev: float = 1.1) -> dict:
    ll = evaluate_probs(P).set_index("model")["logloss"]
    bets = B.BetRule(min_ev=min_ev).select(P)
    r = B.flat_bet(bets)
    lo, hi = B.bootstrap_roi(bets)
    sh = B.split_half_check(P)
    pt = bet_true_p(P, bets, cfg)
    return {
        "races": P["race_id"].nunique(),
        "d_ll_mkt": ll["市場 (オッズそのまま)"] - ll["合成モデル"],     # 正 = 市場より良い
        "d_ll_cal": ll["市場 (較正後)"] - ll["合成モデル"],             # 較正後の市場との差
        "d_ll_true": ll["市場 (オッズそのまま)"] - ll["(参考) 真の確率"],  # 改善できる余地の上限
        "bets": r["bets"],
        "hit_rate": r["hit_rate"],
        "roi": r["roi"],
        "roi_lo90": lo,
        "roi_hi90": hi,
        "true_ev": float((pt * bets["odds_win"]).mean()) if len(bets) else np.nan,
        "claimed_ev": float(bets["ev"].mean()) if len(bets) else np.nan,
        "sh_min_ev": sh.get("chosen_min_ev"),
        "sh_bets": sh.get("second_half_bets", 0),
        "sh_roi": sh.get("second_half_roi", np.nan),
        "sh_lo90": sh.get("second_half_roi_lo90", np.nan),
        "sh_hi90": sh.get("second_half_roi_hi90", np.nan),
        "_bets": bets.assign(race_id=bets["race_id"].astype(str), true_p=pt),
    }


def pooled(rows: list[dict]) -> dict:
    """シードをまたいで賭けをまとめた回収率と区間 (レース ID はシードごとに別物として扱う)."""
    b = pd.concat([r["_bets"].assign(race_id=f"{r['seed']}_" + r["_bets"]["race_id"]) for r in rows])
    if b.empty:
        return {"bets": 0, "roi": np.nan, "roi_lo90": np.nan, "roi_hi90": np.nan, "true_ev": np.nan,
                "claimed_ev": np.nan}
    lo, hi = B.bootstrap_roi(b)
    return {"bets": len(b), "roi": float((b["win"] * b["odds_win"]).mean()), "roi_lo90": lo,
            "roi_hi90": hi, "true_ev": float((b["true_p"] * b["odds_win"]).mean()),
            "claimed_ev": float(b["ev"].mean())}


def fmt(x, nd=3, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "−"
    if pct:
        return f"{x * 100:.0f}%"
    return f"{x:+.{nd}f}" if nd == 4 else f"{x:.{nd}f}"


def ci(lo, hi):
    return "−" if lo is None or np.isnan(lo) else f"{lo * 100:.0f}〜{hi * 100:.0f}%"


def to_markdown(rows: list[dict], scales: list[str]) -> str:
    out = ["### シード別 (期待値 ≥ 1.10 の単勝, 均等買い)", "",
           "| 歪み s | seed | レース | Δ対数損失 vs 市場 | vs 較正市場 | 真の確率 vs 市場 | 点数 | 的中率 | 回収率 | 90%区間 "
           "| 真の期待値 | モデルの見積もり | 前後半: しきい値 | 後半 点数 | 後半 回収率 (90%区間) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for r in rows:
        out.append(
            f"| {r['scale']} | {r['seed']} | {r['races']} | {fmt(r['d_ll_mkt'], 4)} | {fmt(r['d_ll_cal'], 4)} "
            f"| {fmt(r['d_ll_true'], 4)} | {r['bets']} | {fmt(r['hit_rate'], pct=True)} "
            f"| {fmt(r['roi'], pct=True)} | {ci(r['roi_lo90'], r['roi_hi90'])} | {fmt(r['true_ev'], 2)} "
            f"| {fmt(r['claimed_ev'], 2)} | {r['sh_min_ev'] if r['sh_min_ev'] is not None else '−'} | {r['sh_bets']} "
            f"| {fmt(r['sh_roi'], pct=True)} ({ci(r['sh_lo90'], r['sh_hi90'])}) |")
    out += ["", "### 歪みの強さごとの集計 (Δ対数損失はシード平均, 回収率は全シードの賭けをまとめたもの)", "",
            "| 歪み s | Δ対数損失 vs 市場 (平均, 範囲) | vs 較正市場 | 真の確率 vs 市場 | 点数 (合計) | 回収率 | 90%区間 "
            "| 真の期待値 | モデルの見積もり | 区間下限 > 100% のシード | 後半 回収率 (平均) |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for s in scales:
        g = [r for r in rows if r["scale"] == s]
        if not g:
            continue
        d = np.array([r["d_ll_mkt"] for r in g])
        dc = np.mean([r["d_ll_cal"] for r in g])
        dt = np.mean([r["d_ll_true"] for r in g])
        p = pooled(g)
        sig = sum(1 for r in g if not np.isnan(r["roi_lo90"]) and r["roi_lo90"] > 1)
        shr = np.nanmean([r["sh_roi"] for r in g]) if any(not np.isnan(r["sh_roi"]) for r in g) else np.nan
        out.append(
            f"| {s} | {d.mean():+.4f} ({d.min():+.4f}〜{d.max():+.4f}) | {dc:+.4f} | {dt:+.4f} | {p['bets']} "
            f"| {fmt(p['roi'], pct=True)} | {ci(p['roi_lo90'], p['roi_hi90'])} | {fmt(p['true_ev'], 2)} "
            f"| {fmt(p['claimed_ev'], 2)} | {sig}/{len(g)} | {fmt(shr, pct=True)} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scales", default="0,0.25,0.5,1,efficient",
                    help="歪みの強さ (カンマ区切り)。プリセット名 (efficient / realistic) も可")
    ap.add_argument("--seeds", default="42,3,7")
    ap.add_argument("--start", default="2023-04-01")
    ap.add_argument("--end", default="2025-09-30")
    ap.add_argument("--cache", default=str(config.INTERIM_DIR / "robustness"))
    ap.add_argument("--out", default=None, help="Markdown の表を書き出すファイル")
    a = ap.parse_args(argv)

    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    scales = [s.strip() for s in a.scales.split(",")]
    scales = [s if s in PRESETS else f"{float(s):g}" for s in scales]
    rows = []
    for s in scales:
        for seed in [int(x) for x in a.seeds.split(",")]:
            r = summarize(predictions(s, seed, a.start, a.end, cache), make_config(s, seed, a.start, a.end))
            r.update(scale=s, seed=seed)
            rows.append(r)
    md = to_markdown(rows, scales)
    print("※ シミュレーション (人工データ) 上の数値です。実際のばんえい競馬の成績ではありません。\n")
    print(md)
    if a.out:
        Path(a.out).write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
