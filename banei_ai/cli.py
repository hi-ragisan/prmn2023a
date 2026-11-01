"""コマンドライン.

  python -m banei_ai.cli simulate                     検証用データを生成
  python -m banei_ai.cli scrape --start 2023-04-01 --end 2026-03-31 --source netkeiba
  python -m banei_ai.cli update                       最終日から昨日までを追加取得 (日々の運用)
  python -m banei_ai.cli import-csv --in mydata.csv   手持ち CSV (日本語見出し可) を取り込み
  python -m banei_ai.cli backtest                     ウォークフォワード検証 + 歪みレポート
  python -m banei_ai.cli report                       騎手・馬の良し悪し + 歪みの棚卸し
  python -m banei_ai.cli train                        本番モデルを学習して保存
  python -m banei_ai.cli predict --date 2026-11-02    当日の出馬表を取得して予測
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import backtest as B
from . import config, report, schema
from .features import build_features
from .model import BaneiModel, evaluate_probs, walk_forward

DEFAULT_DATA = config.INTERIM_DIR / "races.csv"
DEFAULT_MODEL = config.MODEL_DIR / "model.pkl"


def _load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} がありません。先に scrape / import-csv / simulate を実行してください")
    return schema.normalize(pd.read_csv(path, dtype={"race_id": str, "horse_id": str,
                                                     "jockey_id": str, "trainer_id": str}))


def _features(df: pd.DataFrame, cache: Path | None = None) -> pd.DataFrame:
    print(f"特徴量を作成中... ({len(df)} 行)")
    return build_features(df)


def cmd_simulate(a):
    from .simulate import SimConfig, simulate

    df = simulate(SimConfig(start=a.start, end=a.end, seed=a.seed))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"[シミュレーション] {len(df)} 行 / {df['race_id'].nunique()} レース → {a.out}")
    print("※ 人工データです。実際のばんえい競馬の成績ではありません。")


def cmd_scrape(a):
    if a.source == "netkeiba":
        from .scrape import netkeiba as src
    else:
        from .scrape import keibago as src
    df = src.scrape_range(a.start, a.end, wait=a.wait)
    if df.empty:
        print("データが取得できませんでした")
        return
    out = Path(a.out)
    if out.exists() and not a.overwrite:
        old = pd.read_csv(out, dtype={"race_id": str})
        df = pd.concat([old, df]).drop_duplicates(["race_id", "post"], keep="last")
    df = schema.normalize(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} 行 / {df['race_id'].nunique()} レース → {out}")
    if a.check:
        _check(df)


def cmd_update(a):
    """データの最終日の翌日から昨日までを追加取得."""
    out = Path(a.out)
    if out.exists():
        old = pd.read_csv(out, usecols=["date"])
        start = pd.to_datetime(old["date"]).max() + pd.Timedelta(days=1)
    else:
        start = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    if start > end:
        print("最新です")
        return
    print(f"{start:%Y-%m-%d} 〜 {end:%Y-%m-%d} を取得します")
    a.start, a.end, a.overwrite, a.check = f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", False, False
    cmd_scrape(a)


def _check(df: pd.DataFrame):
    print("\n[データ品質チェック] 列ごとの欠損率")
    print(df.isna().mean().round(3).to_string())
    print("\n先頭 12 行:")
    print(df.head(12).to_string())


def cmd_import(a):
    from .scrape.common import clean_entries, map_headers

    raw = pd.read_csv(a.input)
    m = map_headers(raw.columns)
    # 既に正規列名の列はそのまま
    keep = {c: c for c in raw.columns if c in schema.ALL}
    m = {**m, **keep}
    t = raw[list(m)].rename(columns=m)
    t = clean_entries(t) if "finish" in t and t["finish"].dtype == object else t
    for c, src in [("horse_id", "horse_name"), ("jockey_id", "jockey_name"), ("trainer_id", "trainer_name")]:
        if c not in t and src in t:
            t[c] = t[src]
    if "race_id" not in t:
        t["race_id"] = pd.to_datetime(t["date"]).dt.strftime("%Y%m%d") + t["race_no"].astype(int).map("{:02d}".format)
    if "status" not in t:
        t["status"] = "ok"
    df = schema.normalize(t)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} 行を取り込み → {a.out}")
    _check(df)


def cmd_backtest(a):
    config.ensure_dirs()
    df = _load(a.data)
    X = _features(df)
    print("ウォークフォワード学習中...")
    P = walk_forward(X, block_months=a.block_months)
    P.to_pickle(config.REPORT_DIR / "walkforward_preds.pkl")
    ev_tab = evaluate_probs(P)
    grid = B.threshold_grid(P)
    rule = B.BetRule(min_ev=a.min_ev)
    bets = rule.select(P)
    lo, hi = B.bootstrap_roi(bets)
    per = B.by_period(P, rule)
    cal = B.calibration(P)
    from .market import fit_power_calibration

    alpha = fit_power_calibration(P)
    lines = [
        "# バックテスト結果 (ウォークフォワード・完全サンプル外)\n",
        f"- データ: {a.data}",
        f"- 評価期間: {P['date'].min():%Y-%m-%d} 〜 {P['date'].max():%Y-%m-%d} / {P['race_id'].nunique()} レース",
        f"- 市場の favorite-longshot 指数 α = {alpha:.3f} (α>1: 本命が過小評価・大穴が過大評価)\n",
        "## 確率の精度 (1 レースあたり対数損失、vs_market が正なら市場より良い)\n",
        ev_tab.to_markdown(index=False, floatfmt=".4f"),
        "\n## 期待値しきい値ごとの成績 (単勝・100円均等 / 1/4ケリー)\n",
        grid.to_markdown(index=False, floatfmt=".3f"),
        f"\n## 採用ルール min_ev={a.min_ev}: 回収率 90% 区間 [{lo:.3f}, {hi:.3f}] (レース単位ブートストラップ)\n",
        per.to_markdown(index=False, floatfmt=".3f") if not per.empty else "(該当なし)",
        "\n## 較正 (AI 勝率 vs 実際の勝率)\n",
        cal.to_markdown(floatfmt=".3f"),
        "\n## Stage1 重要特徴量 (最終ブロックのモデル)\n",
    ]
    from .model import Stage1

    s1 = Stage1().fit(X[X["date"] >= X["date"].min() + pd.Timedelta(days=90)])
    lines.append(s1.importance().head(20).to_frame("gain").to_markdown(floatfmt=".0f"))
    lines.append("\n> 注意: 確定オッズでの検証です。購入時点のオッズとのズレ、自分の購入によるオッズ低下で実際の回収率は下がります。")
    out = config.REPORT_DIR / "backtest.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:9]))
    print(grid.to_string(index=False))
    print(f"\n詳細 → {out}")


def cmd_report(a):
    config.ensure_dirs()
    df = _load(a.data)
    X = _features(df)
    since = X["date"].max() - pd.Timedelta(days=a.days)
    jt = report.jockey_table(X, since=since, min_rides=a.min_rides)
    sk = report.jockey_skill_clogit(X[X["date"] >= since], min_rides=a.min_rides)
    ht = report.horse_table(X)
    dist = report.distortion_report(X)
    show = ["name", "rides", "win_rate", "top3_rate", "win_ae", "mkt_ae", "mkt_ae_lo90",
            "mkt_ae_hi90", "popularity", "roi", "verdict"]
    lines = [
        f"# 騎手・馬・オッズの歪みレポート (直近 {a.days} 日)\n",
        "## 騎手: 市場との比較\n",
        "- win_ae: 頭数から見た期待勝利に対する実勝利 (腕＋騎乗馬の質)",
        "- mkt_ae: **オッズが示す期待勝利に対する実勝利** (>1 なら市場が過小評価)",
        "- popularity: 平均市場シェア×頭数 (>1 なら人気を集めやすい騎手)\n",
        jt[show].to_markdown(floatfmt=".3f"),
        "\n## 騎手: 馬の強さを統制した固有の腕 (条件付きロジット係数)\n",
        sk.to_markdown(floatfmt=".3f"),
        "\n## 馬: レーティング上位\n",
        ht.to_markdown(index=False, floatfmt=".3f"),
        "\n## オッズの歪みの棚卸し (A/E = 実勝利 / 市場期待勝利)\n",
        report.to_markdown({k: v[["bets", "wins", "exp_wins", "ae", "ae_lo90", "ae_hi90", "roi", "signal"]]
                            for k, v in dist.items()}),
    ]
    out = config.REPORT_DIR / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(jt[show].head(15).to_string())
    print(f"\n詳細 → {out}")


def cmd_train(a):
    df = _load(a.data)
    X = _features(df)
    print("学習中...")
    m = BaneiModel().fit(X)
    m.save(Path(a.model))
    print(f"モデル保存 → {a.model} (学習データ最終日 {m.trained_until:%Y-%m-%d})")
    print(m.stage2.combo_.coef_table().to_string(index=False))


def cmd_predict(a):
    from .predict import format_table, predict_card

    hist = _load(a.data)
    model = BaneiModel.load(Path(a.model))
    date = pd.Timestamp(a.date)
    if a.card:
        card = pd.read_csv(a.card, dtype={"race_id": str, "horse_id": str, "jockey_id": str})
        card["date"] = pd.to_datetime(card["date"])
    else:
        from .scrape import keibago

        races = []
        f = keibago.Fetcher(wait=a.wait)
        for rn in range(1, 13):
            if a.race and rn != a.race:
                continue
            t = keibago.scrape_card(date, rn, f)
            if t is None:
                if not a.race:
                    break
                continue
            races.append(t)
        if not races:
            raise SystemExit("出馬表を取得できませんでした (--card で CSV を渡すこともできます)")
        card = pd.concat(races, ignore_index=True)
    P = predict_card(hist, card, model, B.BetRule(min_ev=a.min_ev), bankroll=a.bankroll)
    t = format_table(P)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for r, g in t.groupby("R"):
        print(f"\n===== {date:%Y-%m-%d} {int(r)}R =====")
        print(g.drop(columns=["R"]).to_string(index=False))
    out = config.REPORT_DIR / f"pred_{date:%Y%m%d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n→ {out}")
    print("※ 歪み = AI勝率/市場確率 - 1 (%)。期待値 = AI勝率 × 単勝オッズ。締切直前のオッズで再計算推奨。")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="banei_ai", description="ばんえい競馬 予測AI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="検証用データ生成")
    s.add_argument("--start", default="2023-04-01")
    s.add_argument("--end", default="2025-09-30")
    s.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    s.add_argument("--out", default=str(config.INTERIM_DIR / "simulated.csv"))
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("scrape", help="過去成績を取得")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--source", choices=["netkeiba", "keibago"], default="netkeiba")
    s.add_argument("--out", default=str(DEFAULT_DATA))
    s.add_argument("--wait", type=float, default=2.0, help="リクエスト間隔 (秒)")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--check", action="store_true", help="取得後にデータ品質を表示")
    s.set_defaults(fn=cmd_scrape)

    s = sub.add_parser("update", help="最終日から昨日までを追加取得")
    s.add_argument("--source", choices=["netkeiba", "keibago"], default="netkeiba")
    s.add_argument("--out", default=str(DEFAULT_DATA))
    s.add_argument("--wait", type=float, default=2.0)
    s.set_defaults(fn=cmd_update)

    s = sub.add_parser("import-csv", help="手持ち CSV を取り込み")
    s.add_argument("--in", dest="input", required=True)
    s.add_argument("--out", default=str(DEFAULT_DATA))
    s.set_defaults(fn=cmd_import)

    for name, fn, help_ in [("backtest", cmd_backtest, "ウォークフォワード検証"),
                            ("report", cmd_report, "騎手・馬・歪みレポート"),
                            ("train", cmd_train, "本番モデル学習")]:
        s = sub.add_parser(name, help=help_)
        s.add_argument("--data", default=str(DEFAULT_DATA))
        s.add_argument("--model", default=str(DEFAULT_MODEL))
        s.add_argument("--block-months", type=int, default=2)
        s.add_argument("--min-ev", type=float, default=1.10)
        s.add_argument("--days", type=int, default=365)
        s.add_argument("--min-rides", type=int, default=50)
        s.set_defaults(fn=fn)

    s = sub.add_parser("predict", help="当日予測")
    s.add_argument("--date", required=True)
    s.add_argument("--race", type=int, default=None)
    s.add_argument("--card", default=None, help="出馬表 CSV (正規スキーマ, status=entry)")
    s.add_argument("--data", default=str(DEFAULT_DATA))
    s.add_argument("--model", default=str(DEFAULT_MODEL))
    s.add_argument("--min-ev", type=float, default=1.10)
    s.add_argument("--bankroll", type=float, default=100_000)
    s.add_argument("--wait", type=float, default=1.0)
    s.set_defaults(fn=cmd_predict)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
