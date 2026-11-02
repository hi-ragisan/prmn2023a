"""コマンドライン.

  python -m banei_ai.cli simulate                     検証用データを生成
  python -m banei_ai.cli scrape --start 2023-04-01 --end 2026-03-31 --source netkeiba
  python -m banei_ai.cli update                       最終日から昨日までを追加取得 (日々の運用)
  python -m banei_ai.cli import-csv --in mydata.csv   手持ち CSV (日本語見出し可) を取り込み
  python -m banei_ai.cli backtest                     ウォークフォワード検証 + 歪みレポート
  python -m banei_ai.cli report                       騎手・馬の良し悪し + 歪みの棚卸し
  python -m banei_ai.cli train                        本番モデルを学習して保存
  python -m banei_ai.cli predict --date 2026-11-02    当日の出馬表を取得して予測
  python -m banei_ai.cli dashboard --data ...         結果の HTML ダッシュボード
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as B
from . import config, report, schema
from .features import build_features
from .model import BaneiModel, evaluate_probs, walk_forward

DEFAULT_DATA = config.INTERIM_DIR / "races.csv"
DEFAULT_MODEL = config.MODEL_DIR / "model.pkl"
# 1 レースの単勝売上の仮定 (円)。帯広の年間売上 ≈ 550億円 / 約1,650レース ≈ 3,300万円/レース (全券種),
# 単勝のシェア ≈ 8% (2026-03 の実例) → 約 250 万円。通常レース 100〜500 万円、重賞は 1,000 万円超。
# 推定値 (確度: 中)。レースごとの実売上が手に入ったら差し替えること。
DEFAULT_WIN_POOL_YEN = 2_500_000.0


def _load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} がありません。先に scrape / import-csv / simulate を実行してください")
    return schema.normalize(schema.read_csv(path))


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
        old = schema.read_csv(out)
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


IMPORT_EXTRA = {"日付": "date", "開催日": "date", "年月日": "date", "レース": "race_no", "R": "race_no",
                "レース番号": "race_no", "水分": "moisture", "馬場水分": "moisture", "クラス": "race_class",
                "条件": "race_class", "レース名": "race_name"}


def import_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """日本語見出しの CSV → 正規スキーマ."""
    from .scrape.common import clean_entries, map_headers

    m = {c: IMPORT_EXTRA[c] for c in raw.columns if c in IMPORT_EXTRA}
    m.update({c: v for c, v in map_headers([c for c in raw.columns if c not in m]).items()})
    m.update({c: c for c in raw.columns if c in schema.ALL})  # 既に正規列名ならそのまま
    t = raw[list(m)].rename(columns=m)
    t = t.loc[:, ~t.columns.duplicated()]
    t = clean_entries(t)
    if "moisture" in t:
        t["moisture"] = t["moisture"].map(schema.to_float)
    for c, src in [("horse_id", "horse_name"), ("jockey_id", "jockey_name"), ("trainer_id", "trainer_name")]:
        if c not in t and src in t:
            t[c] = t[src]
    if "race_id" not in t:
        t["race_id"] = (pd.to_datetime(t["date"]).dt.strftime("%Y%m%d")
                        + t["race_no"].map(schema.to_float).astype(int).map("{:02d}".format))
    return schema.normalize(t)


def cmd_import(a):
    df = import_frame(pd.read_csv(a.input))
    df.to_csv(a.out, index=False)
    print(f"{len(df)} 行を取り込み → {a.out}")
    _check(df)


def _fmt_yen(x: float) -> str:
    if not np.isfinite(x):
        return "プール額でも黒字"
    return "100円でも赤字" if x <= 0 else f"{x:,.0f}"


def _realism_lines(P: pd.DataFrame, a) -> list[str]:
    """現実性の節: 自分の購入によるオッズ低下 (プールの大きさ) × 締切前オッズのズレ."""
    rule = B.BetRule(min_ev=a.min_ev)
    pools = tuple(sorted({*B.DEFAULT_POOLS, float(a.pool_yen)}))
    sigmas = tuple(sorted({*B.DEFAULT_SIGMAS, float(a.odds_noise)}))
    stakes = B.DEFAULT_STAKES
    coef = B.stage2_market_coef(P)
    dec = B.decision_sets(P, rule, sigmas, a.noise_reps, coef=coef)
    kw = dict(payout_rate=a.payout_rate, decisions=dec)
    rt = B.robustness_table(P, rule, pools, sigmas, stakes, a.noise_reps, **kw)
    be = B.breakeven_table(P, rule, [p for p in pools if p > 0], sigmas, a.noise_reps, **kw)
    k_sig = (float(a.odds_noise),)
    kr = pd.concat([B.kelly_robustness(P, rule, pools, k_sig, n_rep=a.noise_reps, **kw),
                    B.kelly_robustness(P, rule, [p for p in pools if p > 0], k_sig, n_rep=a.noise_reps,
                                       max_stake_pct_pool=1.0, **kw)], ignore_index=True)

    def pool_label(p):
        return "影響なし" if p <= 0 else f"{p / 1e4:,.0f}万円"

    head = rt[(rt["pool_yen"] == a.pool_yen) & (rt["odds_noise"] == a.odds_noise)]
    h = pd.DataFrame({
        "1点の賭け金": head["stake"].map("{:,.0f}円".format),
        "点数": head["bets"],
        "的中率": head["hit_rate"],
        "回収率": head["roi"],
        "90%区間": [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(head["roi_lo90"], head["roi_hi90"])],
        "収支(円)": head["profit"].map("{:,.0f}".format),
        "払戻オッズ比": head["paid_odds_ratio"],
    })
    piv = rt.pivot_table(index=["odds_noise", "pool_yen"], columns="stake", values="roi", sort=True)
    piv.columns = [f"{c:,.0f}円" for c in piv.columns]
    nb = rt.groupby(["odds_noise", "pool_yen"])["bets"].first()
    bt = be.set_index(["odds_noise", "pool_yen"])
    g = piv.reset_index()
    g.insert(2, "点数", nb.reindex(pd.MultiIndex.from_frame(g[["odds_noise", "pool_yen"]])).to_numpy())
    g["損益分岐の賭け金"] = [_fmt_yen(bt.loc[(s, p), "breakeven_stake"]) if p > 0 else "—"
                       for s, p in zip(g["odds_noise"], g["pool_yen"])]
    g["(プール比)"] = [f"{bt.loc[(s, p), 'breakeven_pct_pool']:.2f}%" if p > 0 and np.isfinite(
        bt.loc[(s, p), "breakeven_stake"]) else "—" for s, p in zip(g["odds_noise"], g["pool_yen"])]
    g["利益最大の賭け金"] = [f"{bt.loc[(s, p), 'best_stake']:,.0f}" if p > 0 else "—"
                       for s, p in zip(g["odds_noise"], g["pool_yen"])]
    g["pool_yen"] = g["pool_yen"].map(pool_label)
    g = g.rename(columns={"odds_noise": "σ", "pool_yen": "単勝プール"})
    g_fmt = [".2f", "", ".1f"] + [".3f"] * len(stakes) + ["", "", ""]
    k = pd.DataFrame({
        "初期資金": kr["bankroll0"].map("{:,.0f}".format),
        "単勝プール": kr["pool_yen"].map(pool_label),
        "1点上限": kr["max_stake_cap"].map(lambda x: "なし" if not np.isfinite(x) else f"{x:,.0f}"),
        "資金倍率": kr["growth"],
        "最大DD": kr["max_dd_pct"],
        "平均賭け金": kr["mean_stake"].map("{:,.0f}".format),
        "最後の1/4期間の平均賭け金": kr["last_q_mean_stake"].map("{:,.0f}".format),
        "払戻オッズ比": kr["paid_odds_ratio"],
    })
    guide = []
    if a.pool_yen > 0:
        for o in (2.0, 3.0, 5.0, 10.0, 20.0, 40.0):
            r = {"単勝オッズ": o, "その馬の票(円)": f"{a.payout_rate * a.pool_yen / o:,.0f}"}
            for ev in (1.05, 1.10, 1.20, 1.30):
                opt, b_e = B.optimal_stake(ev / o, o, a.pool_yen, a.payout_rate)
                r[f"EV {ev:.2f}"] = f"{float(opt):,.0f} / {float(b_e):,.0f}"
            guide.append(r)
    # 参考: AI勝率を締切前オッズで計算し直さない (古い AI勝率 × 新しいオッズ) と逆選択が起きる
    naive_sig = tuple(s for s in sigmas if s > 0)
    naive_lines = []
    if naive_sig:
        nv = B.robustness_table(P, rule, (0.0,), naive_sig, (100.0,), a.noise_reps, coef=0.0,
                                payout_rate=a.payout_rate, n_boot=200)
        hon = rt[(rt["pool_yen"] == 0) & (rt["stake"] == 100.0)].set_index("odds_noise")
        naive_lines = [
            "\n### 参考: AI勝率を締切前オッズで計算し直さなかった場合 (プール影響なし, 100円)\n",
            "- 「前に計算した AI勝率 × 最新オッズ」で期待値を判断すると、たまたまオッズが高く表示された馬ばかり買う (逆選択)。"
            "Stage2 の市場係数 b≈1 なので、計算し直せば締切前オッズのズレはほぼ相殺される\n",
            pd.DataFrame({"σ": nv["odds_noise"],
                          "再計算する: 点数": hon.loc[nv["odds_noise"], "bets"].to_numpy(),
                          "再計算する: 回収率": hon.loc[nv["odds_noise"], "roi"].to_numpy(),
                          "しない: 点数": nv["bets"].to_numpy(), "しない: 回収率": nv["roi"].to_numpy()}).to_markdown(
                index=False, floatfmt=[".2f", ".1f", ".3f", ".1f", ".3f"]),
        ]
    exact = coef["max_abs_err"].max()
    guide_lines = [
        f"\n### 1 点の賭け金の目安 (プール {a.pool_yen:,.0f} 円, 各セル = 利益最大 / 損益分岐, 円)\n",
        "- 期待値 EV (自分が買う前のオッズでの p×o) と単勝オッズごとの理論値 (B.optimal_stake, 端数切り捨ては無視)。"
        "利益最大の賭け金 ≈ (EV − 1)/2 × その馬に入っている票\n",
        pd.DataFrame(guide).to_markdown(index=False, floatfmt=".1f"),
    ] if guide else []
    return [
        f"\n## 現実性: 自分の購入によるオッズ低下 × 締切前オッズのズレ (min_ev={a.min_ev})\n",
        f"- 単勝プール (1レースの単勝売上) **{a.pool_yen:,.0f} 円** を仮定 (調査による典型値 約250万円, "
        "通常レース 100〜500万円, 重賞 1,000万円超。レースごとの実売上が取れたら差し替えること), 払戻率 "
        f"{a.payout_rate:.0%}",
        "- 自分の購入 s 円を足した払戻オッズ: o' = R·(T+Σs)/(W_i+s_i), W_i = R·T/(o_i+0.05) を 0.1 刻みで切り捨て (最低 1.0)",
        f"- 締切前オッズ = 確定オッズ × exp(ε), ε~N(0, σ) (レース内で控除率を保って正規化)。"
        f"σ>0 は {a.noise_reps} 回の抽選の平均。判断 (期待値・AI勝率) は締切前オッズ、払戻は確定オッズ",
        f"- AI勝率は Stage2 の市場係数 b (ブロック別 {coef['b'].min():.3f}〜{coef['b'].max():.3f}, "
        f"予測値からの復元誤差 {exact:.1e}) で締切前の市場確率に合わせて計算し直している",
        "- 払戻オッズ比 = 実際に受け取るオッズ ÷ 自分が買わなかった場合の確定オッズ\n",
        f"### 設定シナリオ: プール {a.pool_yen:,.0f} 円, σ={a.odds_noise}\n",
        h.to_markdown(index=False, floatfmt=["", ".1f", ".3f", ".3f", "", "", ".3f"]),
        "\n### 均等買いの回収率: プール × σ × 1点の賭け金\n",
        "- 損益分岐の賭け金: これ以上を毎回賭けると回収率が 1 を下回る額 (100円単位)",
        "- 利益最大の賭け金: 期待収支 = 賭け金 × (回収率 − 1) が最大になる額。自分の影響はおおむね 賭け金/プール で決まるので、"
        "どちらもプールに比例する\n",
        g.to_markdown(index=False, floatfmt=g_fmt),
        *naive_lines,
        *guide_lines,
        f"\n### 1/4ケリー (1点上限 資金の3%) の資金推移, σ={a.odds_noise}\n",
        "- 資金が増えると賭け金が増え、自分の購入でオッズが下がる。1点上限 = プールの 1% で頭打ちにした場合も併記\n",
        k.to_markdown(index=False, floatfmt=".3f"),
        "\n> 前提の限界: 締切前オッズのズレは情報を含まない雑音として扱っている (実際には締切直前の資金は情報を持つことが多く、"
        "その場合の悪化はこれより大きい)。プールは全レース一定と仮定。詳細は docs/backtest_realism.md\n",
    ]


def cmd_backtest(a):
    config.ensure_dirs()
    df = _load(a.data)
    X = _features(df)
    if a.preds:
        print(f"ウォークフォワード予測を読み込み: {a.preds}")
        P = pd.read_pickle(a.preds)
    else:
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
    sh = B.split_half_check(P)
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
        "## 前半で選んだしきい値を後半に適用 (しきい値選びの楽観バイアスを除いた成績)\n",
        (f"- 分割日 {sh['cut']:%Y-%m-%d}: 前半で最良の min_ev = {sh['chosen_min_ev']} (前半回収率 {sh['first_half_roi']:.3f})\n"
         f"- **後半: {sh['second_half_bets']} 点, 回収率 {sh['second_half_roi']:.3f} "
         f"(90% 区間 [{sh['second_half_roi_lo90']:.3f}, {sh['second_half_roi_hi90']:.3f}])**\n"
         "- 区間の下限が 1 を超えていれば、実戦でもプラスが期待できる根拠になる\n")
        if sh.get("chosen_min_ev") is not None else "- 前半に十分な購入点数がありませんでした\n",
        per.to_markdown(index=False, floatfmt=".3f") if not per.empty else "(該当なし)",
        "\n## 較正 (AI 勝率 vs 実際の勝率)\n",
        cal.to_markdown(floatfmt=".3f"),
        "\n## Stage1 重要特徴量 (最終ブロックのモデル)\n",
    ]
    from .model import Stage1

    s1 = Stage1().fit(X[X["date"] >= X["date"].min() + pd.Timedelta(days=90)])
    lines.append(s1.importance().head(20).to_frame("gain").to_markdown(floatfmt=".0f"))
    lines.append("\n> 注意: 確定オッズでの検証です。購入時点のオッズとのズレ、自分の購入によるオッズ低下で実際の回収率は下がります。")
    print("現実性 (プール・締切前オッズ) の感度分析中...")
    realism = _realism_lines(P, a)
    lines += realism
    out = config.REPORT_DIR / "backtest.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:9]))
    print(grid.to_string(index=False))
    print("\n" + "\n".join(l for l in lines if "前半" in l or "後半" in l))
    print("\n" + "\n".join(realism[6:8]))
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
        card = schema.read_csv(a.card)
        card["date"] = pd.to_datetime(card["date"])
    else:
        races = []
        for rn in range(1, 13):
            if a.race and rn != a.race:
                continue
            t = _fetch_card(date, rn, a.card_source, a.wait)
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


def _fetch_card(date: pd.Timestamp, rn: int, source: str, wait: float) -> pd.DataFrame | None:
    """出馬表を取得. 馬・騎手 ID は過去データと同じ取得元に揃える必要がある.

    netkeiba: 出馬表 (ID は netkeiba 体系) + オッズは keiba.go.jp から馬番で結合
    keibago : 出馬表もオッズも keiba.go.jp (過去データも keibago で集めた場合)
    """
    from .scrape import keibago

    if source == "keibago":
        return keibago.scrape_card(date, rn, keibago.Fetcher(wait=wait))
    from .scrape import netkeiba

    t = netkeiba.scrape_card(date, rn, netkeiba.Fetcher(wait=wait))
    if t is None:
        return None
    try:
        f = keibago.Fetcher(wait=wait)
        o = keibago.parse_odds(f.get(keibago.odds_url(date, rn), use_cache=False))
    except Exception as e:  # noqa: BLE001
        print(f"  {rn}R: オッズ取得失敗 {e}")
        o = None
    if o is not None and not o.empty:
        t = t.drop(columns=[c for c in ("odds_win", "place_min", "place_max") if c in t]).merge(
            o, on="post", how="left")
    return t


def main(argv=None):
    import sys

    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "dashboard":
        # 引数はそのまま banei_ai.dashboard に渡す (python -m banei_ai.dashboard と同じ)
        from .dashboard import main as dashboard_main

        return dashboard_main(argv[1:])
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
        if name == "backtest":
            s.add_argument("--pool-yen", type=float, default=DEFAULT_WIN_POOL_YEN,
                           help="1レースの単勝売上 (円) の仮定。自分の購入によるオッズ低下の計算に使う。0 = 影響なし")
            s.add_argument("--odds-noise", type=float, default=0.0,
                           help="締切前オッズのズレ σ (締切前 = 確定 × exp(N(0,σ)))。0 = 確定オッズで判断")
            s.add_argument("--payout-rate", type=float, default=config.WIN_PAYOUT_RATE, help="単勝の払戻率")
            s.add_argument("--noise-reps", type=int, default=5, help="σ>0 のときの抽選回数")
            s.add_argument("--preds", default=None,
                           help="保存済みのウォークフォワード予測 (walkforward_preds.pkl) を使い、学習を省略")
        s.set_defaults(fn=fn)

    s = sub.add_parser("predict", help="当日予測")
    s.add_argument("--date", required=True)
    s.add_argument("--race", type=int, default=None)
    s.add_argument("--card", default=None, help="出馬表 CSV (正規スキーマ, status=entry)")
    s.add_argument("--card-source", choices=["netkeiba", "keibago"], default="netkeiba",
                   help="出馬表の取得元。過去データ (scrape --source) と必ず揃える")
    s.add_argument("--data", default=str(DEFAULT_DATA))
    s.add_argument("--model", default=str(DEFAULT_MODEL))
    s.add_argument("--min-ev", type=float, default=1.10)
    s.add_argument("--bankroll", type=float, default=100_000)
    s.add_argument("--wait", type=float, default=1.0)
    s.set_defaults(fn=cmd_predict)

    sub.add_parser("dashboard", help="結果の HTML ダッシュボード (詳細: dashboard --help)")

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
