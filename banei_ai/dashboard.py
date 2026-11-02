"""結果をひと目で見るための静的 HTML ダッシュボード.

    python -m banei_ai.dashboard --data data/interim/races.csv \\
        [--preds data/reports/walkforward_preds.pkl] [--pred-csv data/reports/pred_20261102.csv] \\
        --out data/reports/dashboard.html

* 1 ファイルで完結 (グラフは Python で生成したインライン SVG。CDN・外部ライブラリなし、閲覧時に通信しない)
* ライト/ダーク両対応 (prefers-color-scheme)、スマホ幅でも横スクロールしない (表は枠内でスクロール)
* 欠けている材料 (オッズなし・購入点数が少ない・予測 CSV なし) があっても、その節だけ省略して出力する

節: サマリー / 確率の較正 / 期待値しきい値と回収率 / 累積損益 / オッズの歪み / 騎手 / 馬 / 本日の予測
(+ backtest.robustness_table があれば「オッズ低下・ズレへの頑健性」)
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as B
from . import config, report, schema

SIM_BANNER = "シミュレーションデータ（実際の成績ではありません）"

SECTION_TITLES = {
    "summary": "サマリー",
    "calibration": "確率の較正",
    "threshold": "期待値しきい値と回収率",
    "pnl": "累積損益",
    "distortion": "オッズの歪み",
    "jockey": "騎手",
    "horse": "馬",
    "today": "本日の予測",
    "robust": "オッズ低下・ズレへの頑健性",
}

EV_GRID = (1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0)

# ---------------------------------------------------------------------------
# 書式
# ---------------------------------------------------------------------------


def esc(s) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def _isnan(x) -> bool:
    try:
        return x is None or (isinstance(x, (float, np.floating)) and math.isnan(x)) or pd.isna(x)
    except (TypeError, ValueError):
        return False


def _pct(x, d: int = 1, sign: bool = False) -> str:
    if _isnan(x):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def _num(x, d: int = 2, sign: bool = False) -> str:
    if _isnan(x):
        return "—"
    return f"{x:+,.{d}f}" if sign else f"{x:,.{d}f}"


def _int(x) -> str:
    return "—" if _isnan(x) else f"{int(round(float(x))):,}"


def _yen(x, sign: bool = True) -> str:
    if _isnan(x):
        return "—"
    return f"{x:+,.0f}円" if sign else f"{x:,.0f}円"


def _ci(lo, hi, fmt=_pct, d: int = 0) -> str:
    if _isnan(lo) or _isnan(hi):
        return "—"
    if fmt is _pct:
        return f"{lo * 100:.{d}f}〜{hi * 100:.{d}f}%"
    return f"{fmt(lo, d)}〜{fmt(hi, d)}"


def _f(x: float) -> str:
    """SVG 座標."""
    return f"{x:.1f}"


class _Skip(Exception):
    """材料が無いので節を省略する (想定内)."""


_NO_P = "ウォークフォワード予測が無いため省略 (理由はサマリーを参照)"


# ---------------------------------------------------------------------------
# SVG 部品
# ---------------------------------------------------------------------------


def _lin(d0: float, d1: float, r0: float, r1: float):
    span = (d1 - d0) or 1.0
    return lambda v: r0 + (v - d0) / span * (r1 - r0)


def _step(span: float, n: int) -> float:
    raw = max(span, 1e-12) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag + 1e-12:
            return m * mag
    return 10 * mag


def _nice(lo: float, hi: float, n: int = 5) -> tuple[float, float, list[float]]:
    if hi <= lo:
        hi = lo + 1.0
    st = _step(hi - lo, n)
    a = math.floor(lo / st + 1e-9) * st
    b = math.ceil(hi / st - 1e-9) * st
    ticks, v = [], a
    while v <= b + st * 1e-6:
        ticks.append(round(v, 10))
        v += st
    return a, b, ticks


def _tip_attrs(tip: str) -> str:
    """ホバー/タップ/フォーカスでツールチップを出すマーク (1 行目が値)."""
    label = tip.replace("\n", " / ")
    return f' class="mk" tabindex="0" role="img" aria-label="{esc(label)}" data-tip="{esc(tip)}"'


def _svg(w: int, h: int, body: str, label: str, extra: str = "", cls: str = "chart") -> str:
    return (f'<svg class="{cls}" viewBox="0 0 {w} {h}" role="group" aria-label="{esc(label)}"'
            f' preserveAspectRatio="xMidYMid meet"{extra}>{body}</svg>')


def _dot(x: float, y: float, cls: str, tip: str, r: float = 4.5) -> str:
    # 2px のサーフェスリング + 24px 以上の透明な当たり判定
    return (f'<g{_tip_attrs(tip)}><circle cx="{_f(x)}" cy="{_f(y)}" r="12" class="hit"/>'
            f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{r}" class="dot {cls}"/></g>')


def _col_path(x: float, y_top: float, w: float, y_base: float, r: float = 4.0) -> str:
    """角丸 (データ端のみ) の縦棒. 基線側は直角."""
    h = y_base - y_top
    if h <= 0.5:
        return f"M{_f(x)},{_f(y_base)}H{_f(x + w)}"
    r = min(r, w / 2, h)
    return (f"M{_f(x)},{_f(y_base)}V{_f(y_top + r)}A{r},{r} 0 0 1 {_f(x + r)},{_f(y_top)}"
            f"H{_f(x + w - r)}A{r},{r} 0 0 1 {_f(x + w)},{_f(y_top + r)}V{_f(y_base)}Z")


def _legend(items: list[tuple[str, str, str]]) -> str:
    """items: (kind[dot|line|bar|ref], css class, label)."""
    out = []
    for kind, cls, lab in items:
        out.append(f'<span class="lg"><i class="key {kind} {cls}" aria-hidden="true"></i>{esc(lab)}</span>')
    return f'<div class="legend">{"".join(out)}</div>'


def _table(df: pd.DataFrame, caption: str = "", row_cls: list[str] | None = None,
           num_cols: set[str] | None = None, raw_cols: set[str] | None = None) -> str:
    """DataFrame (表示用の文字列) → HTML 表. raw_cols の列は HTML をそのまま入れる."""
    num_cols = num_cols or set()
    raw_cols = raw_cols or set()
    head = "".join(f'<th scope="col"{" class=num" if c in num_cols else ""}>{esc(c)}</th>' for c in df.columns)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        cells = []
        for c in df.columns:
            v = r[c]
            v = "" if _isnan(v) else v
            v = v if c in raw_cols else esc(v)
            cells.append(f'<td{" class=num" if c in num_cols else ""}>{v}</td>')
        rc = f' class="{row_cls[i]}"' if row_cls and row_cls[i] else ""
        rows.append(f"<tr{rc}>{''.join(cells)}</tr>")
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return (f'<div class="tbl"><table>{cap}<thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _details(summary: str, body: str, open_: bool = False) -> str:
    return f'<details{" open" if open_ else ""}><summary>{esc(summary)}</summary>{body}</details>'


def _note(msg: str) -> str:
    return f'<p class="note"><span aria-hidden="true">ⓘ </span>{esc(msg)}</p>'


def _status(kind: str, text: str) -> str:
    icon = {"good": "✓", "warn": "!", "bad": "✕", "info": "ⓘ"}[kind]
    return f'<span class="st st-{kind}"><span class="ic" aria-hidden="true">{icon}</span>{esc(text)}</span>'


def _card(title: str, body: str, sub: str = "", wide: bool = False) -> str:
    s = f'<p class="sub">{sub}</p>' if sub else ""
    return f'<div class="card{" wide" if wide else ""}"><h3>{esc(title)}</h3>{s}{body}</div>'


# ---------------------------------------------------------------------------
# 1. サマリー
# ---------------------------------------------------------------------------


def _tile(label: str, value: str, sub: str = "", status: str = "") -> str:
    return (f'<div class="tile"><div class="t-label">{esc(label)}</div>'
            f'<div class="t-value">{esc(value)}</div>'
            f'{f"<div class=t-sub>{esc(sub)}</div>" if sub else ""}{status}</div>')


def section_summary(ctx: dict) -> str:
    P, rule = ctx["P"], ctx["rule"]
    if P is None:
        raise _Skip(ctx.get("p_error") or "ウォークフォワード予測がありません")
    tiles, extra = [], []
    vs = np.nan
    races = P["race_id"].nunique()
    tiles.append(_tile("評価レース数", f"{races:,}",
                       f"{P['date'].min():%Y-%m-%d}〜{P['date'].max():%Y-%m-%d} · 完全サンプル外"))
    # 対数損失
    try:
        from .model import evaluate_probs

        ev = evaluate_probs(P)
        row = ev.set_index("model")
        vs = float(row.loc["合成モデル", "vs_market"])
        st = _status("good", "市場より良い") if vs > 0 else _status("bad", "市場に負けている")
        tiles.append(_tile("対数損失の改善 (市場比)", f"{vs:+.4f}",
                           f"合成 {row.loc['合成モデル', 'logloss']:.4f} / 市場 "
                           f"{row.loc['市場 (オッズそのまま)', 'logloss']:.4f} (小さいほど良い)", st))
        t = ev.assign(logloss=ev["logloss"].map(lambda v: f"{v:.4f}"),
                      vs_market=ev["vs_market"].map(lambda v: f"{v:+.4f}"))
        t.columns = ["モデル", "対数損失 (1レースあたり)", "市場比 (+ が良い)"]
        extra.append(_details("対数損失の内訳", _table(t, num_cols={"対数損失 (1レースあたり)", "市場比 (+ が良い)"})))
    except Exception as e:  # noqa: BLE001
        tiles.append(_tile("対数損失の改善 (市場比)", "—", f"計算できません ({e})"))
    # 採用ルール
    bets = rule.select(P) if P["odds_win"].notna().any() else P.iloc[0:0]
    r = B.flat_bet(bets)
    lo, hi = B.bootstrap_roi(bets)
    ctx["bets"] = bets
    if r["bets"]:
        st = (_status("good", "区間下限 > 100%") if lo > 1 else
              _status("warn", "区間が 100% をまたぐ") if hi >= 1 else _status("bad", "区間上限 < 100%"))
        tiles.append(_tile(f"期待値 ≥ {rule.min_ev:g} の回収率", _pct(r["roi"], 0),
                           f"{r['bets']:,} 点 · 的中 {_pct(r['hit_rate'], 1)} · 90%区間 {_ci(lo, hi)}", st))
    else:
        tiles.append(_tile(f"期待値 ≥ {rule.min_ev:g} の回収率", "—", "購入対象がありません"))
    # 前後半
    sh = B.split_half_check(P) if r["bets"] else {"chosen_min_ev": None}
    if sh.get("chosen_min_ev") is not None:
        lo2, hi2 = sh["second_half_roi_lo90"], sh["second_half_roi_hi90"]
        st = (_status("good", "後半の区間下限 > 100%") if lo2 > 1 else
              _status("warn", "後半の区間が 100% をまたぐ") if (_isnan(hi2) or hi2 >= 1)
              else _status("bad", "後半の区間上限 < 100%"))
        tiles.append(_tile("前後半検証 (後半の回収率)", _pct(sh["second_half_roi"], 0),
                           f"前半で選んだ EV≥{sh['chosen_min_ev']:g} を {sh['cut']:%Y-%m-%d} 以降に適用: "
                           f"{sh['second_half_bets']:,} 点 · 90%区間 {_ci(lo2, hi2)}", st))
    else:
        tiles.append(_tile("前後半検証 (後半の回収率)", "—", "前半に十分な購入点数がありません (30点以上が必要)"))
    ctx["split"] = sh
    verdict = ""
    try:
        good_ll = not _isnan(vs) and vs > 0
        good_sh = sh.get("chosen_min_ev") is not None and sh["second_half_roi_lo90"] > 1
        if good_ll and good_sh:
            verdict = _status("good", "本番投入の目安を満たしています (市場より良い確率 + 後半の回収率の区間下限 > 100%)")
        else:
            why = []
            if not good_ll:
                why.append("確率の精度が市場以下")
            if not good_sh:
                why.append("後半の回収率の区間下限が 100% 以下")
            verdict = _status("warn", "本番投入の目安は未達: " + "・".join(why))
    except Exception:  # noqa: BLE001
        pass
    return (f'<div class="verdict">{verdict}</div><div class="tiles">{"".join(tiles)}</div>' + "".join(extra))


# ---------------------------------------------------------------------------
# 2. 較正
# ---------------------------------------------------------------------------


def _wilson(k: float, n: float, z: float = 1.645) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def _bin_label(iv) -> str:
    try:
        return f"{iv.left * 100:g}〜{iv.right * 100:g}%"
    except AttributeError:
        return str(iv)


def section_calibration(ctx: dict) -> str:
    P = ctx["P"]
    if P is None:
        raise _Skip(_NO_P)
    series = [("p_model", "AI勝率 (合成モデル)", "s1"), ("p_mkt", "市場確率 (オッズ)", "s2")]
    cals = []
    for col, name, cls in series:
        if col in P and P[col].notna().any():
            c = B.calibration(P, col)
            c = c[c["n"] > 0].copy()
            ci = [_wilson(a * n, n) for a, n in zip(c["actual"], c["n"])]
            c["lo"], c["hi"] = [x[0] for x in ci], [x[1] for x in ci]
            cals.append((col, name, cls, c))
    if not cals:
        raise _Skip("確率の列がありません")
    m = max(max(c["pred"].max(), c["actual"].max()) for *_, c in cals)
    _, vmax, ticks = _nice(0, min(1.0, m * 1.08), 5)
    W, H, ml, mr, mt, mb = 380, 360, 46, 14, 14, 36
    sx, sy = _lin(0, vmax, ml, W - mr), _lin(0, vmax, H - mb, mt)
    g = []
    for t in ticks:
        g.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{_f(sy(t))}" y2="{_f(sy(t))}"/>')
        g.append(f'<text class="tk" x="{ml - 6}" y="{_f(sy(t) + 4)}" text-anchor="end">{t * 100:g}%</text>')
        g.append(f'<text class="tk" x="{_f(sx(t))}" y="{H - mb + 16}" text-anchor="middle">{t * 100:g}%</text>')
    g.append(f'<line class="axis" x1="{ml}" x2="{W - mr}" y1="{H - mb}" y2="{H - mb}"/>')
    g.append(f'<line class="ref" x1="{_f(sx(0))}" y1="{_f(sy(0))}" x2="{_f(sx(vmax))}" y2="{_f(sy(vmax))}"/>')
    ang = -math.degrees(math.atan2(sy(0) - sy(vmax), sx(vmax) - sx(0)))
    lx, ly = sx(vmax * 0.97), sy(vmax * 0.97)
    g.append(f'<text class="tk halo" x="{_f(lx)}" y="{_f(ly)}" dy="-5" text-anchor="end" '
             f'transform="rotate({ang:.1f} {_f(lx)} {_f(ly)})">完全一致</text>')
    g.append(f'<text class="lb" x="{(ml + W - mr) / 2}" y="{H - 4}" text-anchor="middle">予測した勝率</text>')
    g.append(f'<text class="lb" x="{ml + 4}" y="{mt + 4}" text-anchor="start">実際の勝率 ↑</text>')
    ends = []
    for col, name, cls, c in cals:
        for iv, r in c.iterrows():
            x = sx(r["pred"])
            y0, y1 = sy(min(r["hi"], vmax)), sy(r["lo"])
            g.append(f'<line class="whisk {cls}" x1="{_f(x)}" x2="{_f(x)}" y1="{_f(y0)}" y2="{_f(y1)}"/>')
        for iv, r in c.iterrows():
            tip = (f"実際 {_pct(r['actual'])}\n{name} · 予測 {_bin_label(iv)} の帯\n"
                   f"予測平均 {_pct(r['pred'])} · {int(r['n']):,} 頭\n90%区間 {_pct(r['lo'])}〜{_pct(r['hi'])}")
            g.append(_dot(sx(r["pred"]), sy(min(r["actual"], vmax)), cls, tip))
        last = c.iloc[-1]
        ends.append((sx(last["pred"]), sy(min(last["actual"], vmax)), name.split(" ")[0]))
    # 直接ラベルは重ならない時だけ (重なるなら凡例とツールチップに任せる)
    if len(ends) < 2 or abs(ends[0][1] - ends[1][1]) >= 16:
        for x, y, lab in ends:
            g.append(f'<text class="lb" x="{_f(x - 9)}" y="{_f(y + 4)}" text-anchor="end">{esc(lab)}</text>')
    svg = _svg(W, H, "".join(g), "確率の較正: 予測した勝率と実際の勝率")
    leg = _legend([("dot", cls, name) for _, name, cls, _ in cals] + [("ref", "", "完全一致 (y = x)")])
    # 表
    rows = {}
    for col, name, cls, c in cals:
        for iv, r in c.iterrows():
            d = rows.setdefault(_bin_label(iv), {"予測の帯": _bin_label(iv)})
            short = "AI" if col == "p_model" else "市場"
            d[f"{short} 頭数"] = _int(r["n"])
            d[f"{short} 予測"] = _pct(r["pred"])
            d[f"{short} 実際"] = _pct(r["actual"])
    tdf = pd.DataFrame(list(rows.values())).fillna("—")
    alpha_txt = ""
    try:
        from .market import fit_power_calibration

        a = fit_power_calibration(P)
        alpha_txt = (f"市場の favorite-longshot 指数 α = {a:.3f}"
                     f"（α > 1: 本命が過小評価・大穴が過大評価）。")
    except Exception:  # noqa: BLE001
        pass
    sub = ("点が対角線に乗っていれば「20%と言った馬は実際に20%勝つ」。縦線は実際の勝率の 90% 区間。" + alpha_txt)
    body = leg + svg + _details("表で見る", _table(tdf, num_cols=set(tdf.columns) - {"予測の帯"}))
    return f'<div class="grid1">{_card("予測した勝率 vs 実際の勝率", body, esc(sub))}</div>'


# ---------------------------------------------------------------------------
# 3. しきい値
# ---------------------------------------------------------------------------


def _threshold_rows(P: pd.DataFrame, rule: B.BetRule) -> pd.DataFrame:
    evs = sorted(set(EV_GRID) | {round(rule.min_ev, 4)})
    rows = []
    for ev in evs:
        b = replace(rule, min_ev=ev).select(P)
        r = B.flat_bet(b)
        lo, hi = B.bootstrap_roi(b, n_boot=1000)
        rows.append({"min_ev": ev, **r, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def section_threshold(ctx: dict) -> str:
    P, rule = ctx["P"], ctx["rule"]
    if P is None:
        raise _Skip(_NO_P)
    if not P["odds_win"].notna().any():
        raise _Skip("オッズが無いため回収率を計算できません")
    T = _threshold_rows(P, rule)
    n = len(T)
    W, ml, mr = 380, 46, 14
    bw = (W - ml - mr) / n
    cx = [ml + (i + 0.5) * bw for i in range(n)]
    di = int(np.argmin(np.abs(T["min_ev"].to_numpy() - rule.min_ev)))
    # (a) 回収率 + 90% 区間
    H, mt, mb = 240, 22, 34
    vals = pd.concat([T["lo"], T["hi"], T["roi"]]).dropna()
    lo_d = min(0.6, vals.min()) if len(vals) else 0.6
    hi_d = max(1.4, vals.max()) if len(vals) else 1.4
    hi_d = min(hi_d, 4.0)
    a, b, ticks = _nice(max(0.0, lo_d), hi_d, 5)
    sy = _lin(a, b, H - mb, mt)
    g = [f'<rect class="hl" x="{_f(cx[di] - bw / 2)}" y="{mt - 16}" width="{_f(bw)}" height="{_f(H - mb - mt + 16)}"/>',
         f'<text class="lb" x="{_f(cx[di])}" y="{mt - 5}" text-anchor="middle">採用</text>']
    for t in ticks:
        g.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{_f(sy(t))}" y2="{_f(sy(t))}"/>')
        g.append(f'<text class="tk" x="{ml - 6}" y="{_f(sy(t) + 4)}" text-anchor="end">{t * 100:g}%</text>')
    g.append(f'<line class="ref" x1="{ml}" x2="{W - mr}" y1="{_f(sy(1))}" y2="{_f(sy(1))}"/>')
    g.append(f'<text class="tk halo" x="{ml + 4}" y="{_f(sy(1) + 14)}" text-anchor="start">損益分岐 100%</text>')
    for i, t in enumerate(T["min_ev"]):
        g.append(f'<text class="tk" x="{_f(cx[i])}" y="{H - mb + 16}" text-anchor="middle">{t:g}</text>')
    g.append(f'<text class="lb" x="{(ml + W - mr) / 2}" y="{H - 3}" text-anchor="middle">期待値しきい値 (AI勝率 × オッズ ≥)</text>')
    pts = [(cx[i], sy(min(r["roi"], b))) for i, r in T.iterrows() if not _isnan(r["roi"])]
    if len(pts) > 1:
        g.append('<path class="line s1" d="M' + " L".join(f"{_f(x)},{_f(y)}" for x, y in pts) + '"/>')
    for i, r in T.iterrows():
        if _isnan(r["roi"]):
            continue
        g.append(f'<line class="whisk s1" x1="{_f(cx[i])}" x2="{_f(cx[i])}" '
                 f'y1="{_f(sy(min(r["hi"], b)))}" y2="{_f(sy(max(r["lo"], a)))}"/>')
    for i, r in T.iterrows():
        if _isnan(r["roi"]):
            continue
        tip = (f"回収率 {_pct(r['roi'])}\n期待値 ≥ {r['min_ev']:g} · {int(r['bets']):,} 点\n"
               f"90%区間 {_ci(r['lo'], r['hi'], d=1)} · 的中 {_pct(r['hit_rate'])}\n損益 {_yen(r['profit'])} (100円均等)")
        g.append(_dot(cx[i], sy(min(r["roi"], b)), "s1", tip))
    rd = T.iloc[di]
    if not _isnan(rd["roi"]):
        ty = sy(min(rd["roi"], b))
        anchor, dx = ("start", 9) if di < n - 1 else ("end", -9)
        g.append(f'<text class="vl halo" x="{_f(cx[di] + dx)}" y="{_f(ty - 6)}" '
                 f'text-anchor="{anchor}">{_pct(rd["roi"], 0)}</text>')
    svg_a = _svg(W, H, "".join(g), "期待値しきい値ごとの回収率と90%区間")
    # (b) 購入点数
    H2, mt2, mb2 = 170, 16, 34
    nmax = max(1, int(T["bets"].max()))
    a2, b2, ticks2 = _nice(0, nmax, 4)
    sy2 = _lin(0, b2, H2 - mb2, mt2)
    cw = min(24.0, bw * 0.6)
    g = [f'<rect class="hl" x="{_f(cx[di] - bw / 2)}" y="{mt2 - 8}" width="{_f(bw)}" height="{_f(H2 - mb2 - mt2 + 8)}"/>']
    for t in ticks2:
        g.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{_f(sy2(t))}" y2="{_f(sy2(t))}"/>')
        g.append(f'<text class="tk" x="{ml - 6}" y="{_f(sy2(t) + 4)}" text-anchor="end">{_int(t)}</text>')
    for i, r in T.iterrows():
        tip = f"{int(r['bets']):,} 点\n期待値 ≥ {r['min_ev']:g}\n的中 {_int(r['hits'])} 回 · 平均オッズ {_num(r['avg_odds'], 1)}"
        y = sy2(r["bets"])
        g.append(f'<g{_tip_attrs(tip)}><rect class="hit" x="{_f(cx[i] - bw / 2)}" y="{mt2}" width="{_f(bw)}" '
                 f'height="{_f(H2 - mb2 - mt2)}"/><path class="bar s1" d="{_col_path(cx[i] - cw / 2, y, cw, H2 - mb2)}"/></g>')
        g.append(f'<text class="tk" x="{_f(cx[i])}" y="{H2 - mb2 + 16}" text-anchor="middle">{r["min_ev"]:g}</text>')
    g.append(f'<line class="axis" x1="{ml}" x2="{W - mr}" y1="{H2 - mb2}" y2="{H2 - mb2}"/>')
    g.append(f'<text class="vl" x="{_f(cx[di])}" y="{_f(sy2(rd["bets"]) - 6)}" text-anchor="middle">{_int(rd["bets"])}</text>')
    g.append(f'<text class="lb" x="{(ml + W - mr) / 2}" y="{H2 - 3}" text-anchor="middle">期待値しきい値</text>')
    svg_b = _svg(W, H2, "".join(g), "期待値しきい値ごとの購入点数")
    tdf = pd.DataFrame({
        "しきい値": [f"≥ {v:g}" for v in T["min_ev"]],
        "点数": T["bets"].map(_int), "的中率": T["hit_rate"].map(_pct),
        "回収率": T["roi"].map(_pct), "90%区間": [_ci(lo, hi, d=1) for lo, hi in zip(T["lo"], T["hi"])],
        "損益(100円)": T["profit"].map(_yen), "最大DD": T["max_dd"].map(lambda v: _yen(v, sign=False)),
        "平均オッズ": T["avg_odds"].map(lambda v: _num(v, 1)),
    })
    sub = (f"購入条件は期待値以外 AI勝率 ≥ {rule.min_prob:.0%}・オッズ ≤ {rule.max_odds:g}倍・"
           f"1レース {rule.max_per_race} 頭まで。帯は採用中のしきい値。しきい値を上げるほど点数が減り、区間が広がる。")
    body = (_card("回収率 (90% 区間つき)", svg_a, esc(sub))
            + _card("購入点数", svg_b + _details("表で見る", _table(tdf, num_cols=set(tdf.columns) - {"しきい値"})),
                    "同じしきい値の並び。回収率と点数は尺度が違うので別のグラフにしています。"))
    return f'<div class="grid2">{body}</div>'


# ---------------------------------------------------------------------------
# 4. 累積損益
# ---------------------------------------------------------------------------


def _pnl_svg(daily: pd.DataFrame, d0: pd.Timestamp, d1: pd.Timestamp, cut, label: str,
             W: int, H: int, n_xticks: int, cls: str) -> str:
    """累積損益の折れ線 (幅違いを 2 枚作って CSS で出し分ける)."""
    ml, mr, mt, mb = 50, 14, 24, 30
    t0 = d0.value
    span = max(d1.value - t0, 1)
    sx = lambda d: ml + (pd.Timestamp(d).value - t0) / span * (W - ml - mr)  # noqa: E731
    lo, hi = min(0.0, daily["cum"].min()), max(0.0, daily["cum"].max())
    a, bb, ticks = _nice(lo, hi if hi > lo else lo + 1000, 5)
    sy = _lin(a, bb, H - mb, mt)
    big = max(abs(a), abs(bb)) >= 10000
    fmt_y = (lambda v: "0" if v == 0 else f"{v / 10000:g}万") if big else (lambda v: f"{v:,.0f}")
    g = []
    for t in ticks:
        g.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{_f(sy(t))}" y2="{_f(sy(t))}"/>')
        g.append(f'<text class="tk" x="{ml - 6}" y="{_f(sy(t) + 4)}" text-anchor="end">{fmt_y(t)}</text>')
    g.append(f'<line class="axis" x1="{ml}" x2="{W - mr}" y1="{_f(sy(0))}" y2="{_f(sy(0))}"/>')
    months = pd.date_range(d0.to_period("M").to_timestamp(), d1, freq="MS")
    months = [m for m in months if m >= d0]
    k = max(1, math.ceil(len(months) / n_xticks))
    for m in months[::k]:
        x = sx(m)
        g.append(f'<line class="axis" x1="{_f(x)}" x2="{_f(x)}" y1="{H - mb}" y2="{H - mb + 4}"/>')
        g.append(f'<text class="tk" x="{_f(x)}" y="{H - mb + 17}" text-anchor="middle">{m.year}/{m.month}</text>')
    g.append(f'<text class="lb" x="{ml - 44}" y="{mt - 10}" text-anchor="start">累積損益 (円)</text>')
    if cut is not None and d0 < cut < d1:
        x = sx(cut)
        g.append(f'<line class="ref" x1="{_f(x)}" x2="{_f(x)}" y1="{mt}" y2="{H - mb}"/>')
        g.append(f'<text class="tk" x="{_f(x + 4)}" y="{mt + 10}" text-anchor="start">後半 →</text>')
    pts = [(sx(d0), sy(0))] + [(sx(d), sy(v)) for d, v in daily["cum"].items()]
    path = "M" + " L".join(f"{_f(x)},{_f(y)}" for x, y in pts)
    g.append(f'<path class="area s1" d="{path} L{_f(pts[-1][0])},{_f(sy(0))} Z"/>')
    g.append(f'<path class="line s1" d="{path}"/>')
    ex, ey = pts[-1]
    g.append(f'<circle class="dot s1" cx="{_f(ex)}" cy="{_f(ey)}" r="4.5"/>')
    last = daily.iloc[-1]
    g.append(f'<text class="vl halo" x="{_f(ex - 8)}" y="{_f(ey - 9 if ey > mt + 20 else ey + 18)}" '
             f'text-anchor="end">{_yen(last["cum"])}</text>')
    # クロスヘア (最も近い日付にスナップ)
    xh = [[round(sx(d), 1), round(sy(r["cum"]), 1),
           f"{_yen(r['cum'])}\n{d:%Y-%m-%d} 時点 · 累計 {int(r['ncum']):,} 点\n回収率 {_pct(r['roi'])} · この日 {_yen(r['pnl'])}"]
          for d, r in daily.iterrows()]
    g.append(f'<line class="xhair" x1="0" x2="0" y1="{mt}" y2="{H - mb}" visibility="hidden"/>')
    g.append('<circle class="xdot dot s1" cx="0" cy="0" r="4.5" visibility="hidden"/>')
    g.append(f'<rect class="hit" x="{ml}" y="{mt}" width="{W - ml - mr}" height="{H - mb - mt}"/>')
    return _svg(W, H, "".join(g), label, cls=cls,
                extra=f' data-xhair="{esc(json.dumps(xh, ensure_ascii=False))}"')


def section_pnl(ctx: dict) -> str:
    P, rule = ctx["P"], ctx["rule"]
    if P is None:
        raise _Skip(_NO_P)
    bets = ctx.get("bets")
    if bets is None:
        bets = rule.select(P) if P["odds_win"].notna().any() else P.iloc[0:0]
    if bets.empty:
        raise _Skip(f"期待値 ≥ {rule.min_ev:g} の購入対象がありません")
    b = bets.sort_values(["date", "race_id"]).assign(pnl=lambda d: d["win"] * d["odds_win"] * 100 - 100,
                                                     ret=lambda d: d["win"] * d["odds_win"] * 100)
    daily = b.groupby("date").agg(pnl=("pnl", "sum"), n=("pnl", "size"), ret=("ret", "sum"))
    daily["cum"] = daily["pnl"].cumsum()
    daily["ncum"] = daily["n"].cumsum()
    daily["roi"] = daily["ret"].cumsum() / (daily["ncum"] * 100)
    d0, d1 = P["date"].min(), P["date"].max()
    cut = (ctx.get("split") or {}).get("cut")
    label = f"期待値 ≥ {rule.min_ev:g} を100円ずつ買った場合の累積損益"
    svg = (_pnl_svg(daily, d0, d1, cut, label, 380, 250, 5, "chart only-narrow")
           + _pnl_svg(daily, d0, d1, cut, label, 720, 300, 9, "chart only-wide"))
    r = B.flat_bet(bets)
    by_m = b.assign(m=b["date"].dt.strftime("%Y-%m")).groupby("m").agg(
        n=("pnl", "size"), hits=("win", "sum"), ret=("ret", "sum"), pnl=("pnl", "sum"))
    by_m["cum"] = by_m["pnl"].cumsum()
    tdf = pd.DataFrame({"月": by_m.index, "点数": by_m["n"].map(_int), "的中": by_m["hits"].map(_int),
                        "回収率": (by_m["ret"] / (by_m["n"] * 100)).map(lambda v: _pct(v, 0)),
                        "損益": by_m["pnl"].map(_yen), "累積": by_m["cum"].map(_yen)})
    sub = (f"期待値 ≥ {rule.min_ev:g} の馬を毎回 100 円ずつ買った場合 (確定オッズ)。"
           f"{r['bets']:,} 点 · 最終 {_yen(r['profit'])} · 最大ドローダウン {_yen(r['max_dd'], sign=False)}。"
           "縦線より右が前後半検証の「後半」。")
    body = svg + _details("月別の表で見る", _table(tdf, num_cols=set(tdf.columns) - {"月"}))
    return f'<div class="grid1">{_card("採用ルールの累積損益", body, esc(sub), wide=True)}</div>'


# ---------------------------------------------------------------------------
# 5. オッズの歪み
# ---------------------------------------------------------------------------

DIST_ORDER = ["オッズ帯", "人気順", "前走着順", "馬場水分適性", "負担重量の増減", "馬体重の増減", "騎手の人気度", "乗り替わり"]
MOIST_NAMES = {5: ["苦手", "やや苦手", "ふつう", "やや得意", "得意"]}


def _cat_labels(name: str, idx: pd.Index) -> list[str]:
    out = []
    for i, v in enumerate(idx):
        if name == "オッズ帯" and hasattr(v, "left"):
            out.append(f"{v.left:g}倍以上" if v.right >= 1000 else f"{v.left:g}〜{v.right:g}倍")
        elif name == "人気順":
            out.append(f"{int(v)}番人気" if not _isnan(v) else "不明")
        elif name == "馬場水分適性":
            names = MOIST_NAMES.get(len(idx))
            out.append(f"Q{i + 1} {names[i]}" if names else f"Q{i + 1}")
        elif name == "乗り替わり":
            out.append({0: "継続騎乗", 1: "乗り替わり"}.get(int(v), str(v)) if not _isnan(v) else "不明")
        else:
            out.append(str(v))
    return out


def _forest(name: str, t: pd.DataFrame, labels: list[str], dom: tuple[float, float, list[float]]) -> str:
    a, b, ticks = dom
    n = len(t)
    W, lw, rw, row, mt, mb = 380, 104, 60, 26, 8, 26
    H = mt + n * row + mb
    sx = _lin(a, b, lw + 6, W - rw)
    g = []
    for tk in ticks:
        g.append(f'<line class="grid" x1="{_f(sx(tk))}" x2="{_f(sx(tk))}" y1="{mt}" y2="{H - mb}"/>')
        g.append(f'<text class="tk" x="{_f(sx(tk))}" y="{H - mb + 16}" text-anchor="middle">{tk:g}</text>')
    if a <= 1 <= b:
        g.append(f'<line class="ref" x1="{_f(sx(1))}" x2="{_f(sx(1))}" y1="{mt}" y2="{H - mb}"/>')
    for i, ((_, r), lab) in enumerate(zip(t.iterrows(), labels)):
        y = mt + (i + 0.5) * row
        sig = r.get("signal", "") or ""
        cls = "pos" if sig == "過小評価" else "neg" if sig == "過大評価" else "neu"
        g.append(f'<text class="lb" x="{lw}" y="{_f(y + 4)}" text-anchor="end">{esc(lab)}</text>')
        if _isnan(r["ae"]):
            g.append(f'<text class="tk" x="{_f(sx(1))}" y="{_f(y + 4)}" text-anchor="middle">勝利なし/データ不足</text>')
            continue
        lo, hi = r["ae_lo90"], r["ae_hi90"]
        x0, x1 = sx(max(a, lo if not _isnan(lo) else r["ae"])), sx(min(b, hi if not _isnan(hi) else r["ae"]))
        g.append(f'<line class="whisk {cls}" x1="{_f(x0)}" x2="{_f(x1)}" y1="{_f(y)}" y2="{_f(y)}"/>')
        if not _isnan(hi) and hi > b:  # 範囲外は矢印で示す
            g.append(f'<path class="arrow {cls}" d="M{_f(x1)},{_f(y - 4)} L{_f(x1 + 5)},{_f(y)} L{_f(x1)},{_f(y + 4)}Z"/>')
        tip = (f"A/E {r['ae']:.2f}\n{name}: {lab}{' · ' + sig if sig else ''}\n"
               f"{int(r['bets']):,} 頭 · 勝利 {int(r['wins'])} / 市場の期待 {r['exp_wins']:.1f}\n"
               f"90%区間 {_num(lo)}〜{_num(hi)} · 単勝回収率 {_pct(r['roi'], 0)}")
        g.append(_dot(sx(min(max(r["ae"], a), b)), y, cls, tip, r=4.5))
        if sig:
            g.append(f'<text class="lb" x="{W - rw + 8}" y="{_f(y + 4)}" text-anchor="start">{esc(sig)}</text>')
    return _svg(W, H, "".join(g), f"オッズの歪み: {name} ごとの A/E と 90%区間")


def section_distortion(ctx: dict) -> str:
    X = ctx["X"]
    if X is None or "p_mkt" not in X or not X["p_mkt"].notna().any():
        raise _Skip("オッズが無いため、市場との比較 (A/E) はできません")
    dist = report.distortion_report(X)
    keys = [k for k in DIST_ORDER if k in dist] + [k for k in dist if k not in DIST_ORDER]
    dist = {k: dist[k] for k in keys if len(dist[k])}
    if not dist:
        raise _Skip("集計できるデータがありません")
    allv = pd.concat([pd.concat([t["ae_lo90"], t["ae_hi90"], t["ae"]]) for t in dist.values()]).replace(
        [np.inf, -np.inf], np.nan).dropna()
    lo = min(0.5, allv.min()) if len(allv) else 0.5
    hi = max(1.5, allv.quantile(0.97)) if len(allv) else 1.5
    dom = _nice(max(0.0, lo), min(hi, 2.5), 5)
    cards, tables = [], []
    for k, t in dist.items():
        labels = _cat_labels(k, t.index)
        cards.append(_card(k, _forest(k, t, labels, dom)))
        tdf = pd.DataFrame({"区分": labels, "頭数": t["bets"].map(_int), "勝利": t["wins"].map(_int),
                            "市場の期待勝利": t["exp_wins"].map(lambda v: _num(v, 1)), "A/E": t["ae"].map(_num),
                            "90%区間": [f"{_num(a)}〜{_num(b)}" for a, b in zip(t["ae_lo90"], t["ae_hi90"])],
                            "単勝回収率": t["roi"].map(lambda v: _pct(v, 0)), "判定": t["signal"].fillna("")})
        tables.append(_table(tdf, caption=k, num_cols={"頭数", "勝利", "市場の期待勝利", "A/E", "90%区間", "単勝回収率"}))
    leg = _legend([("dot", "pos", "過小評価 (区間が 1 より上 = 市場が見落とし、買い材料)"),
                   ("dot", "neg", "過大評価 (区間が 1 より下 = 人気しすぎ、消し材料)"),
                   ("dot", "neu", "有意差なし (区間が 1 をまたぐ)")])
    lead = ('<p class="sub">A/E = 実際の勝利数 ÷ 市場 (オッズ) が予想した勝利数。1.0 なら市場どおり。'
            "点が A/E、横線が 90% 区間。全期間・全出走馬の集計 (切り口ごとに独立)。</p>")
    return (lead + leg + f'<div class="grid2">{"".join(cards)}</div>'
            + _details("表で見る (全切り口)", "".join(tables)))


# ---------------------------------------------------------------------------
# 6. 騎手
# ---------------------------------------------------------------------------


def _mini_ci(ae, lo, hi, sig: str, dom: tuple[float, float]) -> str:
    if _isnan(ae):
        return ""
    a, b = dom
    W, H = 112, 18
    sx = _lin(a, b, 6, W - 6)
    cls = "pos" if "過小" in sig else "neg" if "過大" in sig else "neu"
    x0 = sx(max(a, lo if not _isnan(lo) else ae))
    x1 = sx(min(b, hi if not _isnan(hi) else ae))
    one = f'<line class="ref" x1="{_f(sx(1))}" x2="{_f(sx(1))}" y1="1" y2="{H - 1}"/>' if a <= 1 <= b else ""
    return (f'<svg class="mini" width="{W}" height="{H}" viewBox="0 0 {W} {H}" aria-hidden="true">{one}'
            f'<line class="whisk {cls}" x1="{_f(x0)}" x2="{_f(x1)}" y1="{H / 2}" y2="{H / 2}"/>'
            f'<circle class="dot {cls}" cx="{_f(sx(min(max(ae, a), b)))}" cy="{H / 2}" r="4"/></svg>')


def section_jockey(ctx: dict) -> str:
    X = ctx["X"]
    if X is None or X.empty:
        raise _Skip("データがありません")
    since = X["date"].max() - pd.Timedelta(days=ctx["days"])
    jt = report.jockey_table(X, since=since, min_rides=ctx["min_rides"])
    if jt.empty:
        raise _Skip(f"直近 {ctx['days']} 日で {ctx['min_rides']} 騎乗以上の騎手がいません")
    skill_note = ""
    try:
        sk = report.jockey_skill_clogit(X[X["date"] >= since], min_rides=ctx["min_rides"])
        jt = jt.join(sk[["skill"]], how="left")
    except Exception as e:  # noqa: BLE001
        jt["skill"] = np.nan
        skill_note = f"固有の腕 (条件付きロジット) は計算できませんでした: {e}"
    has_mkt = bool(jt["exp_mkt"].gt(0).any() and jt["mkt_ae"].replace([np.inf, -np.inf], np.nan).notna().any())
    if not has_mkt:
        jt = jt.sort_values("win_ae", ascending=False)
    vals = pd.concat([jt["mkt_ae_lo90"], jt["mkt_ae_hi90"]]).replace([np.inf, -np.inf], np.nan).dropna()
    dom = (max(0.0, math.floor(min(0.7, vals.min()) * 10) / 10) if len(vals) else 0.5,
           min(2.5, math.ceil(max(1.3, vals.max()) * 10) / 10) if len(vals) else 1.5)
    rows = []
    for _, r in jt.iterrows():
        sig = r.get("verdict", "") or ""
        sk_v = r.get("skill", np.nan)
        d = {"騎手": r["name"]}
        if has_mkt:  # スマホで最初に見える列ほど重要なものを左に
            d.update({"市場比 A/E": _num(r["mkt_ae"]),
                      "区間図": _mini_ci(r["mkt_ae"], r["mkt_ae_lo90"], r["mkt_ae_hi90"], sig, dom),
                      "判定": sig})
        d.update({"勝率": _pct(r["win_rate"]),
                  "固有の腕": "—" if _isnan(sk_v) else f"{(math.exp(sk_v) - 1) * 100:+.0f}%"})
        if has_mkt:
            d.update({"人気度": _num(r["popularity"]),
                      "90%区間": f"{_num(r['mkt_ae_lo90'])}〜{_num(r['mkt_ae_hi90'])}",
                      "単勝回収率": _pct(r["roi"], 0)})
        d.update({"騎乗": _int(r["rides"]), "複勝率": _pct(r["top3_rate"]), "頭数比 A/E": _num(r["win_ae"])})
        rows.append(d)
    tdf = pd.DataFrame(rows)
    rcls = ["pick" if "過小" in (v or "") else "" for v in jt.get("verdict", pd.Series([""] * len(jt)))]
    tbl = _table(tdf, row_cls=rcls, raw_cols={"区間図"},
                 num_cols={"騎乗", "勝率", "複勝率", "頭数比 A/E", "人気度", "市場比 A/E", "90%区間", "単勝回収率", "固有の腕"})
    if has_mkt:
        sub = (f"直近 {ctx['days']} 日・{ctx['min_rides']} 騎乗以上。市場比 A/E 順。"
               "市場比 A/E = 実勝利 ÷ オッズが示す期待勝利 (> 1 なら市場が過小評価)。"
               "人気度 > 1 は人気を集めやすい騎手。固有の腕 = 馬の強さを統制した勝ちやすさ (平均騎手比, 縮小推定)。"
               "区間図の縦線は 1.0。")
    else:
        sub = (f"直近 {ctx['days']} 日・{ctx['min_rides']} 騎乗以上。オッズが無いので市場との比較は省略し、"
               "頭数比 A/E (実勝利 ÷ 頭数から見た期待勝利) 順。固有の腕 = 馬の強さを統制した勝ちやすさ (平均騎手比)。")
    out = f'<p class="sub">{esc(sub)}</p>' + tbl
    if skill_note:
        out += _note(skill_note)
    return out


# ---------------------------------------------------------------------------
# 7. 馬
# ---------------------------------------------------------------------------


def section_horse(ctx: dict, top: int = 20, active_days: int = 120) -> str:
    X = ctx["X"]
    if X is None or X.empty:
        raise _Skip("データがありません")
    ht = report.horse_table(X, top=100000)
    recent = ht[ht["date"] >= X["date"].max() - pd.Timedelta(days=active_days)].head(top)
    if recent.empty:
        raise _Skip("最近出走した馬がいません")
    tdf = pd.DataFrame({
        "馬名": recent["horse_name"], "クラス": recent.get("race_class", pd.Series(index=recent.index, dtype=object)),
        "レート": recent["h_rating"].map(lambda v: _num(v, 0)), "出走数": recent["h_n_runs"].map(_int),
        "近走指数": recent["h_ewm_pct"].map(lambda v: "—" if _isnan(v) else f"{(1 - v) * 100:.0f}"),
        "頭数比 A/E": recent.get("h_win_ae", pd.Series(np.nan, index=recent.index)).map(_num),
        "市場比 A/E": recent.get("h_mkt_ae", pd.Series(np.nan, index=recent.index)).map(_num),
        "水分適性": recent.get("h_moist_apt", pd.Series(np.nan, index=recent.index)).map(lambda v: _num(v, 2, sign=True)),
        "最終出走": recent["date"].map(lambda d: f"{d:%Y-%m-%d}"),
        "その着順": recent["finish"].map(lambda v: "—" if _isnan(v) else f"{int(v)}着"),
        "その単勝": recent.get("odds_win", pd.Series(np.nan, index=recent.index)).map(lambda v: _num(v, 1)),
    })
    sub = (f"直近 {active_days} 日以内に出走した馬のレーティング上位 {top} 頭 (最終出走時点)。"
           "近走指数 100 = 毎回1着、0 = 毎回最下位 (時間減衰平均)。"
           "市場比 A/E > 1 はオッズ以上に勝っている馬。水分適性 + は今の水分帯が得意。")
    return f'<p class="sub">{esc(sub)}</p>' + _table(
        tdf, num_cols={"レート", "出走数", "近走指数", "頭数比 A/E", "市場比 A/E", "水分適性", "その単勝"})


# ---------------------------------------------------------------------------
# 8. 本日の予測
# ---------------------------------------------------------------------------

# 判断に使う列を左に (スマホでは左の数列だけが最初に見える)
PRED_COLS = ["馬", "推奨", "AI勝率", "市場確率", "期待値", "単勝", "歪み", "ケリー額", "騎手",
             "プラス要因", "マイナス要因"]


def _is_pick(v) -> bool:
    if _isnan(v):
        return False
    return str(v).strip() in ("◎", "True", "true", "1", "1.0")


def section_today(ctx: dict) -> str:
    pred = ctx["pred"]
    if pred is None:
        raise _Skip("予測 CSV (--pred-csv data/reports/pred_YYYYMMDD.csv) を指定すると、ここに当日の予測と推奨買い目を表示します")
    if pred.empty:
        raise _Skip("予測 CSV が空です")
    t = pred.copy()
    if "推奨" not in t:
        t["推奨"] = ""
    t["_pick"] = t["推奨"].map(_is_pick)
    pct_cols = {"市場確率", "AI勝率", "能力確率", "AI3着内率"}

    def fmt(c, v):
        if c == "推奨":
            return "◎" if _is_pick(v) else ""
        if _isnan(v):
            return "—"
        if c in pct_cols:
            return f"{float(v):.1f}%"
        if c == "歪み":
            return f"{float(v):+.1f}%"
        if c == "期待値":
            return f"{float(v):.2f}"
        if c == "単勝":
            return f"{float(v):.1f}"
        if c == "ケリー額":
            return "" if float(v) == 0 else f"{float(v):,.0f}円"
        if c in ("馬番", "R"):
            return f"{int(float(v))}"
        return str(v)

    picks = t[t["_pick"]]
    out = []
    date_txt = ctx.get("pred_date") or ""
    if len(picks):
        items = []
        for _, r in picks.iterrows():
            rn = f"{fmt('R', r['R'])}R " if "R" in t else ""
            head = f"◎ {rn}{fmt('馬番', r.get('馬番', np.nan))}番 {r.get('馬名', '')}"
            facts = []
            if "単勝" in t:
                facts.append(f"単勝 {fmt('単勝', r['単勝'])}倍")
            if "AI勝率" in t and "市場確率" in t:
                facts.append(f"AI {fmt('AI勝率', r['AI勝率'])} / 市場 {fmt('市場確率', r['市場確率'])}")
            if "期待値" in t:
                facts.append(f"期待値 {fmt('期待値', r['期待値'])}")
            if "ケリー額" in t and fmt("ケリー額", r["ケリー額"]):
                facts.append(f"ケリー {fmt('ケリー額', r['ケリー額'])}")
            plus = r.get("プラス要因", "")
            minus = r.get("マイナス要因", "")
            fac = ""
            if not _isnan(plus) and str(plus):
                fac += f'<div class="fac"><span class="fp">＋</span>{esc(plus)}</div>'
            if not _isnan(minus) and str(minus):
                fac += f'<div class="fac"><span class="fm">－</span>{esc(minus)}</div>'
            items.append(f'<li class="pick-item"><div class="pk-h">{esc(head)}</div>'
                         f'<div class="pk-f">{esc(" · ".join(facts))}</div>{fac}</li>')
        out.append(f'<h3>推奨買い目 {len(picks)} 点{f" ({esc(date_txt)})" if date_txt else ""}</h3>'
                   f'<ul class="picks">{"".join(items)}</ul>')
    else:
        out.append(_note("購入条件を満たす馬はいません (オッズが無い場合も推奨は出ません)"))
    if "馬名" in t:
        t["馬"] = [(f"{fmt('馬番', n)} " if not _isnan(n) else "") + str(m)
                  for n, m in zip(t.get("馬番", pd.Series(np.nan, index=t.index)), t["馬名"])]
    cols = [c for c in PRED_COLS if c in t]
    groups = t.groupby("R", sort=True) if "R" in t else [("", t)]
    races = []
    for rn, g in groups:
        disp = pd.DataFrame({c: [fmt(c, v) for v in g[c]] for c in cols})
        rcls = ["pick" if p else "" for p in g["_pick"]]
        npk = int(g["_pick"].sum())
        title = (f"{fmt('R', rn)}R" if rn != "" else "全馬") + (f" — 推奨 ◎ {npk} 頭" if npk else "")
        races.append(_details(title, _table(disp, row_cls=rcls,
                                            num_cols={"単勝", "市場確率", "AI勝率", "歪み", "期待値", "ケリー額"}),
                              open_=bool(npk)))
    sub = ("歪み = AI勝率 ÷ 市場確率 − 1。期待値 = AI勝率 × 単勝オッズ。◎ の行が推奨 (色付き)。"
           "推奨のあるレースは開いた状態。オッズは締切直前に再計算するのが望ましい。")
    return "".join(out) + f'<p class="sub">{esc(sub)}</p>' + "".join(races)


# ---------------------------------------------------------------------------
# (任意) 頑健性: backtest.robustness_table があれば
# ---------------------------------------------------------------------------


def _robust_label(r: pd.Series, with_pool: bool = True) -> str:
    parts = []
    if with_pool and "pool_yen" in r and not _isnan(r["pool_yen"]):
        v = float(r["pool_yen"])
        parts.append("購入の影響なし" if v == 0 else f"プール {v / 10000:,.0f}万円")
    if "odds_noise" in r and not _isnan(r["odds_noise"]):
        parts.append(f"オッズのズレ {float(r['odds_noise']) * 100:.0f}%")
    if "stake" in r and not _isnan(r["stake"]):
        parts.append(f"1点 {float(r['stake']):,.0f}円")
    if parts:
        return " · ".join(parts)
    return str(r.get("label", ""))


def _roi_rows_svg(rows: list[tuple[str, float, float, float, str]], label: str) -> str:
    """ラベルを上の行に置く横向きの点 + 区間 (長い条件名でもスマホ幅に収まる)."""
    vals = [v for _, roi, lo, hi, _ in rows for v in (roi, lo, hi) if not _isnan(v)]
    a, b, ticks = _nice(max(0.0, min([0.6] + vals)), min(3.0, max([1.4] + vals)), 5)
    W, ml, mr, row, mt, mb = 380, 12, 14, 40, 6, 26
    H = mt + len(rows) * row + mb
    sx = _lin(a, b, ml, W - mr)
    g = []
    for t in ticks:
        g.append(f'<line class="grid" x1="{_f(sx(t))}" x2="{_f(sx(t))}" y1="{mt}" y2="{H - mb}"/>')
        g.append(f'<text class="tk" x="{_f(sx(t))}" y="{H - mb + 16}" text-anchor="middle">{t * 100:g}%</text>')
    if a <= 1 <= b:
        g.append(f'<line class="ref" x1="{_f(sx(1))}" x2="{_f(sx(1))}" y1="{mt}" y2="{H - mb}"/>')
    for i, (lab, roi, lo, hi, tip) in enumerate(rows):
        y0 = mt + i * row
        y = y0 + 27
        g.append(f'<text class="lb halo" x="{ml}" y="{_f(y0 + 13)}" text-anchor="start">{esc(lab)}</text>')
        if _isnan(roi):
            g.append(f'<text class="tk" x="{_f(sx(1))}" y="{_f(y + 4)}" text-anchor="middle">購入なし</text>')
            continue
        cls = "pos" if (not _isnan(lo) and lo > 1) else "neg" if (not _isnan(hi) and hi < 1) else "neu"
        x0 = sx(max(a, lo if not _isnan(lo) else roi))
        x1 = sx(min(b, hi if not _isnan(hi) else roi))
        g.append(f'<line class="whisk {cls}" x1="{_f(x0)}" x2="{_f(x1)}" y1="{_f(y)}" y2="{_f(y)}"/>')
        g.append(_dot(sx(min(max(roi, a), b)), y, cls, tip))
    return _svg(W, H, "".join(g), label)


def _accepts(fn, name: str) -> bool:
    try:
        import inspect

        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _stake_txt(v) -> str:
    if _isnan(v):
        return "—"
    v = float(v)
    if math.isinf(v):
        return "プール額まで黒字"
    if v == 0:
        return "100円でも赤字"
    return f"{v:,.0f}円"


def section_robust(ctx: dict) -> str:
    fn = getattr(B, "robustness_table", None)
    if fn is None:
        raise _Skip("このバージョンには backtest.robustness_table がありません")
    P, rule = ctx["P"], ctx["rule"]
    if P is None:
        raise _Skip(_NO_P)
    if not P["odds_win"].notna().any():
        raise _Skip("オッズが無いため計算できません")
    kw: dict = {"rule": rule}
    dec_fn = getattr(B, "decision_sets", None)
    if dec_fn is not None and _accepts(fn, "decisions"):
        kw["decisions"] = dec_fn(P, rule)  # 締切前オッズの抽選を robustness/breakeven で共有
    R = fn(P, **kw)
    if R is None or len(R) == 0:
        raise _Skip("結果がありません")
    R = R.reset_index(drop=True)
    get = lambda r, c: r[c] if c in r else np.nan  # noqa: E731
    out = []
    # (a) 代表的なプール額での「ズレ × 1点の金額」ごとの回収率
    pools = sorted({float(v) for v in R.get("pool_yen", pd.Series(dtype=float)).dropna() if float(v) > 0})
    if pools:
        pool = 2_500_000.0 if 2_500_000.0 in pools else pools[len(pools) // 2]
        sub_r = R[R["pool_yen"] == pool]
        base = R[(R["pool_yen"] == 0) & (R.get("odds_noise", 0) == 0)].head(1)
        base_rows = [("購入の影響なし · 確定オッズで判断 (従来のバックテスト)", r) for _, r in base.iterrows()]
        chart_src = base_rows + [(_robust_label(r, with_pool=False), r) for _, r in sub_r.iterrows()]
        title = f"プール {pool / 10000:,.0f}万円 の場合の回収率 (90% 区間)"
    else:
        chart_src = [(_robust_label(r), r) for _, r in R.iterrows()]
        title = "条件ごとの回収率 (90% 区間)"
    if len(chart_src) <= 24:
        rows = []
        for lab, r in chart_src:
            tip = (f"回収率 {_pct(get(r, 'roi'))}\n{lab}\n"
                   f"90%区間 {_ci(get(r, 'roi_lo90'), get(r, 'roi_hi90'), d=1)} · {_num(get(r, 'bets'), 0)} 点\n"
                   f"収支 {_yen(get(r, 'profit'))}")
            rows.append((lab, get(r, "roi"), get(r, "roi_lo90"), get(r, "roi_hi90"), tip))
        leg = _legend([("dot", "pos", "区間下限 > 100%"), ("dot", "neu", "区間が 100% をまたぐ"),
                       ("dot", "neg", "区間上限 < 100%")])
        note = ("オッズのズレ = 締切前に見たオッズと確定オッズの差 (σ)。ズレがあると AI 勝率も締切前オッズで計算し直して判断する。"
                "1点の金額が大きいほど自分の購入で払戻オッズが下がる。")
        out.append(_card(title, leg + _roi_rows_svg(rows, title), esc(note)))
    # (b) 損益分岐の賭け金
    be_fn = getattr(B, "breakeven_table", None)
    if be_fn is not None:
        try:
            BE = be_fn(P, **kw)
            if BE is not None and len(BE):
                BE = BE.sort_values(["pool_yen", "odds_noise"]).reset_index(drop=True)
                col = lambda c: BE[c] if c in BE else pd.Series(np.nan, index=BE.index)  # noqa: E731
                tdf = pd.DataFrame({  # スマホで最初に見える左側に結論の列
                    "プール": BE["pool_yen"].map(lambda v: f"{float(v) / 10000:,.0f}万円"),
                    "オッズのズレ": BE["odds_noise"].map(lambda v: _pct(v, 0)),
                    "損益分岐の賭け金": col("breakeven_stake").map(_stake_txt),
                    "最も儲かる賭け金": col("best_stake").map(
                        lambda v: "—" if _isnan(v) else ("なし (赤字)" if float(v) == 0 else f"{float(v):,.0f}円")),
                    "そのときの1点の期待収支": col("best_profit_per_bet").map(_yen),
                    "100円での回収率": col("roi_100").map(_pct),
                })
                rcls = ["pick" if pools and float(v) == pool else "" for v in BE["pool_yen"]]
                out.append(_card("1点あたりいくらまで買えるか", _table(
                    tdf, row_cls=rcls, num_cols=set(tdf.columns) - {"プール", "オッズのズレ"}),
                    esc("損益分岐の賭け金 = 1点の金額をこれ以上にすると回収率が 100% を下回る額 (自分の購入でオッズが下がるため)。"
                        "色付きは上のグラフと同じプール額。"), wide=True))
        except Exception as e:  # noqa: BLE001
            out.append(_note(f"損益分岐の表は作成できませんでした ({type(e).__name__}: {e})"))
    tdf = pd.DataFrame({"条件": [_robust_label(r) for _, r in R.iterrows()]})
    for c, j, f in [("bets", "点数", lambda v: _num(v, 0)), ("hit_rate", "的中率", _pct), ("roi", "回収率", _pct),
                    ("roi_lo90", "下限90", _pct), ("roi_hi90", "上限90", _pct), ("profit", "収支", _yen),
                    ("paid_odds_ratio", "払戻オッズ比", lambda v: _num(v, 3))]:
        if c in R:
            tdf[j] = R[c].map(f)
    rcls = ["pick" if not _isnan(get(r, "roi_lo90")) and get(r, "roi_lo90") > 1 else "" for _, r in R.iterrows()]
    sub = (f"期待値 ≥ {rule.min_ev:g} の均等買いを、自分の購入によるオッズ低下 (プールが小さく 1 点の金額が大きいほど大きい) と"
           "締切前オッズのズレを入れて計算し直した回収率。確定オッズでのバックテストより現実に近い目安。")
    return (f'<p class="sub">{esc(sub)}</p><div class="grid1">{"".join(out)}</div>'
            + _details(f"全条件の表で見る ({len(R)} 行, 色付きは区間下限 > 100%)",
                       _table(tdf, row_cls=rcls, num_cols=set(tdf.columns) - {"条件"})))


# ---------------------------------------------------------------------------
# 組み立て
# ---------------------------------------------------------------------------

CSS = """
:root{color-scheme:light;
--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--ink-3:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--pos:#2a78d6;--neg:#e34948;--neu:#898781;
--good:#0ca30c;--warn:#fab219;--bad:#d03b3b;
--hl:rgba(42,120,214,.08);--pick:#e7eff7;
--banner:#fff3d1;--banner-b:#fab219}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink-2:#c3c2b7;--ink-3:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--pos:#3987e5;--neg:#e66767;--neu:#898781;
--hl:rgba(57,135,229,.12);--pick:#1f2b3a;
--banner:#3a2e0e;--banner-b:#fab219}}
:root[data-theme="dark"]{color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink-2:#c3c2b7;--ink-3:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--pos:#3987e5;--neg:#e66767;--neu:#898781;
--hl:rgba(57,135,229,.12);--pick:#1f2b3a;
--banner:#3a2e0e;--banner-b:#fab219}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.6 system-ui,-apple-system,"Segoe UI","Hiragino Sans","Noto Sans JP",sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:16px}
header h1{font-size:22px;line-height:1.3;margin:4px 0 6px}
.meta{color:var(--ink-2);font-size:13px;margin:0;overflow-wrap:anywhere}
.banner{display:flex;gap:10px;align-items:flex-start;background:var(--banner);border-left:6px solid var(--banner-b);
border-radius:8px;padding:12px 14px;margin:12px 0}
.banner .ic{font-size:22px;line-height:1.2}
.banner strong{display:block;font-size:17px}
.banner p{margin:2px 0 0;font-size:13px;color:var(--ink-2)}
nav ul{display:flex;flex-wrap:wrap;gap:6px;list-style:none;padding:0;margin:12px 0 4px}
nav a{display:inline-block;padding:4px 10px;border:1px solid var(--border);border-radius:999px;
background:var(--surface);color:var(--ink);text-decoration:none;font-size:13px}
nav a:hover,nav a:focus-visible{background:var(--hl)}
.sec{margin:26px 0}
.sec>h2{font-size:19px;margin:0 0 8px;padding-bottom:4px;border-bottom:1px solid var(--grid)}
h3{font-size:15px;margin:0 0 4px}
.sub{color:var(--ink-2);font-size:13px;margin:0 0 8px}
.note{color:var(--ink-2);font-size:13px;background:var(--surface);border:1px dashed var(--axis);
border-radius:8px;padding:8px 12px;margin:8px 0}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px;min-width:0}
.grid1,.grid2{display:grid;gap:12px;grid-template-columns:minmax(0,1fr)}
@media (min-width:860px){.grid2{grid-template-columns:repeat(2,minmax(0,1fr))}}
.grid1 .card{max-width:600px}.grid1 .card.wide{max-width:900px}
.tiles{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(min(100%,230px),1fr))}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px;min-width:0}
.t-label{font-size:13px;color:var(--ink-2)}
.t-value{font-size:30px;font-weight:600;line-height:1.25;margin:2px 0}
.t-sub{font-size:12.5px;color:var(--ink-2);overflow-wrap:anywhere}
.st{display:inline-flex;gap:6px;align-items:baseline;font-size:13px;margin-top:6px}
.st .ic{display:inline-grid;place-items:center;width:18px;height:18px;border-radius:50%;
font-size:12px;font-weight:700;color:#fff;flex:none;transform:translateY(3px)}
.st-good .ic{background:var(--good)}.st-warn .ic{background:var(--warn);color:#0b0b0b}
.st-bad .ic{background:var(--bad)}.st-info .ic{background:var(--ink-3)}
.verdict{margin:0 0 10px}.verdict .st{font-size:14px;margin:0}
svg.chart{display:block;width:100%;height:auto;max-width:520px;margin:0 auto;overflow:visible}
.card.wide svg.chart{max-width:none}
svg.chart.only-wide{display:none}
@media (min-width:700px){svg.chart.only-wide{display:block}svg.chart.only-narrow{display:none}}
svg text{font-family:inherit}
.tk{fill:var(--ink-3);font-size:12px;font-variant-numeric:tabular-nums}
.lb{fill:var(--ink-2);font-size:12.5px}
.vl{fill:var(--ink);font-size:13px;font-weight:600}
.halo{paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.ref{stroke:var(--ink-3);stroke-width:1}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.line.s1{stroke:var(--s1)}
.area.s1{fill:var(--s1);opacity:.1;stroke:none}
.whisk{stroke-width:1.5;stroke-linecap:round}
.whisk.s1{stroke:var(--s1)}.whisk.s2{stroke:var(--s2)}
.whisk.pos{stroke:var(--pos)}.whisk.neg{stroke:var(--neg)}.whisk.neu{stroke:var(--neu)}
.dot{stroke:var(--surface);stroke-width:2}
.dot.s1,.bar.s1{fill:var(--s1)}.dot.s2{fill:var(--s2)}
.dot.pos,.arrow.pos{fill:var(--pos)}.dot.neg,.arrow.neg{fill:var(--neg)}.dot.neu,.arrow.neu{fill:var(--neu)}
.hit{fill:transparent;stroke:none}
.hl{fill:var(--hl)}
.xhair{stroke:var(--ink-3);stroke-width:1;pointer-events:none}
.xdot{pointer-events:none}
.mk{cursor:pointer}.mk:focus{outline:none}.mk:focus-visible{outline:2px solid var(--s1);outline-offset:1px}
.mk:hover .dot,.mk:focus-visible .dot{stroke:var(--ink);stroke-width:1.5}
.mk:hover .bar,.mk:focus-visible .bar{opacity:.8}
svg.mini{display:block}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12.5px;color:var(--ink-2);margin:4px 0 6px}
.lg{display:inline-flex;align-items:center;gap:6px}
.key{display:inline-block;flex:none}
.key.dot{width:10px;height:10px;border-radius:50%}
.key.line{width:16px;height:2px;border-radius:1px}
.key.ref{width:16px;height:1px;background:var(--ink-3)}
.key.s1{background:var(--s1)}.key.s2{background:var(--s2)}
.key.pos{background:var(--pos)}.key.neg{background:var(--neg)}.key.neu{background:var(--neu)}
details{margin:8px 0}
summary{cursor:pointer;color:var(--ink-2);font-size:13px;padding:4px 0}
details[open]>summary{color:var(--ink)}
.tbl{overflow-x:auto;-webkit-overflow-scrolling:touch;max-width:100%;border:1px solid var(--border);
border-radius:8px;background:var(--surface);margin:6px 0}
table{border-collapse:separate;border-spacing:0;font-size:13px;width:100%}
caption{text-align:left;font-weight:600;padding:8px 10px 2px;font-size:13px}
th,td{padding:5px 9px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap;vertical-align:middle}
th{color:var(--ink-2);font-weight:600;font-size:12px;background:var(--surface)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
th:first-child,td:first-child{position:sticky;left:0;background:var(--surface);z-index:1}
tr.pick td{background:var(--pick)}
tr.pick td:first-child{box-shadow:inset 3px 0 0 var(--s1)}
tbody tr:last-child td{border-bottom:none}
.picks{list-style:none;padding:0;margin:6px 0 10px;display:grid;gap:8px;
grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr))}
.pick-item{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--s1);
border-radius:8px;padding:10px 12px;min-width:0}
.pk-h{font-weight:600}
.pk-f{font-size:13px;color:var(--ink-2);overflow-wrap:anywhere}
.fac{font-size:13px;overflow-wrap:anywhere}
.fp,.fm{display:inline-block;width:1.3em;font-weight:700}
.fp{color:var(--good)}.fm{color:var(--bad)}
#tip{position:fixed;z-index:10;pointer-events:none;max-width:min(300px,calc(100vw - 24px));
background:var(--surface);color:var(--ink-2);border:1px solid var(--border);border-radius:8px;
box-shadow:0 4px 16px rgba(0,0,0,.18);padding:8px 10px;font-size:12.5px;line-height:1.45}
#tip strong{display:block;color:var(--ink);font-size:15px}
footer{color:var(--ink-3);font-size:12px;margin:32px 0 8px;border-top:1px solid var(--grid);padding-top:10px}
@media (max-width:520px){.wrap{padding:16px}.t-value{font-size:26px}header h1{font-size:19px}}
@media print{nav,#tip{display:none}details{display:block}.card{break-inside:avoid}}
"""

JS = """
(function(){
var tip=document.getElementById('tip');if(!tip)return;
function place(x,y){var w=tip.offsetWidth,h=tip.offsetHeight,vw=window.innerWidth,vh=window.innerHeight;
var l=x+14,t=y+14;if(l+w>vw-8)l=Math.max(8,x-w-14);if(t+h>vh-8)t=Math.max(8,y-h-14);
tip.style.left=l+'px';tip.style.top=t+'px';}
function show(text,x,y){tip.replaceChildren();var ls=text.split('\\n');var s=document.createElement('strong');
s.textContent=ls[0];tip.appendChild(s);for(var i=1;i<ls.length;i++){var d=document.createElement('div');
d.textContent=ls[i];tip.appendChild(d);}tip.hidden=false;place(x,y);}
function hide(){tip.hidden=true;}
document.addEventListener('pointerover',function(e){var el=e.target.closest&&e.target.closest('[data-tip]');
if(el)show(el.getAttribute('data-tip'),e.clientX,e.clientY);});
document.addEventListener('pointerout',function(e){var el=e.target.closest&&e.target.closest('[data-tip]');
if(el&&!(e.relatedTarget&&el.contains(e.relatedTarget)))hide();});
document.addEventListener('focusin',function(e){var el=e.target.closest&&e.target.closest('[data-tip]');
if(el){var r=el.getBoundingClientRect();show(el.getAttribute('data-tip'),r.right,r.top);}});
document.addEventListener('focusout',hide);
window.addEventListener('scroll',hide,{passive:true});
document.querySelectorAll('svg[data-xhair]').forEach(function(svg){
var pts=JSON.parse(svg.getAttribute('data-xhair'));if(!pts.length)return;
var ln=svg.querySelector('.xhair'),dt=svg.querySelector('.xdot');
function move(e){var r=svg.getBoundingClientRect(),vb=svg.viewBox.baseVal;
var x=(e.clientX-r.left)*vb.width/r.width,lo=0,hi=pts.length-1;
while(hi-lo>1){var m=(lo+hi)>>1;if(pts[m][0]<x)lo=m;else hi=m;}
var p=(Math.abs(pts[lo][0]-x)<=Math.abs(pts[hi][0]-x))?pts[lo]:pts[hi];
ln.setAttribute('x1',p[0]);ln.setAttribute('x2',p[0]);ln.setAttribute('visibility','visible');
dt.setAttribute('cx',p[0]);dt.setAttribute('cy',p[1]);dt.setAttribute('visibility','visible');
show(p[2],e.clientX,e.clientY);}
svg.addEventListener('pointermove',move);svg.addEventListener('pointerdown',move);
svg.addEventListener('pointerleave',function(){ln.setAttribute('visibility','hidden');
dt.setAttribute('visibility','hidden');hide();});});
})();
"""


def _section(sid: str, fn, ctx: dict) -> str:
    try:
        body = fn(ctx)
    except _Skip as e:
        body = _note(f"省略: {e}")
    except Exception as e:  # noqa: BLE001  (1 つの節の失敗でダッシュボード全体を落とさない)
        body = _note(f"省略: この節を作成できませんでした ({type(e).__name__}: {e})")
    return (f'<section id="{sid}" class="sec" aria-labelledby="h-{sid}">'
            f'<h2 id="h-{sid}">{esc(SECTION_TITLES[sid])}</h2>{body}</section>')


def _pred_date_from_name(path: str | Path | None) -> str:
    if not path:
        return ""
    m = re.search(r"(\d{4})(\d{2})(\d{2})", Path(path).name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def build_dashboard(
    df: pd.DataFrame,
    P: pd.DataFrame | None = None,
    pred: pd.DataFrame | None = None,
    *,
    X: pd.DataFrame | None = None,
    rule: B.BetRule | None = None,
    days: int = 365,
    min_rides: int = 50,
    compute_walkforward: bool = True,
    block_months: int = 2,
    pred_date: str = "",
    sources: dict[str, str] | None = None,
    verbose: bool = False,
) -> str:
    """ダッシュボードの HTML 文字列を返す.

    df   : 過去成績 (schema.normalize 済み)。騎手・馬・歪みの集計に使う
    P    : walk_forward の出力 (無ければ compute_walkforward=True のとき計算する)
    pred : predict.format_table の出力 (pred_YYYYMMDD.csv を読んだもの)
    """
    rule = rule or B.BetRule()
    sources = sources or {}
    is_sim = "_p_true" in df.columns or (P is not None and "_p_true" in P.columns)
    p_error = ""
    if X is None and len(df):
        from .features import build_features

        if verbose:
            print(f"特徴量を作成中... ({len(df)} 行)")
        X = build_features(df)
    if P is None and compute_walkforward and X is not None:
        from .model import walk_forward

        if verbose:
            print("ウォークフォワード学習中... (--preds で保存済みの予測を渡すと省略できます)")
        try:
            P = walk_forward(X, block_months=block_months, verbose=verbose)
        except Exception as e:  # noqa: BLE001
            p_error = f"ウォークフォワード予測を作成できませんでした ({e})"
    elif P is None:
        p_error = "ウォークフォワード予測 (--preds) が指定されていません"
    if P is not None:
        P = P.copy()
        P["date"] = pd.to_datetime(P["date"])
        if "odds_win" not in P:
            P["odds_win"] = np.nan
    ctx = {"df": df, "X": X, "P": P, "pred": pred, "rule": rule, "days": days, "min_rides": min_rides,
           "p_error": p_error, "pred_date": pred_date}

    order = [("summary", section_summary), ("calibration", section_calibration),
             ("threshold", section_threshold), ("pnl", section_pnl), ("distortion", section_distortion),
             ("jockey", section_jockey), ("horse", section_horse), ("today", section_today)]
    if getattr(B, "robustness_table", None) is not None:
        order.append(("robust", section_robust))
    sections = []
    for sid, fn in order:
        if verbose:
            print(f"  {SECTION_TITLES[sid]}")
        sections.append(_section(sid, fn, ctx))
    nav = "".join(f'<li><a href="#{sid}">{esc(SECTION_TITLES[sid])}</a></li>' for sid, _ in order)

    meta = []
    if len(df):
        d = pd.to_datetime(df["date"])
        meta.append(f"データ: {d.min():%Y-%m-%d}〜{d.max():%Y-%m-%d} · {df['race_id'].nunique():,} レース · {len(df):,} 頭")
    for k, v in sources.items():
        if v:
            meta.append(f"{k}: {v}")
    meta.append(f"作成: {datetime.now():%Y-%m-%d %H:%M}")
    banner = ""
    if is_sim:
        banner = (f'<div class="banner" role="note"><span class="ic" aria-hidden="true">⚠</span><div>'
                  f'<strong>{esc(SIM_BANNER)}</strong><p>検証用の人工データ (banei_ai.simulate) で作った表示です。'
                  "回収率などの数字は実際のばんえい競馬の収益性について何も保証しません。</p></div></div>")
    foot = ("確定オッズでの検証です。購入時点のオッズとのズレや、自分の購入によるオッズ低下で実際の回収率は下がります。"
            "馬券の購入は自己責任で。")
    title = "ばんえい単勝AI ダッシュボード" + ("（シミュレーション）" if is_sim else "")
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
<h1>{esc(title)}</h1>
<p class="meta">{'<br>'.join(esc(m) for m in meta)}</p>
{banner}
<nav aria-label="目次"><ul>{nav}</ul></nav>
</header>
<main>
{''.join(sections)}
</main>
<footer>{esc(foot)} banei_ai {esc(_version())}</footer>
</div>
<div id="tip" role="tooltip" hidden></div>
<script>{JS}</script>
</body>
</html>
"""


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return ""


def read_pred_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_dashboard(out: str | Path, data: str | Path, preds: str | Path | None = None,
                    pred_csv: str | Path | None = None, **kw) -> Path:
    df = schema.normalize(schema.read_csv(data))
    P = pd.read_pickle(preds) if preds else None
    pred = read_pred_csv(pred_csv) if pred_csv else None
    srcs = {"成績データ": str(data), "予測 (walk-forward)": str(preds) if preds else "",
            "当日予測": str(pred_csv) if pred_csv else ""}
    doc = build_dashboard(df, P, pred, pred_date=_pred_date_from_name(pred_csv), sources=srcs, **kw)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="banei_ai.dashboard", description="結果をひと目で見る HTML ダッシュボード")
    ap.add_argument("--data", default=str(config.INTERIM_DIR / "races.csv"), help="過去成績 CSV")
    ap.add_argument("--preds", default=None,
                    help="walk_forward の予測 pickle (backtest が保存する data/reports/walkforward_preds.pkl)。"
                         "省略時はその場で計算 (数分かかる)")
    ap.add_argument("--pred-csv", default=None, help="当日予測 CSV (data/reports/pred_YYYYMMDD.csv)")
    ap.add_argument("--out", default=str(config.REPORT_DIR / "dashboard.html"))
    ap.add_argument("--min-ev", type=float, default=1.10, help="採用ルールの期待値しきい値")
    ap.add_argument("--days", type=int, default=365, help="騎手表の集計期間 (日)")
    ap.add_argument("--min-rides", type=int, default=50)
    ap.add_argument("--block-months", type=int, default=2)
    ap.add_argument("--no-walkforward", action="store_true",
                    help="--preds が無いときにウォークフォワードを計算しない (予測系の節は省略)")
    a = ap.parse_args(argv)
    if not Path(a.data).exists():
        raise SystemExit(f"{a.data} がありません。先に scrape / import-csv / simulate を実行してください")
    out = write_dashboard(a.out, a.data, a.preds, a.pred_csv, rule=B.BetRule(min_ev=a.min_ev), days=a.days,
                          min_rides=a.min_rides, compute_walkforward=not a.no_walkforward,
                          block_months=a.block_months, verbose=True)
    print(f"ダッシュボード → {out}")


if __name__ == "__main__":
    main()
