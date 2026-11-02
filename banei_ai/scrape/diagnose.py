"""スクレイパーの診断ツール.

実ページ (URL) または保存した HTML ファイルを読み、パーサーが
  * どの表を選んだか・見出し→列の対応・対応しなかった見出し
  * パース結果 (先頭 12 行)
  * 馬場水分・クラス・レース名 (どこから読んだか)
  * 馬・騎手・調教師 ID の対応率 (リンクから ID が取れた割合)
を表示する。ずれていたら common.HEADER_RULES / find_moisture などを直す手がかりにする。

使い方:
  python -m banei_ai.scrape.diagnose "https://nar.netkeiba.com/race/result.html?race_id=202665110101"
  python -m banei_ai.scrape.diagnose saved.html --source keibago
  python -m banei_ai.scrape.diagnose saved.html --source keibago --kind deba

ブラウザで「ページを保存 (HTML のみ)」したファイルでも、curl -o で保存したファイルでもよい。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from . import common, keibago, netkeiba

KINDS = {
    "netkeiba": ["result", "card"],
    "keibago": ["result", "deba", "odds", "racelist"],
}
SHOW_COLS = ["post", "frame", "finish", "status", "horse_name", "horse_id", "sex", "age",
             "carried_weight", "jockey_name", "jockey_id", "trainer_name", "trainer_id",
             "horse_weight", "horse_weight_diff", "time_sec", "odds_win", "popularity",
             "place_min", "place_max", "race_no", "race_name", "race_class", "moisture", "weather"]


def detect_source(target: str, html: str) -> str:
    if "keiba.go.jp" in target or "KeibaWeb" in html[:200000]:
        return "keibago"
    return "netkeiba"


def detect_kind(source: str, target: str, html: str) -> str:
    t = target.lower()
    if source == "netkeiba":
        return "card" if ("shutuba" in t or "HorseList" in html) else "result"
    if "racelist" in t:
        return "racelist"
    if "debatable" in t:
        return "deba"
    if "oddstanfuku" in t:
        return "odds"
    if "racemarktable" in t:
        return "result"
    if 'class="tBorder"' in html or "class='tBorder'" in html:
        return "deba"
    if "odd_popular_table" in html or ("複勝" in html and "単勝" in html and "成績表" not in html):
        return "odds"
    if keibago.parse_race_list(html) is not None and "成績表" not in html:
        return "racelist"
    return "result"


def _date_race(target: str, source: str) -> tuple[pd.Timestamp, int]:
    """URL から日付・レース番号を推定 (分からなければ今日・1R)."""
    q = parse_qs(urlparse(target).query) if target.startswith("http") else {}
    try:
        if source == "netkeiba" and "race_id" in q:
            rid = q["race_id"][0]
            return pd.Timestamp(f"{rid[:4]}-{rid[6:8]}-{rid[8:10]}"), int(rid[10:12])
        if source == "keibago" and "k_raceDate" in q:
            return (pd.Timestamp(q["k_raceDate"][0].replace("/", "-")),
                    int(q.get("k_raceNo", ["1"])[0]))
    except (ValueError, IndexError):
        pass
    m = re.search(r"(20\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", target)  # race_id を含むファイル名
    if m:
        try:
            return pd.Timestamp(f"{m.group(1)}-{m.group(3)}-{m.group(4)}"), int(m.group(5))
        except ValueError:
            pass
    return pd.Timestamp.today().normalize(), 1


def load(target: str, source: str | None) -> tuple[str, str | None, str]:
    """(html, 文字コード, 取得元)。URL は既存の Fetcher で取得 (キャッシュしない)."""
    if re.match(r"https?://", target):
        src = source or ("keibago" if "keiba.go.jp" in target else "netkeiba")
        f = keibago.Fetcher(wait=0) if src == "keibago" else common.Fetcher(
            wait=0, encoding=netkeiba.ENCODING_HINT)
        html = f.get(target, use_cache=False)
        return html, f.last_encoding, src
    raw = Path(target).read_bytes()
    html, enc = common.decode_html(raw)
    return html, enc, source or detect_source(target, html)


def _print_table_overview(html: str, out) -> dict | None:
    infos = common.inspect_tables(html)
    out(f"\n[表の一覧] pandas.read_html で {len(infos)} 個")
    for x in infos:
        mapped = sum(k is not None for k in x["keys"])
        hdr = "pandas" if x["header_row"] is None else f"{x['header_row']} 行目"
        mark = "○" if x["usable"] else " "
        out(f"  {mark} #{x['index']:<3} {x['shape'][0]:>3}行×{x['shape'][1]:<3}列  見出し:{hdr:<8} "
            f"対応した見出し {mapped} 個  馬番が数値の行 {x['n_rows']}")
    cands = [x for x in infos if x["usable"] and x["n_rows"] > 0]
    if not cands:
        out("  → 「馬名」と「馬番/着順」の見出しを持つ表が見つかりません")
        return None
    best = max(cands, key=lambda x: x["n_rows"])
    out(f"\n[採用した表] #{best['index']} (馬番が数値の行が最多)")
    out("  見出し → 正規列名:")
    for h, k in zip(best["headers"], best["keys"]):
        if k:
            out(f"    {h!r:<22} → {k}")
    unmapped = [h for h, k in zip(best["headers"], best["keys"]) if not k and h and h != "nan"]
    out(f"  対応しなかった見出し: {unmapped if unmapped else 'なし'}")
    missing = [k for k in ("finish", "post", "horse_name", "jockey_name", "carried_weight",
                           "odds_win", "popularity", "time", "horse_weight")
               if k not in best["keys"]]
    out(f"  見つからなかった正規列: {missing if missing else 'なし'}")
    return best


def _rate(t: pd.DataFrame, id_col: str, name_col: str) -> str:
    r = common.id_hit_rate(t, id_col, name_col)
    return "—" if np.isnan(r) else f"{r:.0%}"


def diagnose(html: str, source: str, kind: str | None = None, target: str = "",
             encoding: str | None = None, out=print) -> dict:
    """診断結果を表示し、要約を dict で返す (テスト用)."""
    kind = kind or detect_kind(source, target, html)
    date, rn = _date_race(target, source)
    title = common._soup(html).title
    out("=" * 70)
    out(f"取得元: {source} / ページ種別: {kind} / 文字コード: {encoding or '不明'} / {len(html):,} 文字")
    out(f"<title>: {title.get_text(strip=True) if title else '(なし)'}")
    n_bad = html.count("�")
    warns = []
    if n_bad >= 20:
        warns.append(f"置換文字 (U+FFFD) が {n_bad} 個 → 文字コードの判定が誤っている可能性")

    t: pd.DataFrame | None = None
    if kind == "racelist":
        t = keibago.parse_race_list(html)
        out("\n[RaceList] レース一覧の表: " + ("見つからない" if t is None else f"{len(t)} レース"))
    elif kind == "odds":
        t = keibago.parse_odds(html)
        out("\n[OddsTanFuku] 「馬番」「単勝」の見出しを持つ表: " + ("見つからない" if t is None else f"{len(t)} 頭"))
    elif kind == "deba":
        n_blocks = len(common._soup(html).select("tr.tBorder"))
        out(f"\n[DebaTable] 専用パーサー: <tr class=\"tBorder\"> のブロック {n_blocks} 個")
        t = keibago.parse_card_page(html, date, rn)
        if n_blocks == 0:
            _print_table_overview(html, out)
    else:
        _print_table_overview(html, out)
        if source == "netkeiba":
            rid = netkeiba.race_id(date, rn)
            t = (netkeiba.parse_card(html, date, rn) if kind == "card"
                 else netkeiba.parse_result(html, rid, date, rn))
        else:
            t = keibago.parse_result_page(html, date, rn)

    moist, where = common.find_moisture_detail(html)
    out("\n[レース情報]")
    out(f"  馬場水分: {moist if not np.isnan(moist) else '見つからない'}"
        + (f"   (読んだ場所: {where})" if where else ""))
    out(f"  クラス: {common.find_race_class(html)}   レース名: {common.find_race_name(html)}")
    if np.isnan(moist) and kind != "odds":
        warns.append("馬場水分が見つからない (keibago は RaceList の「馬場」欄で補完される)")

    summary = {"source": source, "kind": kind, "encoding": encoding, "moisture": moist,
               "rows": 0 if t is None else len(t), "table": t}
    if t is None or t.empty:
        out("\n[パース結果] 0 行")
        warns.append("行が 1 つも取れていない")
    else:
        cols = [c for c in SHOW_COLS if c in t.columns]
        with pd.option_context("display.width", 250, "display.max_columns", 40,
                               "display.max_colwidth", 16):
            out(f"\n[パース結果] {len(t)} 行 (先頭 12 行)")
            out(t[cols].head(12).to_string(index=False))
        if "horse_id" in t:
            rates = {c: _rate(t, f"{c}_id", f"{c}_name") for c in ("horse", "jockey", "trainer")}
            out(f"\n[ID 対応率] (リンクから ID が取れた割合)  馬 {rates['horse']} / "
                f"騎手 {rates['jockey']} / 調教師 {rates['trainer']}")
            summary["id_rates"] = rates
            if common.id_hit_rate(t, "horse_id", "horse_name") < 0.9:
                warns.append("馬 ID の多くがリンクから取れていない (馬名で代用される)")
        if kind == "result" and "finish" in t and t["finish"].notna().sum() == 0:
            warns.append("着順が 1 つも数値になっていない (結果未確定か、着順列の誤認)")
        if kind in ("result", "odds") and ("odds_win" not in t or t["odds_win"].isna().all()):
            warns.append("単勝オッズが空 (keiba.go.jp 成績表は過去の年のオッズが空欄との報告あり)")
        if "status" in t:
            out(f"  status の内訳: {t['status'].value_counts().to_dict()}")
    out("\n[警告]" + ("".join(f"\n  - {w}" for w in warns) if warns else " なし"))
    summary["warnings"] = warns
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m banei_ai.scrape.diagnose",
                                 description="スクレイパーの診断 (URL または保存した HTML)")
    ap.add_argument("target", help="URL または保存した HTML ファイルのパス")
    ap.add_argument("--source", choices=["netkeiba", "keibago"], default=None,
                    help="取得元 (省略時は URL・内容から推定)")
    ap.add_argument("--kind", choices=sorted({k for v in KINDS.values() for k in v}), default=None,
                    help="ページ種別 (省略時は推定). netkeiba: result/card, "
                         "keibago: result/deba/odds/racelist")
    a = ap.parse_args(argv)
    try:
        html, enc, source = load(a.target, a.source)
    except Exception as e:  # noqa: BLE001
        print(f"読み込みに失敗しました: {e}")
        if re.match(r"https?://", a.target):
            print("(ブラウザでページを開き「名前を付けて保存」した HTML を渡すこともできます)")
        return 2
    if a.kind and a.kind not in KINDS[source]:
        print(f"--kind {a.kind} は {source} では使えません ({KINDS[source]})")
        return 2
    s = diagnose(html, source, a.kind, target=a.target, encoding=enc)
    return 0 if s["rows"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
