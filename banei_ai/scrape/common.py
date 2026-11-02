"""HTTP 取得 (キャッシュ・レート制限付き) と、見出し語ベースの表パーサー.

実ページ構造の根拠は docs/scraper_verification.md を参照 (開発環境から実サイトには
アクセスできなかったため、公開されている実ページの保存物・他者のパーサーから確認した)。
"""
from __future__ import annotations

import codecs
import hashlib
import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, schema

UA = "Mozilla/5.0 (personal research; banei-ai)"

_META_CHARSET = re.compile(rb"<meta[^>]{0,200}?charset\s*=\s*[\"']?\s*([A-Za-z0-9_\-]+)", re.I)
# キャッシュ済み HTML がこの数以上の置換文字 (U+FFFD) を含むなら文字化けとみなして取り直す
_MOJIBAKE_LIMIT = 20


def _codec_name(enc: str | None) -> str | None:
    if not enc:
        return None
    try:
        return codecs.lookup(enc.strip()).name
    except LookupError:
        return None


def decode_html(content: bytes, header_encoding: str | None = None,
                hint: str | None = None) -> tuple[str, str]:
    """バイト列 → 文字列. (文字列, 採用した文字コード) を返す.

    優先順: BOM > HTTP ヘッダーの charset > <meta charset> > hint > utf-8 > euc-jp > cp932。
    各候補で「厳密に」デコードを試し、失敗したら次へ進む (UTF-8 のページを EUC-JP と
    決め打ちして全文字化けする事故を防ぐ)。nar.netkeiba.com は以前 EUC-JP だったが、
    現在は UTF-8 で配信されているという報告があるため、決め打ちはしない。
    どれも厳密には通らなければ、最初の候補で置換デコードする。
    """
    cands: list[str] = []
    if content.startswith(b"\xef\xbb\xbf"):
        cands.append("utf-8-sig")
    hdr = _codec_name(header_encoding)
    if hdr and hdr not in ("latin-1", "iso8859-1", "cp1252"):  # requests の既定値は無視
        cands.append(hdr)
    m = _META_CHARSET.search(content[:4096])
    if m:
        meta = _codec_name(m.group(1).decode("ascii", "ignore"))
        if meta:
            cands.append(meta)
    for enc in (hint, "utf-8", "euc-jp", "euc_jis_2004", "cp932"):
        e = _codec_name(enc)
        if e:
            cands.append(e)
    seen: list[str] = []
    for enc in cands:
        if enc in seen:
            continue
        seen.append(enc)
        try:
            return content.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return content.decode(seen[0], errors="replace"), seen[0]


class Fetcher:
    """キャッシュ・レート制限付きの GET.

    encoding はページが文字コードを宣言していない場合のヒント (決め打ちではない)。
    """

    def __init__(self, cache_dir: Path | None = None, wait: float = 2.0, encoding: str | None = None):
        import requests

        self.session = requests.Session()
        self.session.headers["User-Agent"] = UA
        self.cache_dir = cache_dir or config.RAW_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.wait = wait
        self.encoding = encoding
        self._last = 0.0
        self.last_encoding: str | None = None

    def _path(self, url: str) -> Path:
        h = hashlib.sha1(url.encode()).hexdigest()[:20]
        return self.cache_dir / f"{h}.html"

    def invalidate(self, url: str) -> None:
        """パースできなかったページ (開催なし・結果未確定) のキャッシュを消す."""
        p = self._path(url)
        if p.exists():
            p.unlink()

    # 一時的な失敗 (接続リセット・タイムアウト・429/5xx) は間隔を空けて再試行する。
    # 403/404 などの拒否・不存在は再試行しない (サイトの意思に反するアクセスを繰り返さない)。
    RETRY_STATUS = {429, 500, 502, 503, 504}
    RETRY_WAITS = (5.0, 15.0, 45.0)

    def _get_with_retry(self, url: str):
        import requests

        for attempt in range(len(self.RETRY_WAITS) + 1):
            dt = time.time() - self._last
            if dt < self.wait:
                time.sleep(self.wait - dt)
            try:
                r = self.session.get(url, timeout=30)
                self._last = time.time()
                if getattr(r, "status_code", 200) in self.RETRY_STATUS and attempt < len(self.RETRY_WAITS):
                    raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
                r.raise_for_status()
                return r
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                self._last = time.time()
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status is not None and status not in self.RETRY_STATUS:
                    raise
                if attempt >= len(self.RETRY_WAITS):
                    raise
                w = self.RETRY_WAITS[attempt]
                print(f"  再試行 {attempt + 1}/{len(self.RETRY_WAITS)} ({w:.0f}秒後): {type(e).__name__}", flush=True)
                time.sleep(w)

    def get(self, url: str, use_cache: bool = True) -> str:
        p = self._path(url)
        if use_cache and p.exists():
            text = p.read_text(encoding="utf-8")
            if text.count("�") < _MOJIBAKE_LIMIT:
                return text
            # 以前の版で文字コードを決め打ちして化けたキャッシュは取り直す
        r = self._get_with_retry(url)
        ctype = r.headers.get("Content-Type", "")
        header_enc = r.encoding if "charset" in ctype.lower() else None
        text, self.last_encoding = decode_html(r.content, header_enc, self.encoding)
        if use_cache:
            p.write_text(text, encoding="utf-8")
        return text


# 見出し語 → 正規列名 (上から順に判定。「馬番」は「枠」より先に見る)
HEADER_RULES: list[tuple[str, list[str]]] = [
    ("finish", ["着順", "着", "順位"]),
    ("post", ["馬番"]),
    ("frame", ["枠"]),
    ("horse_name", ["馬名"]),
    ("sex_age", ["性齢", "性令", "性別"]),
    ("carried_weight", ["負担重量", "斤量", "重量"]),
    ("jockey_name", ["騎手"]),
    ("trainer_name", ["調教師", "厩舎"]),
    ("horse_weight", ["馬体重"]),
    ("horse_weight_diff", ["増減"]),
    ("time", ["タイム", "走破"]),
    ("odds_win", ["単勝オッズ", "単勝", "オッズ"]),
    ("popularity", ["人気"]),
]
# 見出しとみなす最大文字数 (ページ全体を包む外枠の表のセルを見出しと誤認しないため)
_MAX_HEADER_LEN = 25


def _flatten(col) -> str:
    if isinstance(col, tuple):
        parts = []
        for c in col:
            c = str(c)
            if c.startswith("Unnamed") or c in parts:
                continue
            parts.append(c)
        return " ".join(parts)
    return str(col)


def header_name(col) -> str:
    """見出しセル → 空白を除いた文字列 (pandas が重複見出しに付ける '.1' も除く)."""
    s = re.sub(r"\s+", "", _flatten(col))
    return re.sub(r"\.\d+$", "", s)


def _map_names(names: list[str]) -> list[str | None]:
    """見出し文字列のリスト → 位置ごとの正規列名 (なければ None)."""
    out: list[str | None] = [None] * len(names)
    used: set[str] = set()
    for i, name in enumerate(names):
        if not name or name == "nan" or len(name) > _MAX_HEADER_LEN:
            continue
        # keiba.go.jp の成績表は馬体重の増減が「差」という単独の列
        if name == "差" and "horse_weight_diff" not in used:
            out[i] = "horse_weight_diff"
            used.add("horse_weight_diff")
            continue
        for key, words in HEADER_RULES:
            if key in used:
                continue
            if key == "finish" and name not in ("着順", "着", "順位") and "着順" not in name:
                continue
            if key == "popularity" and "人気" not in name:
                continue
            if any(w in name for w in words):
                # 「単勝人気」は人気、「単勝オッズ」はオッズ
                if key == "odds_win" and "人気" in name:
                    continue
                out[i] = key
                used.add(key)
                break
    return out


def map_headers(columns) -> dict:
    """DataFrame の列 → 正規列名 の対応を作る."""
    cols = list(columns)
    keys = _map_names([header_name(c) for c in cols])
    return {c: k for c, k in zip(cols, keys) if k is not None}


def _usable(keys) -> bool:
    ks = {k for k in keys if k}
    return "horse_name" in ks and bool({"post", "finish"} & ks)


def read_tables(html: str) -> list[pd.DataFrame]:
    """ページ内の全ての表. 表が無ければ空リスト.

    flavor を lxml に固定する (既定では表が無いとき bs4+html5lib に切り替わり、
    html5lib が入っていないと ValueError ではなく ImportError で落ちる = 非開催日で停止していた)。
    """
    try:
        return pd.read_html(StringIO(html), flavor="lxml")
    except (ValueError, IndexError, ImportError):
        return []


def inspect_tables(html: str, tables: list[pd.DataFrame] | None = None) -> list[dict]:
    """ページ内の各表について、見出し行の位置と 見出し→正規列名 の対応を調べる.

    見出しが <th> でなく <td> で書かれた表 (keiba.go.jp の成績表は
    「４R 成績表」というタイトル行の次に <tr class="dbitem"><td>着<br>順</td>... が来る)
    では pandas が見出しを認識しないので、先頭 6 行から見出し行を探す。
    """
    tables = read_tables(html) if tables is None else tables
    out = []
    for i, t in enumerate(tables):
        names = [header_name(c) for c in t.columns]
        keys = _map_names(names)
        header_row = None  # None = pandas が認識した見出しを使用
        if not _usable(keys):
            for r in range(min(6, len(t))):
                cand = ["" if pd.isna(v) else header_name(v) for v in t.iloc[r]]
                ck = _map_names(cand)
                if _usable(ck):
                    names, keys, header_row = cand, ck, r
                    break
        body = t.iloc[header_row + 1:] if header_row is not None else t
        usable = _usable(keys)
        n_rows = 0
        if usable:
            pos = keys.index("post") if "post" in keys else keys.index("finish")
            n_rows = int(body.iloc[:, pos].map(schema.to_float).notna().sum())
        out.append({"index": i, "shape": t.shape, "header_row": header_row, "headers": names,
                    "keys": keys, "usable": usable, "n_rows": n_rows, "body": body})
    return out


def find_entry_table(html: str) -> pd.DataFrame | None:
    """ページ内の表のうち「馬名」と「馬番」(または着順) の見出しを持ち、
    馬番が数値の行が最も多い表を、正規列名で返す."""
    infos = [x for x in inspect_tables(html) if x["usable"] and x["n_rows"] > 0]
    if not infos:
        return None
    best = max(infos, key=lambda x: x["n_rows"])
    pos = [i for i, k in enumerate(best["keys"]) if k]
    t = best["body"].iloc[:, pos].copy()
    t.columns = [best["keys"][i] for i in pos]
    t = t.loc[:, ~t.columns.duplicated()]
    return t.reset_index(drop=True)


_MARKS = "☆▲△★◇◆□■▽▼"


def norm_name(v) -> str:
    """空白・減量記号 (☆▲△★◇ 等)・末尾の所属 ((帯広) / （帯広）) を除いた名前."""
    s = re.sub(r"\s+", "", str(v))
    s = re.sub(f"^[{_MARKS}]+", "", s)
    return re.sub(r"[（(][^（()）]{1,6}[)）]$", "", s)


def link_ids(html: str, pattern: str) -> dict[str, str]:
    """<a href="...{pattern}ID...">名前</a> から 名前→ID の辞書を作る."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out: dict[str, str] = {}
    rx = re.compile(pattern)
    for a in soup.find_all("a", href=True):
        m = rx.search(a["href"])
        if m:
            name = norm_name(a.get_text())
            if name and name not in out:
                out[name] = m.group(1)
    return out


def attach_ids(t: pd.DataFrame, name_col: str, id_col: str, mapping: dict[str, str]) -> pd.DataFrame:
    """名前→ID 対応で ID を付ける.

    セルに父・母名などが一緒に入っていて完全一致しない場合は、リンク文字列がセルの先頭に
    一致するもの (最長一致)、次に末尾に一致するもの (netkeiba の厩舎セルは「帯広」などの
    所属が名前の前に付く) を採用し、名前もリンク文字列に置き換える。
    見つからなければ名前を ID にする。
    """
    if name_col not in t:
        t[id_col] = np.nan
        return t
    names = sorted(mapping, key=len, reverse=True)
    ids, fixed = [], []
    for v in t[name_col].map(str):
        if v in mapping:
            ids.append(mapping[v])
            fixed.append(v)
            continue
        hit = next((n for n in names if v.startswith(n)), None)
        if hit is None:
            hit = next((n for n in names if len(n) >= 2 and v.endswith(n)), None)
        ids.append(mapping[hit] if hit else v)
        fixed.append(hit or v)
    t[name_col] = fixed
    t[id_col] = ids
    return t


def id_hit_rate(t: pd.DataFrame, id_col: str, name_col: str) -> float:
    """ID 列のうち、リンクから ID が取れた (名前のままでない) 割合."""
    if id_col not in t or name_col not in t or len(t) == 0:
        return float("nan")
    ok = t[id_col].map(str) != t[name_col].map(str)
    return float(ok.mean())


# ---- レース情報 (馬場水分・クラス・レース名) ----

# レース名・条件が書かれていそうな要素 (ページ全体だとナビの他レース名や、
# 出馬表の各馬の過去走欄を拾うため、まずここを見る)
#   netkeiba: .RaceName / .RaceData01 (「18:55発走 / 直200m / 天候:晴 / 水分量:1.6」)
#   keiba.go.jp 出馬表・オッズ: section.raceTitle (h3 = レース名, ul.dataArea = 「天候：曇　馬場：…」)
#   keiba.go.jp 成績表: span.plus1bold02 = レース名
_INFO_SELECTORS = (".RaceData01, .RaceData02, .RaceList_Item02, .raceTitle .dataArea, "
                   ".raceTitle, .racedata, .race_data")
_HEADER_SELECTORS = (".RaceName, .plus1bold02, .raceTitle h3, .RaceData02, .RaceList_Item02, "
                     ".raceTitle, .race_title, h1, h2, h3, h4, title")

_MOIST_HEADER_RX = [
    # netkeiba ばんえい: 「水分量:1.6」(% なし)。他サイト: 「水分 3.2%」
    re.compile(r"水分量?\s*:?\s*(\d+(?:\.\d+)?)\s*%?"),
    # keiba.go.jp: ばんえいは「馬場」に良/重ではなく水分の数値が入る
    re.compile(r"馬場\s*:\s*(\d+(?:\.\d+)?)\s*%?"),
    re.compile(r"馬場[^0-9/]{0,12}?(\d+\.\d)\s*%"),
    re.compile(r"\((\d+\.\d)\s*%\)"),
]
# ページ全体で探すときは「水分」「馬場:」の直後の数値だけ (過去走欄の誤検出を避ける)
_MOIST_PAGE_RX = [re.compile(r"水分量?\s*:\s*(\d+(?:\.\d+)?)"),
                  re.compile(r"水分量?\s*(\d+(?:\.\d+)?)\s*%"),
                  re.compile(r"馬場\s*:\s*(\d+(?:\.\d+)?)\s*%?")]
MOISTURE_RANGE = (0.0, 20.0)


def _soup(html: str):
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "lxml")


def _header_texts(soup) -> list[tuple[str, str]]:
    """(どこから取ったか, テキスト) のリスト。レース情報が書かれていそうな短い要素."""
    out = []
    for el in soup.select(_INFO_SELECTORS):
        out.append((f"{el.name}.{'.'.join(el.get('class') or [])}", el.get_text(" ")))
    # 「天候」を含む文字列の親要素 (keiba.go.jp 成績表のようにクラスの無い表セルに書かれる場合)
    for s in soup.find_all(string=re.compile("天候")):
        el = s.parent
        for _ in range(3):
            if el is None:
                break
            txt = el.get_text(" ")
            if len(txt) > 400:
                break
            out.append((f"天候を含む<{el.name}>", txt))
            el = el.parent
    return out


def find_moisture_detail(html: str) -> tuple[float, str | None]:
    """馬場水分 (%) と、それを見つけた場所."""
    soup = _soup(html)
    lo, hi = MOISTURE_RANGE
    for where, text in _header_texts(soup):
        text = schema.norm_num_text(text)
        for rx in _MOIST_HEADER_RX:
            for m in rx.finditer(text):
                v = float(m.group(1))
                if lo <= v <= hi:
                    return v, f"{where}: {m.group(0).strip()}"
    text = schema.norm_num_text(soup.get_text(" "))
    for rx in _MOIST_PAGE_RX:
        for m in rx.finditer(text):
            v = float(m.group(1))
            if lo <= v <= hi:
                return v, f"ページ全体: {m.group(0).strip()}"
    return np.nan, None


def find_moisture(html: str) -> float:
    """ページ本文から馬場水分 (%) を探す (レース情報欄を優先)."""
    return find_moisture_detail(html)[0]


_CLASS_RX = re.compile(r"(?<![A-Za-z0-9])(A1|A2|B1|B2|B3|B4|C1|C2|OP)(?![0-9])|(オープン)")


def find_race_class(html: str) -> str | None:
    soup = _soup(html)
    texts = [el.get_text(" ") for el in soup.select(_HEADER_SELECTORS)] + [soup.get_text(" ")]
    for text in texts:
        text = schema.norm_num_text(text)
        m = _CLASS_RX.search(text)
        if m:
            return m.group(1) or "オープン"
        if "重賞" in text or "記念" in text:
            return "重賞"
        m = re.search(r"([23])歳", text)
        if m:
            return f"{m.group(1)}歳"
    return None


def find_race_name(html: str) -> str | None:
    """レース名 (netkeiba .RaceName / keiba.go.jp 出馬表 h3 / 成績表 .plus1bold02)."""
    soup = _soup(html)
    for sel in (".RaceName", ".raceTitle h3", ".plus1bold02"):
        el = soup.select_one(sel)
        if el is not None:
            s = re.sub(r"\s+", " ", el.get_text(" ")).strip()
            if s:
                return s
    return None


def find_race_meta(html: str) -> dict:
    """レースの付帯情報 (netkeiba .RaceData02): 1着本賞金 (万円)・出走条件.

    netkeiba のばんえい成績ページには A1/B2 などのクラス名が無いが、1着賞金がクラスの高さをよく表す
    (例: 下級 30万円 / 中級 34万円 / 上級 65万円 …)。特徴量では log(1着賞金) をクラスの指標に使う。
    """
    soup = _soup(html)
    el = soup.select_one(".RaceData02")
    text = re.sub(r"\s+", " ", el.get_text(" ")).strip() if el is not None else ""
    out: dict = {"prize1": np.nan, "race_condition": None}
    m = re.search(r"本賞金\s*[:：]\s*([\d.]+)", text)
    if m:
        out["prize1"] = float(m.group(1))
    m = re.search(r"((?:混合|牝馬|牡馬)?\S*[2-9２-９]歳\S*)", text)
    if m:
        out["race_condition"] = schema.norm_num_text(m.group(1))
    return out


_DNF_RX = "中止|失格"
_SCRATCH_RX = "取消|除外|停止"


def clean_entries(t: pd.DataFrame) -> pd.DataFrame:
    """パース直後の表 → 正規スキーマに近い形へ."""
    t = t.copy()
    for c in ("horse_name", "jockey_name", "trainer_name"):
        if c in t:
            t[c] = t[c].map(norm_name)
    t["post"] = t["post"].map(schema.to_float) if "post" in t else np.nan
    # 注記行・見出しの繰り返し行などを除く
    keep = t["post"].notna()
    if "horse_name" in t:
        keep &= t["horse_name"].notna() & (t["horse_name"] != "nan") & (t["horse_name"] != "")
    t = t[keep].copy()
    raw_fin = t["finish"].astype(str) if "finish" in t else pd.Series("", index=t.index)
    # 着順列が無い表 (出馬表) では、オッズ・馬体重などの欄に「取消」「除外」と出る
    other = [c for c in t.columns if c not in ("horse_name", "jockey_name", "trainer_name",
                                                "post", "frame", "finish")]
    raw_row = (t[other].map(lambda v: "" if pd.isna(v) else str(v)).agg("".join, axis=1)
               if other else pd.Series("", index=t.index))
    t["finish"] = pd.to_numeric(raw_fin.str.extract(r"^\s*(\d+)")[0], errors="coerce")
    if "status" not in t:
        t["status"] = "ok"
        t.loc[raw_fin.str.contains(_DNF_RX, na=False), "status"] = "dnf"
        t.loc[raw_fin.str.contains(_SCRATCH_RX, na=False), "status"] = "scratch"
        t.loc[t["finish"].isna() & ~raw_fin.str.contains(_DNF_RX, na=False)
              & raw_row.str.contains(_SCRATCH_RX, na=False), "status"] = "scratch"
    if "sex_age" in t:
        sa = t["sex_age"].astype(str)
        t["sex"] = sa.str.extract(r"([牡牝セ騸])")[0].replace({"騸": "セ"})
        t["age"] = pd.to_numeric(sa.str.extract(r"(\d+)")[0], errors="coerce")
    if "carried_weight" in t:
        t["carried_weight"] = t["carried_weight"].map(schema.to_float)
    if "horse_weight" in t:
        hw = t["horse_weight"].map(schema.norm_num_text)
        if "horse_weight_diff" not in t:
            t["horse_weight_diff"] = pd.to_numeric(
                hw.str.extract(r"\(\s*([+\-]?\d+)\s*\)")[0], errors="coerce")
        else:
            t["horse_weight_diff"] = t["horse_weight_diff"].map(schema.to_float)
        t["horse_weight"] = pd.to_numeric(hw.str.extract(r"(\d{3,4})")[0], errors="coerce")
    if "time" in t:
        t["time_sec"] = t["time"].map(schema.parse_time_to_sec)
    for c in ("odds_win", "popularity"):
        if c in t:
            t[c] = t[c].map(schema.to_float)
    if "odds_win" in t:
        t.loc[t["odds_win"] <= 0, "odds_win"] = np.nan
    return t
