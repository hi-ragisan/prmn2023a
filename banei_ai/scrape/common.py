"""HTTP 取得 (キャッシュ・レート制限付き) と、見出し語ベースの表パーサー."""
from __future__ import annotations

import hashlib
import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, schema

UA = "Mozilla/5.0 (personal research; banei-ai)"


class Fetcher:
    def __init__(self, cache_dir: Path | None = None, wait: float = 2.0, encoding: str | None = None):
        import requests

        self.session = requests.Session()
        self.session.headers["User-Agent"] = UA
        self.cache_dir = cache_dir or config.RAW_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.wait = wait
        self.encoding = encoding
        self._last = 0.0

    def _path(self, url: str) -> Path:
        h = hashlib.sha1(url.encode()).hexdigest()[:20]
        return self.cache_dir / f"{h}.html"

    def get(self, url: str, use_cache: bool = True) -> str:
        p = self._path(url)
        if use_cache and p.exists():
            return p.read_text(encoding="utf-8")
        dt = time.time() - self._last
        if dt < self.wait:
            time.sleep(self.wait - dt)
        r = self.session.get(url, timeout=30)
        self._last = time.time()
        r.raise_for_status()
        if self.encoding:
            r.encoding = self.encoding
        elif r.encoding is None or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding
        text = r.text
        p.write_text(text, encoding="utf-8")
        return text


# 見出し語 → 正規列名 (上から順に判定。「馬番」は「枠」より先に見る)
HEADER_RULES: list[tuple[str, list[str]]] = [
    ("finish", ["着順", "着"]),
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


def map_headers(columns) -> dict:
    """DataFrame の列 → 正規列名 の対応を作る."""
    mapping: dict = {}
    used: set[str] = set()
    for col in columns:
        name = re.sub(r"\s+", "", _flatten(col))
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
                mapping[col] = key
                used.add(key)
                break
    return mapping


def find_entry_table(html: str) -> pd.DataFrame | None:
    """ページ内の表のうち「馬名」列を持つ最大の表を、正規列名で返す."""
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return None
    best = None
    for t in tables:
        m = map_headers(t.columns)
        if "horse_name" in m.values() and ("post" in m.values() or "finish" in m.values()):
            if best is None or len(t) > len(best[0]):
                best = (t, m)
    if best is None:
        return None
    t, m = best
    t = t[list(m)].copy()
    t.columns = [m[c] for c in t.columns]
    t = t.loc[:, ~t.columns.duplicated()]
    return t


def norm_name(v) -> str:
    """空白と減量記号 (☆▲△★◇) を除いた名前."""
    s = re.sub(r"\s+", "", str(v))
    return re.sub(r"^[☆▲△★◇]+", "", s)


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


def find_moisture(html: str) -> float:
    """ページ本文から馬場水分 (%) を探す."""
    from bs4 import BeautifulSoup

    text = BeautifulSoup(html, "lxml").get_text(" ")
    for rx in [r"水分\s*[:：]?\s*(\d+(?:\.\d+)?)\s*[%％]",
               r"馬場[^0-9]{0,12}(\d+\.\d)\s*[%％]",
               r"\((\d+\.\d)\s*[%％]\)"]:
        m = re.search(rx, text)
        if m:
            return float(m.group(1))
    return np.nan


def find_race_class(html: str) -> str | None:
    from bs4 import BeautifulSoup

    text = BeautifulSoup(html, "lxml").get_text(" ")
    text = text.translate(str.maketrans("ＡＢＣ０１２３４５６７８９－", "ABC0123456789-"))
    m = re.search(r"\b(A1|A2|B1|B2|B3|B4|C1|C2|オープン|OP)[-\s]?\d*", text)
    if m:
        return m.group(1)
    if "重賞" in text or "記念" in text:
        return "重賞"
    m = re.search(r"([23])歳", text)
    return f"{m.group(1)}歳" if m else None


def clean_entries(t: pd.DataFrame) -> pd.DataFrame:
    """パース直後の表 → 正規スキーマに近い形へ."""
    t = t.copy()
    for c in ("horse_name", "jockey_name", "trainer_name"):
        if c in t:
            t[c] = t[c].map(norm_name)
    t["post"] = t["post"].map(schema.to_float) if "post" in t else np.nan
    raw_fin = t["finish"].astype(str) if "finish" in t else pd.Series("", index=t.index)
    t["finish"] = pd.to_numeric(raw_fin.str.extract(r"^(\d+)")[0], errors="coerce")
    t["status"] = "ok"
    t.loc[raw_fin.str.contains("中止|失格", na=False), "status"] = "dnf"
    t.loc[raw_fin.str.contains("取消|除外", na=False), "status"] = "scratch"
    if "sex_age" in t:
        sa = t["sex_age"].astype(str)
        t["sex"] = sa.str.extract(r"([牡牝セ騸])")[0].replace({"騸": "セ"})
        t["age"] = pd.to_numeric(sa.str.extract(r"(\d+)")[0], errors="coerce")
    if "carried_weight" in t:
        t["carried_weight"] = t["carried_weight"].map(schema.to_float)
    if "horse_weight" in t:
        hw = t["horse_weight"].astype(str)
        if "horse_weight_diff" not in t:
            t["horse_weight_diff"] = pd.to_numeric(
                hw.str.extract(r"\(([+\-−]?\d+)\)")[0].str.replace("−", "-"), errors="coerce")
        else:
            t["horse_weight_diff"] = t["horse_weight_diff"].map(schema.to_float)
        t["horse_weight"] = pd.to_numeric(hw.str.extract(r"(\d{3,4})")[0], errors="coerce")
    if "time" in t:
        t["time_sec"] = t["time"].map(schema.parse_time_to_sec)
    for c in ("odds_win", "popularity"):
        if c in t:
            t[c] = t[c].map(schema.to_float)
    return t
