"""出走データの正規スキーマ.

スクレイパー・CSV 取り込み・シミュレーターはすべてこの形式の DataFrame を返す。
1 行 = 1 レース × 1 頭。
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

# (列名, 型, 必須か, 説明)
COLUMNS: list[tuple[str, str, bool, str]] = [
    ("race_id", "str", True, "レースID (YYYYMMDD + RR など一意なら何でも可)"),
    ("date", "datetime", True, "開催日"),
    ("race_no", "int", True, "レース番号"),
    ("race_name", "str", False, "レース名"),
    ("race_class", "str", False, "クラス (A1, B2, オープン 等)"),
    ("moisture", "float", False, "馬場水分 (%)"),
    ("post", "int", True, "馬番"),
    ("horse_id", "str", True, "馬ID (無ければ馬名)"),
    ("horse_name", "str", False, "馬名"),
    ("sex", "str", False, "性別 (牡/牝/セ)"),
    ("age", "float", False, "馬齢"),
    ("jockey_id", "str", True, "騎手ID (無ければ騎手名)"),
    ("jockey_name", "str", False, "騎手名"),
    ("trainer_id", "str", False, "調教師ID (無ければ調教師名)"),
    ("carried_weight", "float", False, "負担重量 (kg) — ソリ+騎手の重さ"),
    ("horse_weight", "float", False, "馬体重 (kg)"),
    ("horse_weight_diff", "float", False, "馬体重増減 (kg)"),
    ("odds_win", "float", False, "単勝オッズ (確定)"),
    ("popularity", "float", False, "単勝人気"),
    ("finish", "float", False, "着順 (中止は NaN、status='dnf')"),
    ("time_sec", "float", False, "走破タイム (秒)"),
    ("status", "str", True, "ok / dnf(競走中止) / scratch(取消・除外)"),
]

REQUIRED = [c for c, _, req, _ in COLUMNS if req]
ALL = [c for c, *_ in COLUMNS]


def parse_time_to_sec(v) -> float:
    """'1:45.3' / '1分45秒3' / '105.3' → 105.3 秒."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    if not s or s in {"-", "--", "nan"}:
        return np.nan
    m = re.match(r"^(\d+)[:分](\d+)(?:[.秒](\d+))?$", s)
    if m:
        mi, se, fr = m.group(1), m.group(2), m.group(3) or "0"
        return int(mi) * 60 + int(se) + float("0." + fr)
    try:
        return float(s)
    except ValueError:
        return np.nan


ID_COLS = ["race_id", "horse_id", "jockey_id", "trainer_id"]


def read_csv(path, **kw) -> pd.DataFrame:
    """ID 列 (先頭ゼロあり) を必ず文字列で読む."""
    return pd.read_csv(path, dtype={c: str for c in ID_COLS}, **kw)


def norm_num_text(v) -> str:
    """全角数字・各種マイナス記号 (− U+2212, － U+FF0D) を半角に."""
    s = unicodedata.normalize("NFKC", str(v))
    return s.replace("\u2212", "-").replace("\u2012", "-").replace("\u2013", "-")


def to_float(v) -> float:
    if v is None:
        return np.nan
    s = norm_num_text(v).strip().replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """型を揃え、欠けている任意列を補い、検証する."""
    df = df.copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"必須列がありません: {missing}")
    for col, typ, _, _ in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        if typ == "datetime":
            df[col] = pd.to_datetime(df[col])
        elif typ in ("int", "float"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif typ == "str":
            df[col] = df[col].astype("string")
    if df["trainer_id"].isna().all():
        df["trainer_id"] = "unknown"
    df["jockey_id"] = df["jockey_id"].fillna("unknown")
    df["race_id"] = df["race_id"].astype(str)
    df["status"] = df["status"].fillna("ok")
    df = df[ALL + [c for c in df.columns if c not in ALL]]
    df = df.sort_values(["date", "race_id", "post"]).reset_index(drop=True)
    dup = df.duplicated(["race_id", "post"])
    if dup.any():
        raise ValueError(f"race_id+post の重複が {int(dup.sum())} 行あります")
    return df


def runners_only(df: pd.DataFrame) -> pd.DataFrame:
    """取消・除外を除き、実際に出走した馬だけ残す."""
    out = df[df["status"] != "scratch"].copy()
    out["n_runners"] = out.groupby("race_id")["post"].transform("size")
    return out
