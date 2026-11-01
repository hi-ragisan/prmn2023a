"""keiba.go.jp (地方競馬全国協会 公式) からばんえい (帯広, babaCode=3) を取得.

  出馬表  DebaTable       結果  RaceMarkTable       単勝・複勝オッズ  OddsTanFuku
公式なので当日の出馬表・オッズ取得に向く。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .common import Fetcher, clean_entries, find_entry_table, find_moisture, find_race_class, link_ids

BASE = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo"


def _q(date: pd.Timestamp, race_no: int | None = None) -> str:
    q = f"k_raceDate={date:%Y/%m/%d}&k_babaCode={config.KEIBAGO_BANEI_CODE}"
    if race_no is not None:
        q += f"&k_raceNo={race_no}"
    return q


def race_list_url(date):
    return f"{BASE}/RaceList?{_q(date)}"


def card_url(date, rn):
    return f"{BASE}/DebaTable?{_q(date, rn)}"


def result_url(date, rn):
    return f"{BASE}/RaceMarkTable?{_q(date, rn)}"


def odds_url(date, rn):
    return f"{BASE}/OddsTanFuku?{_q(date, rn)}"


def _ids(t: pd.DataFrame, html: str) -> pd.DataFrame:
    horses = link_ids(html, r"k_lineageLoginCode=(\w+)")
    jockeys = link_ids(html, r"k_riderLicenseNo=(\w+)")
    trainers = link_ids(html, r"k_trainerLicenseNo=(\w+)")
    t["horse_id"] = t["horse_name"].map(horses).fillna(t["horse_name"])
    t["jockey_id"] = t["jockey_name"].map(jockeys).fillna(t["jockey_name"]) if "jockey_name" in t else np.nan
    if "trainer_name" in t:
        t["trainer_id"] = t["trainer_name"].map(trainers).fillna(t["trainer_name"])
    return t


def parse_odds(html: str) -> pd.DataFrame | None:
    """単勝オッズ表 → [post, odds_win] (複勝の下限/上限があれば place_min/place_max)."""
    from io import StringIO

    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return None
    for t in tables:
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        if any("馬番" in c for c in cols) and any("単勝" in c for c in cols):
            t.columns = cols
            post_c = next(c for c in cols if "馬番" in c)
            win_c = next(c for c in cols if "単勝" in c)
            out = pd.DataFrame({
                "post": pd.to_numeric(t[post_c], errors="coerce"),
                "odds_win": pd.to_numeric(t[win_c].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce"),
            })
            place_c = [c for c in cols if "複勝" in c]
            if place_c:
                rng = t[place_c[0]].astype(str).str.extract(r"(\d+\.?\d*)\D+(\d+\.?\d*)")
                out["place_min"] = pd.to_numeric(rng[0], errors="coerce")
                out["place_max"] = pd.to_numeric(rng[1], errors="coerce")
            return out.dropna(subset=["post"])
    return None


def scrape_result(date: pd.Timestamp, rn: int, f: Fetcher, with_odds: bool = True) -> pd.DataFrame | None:
    # 直近 2 日以内は結果が未確定・訂正の可能性があるのでキャッシュしない
    cache = date < pd.Timestamp.today().normalize() - pd.Timedelta(days=2)
    url = result_url(date, rn)
    html = f.get(url, use_cache=cache)
    t = find_entry_table(html)
    if t is None or t.empty:
        f.invalidate(url)
        return None
    t = _ids(clean_entries(t), html)
    t["race_id"] = f"{date:%Y%m%d}{rn:02d}"
    t["date"] = date
    t["race_no"] = rn
    t["moisture"] = find_moisture(html)
    t["race_class"] = find_race_class(html)
    if with_odds and ("odds_win" not in t or t["odds_win"].isna().all()):
        o = parse_odds(f.get(odds_url(date, rn), use_cache=cache))
        if o is not None:
            t = t.drop(columns=["odds_win"], errors="ignore").merge(o, on="post", how="left")
    return t


def scrape_day(date: pd.Timestamp, f: Fetcher | None = None, max_races: int = 12) -> pd.DataFrame:
    f = f or Fetcher()
    out = []
    for rn in range(1, max_races + 1):
        try:
            t = scrape_result(date, rn, f)
        except Exception as e:  # noqa: BLE001
            print(f"  {date:%Y-%m-%d} {rn}R: 取得失敗 {e}")
            break
        if t is None:
            break
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def scrape_range(start: str, end: str, days_of_week=None, wait: float = 2.0) -> pd.DataFrame:
    f = Fetcher(wait=wait)
    frames = []
    for d in pd.date_range(start, end):
        if days_of_week and d.dayofweek not in days_of_week:
            continue
        df = scrape_day(d, f)
        if not df.empty:
            print(f"{d:%Y-%m-%d}: {df['race_id'].nunique()} R")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def scrape_card(date: pd.Timestamp, rn: int, f: Fetcher | None = None) -> pd.DataFrame | None:
    """当日の出馬表 + 現在の単勝オッズ (キャッシュしない)."""
    f = f or Fetcher()
    html = f.get(card_url(date, rn), use_cache=False)
    t = find_entry_table(html)
    if t is None or t.empty:
        return None
    t = _ids(clean_entries(t), html)
    t["race_id"] = f"{date:%Y%m%d}{rn:02d}"
    t["date"] = date
    t["race_no"] = rn
    t["moisture"] = find_moisture(html)
    t["race_class"] = find_race_class(html)
    t["status"] = t["status"].where(t["status"].eq("scratch"), "entry")
    t["finish"] = np.nan
    t["time_sec"] = np.nan
    try:
        o = parse_odds(f.get(odds_url(date, rn), use_cache=False))
    except Exception:  # noqa: BLE001
        o = None
    if o is not None:
        t = t.drop(columns=["odds_win"], errors="ignore").merge(o, on="post", how="left")
    return t
