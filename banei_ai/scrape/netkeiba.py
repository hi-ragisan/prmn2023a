"""nar.netkeiba.com からばんえい (帯広) の結果を取得.

race_id = YYYY + 場コード(65) + MM + DD + RR  (例: 202465123001 = 2024/12/30 帯広 1R)
結果ページの表に 単勝オッズ・人気・タイム・馬体重 がまとまっているので学習データ作りに向く。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .common import Fetcher, clean_entries, find_entry_table, find_moisture, find_race_class, link_ids

BASE = "https://nar.netkeiba.com"


def race_id(date: pd.Timestamp, race_no: int) -> str:
    return f"{date:%Y}{config.NETKEIBA_BANEI_CODE}{date:%m%d}{race_no:02d}"


def result_url(rid: str) -> str:
    return f"{BASE}/race/result.html?race_id={rid}"


def shutuba_url(rid: str) -> str:
    return f"{BASE}/race/shutuba.html?race_id={rid}"


def parse_result(html: str, rid: str, date: pd.Timestamp, race_no: int) -> pd.DataFrame | None:
    t = find_entry_table(html)
    if t is None or t.empty:
        return None
    t = clean_entries(t)
    horses = link_ids(html, r"/horse/(?!ped|result|top|search)(\w+)")
    jockeys = link_ids(html, r"/jockey/(?:result/recent/)?(?!result|profile|search)(\w+)")
    trainers = link_ids(html, r"/trainer/(?:result/recent/)?(?!result|profile|search)(\w+)")
    t["horse_id"] = t["horse_name"].map(horses).fillna(t["horse_name"])
    t["jockey_id"] = t["jockey_name"].map(jockeys).fillna(t["jockey_name"])
    if "trainer_name" in t:
        t["trainer_id"] = t["trainer_name"].map(trainers).fillna(t["trainer_name"])
    t["race_id"] = rid
    t["date"] = date
    t["race_no"] = race_no
    t["moisture"] = find_moisture(html)
    t["race_class"] = find_race_class(html)
    return t


def scrape_day(date: pd.Timestamp, fetcher: Fetcher | None = None, max_races: int = 12) -> pd.DataFrame:
    """1 日分。1R が取れなければ開催なしとみなす."""
    fetcher = fetcher or Fetcher(encoding="euc-jp")
    out = []
    # 直近 2 日以内は結果が未確定・訂正の可能性があるのでキャッシュしない
    recent = date >= pd.Timestamp.today().normalize() - pd.Timedelta(days=2)
    for rn in range(1, max_races + 1):
        rid = race_id(date, rn)
        url = result_url(rid)
        try:
            html = fetcher.get(url, use_cache=not recent)
        except Exception as e:  # noqa: BLE001
            print(f"  {rid}: 取得失敗 {e}")
            break
        df = parse_result(html, rid, date, rn)
        if df is None or df["finish"].notna().sum() == 0:
            fetcher.invalidate(url)
            break
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def scrape_range(start: str, end: str, days_of_week=None, wait: float = 2.0) -> pd.DataFrame:
    """期間内の全日を総当たりで取得 (非開催日は 1R が無いので 1 リクエストで飛ばす)."""
    f = Fetcher(encoding="euc-jp", wait=wait)
    frames = []
    for d in pd.date_range(start, end):
        if days_of_week and d.dayofweek not in days_of_week:
            continue
        df = scrape_day(d, f)
        if not df.empty:
            print(f"{d:%Y-%m-%d}: {df['race_id'].nunique()} R")
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def scrape_card(date: pd.Timestamp, race_no: int, fetcher: Fetcher | None = None) -> pd.DataFrame | None:
    """出馬表 (当日予測用)。オッズは JS 読込のため取れないことが多い → keibago.odds を併用."""
    fetcher = fetcher or Fetcher(encoding="euc-jp")
    rid = race_id(date, race_no)
    html = fetcher.get(shutuba_url(rid), use_cache=False)
    t = parse_result(html, rid, date, race_no)
    if t is not None:
        t["status"] = t["status"].where(t["status"].eq("scratch"), "entry")
        t["finish"] = np.nan
    return t
