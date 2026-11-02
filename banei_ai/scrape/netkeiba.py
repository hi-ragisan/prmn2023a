"""nar.netkeiba.com からばんえい (帯広) の結果を取得.

race_id = YYYY + 場コード(65 = 帯広(ば)) + MM + DD + RR  (例: 202465123001 = 2024/12/30 帯広 1R)
結果ページの表に 単勝オッズ・人気・タイム・馬体重 がまとまっているので学習データ作りに向く。

実ページ構造の根拠 (docs/scraper_verification.md):
* 文字コード: 以前は EUC-JP だったが、現在は UTF-8 との報告が複数ある → 宣言から自動判定。
* レース情報欄 (.RaceData01): 「18:55発走 / 直200m / 天候:晴 / 水分量:1.6」(% なし)。
* 結果表: データ行の <td class="Result_Num"><div class="Rank">1</div>、<td class="Horse_Info">…
* 出馬表 (shutuba): 行は <tr class="HorseList">、列は 枠/馬番/印/馬名/性齢/斤量/騎手/厩舎/馬体重/オッズ/人気。
  オッズは JavaScript で後から入るため HTML には入っていないことが多い → keiba.go.jp のオッズを併用。
* ばんえいの馬 ID は "B201900123" のように英字で始まることがある (数値と決めつけない)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .common import (Fetcher, attach_ids, clean_entries, find_entry_table, find_moisture,
                     find_race_class, find_race_meta, find_race_name, link_ids)

BASE = "https://nar.netkeiba.com"

HORSE_ID_RX = r"/horse/(?!ped|result|top|search)(\w+)"
JOCKEY_ID_RX = r"/jockey/(?:result/recent/)?(?!result|profile|search)(\w+)"
TRAINER_ID_RX = r"/trainer/(?:result/recent/)?(?!result|profile|search)(\w+)"
# ページが文字コードを宣言していない場合のヒント (宣言があればそちらを優先)
ENCODING_HINT = "utf-8"


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
    t = attach_ids(t, "horse_name", "horse_id", link_ids(html, HORSE_ID_RX))
    t = attach_ids(t, "jockey_name", "jockey_id", link_ids(html, JOCKEY_ID_RX))
    if "trainer_name" in t:
        t = attach_ids(t, "trainer_name", "trainer_id", link_ids(html, TRAINER_ID_RX))
    t["race_id"] = rid
    t["date"] = date
    t["race_no"] = race_no
    t["moisture"] = find_moisture(html)
    t["race_class"] = find_race_class(html)
    t["race_name"] = find_race_name(html)
    meta = find_race_meta(html)
    t["prize1"] = meta["prize1"]
    t["race_condition"] = meta["race_condition"]
    return t


def _fill_moisture_from_keibago(df: pd.DataFrame, date: pd.Timestamp, wait: float) -> pd.DataFrame:
    """netkeiba から馬場水分が読めなかった日は、keiba.go.jp のレース一覧 (1 日 1 リクエスト) で補う.

    馬場水分はレース単位の値なので、取得元を混ぜても馬・騎手 ID の体系には影響しない。
    """
    if df.empty or df["moisture"].notna().all():
        return df
    try:
        from . import keibago

        cache = date < pd.Timestamp.today().normalize() - pd.Timedelta(days=2)
        rl = keibago.race_list(date, keibago.Fetcher(wait=wait), use_cache=cache)
    except Exception as e:  # noqa: BLE001
        print(f"  {date:%Y-%m-%d}: 馬場水分の補完に失敗 {e}")
        return df
    if rl is None or rl.empty:
        return df
    m = rl.set_index("race_no")["moisture"]
    fill = df["race_no"].map(m)
    df["moisture"] = df["moisture"].fillna(fill)
    return df


def scrape_day(date: pd.Timestamp, fetcher: Fetcher | None = None, max_races: int = 12,
               fill_moisture: bool = True) -> pd.DataFrame:
    """1 日分。1R が取れなければ開催なしとみなす."""
    fetcher = fetcher or Fetcher(encoding=ENCODING_HINT)
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
    if not out:
        return pd.DataFrame()
    day = pd.concat(out, ignore_index=True)
    if fill_moisture:
        day = _fill_moisture_from_keibago(day, date, fetcher.wait)
    return day


def scrape_range(start: str, end: str, days_of_week=None, wait: float = 2.0) -> pd.DataFrame:
    """期間内の全日を総当たりで取得 (非開催日は 1R が無いので 1 リクエストで飛ばす)."""
    f = Fetcher(encoding=ENCODING_HINT, wait=wait)
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


def parse_card(html: str, date: pd.Timestamp, race_no: int) -> pd.DataFrame | None:
    """出馬表 HTML → DataFrame (純粋関数)。取消・除外は status='scratch'、他は 'entry'."""
    t = parse_result(html, race_id(date, race_no), date, race_no)
    if t is not None:
        t["status"] = t["status"].where(t["status"].eq("scratch"), "entry")
        t["finish"] = np.nan
    return t


def scrape_card(date: pd.Timestamp, race_no: int, fetcher: Fetcher | None = None) -> pd.DataFrame | None:
    """出馬表 (当日予測用)。オッズは JS 読込のため取れないことが多い → keibago.odds を併用."""
    fetcher = fetcher or Fetcher(encoding=ENCODING_HINT)
    rid = race_id(date, race_no)
    html = fetcher.get(shutuba_url(rid), use_cache=False)
    return parse_card(html, date, race_no)
