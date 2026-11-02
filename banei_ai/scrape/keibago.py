"""keiba.go.jp (地方競馬全国協会 公式) からばんえい (帯広, babaCode=3) を取得.

  レース一覧 RaceList   出馬表 DebaTable   結果 RaceMarkTable   単勝・複勝オッズ OddsTanFuku
公式なので当日の出馬表・オッズ取得に向く。

実ページ構造の根拠 (docs/scraper_verification.md):
* 文字コードは UTF-8 (<meta charset="UTF-8">)。
* RaceList: 見出し「競走 / 発走時刻 / 変更 / 競走種類 / 競走名・出馬表 / コース / 天候 / 馬場 / 頭数 / …」。
  ばんえいでは「馬場」欄に馬場水分の数値 (例 "1.7") が入る。
* RaceMarkTable (成績表): 見出しが <th> ではなく <tr class="dbitem"><td>着<br>順</td>… で、
  その前に「４R 成績表」というタイトル行がある。馬体重の増減は「差」という別の列。
  騎手セルは「沖静男(金沢)」のように所属付き。
  ⚠️ 単勝オッズ列は過去の年の分が空欄 (当年分のみ) との報告あり → 過去データは人気のみ。
* DebaTable (出馬表): 1 頭が <tr class="tBorder"> から始まる 5 行のブロック。
  見出しは「競走馬」(colspan=3) で「馬名」という語が無く、過去 5 走の欄 (着順「取消」などを含む) が並ぶ。
  → 見出し語方式では読めないので専用パーサー parse_deba_table を使う。
* OddsTanFuku: 複勝オッズは colspan=2 の見出しの下に「1.1-」「1.2」と 2 セルに分かれる。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import config, schema
from . import common
from .common import (attach_ids, clean_entries, find_entry_table, find_moisture,
                     find_race_class, find_race_name, link_ids, norm_name)

BASE = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo"

HORSE_ID_RX = r"k_lineageLoginCode=(\w+)"
JOCKEY_ID_RX = r"k_riderLicenseNo=(\w+)"
TRAINER_ID_RX = r"k_trainerLicenseNo=(\w+)"


class Fetcher(common.Fetcher):
    """keiba.go.jp 用. 文字コードが宣言されていない場合は UTF-8 とみなす."""

    def __init__(self, cache_dir=None, wait: float = 2.0, encoding: str | None = "utf-8"):
        super().__init__(cache_dir=cache_dir, wait=wait, encoding=encoding)


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
    horses = link_ids(html, HORSE_ID_RX)
    jockeys = link_ids(html, JOCKEY_ID_RX)
    trainers = link_ids(html, TRAINER_ID_RX)
    t = attach_ids(t, "horse_name", "horse_id", horses)
    t = attach_ids(t, "jockey_name", "jockey_id", jockeys)
    if "trainer_name" in t:
        t = attach_ids(t, "trainer_name", "trainer_id", trainers)
    return t


def _text(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ")).strip() if el is not None else ""


def _num(s) -> float:
    return schema.to_float(s) if s is not None else np.nan


# ---------------- RaceList ----------------

_RACE_LIST_COLS = {"競走": "race", "発走時刻": "start_time", "競走名": "race_name",
                   "コース": "course", "天候": "weather", "馬場": "baba", "頭数": "n_entries"}
# 見出しが読めなかったときの既定位置 (実ページの列順)
_RACE_LIST_DEFAULT_POS = {"race": 0, "start_time": 1, "race_name": 4, "course": 5,
                          "weather": 6, "baba": 7, "n_entries": 8}


def parse_race_list(html: str) -> pd.DataFrame | None:
    """RaceList → [race_no, race_name, race_class, moisture, weather, start_time, n_entries].

    レース一覧の表 (見出しに「競走」と「馬場」がある) が無ければ None、
    表はあるがレース行が無ければ空の DataFrame を返す。
    """
    soup = common._soup(html)
    tables = []  # (table, 見出し位置)
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            names = [re.sub(r"\s+", "", th.get_text()) for th in tr.find_all("th", recursive=False)]
            if "競走" in names and any("馬場" in n for n in names):
                pos = {}
                for i, n in enumerate(names):
                    for word, key in _RACE_LIST_COLS.items():
                        if key not in pos and (n == word or (word != "競走" and n.startswith(word))):
                            pos[key] = i
                tables.append((table, pos))
                break
    found_table = bool(tables)
    if not tables:
        # 見出しが読めない場合は「1R」で始まり 8 セル以上ある行を既定の列位置で読む
        tables = [(table, {}) for table in soup.find_all("table")]
    rows = []
    for table, pos in tables:
        for tr in table.find_all("tr"):
            tds = tr.find_all("td", recursive=False)
            if not tds:
                continue
            cells = [_text(td) for td in tds]
            m = re.fullmatch(r"(\d{1,2})\s*R", cells[0])
            if not m or (not pos and len(cells) < 8):
                continue
            p = {**_RACE_LIST_DEFAULT_POS, **pos}

            def cell(key):
                i = p.get(key)
                return cells[i] if i is not None and i < len(cells) else ""

            baba = schema.norm_num_text(cell("baba"))
            mm = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%?\s*", baba)
            moist = float(mm.group(1)) if mm else np.nan
            name = cell("race_name")
            rows.append({"race_no": int(m.group(1)), "race_name": name or None,
                         "race_class": find_race_class(f"<h3>{name}</h3>") if name else None,
                         "moisture": moist, "weather": cell("weather") or None,
                         "start_time": cell("start_time") or None,
                         "n_entries": _num(cell("n_entries"))})
            found_table = True
    if not found_table:
        return None
    if not rows:
        return pd.DataFrame(columns=["race_no", "race_name", "race_class", "moisture", "weather",
                                     "start_time", "n_entries"])
    return pd.DataFrame(rows).drop_duplicates("race_no").sort_values("race_no").reset_index(drop=True)


# ---------------- DebaTable ----------------

_SEX_AGE_RX = re.compile(r"^(牡|牝|セン?|騸)\s*(\d{1,2})$")
_WEIGHT_RECORD_RX = re.compile(r"(\d{2,4}(?:\.\d)?)\s*\d+-\d+-\d+-\d+")
_HORSE_WEIGHT_RX = re.compile(r"(\d{3,4})\s*(?:\(\s*([+\-]?\d+)\s*\))?")


def parse_deba_table(html: str) -> pd.DataFrame | None:
    """keiba.go.jp 出馬表 (DebaTable) → 1 行 1 頭.

    ブロック構造 (実ページで確認):
      行1 tr.tBorder: td.courseNum(枠, 同枠 2 頭目は省略) td.horseNum(馬番) a.horseName
                      a.jockeyName(+span.jockeyarea 所属) td.odds_weight(「8.1 (3人気)」)
                      td.result(着別成績の入れ子表) 過去5走
      行2: 性齢 / 毛色 / 生年月日 / 「負担重量　騎手成績(0-0-0-0)」 / 過去5走
      行3: 父 / 調教師リンク / td.odds_weight(馬体重「1061(+26)」) / 過去5走
      行4: 母 / 馬主 / 過去5走
      行5: 母父 / 生産牧場 / td.info(変更情報: 取消・除外) / 過去5走
    過去走の欄は読まない (「取消」や過去の馬場水分が入っているため)。
    """
    soup = common._soup(html)
    blocks = soup.select("tr.tBorder")
    if not blocks:
        return None
    rows = []
    last_frame = np.nan
    for tr in blocks:
        post = _num(_text(tr.select_one("td.horseNum")))
        if np.isnan(post):
            continue
        ftd = tr.select_one("td.courseNum")
        frame = _num(_text(ftd)) if ftd is not None else last_frame
        last_frame = frame
        rec = {"frame": frame, "post": post}

        a = tr.select_one("a.horseName") or tr.find("a", href=re.compile(HORSE_ID_RX))
        rec["horse_name"] = norm_name(_text(a)) if a is not None else None
        m = re.search(HORSE_ID_RX, a.get("href", "")) if a is not None else None
        rec["horse_id"] = m.group(1) if m else rec["horse_name"]

        j = tr.select_one("a.jockeyName") or tr.find("a", href=re.compile(JOCKEY_ID_RX))
        if j is not None:
            for sp in j.select("span.jockeyarea"):
                sp.extract()
        rec["jockey_name"] = norm_name(_text(j)) if j is not None else None
        m = re.search(JOCKEY_ID_RX, j.get("href", "")) if j is not None else None
        rec["jockey_id"] = m.group(1) if m else rec["jockey_name"]

        # 1 行目の odds_weight = 単勝オッズと人気
        ow = tr.select_one("td.odds_weight")
        ow_text = schema.norm_num_text(_text(ow))
        mo = re.match(r"\s*(\d+(?:\.\d+)?)", ow_text)
        rec["odds_win"] = float(mo.group(1)) if mo else np.nan
        mp = re.search(r"\(\s*(\d+)\s*人気\s*\)", ow_text)
        rec["popularity"] = float(mp.group(1)) if mp else np.nan
        status_text = ow_text

        # 次の tBorder までの補助行
        subs = []
        sib = tr.find_next_sibling("tr")
        while sib is not None and "tBorder" not in (sib.get("class") or []) and len(subs) < 6:
            subs.append(sib)
            sib = sib.find_next_sibling("tr")
        rec.update({"sex_age": None, "carried_weight": np.nan, "trainer_name": None,
                    "trainer_id": None, "horse_weight": np.nan, "horse_weight_diff": np.nan})
        for sub in subs:
            cells = sub.find_all("td", recursive=False)
            # 末尾 5 セルは過去 5 走の欄 (「1:37.5 1-1-1-1 39.5」などを負担重量と誤認しないよう読まない)
            own = cells[:-5] if len(cells) > 5 else cells
            for td in own:
                txt = schema.norm_num_text(_text(td))
                if rec["sex_age"] is None and _SEX_AGE_RX.match(txt.replace(" ", "")):
                    rec["sex_age"] = txt.replace(" ", "")
                if np.isnan(rec["carried_weight"]):
                    mw = _WEIGHT_RECORD_RX.search(txt)
                    if mw:
                        rec["carried_weight"] = float(mw.group(1))
            ta = sub.find("a", href=re.compile(TRAINER_ID_RX))
            if ta is not None and rec["trainer_name"] is None:
                rec["trainer_name"] = norm_name(_text(ta))
                rec["trainer_id"] = re.search(TRAINER_ID_RX, ta["href"]).group(1)
            hw = sub.select_one("td.odds_weight")
            if hw is not None and np.isnan(rec["horse_weight"]):
                hwt = schema.norm_num_text(_text(hw))
                mh = _HORSE_WEIGHT_RX.search(hwt)
                if mh:
                    rec["horse_weight"] = float(mh.group(1))
                    if mh.group(2):
                        rec["horse_weight_diff"] = float(mh.group(2))
                status_text += hwt
            for info in sub.select("td.info"):
                status_text += _text(info)
        rec["status"] = "scratch" if re.search(common._SCRATCH_RX, status_text) else "entry"
        if rec["status"] == "scratch":
            rec["odds_win"] = np.nan
        sa = rec.pop("sex_age") or ""
        ms = re.match(r"(牡|牝|セン?|騸)(\d+)", sa)
        rec["sex"] = {"セン": "セ", "騸": "セ"}.get(ms.group(1), ms.group(1)) if ms else np.nan
        rec["age"] = float(ms.group(2)) if ms else np.nan
        if rec["trainer_id"] is None:
            rec["trainer_id"] = rec["trainer_name"]
        rows.append(rec)
    if not rows:
        return None
    t = pd.DataFrame(rows).drop_duplicates("post").reset_index(drop=True)
    t["finish"] = np.nan
    t["time_sec"] = np.nan
    return t


# ---------------- OddsTanFuku ----------------

def _first_num(v) -> float:
    m = re.search(r"\d+(?:\.\d+)?", schema.norm_num_text(v))
    return float(m.group(0)) if m else np.nan


def parse_odds(html: str) -> pd.DataFrame | None:
    """単勝・複勝オッズ表 → [post, odds_win, place_min, place_max].

    複勝は「1.1-」「1.2」と 2 セルに分かれる形 (実ページ) と、1 セルに「1.1-1.2」の形の両方に対応。
    """
    for t in common.read_tables(html):
        cols = [common.header_name(c) for c in t.columns]
        post_i = next((i for i, c in enumerate(cols) if "馬番" in c), None)
        win_i = next((i for i, c in enumerate(cols) if "単勝" in c), None)
        if post_i is None or win_i is None:
            continue
        out = pd.DataFrame({
            "post": pd.to_numeric(t.iloc[:, post_i].map(schema.to_float), errors="coerce"),
            "odds_win": t.iloc[:, win_i].map(_first_num),
        })
        place_i = [i for i, c in enumerate(cols) if "複勝" in c]
        if len(place_i) >= 2:
            out["place_min"] = t.iloc[:, place_i[0]].map(_first_num)
            out["place_max"] = t.iloc[:, place_i[1]].map(_first_num)
        elif len(place_i) == 1:
            nums = t.iloc[:, place_i[0]].astype(str).map(
                lambda s: re.findall(r"\d+(?:\.\d+)?", schema.norm_num_text(s)))
            out["place_min"] = nums.map(lambda x: float(x[0]) if x else np.nan)
            out["place_max"] = nums.map(lambda x: float(x[1]) if len(x) > 1 else (float(x[0]) if x else np.nan))
        for c in ("odds_win", "place_min", "place_max"):
            if c in out:
                out.loc[out[c] <= 0, c] = np.nan  # 0 は発売対象外 (取消など)
        out = out.dropna(subset=["post"]).drop_duplicates("post")
        out["post"] = out["post"].astype(float)
        return out.reset_index(drop=True)
    return None


# ---------------- 成績表 ----------------

# 2026 年版の成績表は各セルに 1 文字のクラスが付く (他者のパーサーより: a=着順 c=馬番 d=馬名
# j=馬体重「1061(+26)」 k=タイム)。見出し語で読めなかったときだけ、この対応で読む。
_CLASSED_RESULT_COLS = {"a": "finish", "c": "post", "d": "horse_name", "j": "horse_weight", "k": "time"}


def _parse_classed_result(html: str) -> pd.DataFrame | None:
    soup = common._soup(html)
    rows = []
    for tr in soup.find_all("tr"):
        cells = {}
        for td in tr.find_all("td", recursive=False):
            for c in td.get("class") or []:
                if c in _CLASSED_RESULT_COLS and _CLASSED_RESULT_COLS[c] not in cells:
                    cells[_CLASSED_RESULT_COLS[c]] = _text(td)
        if not {"finish", "post", "horse_name"} <= set(cells):
            continue
        ja = tr.find("a", href=re.compile(JOCKEY_ID_RX))
        ta = tr.find("a", href=re.compile(TRAINER_ID_RX))
        cells["jockey_name"] = _text(ja) if ja is not None else None
        cells["trainer_name"] = _text(ta) if ta is not None else None
        rows.append(cells)
    return pd.DataFrame(rows) if rows else None


def parse_result_page(html: str, date: pd.Timestamp, rn: int) -> pd.DataFrame | None:
    """RaceMarkTable の HTML → 正規スキーマに近い DataFrame (ネットに出ない純粋関数)."""
    t = find_entry_table(html)
    if t is None or t.empty:
        t = _parse_classed_result(html)
    if t is None or t.empty:
        return None
    t = _ids(clean_entries(t), html)
    t["race_id"] = f"{date:%Y%m%d}{rn:02d}"
    t["date"] = date
    t["race_no"] = rn
    t["moisture"] = find_moisture(html)
    t["race_class"] = find_race_class(html)
    t["race_name"] = find_race_name(html)
    return t


def _fill_from_meta(t: pd.DataFrame, meta: dict | None) -> pd.DataFrame:
    """RaceList の情報 (馬場水分・レース名・クラス) で欠けている値を補う."""
    if not meta:
        return t
    for col in ("moisture", "race_name", "race_class"):
        v = meta.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if col not in t or t[col].isna().all():
            t[col] = v
    return t


def scrape_result(date: pd.Timestamp, rn: int, f: common.Fetcher, with_odds: bool = True,
                  meta: dict | None = None) -> pd.DataFrame | None:
    # 直近 2 日以内は結果が未確定・訂正の可能性があるのでキャッシュしない
    cache = date < pd.Timestamp.today().normalize() - pd.Timedelta(days=2)
    url = result_url(date, rn)
    html = f.get(url, use_cache=cache)
    t = parse_result_page(html, date, rn)
    if t is None or t["finish"].notna().sum() == 0:  # 表が無い・結果未確定
        f.invalidate(url)
        return None
    t = _fill_from_meta(t, meta)
    # 過去の年は成績表の単勝オッズが空欄。OddsTanFuku も遡れるのは半年あまり
    if with_odds and ("odds_win" not in t or t["odds_win"].isna().all()):
        try:
            o = parse_odds(f.get(odds_url(date, rn), use_cache=cache))
        except Exception:  # noqa: BLE001
            o = None
        if o is not None and o["odds_win"].notna().any():
            t = t.drop(columns=[c for c in ("odds_win", "place_min", "place_max") if c in t]).merge(
                o, on="post", how="left")
    return t


def race_list(date: pd.Timestamp, f: common.Fetcher, use_cache: bool = True) -> pd.DataFrame | None:
    url = race_list_url(date)
    html = f.get(url, use_cache=use_cache)
    rl = parse_race_list(html)
    if rl is None or rl.empty:
        f.invalidate(url)
    return rl


def scrape_day(date: pd.Timestamp, f: common.Fetcher | None = None, max_races: int = 12) -> pd.DataFrame:
    """1 日分. RaceList でレース番号と馬場水分を得てから各レースの成績表を取る.

    RaceList が読めない (表が無い) 場合は従来どおり 1R から順に成績表を試す。
    """
    f = f or Fetcher()
    cache = date < pd.Timestamp.today().normalize() - pd.Timedelta(days=2)
    try:
        rl = race_list(date, f, use_cache=cache)
    except Exception as e:  # noqa: BLE001
        print(f"  {date:%Y-%m-%d} RaceList 取得失敗 {e}")
        rl = None
    if rl is not None and rl.empty:
        return pd.DataFrame()  # レース一覧の表はあるがレースが無い = 開催なし
    metas = {int(r.race_no): r._asdict() for r in rl.itertuples()} if rl is not None else {}
    numbers = sorted(metas) if metas else range(1, max_races + 1)
    out = []
    for rn in numbers:
        try:
            t = scrape_result(date, rn, f, meta=metas.get(rn))
        except Exception as e:  # noqa: BLE001
            print(f"  {date:%Y-%m-%d} {rn}R: 取得失敗 {e}")
            if metas:
                continue
            break
        if t is None:
            if metas:
                continue
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


def parse_card_page(html: str, date: pd.Timestamp, rn: int) -> pd.DataFrame | None:
    """DebaTable の HTML → 出馬表 (純粋関数). 専用パーサーが読めなければ見出し語方式で試す."""
    t = parse_deba_table(html)
    if t is None or t.empty:
        t = find_entry_table(html)
        if t is None or t.empty:
            return None
        t = _ids(clean_entries(t), html)
        t["status"] = t["status"].where(t["status"].eq("scratch"), "entry")
        t["finish"] = np.nan
        t["time_sec"] = np.nan
    t["race_id"] = f"{date:%Y%m%d}{rn:02d}"
    t["date"] = date
    t["race_no"] = rn
    t["moisture"] = find_moisture(html)
    t["race_class"] = find_race_class(html)
    t["race_name"] = find_race_name(html)
    return t


def scrape_card(date: pd.Timestamp, rn: int, f: common.Fetcher | None = None) -> pd.DataFrame | None:
    """当日の出馬表 + 現在の単勝オッズ (キャッシュしない)."""
    f = f or Fetcher()
    html = f.get(card_url(date, rn), use_cache=False)
    t = parse_card_page(html, date, rn)
    if t is None:
        return None
    if t["moisture"].isna().all():
        try:
            rl = parse_race_list(f.get(race_list_url(date), use_cache=False))
            if rl is not None and (rl["race_no"] == rn).any():
                t = _fill_from_meta(t, rl[rl["race_no"] == rn].iloc[0].to_dict())
        except Exception:  # noqa: BLE001
            pass
    try:
        o = parse_odds(f.get(odds_url(date, rn), use_cache=False))
    except Exception:  # noqa: BLE001
        o = None
    if o is not None and o["odds_win"].notna().any():
        t = t.drop(columns=[c for c in ("odds_win", "place_min", "place_max") if c in t]).merge(
            o, on="post", how="left")
    return t
