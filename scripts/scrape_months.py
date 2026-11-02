"""netkeiba から月単位で過去成績を取得し、月ごとの CSV に保存する (途中再開可).

  python scripts/scrape_months.py --from 2026-09 --to 2023-04            # 新しい月から順に
  python scripts/scrape_months.py --from 2026-09 --to 2023-04 --parity 0 # 偶数番目の月だけ (2 本並行用)
  python scripts/scrape_months.py --merge                                # 月別 CSV → data/interim/races.csv

* 取得済みの月 (data/interim/months/YYYY-MM.csv) は飛ばす。HTML は data/raw/ にキャッシュされる。
* 2 秒間隔 (--wait) を守ること。取得データは私的利用に限り、公開・再配布しない (利用規約)。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from banei_ai import config, schema  # noqa: E402
from banei_ai.scrape import netkeiba  # noqa: E402

MONTH_DIR = config.INTERIM_DIR / "months"


def months_desc(start: str, end: str) -> list[pd.Period]:
    a, b = pd.Period(start, "M"), pd.Period(end, "M")
    hi, lo = max(a, b), min(a, b)
    out = []
    m = hi
    while m >= lo:
        out.append(m)
        m -= 1
    return out


def scrape_month(m: pd.Period, fetcher, today: pd.Timestamp) -> pd.DataFrame:
    start = m.start_time
    end = min(m.end_time.normalize(), today - pd.Timedelta(days=1))
    frames = []
    for d in pd.date_range(start, end):
        df = netkeiba.scrape_day(d, fetcher)
        if not df.empty:
            print(f"{d:%Y-%m-%d}: {df['race_id'].nunique()} R", flush=True)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def enrich() -> None:
    """取得済みの月別 CSV に、キャッシュ済み HTML から 1着賞金・出走条件を後付けする (通信なし)."""
    from banei_ai.scrape.common import find_race_meta

    f = netkeiba.Fetcher(encoding=netkeiba.ENCODING_HINT)
    for path in sorted(MONTH_DIR.glob("*.csv")):
        if path.stat().st_size < 10:
            continue
        df = schema.read_csv(path)
        if "prize1" in df and df["prize1"].notna().all():
            continue
        metas = {}
        for rid in df["race_id"].unique():
            p = f._path(netkeiba.result_url(rid))
            if p.exists():
                metas[rid] = find_race_meta(p.read_text(encoding="utf-8"))
        df["prize1"] = df["race_id"].map(lambda r: metas.get(r, {}).get("prize1"))
        df["race_condition"] = df["race_id"].map(lambda r: metas.get(r, {}).get("race_condition"))
        df.to_csv(path, index=False)
        print(f"[enrich] {path.stem}: {len(metas)}/{df['race_id'].nunique()} レース", flush=True)


def merge() -> None:
    files = sorted(MONTH_DIR.glob("*.csv"))
    if not files:
        raise SystemExit("月別 CSV がありません")
    frames = [schema.read_csv(f) for f in files if f.stat().st_size > 10]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["race_id", "post"], keep="last")
    df = schema.normalize(df)
    out = config.INTERIM_DIR / "races.csv"
    df.to_csv(out, index=False)
    print(f"{len(files)} か月 → {len(df)} 行 / {df['race_id'].nunique()} レース "
          f"({df['date'].min():%Y-%m-%d}〜{df['date'].max():%Y-%m-%d}) → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None, help="新しい側の月 YYYY-MM")
    ap.add_argument("--to", dest="end", default=None, help="古い側の月 YYYY-MM")
    ap.add_argument("--parity", type=int, choices=[0, 1], default=None, help="2 本並行時の担当 (0/1)")
    ap.add_argument("--wait", type=float, default=2.0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--enrich", action="store_true", help="キャッシュから 1着賞金などを後付け (通信なし)")
    a = ap.parse_args()
    if a.enrich:
        enrich()
    if a.merge:
        merge()
    if a.enrich or a.merge:
        return
    MONTH_DIR.mkdir(parents=True, exist_ok=True)
    f = netkeiba.Fetcher(encoding=netkeiba.ENCODING_HINT, wait=a.wait)
    today = pd.Timestamp.today().normalize()
    months = months_desc(a.start, a.end)
    if a.parity is not None:
        months = [m for i, m in enumerate(months) if i % 2 == a.parity]
    for m in months:
        out = MONTH_DIR / f"{m}.csv"
        if out.exists():
            print(f"[skip] {m} (取得済み)", flush=True)
            continue
        t0 = time.time()
        try:
            df = scrape_month(m, f, today)
        except Exception as e:  # noqa: BLE001
            # 欠けた月は保存しない (次回の実行でやり直す。取得済みページはキャッシュから読むので速い)
            print(f"[fail] {m}: {type(e).__name__} — 後でやり直します", flush=True)
            continue
        tmp = out.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.rename(out)  # 月の途中で止まっても中途半端なファイルを残さない
        n = df["race_id"].nunique() if not df.empty else 0
        print(f"[done] {m}: {n} レース, {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
