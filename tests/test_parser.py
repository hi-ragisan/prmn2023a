"""見出し語ベースのパーサーのテスト.

注: 下の HTML は実サイトの取得物ではなく、典型的な成績表の形を手で模したもの。
実ページでの確認は `cli scrape --check` で行うこと。
"""
import pandas as pd
import pytest

from banei_ai.scrape import netkeiba
from banei_ai.scrape.common import map_headers

HTML = """
<html><body>
<div class="RaceName">ばんえい特別 B2-1</div>
<div class="RaceData01">14:50発走 / 直200m / 天候:晴 / 馬場:重 水分 3.2%</div>
<table class="RaceTable01">
<tr><th>着順</th><th>枠</th><th>馬番</th><th>馬名</th><th>性齢</th><th>斤量</th><th>騎手</th>
<th>タイム</th><th>着差</th><th>人気</th><th>単勝オッズ</th><th>厩舎</th><th>馬体重(増減)</th></tr>
<tr><td>1</td><td>3</td><td>3</td><td><a href="https://db.netkeiba.com/horse/2019100001">アオノキセキ</a></td>
<td>牡5</td><td>690</td><td><a href="/jockey/result/recent/05001/">鈴木恵</a></td><td>1:52.4</td><td></td>
<td>2</td><td>3.4</td><td><a href="/trainer/result/recent/01001/">坂本東</a></td><td>1050(+6)</td></tr>
<tr><td>2</td><td>1</td><td>1</td><td><a href="https://db.netkeiba.com/horse/2018100002">ベニスズラン</a></td>
<td>牝6</td><td>670</td><td><a href="/jockey/result/recent/05002/">☆金田利</a></td><td>1:55.0</td><td>2.6</td>
<td>1</td><td>2.1</td><td><a href="/trainer/result/recent/01002/">松井浩</a></td><td>980(-4)</td></tr>
<tr><td>中止</td><td>2</td><td>2</td><td><a href="https://db.netkeiba.com/horse/2017100003">クロガネ</a></td>
<td>セ7</td><td>700</td><td><a href="/jockey/result/recent/05003/">西謙</a></td><td></td><td></td>
<td>3</td><td>8.8</td><td><a href="/trainer/result/recent/01003/">金田勇</a></td><td>1010(0)</td></tr>
</table></body></html>
"""


def test_map_headers():
    m = map_headers(["着順", "枠", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム", "人気", "単勝オッズ", "厩舎", "馬体重(増減)"])
    assert m["馬番"] == "post" and m["枠"] == "frame"
    assert m["斤量"] == "carried_weight"
    assert m["単勝オッズ"] == "odds_win" and m["人気"] == "popularity"
    assert m["厩舎"] == "trainer_name"


def test_parse_result_table():
    df = netkeiba.parse_result(HTML, "202465010101", pd.Timestamp("2026-01-01"), 1)
    assert df is not None and len(df) == 3
    r = df.set_index("post")
    assert r.loc[3, "finish"] == 1 and r.loc[3, "time_sec"] == pytest.approx(112.4)
    assert r.loc[3, "horse_id"] == "2019100001"
    assert r.loc[1, "jockey_id"] == "05002"  # ☆ 減量記号を除いて名前→ID 対応
    assert r.loc[1, "horse_weight"] == 980 and r.loc[1, "horse_weight_diff"] == -4
    assert r.loc[2, "status"] == "dnf"
    assert r.loc[3, "carried_weight"] == 690 and r.loc[3, "odds_win"] == pytest.approx(3.4)
    assert r.loc[1, "sex"] == "牝" and r.loc[1, "age"] == 6
    assert df["moisture"].iloc[0] == pytest.approx(3.2)
    assert df["race_class"].iloc[0] == "B2"


def test_race_id_format():
    assert netkeiba.race_id(pd.Timestamp("2024-12-30"), 1) == "202465123001"


def test_import_japanese_csv():
    from banei_ai.cli import import_frame

    raw = pd.DataFrame({
        "日付": ["2026/01/03"] * 3, "レース": [5, 5, 5], "馬場水分": ["2.4%"] * 3,
        "着順": [2, 1, "中止"], "馬番": [1, 2, 3], "馬名": ["A", "B", "C"], "性齢": ["牡4", "牝5", "セ6"],
        "負担重量": [700, 680, 690], "騎手": ["☆山田", "佐藤", "鈴木"], "タイム": ["1:50.2", "1:48.0", ""],
        "単勝": [3.2, 2.0, 9.9], "人気": [2, 1, 3], "馬体重": ["1012(+4)", "995(-2)", "1050(0)"],
    })
    df = import_frame(raw)
    r = df.set_index("post")
    assert r.loc[2, "finish"] == 1 and r.loc[3, "status"] == "dnf"
    assert r.loc[1, "jockey_id"] == "山田" and r.loc[1, "age"] == 4
    assert r.loc[1, "time_sec"] == pytest.approx(110.2) and r.loc[1, "horse_weight_diff"] == 4
    assert df["moisture"].iloc[0] == pytest.approx(2.4) and df["race_id"].iloc[0] == "2026010305"


def test_review_regressions():
    """レビューで見つかった不具合の再発防止."""
    import numpy as np

    from banei_ai import market, schema
    from banei_ai.clogit import harville_top2, place_prob
    from banei_ai.scrape.common import find_race_class

    # 各種マイナス記号・全角
    assert schema.to_float("−12") == -12 and schema.to_float("－８") == -8 and schema.to_float("＋6") == 6
    # 日本語に挟まれたクラス名・オープン
    assert find_race_class("<h2>混合A1-1組</h2>") == "A1"
    assert find_race_class("<h2>ばんえいオープン</h2>") == "オープン"
    # 欠けた人気はレース全体で順位付け
    d = pd.DataFrame({"race_id": ["r"] * 4, "odds_win": [2.0, 3.0, 5.0, 9.0], "popularity": [1, 2, np.nan, np.nan]})
    assert market.add_market_probs(d)["popularity"].tolist() == [1, 2, 3, 4]
    # 複勝: 7 頭以下は 2 着まで
    p = np.array([0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
    assert place_prob(p).sum() == pytest.approx(2.0)
    assert harville_top2(p).sum() == pytest.approx(2.0)
    assert np.isnan(place_prob(p[:4])).all()


def test_ids_keep_leading_zeros(tmp_path):
    from banei_ai import schema

    f = tmp_path / "x.csv"
    pd.DataFrame({"race_id": ["2024010101"], "jockey_id": ["05385"], "trainer_id": ["01001"]}).to_csv(f, index=False)
    d = schema.read_csv(f)
    assert d.loc[0, "jockey_id"] == "05385" and d.loc[0, "trainer_id"] == "01001"
