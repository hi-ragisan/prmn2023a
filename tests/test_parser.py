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
