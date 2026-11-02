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


def test_attach_ids_with_pedigree_in_cell():
    from banei_ai.scrape.common import attach_ids

    t = pd.DataFrame({"horse_name": ["アオノキセキ父:キタノ母:ハル", "ベニスズラン", "ナゾノウマ"]})
    t = attach_ids(t, "horse_name", "horse_id", {"アオノキセキ": "H1", "ベニスズラン": "H2"})
    assert t["horse_id"].tolist() == ["H1", "H2", "ナゾノウマ"]
    assert t["horse_name"].tolist() == ["アオノキセキ", "ベニスズラン", "ナゾノウマ"]


# ---------------------------------------------------------------------------
# 以下は実ページの構造を「観察してから手で再構成した」小さな HTML (実ページのコピーではない)。
# 観察元 (詳細は docs/scraper_verification.md):
#   keiba.go.jp: 公開リポジトリに保存されていた実ページ (金沢 2025-11-10 の RaceList / DebaTable /
#     RaceMarkTable、佐賀 2026-09-20 の OddsTanFuku。テンプレートは全場共通) と、
#     ばんえいを実際に取得しているパーサー (列位置・馬場水分の位置・負担重量の書式)。
#   netkeiba: 他者のパーサー (2026 年版) と検索結果の抜粋 (「天候:晴 / 水分量:1.6」)。
# 馬名・騎手名・ID・数値は架空。ばんえい固有の部分 (成績表の単勝列・「馬場：1.8」の書式) は推定を含む。
# ---------------------------------------------------------------------------

# keiba.go.jp 成績表 (RaceMarkTable, 旧来レイアウト): 見出しが <td> で、前にタイトル行がある。
# 馬体重の増減は「差」列。騎手は「名前(所属)」。ばんえいは「馬場」に水分の数値が入る (推定)。
KEIBAGO_RESULT = """
<html><head><META http-equiv=content-type content="text/html; charset=UTF-8" /><title>レース成績表</title></head>
<body>
<table class="bs" border="0"><tr><td nowrap>
2026年11月1日 （日）
<span class="plus1bold01">帯　広第１競走</span>
直線
200ｍ

天候：晴　馬場：1.8
</td></tr></table>
<span class="midium"><NOBR><span class="plus1bold02">Ｃ２－１</span></NOBR>
<BR>（ばんえい系　一般）</span>
<table class="bs" border="0"><tr><td class="dbtbl">
<table class="bs" border=1 width="640">
<tr><td colspan="15" class="dbtitle"><span class="plus1bold01">１R        成績表</span></td></tr>
<tr class="dbitem"><td nowrap>  着<br>  順  </td><td nowrap>  枠<br>  番  </td><td nowrap>  馬<br>  番  </td>
<td nowrap>馬名</td><td nowrap>所属</td><td nowrap>性齢</td><td nowrap>
  負担<br>
  重量
</td><td nowrap>  騎手<br>  (所属)  </td><td nowrap>調教師</td><td nowrap>馬体重</td><td nowrap>差</td>
<td nowrap>タイム</td><td nowrap>着差</td><td nowrap>  人<br>  気  </td><td nowrap>単勝<br>オッズ</td></tr>
<tr bgcolor="#FFFFFF" align="center">
<td nowrap><span class="bold">1</span></td><td nowrap>3</td><td nowrap>3</td>
<td nowrap class="dbdata3"><span class="bold"><a href='
/KeibaWeb/DataRoom/HorseMarkInfo?k_lineageLoginCode=30123456789
' target="_blank">
アオノキセキ
</a></span></td>
<td nowrap>帯広</td><td nowrap><span class=''>牡 5 </span></td><td nowrap class="dbdata2"> 690</td>
<td nowrap><a href='/KeibaWeb/DataRoom/RiderMark?k_riderLicenseNo=38001' target="_blank">
 鈴木恵(帯広)
</a><br></td>
<td nowrap><a href='/KeibaWeb/DataRoom/TrainerMark?k_trainerLicenseNo=18001' target="_blank">
坂本東
</a></td>
<td nowrap class="dbdata2">1012</td><td nowrap>+4</td><td nowrap class="dbdata2">
1:52.4
</td><td nowrap>&nbsp;</td><td nowrap>2</td><td nowrap>3.4</td></tr>
<tr bgcolor="#FFFFFF" align="center">
<td nowrap><span class="bold">2</span></td><td nowrap>1</td><td nowrap>1</td>
<td nowrap class="dbdata3"><span class="bold"><a href='/KeibaWeb/DataRoom/HorseMarkInfo?k_lineageLoginCode=30123456790' target="_blank">
ベニスズラン</a></span></td>
<td nowrap>帯広</td><td nowrap><span class='red'>牝 6 </span></td><td nowrap class="dbdata2"> 670</td>
<td nowrap><a href='/KeibaWeb/DataRoom/RiderMark?k_riderLicenseNo=38002' target="_blank">
 ☆金田利(帯広)</a><br></td>
<td nowrap><a href='/KeibaWeb/DataRoom/TrainerMark?k_trainerLicenseNo=18002' target="_blank">松井浩</a></td>
<td nowrap class="dbdata2">980</td><td nowrap>-6</td><td nowrap class="dbdata2">1:55.0</td><td nowrap>2.6</td>
<td nowrap>1</td><td nowrap>2.1</td></tr>
<tr bgcolor="#FFFFFF" align="center">
<td nowrap>競走中止</td><td nowrap>2</td><td nowrap>2</td>
<td nowrap><a href='/KeibaWeb/DataRoom/HorseMarkInfo?k_lineageLoginCode=30123456791'>クロガネ</a></td>
<td nowrap>帯広</td><td nowrap>セ 7</td><td nowrap> 700</td>
<td nowrap><a href='/KeibaWeb/DataRoom/RiderMark?k_riderLicenseNo=38003'>西謙(帯広)</a></td>
<td nowrap><a href='/KeibaWeb/DataRoom/TrainerMark?k_trainerLicenseNo=18003'>金田勇</a></td>
<td nowrap>1010</td><td nowrap>0</td><td nowrap></td><td nowrap></td><td nowrap>3</td><td nowrap>8.8</td></tr>
<tr bgcolor="#FFFFFF" align="center">
<td nowrap>出走取消</td><td nowrap>4</td><td nowrap>4</td>
<td nowrap><a href='/KeibaWeb/DataRoom/HorseMarkInfo?k_lineageLoginCode=30123456792'>シロガネ</a></td>
<td nowrap>帯広</td><td nowrap>牝 4</td><td nowrap> 650</td>
<td nowrap><a href='/KeibaWeb/DataRoom/RiderMark?k_riderLicenseNo=38004'>東涼(帯広)</a></td>
<td nowrap><a href='/KeibaWeb/DataRoom/TrainerMark?k_trainerLicenseNo=18004'>林豊</a></td>
<td nowrap></td><td nowrap></td><td nowrap></td><td nowrap></td><td nowrap></td><td nowrap></td></tr>
</table></td></tr></table>
<table border="1"><tr><td>単勝</td><td>3</td><td>340円</td><td>2</td></tr></table>
</body></html>
"""


def test_keibago_result_td_header_table():
    """見出しが <td> の成績表 (旧コードでは表を見つけられず 0 レースになっていた)."""
    from banei_ai.scrape import keibago

    df = keibago.parse_result_page(KEIBAGO_RESULT, pd.Timestamp("2026-11-01"), 1)
    assert df is not None and len(df) == 4
    r = df.set_index("post")
    assert r.loc[3, "finish"] == 1 and r.loc[3, "time_sec"] == pytest.approx(112.4)
    assert r.loc[3, "horse_id"] == "30123456789" and r.loc[3, "horse_name"] == "アオノキセキ"
    assert r.loc[3, "jockey_name"] == "鈴木恵" and r.loc[3, "jockey_id"] == "38001"  # (所属) を除く
    assert r.loc[1, "jockey_id"] == "38002"  # ☆ と (帯広) を除いて対応
    assert r.loc[3, "trainer_id"] == "18001"
    assert r.loc[3, "carried_weight"] == 690 and r.loc[3, "horse_weight"] == 1012
    assert r.loc[3, "horse_weight_diff"] == 4 and r.loc[1, "horse_weight_diff"] == -6  # 「差」列
    assert r.loc[3, "odds_win"] == pytest.approx(3.4) and r.loc[1, "popularity"] == 1
    assert r.loc[2, "status"] == "dnf" and r.loc[4, "status"] == "scratch"
    assert r.loc[1, "sex"] == "牝" and r.loc[1, "age"] == 6
    assert df["moisture"].iloc[0] == pytest.approx(1.8)  # 「馬場：1.8」(% なし)
    assert df["race_class"].iloc[0] == "C2" and df["race_name"].iloc[0] == "Ｃ２－１"
    assert df["race_id"].iloc[0] == "2026110101"


# keiba.go.jp 出馬表 (DebaTable): 1 頭 = tr.tBorder から始まる 5 行。過去走欄に「取消」や
# 過去の馬場状態が入り、見出しには「馬名」の語が無い。同枠 2 頭目は枠セルが省略される。
def _deba_block(frame_td, post, hid, name, jid, jockey, odds_cell, sex_age, weight_cell,
                tid, trainer, hw_cell, info, past_rank):
    past = "".join(
        f'<td><div class="raceInfo"><span class="pastRank ">{past_rank}</span>'
        f'26.10.{20 + i}　重(2.{i})　10頭<br>帯広　直200　{i + 1}番</div></td>' for i in range(5))

    def sub(txt):
        return "".join(f'<td class="">{txt}</td>' for _ in range(5))

    return f"""
<tr class="tBorder">{frame_td}
<td rowspan="5" class="horseNum">{post}</td>
<td colspan="3"><a class="horseName" target="_blank" href="../DataRoom/HorseMarkInfo?k_lineageLoginCode={hid}">{name}</a></td>
<td colspan="1"><a class="jockeyName" target="_blank" href="../DataRoom/RiderMark?k_riderLicenseNo={jid}">
{jockey}<span class="jockeyarea">（帯広）</span></a></td>
<td class="odds_weight" rowspan="2">{odds_cell}</td>
<td rowspan="5" class="result rBorder"><table class="arrival noBorder">
<tr><td>全&nbsp;&nbsp;</td><td class="alignR">1-</td><td class="alignR">2-</td><td class="alignR">0-</td><td class="alignR">15</td></tr>
<tr><td colspan="2">1:48.2</td><td colspan="3">1:48.2</td></tr></table></td>
{past}
</tr>
<tr><td class="noBorder"><span class="male">{sex_age}</span></td><td class="noBorder">鹿毛</td><td>20.04.01生</td>
<td>{weight_cell}</td>{sub('<a href="../TodayRaceInfo/RaceMarkTable?k_raceDate=2026%2f10%2f20" class="race">Ｃ２－１</a>')}</tr>
<tr><td colspan="3">父ウマ</td><td colspan="1"><a target="_blank" href="../DataRoom/TrainerMark?k_trainerLicenseNo={tid}">
{trainer}（帯広）</a></td><td rowspan="2" class="odds_weight">{hw_cell}</td>{sub('3人　1050　鈴木恵 690')}</tr>
<tr><td colspan="3">母ウマ</td><td colspan="1">馬主名</td>{sub('1:52.4　1-1-1-1　0.0')}</tr>
<tr><td colspan="3">（母父ウマ）</td><td colspan="1">生産牧場</td><td class="info">{info}</td>{sub('0.0　ライバル')}</tr>
"""


KEIBAGO_DEBA = (
    """<html><head><meta charset="UTF-8"><title>出馬表｜地方競馬情報サイト</title></head><body>
<article class="raceCard"><h4>2026年11月1日（日）　帯　広　第1競走　14:45発走</h4>
<section class="raceTitle"><h3>Ｃ２－１</h3>
<ul class="dataArea"><li>直線　200ｍ　　天候：晴　　馬場：1.8　　ばんえい　一般</li></ul></section>
<section class="cardTable"><table><tbody>
<tr><th rowspan="2" class="a">枠番</th><th rowspan="2" class="b">馬番</th><th colspan="3" rowspan="2" class="c">競走馬</th>
<th colspan="1" rowspan="2" class="d">騎手・調教師<br>馬主・生産牧場</th><th rowspan="2" class="e">オッズ<br>馬体重<br>変更情報</th>
<th rowspan="2" class="k">着別成績<br>最高タイム</th><th colspan="5" class="l">競走成績</th></tr>
<tr><th class="f">前走</th><th class="g">前々走</th><th class="h">3走前</th><th class="i">4走前</th><th class="rBorder j">5走前</th></tr>
<tr><td colspan="11"><div class="raceInfo"><span class="pastRank rank_02">着順</span>競走日　馬場状態（馬場水分）　出走頭数</div></td></tr>
"""
    + _deba_block('<td rowspan="10" class="courseNum course_01">1</td>', 1, "30123456789", "アオノキセキ",
                  "38001", "鈴木恵", '<span class="odds_Black odds">8.1</span><br>(3人気)', "牡5",
                  "☆ 690　0-0-0-0", "18001", "坂本東", "1061<br>(+26)", "", "取消")
    + _deba_block("", 2, "30123456790", "ベニスズラン", "38002", "金田利",
                  '<span class="odds_Black odds">2.1</span><br>(1人気)', "牝6", "  670　1-0-0-3",
                  "18002", "松井浩", "980<br>(-4)", "", "1")
    + _deba_block('<td rowspan="5" class="courseNum course_02">2</td>', 3, "30123456791", "クロガネ",
                  "38003", "西謙", "", "セ7", "  700　0-0-0-1", "18003", "金田勇", "", "出走取消", "2")
    + "</tbody></table></section></article></body></html>"
)


def test_keibago_deba_table_blocks():
    """出馬表は専用パーサーで読む (旧コードでは「馬名」見出しが無く常に None だった)."""
    from banei_ai.scrape import keibago

    t = keibago.parse_card_page(KEIBAGO_DEBA, pd.Timestamp("2026-11-01"), 1)
    assert t is not None and t["post"].tolist() == [1, 2, 3]
    r = t.set_index("post")
    assert r.loc[1, "frame"] == 1 and r.loc[2, "frame"] == 1 and r.loc[3, "frame"] == 2  # 同枠は枠セル省略
    assert r.loc[1, "horse_id"] == "30123456789" and r.loc[1, "horse_name"] == "アオノキセキ"
    assert r.loc[1, "jockey_name"] == "鈴木恵" and r.loc[1, "jockey_id"] == "38001"
    assert r.loc[1, "trainer_name"] == "坂本東" and r.loc[1, "trainer_id"] == "18001"
    assert r.loc[1, "carried_weight"] == 690 and r.loc[2, "carried_weight"] == 670
    assert r.loc[1, "horse_weight"] == 1061 and r.loc[1, "horse_weight_diff"] == 26
    assert r.loc[2, "horse_weight_diff"] == -4
    assert r.loc[1, "odds_win"] == pytest.approx(8.1) and r.loc[1, "popularity"] == 3
    assert r.loc[1, "sex"] == "牡" and r.loc[1, "age"] == 5 and r.loc[3, "sex"] == "セ"
    # 過去走の「取消」では取消扱いにしない。変更情報欄の「出走取消」だけが取消
    assert r.loc[1, "status"] == "entry" and r.loc[3, "status"] == "scratch"
    # 馬場水分は過去走欄 (2.x) ではなくレース情報欄の値
    assert t["moisture"].iloc[0] == pytest.approx(1.8)
    assert t["race_class"].iloc[0] == "C2"
    assert t["finish"].isna().all()


# keiba.go.jp OddsTanFuku: 複勝は colspan=2 の見出しの下で「1.1-」「1.2」と 2 セルに分かれる
KEIBAGO_ODDS = """
<html><head><meta charset="UTF-8"></head><body>
<table class="odd_popular_table_02"><thead><tr>
<th>枠</th><th>馬番</th><th>馬名</th><th>単勝<br>オッズ</th><th colspan="2">複勝オッズ<br>(3着払い)</th>
<th>性齢</th><th>馬体重<br><span>(増減)</span></th><th>負担<br>重量</th><th>騎手<br><span>(所属)</span></th>
<th>所属</th><th>調教師</th><th>変更情報</th></tr></thead><tbody>
<tr><td class="odd_post1" rowspan="2">1</td><td>1</td><td>アオノキセキ</td><td class="odd_red">3.4</td>
<td class="odd_red">1.1-</td><td class="odd_red">1.6</td><td>牡5</td><td>1061(+26)</td><td>690</td>
<td>鈴木恵<span>（帯広）</span></td><td>帯広</td><td>坂本東</td><td></td></tr>
<tr><td>2</td><td>ベニスズラン</td><td>12.5</td><td>2.4-</td><td>4.0</td><td>牝6</td><td>980(-4)</td>
<td>670</td><td>金田利</td><td>帯広</td><td>松井浩</td><td></td></tr>
<tr><td class="odd_post2">2</td><td>3</td><td>クロガネ</td><td>取消</td><td></td><td></td><td>セ7</td><td></td>
<td>700</td><td>西謙</td><td>帯広</td><td>金田勇</td><td>出走取消</td></tr>
</tbody></table></body></html>
"""


def test_keibago_odds_split_place_cells():
    from banei_ai.scrape import keibago

    o = keibago.parse_odds(KEIBAGO_ODDS).set_index("post")
    assert o.loc[1, "odds_win"] == pytest.approx(3.4) and o.loc[2, "odds_win"] == pytest.approx(12.5)
    # 旧コードは「1.1-」から 2 つ目の数値を探して 1 / 1 のような誤った値を返していた
    assert o.loc[1, "place_min"] == pytest.approx(1.1) and o.loc[1, "place_max"] == pytest.approx(1.6)
    assert o.loc[2, "place_min"] == pytest.approx(2.4) and o.loc[2, "place_max"] == pytest.approx(4.0)
    assert pd.isna(o.loc[3, "odds_win"]) and pd.isna(o.loc[3, "place_min"])
    # 1 セルに範囲が入る形にも対応
    one = (KEIBAGO_ODDS.replace('<th colspan="2">複勝オッズ', "<th>複勝オッズ")
           .replace('<td class="odd_red">1.1-</td><td class="odd_red">1.6</td>', '<td class="odd_red">1.1-1.6</td>')
           .replace("<td>2.4-</td><td>4.0</td>", "<td>2.4 - 4.0</td>")
           .replace("<td>取消</td><td></td><td></td>", "<td>取消</td><td></td>"))
    o1 = keibago.parse_odds(one).set_index("post")
    assert o1.loc[1, "place_min"] == pytest.approx(1.1) and o1.loc[2, "place_max"] == pytest.approx(4.0)


# keiba.go.jp RaceList: ばんえいは「馬場」欄に水分の数値 (例 1.8) が入る。下に変更情報の表もある。
KEIBAGO_RACE_LIST = """
<html><head><meta charset="UTF-8"></head><body><section class="raceTable"><table><tbody>
<tr><th colspan="13">2026年11月1日（日）  帯広ば競馬　当日メニュー</th></tr>
<tr class="subHeader"><th class="a">競走</th><th class="b">発走時刻</th><th class="c">変更</th><th class="d">競走種類</th>
<th class="e">競走名・出馬表</th><th class="f">コース</th><th class="g">天候</th><th class="h">馬場</th><th class="i">頭数</th>
<th class="k">オッズ・映像・成績</th></tr>
<tr class="data"><td>1R</td><td>14:45</td><td><span class="timechange"></span></td><td></td>
<td><a href="../TodayRaceInfo/DebaTable?k_raceDate=2026%2f11%2f01&k_raceNo=1&k_babaCode=3">Ｃ２－１</a></td>
<td>  直200m</td><td>晴</td><td>1.8</td><td>10</td><td><a class="chartBtn">成績</a></td></tr>
<tr class="data"><td>2R</td><td>15:20</td><td></td><td>特</td><td><a>オープン記念</a></td>
<td>  直200m</td><td>晴</td><td>2.1</td><td>9</td><td></td></tr>
</tbody></table></section>
<table class="changeInfo"><tr><th colspan="6">変更情報</th></tr>
<tr><th class="a">競走</th><th class="b">馬番</th><th class="c">馬名</th><th class="d">変更区分</th><th class="e">変更理由</th><th class="f">変更内容</th></tr>
<tr><td>3R</td><td>5</td><td>クロガネ</td><td>出走取消</td><td>疾病</td><td></td></tr></table>
</body></html>
"""


def test_keibago_race_list():
    from banei_ai.scrape import keibago

    rl = keibago.parse_race_list(KEIBAGO_RACE_LIST)
    assert rl["race_no"].tolist() == [1, 2]  # 変更情報の「3R」行はレースとして数えない
    assert rl["moisture"].tolist() == [pytest.approx(1.8), pytest.approx(2.1)]
    assert rl.loc[0, "race_class"] == "C2" and rl.loc[1, "race_class"] == "オープン"
    assert rl.loc[0, "weather"] == "晴" and rl.loc[0, "n_entries"] == 10
    # 表はあるがレースが無い日 → 空、表そのものが無いページ → None
    empty = KEIBAGO_RACE_LIST.split('<tr class="data">')[0] + "</tbody></table></section></body></html>"
    assert keibago.parse_race_list(empty).empty
    assert keibago.parse_race_list("<html><body><p>開催情報がありません</p></body></html>") is None


def test_moisture_formats_and_scope():
    from banei_ai.scrape.common import find_moisture

    # netkeiba ばんえいのレース情報欄 (検索結果の抜粋より): 「水分量:1.6」(% なし)
    nk = '<div class="RaceData01">18:55発走 / 直200m / 天候:晴 / <span>水分量:1.6</span></div>'
    assert find_moisture(nk) == pytest.approx(1.6)
    assert find_moisture('<div class="RaceData01">天候:雪 / 水分量：１．３％</div>') == pytest.approx(1.3)
    assert find_moisture("<p>天候：晴　馬場：1.8%</p>") == pytest.approx(1.8)
    # 平地の馬場状態 (良/不良) は水分ではない
    assert pd.isna(find_moisture("<li>天候：曇　　馬場：不良</li>"))
    # 出馬表: 過去走欄の値ではなくレース情報欄の値を採る
    assert find_moisture(KEIBAGO_DEBA) == pytest.approx(1.8)


# netkeiba 結果表: ばんえいの馬 ID は英字始まりのことがある。厩舎セルは所属が名前の前に付く (推定)。
NETKEIBA_RESULT = """
<html><head><meta charset="UTF-8"></head><body>
<div class="RaceName">ばんえい大賞典</div>
<div class="RaceData01">15:55発走 / 直200m / 天候:晴 / 水分量:0.8</div>
<table class="RaceTable01 RaceCommon_Table ResultRefund Table_Show_All" id="All_Result_Table">
<thead><tr class="Header"><th>着順</th><th>枠</th><th>馬番</th><th>馬名</th><th>性齢</th><th>斤量</th><th>騎手</th>
<th>タイム</th><th>着差</th><th>人気</th><th>単勝<br>オッズ</th><th>後3F</th><th>厩舎</th><th>馬体重<br>(増減)</th></tr></thead>
<tbody>
<tr><td class="Result_Num"><div class="Rank">1</div></td><td>2</td><td>2</td>
<td class="Horse_Info"><span class="Horse_Name"><a href="https://db.netkeiba.com/horse/B201900123/">アオノキセキ</a></span></td>
<td>牡4</td><td>710</td><td class="Jockey"><a href="https://db.netkeiba.com/jockey/result/recent/B0049/">鈴木恵</a></td>
<td>2:01.3</td><td></td><td>1</td><td>2.0</td><td></td>
<td class="Trainer"><span class="Label1">帯広</span><a href="https://db.netkeiba.com/trainer/result/recent/B1001/">坂本東</a></td>
<td>1050(+6)</td></tr>
<tr><td class="Result_Num"><div class="Rank">除外</div></td><td>1</td><td>1</td>
<td class="Horse_Info"><span class="Horse_Name"><a href="https://db.netkeiba.com/horse/B201900124/">シロガネ</a></span></td>
<td>牝4</td><td>690</td><td class="Jockey"><a href="https://db.netkeiba.com/jockey/result/recent/B0050/">西謙</a></td>
<td></td><td></td><td></td><td>---</td><td></td>
<td class="Trainer"><span class="Label1">帯広</span><a href="https://db.netkeiba.com/trainer/result/recent/B1002/">松井浩</a></td>
<td>計不</td></tr>
</tbody></table></body></html>
"""


def test_netkeiba_letter_ids_and_trainer_prefix():
    df = netkeiba.parse_result(NETKEIBA_RESULT, "202465070711", pd.Timestamp("2024-07-07"), 11)
    r = df.set_index("post")
    assert r.loc[2, "horse_id"] == "B201900123" and r.loc[2, "jockey_id"] == "B0049"
    assert r.loc[2, "trainer_id"] == "B1001" and r.loc[2, "trainer_name"] == "坂本東"  # 「帯広坂本東」→ 末尾一致
    assert r.loc[2, "time_sec"] == pytest.approx(121.3) and r.loc[2, "horse_weight_diff"] == 6
    assert r.loc[1, "status"] == "scratch" and pd.isna(r.loc[1, "odds_win"])
    assert df["moisture"].iloc[0] == pytest.approx(0.8) and df["race_name"].iloc[0] == "ばんえい大賞典"


def test_netkeiba_card_scratch_and_js_odds():
    """出馬表: オッズは JS で入るため HTML では「---.-」。取消は着順列が無いので他の欄から判定."""
    html = """<html><head><meta charset="UTF-8"></head><body><div class="RaceData01">天候:曇 / 水分量:2.2</div>
<table class="Shutuba_Table RaceTable01"><thead><tr class="Header"><th>枠</th><th>馬番</th><th>印</th><th>馬名</th><th>性齢</th>
<th>斤量</th><th>騎手</th><th>厩舎</th><th>馬体重<br>(増減)</th><th>オッズ<br>更新</th><th>人気</th></tr></thead><tbody>
<tr class="HorseList"><td>1</td><td>1</td><td></td><td><a href="https://db.netkeiba.com/horse/B201900123">アオノキセキ</a></td>
<td>牡4</td><td>710</td><td><a href="https://db.netkeiba.com/jockey/result/recent/B0049/">鈴木恵</a></td><td>帯広坂本東</td>
<td></td><td><span id="odds-1_01">---.-</span></td><td>**</td></tr>
<tr class="HorseList Cancel"><td>2</td><td>2</td><td></td><td><a href="https://db.netkeiba.com/horse/B201900124">シロガネ</a></td>
<td>牝4</td><td>690</td><td><a href="https://db.netkeiba.com/jockey/result/recent/B0050/">西謙</a></td><td>帯広松井浩</td>
<td>取消</td><td>取消</td><td></td></tr></tbody></table></body></html>"""
    t = netkeiba.parse_card(html, pd.Timestamp("2025-08-10"), 11)
    r = t.set_index("post")
    assert r.loc[1, "status"] == "entry" and r.loc[2, "status"] == "scratch"
    assert t["odds_win"].isna().all() and t["finish"].isna().all()
    assert r.loc[1, "horse_id"] == "B201900123" and t["moisture"].iloc[0] == pytest.approx(2.2)


def test_header_words_keibago():
    m = map_headers(["着 順", "枠 番", "馬 番", "馬名", "所属", "性齢", "負担 重量", "騎手 (所属)", "調教師",
                     "馬体重", "差", "タイム", "着差", "上り 3F", "人 気", "単勝 オッズ"])
    assert m["着 順"] == "finish" and m["馬 番"] == "post" and m["枠 番"] == "frame"
    assert m["負担 重量"] == "carried_weight" and m["騎手 (所属)"] == "jockey_name"
    assert m["差"] == "horse_weight_diff" and "着差" not in m and "所属" not in m
    assert m["人 気"] == "popularity" and m["単勝 オッズ"] == "odds_win"


def test_decode_html_detects_charset():
    from banei_ai.scrape.common import decode_html

    body = "<html><body>着順 馬名 ばんえい 天候:晴 / 水分量:1.6</body></html>"
    # UTF-8 のページを EUC-JP と決め打ちしない (nar.netkeiba.com は現在 UTF-8 との報告)
    text, enc = decode_html(('<meta charset="UTF-8">' + body).encode("utf-8"), None, "euc-jp")
    assert "水分量:1.6" in text and enc == "utf-8"
    text, enc = decode_html(('<meta charset="EUC-JP">' + body).encode("euc-jp"), None, "utf-8")
    assert "水分量:1.6" in text and enc == "euc_jp"
    # HTTP ヘッダーが誤っていても厳密デコードに失敗すれば次の候補へ
    text, _ = decode_html(body.encode("utf-8"), "EUC-JP", None)
    assert "ばんえい" in text


def test_fetcher_decodes_and_refetches_mojibake_cache(tmp_path):
    from banei_ai.scrape.common import Fetcher

    class Resp:
        content = '<html><meta charset="UTF-8"><body>水分量:1.6</body></html>'.encode("utf-8")
        headers = {"Content-Type": "text/html; charset=EUC-JP"}
        encoding = "EUC-JP"

        def raise_for_status(self):
            pass

    f = Fetcher(cache_dir=tmp_path, wait=0, encoding="euc-jp")
    calls = []

    def fake_get(url, timeout=30):
        calls.append(url)
        return Resp()

    f.session.get = fake_get
    url = "https://example.invalid/race/result.html?race_id=1"
    f._path(url).write_text("�" * 50, encoding="utf-8")  # 旧版で化けたキャッシュ
    assert "水分量:1.6" in f.get(url) and len(calls) == 1
    assert "水分量:1.6" in f.get(url) and len(calls) == 1  # 正常なキャッシュは再取得しない


class _FakeFetcher:
    """URL の一部 → HTML の辞書で応答する (ネットに出ない)."""

    def __init__(self, pages):
        self.pages, self.calls, self.wait = pages, [], 0

    def get(self, url, use_cache=True):
        self.calls.append(url)
        for key, html in self.pages.items():
            if key in url:
                return html
        return "<html><body>データがありません</body></html>"

    def invalidate(self, url):
        pass


def test_keibago_scrape_day_uses_race_list():
    from banei_ai.scrape import keibago

    no_moist = KEIBAGO_RESULT.replace("天候：晴　馬場：1.8", "天候：晴")
    f = _FakeFetcher({"RaceList": KEIBAGO_RACE_LIST, "RaceMarkTable?k_raceDate=2025/11/03&k_babaCode=3&k_raceNo=1": no_moist})
    df = keibago.scrape_day(pd.Timestamp("2025-11-03"), f)
    assert df["race_no"].unique().tolist() == [1]  # 2R は成績表が無い (未確定) ので除く
    assert df["moisture"].iloc[0] == pytest.approx(1.8)  # 成績表に無い水分を RaceList から補完
    assert not any("k_raceNo=3" in u for u in f.calls)  # RaceList に無いレースは取りにいかない
    # 開催なし: レース一覧の表はあるが行が無い → 1 リクエストで終わる
    empty = KEIBAGO_RACE_LIST.split('<tr class="data">')[0] + "</tbody></table></section></body></html>"
    f2 = _FakeFetcher({"RaceList": empty})
    assert keibago.scrape_day(pd.Timestamp("2025-11-04"), f2).empty and len(f2.calls) == 1


def test_keibago_classed_result_layout_fallback():
    """2026 年版の成績表 (セルに a/c/d/j/k のクラス) で見出し語が読めない場合の予備."""
    from banei_ai.scrape import keibago

    html = """<html><body><table><tr><td class="a">1</td><td class="b">3</td><td class="c">3</td>
<td class="d"><a href="/KeibaWeb/DataRoom/HorseMarkInfo?k_lineageLoginCode=30123456789">アオノキセキ</a></td>
<td class="h"><a href="/KeibaWeb/DataRoom/RiderMark?k_riderLicenseNo=38001">鈴木恵(帯広)</a></td>
<td class="j">1012(+4)</td><td class="k">1:52.4</td></tr>
<tr><td class="a">取消</td><td class="b">1</td><td class="c">1</td>
<td class="d"><a href="/KeibaWeb/DataRoom/HorseMarkInfo?k_lineageLoginCode=30123456790">ベニスズラン</a></td>
<td class="h"><a href="/KeibaWeb/DataRoom/RiderMark?k_riderLicenseNo=38002">金田利(帯広)</a></td>
<td class="j"></td><td class="k"></td></tr></table></body></html>"""
    df = keibago.parse_result_page(html, pd.Timestamp("2026-11-01"), 1)
    r = df.set_index("post")
    assert r.loc[3, "finish"] == 1 and r.loc[3, "horse_id"] == "30123456789"
    assert r.loc[3, "jockey_id"] == "38001" and r.loc[3, "horse_weight_diff"] == 4
    assert r.loc[3, "time_sec"] == pytest.approx(112.4) and r.loc[1, "status"] == "scratch"


def test_netkeiba_moisture_fallback_from_keibago(monkeypatch):
    from banei_ai.scrape import keibago

    no_moist = NETKEIBA_RESULT.replace(" / 水分量:0.8", "")
    f = _FakeFetcher({"race_id=202465110201": no_moist})  # 2024-11-02 1R (2R は表なし = 終了)
    monkeypatch.setattr(keibago, "race_list", lambda date, fetcher, use_cache=True: pd.DataFrame(
        {"race_no": [1, 2], "moisture": [2.4, 2.5]}))
    df = netkeiba.scrape_day(pd.Timestamp("2024-11-02"), f)
    assert df["race_no"].unique().tolist() == [1] and df["moisture"].iloc[0] == pytest.approx(2.4)


def test_diagnose_saved_html(tmp_path, capsys):
    from banei_ai.scrape import diagnose

    p = tmp_path / "RaceMarkTable_2026110101.html"
    p.write_bytes(KEIBAGO_RESULT.encode("utf-8"))
    assert diagnose.main([str(p), "--source", "keibago"]) == 0
    out = capsys.readouterr().out
    assert "採用した表" in out and "horse_weight_diff" in out and "対応しなかった見出し" in out
    assert "馬場水分: 1.8" in out and "ID 対応率" in out and "馬 100%" in out
    s = diagnose.diagnose(KEIBAGO_DEBA, "keibago", target="deba.html", out=lambda *_: None)
    assert s["kind"] == "deba" and s["rows"] == 3
    p3 = tmp_path / "netkeiba.html"
    p3.write_bytes(NETKEIBA_RESULT.encode("utf-8"))
    assert diagnose.main([str(p3)]) == 0
    assert "水分量:0.8" in capsys.readouterr().out
