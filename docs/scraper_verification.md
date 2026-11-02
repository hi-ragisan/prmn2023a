# スクレイパーの実ページ検証メモ (2026-11-02)

## 0. 前提と結論

- **開発環境から実サイトには直接アクセスできなかった**（nar.netkeiba.com・www.keiba.go.jp とも、
  シェルからも Web 取得ツールからも名前解決できず）。**ばんえい（帯広）の実ページそのものは一度も見ていない。**
- 代わりに次の 3 つで実ページの構造を確認した。
  1. **公開リポジトリに保存されていた keiba.go.jp の実ページ**（他場。サイトのテンプレートは全場共通）
     - 金沢 2025-11-10: RaceList / DebaTable(4R) / RaceMarkTable(4R) — `kako-jun/kanazawa-dirt-one-spear`
       （`backend/data/html/2025/20251110/` の `race_list.html`, `race_04_deba.html`, `race_04_result.html`）
     - 佐賀 2026-09-20: OddsTanFuku(1R) — `kantsu888-droid/oba-odds-export`（`data/2026-09-20/saga_1R_123811.html`）
  2. **実際に動いている他者のパーサー**（列位置・クラス名・書式の根拠）
     - `shirokuma89dev/BambaObihiroML`（**ばんえい専用**、keiba.go.jp から 2018 年以降を取得して分析しているプロジェクト。
       `src/scraper/banei_scraper.py`, `banei_race_day_scraper.py`, `scrape_live_odds.py`, `AGENTS.md`）
     - `kkkaoru/horse-racing-data`（keiba.go.jp、`apps/sync-realtime-data-hot/src/keiba-go.ts`、2026-05 時点の 2 レイアウト対応）
     - `kanta13jp1/my_web_app`（netkeiba、`scripts/fetch_horse_racing.py`、「2026-04 HTML構造対応版」）
     - `yuyouyu80-sudo/netkeiba-nar-pipeline`, `k29770485-svg/keiba-`（netkeiba NAR の文字コード）
     - `craftman1202/HourseRacing_Local`（OddsTanFuku の複勝列）, `poi-ai/keibaAI`（場コード表）, `nagashima-w/keiba-prediction`
  3. **検索結果の抜粋**（L1 調査レーン経由）: nar.netkeiba.com のばんえい結果ページの
     「18:55発走 / 直200m / 天候:晴 / 水分量:1.6」など。
- 保存されていた実ページに対して、**旧コードは keiba.go.jp の成績表・出馬表をどちらも読めず（None）、複勝オッズは誤った値を返した**。
  新コードではすべて正しく読めることを手元で確認した（実ページはリポジトリに含めていない。テストは観察した構造を手で再構成した小さな HTML）。

## 1. 見つかった不具合と修正

| # | 不具合 | 根拠 | 影響 | 修正 |
|---|---|---|---|---|
| 1 | keiba.go.jp 成績表 (RaceMarkTable) の見出しが `<th>` でなく `<tr class="dbitem"><td>着<br>順</td>…`、その前に「４R 成績表」のタイトル行 → pandas が見出しを認識せず、表を選べない | 実ページ（金沢）で旧コードが `None` | `--source keibago` で**全日 0 レース** | 先頭 6 行から見出し行を探す (`common.inspect_tables`) |
| 2 | keiba.go.jp 出馬表 (DebaTable) は見出しが「競走馬」(馬名の語が無い)、1 頭 = `tr.tBorder` から始まる 5 行ブロック、過去 5 走の欄付き | 実ページ（金沢）で旧コードが `None` | `--card-source keibago` の予測が不可能 | 専用パーサー `keibago.parse_deba_table` |
| 3 | OddsTanFuku の複勝が colspan=2 の見出しの下で「1.1-」「1.2」の 2 セルに分かれる | 実ページ（佐賀）。旧コードは 1.1/1.2 → **1/1**、1.8/3.1 → **1/8** と誤読 | 複勝下限・複勝期待値が誤り | 2 セル・1 セル両対応 (`keibago.parse_odds`) |
| 4 | netkeiba を EUC-JP と決め打ち。nar.netkeiba.com は現在 UTF-8 との報告 | 2 件（NAR の result.html を `encoding="utf-8"` で取得するコード、「NARはUTF-8」と書いた資料）。直接は未確認 | UTF-8 なら**全ページ文字化け**（実ページで試すと置換文字 2,387 個） | HTTP ヘッダー・`<meta charset>`・厳密デコードで自動判定 (`common.decode_html`)。`encoding` 引数はヒント扱いに。化けたキャッシュは取り直す |
| 5 | netkeiba ばんえいの馬場水分は「水分量:1.6」(「量」が入り、% なし) → 旧正規表現に一致せず NaN | 検索結果の抜粋 5 件（2023〜2025 の重賞など） | 馬場水分の特徴量が全滅 | 「水分量?」「% 任意」に対応 |
| 6 | keiba.go.jp の馬場水分は RaceList の「馬場」欄に数値だけ（例 `1.7`）。成績表・出馬表のレース情報欄は平地だと「馬場：不良」 | BambaObihiroML が RaceList の 8 列目を `float()` で読んで分析に使っている | 成績表から読めない場合 NaN | 「馬場：数値」を水分と解釈 + **RaceList から補完**（keibago は RaceList を先に取得してレース番号も得る） |
| 7 | ページ全体から水分を探すと、出馬表の過去走欄（凡例「馬場状態（馬場水分）」）の値を拾い得る | 実ページ（金沢）の出馬表 | 誤った水分 | レース情報欄（`.RaceData01`、`.raceTitle .dataArea`、「天候」を含む短い要素）を優先し、ページ全体は「水分量:」「馬場:」直後の数値のみ |
| 8 | keiba.go.jp の騎手名は「沖静男(金沢)」「田知弘（金沢）」と所属付き | 実ページ | 騎手名が「名前(帯広)」になる（ID は取れていた） | `norm_name` で末尾の (所属) を除去 |
| 9 | keiba.go.jp 成績表の馬体重増減は「差」という単独の列 | 実ページ | 馬体重増減が NaN | 見出し「差」→ `horse_weight_diff`（「着差」は対象外） |
| 10 | 取消・除外の表記: 着順欄に「出走取消」「競走除外」「出場停止」、出馬表は「変更情報」欄 (`td.info`) | kkkaoru のパーサーの表記一覧、実ページの出馬表 | 出馬表で取消馬を出走扱い | 着順欄に加え、出馬表ではオッズ・馬体重などの欄からも判定（過去走欄は見ない） |
| 11 | 表が 1 つも無いページ（非開催日など）で `pd.read_html` が bs4+html5lib に切り替わり、html5lib 未導入だと `ImportError` | 手元で再現（requirements に html5lib なし） | **非開催日に当たるとスクレイピングが例外で停止** | `flavor="lxml"` に固定し例外を捕捉 |
| 12 | keiba.go.jp で結果未確定の成績表（着順なし）をそのまま取り込む | コード読み | 着順が全部 NaN のレースが混入 | 着順が 1 つも無ければ未確定として除外 |
| 13 | netkeiba の厩舎セルは所属が名前の前に付くことがある（「帯広坂本東」） | kanta13jp1 の NAR 用分割処理（ばんえいでの表記は未確認） | 調教師 ID が名前のまま | ID 付与で末尾一致も試す |

このほか、keiba.go.jp 成績表の **2026 年版レイアウト**（各セルに 1 文字のクラス a=着順 c=馬番 d=馬名 j=馬体重 k=タイム。kkkaoru のパーサーより）
で見出し語が読めなかった場合の予備パーサーを入れた。

## 2. データの取れ方に関する重要な事実

- **keiba.go.jp 成績表の単勝オッズは過去の年が空欄**（当年分のみ。人気は全年あり）— BambaObihiroML の AGENTS.md
  「実オッズはサイト側で過去年が空欄化しており2026年のみ取得可」。OddsTanFuku も遡れるのは半年あまり
  （2026-08-20 時点で 2026-02-02 以降は取れ、1 月分は取れなかった）。
  → **`--source keibago` で集めた過去データにはオッズが無く、オッズの歪みの学習・検証ができない。過去データは netkeiba で集めること。**
- 場コード: netkeiba は **65 = 帯広(ば)**（33 は廃止された平地の帯広。混同しないこと）、keiba.go.jp は **k_babaCode=3 = 帯広ば**
  （実ページのナビのリンク文字が「帯広ば」）。複数の資料・実 URL で一致。
- netkeiba の race_id = YYYY + 65 + MM + DD + RR（例 202465070711 = 2024-07-07 11R）。12R まである日がある。金曜開催もある。
- netkeiba のばんえいの馬 ID・騎手 ID は英字始まりがある（`B201900123`, `B0049`）。数値と決めつけない（正規表現は `\w+`）。
- 文字コード: keiba.go.jp は UTF-8（実ページの `<meta charset="UTF-8">`）。
- keiba.go.jp の実ページ内リンクは `k_raceDate=2025%2f11%2f10`（スラッシュをエンコード）だが、
  エンコードしない `2026/08/29` でも動いている実例がある（BambaObihiroML）。

## 3. 確認できた実ページ構造（要点）

### keiba.go.jp（実ページで確認。ばんえい固有の部分は根拠を併記）
- **RaceList**: `section.raceTable > table`。見出し `<th>`:「競走 / 発走時刻 / 変更 / 競走種類 / 競走名・出馬表 / コース / 天候 / 馬場 / 頭数 / オッズ・映像・成績」。
  行は `1R, 12:25, …`。ばんえいは「馬場」に水分の数値（BambaObihiroML）。下に「変更情報」の表（行頭が「3R」など）がある。
- **RaceMarkTable（旧来レイアウト）**: レース情報は `<table class="bs"><tr><td nowrap>…天候：曇　馬場：不良</td>`、
  レース名は `span.plus1bold02`。成績表は `table.bs[border=1]`、データ行 `tr[bgcolor="#FFFFFF"]`。金沢の列は
  「着順 / 枠番 / 馬番 / 馬名 / 所属 / 性齢 / 負担重量 / 騎手(所属) / 調教師 / 馬体重 / 差 / タイム / 着差 / 上り3F / 人気」。
  ばんえいは BambaObihiroML が 14 列目=人気・15 列目=単勝オッズとして読んでいるので、上り3F の代わりに単勝列がある可能性が高い（未確認）。
- **DebaTable**: `section.cardTable > table`。1 頭 5 行:
  行1 `tr.tBorder`（`td.courseNum`=枠（同枠 2 頭目は rowspan で省略）、`td.horseNum`=馬番、`a.horseName`、`a.jockeyName`+`span.jockeyarea`、
  `td.odds_weight`=「8.1 (3人気)」、`td.result`=着別成績の入れ子表、過去 5 走）／行2 性齢・毛色・生年月日・「負担重量　騎手成績」
  （ばんえいは「☆ 460　0-0-0-0」の形 — BambaObihiroML）／行3 父・調教師リンク・`td.odds_weight`=馬体重「1061(+26)」／
  行4 母・馬主／行5 母父・生産牧場・`td.info`=変更情報。
- **OddsTanFuku**: `table.odd_popular_table_02`。見出し「枠 / 馬番 / 馬名 / 単勝オッズ / 複勝オッズ(2着払い or 3着払い)[colspan=2] / 性齢 / 馬体重(増減) / 負担重量 / 騎手(所属) / 所属 / 調教師 / 変更情報」。
- **リンク**: 馬 `/KeibaWeb/DataRoom/HorseMarkInfo?k_lineageLoginCode=30043400296`、騎手 `RiderMark?k_riderLicenseNo=30868`、
  調教師 `TrainerMark?k_trainerLicenseNo=11433`。

### netkeiba（他者のパーサーと検索抜粋。実ページは未確認）
- レース情報 `.RaceData01`: 「18:55発走 / 直200m / 天候:晴 / 水分量:1.6」。レース名 `.RaceName`。
- 結果表: データ行に `td.Result_Num > div.Rank`、`td.Horse_Info > span.Horse_Name > a`、払戻は `tr.Tansho` など。
  見出し語（着順・枠・馬番・馬名・性齢・斤量・騎手・タイム・着差・人気・単勝オッズ・厩舎・馬体重(増減)）は中央競馬と同じテンプレートという前提で、ばんえいでは未確認。
- 出馬表: `tr.HorseList`、列は 枠 / 馬番 / 印 / 馬名 / 性齢 / 斤量 / 騎手 / 厩舎 / 馬体重 / 単勝 / 人気。
  単勝オッズは JavaScript で後から入るのが通例（HTML には「---.-」）→ 予測時は keiba.go.jp のオッズを馬番で結合（従来どおり）。

## 4. まだ確認できていないこと（残るリスク）

1. **ばんえいの実ページそのもの**（上記の構造は他場の同一テンプレートと他者のコードからの推定を含む）。
2. keiba.go.jp 成績表の「馬場：」欄の書式（数値か「%」付きか）→ どちらにも対応し、さらに RaceList で補完するので致命的ではない。
3. keiba.go.jp 成績表の 2026 年版レイアウトの見出し文字列（`<th>` に見出し語があれば通常の方式で読める。無ければ予備パーサーで主要列のみ）。
4. netkeiba 結果表の見出し語と、取消・除外・中止の表記、厩舎セルの所属表記。
5. netkeiba が現在 UTF-8 か EUC-JP か（自動判定にしたのでどちらでも読めるはず）。
6. netkeiba 出馬表のオッズが HTML に入っているか（入っていなくても keiba.go.jp で補う）。
7. アクセス制限: netkeiba は短い間隔だと 403/429 を返すとの報告（1.2 秒以上推奨）。既定の 2 秒間隔を短くしないこと。
8. `cli.py` は `netkeiba.Fetcher(encoding="euc-jp")` を渡しているが、今は「宣言が無いときのヒント」なので問題ない（cli.py はこの作業の担当外のため未変更）。
9. netkeiba で水分が取れなかった日は keiba.go.jp の RaceList（1 日 1 リクエスト）で補完する。水分はレース単位の値なので ID 体系は混ざらない。

## 5. 手元での確認手順

1. **1 ページずつ診断する**（URL でも、ブラウザで保存した HTML でもよい）:
   ```bash
   python -m banei_ai.scrape.diagnose "https://nar.netkeiba.com/race/result.html?race_id=202665110101"
   python -m banei_ai.scrape.diagnose "https://nar.netkeiba.com/race/shutuba.html?race_id=202665110201"
   python -m banei_ai.scrape.diagnose "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/RaceList?k_raceDate=2026/11/01&k_babaCode=3"
   python -m banei_ai.scrape.diagnose "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/RaceMarkTable?k_raceDate=2026/11/01&k_babaCode=3&k_raceNo=1"
   python -m banei_ai.scrape.diagnose "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/DebaTable?k_raceDate=2026/11/01&k_babaCode=3&k_raceNo=1"
   python -m banei_ai.scrape.diagnose "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/OddsTanFuku?k_raceDate=2026/11/01&k_babaCode=3&k_raceNo=1"
   python -m banei_ai.scrape.diagnose saved.html --source keibago --kind deba   # 保存ファイル
   ```
   見るところ:
   - `文字コード` と警告「置換文字 (U+FFFD) が … 個」が出ていないこと
   - `[採用した表]` の「見出し → 正規列名」と「見つからなかった正規列」（結果ページで `odds_win` が無いのは keiba.go.jp 過去年なら正常）
   - `馬場水分` とその「読んだ場所」（netkeiba なら `.RaceData01: 水分量:1.6` のように出るはず）
   - `ID 対応率` が馬・騎手・調教師とも 100% 近いこと
   - `status の内訳` に取消・中止がある日は scratch / dnf が数えられていること
2. **1 週間分を取得して欠損率を見る**:
   ```bash
   python -m banei_ai.cli scrape --start 2026-10-25 --end 2026-11-02 --source netkeiba --check
   ```
   `moisture`・`odds_win`・`horse_weight`・`carried_weight` の欠損率がほぼ 0（取消馬を除く）、
   `horse_id`・`jockey_id` が名前ではなく ID（netkeiba は `B…` の英数字、keiba.go.jp は数字）になっていること。
   値の目安: 馬場水分 0.5〜6 % 程度、負担重量 480〜1000 kg、馬体重 800〜1200 kg、タイム 1〜4 分。
3. **ずれていた場合の直し方**:
   - 見出し語が違う → `banei_ai/scrape/common.py` の `HEADER_RULES`
   - 馬場水分の書式が違う → `common.py` の `_MOIST_HEADER_RX`
   - keiba.go.jp 出馬表 → `keibago.parse_deba_table`、レース一覧 → `keibago.parse_race_list`
   - 直したら、そのページの構造を**手で再構成した小さな HTML** を `tests/test_parser.py` に足す
     （実ページの HTML そのものはリポジトリに入れない: 両サイトの利用規約は転載・公開を禁じている）。
