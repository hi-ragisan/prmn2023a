#!/usr/bin/env bash
# 毎日の運用例:
#   1) 昨日までの結果を追加取得
#   2) 月曜 (またはモデル未作成) なら再学習
#   3) 今日の出馬表と現在オッズで予測 (締切直前にもう一度実行するのがおすすめ)
# 使い方: scripts/daily.sh            # 全レース
#         scripts/daily.sh --race 7   # 7R だけ
set -euo pipefail
cd "$(dirname "$0")/.."
python -m banei_ai.cli update
if [ "$(date +%u)" = "1" ] || [ ! -f data/models/model.pkl ]; then
  python -m banei_ai.cli train
fi
# 出馬表の取得元は過去データと同じ netkeiba (ID 体系を揃えるため)。オッズは公式サイトから結合
python -m banei_ai.cli predict --date "$(date +%Y-%m-%d)" --card-source netkeiba "$@"
