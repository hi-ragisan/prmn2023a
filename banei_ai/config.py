"""プロジェクト全体の定数・パス."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # スクレイピングした HTML キャッシュ
INTERIM_DIR = DATA_DIR / "interim"  # パース済み CSV
MODEL_DIR = DATA_DIR / "models"
REPORT_DIR = DATA_DIR / "reports"

# 帯広競馬場 (ばんえい) の場コード
NETKEIBA_BANEI_CODE = "65"   # nar.netkeiba.com の race_id で使う場コード
KEIBAGO_BANEI_CODE = "3"     # keiba.go.jp の k_babaCode

# 単勝の払戻率 (地方競馬の単勝は概ね 80%)。市場確率は各レースで正規化するので
# 主に「控除を考慮した理論期待値」の参考値として使う。
WIN_PAYOUT_RATE = 0.80

# 馬場水分の区分 (ばんえいは平地競馬と用語が逆: 水分が高いほど「軽馬場」=時計が速い、
# 乾いているほど「重馬場」=時計がかかる。docs/research_banei.md 参照)
MOISTURE_BINS = [0.0, 1.5, 2.5, 4.0, 100.0]
MOISTURE_LABELS = ["重(乾燥)", "標準", "やや軽", "軽"]

# 時間減衰の半減期 (日)
JOCKEY_HALF_LIFE_DAYS = 240
TRAINER_HALF_LIFE_DAYS = 365
HORSE_FORM_HALF_LIFE_DAYS = 120

# 経験ベイズの事前分布の強さ (擬似レース数)
JOCKEY_PRIOR_STRENGTH = 60.0
TRAINER_PRIOR_STRENGTH = 80.0
COMBO_PRIOR_STRENGTH = 8.0

RANDOM_SEED = 42


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, INTERIM_DIR, MODEL_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
