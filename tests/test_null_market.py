"""帰無仮説の検証: 歪みのない市場で、パイプラインが「エッジ」をでっち上げないこと.

SimConfig.preset("efficient") の市場は、真の強さ (モデルが見られない調教の気配なども含む) を
完全に知り、実際の勝敗に対して較正した確率を付ける (埋め込んだ歪み 0, market_info_sd = 0)。
モデルの特徴量はすべて過去の結果から作る「真の強さの粗い代理」なので、市場に足せる情報は原理的にゼロ。
よってパイプラインが正しければ
  1. 合成モデルの対数損失は、市場より意味のある幅で良くなってはいけない
  2. 期待値 ≥ 1.10 の賭けの回収率が「有意に 100% 超え」(90% 区間の下限 > 1) になってはいけない
  3. 選ばれた賭けの *真の* 期待値 (シミュレーションの真の勝率 × オッズ) は 1 を下回る (控除 20% → 約 0.8)

しきい値の根拠 (scratch 実験: 2023-04〜2024-12, サンプル外 約 836 レース/シード, seed 3/42/7/11/5/1。
※ すべて人工データ上の数値):
  Δ対数損失 (市場 − 合成, 正 = モデルが良い): −0.0080〜+0.0001 (平均 −0.0031, シード間 sd 0.003)。
      1 シード内のレース単位の標準誤差 ≈ 0.003〜0.004。しきい値 +0.005 は帰無分布の平均 +2.5sd 超で、
      歪みを埋め込んだ既定の市場 (同じ seed 3) の +0.027 の 1/5 以下。
  期待値 ≥ 1.10 の賭け: 0〜45 点/シード。90% 区間の下限は最大 0.85 (seed 42)、seed 3 は 0.22。
      実現回収率は運に大きく振れる (seed 42 は 12 点で 251%、6 シード合算 68 点で 117%、9 勝 vs 真の期待 4.8 勝)。
      このため「回収率の上限」ではなく「下限 < 1 (有意なエッジを主張しない)」と、運に左右されない
      「真の期待値 < 1」で判定する。区間の上限は点数が少ないと 400% を超えることもあり判定に使えない。
  選ばれた賭けの真の期待値: 0.73〜0.79 (モデル自身の見積もりは 1.14〜1.20 = 選択バイアス/勝者の呪い)。
  リーク検査 (6 シード合算の条件付きロジット): 真の勝率を入れた上での s1 の追加係数 0.06 ± 0.05 (有意でない)。

検出力の確認として、同じ seed で歪みを埋め込んだ既定の市場では Δ対数損失がはっきり正になることも見る
(パイプラインが壊れて「常に市場をそのまま返す」状態でも帰無検定は通ってしまうため)。
"""
import numpy as np
import pandas as pd
import pytest

from banei_ai import backtest as B
from banei_ai import config, market
from banei_ai.features import build_features, prepare
from banei_ai.model import evaluate_probs, walk_forward
from banei_ai.simulate import SimConfig, simulate

PERIOD = dict(start="2023-04-01", end="2024-12-31", seed=3)


def _true_p(P: pd.DataFrame) -> pd.Series:
    return P["_p_true"] / P.groupby("race_id")["_p_true"].transform("sum")


def _dll(P: pd.DataFrame) -> tuple[float, float]:
    t = evaluate_probs(P).set_index("model")["logloss"]
    return (t["市場 (オッズそのまま)"] - t["合成モデル"], t["市場 (較正後)"] - t["合成モデル"])


@pytest.fixture(scope="module")
def null_preds():
    return walk_forward(build_features(simulate(SimConfig.preset("efficient", **PERIOD))), verbose=False)


# ---------------------------------------------------------------- シミュレーター自体の性質 (速い)

def test_default_output_unchanged():
    """既定の出力は歪みスケール導入前 (fbab895) と完全に同じ (既存テスト・README の数値の前提)."""
    kw = dict(start="2023-04-01", end="2023-07-31", seed=42)
    a = simulate(SimConfig(**kw))
    # fbab895 の simulate() で取った指紋 (numpy の乱数列が変わらない限り不変)
    assert (len(a), a["race_id"].nunique(), int(a["finish"].sum())) == (4737, 476, 25815)
    assert a["odds_win"].sum() == pytest.approx(67160.6, abs=1e-6)
    explicit = SimConfig(distortion=1.0, d_flb=1.0, d_fame=1.0, d_overreact=1.0, d_moist=1.0,
                         d_weight_drop=1.0, payout_rate=config.WIN_PAYOUT_RATE, **kw)
    for cfg in (SimConfig.preset("default", **kw), explicit):
        pd.testing.assert_frame_equal(a, simulate(cfg), check_exact=True)


def test_presets_share_race_outcomes():
    """同じ seed なら、市場の設定を変えてもレース結果は同じでオッズだけが変わる (共通乱数)."""
    kw = dict(start="2023-04-01", end="2023-07-31", seed=5)
    a = simulate(SimConfig(**kw))
    for cfg in (SimConfig.preset("efficient", **kw), SimConfig(distortion=0.5, **kw)):
        b = simulate(cfg)
        changed = {c for c in a.columns if not a[c].equals(b[c])}
        assert changed == {"odds_win", "popularity"}


def test_efficient_market_is_calibrated_and_default_is_not():
    """効率的な市場は実際の勝敗に対して較正済み (p ∝ p_mkt^α で α ≈ 1)、既定は大穴過剰人気 (α > 1)."""
    kw = dict(start="2023-04-01", end="2024-09-30", seed=11)
    eff = prepare(simulate(SimConfig.preset("efficient", **kw)))
    dft = prepare(simulate(SimConfig(**kw)))
    # 4 シード × 3 年で α = 0.99〜1.02 (効率的) / 1.19〜1.27 (既定)。1.5 年 1 シードなので幅を持たせる
    assert 0.92 < market.fit_power_calibration(eff) < 1.08
    assert market.fit_power_calibration(dft) > 1.12
    # 全馬を均等に買った回収率 ≈ 払戻率 (歪みが無ければどの馬も期待値 ≈ 0.8)
    assert 0.72 < (eff["win"] * eff["odds_win"]).mean() < 0.88
    # 効率的な市場では、どの馬も真の期待値 (真の勝率 × オッズ) は 1 未満
    assert (_true_p(eff) * eff["odds_win"]).max() < 1.0


def test_realistic_preset_matches_real_world_stats():
    """realistic は L1 調査の実データ統計 (帯広) に概ね合う. 範囲は 1 年 1 シードのぶれを見込んだもの."""
    d = prepare(simulate(SimConfig.preset("realistic", start="2023-04-01", end="2024-03-31", seed=42)))
    d = d[d["p_mkt"].notna()]
    fav = d.loc[d.groupby("race_id")["odds_win"].idxmin()]
    assert 10.5 < d["race_id"].nunique() / d["date"].nunique() <= 11       # 実: 11R/日
    assert 9.3 < d.groupby("race_id").size().mean() < 9.9                  # 実: 推定 9.3〜9.8 (未確認)
    assert 0.30 < fav["win"].mean() < 0.42                                 # 実: 34〜37%
    assert 600 < d.loc[d["win"] == 1, "odds_win"].mean() * 100 < 750       # 実: 平均配当 661 円


# ---------------------------------------------------------------- 帰無仮説の検定 (ウォークフォワード)

def test_null_market_no_logloss_edge(null_preds):
    d_mkt, d_cal = _dll(null_preds)
    assert d_mkt < 0.005, f"効率的な市場に対して合成モデルが {d_mkt:+.4f} 良い (偽のエッジ?)"
    assert d_cal < 0.005, f"較正後の市場に対して合成モデルが {d_cal:+.4f} 良い (偽のエッジ?)"


def test_null_market_no_significant_roi(null_preds):
    bets = B.BetRule(min_ev=1.10).select(null_preds)
    if len(bets):
        lo, hi = B.bootstrap_roi(bets)
        assert lo < 1.0, f"効率的な市場で回収率 90% 区間の下限が {lo:.2f} > 1 (偽のエッジ)"
        # 運に左右されない判定: 選ばれた賭けの真の期待値は 1 未満
        true_ev = float((_true_p(null_preds)[bets.index] * bets["odds_win"]).mean())
        assert true_ev < 1.0
        # モデルは期待値 ≥ 1.1 と思って買っている → 真の期待値との差が選択バイアス
        assert bets["ev"].mean() > true_ev
    sh = B.split_half_check(null_preds)
    if sh.get("second_half_bets"):
        assert sh["second_half_roi_lo90"] < 1.0, "前半で選んだしきい値が後半でも有意に勝っている (偽のエッジ)"


def test_pipeline_detects_planted_distortions():
    """検出力: 同じ seed・期間で歪みを埋め込んだ既定の市場なら、はっきり市場に勝つ.

    実測 (seed 3): Δ対数損失 vs 市場 +0.0270, vs 較正後の市場 +0.0103, 期待値 ≥ 1.1 が 159 点で真の期待値 1.11.
    """
    P = walk_forward(build_features(simulate(SimConfig(**PERIOD))), verbose=False)
    d_mkt, d_cal = _dll(P)
    assert d_mkt > 0.015 and d_cal > 0.005  # 帰無のしきい値 0.005 を上回る
    bets = B.BetRule(min_ev=1.10).select(P)
    assert len(bets) > 50
    assert float((_true_p(P)[bets.index] * bets["odds_win"]).mean()) > 1.0  # 本物のエッジ
