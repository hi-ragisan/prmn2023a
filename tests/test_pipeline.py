"""学習 → 当日予測までの通し動作 (シミュレーションデータ)."""
import numpy as np
import pandas as pd
import pytest

from banei_ai.features import build_features
from banei_ai.model import BaneiModel, evaluate_probs, walk_forward
from banei_ai.predict import format_table, predict_card
from banei_ai.simulate import SimConfig, simulate


@pytest.fixture(scope="module")
def data():
    return simulate(SimConfig(start="2023-04-01", end="2024-12-31", seed=3))


def test_walk_forward_and_market_sanity(data):
    X = build_features(data)
    P = walk_forward(X, verbose=False)
    # 予測は完全サンプル外: 学習開始 (270 日) より後だけ
    assert P["date"].min() >= X["date"].min() + pd.Timedelta(days=270)
    assert np.allclose(P.groupby("race_id")["p_model"].sum(), 1.0)
    t = evaluate_probs(P).set_index("model")["logloss"]
    assert t["市場 (オッズそのまま)"] < t["一様 (1/頭数)"]
    # 市場確率 (較正後) だけで期待値 > 1.2 の馬はほぼ見つからない
    assert ((P["p_mkt_cal"] * P["odds_win"]) > 1.2).mean() < 0.01


def test_train_and_predict_card(data):
    last = data["date"].max()
    hist = data[data["date"] < last]
    card = data[data["date"] == last].drop(columns=["finish", "time_sec", "_p_true"])
    X = build_features(hist)
    m = BaneiModel().fit(X)
    P = predict_card(hist, card, m)
    assert len(P) == len(card)
    assert np.allclose(P.groupby("race_id")["p_model"].sum(), 1.0)
    assert np.allclose(P.groupby("race_id")["p_top3"].sum(), 3.0)
    assert (P["plus_factors"].str.len() > 0).mean() > 0.5
    t = format_table(P)
    assert {"AI勝率", "歪み", "期待値", "プラス要因"} <= set(t.columns)
