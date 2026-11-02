"""HTML ダッシュボードのスモークテスト (小さなシミュレーション・学習なしで高速に).

ウォークフォワード (LightGBM) は重いので、ここでは市場確率と真の確率を混ぜた
「それらしい予測」P を作って描画だけを検査する。
"""
import re
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from banei_ai import backtest as B
from banei_ai import dashboard as D
from banei_ai import schema
from banei_ai.features import build_features
from banei_ai.predict import format_table
from banei_ai.simulate import SimConfig, simulate

MAIN_SECTIONS = ["summary", "calibration", "threshold", "pnl", "distortion", "jockey", "horse", "today"]


@pytest.fixture(scope="module")
def sim():
    df = simulate(SimConfig(start="2023-04-01", end="2023-09-30", seed=11, n_horses=260, races_per_day=5))
    return schema.normalize(df)


@pytest.fixture(scope="module")
def feats(sim):
    return build_features(sim)


def _norm(s: pd.Series, rid: pd.Series) -> pd.Series:
    return s / s.groupby(rid).transform("sum")


def _fake_preds(X: pd.DataFrame) -> pd.DataFrame:
    """walk_forward の出力と同じ列を持つ、学習なしの予測."""
    P = X[(X["date"] >= X["date"].min() + pd.Timedelta(days=45)) & X["p_mkt"].notna()].copy()
    P["s1"] = np.log(P["_p_true"])
    P["block"] = 0
    P["p_fund"] = _norm(P["_p_true"] ** 0.7, P["race_id"])
    P["p_mkt_cal"] = P["p_mkt"]
    P["p_model"] = _norm(0.5 * P["p_mkt"] + 0.5 * P["_p_true"], P["race_id"])
    return P.reset_index(drop=True)


def _pred_csv(X: pd.DataFrame, path) -> pd.DataFrame:
    """predict.format_table と同じ形式の当日予測 CSV."""
    d = X[X["date"] == X["date"].max()].copy()
    d["p_model"] = _norm(0.5 * d["p_mkt"] + 0.5 * d["_p_true"], d["race_id"])
    d["p_fund"] = d["_p_true"]
    d["ev"] = d["p_model"] * d["odds_win"]
    d["edge"] = d["p_model"] / d["p_mkt"] - 1
    d["fair_odds"] = 1 / d["p_model"]
    d["bet"] = d["ev"] >= 1.1
    d.loc[d["ev"].idxmax(), "bet"] = True  # 少なくとも 1 頭は推奨
    d["kelly_stake"] = np.where(d["bet"], 500.0, 0.0)
    d["plus_factors"] = "騎手の腕・近走着順"
    d["minus_factors"] = "負担重量"
    t = format_table(d)
    t.to_csv(path, index=False, encoding="utf-8-sig")
    return D.read_pred_csv(path)


class _TagChecker(HTMLParser):
    """開始/終了タグの対応を確かめる (自己終了の SVG 要素は対象外)."""

    VOID = {"meta", "br", "img", "input", "link", "hr", "col", "area", "base", "source", "wbr"}

    def __init__(self):
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.errors.append(f"</{tag}> (open: {self.stack[-3:]})")


def _check_html(doc: str) -> None:
    assert doc.lstrip().startswith("<!DOCTYPE html>")
    assert doc.rstrip().endswith("</html>")
    assert '<meta charset="utf-8">' in doc and 'name="viewport"' in doc
    p = _TagChecker()
    p.feed(doc)
    p.close()
    assert not p.errors, p.errors[:5]
    assert not p.stack, p.stack
    # 閲覧時に通信しない (外部 CSS/JS/画像なし)
    assert not re.search(r'(src|href)\s*=\s*"(https?:)?//', doc)
    assert "<link" not in doc
    assert "prefers-color-scheme:dark" in doc.replace(" ", "")


def _h2(doc: str) -> list[str]:
    return re.findall(r"<h2[^>]*>(.*?)</h2>", doc)


def test_full_dashboard(sim, feats, tmp_path, monkeypatch):
    P = _fake_preds(feats)
    pred = _pred_csv(feats, tmp_path / "pred_20230930.csv")

    def fake_robustness(P, rule=None, **kw):
        b = (rule or B.BetRule()).select(P)
        r = B.flat_bet(b)
        return pd.DataFrame([{"label": f"pool={p:,} noise=0.10 stake=100", "pool_yen": p, "odds_noise": 0.1,
                              "stake": 100, "bets": r["bets"], "hit_rate": r["hit_rate"], "roi": r["roi"] - k,
                              "roi_lo90": r["roi"] - k - 0.2, "roi_hi90": r["roi"] - k + 0.2, "profit": 0.0}
                             for k, p in [(0.0, 0), (0.1, 1_000_000)]])

    monkeypatch.setattr(B, "robustness_table", fake_robustness, raising=False)
    doc = D.build_dashboard(sim, P, pred, X=feats, pred_date="2023-09-30", min_rides=20)
    _check_html(doc)
    heads = _h2(doc)
    for sid in MAIN_SECTIONS + ["robust"]:
        assert D.SECTION_TITLES[sid] in heads
    assert D.SIM_BANNER in doc
    # 各節が省略されずに描画されている
    assert "省略" not in doc, re.findall(r'<p class="note">(.*?)</p>', doc)
    assert doc.count("<svg") >= 10
    assert "data-xhair" in doc  # 累積損益のクロスヘア
    assert "推奨買い目" in doc and 'class="pick"' in doc
    assert "市場比 A/E" in doc  # 騎手表
    assert "固有の腕" in doc and "計算できませんでした" not in doc
    assert "プール 100万円" in doc  # 頑健性 (robustness_table があるときだけの節)


def test_cli_writes_file(sim, feats, tmp_path):
    data = tmp_path / "sim.csv"
    sim.to_csv(data, index=False)
    preds = tmp_path / "wf.pkl"
    _fake_preds(feats).to_pickle(preds)
    pred_csv = tmp_path / "pred_20230930.csv"
    _pred_csv(feats, pred_csv)
    out = tmp_path / "out" / "dashboard.html"
    D.main(["--data", str(data), "--preds", str(preds), "--pred-csv", str(pred_csv), "--out", str(out),
            "--min-rides", "20"])
    assert out.exists()
    doc = out.read_text(encoding="utf-8")
    _check_html(doc)
    for sid in MAIN_SECTIONS:
        assert D.SECTION_TITLES[sid] in _h2(doc)
    assert D.SIM_BANNER in doc
    assert "2023-09-30" in doc  # 予測 CSV のファイル名から日付


def test_missing_pieces_are_skipped(sim):
    """オッズなし・予測なし・期間が短すぎてウォークフォワード不可でも、落ちずに出力する."""
    d = sim[sim["date"] < "2023-08-01"].drop(columns=["_p_true"]).copy()
    d["odds_win"] = np.nan
    d["popularity"] = np.nan
    doc = D.build_dashboard(d, None, None, compute_walkforward=True, min_rides=20)
    _check_html(doc)
    heads = _h2(doc)
    for sid in MAIN_SECTIONS:
        assert D.SECTION_TITLES[sid] in heads
    assert D.SIM_BANNER not in doc  # _p_true が無ければ実データ扱い
    notes = re.findall(r'<p class="note">(.*?)</p>', doc)
    assert any("ウォークフォワード" in n for n in notes)
    assert any("オッズが無い" in n for n in notes)
    assert any("予測 CSV" in n for n in notes)
    # 騎手・馬はオッズが無くても出る
    assert "頭数比 A/E" in doc and "レート" in doc


def test_few_bets_and_formatting():
    """購入点数 0 でも KPI・グラフが壊れない. 書式関数は NaN を — にする."""
    rid = np.repeat([f"r{i}" for i in range(40)], 5)
    P = pd.DataFrame({
        "race_id": rid, "date": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.repeat(range(40), 5), "D"),
        "post": np.tile(range(1, 6), 40), "odds_win": 4.0, "p_mkt": 0.2, "log_p_mkt": np.log(0.2),
        "p_mkt_cal": 0.2, "p_fund": 0.2, "p_model": 0.2, "win": np.tile([1, 0, 0, 0, 0], 40)})
    ctx = {"P": P, "rule": B.BetRule(), "X": None, "pred": None, "days": 365, "min_rides": 50, "p_error": ""}
    html = D._section("summary", D.section_summary, ctx)
    assert "購入対象がありません" in html
    assert "省略" in D._section("pnl", D.section_pnl, ctx)
    assert "<svg" in D._section("threshold", D.section_threshold, ctx)
    assert D._pct(np.nan) == "—" and D._yen(float("nan")) == "—"
    assert D._pct(0.1234) == "12.3%" and D._yen(1234.0) == "+1,234円"
