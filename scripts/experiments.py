"""README の比較実験 (Stage1 の学習方式 / Stage2 の市場系特徴量) を再現する.

  python scripts/experiments.py                  # シミュレーションの 3 シード
  python scripts/experiments.py --data data/interim/races.csv   # 実データで比較

実データで回すときは、シミュレーションと違って「答え」を知らないので、
ここで良かった設定をそのまま採用してよい (ただし期間を分けて再確認すること)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from banei_ai import model as M  # noqa: E402
from banei_ai import schema  # noqa: E402
from banei_ai.features import build_features  # noqa: E402
from banei_ai.simulate import SimConfig, simulate  # noqa: E402

STAGE1_OBJECTIVES = ["lambdarank", "rank_xendcg", "binary"]
STAGE2_SETS = {
    "s1+市場のみ": [],
    "+騎手人気度・前走市場シェア (既定)": ["j_popularity", "h_last_pmkt_share"],
    "+前走着順・馬体重増減": ["j_popularity", "h_last_pmkt_share", "h_last1_pct", "horse_weight_diff"],
}


def run(X: pd.DataFrame, objective: str, extra: list[str]) -> dict:
    orig_fit, orig_extra = M.Stage1.fit, list(M.STAGE2_MARKET_EXTRA)

    def fit(self, d):
        self.params = {**self.params, "objective": objective}
        if objective == "binary":
            import lightgbm as lgb

            d = d.sort_values(["date", "race_id"])
            self.features = [c for c in self.features if c in d.columns and d[c].notna().any()]
            ds = lgb.Dataset(d[self.features].astype(float), label=d["win"])
            self.booster = lgb.train(self.params, ds, num_boost_round=self.n_rounds)
            return self
        return orig_fit(self, d)

    M.Stage1.fit = fit
    M.STAGE2_MARKET_EXTRA[:] = extra
    try:
        P = M.walk_forward(X, verbose=False)
    finally:
        M.Stage1.fit = orig_fit
        M.STAGE2_MARKET_EXTRA[:] = orig_extra
    ev = M.evaluate_probs(P).set_index("model")["vs_market"]
    sel = P[P["p_model"] * P["odds_win"] >= 1.1]
    return {"fund_vs_mkt": ev["ファンダメンタル"], "model_vs_mkt": ev["合成モデル"],
            "bets_ev1.1": len(sel), "roi_ev1.1": (sel["win"] * sel["odds_win"]).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--seeds", default="42,7,123")
    a = ap.parse_args()
    if a.data:
        datasets = {"real": build_features(schema.normalize(pd.read_csv(a.data, dtype={"race_id": str})))}
    else:
        datasets = {f"sim{s}": build_features(simulate(SimConfig(end="2025-09-30", seed=int(s))))
                    for s in a.seeds.split(",")}
    rows = []
    for name, X in datasets.items():
        for obj in STAGE1_OBJECTIVES:
            rows.append({"data": name, "exp": f"stage1={obj}", **run(X, obj, STAGE2_SETS["+騎手人気度・前走市場シェア (既定)"])})
            print(rows[-1], flush=True)
        for label, extra in STAGE2_SETS.items():
            rows.append({"data": name, "exp": f"stage2={label}", **run(X, "rank_xendcg", extra)})
            print(rows[-1], flush=True)
    print(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()
