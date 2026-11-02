"""条件付きロジット (レース内多項ロジット).

  P(馬 i が勝つ | レース r) = exp(x_i·β) / Σ_{j∈r} exp(x_j·β)

競馬の勝率モデルの定番 (Bolton & Chapman 1986, Benter 1994)。
レース内で確率の和が 1 になり、オッズ (市場確率) と自然に合成できる。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _groups(race_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ソート済み race_ids の各グループ開始位置と長さ."""
    change = np.r_[True, race_ids[1:] != race_ids[:-1]]
    starts = np.flatnonzero(change)
    lens = np.diff(np.r_[starts, len(race_ids)])
    return starts, lens


def softmax_by_group(u: np.ndarray, starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    m = np.maximum.reduceat(u, starts)
    e = np.exp(u - np.repeat(m, lens))
    s = np.add.reduceat(e, starts)
    return e / np.repeat(s, lens)


@dataclass
class ConditionalLogit:
    l2: float = 1e-3
    coef_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    columns: list[str] | None = None

    def _prep(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if fit:
            self.mean_ = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
            self.scale_ = np.where(sd > 0, sd, 1.0)
        Z = (X - self.mean_) / self.scale_
        return np.nan_to_num(Z, nan=0.0)  # 欠損 = 平均

    def fit(self, X, y, race_ids, columns: list[str] | None = None,
            weights=None) -> "ConditionalLogit":
        """weights: 行ごとの重み (同じレース内は同じ値。レースの先頭行の値を使う)。省略時は全レース 1."""
        race_ids = np.asarray(race_ids)
        order = np.argsort(race_ids, kind="stable")
        Z = self._prep(np.asarray(X)[order], fit=True)
        y = np.asarray(y, dtype=float)[order]
        starts, lens = _groups(race_ids[order])
        # 勝ち馬がいないレース (全頭中止など) は除外
        has_win = np.add.reduceat(y, starts) > 0
        keep = np.repeat(has_win, lens)
        Z, y = Z[keep], y[keep]
        starts, lens = _groups(race_ids[order][keep])
        # 同着は勝利を等分
        ysum = np.repeat(np.add.reduceat(y, starts), lens)
        y = y / ysum
        n_races = len(starts)
        if weights is not None:
            w_race = np.asarray(weights, dtype=float)[order][keep][starts]
            w_row = np.repeat(w_race, lens)
            w_tot = float(w_race.sum())

        def f(b):
            u = Z @ b
            m = np.maximum.reduceat(u, starts)
            e = np.exp(u - np.repeat(m, lens))
            s = np.add.reduceat(e, starts)
            lse = np.log(s) + m
            p = e / np.repeat(s, lens)
            if weights is None:  # 従来と完全に同じ計算
                nll = -(y @ u - lse.sum()) / n_races + 0.5 * self.l2 * b @ b
                g = -(Z.T @ (y - p)) / n_races + self.l2 * b
            else:
                nll = -((w_row * y) @ u - w_race @ lse) / w_tot + 0.5 * self.l2 * b @ b
                g = -(Z.T @ (w_row * (y - p))) / w_tot + self.l2 * b
            return nll, g

        b0 = np.zeros(Z.shape[1])
        r = minimize(f, b0, jac=True, method="L-BFGS-B")
        self.coef_ = r.x
        self.columns = columns
        self.nll_ = r.fun
        return self

    def utility(self, X) -> np.ndarray:
        return self._prep(np.asarray(X), fit=False) @ self.coef_

    def predict_proba(self, X, race_ids) -> np.ndarray:
        race_ids = np.asarray(race_ids)
        order = np.argsort(race_ids, kind="stable")
        u = self.utility(np.asarray(X)[order])
        starts, lens = _groups(race_ids[order])
        p = softmax_by_group(u, starts, lens)
        out = np.empty_like(p)
        out[order] = p
        return out

    def coef_table(self) -> pd.DataFrame:
        cols = self.columns or [f"x{i}" for i in range(len(self.coef_))]
        return pd.DataFrame({"feature": cols, "coef_std": self.coef_,
                             "coef_raw": self.coef_ / self.scale_})


def race_logloss(p: np.ndarray, y: np.ndarray, race_ids: np.ndarray) -> float:
    """勝ち馬に付けた確率の平均対数損失 (1 レースあたり)."""
    y = np.asarray(y)
    mask = y == 1
    rid = np.asarray(race_ids)[mask]
    # 同着レースは平均
    s = pd.Series(-np.log(np.clip(np.asarray(p)[mask], 1e-9, 1))).groupby(rid).mean()
    return float(s.mean())


def harville_top2(p: np.ndarray) -> np.ndarray:
    """Harville 式で 2 着以内確率."""
    p = np.asarray(p, dtype=float)
    if len(p) <= 2:
        return np.ones(len(p))
    out = p.copy()
    for i in range(len(p)):
        for j in range(len(p)):
            if j != i and p[j] < 1:
                out[i] += p[j] * p[i] / (1 - p[j])
    return np.clip(out, 0, 1)


def place_prob(p: np.ndarray) -> np.ndarray:
    """複勝が的中する確率. 8 頭以上: 3 着以内 / 5〜7 頭: 2 着以内 / 4 頭以下: 発売なし (NaN)."""
    n = len(p)
    if n <= 4:
        return np.full(n, np.nan)
    return harville_top3(p) if n >= 8 else harville_top2(p)


def harville_top3(p: np.ndarray) -> np.ndarray:
    """Harville 式で 3 着以内確率を計算 (複勝の期待値用)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n <= 3:
        return np.ones(n)
    out = p.copy()
    for i in range(n):
        acc = 0.0
        for j in range(n):
            if j == i:
                continue
            pj = p[j]
            rem1 = 1 - pj
            # 2着が i
            acc += pj * p[i] / rem1
            for k in range(n):
                if k in (i, j):
                    continue
                pk = p[k]
                rem2 = rem1 - pk
                if rem2 <= 0:
                    continue
                # 1着 j, 2着 k, 3着 i
                acc += pj * (pk / rem1) * (p[i] / rem2)
        out[i] += acc
    return np.clip(out, 0, 1)
