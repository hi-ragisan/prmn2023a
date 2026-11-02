"""馬券シミュレーション (単勝).

注意:
  * 学習データのオッズは「確定オッズ」。実際に買う時点 (締切前) のオッズとはズレる。
  * ばんえいは売上 (プール) が小さく、自分の購入でオッズが下がる。大口購入ほど期待値は目減りする。
  * よってバックテストの回収率は「上限の目安」と考えること。

現実性の補正 (後半の「現実性」節, docs/backtest_realism.md):
  * 自分の購入による払戻低下 (パリミュチュエル): `impact_odds`, `flat_bet(pool_yen=...)`, `kelly_bet(pool_yen=...)`
  * 締切前オッズ → 確定オッズのズレ: `preclose_view` (判断は締切前オッズ、払戻は確定オッズ)
  * まとめ: `robustness_table`, `breakeven_table`, `kelly_robustness`
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class BetRule:
    min_ev: float = 1.10        # 期待値 p × オッズ の下限
    min_prob: float = 0.03      # 勝率の下限 (超大穴の過信を避ける)
    max_odds: float = 50.0      # オッズの上限
    min_edge: float = 0.0       # p_model / p_mkt - 1 の下限 (歪みの大きさ)
    max_per_race: int = 2       # 1 レースで買う最大頭数 (EV 上位から)

    def select(self, P: pd.DataFrame, p_col: str = "p_model") -> pd.DataFrame:
        d = P[P["odds_win"].notna()].copy()
        d["ev"] = d[p_col] * d["odds_win"]
        d["edge"] = d[p_col] / d["p_mkt"] - 1
        m = (
            (d["ev"] >= self.min_ev)
            & (d[p_col] >= self.min_prob)
            & (d["odds_win"] <= self.max_odds)
            & (d["edge"] >= self.min_edge)
        )
        d = d[m].sort_values(["race_id", "ev"], ascending=[True, False])
        d = d.groupby("race_id").head(self.max_per_race)
        return d


def _max_drawdown(cum: np.ndarray) -> float:
    if len(cum) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.r_[0.0, cum])
    return float((peak[1:] - cum).max())


ODDS_TICKS = 10  # 払戻は 10 円単位で切り捨て → 単勝オッズは 0.1 刻み (最低 1.0 = 元返し)


def floor_odds(x) -> np.ndarray:
    """払戻の端数切り捨て: 0.1 刻みに切り下げ、最低 1.0."""
    x = np.asarray(x, dtype=float)
    return np.maximum(1.0, np.floor(x * ODDS_TICKS + 1e-9) / ODDS_TICKS)


def impact_odds(odds, stakes, race_ids, pool_yen, payout_rate: float = config.WIN_PAYOUT_RATE) -> np.ndarray:
    """自分の購入を加えた後の実効オッズ (パリミュチュエル).

    確定オッズ o_i, 単勝の総売上 T (pool_yen), 払戻率 R のとき、馬 i への票は W_i = R·T / o_i。
    自分が馬 i に s_i 円を足すと (同じレースの自分の購入はすべて総売上に入る)
        o_i' = floor_0.1( R·(T + Σ_j s_j) / (W_i + s_i) ),  最低 1.0
    表示オッズ o_i 自体が切り捨て後の値なので、真の比 R·T/W_i は [o_i, o_i+0.1) のどこか。
    その中点 o_i + 0.05 を使う (s=0 なら o_i をそのまま再現し、1 円でも買うと 1 刻み下がるという偏りを避ける)。

    odds, stakes, race_ids は購入した行ごとの配列 (stakes はスカラー可)。pool_yen はスカラーか行ごとの配列。
    pool_yen <= 0 の行は影響なし (odds をそのまま返す)。
    """
    odds = np.asarray(odds, dtype=float)
    s = np.broadcast_to(np.asarray(stakes, dtype=float), odds.shape)
    T = np.broadcast_to(np.asarray(pool_yen, dtype=float), odds.shape)
    if odds.size == 0 or not np.any(T > 0):
        return odds.copy()
    codes = pd.factorize(np.asarray(race_ids))[0]
    ours = np.bincount(codes, weights=s)[codes]  # 同じレースで自分が買った合計
    with np.errstate(divide="ignore", invalid="ignore"):
        W = payout_rate * T / (odds + 0.5 / ODDS_TICKS)
        eff = floor_odds(payout_rate * (T + ours) / (W + s))
    return np.where(T > 0, eff, odds)


def _impact_race(odds: np.ndarray, s: np.ndarray, T: np.ndarray, payout_rate: float) -> np.ndarray:
    """impact_odds の 1 レース分 (ケリーのループ用に高速化。式は同じ)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        W = payout_rate * T / (odds + 0.5 / ODDS_TICKS)
        eff = floor_odds(payout_rate * (T + s.sum()) / (W + s))
    return np.where(T > 0, eff, odds)


def optimal_stake(p, odds, pool_yen, payout_rate: float = config.WIN_PAYOUT_RATE) -> tuple[np.ndarray, np.ndarray]:
    """1 点だけ買うときの (利益最大の賭け金, 損益分岐の賭け金). 端数切り捨ては無視した連続近似.

    期待収支 f(s) = s·(p·o'(s) − 1), o'(s) = R(T+s)/(W+s), W = R·T/o (その馬に入っている票)。
      損益分岐 s* = (p·R·T − W) / (1 − p·R)          (= W·(EV−1)/(1−p·R))
      利益最大 s_opt = W·(√(1 + s*/W) − 1)          (s* が W より十分小さければ ≈ s*/2)
    EV = p·o ≤ 1 なら 0。
    """
    p = np.asarray(p, dtype=float)
    o = np.asarray(odds, dtype=float)
    T = np.asarray(pool_yen, dtype=float)
    W = payout_rate * T / o
    be = np.maximum((p * payout_rate * T - W) / (1 - p * payout_rate), 0.0)
    opt = W * (np.sqrt(1 + be / W) - 1)
    return opt, be


def _pool_arg(b: pd.DataFrame, pool_yen):
    """pool_yen: スカラー or 列名 (レースごとの単勝売上がデータにある場合)."""
    if isinstance(pool_yen, str):
        return b[pool_yen].fillna(0).to_numpy(dtype=float)
    return 0.0 if pool_yen is None else pool_yen


def flat_bet(bets: pd.DataFrame, stake: float = 100.0, pool_yen=0.0,
             payout_rate: float = config.WIN_PAYOUT_RATE, pay_col: str = "odds_win") -> dict:
    """均等買い.

    pool_yen > 0 なら自分の購入による払戻低下 (impact_odds) を反映する。
    pay_col は払戻に使うオッズの列 (締切前オッズで判断する場合は確定オッズの列 "odds_final")。
    既定 (pool_yen=0, pay_col="odds_win") は従来どおり。
    """
    if bets.empty:
        return {"bets": 0, "hits": 0, "hit_rate": np.nan, "roi": np.nan,
                "profit": 0.0, "max_dd": 0.0, "avg_odds": np.nan}
    b = bets.sort_values(["date", "race_id"])
    pool = _pool_arg(b, pool_yen)
    realistic = np.any(np.asarray(pool) > 0) or pay_col != "odds_win"
    odds = b[pay_col]
    if realistic:
        odds = pd.Series(impact_odds(odds, stake, b["race_id"], pool, payout_rate), index=b.index)
    ret = b["win"] * odds * stake
    pnl = ret - stake
    cum = pnl.cumsum().to_numpy()
    out = {
        "bets": len(b),
        "hits": int(b["win"].sum()),
        "hit_rate": float(b["win"].mean()),
        "roi": float(ret.sum() / (stake * len(b))),
        "profit": float(pnl.sum()),
        "max_dd": _max_drawdown(cum),
        "avg_odds": float(b["odds_win"].mean()),
    }
    if realistic:
        out["avg_paid_odds"] = float(odds.mean())
    return out


def kelly_bet(bets: pd.DataFrame, p_col: str = "p_model", bankroll: float = 100_000.0,
              fraction: float = 0.25, cap: float = 0.03, pool_yen=0.0,
              payout_rate: float = config.WIN_PAYOUT_RATE, pay_col: str = "odds_win",
              max_stake: float | None = None, detail: bool = False) -> dict:
    """分数ケリー: 賭け率 f = fraction × (p·o − 1)/(o − 1), 1 点あたり資金の cap まで.

    賭け金は判断時のオッズ odds_win で決め、払戻は pay_col のオッズに自分の購入の影響
    (pool_yen > 0 のとき impact_odds) を加えたもので計算する。資金が増えるほど賭け金が増え、影響も大きくなる。
    max_stake: 1 点あたりの上限 (円)。detail=True で賭け金・実効オッズの統計も返す。
    """
    if bets.empty:
        out = {"final_bankroll": bankroll, "growth": 1.0, "max_dd_pct": 0.0}
        if detail:
            out.update({"bets_placed": 0, "total_staked": 0.0, "mean_stake": np.nan,
                        "max_stake": 0.0, "last_q_mean_stake": np.nan, "paid_odds_ratio": np.nan})
        return out
    b = bets.sort_values(["date", "race_id"])
    p = b[p_col].to_numpy(dtype=float)
    o = b["odds_win"].to_numpy(dtype=float)
    pay = b[pay_col].to_numpy(dtype=float)
    win = b["win"].to_numpy(dtype=float)
    pool = np.broadcast_to(np.asarray(_pool_arg(b, pool_yen), dtype=float), o.shape)
    impact = bool(np.any(pool > 0))
    key = pd.MultiIndex.from_arrays([b["date"], b["race_id"]])
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(b)]
    with np.errstate(divide="ignore", invalid="ignore"):
        f_all = np.where(o > 1, fraction * (p * o - 1) / (o - 1), 0.0)
    bk = bankroll
    peak = bk
    max_dd = 0.0
    stakes = np.zeros(len(b))
    paid = pay.copy()
    for a, z in zip(starts, ends):
        base = bk  # 同一レース内は同じ資金を基準
        pnl = 0.0
        st = np.floor(np.minimum(np.maximum(f_all[a:z], 0), cap) * base / 100) * 100
        if max_stake is not None:
            st = np.minimum(st, np.floor(max_stake / 100) * 100)
        po = pay[a:z]
        if impact:
            po = _impact_race(po, st, pool[a:z], payout_rate)
        for k in range(z - a):
            pnl += st[k] * (po[k] * win[a + k] - 1)
        stakes[a:z] = st
        paid[a:z] = po
        bk += pnl
        peak = max(peak, bk)
        max_dd = max(max_dd, (peak - bk) / peak)
    out = {"final_bankroll": bk, "growth": bk / bankroll, "max_dd_pct": max_dd}
    if detail:
        placed = stakes > 0
        t = b["date"].to_numpy().astype("int64")
        q = t >= np.quantile(t, 0.75)  # 最後の 1/4 の期間 (資金が増えた後の賭け金)
        out.update({
            "bets_placed": int(placed.sum()),
            "total_staked": float(stakes.sum()),
            "mean_stake": float(stakes[placed].mean()) if placed.any() else np.nan,
            "max_stake": float(stakes.max()),
            "last_q_mean_stake": float(stakes[placed & q].mean()) if (placed & q).any() else np.nan,
            # 賭け金加重の (実際に受け取るオッズ / 自分が買わなかった場合の確定オッズ)
            "paid_odds_ratio": float((stakes * paid / pay).sum() / stakes.sum()) if placed.any() else np.nan,
        })
    return out


def threshold_grid(P: pd.DataFrame, p_col: str = "p_model",
                   evs=(1.0, 1.05, 1.1, 1.2, 1.3, 1.5)) -> pd.DataFrame:
    rows = []
    for ev in evs:
        rule = BetRule(min_ev=ev)
        b = rule.select(P, p_col)
        r = {"min_ev": ev, **flat_bet(b)}
        r.update(kelly_bet(b, p_col))
        rows.append(r)
    return pd.DataFrame(rows)


def split_half_check(P: pd.DataFrame, p_col: str = "p_model",
                     evs=(1.0, 1.05, 1.1, 1.2, 1.3), min_bets: int = 30) -> dict:
    """前半でしきい値を選び、後半にそのまま適用 (しきい値選びの楽観バイアスを除いた成績)."""
    races = P[["date", "race_id"]].drop_duplicates().sort_values("date")
    cut = races["date"].iloc[len(races) // 2]
    first, second = P[P["date"] < cut], P[P["date"] >= cut]
    best, best_roi = None, -np.inf
    for ev in evs:
        r = flat_bet(BetRule(min_ev=ev).select(first, p_col))
        if r["bets"] >= min_bets and r["roi"] > best_roi:
            best, best_roi = ev, r["roi"]
    if best is None:
        return {"cut": cut, "chosen_min_ev": None}
    b2 = BetRule(min_ev=best).select(second, p_col)
    r2 = flat_bet(b2)
    lo, hi = bootstrap_roi(b2)
    return {"cut": cut, "chosen_min_ev": best, "first_half_roi": best_roi,
            "second_half_bets": r2["bets"], "second_half_roi": r2["roi"],
            "second_half_roi_lo90": lo, "second_half_roi_hi90": hi}


def by_period(P: pd.DataFrame, rule: BetRule, p_col: str = "p_model", freq: str = "QS") -> pd.DataFrame:
    """期間別の安定性 (どの時期でも勝てているか)."""
    b = rule.select(P, p_col)
    if b.empty:
        return pd.DataFrame()
    b["period"] = b["date"].dt.to_period(freq[0])
    rows = []
    for k, g in b.groupby("period"):
        rows.append({"period": str(k), **flat_bet(g)})
    return pd.DataFrame(rows)


def calibration(P: pd.DataFrame, p_col: str = "p_model", bins=(0, .02, .05, .1, .15, .2, .3, .45, 1.0)) -> pd.DataFrame:
    d = P[P[p_col].notna()].copy()
    d["bin"] = pd.cut(d[p_col], bins)
    g = d.groupby("bin", observed=True)
    return pd.DataFrame({"n": g.size(), "pred": g[p_col].mean(), "actual": g["win"].mean()})


def bootstrap_roi(bets: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """レース単位ブートストラップによる回収率の 90% 区間."""
    if bets.empty:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    g = bets.assign(ret=bets["win"] * bets["odds_win"]).groupby("race_id").agg(
        ret=("ret", "sum"), n=("ret", "size"))
    r, n = g["ret"].to_numpy(), g["n"].to_numpy()
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    rois = r[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(np.quantile(rois, 0.05)), float(np.quantile(rois, 0.95))


# ---------------------------------------------------------------------------
# 現実性: 締切前オッズのズレ (購入判断は締切前、払戻は確定オッズ)
# ---------------------------------------------------------------------------

def _race_codes(race_ids) -> np.ndarray:
    return pd.factorize(np.asarray(race_ids))[0]


def _group_sum(x: np.ndarray, codes: np.ndarray) -> np.ndarray:
    return np.bincount(codes, weights=x)[codes]


def _softmax_by_race(u: np.ndarray, codes: np.ndarray) -> np.ndarray:
    m = pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u - m)
    return e / _group_sum(e, codes)


def _full_odds(P: pd.DataFrame) -> np.ndarray:
    """レース内の全馬に市場確率がある行 (Stage2 が市場と合成した行。それ以外は p_model = p_fund)."""
    return P["p_mkt"].notna().groupby(P["race_id"]).transform("all").to_numpy()


def _stage2_cols(P: pd.DataFrame) -> list[str]:
    from .model import STAGE2_MARKET_EXTRA

    return ["s1", "log_p_mkt"] + [c for c in STAGE2_MARKET_EXTRA if c in P]


def stage2_market_coef(P: pd.DataFrame, p_col: str = "p_model", features: list[str] | None = None) -> pd.DataFrame:
    """ブロックごとの Stage2 (条件付きロジット) の log_p_mkt 係数 b (生の単位) を予測値から逆算する.

    walk_forward の出力には Stage2 の学習に使われた最初の数ブロックが含まれないので、再学習では
    同じ Stage2 を作れない。代わりに条件付きロジットでは同じレース内で
        log p_model_i − log p_model_j = Σ_k β_k (x_ik − x_jk)
    が厳密に成り立つことを使い、レース内で中心化した最小二乗で β を求める
    (Stage2 は欠損を学習時の平均で埋めるので、欠損のある列には欠損ダミーを加える)。

    戻り値: block, b (log_p_mkt の係数), n (行数), max_abs_err
      max_abs_err = 復元した効用から計算し直した p_model と元の p_model の最大差。
      ~1e-12 なら Stage2 を厳密に復元できている。大きい場合 (Stage2 の形が変わった等) は b は線形近似。
    """
    features = features or _stage2_cols(P)
    d = P[_full_odds(P)]
    blocks = d["block"] if "block" in d else pd.Series(0, index=d.index)
    rows = []
    for k, g in d.groupby(blocks.to_numpy()):
        codes = _race_codes(g["race_id"])
        cols, ib = [], None
        for c in features:
            x = g[c].to_numpy(dtype=float)
            miss = np.isnan(x)
            if c == "log_p_mkt":
                ib = len(cols)
            cols.append(np.where(miss, 0.0, x))
            if miss.any():
                cols.append(miss.astype(float))
        X = np.column_stack(cols)
        p = g[p_col].to_numpy(dtype=float)
        y = np.log(np.clip(p, 1e-300, None))
        cnt = np.bincount(codes).astype(float)
        means = np.column_stack([np.bincount(codes, weights=X[:, j]) / cnt for j in range(X.shape[1])])
        Xc = X - means[codes]
        yc = y - (np.bincount(codes, weights=y) / cnt)[codes]
        beta = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        err = float(np.abs(_softmax_by_race(X @ beta, codes) - p).max())
        rows.append({"block": k, "b": float(beta[ib]), "n": len(g), "max_abs_err": err})
    return pd.DataFrame(rows)


def _coef_per_row(P: pd.DataFrame, coef) -> np.ndarray:
    if np.isscalar(coef):
        return np.full(len(P), float(coef))
    m = coef.set_index("block")["b"] if isinstance(coef, pd.DataFrame) else pd.Series(coef)
    if "block" not in P:
        return np.full(len(P), float(m.iloc[0]))
    return P["block"].map(m).to_numpy(dtype=float)


def reprice_model(P: pd.DataFrame, new_log_p_mkt, coef=None, p_col: str = "p_model") -> np.ndarray:
    """市場確率だけを入れ替えたときの Stage2 の出力.

    u_new = log p_model + b·(log p̃_mkt − log p_mkt) をレース内で softmax。条件付きロジットなので、
    Stage2 に新しい log_p_mkt を入れて予測し直した値と厳密に一致する。
    市場と合成していないレース (オッズ欠けあり) は p_model をそのまま返す。
    """
    if coef is None:
        coef = stage2_market_coef(P, p_col)
    p = P[p_col].to_numpy(dtype=float).copy()
    full = _full_odds(P)
    if not full.any():
        return p
    b = _coef_per_row(P, coef)[full]
    delta = np.asarray(new_log_p_mkt, dtype=float)[full] - P["log_p_mkt"].to_numpy(dtype=float)[full]
    u = np.log(np.clip(p[full], 1e-300, None)) + b * delta
    p[full] = _softmax_by_race(u, _race_codes(P["race_id"].to_numpy()[full]))
    return p


def preclose_view(P: pd.DataFrame, sigma: float, rng: np.random.Generator | None = None, coef=None,
                  p_col: str = "p_model", seed: int = 0) -> pd.DataFrame:
    """購入判断時 (締切前) のオッズで見た予測表.

    締切前オッズ = 確定オッズ × exp(ε), ε ~ N(0, σ) を馬ごとに独立に与え、レース内で
    Σ 1/オッズ (= 控除込みの合計, overround) が確定オッズと同じになるよう正規化し直す。
    市場確率・log_p_mkt を締切前オッズから計算し直し、p_model も reprice_model で計算し直す。

    戻り値: P のコピー。odds_win / p_mkt / log_p_mkt / p_model が締切前の値 (購入判断用)、
           odds_final が確定オッズ (払戻用)。sigma=0 なら値は変わらない (odds_final だけ追加)。
    """
    d = P.copy()
    d["odds_final"] = d["odds_win"]
    if sigma <= 0:
        return d
    rng = rng if rng is not None else np.random.default_rng(seed)
    eps = sigma * rng.standard_normal(len(d))
    o = d["odds_win"].to_numpy(dtype=float)
    has = np.isfinite(o) & (o > 0)
    codes = _race_codes(d["race_id"])
    q = np.where(has, 1.0 / np.where(has, o, 1.0), 0.0)
    qn = q * np.exp(-eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        qn = qn * (_group_sum(q, codes) / _group_sum(qn, codes))
        d["odds_win"] = np.where(has, 1.0 / qn, np.nan)
        pm = np.where(has, qn / _group_sum(qn, codes), np.nan)
    new_log = np.where(has, np.log(np.clip(pm, 1e-4, None)), d["log_p_mkt"].to_numpy(dtype=float))
    d[p_col] = reprice_model(P, new_log, coef, p_col)
    d["p_mkt"] = pm
    d["log_p_mkt"] = new_log
    return d


# ---------------------------------------------------------------------------
# 現実性: プールの大きさ × オッズのズレ × 賭け金 の感度分析
# ---------------------------------------------------------------------------

DEFAULT_POOLS = (0.0, 500_000.0, 1_000_000.0, 2_500_000.0, 5_000_000.0)
DEFAULT_SIGMAS = (0.0, 0.05, 0.1, 0.2)
DEFAULT_STAKES = (100.0, 1_000.0, 5_000.0, 10_000.0)
_VIEW_COLS = ["race_id", "date", "block", "win", "odds_win", "p_mkt"]


def decision_sets(P: pd.DataFrame, rule: BetRule | None = None, sigmas=DEFAULT_SIGMAS, n_rep: int = 5,
                  seed: int = 0, coef=None, p_col: str = "p_model") -> dict[float, list[pd.DataFrame]]:
    """σ ごとに締切前オッズを n_rep 回引き、それぞれで購入ルールが選ぶ馬券 (odds_final 列つき) を返す.

    σ=0 は 1 回だけ (確定オッズで判断 = 従来どおり)。乱数は σ 間で共通 (ε = σ·z) なので比較がぶれにくい。
    """
    rule = rule or BetRule()
    cols = [c for c in _VIEW_COLS + [p_col] + _stage2_cols(P) if c in P]
    Q = P[list(dict.fromkeys(cols))]
    if coef is None and any(s > 0 for s in sigmas):
        coef = stage2_market_coef(P, p_col)
    out = {}
    for sg in sigmas:
        reps = 1 if sg <= 0 else n_rep
        out[float(sg)] = [rule.select(preclose_view(Q, sg, np.random.default_rng([seed, r]), coef, p_col), p_col)
                          for r in range(reps)]
    return out


def _stack_reps(bets: list[pd.DataFrame]) -> pd.DataFrame:
    """複数回の抽選の馬券を 1 つに (レース ID は回ごとに別物として扱う)."""
    parts = [b.assign(_rep=i) for i, b in enumerate(bets)]
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not d.empty:
        d["_key"] = d["race_id"].astype(str) + "#" + d["_rep"].astype(str)
    return d


def _ratio_ci(ret: np.ndarray, cost: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """レース単位ブートストラップ (bootstrap_roi と同じ手順) の 90% 区間."""
    if len(ret) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ret), size=(n_boot, len(ret)))
    rois = ret[idx].sum(axis=1) / cost[idx].sum(axis=1)
    return float(np.quantile(rois, 0.05)), float(np.quantile(rois, 0.95))


def robustness_table(P: pd.DataFrame, rule: BetRule | None = None, pools=DEFAULT_POOLS, sigmas=DEFAULT_SIGMAS,
                     stakes=DEFAULT_STAKES, n_rep: int = 5, payout_rate: float = config.WIN_PAYOUT_RATE,
                     seed: int = 0, n_boot: int = 2000, coef=None, p_col: str = "p_model",
                     decisions: dict | None = None) -> pd.DataFrame:
    """均等買いの回収率を (プール × 締切前オッズのズレ σ × 1 点の賭け金) ごとに計算する.

    pool_yen=0 は自分の購入の影響なし、odds_noise=0 は確定オッズで判断 (従来のバックテスト)。
    σ>0 は n_rep 回の抽選の平均。90% 区間はレース単位ブートストラップ (各レースの払戻・購入額を抽選間で平均)。
    1 行 = 1 シナリオ: label, pool_yen, odds_noise, stake, bets (1 回あたり), hit_rate, roi, roi_lo90, roi_hi90,
    profit (1 回あたりの収支, 円), paid_odds_ratio (実際の払戻オッズ / 自分が買わなかった場合の確定オッズ, 平均)。
    """
    decisions = decisions or decision_sets(P, rule, sigmas, n_rep, seed, coef, p_col)
    rows = []
    for sg in sigmas:
        reps = decisions[float(sg)]
        d = _stack_reps(reps)
        for pool in pools:
            for stake in stakes:
                row = {"label": f"pool={pool:,.0f} noise={sg:.2f} stake={stake:,.0f}",
                       "pool_yen": float(pool), "odds_noise": float(sg), "stake": float(stake)}
                if d.empty:
                    rows.append({**row, "bets": 0.0, "hit_rate": np.nan, "roi": np.nan, "roi_lo90": np.nan,
                                 "roi_hi90": np.nan, "profit": 0.0, "paid_odds_ratio": np.nan})
                    continue
                fin = d["odds_final"].to_numpy(dtype=float)
                eff = impact_odds(fin, stake, d["_key"], pool, payout_rate)
                w = d["win"].to_numpy(dtype=float)
                ret = w * eff * stake
                g = pd.DataFrame({"race_id": d["race_id"].to_numpy(), "ret": ret, "cost": stake}).groupby(
                    "race_id").sum()
                lo, hi = _ratio_ci(g["ret"].to_numpy() / len(reps), g["cost"].to_numpy() / len(reps), n_boot, seed)
                rows.append({**row, "bets": len(d) / len(reps), "hit_rate": float(w.mean()),
                             "roi": float(ret.sum() / (stake * len(d))), "roi_lo90": lo, "roi_hi90": hi,
                             "profit": float((ret.sum() - stake * len(d)) / len(reps)),
                             "paid_odds_ratio": float((eff / fin).mean())})
    return pd.DataFrame(rows)


def _roi_curve(d: pd.DataFrame, stakes: np.ndarray, pool, payout_rate: float) -> np.ndarray:
    fin = d["odds_final"].to_numpy(dtype=float)
    w = d["win"].to_numpy(dtype=float)
    codes = _race_codes(d["_key"])
    return np.array([float((w * impact_odds(fin, s, codes, pool, payout_rate)).mean()) for s in stakes])


def breakeven_table(P: pd.DataFrame, rule: BetRule | None = None, pools=DEFAULT_POOLS[1:], sigmas=DEFAULT_SIGMAS,
                    n_rep: int = 5, payout_rate: float = config.WIN_PAYOUT_RATE, seed: int = 0, coef=None,
                    p_col: str = "p_model", n_grid: int = 160, decisions: dict | None = None) -> pd.DataFrame:
    """「1 点の賭け金をいくらにすると歪み (エッジ) が消えるか」.

    均等買いの回収率 ROI(s) を賭け金 s の関数として計算し、
      breakeven_stake: ROI が 1 を下回る最小の賭け金 (100 円単位, 0 = 100 円でもすでにマイナス, inf = プール額まで黒字)
      best_stake     : 期待収支 s·(ROI(s) − 1) が最大になる賭け金 (グリッド上, 0 = どの額でも赤字)
    をプール × σ ごとに返す。自分の購入の影響は s / プール だけで決まるので、どちらもプールにほぼ比例する。
    """
    decisions = decisions or decision_sets(P, rule, sigmas, n_rep, seed, coef, p_col)
    rows = []
    for sg in sigmas:
        reps = decisions[float(sg)]
        d = _stack_reps(reps)
        for pool in pools:
            if pool <= 0:
                continue
            row = {"pool_yen": float(pool), "odds_noise": float(sg), "bets": len(d) / max(len(reps), 1)}
            if d.empty:
                rows.append({**row, "roi_100": np.nan, "breakeven_stake": np.nan, "breakeven_pct_pool": np.nan,
                             "best_stake": np.nan, "best_profit_per_bet": np.nan})
                continue
            grid = np.unique(np.round(np.geomspace(100, pool, n_grid) / 100) * 100)
            roi = _roi_curve(d, grid, pool, payout_rate)
            below = np.flatnonzero(roi < 1.0)
            if len(below) == 0:
                be = np.inf
            elif below[0] == 0:
                be = 0.0
            else:  # 100 円単位の二分探索
                lo, hi = int(grid[below[0] - 1] // 100), int(grid[below[0]] // 100)
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if _roi_curve(d, np.array([mid * 100.0]), pool, payout_rate)[0] < 1.0:
                        hi = mid
                    else:
                        lo = mid
                be = hi * 100.0
            prof = grid * (roi - 1)
            j = int(np.argmax(prof))
            rows.append({**row, "roi_100": float(roi[0]), "breakeven_stake": be,
                         "breakeven_pct_pool": be / pool * 100,
                         "best_stake": float(grid[j]) if prof[j] > 0 else 0.0,
                         "best_profit_per_bet": float(max(prof[j], 0.0))})
    return pd.DataFrame(rows)


def kelly_robustness(P: pd.DataFrame, rule: BetRule | None = None, pools=DEFAULT_POOLS, sigmas=(0.0, 0.1),
                     bankrolls=(100_000.0, 1_000_000.0), n_rep: int = 5, fraction: float = 0.25, cap: float = 0.03,
                     max_stake_pct_pool: float | None = None, payout_rate: float = config.WIN_PAYOUT_RATE,
                     seed: int = 0, coef=None, p_col: str = "p_model", decisions: dict | None = None) -> pd.DataFrame:
    """分数ケリーの資金推移に自分の購入の影響を入れる (資金が増えるほど賭け金が増え、オッズを下げる).

    max_stake_pct_pool: 1 点の上限をプールの何 % にするか (None = 上限なし)。
    σ>0 は n_rep 回の抽選の中央値 (growth) / 平均 (その他)。
    """
    decisions = decisions or decision_sets(P, rule, tuple(sigmas), n_rep, seed, coef, p_col)
    rows = []
    for sg in sigmas:
        reps = decisions[float(sg)]
        for bk0 in bankrolls:
            for pool in pools:
                ms = pool * max_stake_pct_pool / 100 if (max_stake_pct_pool and pool > 0) else None
                rs = [kelly_bet(b, p_col, bankroll=bk0, fraction=fraction, cap=cap, pool_yen=pool,
                                payout_rate=payout_rate, pay_col="odds_final", max_stake=ms, detail=True)
                      for b in reps]
                t = pd.DataFrame(rs)
                rows.append({"odds_noise": float(sg), "bankroll0": float(bk0), "pool_yen": float(pool),
                             "max_stake_cap": ms if ms is not None else np.nan,
                             "growth": float(t["growth"].median()), "max_dd_pct": float(t["max_dd_pct"].mean()),
                             "mean_stake": float(t["mean_stake"].mean()),
                             "last_q_mean_stake": float(t["last_q_mean_stake"].mean()),
                             "max_stake": float(t["max_stake"].mean()),
                             "paid_odds_ratio": float(t["paid_odds_ratio"].mean())})
    return pd.DataFrame(rows)
