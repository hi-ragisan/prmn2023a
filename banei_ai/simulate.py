"""検証用のばんえい競馬シミュレーター.

!!! 注意 !!!
ここで生成するデータは「パイプラインが正しく動くか」「既知の歪みを検出できるか」を
確かめるための人工データであり、実際のばんえい競馬の成績ではない。
このデータ上のバックテスト回収率は、実馬券での収益性について何も保証しない。

市場 (オッズ) には意図的に次の歪みを入れてある:
  1. 大穴の過剰人気 (favorite-longshot bias): 市場確率 q ∝ p^0.85
  2. 人気騎手バイアス: 騎手ごとの「知名度」分だけ市場が過大評価
  3. 前走着順への過剰反応
  4. 馬場水分適性を市場が無視
  5. 馬体重の大幅減を市場が軽視
一方で市場は、モデルが見られない情報 (調教の気配など) も一部持っている。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

CLASS_BASE_WEIGHT = {
    "A1": 720, "A2": 700, "B1": 685, "B2": 675, "B3": 665,
    "B4": 655, "C1": 640, "C2": 620,
}
CLASS_ORDER = list(CLASS_BASE_WEIGHT)


@dataclass
class SimConfig:
    start: str = "2023-04-01"
    end: str = "2025-03-31"
    n_horses: int = 650
    n_jockeys: int = 22
    n_trainers: int = 30
    races_per_day: int = 11
    seed: int = config.RANDOM_SEED
    noise_scale: float = 0.45     # レースの偶然性 (Gumbel スケール)
    stop_prob: float = 0.07       # 第2障害で止まる確率
    flb_alpha: float = 0.85       # 大穴過剰人気の強さ (1.0 なら歪みなし)
    market_info_sd: float = 0.20  # 市場の能力認識の誤差
    market_private_sd: float = 0.25  # 市場だけが知る情報 (調教など) の大きさ
    market_temp: float = 0.7      # 市場の確信度 (自分の認識誤差を考慮して確率を縮める)


def _race_days(start: str, end: str) -> list[pd.Timestamp]:
    days = pd.date_range(start, end, freq="D")
    # ばんえいは主に土・日・月開催
    return [d for d in days if d.dayofweek in (5, 6, 0)]


def _moisture_for(day: pd.Timestamp, rng: np.random.Generator) -> float:
    # 冬 (12-3月) は水分が高く軽い馬場、夏は乾燥
    winter = day.month in (12, 1, 2, 3)
    base = 3.5 if winter else 1.6
    m = rng.gamma(shape=4.0, scale=base / 4.0)
    return float(np.clip(round(m, 1), 0.4, 9.5))


def simulate(cfg: SimConfig | None = None) -> pd.DataFrame:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)

    # ---------- 騎手 ----------
    J = cfg.n_jockeys
    j_skill = rng.normal(0, 0.30, J)
    j_fame = rng.normal(0, 0.25, J) + 0.5 * j_skill  # 人気騎手ほど過大評価
    j_names = [f"騎手{j:02d}" for j in range(J)]

    # ---------- 調教師 ----------
    T = cfg.n_trainers
    t_skill = rng.normal(0, 0.15, T)

    # ---------- 馬 ----------
    horses: list[dict] = []

    def new_horse(age: int, day: pd.Timestamp) -> dict:
        h = {
            "id": f"H{len(horses):05d}",
            "ability": rng.normal(0, 1.0),
            "moist_apt": rng.normal(0, 0.20),
            "sex": rng.choice(["牡", "牝", "セ"], p=[0.5, 0.4, 0.1]),
            "birth_year": day.year - age,
            "trainer": int(rng.integers(T)),
            "jockey": None,
            "weight": rng.normal(1000, 60),
            "last_weight": np.nan,
            "next_ok": day + pd.Timedelta(days=int(rng.integers(0, 14))),
            "merit": 0.0,
            "last_finish_pct": np.nan,
            "active": True,
        }
        h["merit"] = h["ability"] + rng.normal(0, 0.5)
        # 騎手は能力の高い馬ほど上手い騎手が付きやすい
        score = j_skill * 1.5 + rng.normal(0, 0.6, J) + 0.3 * h["ability"] * j_skill
        h["jockey"] = int(np.argmax(score))
        horses.append(h)
        return h

    days = _race_days(cfg.start, cfg.end)
    for _ in range(cfg.n_horses):
        new_horse(int(rng.integers(2, 10)), days[0])

    rows: list[dict] = []
    last_season = days[0].year if days[0].month >= 4 else days[0].year - 1

    for day in days:
        season = day.year if day.month >= 4 else day.year - 1
        if season != last_season:
            last_season = season
            # 新年度: 高齢馬・一部の馬が引退し、2歳馬が入厩
            for h in horses:
                age = day.year - h["birth_year"]
                if h["active"] and (age >= 11 or rng.random() < 0.12):
                    h["active"] = False
            n_new = cfg.n_horses - sum(h["active"] for h in horses)
            for _ in range(max(n_new, 0)):
                new_horse(2, day)

        moist = _moisture_for(day, rng)
        moist_z = (moist - 2.5) / 1.5

        active = [h for h in horses if h["active"] and h["next_ok"] <= day]
        rng.shuffle(active)
        target = cfg.races_per_day * 10
        runners = active[:target]
        if len(runners) < 16:
            continue
        # クラス分け: 公開されている実績 (merit) 順に並べて 10 頭ずつ
        runners.sort(key=lambda h: -h["merit"])
        all_merits = np.array([h["merit"] for h in horses if h["active"]])
        qs = np.quantile(all_merits, np.linspace(1, 0, len(CLASS_ORDER) + 1)[1:-1])

        chunks = [runners[i:i + 10] for i in range(0, len(runners), 10)]
        if len(chunks) > 1 and len(chunks[-1]) < 6:
            chunks[-2].extend(chunks.pop())
        chunks = chunks[: cfg.races_per_day]
        # 下級条件から順に組む
        for race_no, field in enumerate(reversed(chunks), start=1):
            field = field[:10]
            n = len(field)
            med_merit = float(np.median([h["merit"] for h in field]))
            cls_idx = int(np.searchsorted(-qs, -med_merit))
            cls = CLASS_ORDER[min(cls_idx, len(CLASS_ORDER) - 1)]
            race_id = f"{day:%Y%m%d}{race_no:02d}"

            merits = np.array([h["merit"] for h in field])
            merit_rank = merits.argsort()[::-1].argsort()  # 0 = 実績最上位
            base_w = CLASS_BASE_WEIGHT[cls]
            cw = np.array([
                base_w + 10 * (n - 1 - merit_rank[i]) // 2 * 2
                - (20 if h["sex"] == "牝" else 0)
                + int(rng.integers(-1, 2)) * 5
                for i, h in enumerate(field)
            ], dtype=float)
            cw_rel = cw - cw.mean()

            # 騎手: 主戦騎手が乗れることが多いが乗り替わりもある
            jockeys = []
            used: set[int] = set()
            for h in field:
                j = h["jockey"]
                if j in used or rng.random() < 0.25:
                    cand = [k for k in range(J) if k not in used]
                    j = int(rng.choice(cand))
                used.add(j)
                jockeys.append(j)
            jockeys = np.array(jockeys)

            # 馬体重
            w_now = np.array([h["weight"] + rng.normal(0, 8) for h in field])
            w_diff = np.array([
                (w_now[i] - h["last_weight"]) if not np.isnan(h["last_weight"]) else np.nan
                for i, h in enumerate(field)
            ])
            big_drop = np.nan_to_num(w_diff, nan=0.0) <= -20

            ability = np.array([h["ability"] for h in field])
            apt = np.array([h["moist_apt"] for h in field])
            age = np.array([day.year - h["birth_year"] for h in field], dtype=float)
            age_eff = -0.04 * (age - 6.0) ** 2 + 0.3
            trn = np.array([t_skill[h["trainer"]] for h in field])
            private = rng.normal(0, cfg.market_private_sd, n)  # 調教の気配など

            strength = (
                ability + age_eff + j_skill[jockeys] + trn
                + apt * moist_z - 0.012 * cw_rel
                - 0.25 * big_drop + private
            )
            perf = strength + rng.gumbel(0, cfg.noise_scale, n)
            stopped = rng.random(n) < cfg.stop_prob
            perf = perf - stopped * rng.uniform(0.8, 2.5, n)
            dnf = rng.random(n) < 0.004

            order = np.argsort(-perf)
            finish = np.empty(n)
            finish[order] = np.arange(1, n + 1)

            # 真の勝率 (Gumbel ノイズ下の多項ロジット) ※停止リスクは全馬共通とみなす
            p_true = np.exp(strength / cfg.noise_scale)
            p_true /= p_true.sum()

            # ---------- 市場 ----------
            last_pct = np.array([h["last_finish_pct"] for h in field])
            overreact = np.where(np.isnan(last_pct), 0.0, 0.20 * (0.5 - last_pct))
            mkt_view = (
                ability + rng.normal(0, cfg.market_info_sd, n)
                + age_eff + j_skill[jockeys] + j_fame[jockeys] * 0.4 + trn
                - 0.012 * cw_rel + private + overreact
                # 馬場水分適性・馬体重減は無視
            )
            q = np.exp(cfg.market_temp * mkt_view / cfg.noise_scale)
            q /= q.sum()
            q = q ** cfg.flb_alpha
            q /= q.sum()
            odds = np.maximum(1.0, np.floor(config.WIN_PAYOUT_RATE / q * 10) / 10)
            pop = pd.Series(odds).rank(method="first").to_numpy()

            base_time = 118 - 5.0 * (moist - 2.0) + 0.06 * (base_w - 650)
            win_time = base_time + rng.normal(0, 2.0)
            gaps = np.sort(perf)[::-1]
            gaps = (gaps[0] - perf) * 6.0
            times = np.round(win_time + gaps, 1)

            for i, h in enumerate(field):
                status = "dnf" if dnf[i] else "ok"
                rows.append({
                    "race_id": race_id,
                    "date": day,
                    "race_no": race_no,
                    "race_name": f"{cls}-{race_no}",
                    "race_class": cls,
                    "moisture": moist,
                    "post": i + 1,
                    "horse_id": h["id"],
                    "horse_name": f"シミュ{h['id']}",
                    "sex": h["sex"],
                    "age": day.year - h["birth_year"],
                    "jockey_id": f"J{jockeys[i]:02d}",
                    "jockey_name": j_names[jockeys[i]],
                    "trainer_id": f"T{h['trainer']:02d}",
                    "carried_weight": cw[i],
                    "horse_weight": round(w_now[i]),
                    "horse_weight_diff": w_diff[i] if np.isnan(w_diff[i]) else round(w_diff[i]),
                    "odds_win": odds[i],
                    "popularity": pop[i],
                    "finish": np.nan if dnf[i] else finish[i],
                    "time_sec": np.nan if dnf[i] else times[i],
                    "status": status,
                    "_p_true": p_true[i],  # 検証用の真の勝率 (実データには無い)
                })

            # 状態更新
            for i, h in enumerate(field):
                pct = (finish[i] - 1) / max(n - 1, 1)
                h["last_finish_pct"] = 1.0 if dnf[i] else pct
                h["merit"] = 0.8 * h["merit"] + 0.2 * (h["ability"] + (0.5 - pct) * 1.5)
                h["ability"] += rng.normal(0, 0.03)
                h["last_weight"] = w_now[i]
                h["weight"] += rng.normal(0, 6)
                h["next_ok"] = day + pd.Timedelta(days=int(rng.integers(9, 22)))

    df = pd.DataFrame(rows)
    # 馬番をランダム化 (ばんえいでは馬番の有利不利は小さい)
    df["post"] = df.groupby("race_id")["post"].transform(
        lambda s: rng.permutation(len(s)) + 1
    )
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="検証用シミュレーションデータを生成")
    ap.add_argument("--start", default="2023-04-01")
    ap.add_argument("--end", default="2025-03-31")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--out", default=str(config.INTERIM_DIR / "simulated.csv"))
    a = ap.parse_args(argv)
    config.ensure_dirs()
    df = simulate(SimConfig(start=a.start, end=a.end, seed=a.seed))
    df.to_csv(a.out, index=False)
    print(f"{len(df)} 行 / {df['race_id'].nunique()} レース → {a.out}")


if __name__ == "__main__":
    main()
