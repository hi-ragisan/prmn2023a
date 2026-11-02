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

歪みの強さは SimConfig.distortion (0〜1, 既定 1 = 従来どおり) でまとめて、
または d_flb / d_fame / d_overreact / d_moist / d_weight_drop で個別に調整できる。
  distortion = 0 → 歪みなし: 市場の見立て = 真の強さ + 誤差 (market_info_sd)、
                   確率は誤差を織り込んで (実際の勝敗に対して) 較正済み。
注意: market_info_sd > 0 の誤差は「真の強さに後から乗る測定誤差」なので、市場は公開情報 (近走成績) で
自分の誤差を補正していない。この分はモデルが正当に拾える小さな情報になる (半強度の効率性は満たさない)。
帰無仮説の検証には market_info_sd = 0 (市場が真の強さを完全に知る) の "efficient" を使う。
プリセット:
  SimConfig.preset("default")    従来どおり (既存テスト・README の数値はこれ)
  SimConfig.preset("efficient")  歪みなし + 市場は真の強さを完全に知る (帰無仮説の検証用)
  SimConfig.preset("realistic")  実データの統計 (人気別勝率・回収率・頭数・払戻率) に合わせた中間設定
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
    # ---- 以下は追加フィールド。既定値では従来と完全に同じデータを生成する ----
    # 埋め込んだ歪みの全体スケール (0 = 歪みなし, 1 = 従来)。1 を超えると誇張。
    distortion: float = 1.0
    # 歪みごとの個別スケール。None なら distortion を使う。
    d_flb: float | None = None        # 1. 大穴過剰人気 (flb_alpha と、較正温度からの過小確信 market_temp)
    d_fame: float | None = None       # 2. 人気騎手の過大評価
    d_overreact: float | None = None  # 3. 前走着順への過剰反応
    d_moist: float | None = None      # 4. 馬場水分適性の無視 (0 なら市場も適性を織り込む)
    d_weight_drop: float | None = None  # 5. 馬体重大幅減の軽視 (0 なら市場も織り込む)
    # d_flb = 0 のときの市場温度。None なら market_info_sd から自動計算 (calibrated_temp)。
    market_temp_calibrated: float | None = None
    payout_rate: float | None = None  # 単勝払戻率。None なら config.WIN_PAYOUT_RATE
    max_field: int = 10               # 最大頭数 (ばんえいは 10 頭立てが上限)
    scratch_prob: float = 0.0         # 出走取消・除外の確率 (頭数のばらつき)。0 なら乱数を消費しない
    min_field: int = 5                # 取消後の最少頭数

    def scale(self, name: str) -> float:
        """歪み name (flb / fame / overreact / moist / weight_drop) の実効スケール."""
        v = getattr(self, f"d_{name}")
        v = self.distortion if v is None else v
        if v < 0:
            raise ValueError(f"d_{name} / distortion は 0 以上: {v}")
        return float(v)

    def calibrated_temp(self) -> float:
        """歪みなしの市場が使う「較正された」温度.

        市場の見立て v = 真の強さ s + 誤差 e (sd = market_info_sd) のとき、softmax(T·v/noise) が
        *実際の勝敗* に対して較正される (p_win ∝ p_mkt^α で α ≈ 1) 温度 T。
          - 誤差の分だけ確率を縮める (errors-in-variables の減衰): 1 / (1 + (sd/CAL_TAU)^2)
          - 第2障害での停止 (_p_true には含まれない偶然) の分だけさらに縮める: 1 − CAL_STOP·stop_prob
        定数は sd ∈ {0, 0.1, 0.2}, stop_prob ∈ {0, 0.07}, seed 42/3/7/11 (各3年) で
        T=1 の市場に対する α を実測して当てはめた (各点 ±0.02 程度で一致)。
        """
        if self.market_temp_calibrated is not None:
            return float(self.market_temp_calibrated)
        return (1.0 - CAL_STOP * self.stop_prob) / (1.0 + (self.market_info_sd / CAL_TAU) ** 2)

    def market_shape(self) -> tuple[float, float]:
        """(温度, flb のべき α) の実効値. d_flb = 1 なら従来の (market_temp, flb_alpha) と厳密に一致."""
        s = self.scale("flb")
        # s = 1 のとき (1 - s) = 0.0 なので浮動小数点でも元の値と完全一致する
        temp = self.market_temp + (1.0 - s) * (self.calibrated_temp() - self.market_temp)
        alpha = self.flb_alpha + (1.0 - s) * (1.0 - self.flb_alpha)
        return temp, alpha

    @classmethod
    def preset(cls, name: str = "default", **overrides) -> "SimConfig":
        """名前付きプリセット (overrides で個別フィールドを上書き)."""
        if name not in PRESETS:
            raise ValueError(f"未知のプリセット {name!r}: {sorted(PRESETS)}")
        return cls(**{**PRESETS[name], **overrides})


# calibrated_temp の当てはめ定数
CAL_TAU = 0.49   # レース内の真の強さのばらつきに相当 (強さの単位)
CAL_STOP = 0.43  # 停止の偶然による確率の平坦化

PRESETS: dict[str, dict] = {
    # 従来の既定 (歪み 5 種類を全て埋め込み)
    "default": {},
    # 帰無仮説用: 歪みなし。市場は真の強さ (モデルが見られない調教の気配なども含む) を完全に知り、
    # 停止の偶然を織り込んで較正した確率を付ける。モデルの特徴量はすべて真の強さの関数なので、
    # 市場に足せる情報は原理的にゼロ。残る誤差はオッズの 0.1 倍単位の切り捨てだけ。
    # (market_info_sd > 0 だと市場は公開情報で自分の誤差を補正しないため、モデルが拾える情報が
    #  わずかに残る。scratch 実験 6 シードで s1 の追加係数 0.18±0.05。ただし控除 20% は越えない)
    "efficient": dict(distortion=0.0, market_info_sd=0.0),
    # 実データの統計に合わせた中間設定。値は暫定 (TODO: L1 の実データ統計で較正)。
    "realistic": dict(distortion=0.25),
}


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
    s_fame, s_over = cfg.scale("fame"), cfg.scale("overreact")
    s_moist, s_drop = cfg.scale("moist"), cfg.scale("weight_drop")
    mkt_temp, mkt_alpha = cfg.market_shape()
    payout = config.WIN_PAYOUT_RATE if cfg.payout_rate is None else cfg.payout_rate
    F = cfg.max_field

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
        target = cfg.races_per_day * F
        runners = active[:target]
        if len(runners) < 16:
            continue
        # クラス分け: 公開されている実績 (merit) 順に並べて F (既定 10) 頭ずつ
        runners.sort(key=lambda h: -h["merit"])
        all_merits = np.array([h["merit"] for h in horses if h["active"]])
        qs = np.quantile(all_merits, np.linspace(1, 0, len(CLASS_ORDER) + 1)[1:-1])

        chunks = [runners[i:i + F] for i in range(0, len(runners), F)]
        if len(chunks) > 1 and len(chunks[-1]) < 6:
            chunks[-2].extend(chunks.pop())
        chunks = chunks[: cfg.races_per_day]
        # 下級条件から順に組む
        for race_no, field in enumerate(reversed(chunks), start=1):
            field = field[:F]
            if cfg.scratch_prob > 0:  # 出走取消・除外 (既定 0 では乱数を消費しない)
                keep = rng.random(len(field)) >= cfg.scratch_prob
                if keep.sum() < cfg.min_field:
                    keep[: cfg.min_field] = True
                field = [h for h, k in zip(field, keep) if k]
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
            # 歪みのスケールが 1 のとき、以下の演算は従来のコードと浮動小数点レベルで一致する
            # (x * 1.0 == x, 0.20 * 1.0 == 0.20)。乱数の消費順も変えない。
            overreact = np.where(np.isnan(last_pct), 0.0, (0.20 * s_over) * (0.5 - last_pct))
            mkt_view = (
                ability + rng.normal(0, cfg.market_info_sd, n)
                + age_eff + j_skill[jockeys] + j_fame[jockeys] * (0.4 * s_fame) + trn
                - 0.012 * cw_rel + private + overreact
            )
            # 歪み 4/5: 既定 (スケール 1) では馬場水分適性・馬体重減を市場は完全に無視
            if s_moist != 1.0:
                mkt_view = mkt_view + (1.0 - s_moist) * (apt * moist_z)
            if s_drop != 1.0:
                mkt_view = mkt_view - (1.0 - s_drop) * (0.25 * big_drop)
            q = np.exp(mkt_temp * mkt_view / cfg.noise_scale)
            q /= q.sum()
            q = q ** mkt_alpha
            q /= q.sum()
            odds = np.maximum(1.0, np.floor(payout / q * 10) / 10)
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
    ap.add_argument("--preset", default="default", choices=sorted(PRESETS))
    ap.add_argument("--distortion", type=float, default=None, help="歪みの全体スケール (0〜1)")
    ap.add_argument("--out", default=str(config.INTERIM_DIR / "simulated.csv"))
    a = ap.parse_args(argv)
    config.ensure_dirs()
    kw = dict(start=a.start, end=a.end, seed=a.seed)
    if a.distortion is not None:
        kw["distortion"] = a.distortion
    df = simulate(SimConfig.preset(a.preset, **kw))
    df.to_csv(a.out, index=False)
    print(f"{len(df)} 行 / {df['race_id'].nunique()} レース → {a.out}")


if __name__ == "__main__":
    main()
