"""
特徴量エンジニアリング

入力: data/raw/{year}_races.csv (2015〜2024)
出力: data/processed/features.csv

注意: 過去成績はすべて shift(1) で「そのレースより前のデータのみ」を参照（リーク防止）
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2015, 2025))

# ─── 定数マップ ───────────────────────────────────────────────────────────────

SURFACE_MAP    = {"芝": 0, "ダート": 1, "障": 2}
CONDITION_MAP  = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
WEATHER_MAP    = {"晴": 0, "曇": 1, "雨": 2, "小雨": 3}


# ─── データ読み込み ───────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    dfs = []
    for year in YEARS:
        p = RAW_DIR / f"{year}_races.csv"
        if p.exists():
            dfs.append(pd.read_csv(p, dtype=str))
    df = pd.concat(dfs, ignore_index=True)
    print(f"読み込み完了: {len(df):,} 行 ({len(dfs)} ファイル)")
    return df


# ─── 前処理 ──────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 日付
    df["date"] = pd.to_datetime(
        df["date"].str.replace(r"(\d{4})年(\d{1,2})月(\d{1,2})日",
                               lambda m: m.group(0), regex=True),
        format="%Y年%m月%d日", errors="coerce",
    )

    # 着順: 中止/除外/降着は NaN
    df["order_num"] = pd.to_numeric(
        df["order"].str.extract(r"^(\d+)", expand=False), errors="coerce"
    )

    # 数値変換
    df["distance_m"]      = pd.to_numeric(df["distance"],  errors="coerce")
    df["last_3f"]         = pd.to_numeric(df["last_3f"],   errors="coerce")
    df["odds"]            = pd.to_numeric(df["odds"].replace("---", np.nan), errors="coerce")
    df["weight_carried"]  = pd.to_numeric(df["weight"],    errors="coerce")

    # 馬体重: "536(0)" → 536
    df["horse_weight_kg"] = pd.to_numeric(
        df["horse_weight"].str.extract(r"^(\d+)", expand=False), errors="coerce"
    )

    # エンコード
    df["surface_enc"]   = df["surface"].map(SURFACE_MAP)
    df["condition_enc"] = df["condition"].map(CONDITION_MAP)
    df["weather_enc"]   = df["weather"].map(WEATHER_MAP)

    # ターゲット: 0/1 (中止・除外は NaN)
    df["is_win"] = np.where(
        df["order_num"].isna(), np.nan, (df["order_num"] == 1).astype(float)
    )

    # 距離帯（200m 刻み）
    df["dist_band"] = (df["distance_m"] // 200 * 200).astype("Int64")

    # 時系列順にソート（同一 date 内は race_id → horse_num で一意）
    df = df.sort_values(["date", "race_id", "horse_num"]).reset_index(drop=True)

    return df


# ─── 特徴量ヘルパー ───────────────────────────────────────────────────────────

def _grp_rolling_mean(df: pd.DataFrame, group_cols: str | list[str],
                      value_col: str, window: int) -> pd.Series:
    """
    groupby(group_cols)[value_col] の shift(1) + rolling(window) mean。
    ※ groupby.rolling は transform(lambda) より ~10倍高速。
    """
    cols = [group_cols] if isinstance(group_cols, str) else group_cols
    shifted = df.groupby(cols)[value_col].shift(1)
    keys = [df[c] for c in cols]
    n_levels = len(cols)
    result = (
        shifted.groupby(keys)
        .rolling(window, min_periods=1).mean()
        .reset_index(level=list(range(n_levels)), drop=True)
        .sort_index()
    )
    return result


def _grp_expanding_mean(df: pd.DataFrame, group_cols: str | list[str],
                        value_col: str) -> pd.Series:
    """
    groupby(group_cols)[value_col] の shift(1) + expanding mean（全過去期間）。
    """
    cols = [group_cols] if isinstance(group_cols, str) else group_cols
    shifted = df.groupby(cols)[value_col].shift(1)
    keys = [df[c] for c in cols]
    n_levels = len(cols)
    result = (
        shifted.groupby(keys)
        .expanding().mean()
        .reset_index(level=list(range(n_levels)), drop=True)
        .sort_index()
    )
    return result


# ─── 特徴量計算 ──────────────────────────────────────────────────────────────

def add_horse_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  馬の過去成績を計算中...")

    # 直近3走・5走の平均着順・平均上がり3F
    for n, lbl in [(3, "3"), (5, "5")]:
        df[f"horse_avg_order_{lbl}"]  = _grp_rolling_mean(df, "horse_id", "order_num", n)
        df[f"horse_avg_last3f_{lbl}"] = _grp_rolling_mean(df, "horse_id", "last_3f",   n)

    # 同距離帯 (±200m) の勝率
    df["horse_win_rate_dist"]    = _grp_expanding_mean(df, ["horse_id", "dist_band"], "is_win")
    # 同 venue の勝率
    df["horse_win_rate_venue"]   = _grp_expanding_mean(df, ["horse_id", "venue"],     "is_win")
    # 芝/ダート別勝率
    df["horse_win_rate_surface"] = _grp_expanding_mean(df, ["horse_id", "surface"],   "is_win")

    return df


def add_jockey_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  騎手の過去成績を計算中...")

    # 直近 100 走勝率
    df["jockey_win_rate_100"]   = _grp_rolling_mean(df,    "jockey",           "is_win", 100)
    # 騎手×venue 勝率（全過去）
    df["jockey_win_rate_venue"] = _grp_expanding_mean(df, ["jockey", "venue"], "is_win")

    return df


# ─── メイン ──────────────────────────────────────────────────────────────────

def build():
    import time
    t0 = time.time()
    print("=== 特徴量エンジニアリング開始 ===")

    df = load_raw()
    df = preprocess(df)
    print(f"前処理完了: {len(df):,} 行  ({time.time()-t0:.1f}s)")
    print(f"  date 範囲: {df['date'].min().date()} 〜 {df['date'].max().date()}")

    df = add_horse_features(df)
    print(f"  馬特徴量完了 ({time.time()-t0:.1f}s)")

    df = add_jockey_features(df)
    print(f"  騎手特徴量完了 ({time.time()-t0:.1f}s)")

    # 出力カラム
    feature_cols = [
        # ID・メタ
        "race_id", "date", "venue", "horse_id", "horse_name", "jockey",
        # レース条件
        "distance_m", "surface_enc", "condition_enc", "weather_enc",
        "weight_carried", "horse_weight_kg",
        # 馬の過去成績
        "horse_avg_order_3", "horse_avg_order_5",
        "horse_avg_last3f_3", "horse_avg_last3f_5",
        "horse_win_rate_dist", "horse_win_rate_venue", "horse_win_rate_surface",
        # 騎手
        "jockey_win_rate_100", "jockey_win_rate_venue",
        # ターゲット
        "is_win", "odds",
    ]

    out = df[feature_cols].copy()
    out_path = OUT_DIR / "features.csv"
    out.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print()
    print("=== 完了 ===")
    print(f"出力: {out_path}")
    print(f"処理時間: {elapsed:.1f}s")
    print(f"行数: {len(out):,}")
    print(f"カラム数: {len(out.columns)}")
    print()
    print(f"{'カラム名':<30} {'non-null':>9}  {'%':>6}")
    print("-" * 50)
    for col in out.columns:
        non_null = int(out[col].notna().sum())
        pct = non_null / len(out) * 100
        print(f"{col:<30} {non_null:>9,}  {pct:>5.1f}%")


if __name__ == "__main__":
    build()
