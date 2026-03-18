"""
当日出走馬の勝率予測パイプライン

フロー:
  1. netkeiba から当日出馬表をスクレイピング (scrape_entries.py)
  2. src/model/horse_stats.csv / jockey_stats.csv で historical 特徴量を補完
  3. LightGBM モデル (lgbm_model.txt) で勝率を予測
  4. pred_prob >= MIN_PRED_PROB かつ odds <= MAX_ODDS の買い目を
     web/predictions/latest.json に保存

Usage:
    python src/predict/predict_today.py                # 今日 (JST)
    python src/predict/predict_today.py --date 20260315  # 日付指定
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# パスを追加して scrape_entries をインポート
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.predict.scrape_entries import scrape_today

# ─── 設定 ────────────────────────────────────────────────────────────────────

JST = timezone(timedelta(hours=9))

ROOT        = Path(__file__).resolve().parents[2]
MODEL_TXT   = ROOT / "src" / "model" / "lgbm_model.txt"
HORSE_STATS = ROOT / "src" / "model" / "horse_stats.csv"
JOCKEY_STATS= ROOT / "src" / "model" / "jockey_stats.csv"
OUT_PATH    = ROOT / "docs" / "predictions" / "latest.json"

MIN_PRED_PROB = 0.08
MAX_ODDS      = 20.0

FEATURE_COLS = [
    "distance_m",
    "surface_enc",
    "condition_enc",
    "weather_enc",
    "weight_carried",
    "horse_weight_kg",
    "horse_avg_order_3",
    "horse_avg_order_5",
    "horse_avg_last3f_3",
    "horse_avg_last3f_5",
    "horse_win_rate_dist",
    "horse_win_rate_venue",
    "horse_win_rate_surface",
    "jockey_win_rate_100",
    "jockey_win_rate_venue",
    "popularity",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── 出馬表 → DataFrame ───────────────────────────────────────────────────────

def races_to_df(races: list[dict]) -> pd.DataFrame:
    """scrape_today() の結果を flat な DataFrame に変換。"""
    rows = []
    for race in races:
        for h in race["horses"]:
            rows.append({
                "race_id":        race["race_id"],
                "race_name":      race["race_name"],
                "venue":          race["venue"],
                "race_num":       race["race_num"],
                "surface_raw":    race.get("surface_raw", ""),
                "distance_m":     race.get("distance_m"),
                "condition_raw":  race.get("condition_raw", ""),
                "weather_raw":    race.get("weather_raw", ""),
                "surface_enc":    race.get("surface_enc"),
                "condition_enc":  race.get("condition_enc"),
                "weather_enc":    race.get("weather_enc"),
                "horse_num":      h["horse_num"],
                "horse_name":     h["horse_name"],
                "horse_id":       h["horse_id"],
                "jockey":         h["jockey"],
                "weight_carried": h.get("weight_carried"),
                "horse_weight_kg":h.get("horse_weight_kg"),
                "odds":           h.get("odds"),
                "popularity_raw": h.get("popularity"),   # スクレイピング済み人気
            })
    return pd.DataFrame(rows)


# ─── historical 特徴量の結合 ──────────────────────────────────────────────────

def merge_stats(df: pd.DataFrame) -> pd.DataFrame:
    """horse_stats.csv / jockey_stats.csv を結合。未知馬・騎手は NaN。"""
    horse_stats = pd.read_csv(HORSE_STATS, index_col="horse_id", dtype={"horse_id": str})
    jockey_stats = pd.read_csv(JOCKEY_STATS, index_col="jockey")

    df["horse_id"] = df["horse_id"].astype(str)
    df = df.merge(horse_stats, on="horse_id", how="left")
    df = df.merge(jockey_stats, on="jockey",    how="left")

    logger.info(
        f"stats merge: 馬 {df['horse_avg_order_3'].notna().sum()}/{len(df)} 件マッチ, "
        f"騎手 {df['jockey_win_rate_100'].notna().sum()}/{len(df)} 件マッチ"
    )
    return df


# ─── popularity 計算 ─────────────────────────────────────────────────────────

def add_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """
    人気を確定する。
    スクレイピング済みの popularity_raw を優先し、未取得の場合は
    レース内オッズ昇順ランクで代替する。
    """
    odds_rank = df.groupby("race_id")["odds"].rank(method="min", ascending=True)
    if "popularity_raw" in df.columns:
        df["popularity"] = df["popularity_raw"].where(
            df["popularity_raw"].notna(), odds_rank
        )
    else:
        df["popularity"] = odds_rank
    return df


# ─── フィルタ & JSON 生成 ─────────────────────────────────────────────────────

def _nan_to_none(v):
    """JSON シリアライズ用: float NaN → None、numpy型 → Python ネイティブ型に変換。"""
    if v is None:
        return None
    try:
        if np.isnan(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def build_json(df: pd.DataFrame, target_date: str) -> dict:
    """
    フィルタ済み DataFrame を latest.json 形式の dict に変換。
    pred_prob >= MIN_PRED_PROB かつ (odds <= MAX_ODDS または odds 未確定) の行のみ。
    オッズ未確定 (NaN) の場合はオッズフィルターをスキップする。
    """
    odds_ok = df["odds"].isna() | (df["odds"] <= MAX_ODDS)
    mask = (df["pred_prob"] >= MIN_PRED_PROB) & odds_ok
    bets = df[mask].copy()

    generated_at = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S+09:00")

    races_out = []
    for race_id, grp in bets.groupby("race_id"):
        first = grp.iloc[0]
        horses_out = []
        for _, row in grp.sort_values("horse_num").iterrows():
            horses_out.append({
                "horse_num":       int(row["horse_num"]),
                "horse_name":      row["horse_name"],
                "jockey":          row["jockey"],
                "odds":            _nan_to_none(row["odds"]),
                "popularity":      _nan_to_none(row["popularity"]),
                "pred_prob":       round(float(row["pred_prob"]), 4),
                "weight_carried":  _nan_to_none(row["weight_carried"]),
                "horse_weight_kg": _nan_to_none(row["horse_weight_kg"]),
            })
        races_out.append({
            "race_id":    race_id,
            "race_name":  first["race_name"],
            "venue":      first["venue"],
            "race_num":   int(first["race_num"]),
            "surface":    first.get("surface_raw", ""),
            "distance_m": _nan_to_none(first.get("distance_m")),
            "condition":  first.get("condition_raw", ""),
            "horses":     horses_out,
        })

    return {
        "generated_at":  generated_at,
        "updated_at":    generated_at,   # index.html 後方互換
        "target_date":   target_date,
        "filter_conditions": {
            "min_pred_prob": MIN_PRED_PROB,
            "max_odds":      MAX_ODDS,
        },
        "total_bets":    len(bets),
        "total_races":   len(races_out),
        "races":         races_out,
    }


# ─── メイン ──────────────────────────────────────────────────────────────────

def main(date_str: str | None = None):
    if date_str is None:
        date_str = datetime.now(JST).strftime("%Y%m%d")

    target_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    logger.info(f"=== 予測開始: {target_date} ===")

    # ── 1. モデル読み込み ──
    if not MODEL_TXT.exists():
        logger.error(f"モデルが見つかりません: {MODEL_TXT}")
        sys.exit(1)
    model = lgb.Booster(model_file=str(MODEL_TXT))
    logger.info(f"モデル読み込み: {MODEL_TXT.name} (rounds={model.num_trees()})")

    # ── 2. 出馬表スクレイピング ──
    logger.info("出馬表スクレイピング中...")
    races = scrape_today(date_str)

    if not races:
        logger.warning("出走馬が取得できませんでした。空の JSON を出力します。")
        result = {
            "generated_at":  datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            "updated_at":    datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            "target_date":   target_date,
            "filter_conditions": {"min_pred_prob": MIN_PRED_PROB, "max_odds": MAX_ODDS},
            "total_bets":    0,
            "total_races":   0,
            "races":         [],
            "note":          "出走馬データを取得できませんでした",
        }
        _save(result)
        return

    total_horses = sum(len(r["horses"]) for r in races)
    logger.info(f"取得: {len(races)}レース {total_horses}頭")

    # ── 3. DataFrame 化 & 特徴量結合 ──
    df = races_to_df(races)
    df = merge_stats(df)
    df = add_popularity(df)

    # ── 4. 予測 ──
    # None を NaN に変換して LightGBM が受け付ける float 型に統一
    X = df[FEATURE_COLS].astype(float)
    df["pred_prob"] = model.predict(X)
    logger.info(f"予測完了: pred_prob 中央値={df['pred_prob'].median():.4f}")

    # ── 5. フィルタ & JSON 保存 ──
    result = build_json(df, target_date)
    logger.info(
        f"フィルタ後: {result['total_bets']}買い目 / {result['total_races']}レース "
        f"(pred>={MIN_PRED_PROB}, odds<={MAX_ODDS})"
    )
    _save(result)


def _save(data: dict):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"保存: {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="対象日 YYYYMMDD (省略時: JST 今日)")
    args = parser.parse_args()
    main(args.date)
