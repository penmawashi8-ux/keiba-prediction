"""
LightGBM モデル学習スクリプト

入力: data/processed/features.csv
出力: src/model/lgbm_model.pkl

学習: 2015〜2022年  /  検証: 2023年  /  テスト: 2024年
ターゲット: is_win (1着=1, それ以外=0)
"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FEATURES_PATH  = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.csv"
MODEL_DIR      = Path(__file__).resolve().parent
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH_PKL = MODEL_DIR / "lgbm_model.pkl"   # ローカル用 (gitignore 対象)
MODEL_PATH_TXT = MODEL_DIR / "lgbm_model.txt"   # GitHub Actions 用 (コミット可)
HORSE_STATS    = MODEL_DIR / "horse_stats.csv"   # 馬の最新成績 (コミット可)
JOCKEY_STATS   = MODEL_DIR / "jockey_stats.csv"  # 騎手の最新成績 (コミット可)

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
    "jockey_win_rate_venue",   # 仕様の jockey_win_rate_course に相当
    "popularity",              # レース内オッズ昇順ランク (1=1番人気)
]

LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "learning_rate":    0.05,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "verbose":          -1,
    "seed":             42,
}

# ─── データ読み込み・分割 ─────────────────────────────────────────────────────

def load_and_split():
    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # is_win が NaN（中止・除外）は学習対象外
    df_valid = df.dropna(subset=["is_win"])

    train = df_valid[df_valid["year"] <= 2022]
    valid = df_valid[df_valid["year"] == 2023]
    test  = df_valid[df_valid["year"] == 2024]

    print(f"学習: {len(train):,}行 ({train['year'].min()}〜{train['year'].max()})")
    print(f"検証: {len(valid):,}行 ({valid['year'].min()})")
    print(f"テスト: {len(test):,}行 ({test['year'].min()})")
    print(f"勝率 — 学習:{train['is_win'].mean():.4f}  検証:{valid['is_win'].mean():.4f}  テスト:{test['is_win'].mean():.4f}")
    return train, valid, test, df  # df は stats export 用（NaN行含む全データ）


# ─── 学習 ────────────────────────────────────────────────────────────────────

def train_model(train: pd.DataFrame, valid: pd.DataFrame) -> lgb.Booster:
    X_tr = train[FEATURE_COLS]
    y_tr = train["is_win"].astype(int)
    X_va = valid[FEATURE_COLS]
    y_va = valid["is_win"].astype(int)

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain,          free_raw_data=False)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=callbacks,
    )
    return model


# ─── 評価: AUC ───────────────────────────────────────────────────────────────

def evaluate_auc(model: lgb.Booster, df: pd.DataFrame, label: str) -> float:
    X    = df[FEATURE_COLS]
    y    = df["is_win"].astype(int)
    pred = model.predict(X)
    auc  = roc_auc_score(y, pred)
    print(f"  AUC [{label}]: {auc:.4f}")
    return auc


# ─── 特徴量重要度 ─────────────────────────────────────────────────────────────

def show_importance(model: lgb.Booster):
    imp = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=model.feature_name(),
    ).sort_values(ascending=False)

    print("\n特徴量重要度 Top 10 (gain):")
    print(f"  {'特徴量':<30} {'重要度':>10}")
    print("  " + "-" * 42)
    for name, val in imp.head(10).items():
        bar = "█" * int(val / imp.iloc[0] * 20)
        print(f"  {name:<30} {val:>10,.1f}  {bar}")
    return imp


# ─── 期待値シミュレーション ───────────────────────────────────────────────────

def ev_simulation(model: lgb.Booster, df: pd.DataFrame, label: str):
    """
    予測勝率 > 1/odds (期待値>1) の買い目のみ単勝購入したと仮定して
    的中率・回収率を算出する。
    """
    d = df.dropna(subset=["odds"]).copy()
    d["pred_prob"]    = model.predict(d[FEATURE_COLS])
    d["implied_prob"] = 1.0 / d["odds"]

    bets = d[d["pred_prob"] > d["implied_prob"]]
    if len(bets) == 0:
        print(f"\n期待値シミュレーション [{label}]: 対象買い目なし")
        return

    n      = len(bets)
    n_wins = int(bets["is_win"].sum())
    rec    = (bets["is_win"] * bets["odds"]).sum() / n

    print(f"\n期待値シミュレーション [{label}]:")
    print(f"  対象買い目数    : {n:,}")
    print(f"  的中数          : {n_wins:,}")
    print(f"  的中率          : {n_wins/n:.2%}")
    print(f"  回収率          : {rec:.2%}  (100円賭け換算)")
    print(f"  pred_prob 中央値: {bets['pred_prob'].median():.4f}")
    print(f"  odds 中央値     : {bets['odds'].median():.1f}")


# ─── 閾値チューニング ────────────────────────────────────────────────────────

PROB_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15]
ODDS_LIMITS     = [10, 20, 30, 50, None]   # None = 制限なし

def _recovery(bets: pd.DataFrame) -> tuple[int, float, float]:
    """(買い目数, 的中率, 回収率) を返す。"""
    n = len(bets)
    if n == 0:
        return 0, 0.0, 0.0
    hits = int(bets["is_win"].sum())
    rec  = (bets["is_win"] * bets["odds"]).sum() / n
    return n, hits / n, rec


def tune_threshold(model: lgb.Booster, df: pd.DataFrame, label: str):
    """
    pred_prob 下限 × odds 上限の全組み合わせを探索し、
    回収率が最大の条件を特定する。
    """
    d = df.dropna(subset=["odds"]).copy()
    d["pred_prob"]    = model.predict(d[FEATURE_COLS])
    d["implied_prob"] = 1.0 / d["odds"]
    d = d[d["pred_prob"] > d["implied_prob"]]

    odds_labels = [str(o) if o else "∞" for o in ODDS_LIMITS]
    col_w = 9

    print(f"\n閾値チューニング [{label}]  ─ 回収率 (買い目数)")
    print(f"  {'':<6}  " + "  ".join(
        f"{'≤'+ol:>{col_w}}" if ol != "∞" else f"{'制限なし':>{col_w}}"
        for ol in odds_labels
    ))
    print("  " + "-" * (8 + (col_w + 2) * len(ODDS_LIMITS)))

    best = {"rec": 0.0, "prob": None, "odds": None, "n": 0}
    for prob_th in PROB_THRESHOLDS:
        row_cells = []
        sub = d[d["pred_prob"] >= prob_th]
        for odds_lim in ODDS_LIMITS:
            bets = sub if odds_lim is None else sub[sub["odds"] <= odds_lim]
            n, hit, rec = _recovery(bets)
            row_cells.append(f"{rec:6.1%}({n:,})")
            if rec > best["rec"]:
                best = {"rec": rec, "prob": prob_th, "odds": odds_lim, "n": n}
        print(f"  {prob_th:>5.2f}  " + "  ".join(f"{c:>{col_w}}" for c in row_cells))

    print()
    odds_str = f"≤{best['odds']}" if best["odds"] else "制限なし"
    print(f"  ★ 最高回収率: {best['rec']:.2%}  "
          f"(pred_prob≥{best['prob']}, odds {odds_str}, 買い目={best['n']:,})")


# ─── 人気別回収率分析 ─────────────────────────────────────────────────────────

def popularity_analysis(model: lgb.Booster, df: pd.DataFrame, label: str):
    """
    popularity (レース内オッズ順位) でグループ化し、
    EV+ フィルタ前後の的中率・回収率を出力する。
    """
    d = df.dropna(subset=["odds"]).copy()
    d["pred_prob"]    = model.predict(d[FEATURE_COLS])
    d["implied_prob"] = 1.0 / d["odds"]

    def band(r):
        if r <= 3:  return "1〜3番人気"
        if r <= 6:  return "4〜6番人気"
        return              "7番人気以下"

    d["pop_band"] = d["popularity"].map(band)

    BANDS     = ["1〜3番人気", "4〜6番人気", "7番人気以下"]
    col_names = ["人気帯", "買い目数", "的中数", "的中率", "回収率",
                 "(EV+)買い目", "(EV+)的中率", "(EV+)回収率"]
    W = [12, 8, 7, 8, 8, 10, 10, 10]

    print(f"\n人気別回収率分析 [{label}]")
    print("  " + "  ".join(f"{n:>{w}}" for n, w in zip(col_names, W)))
    print("  " + "-" * (sum(W) + 2 * len(W)))

    for band_name in BANDS:
        g      = d[d["pop_band"] == band_name]
        n      = len(g)
        n_win  = int(g["is_win"].sum())
        rec    = (g["is_win"] * g["odds"]).sum() / n if n else 0.0
        ev     = g[g["pred_prob"] > g["implied_prob"]]
        ne     = len(ev)
        ne_win = int(ev["is_win"].sum())
        rece   = (ev["is_win"] * ev["odds"]).sum() / ne if ne else 0.0
        print(
            f"  {band_name:>{W[0]}}  "
            f"{n:>{W[1]},}  "
            f"{n_win:>{W[2]},}  "
            f"{n_win/n:>{W[3]}.2%}  "
            f"{rec:>{W[4]}.2%}  "
            f"{ne:>{W[5]},}  "
            f"{ne_win/ne if ne else 0:>{W[6]}.2%}  "
            f"{rece:>{W[7]}.2%}"
        )


# ─── 条件別回収率比較 ─────────────────────────────────────────────────────────

def condition_comparison(model: lgb.Booster, valid: pd.DataFrame, test: pd.DataFrame):
    """
    以下 3 条件での買い目数・的中率・回収率を 2023/2024 で比較する。
      A: pred_prob ≥ 0.08 かつ odds ≤ 20
      B: A かつ popularity ≤ 6
      C: A かつ popularity ≤ 9
    """
    CONDITIONS = [
        ("pred≥0.08 & odds≤20",              lambda d: (d["pred_prob"] >= 0.08) & (d["odds"] <= 20)),
        ("pred≥0.08 & odds≤20 & pop≤6",      lambda d: (d["pred_prob"] >= 0.08) & (d["odds"] <= 20) & (d["popularity"] <= 6)),
        ("pred≥0.08 & odds≤20 & pop≤9",      lambda d: (d["pred_prob"] >= 0.08) & (d["odds"] <= 20) & (d["popularity"] <= 9)),
    ]

    col_names = ["条件", "2023買い目", "2023的中率", "2023回収率", "2024買い目", "2024的中率", "2024回収率"]
    W         = [28, 10, 10, 10, 10, 10, 10]

    print("\n条件別回収率比較  (EV+ ベース: pred_prob > 1/odds)")
    print("  " + "  ".join(f"{n:>{w}}" for n, w in zip(col_names, W)))
    print("  " + "-" * (sum(W) + 2 * len(W)))

    for label, dfs in [("2023", valid), ("2024", test)]:
        d = dfs.dropna(subset=["odds"]).copy()
        d["pred_prob"]    = model.predict(d[FEATURE_COLS])
        d["implied_prob"] = 1.0 / d["odds"]
        # EV+ ベース
        if label == "2023":
            valid_pred = d
        else:
            test_pred = d

    rows = []
    for cond_label, cond_fn in CONDITIONS:
        row = [cond_label]
        for d in [valid_pred, test_pred]:
            bets = d[d["pred_prob"] > d["implied_prob"]]  # EV+ フィルタ
            bets = bets[cond_fn(bets)]
            n      = len(bets)
            n_win  = int(bets["is_win"].sum())
            rec    = (bets["is_win"] * bets["odds"]).sum() / n if n else 0.0
            hit    = n_win / n if n else 0.0
            row += [n, hit, rec]
        rows.append(row)

    for r in rows:
        cond, n23, hit23, rec23, n24, hit24, rec24 = r
        print(
            f"  {cond:>{W[0]}}  "
            f"{n23:>{W[1]},}  "
            f"{hit23:>{W[2]}.2%}  "
            f"{rec23:>{W[3]}.2%}  "
            f"{n24:>{W[4]},}  "
            f"{hit24:>{W[5]}.2%}  "
            f"{rec24:>{W[6]}.2%}"
        )


# ─── 馬・騎手の最新成績を export（予測パイプライン用） ────────────────────────

HORSE_HIST_COLS  = [
    "horse_avg_order_3", "horse_avg_order_5",
    "horse_avg_last3f_3", "horse_avg_last3f_5",
    "horse_win_rate_dist", "horse_win_rate_venue", "horse_win_rate_surface",
]
JOCKEY_HIST_COLS = ["jockey_win_rate_100", "jockey_win_rate_venue"]

def export_stats(df: pd.DataFrame):
    """
    全期間データから馬・騎手ごとの「最新の」historical特徴量を抽出し CSV 保存。
    predict_today.py がこれを読み込んで当日出走馬に結合する。
    """
    df_sorted = df.sort_values("date")

    horse_stats = (
        df_sorted.dropna(subset=["horse_id"])
        .groupby("horse_id")[HORSE_HIST_COLS]
        .last()
    )
    horse_stats.to_csv(HORSE_STATS)

    jockey_stats = (
        df_sorted.dropna(subset=["jockey"])
        .groupby("jockey")[JOCKEY_HIST_COLS]
        .last()
    )
    jockey_stats.to_csv(JOCKEY_STATS)

    print(f"stats export: 馬 {len(horse_stats):,}頭 → {HORSE_STATS.name}")
    print(f"stats export: 騎手 {len(jockey_stats):,}人 → {JOCKEY_STATS.name}")


# ─── モデル保存 ──────────────────────────────────────────────────────────────

def save_model(model: lgb.Booster):
    # pickle (.pkl) — ローカル利用 (gitignore 対象)
    with open(MODEL_PATH_PKL, "wb") as f:
        pickle.dump(model, f)
    # LightGBM native text (.txt) — GitHub Actions 用にコミット可能
    model.save_model(str(MODEL_PATH_TXT))
    print(f"\nモデル保存: {MODEL_PATH_PKL.name}  /  {MODEL_PATH_TXT.name}")


# ─── メイン ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LightGBM 学習開始  (特徴量: popularity 追加)")
    print("=" * 60)

    train, valid, test, df_full = load_and_split()

    print("\n--- 学習中 ---")
    model = train_model(train, valid)
    print(f"最適 round: {model.best_iteration}")

    print("\n--- AUC ---")
    evaluate_auc(model, valid, "2023 検証")
    evaluate_auc(model, test,  "2024 テスト")

    show_importance(model)

    ev_simulation(model, valid, "2023 検証")
    ev_simulation(model, test,  "2024 テスト")

    print("\n--- 閾値チューニング ---")
    tune_threshold(model, valid, "2023 検証")
    tune_threshold(model, test,  "2024 テスト")

    print("\n--- 人気別回収率分析 ---")
    popularity_analysis(model, valid, "2023 検証")
    popularity_analysis(model, test,  "2024 テスト")

    print("\n--- 条件別回収率比較 ---")
    condition_comparison(model, valid, test)

    save_model(model)

    print("\n--- 予測パイプライン用 stats export ---")
    export_stats(df_full)

    print("\n完了")


if __name__ == "__main__":
    main()
