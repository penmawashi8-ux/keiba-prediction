# keiba-prediction

競馬レース結果を予測する機械学習プロジェクトです。
スクレイピングで収集したデータをもとに LightGBM モデルで予測し、GitHub Pages で結果を公開します。

## 概要

- **データ収集**: `src/scraper/` のスクリプトで競馬情報をスクレイピング
- **特徴量エンジニアリング**: `src/features/` で学習用特徴量を生成
- **モデル学習・推論**: `src/model/` で LightGBM モデルを管理
- **公開**: `web/` 以下を GitHub Pages で配信（毎週土曜自動更新）

## ディレクトリ構成

```
keiba-prediction/
├── data/
│   ├── raw/           # 生データ（gitignore対象）
│   └── processed/     # 加工済みデータ（gitignore対象）
├── src/
│   ├── scraper/       # スクレイピングスクリプト
│   ├── features/      # 特徴量エンジニアリング
│   └── model/         # モデル学習・推論
├── web/
│   ├── index.html     # GitHub Pages トップページ
│   └── predictions/   # 予測結果 JSON（公開）
├── .github/
│   └── workflows/
│       └── update_predictions.yml  # 自動更新ワークフロー
├── requirements.txt
└── README.md
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

1. スクレイピング実行:
   ```bash
   python src/scraper/main.py
   ```

2. 特徴量生成:
   ```bash
   python src/features/build_features.py
   ```

3. 予測実行:
   ```bash
   python src/model/predict.py
   ```

予測結果は `web/predictions/` に JSON 形式で出力されます。

## 自動更新

GitHub Actions により毎週土曜 8:00 JST に自動でスクレイピング → 予測 → GitHub Pages 更新が実行されます。

## 技術スタック

- Python 3.11+
- requests / BeautifulSoup4（スクレイピング）
- pandas（データ処理）
- scikit-learn / LightGBM（機械学習）
- GitHub Actions（自動化）
- GitHub Pages（公開）
