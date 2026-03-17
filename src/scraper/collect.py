"""
netkeiba.com JRA競馬データ収集スクリプト

Usage:
    python collect.py --year 2023
    python collect.py --year 2023 --venue 05  # 東京のみ
"""

import argparse
import csv
import logging
import random
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ─── 設定 ────────────────────────────────────────────────────────────────────

BASE_URL = "https://db.netkeiba.com"
ENCODING = "EUC-JP"

VENUES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FIELDNAMES = [
    "race_id", "race_name", "date", "venue", "round", "day", "race_num",
    "distance", "surface", "track_condition", "weather",
    "order", "horse_name", "horse_id", "jockey", "weight_carried",
    "time", "last_3f", "odds", "popularity", "horse_weight",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ─── ログ設定 ─────────────────────────────────────────────────────────────────

def setup_logger(year: int) -> logging.Logger:
    log_path = DATA_DIR / f"{year}_collect.log"
    logger = logging.getLogger("collect")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─── スクレイピング ───────────────────────────────────────────────────────────

def fetch_html(url: str) -> bytes | None:
    """HTMLを取得して生バイト列を返す（EUC-JP前提）。"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as e:
        return None


def parse_race_page(race_id: str, html: bytes) -> list[dict]:
    """レース結果ページをパースしてレコードリストを返す。"""
    soup = BeautifulSoup(html, "lxml", from_encoding=ENCODING)

    # ── レース基本情報 ─────────────────────────────────────────────
    race_name = ""
    name_tag = soup.find("h1", class_=re.compile(r"RaceName"))
    if name_tag:
        race_name = name_tag.get_text(strip=True)

    # 日付・競馬場
    race_data_tag = soup.find("div", class_=re.compile(r"RaceData01"))
    date_str = ""
    if race_data_tag:
        spans = race_data_tag.find_all("span")
        if spans:
            date_str = spans[0].get_text(strip=True)  # 例: 2023年1月5日

    # 距離・芝ダート・天気・馬場
    surface, distance, weather, track_condition = "", "", "", ""
    race_data2 = soup.find("div", class_=re.compile(r"RaceData02"))
    if race_data2:
        text = race_data2.get_text(" ", strip=True)
        # 距離・芝ダート
        m = re.search(r"([芝ダ障])(\d+)m", text)
        if m:
            surface = "芝" if m.group(1) == "芝" else ("障" if m.group(1) == "障" else "ダート")
            distance = m.group(2)
        # 天気
        m2 = re.search(r"天候\s*[:：]\s*(\S+)", text)
        if m2:
            weather = m2.group(1)
        # 馬場
        m3 = re.search(r"馬場\s*[:：]\s*(\S+)", text)
        if m3:
            track_condition = m3.group(1)

    # race_id の構造から venue/round/day/race_num を取り出す
    # race_id = {year(4)}{venue(2)}{round(2)}{day(2)}{race_num(2)}
    year_str   = race_id[0:4]
    venue_code = race_id[4:6]
    round_num  = race_id[6:8].lstrip("0") or "0"
    day_num    = race_id[8:10].lstrip("0") or "0"
    race_num   = race_id[10:12].lstrip("0") or "0"
    venue_name = VENUES.get(venue_code, venue_code)

    base = {
        "race_id": race_id,
        "race_name": race_name,
        "date": date_str,
        "venue": venue_name,
        "round": round_num,
        "day": day_num,
        "race_num": race_num,
        "distance": distance,
        "surface": surface,
        "track_condition": track_condition,
        "weather": weather,
    }

    # ── 着順テーブル ──────────────────────────────────────────────
    records = []
    result_table = soup.find("table", class_=re.compile(r"ResultTableWrap|race_table_01"))
    if result_table is None:
        return []

    rows = result_table.find_all("tr")[1:]  # ヘッダ除外
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 10:
            continue

        def cell(i):
            return cells[i].get_text(strip=True) if i < len(cells) else ""

        # horse_id を href から取得
        horse_id = ""
        horse_link = row.find("a", href=re.compile(r"/horse/"))
        if horse_link:
            m = re.search(r"/horse/(\d+)", horse_link["href"])
            if m:
                horse_id = m.group(1)

        rec = {**base,
            "order":          cell(0),
            "horse_name":     cell(3),
            "horse_id":       horse_id,
            "jockey":         cell(6),
            "weight_carried": cell(5),
            "time":           cell(7),
            "last_3f":        cell(11),
            "odds":           cell(12),
            "popularity":     cell(13),
            "horse_weight":   cell(14),
        }
        records.append(rec)

    return records


# ─── race_id 生成 ─────────────────────────────────────────────────────────────

def generate_race_ids(year: int, venue_filter: str | None = None) -> list[str]:
    """
    指定年のすべての race_id 候補を生成する。
    netkeiba の構造: 各開催 最大12R × 最大8日 × 回 最大6 × 競馬場10場
    実際に存在しないレースは fetch 時にスキップする。
    """
    ids = []
    venues = [venue_filter] if venue_filter else list(VENUES.keys())
    for v in venues:
        for r in range(1, 7):       # 回 1〜6
            for d in range(1, 9):   # 日 1〜8
                for n in range(1, 13):  # レース番号 1〜12
                    ids.append(f"{year}{v}{r:02d}{d:02d}{n:02d}")
    return ids


# ─── メイン処理 ──────────────────────────────────────────────────────────────

def collect(year: int, venue_filter: str | None, logger: logging.Logger) -> None:
    out_path = DATA_DIR / f"{year}_races.csv"
    race_ids = generate_race_ids(year, venue_filter)

    logger.info(f"収集開始: year={year}, venue={venue_filter or 'all'}, 候補={len(race_ids)}件")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        skipped = 0
        saved = 0

        for race_id in race_ids:
            url = f"{BASE_URL}/race/{race_id}/"
            content = fetch_html(url)

            if content is None:
                logger.warning(f"SKIP (fetch error): {race_id}")
                skipped += 1
                time.sleep(random.uniform(2, 4))
                continue

            try:
                records = parse_race_page(race_id, content)
            except Exception as e:
                logger.warning(f"SKIP (parse error): {race_id} - {e}")
                skipped += 1
                time.sleep(random.uniform(2, 4))
                continue

            if not records:
                # レースが存在しない（空ページ）は静かにスキップ
                logger.debug(f"NO DATA: {race_id}")
                time.sleep(random.uniform(2, 4))
                continue

            writer.writerows(records)
            f.flush()
            saved += len(records)
            logger.info(f"OK: {race_id} ({len(records)}頭)")

            time.sleep(random.uniform(2, 4))

    logger.info(f"完了: {saved}レコード保存, {skipped}件スキップ → {out_path}")


# ─── エントリーポイント ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="netkeiba JRAレース結果収集")
    parser.add_argument("--year",  type=int, required=True, help="対象年 (例: 2023)")
    parser.add_argument("--venue", type=str, default=None,
                        help="競馬場コード絞り込み (例: 05=東京)")
    args = parser.parse_args()

    logger = setup_logger(args.year)
    collect(args.year, args.venue, logger)
