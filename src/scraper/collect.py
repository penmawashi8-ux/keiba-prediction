"""
netkeiba.com JRA競馬データ収集スクリプト（非同期並列版）

Usage:
    python collect.py --year 2023
    python collect.py --year 2023 --venue 05  # 東京のみ
    python collect.py --year 2023 --workers 10  # 並列数指定（デフォルト: 10）
"""

import argparse
import asyncio
import csv
import logging
import random
import re
from pathlib import Path

import aiohttp
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
    "race_id", "race_name", "date", "venue", "course", "distance",
    "surface", "condition", "weather",
    "horse_num", "order", "horse_name", "horse_id",
    "jockey", "weight", "time", "last_3f", "odds", "popularity", "horse_weight",
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


# ─── パース ──────────────────────────────────────────────────────────────────

def parse_race_page(race_id: str, html: bytes) -> list[dict]:
    """レース結果ページをパースしてFIELDNAMESのdictのリストを返す。"""
    soup = BeautifulSoup(html, "lxml", from_encoding=ENCODING)

    # race_name / surface / distance / course / weather / condition は
    # dl.racedata に集約されている
    # dd 構造例: "3歳未勝利 芝右2000m / 天候 : 晴 / 芝 : 良 / 発走 : 10:01"
    #           "御堂筋ステークス(3勝) 芝右 外2400m / 天候 : 雨 / 芝 : 重 / ..."
    race_name = ""
    surface, distance, course, weather, condition = "", "", "", "", ""
    racedata_dl = soup.find("dl", class_=re.compile(r"racedata"))
    if racedata_dl:
        text = racedata_dl.get_text(" ", strip=True)
        # race_name: dd の先頭から surface 記号の手前まで
        dd = racedata_dl.find("dd")
        if dd:
            dd_text = dd.get_text(strip=True)
            m_name = re.match(r"^(.+?)\s*[芝ダ障](?=\s*(?:右|左|直|\d))", dd_text)
            race_name = m_name.group(1).strip() if m_name else dd_text
        # surface / distance: "芝右2000m" / "芝右 外2400m" / "ダ1600m"
        # \s* で "右 外" のような空白入りも許容
        m = re.search(r"([芝ダ障])\s*(?:右\s*外|右\s*内|右|左\s*外|左\s*内|左|直線)?\s*(\d+)m", text)
        if m:
            surface = "芝" if m.group(1) == "芝" else ("障" if m.group(1) == "障" else "ダート")
            distance = m.group(2)
        # course direction (空白を正規化: "右 外" → "右外")
        m2 = re.search(r"[芝ダ障]\s*(右\s*外|右\s*内|右|左\s*外|左\s*内|左|直線)", text)
        if m2:
            course = re.sub(r"\s+", "", m2.group(1))
        # weather: "天候 : 晴"
        m3 = re.search(r"天候\s*[:：]\s*(\S+)", text)
        if m3:
            weather = m3.group(1)
        # condition: "芝 : 良" / "ダート : 良" (スペースで course 記号と区別)
        m4 = re.search(r"(?:芝|ダート|障)\s+[:：]\s*(\S+)", text)
        if m4:
            condition = m4.group(1)

    # date: ページ内の日付パターンを検索
    date_str = ""
    raw_text = html.decode(ENCODING, errors="replace")
    m_date = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日)", raw_text)
    if m_date:
        date_str = m_date.group(1)

    venue_code = race_id[4:6]
    venue_name = VENUES.get(venue_code, venue_code)

    base = {
        "race_id":   race_id,
        "race_name": race_name,
        "date":      date_str,
        "venue":     venue_name,
        "course":    course,
        "distance":  distance,
        "surface":   surface,
        "condition": condition,
        "weather":   weather,
    }

    result_table = soup.find("table", class_=re.compile(r"ResultTableWrap|race_table_01"))
    if result_table is None:
        return []

    records = []
    rows = result_table.find_all("tr")[1:]
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 10:
            continue

        def cell(i: int) -> str:
            return cells[i].get_text(strip=True) if i < len(cells) else ""

        horse_id = ""
        horse_link = row.find("a", href=re.compile(r"/horse/"))
        if horse_link:
            m = re.search(r"/horse/(\d+)", horse_link["href"])
            if m:
                horse_id = m.group(1)

        records.append({
            **base,
            "horse_num":    cell(2),
            "order":        cell(0),
            "horse_name":   cell(3),
            "horse_id":     horse_id,
            "jockey":       cell(6),
            "weight":       cell(5),
            "time":         cell(7),
            "last_3f":      cell(11),
            "odds":         cell(12),
            "popularity":   cell(13),
            "horse_weight": cell(14),
        })

    return records


# ─── race_id 生成 ─────────────────────────────────────────────────────────────

def generate_race_ids(year: int, venue_filter: str | None = None) -> list[str]:
    ids = []
    venues = [venue_filter] if venue_filter else list(VENUES.keys())
    for v in venues:
        for r in range(1, 7):
            for d in range(1, 9):
                for n in range(1, 13):
                    ids.append(f"{year}{v}{r:02d}{d:02d}{n:02d}")
    return ids


# ─── 非同期フェッチ ───────────────────────────────────────────────────────────

async def fetch_html(session: aiohttp.ClientSession, url: str) -> bytes | None:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return None
            return await resp.read()
    except Exception:
        return None


async def process_race(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    race_id: str,
    logger: logging.Logger,
) -> list[dict]:
    async with semaphore:
        url = f"{BASE_URL}/race/{race_id}/"
        content = await fetch_html(session, url)

        # ワーカーごとに短いランダム待機（サーバー負荷分散）
        await asyncio.sleep(random.uniform(0.3, 0.5))

        if content is None:
            logger.warning(f"SKIP (fetch error): {race_id}")
            return []

        try:
            records = parse_race_page(race_id, content)
        except Exception as e:
            logger.warning(f"SKIP (parse error): {race_id} - {e}")
            return []

        if records:
            logger.info(f"OK: {race_id} ({len(records)}頭)")
        else:
            logger.debug(f"NO DATA: {race_id}")

        return records


# ─── メイン処理 ──────────────────────────────────────────────────────────────

async def collect_async(year: int, venue_filter: str | None, workers: int, logger: logging.Logger, limit: int | None = None) -> None:
    out_path = DATA_DIR / f"{year}_races.csv"
    race_ids = generate_race_ids(year, venue_filter)
    if limit:
        race_ids = race_ids[:limit]

    logger.info(f"収集開始: year={year}, venue={venue_filter or 'all'}, 候補={len(race_ids)}件, 並列数={workers}")

    semaphore = asyncio.Semaphore(workers)
    connector = aiohttp.TCPConnector(limit=workers)

    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        tasks = [
            process_race(session, semaphore, race_id, logger)
            for race_id in race_ids
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

            saved = 0
            skipped = 0

            for coro in asyncio.as_completed(tasks):
                records = await coro
                if records:
                    writer.writerows(records)
                    f.flush()
                    saved += len(records)
                elif records is not None:
                    skipped += 1

    logger.info(f"完了: {saved}レコード保存, {skipped}件スキップ → {out_path}")


def collect(year: int, venue_filter: str | None, workers: int, logger: logging.Logger, limit: int | None = None) -> None:
    asyncio.run(collect_async(year, venue_filter, workers, logger, limit))


# ─── エントリーポイント ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="netkeiba JRAレース結果収集（並列版）")
    parser.add_argument("--year",    type=int, required=True, help="対象年 (例: 2023)")
    parser.add_argument("--venue",   type=str, default=None,  help="競馬場コード絞り込み (例: 05=東京)")
    parser.add_argument("--workers", type=int, default=10,    help="並列ワーカー数 (デフォルト: 10)")
    parser.add_argument("--limit",   type=int, default=None,  help="処理するrace_idの上限数（テスト用）")
    args = parser.parse_args()

    logger = setup_logger(args.year)
    collect(args.year, args.venue, args.workers, logger, args.limit)
