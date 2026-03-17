"""
当日の出馬表スクレイパー

netkeiba.com から当日開催レースの出走馬情報を取得する。

URL 構造:
  レース一覧: https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={YYYYMMDD}
  出馬表:     https://race.netkeiba.com/race/shutuba.html?race_id={race_id}
"""

import asyncio
import logging
import random
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup

# ─── 定数 ────────────────────────────────────────────────────────────────────

JST = timezone(timedelta(hours=9))

RACE_LIST_URL = "https://race.netkeiba.com/top/race_list_sub.html"
SHUTUBA_URL   = "https://race.netkeiba.com/race/shutuba.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://race.netkeiba.com/",
}

SURFACE_MAP   = {"芝": 0, "ダート": 1, "障": 2}
CONDITION_MAP = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
WEATHER_MAP   = {"晴": 0, "曇": 1, "雨": 2, "小雨": 3}

VENUES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

logger = logging.getLogger(__name__)


# ─── レース一覧取得 ───────────────────────────────────────────────────────────

async def fetch_race_ids(
    session: aiohttp.ClientSession,
    date_str: str,          # "YYYYMMDD"
) -> list[str]:
    """当日の全 race_id を返す。"""
    url = f"{RACE_LIST_URL}?kaisai_date={date_str}"
    logger.info(f"Fetching race list: {url}")
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            logger.info(f"race_list HTTP status: {resp.status}")
            if resp.status != 200:
                logger.warning(f"race_list fetch failed: HTTP {resp.status}")
                return []
            html = await resp.text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"race_list fetch error: {e}")
        return []

    soup = BeautifulSoup(html, "lxml")
    race_ids: list[str] = []
    for a in soup.find_all("a", href=True):
        m = re.search(r"race_id=(\d{12,})", a["href"])
        if m:
            race_ids.append(m.group(1))

    # 重複除去・順序保持
    seen: set[str] = set()
    unique: list[str] = []
    for rid in race_ids:
        if rid not in seen:
            seen.add(rid)
            unique.append(rid)

    logger.info(f"race_list: {date_str} → {len(unique)} races found")
    if len(unique) == 0:
        logger.warning(f"No race_ids in HTML. Total <a> tags: {len(soup.find_all('a'))}")
    return unique


# ─── 出馬表パース ─────────────────────────────────────────────────────────────

def _parse_shutuba(race_id: str, html: str) -> Optional[dict]:
    """
    出馬表 HTML をパースして race_info + horses のdict を返す。
    パース失敗時は None。
    """
    soup = BeautifulSoup(html, "lxml")

    # ── レース情報 ──
    race_name = ""
    surface_raw = ""
    distance_m: Optional[int] = None
    condition_raw = ""
    weather_raw = ""
    race_num = int(race_id[-2:]) if race_id[-2:].isdigit() else 0
    venue_code = race_id[4:6]
    venue = VENUES.get(venue_code, venue_code)

    # タイトル: "RaceName 芝1200m" 等
    for tag in soup.find_all(["h1", "h2", "div"], class_=re.compile(r"RaceName|race_name|RaceNum", re.I)):
        text = tag.get_text(strip=True)
        if text and not race_name:
            # 数字のみのタグ（R番号）は除外
            if not re.fullmatch(r"\d+R?", text):
                race_name = text

    # レースデータ行: "芝1200m（良）" / "発走 10:00 / ダ1800m"
    for tag in soup.find_all(["div", "span", "p"], class_=re.compile(r"RaceData|race_data|RaceInfo", re.I)):
        text = tag.get_text(" ", strip=True)
        # 距離・芝ダ
        m = re.search(r"(芝|ダート|ダ|障)([\d,]+)m", text)
        if m:
            surface_key = m.group(1).replace("ダ", "ダート")
            surface_raw = surface_key
            distance_m = int(m.group(2).replace(",", ""))
        # 馬場状態
        m2 = re.search(r"[（(](良|稍重|重|不良)[）)]", text)
        if m2:
            condition_raw = m2.group(1)
        # 天候
        m3 = re.search(r"天候\s*[：:]\s*(\S+)", text)
        if m3:
            weather_raw = m3.group(1)

    # ── 出走馬テーブル ──
    table = soup.find("table", class_=re.compile(r"Shutuba|shutuba|RaceTable", re.I))
    if table is None:
        # フォールバック: id で探す
        table = soup.find("table", id=re.compile(r"tableWrapper|RaceTable", re.I))
    if table is None:
        logger.warning(f"shutuba table not found: {race_id}")
        return None

    horses = []
    for row in table.find_all("tr"):
        # HorseList クラス or td が十分ある行
        cells = row.find_all("td")
        if len(cells) < 5:
            continue

        def _text(tag) -> str:
            return tag.get_text(strip=True) if tag else ""

        # 馬番: Umaban クラス or 2列目
        umaban_td = row.find("td", class_=re.compile(r"Umaban|umaban", re.I))
        horse_num_str = _text(umaban_td) if umaban_td else _text(cells[1])
        if not horse_num_str.isdigit():
            continue
        horse_num = int(horse_num_str)

        # 馬名 + horse_id
        # 旧: class="HorseName" / 新: class="HorseInfo"
        horse_name = ""
        horse_id   = ""
        name_td    = row.find("td", class_=re.compile(r"HorseInfo|HorseName|horsename", re.I))
        if name_td:
            a = name_td.find("a", href=re.compile(r"/horse/"))
            if a:
                horse_name = a.get_text(strip=True)
                m = re.search(r"/horse/(\d+)", a["href"])
                if m:
                    horse_id = m.group(1)

        # 騎手
        jockey = ""
        jockey_td = row.find("td", class_=re.compile(r"Jockey|jockey", re.I))
        if jockey_td:
            a = jockey_td.find("a")
            jockey = _text(a) if a else _text(jockey_td)

        # 斤量
        # 旧: class="Futan" / 新: class="Txt_C" のみのセル (weight_carried)
        futan_td = row.find("td", class_=re.compile(r"Futan|futan", re.I))
        if futan_td is None:
            # 新構造: class がちょうど ["Txt_C"] だけのセルが斤量
            for td in row.find_all("td"):
                if td.get("class", []) == ["Txt_C"]:
                    futan_td = td
                    break
        weight_carried_str = _text(futan_td) if futan_td else ""
        try:
            weight_carried = float(weight_carried_str)
        except ValueError:
            weight_carried = None

        # 馬体重: "480(+2)" → 480
        # 旧: class="HorseWeight" / 新: class="Weight"
        hw_td = row.find("td", class_=re.compile(r"Weight|HorseWeight|horseweight", re.I))
        horse_weight_kg = None
        if hw_td:
            m = re.search(r"^(\d+)", _text(hw_td))
            if m:
                horse_weight_kg = int(m.group(1))

        # オッズ
        # 旧: class="Odds" / 新: class="Txt_R Popular" など
        odds_td = row.find("td", class_=re.compile(r"Txt_R|Odds|odds", re.I))
        odds = None
        if odds_td:
            try:
                odds = float(_text(odds_td).replace(",", ""))
            except ValueError:
                pass

        if not horse_name:
            continue

        horses.append({
            "horse_num":      horse_num,
            "horse_name":     horse_name,
            "horse_id":       horse_id,
            "jockey":         jockey,
            "weight_carried": weight_carried,
            "horse_weight_kg": horse_weight_kg,
            "odds":           odds,
        })

    if not horses:
        logger.warning(f"no horses parsed: {race_id}")
        return None

    return {
        "race_id":       race_id,
        "race_name":     race_name or f"{race_num}R",
        "venue":         venue,
        "race_num":      race_num,
        "surface_raw":   surface_raw,
        "distance_m":    distance_m,
        "condition_raw": condition_raw,
        "weather_raw":   weather_raw,
        # エンコード済み（NaN fallback）
        "surface_enc":   SURFACE_MAP.get(surface_raw),
        "condition_enc": CONDITION_MAP.get(condition_raw),
        "weather_enc":   WEATHER_MAP.get(weather_raw),
        "horses":        horses,
    }


# ─── 出馬表フェッチ ───────────────────────────────────────────────────────────

async def fetch_shutuba(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    race_id: str,
) -> Optional[dict]:
    url = f"{SHUTUBA_URL}?race_id={race_id}"
    async with semaphore:
        await asyncio.sleep(random.uniform(0.5, 1.0))
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    logger.warning(f"shutuba fetch failed {race_id}: HTTP {resp.status}")
                    return None
                html = await resp.text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"shutuba fetch error {race_id}: {e}")
            return None

    result = _parse_shutuba(race_id, html)
    if result:
        logger.info(f"OK {race_id}: {result['race_name']} ({len(result['horses'])}頭)")
    return result


# ─── メインエントリ ───────────────────────────────────────────────────────────

async def scrape_today_async(date_str: Optional[str] = None) -> list[dict]:
    """
    当日の全レースを非同期スクレイピングし、race dict のリストを返す。
    date_str: "YYYYMMDD" 省略時は JST 今日
    """
    if date_str is None:
        date_str = datetime.now(JST).strftime("%Y%m%d")

    connector = aiohttp.TCPConnector(limit=5)
    semaphore = asyncio.Semaphore(5)

    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        race_ids = await fetch_race_ids(session, date_str)
        if not race_ids:
            logger.warning("No races found for " + date_str)
            return []

        tasks = [fetch_shutuba(session, semaphore, rid) for rid in race_ids]
        results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


def scrape_today(date_str: Optional[str] = None) -> list[dict]:
    return asyncio.run(scrape_today_async(date_str))
