"""
当日の出馬表スクレイパー

netkeiba.com から当日開催レースの出走馬情報を取得する。

URL 構造:
  レース一覧: https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={YYYYMMDD}
  出馬表:     https://race.netkeiba.com/race/shutuba.html?race_id={race_id}

race_list_sub.html はJavaScriptで動的描画されるため、静的HTMLには
グレードレース等の一部リンクしか含まれない（4件程度）。
そこで取得した race_id のプレフィックス (YYYYVVKKDD, 10桁) ごとに
R01〜R12 を全生成することで全レースを補完する。

過去日付では race_list_sub.html が 404 を返すため、
race_result_sub.html / db.netkeiba.com をフォールバックとして使用する。
"""

import asyncio
import json
import logging
import random
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup

# ─── 定数 ────────────────────────────────────────────────────────────────────

JST = timezone(timedelta(hours=9))

RACE_LIST_URL    = "https://race.netkeiba.com/top/race_list_sub.html"
RACE_RESULT_URL  = "https://race.netkeiba.com/top/race_result_sub.html"
DB_RACE_LIST_URL = "https://db.netkeiba.com/"
SHUTUBA_URL      = "https://race.netkeiba.com/race/shutuba.html"
SHUTUBA_SP_URL   = "https://race.sp.netkeiba.com/race/shutuba.html"
ODDS_URL         = "https://race.netkeiba.com/odds/index.html"

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

async def _fetch_race_ids_from_url(
    session: aiohttp.ClientSession,
    url: str,
    encoding: str = "utf-8",
) -> list[str]:
    """指定URLからrace_idを収集して返す。"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                logger.warning(f"race_list fetch failed: HTTP {resp.status} ({url})")
                return []
            html = await resp.text(encoding=encoding, errors="replace")
    except Exception as e:
        logger.warning(f"race_list fetch error ({url}): {e}")
        return []

    # 生HTMLテキストから直接検索（JavaScript内の記述も含む）
    race_ids: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"race_id=(\d{12,})", html):
        rid = m.group(1)
        if rid not in seen:
            seen.add(rid)
            race_ids.append(rid)

    return race_ids


async def fetch_race_ids(
    session: aiohttp.ClientSession,
    date_str: str,          # "YYYYMMDD"
) -> list[str]:
    """
    当日の全 race_id を返す。

    race_list_sub.html はJavaScriptで動的描画されるため、静的HTMLには
    グレードレース等の一部リンクしか含まれない。
    そこで取得できた race_id のプレフィックス (YYYYVVKKDD, 10桁) ごとに
    R01〜R12 を全生成し、全レースを取得できるよう補完する。

    過去日付では race_list_sub.html が 404 を返すため、
    race_result_sub.html / db.netkeiba.com をフォールバックとして試みる。
    """
    # 試行する URL リスト（上から順に試し、seed IDs が取得できた時点で使用）
    candidate_urls = [
        (f"{RACE_LIST_URL}?kaisai_date={date_str}", "utf-8"),
        (f"{RACE_RESULT_URL}?kaisai_date={date_str}", "utf-8"),
        (f"{DB_RACE_LIST_URL}?pid=race_list&date={date_str}", "euc_jp"),
        (f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}", "utf-8"),
        (f"https://race.netkeiba.com/top/race_result.html?kaisai_date={date_str}", "utf-8"),
    ]

    seed_ids: list[str] = []
    for url, enc in candidate_urls:
        seed_ids = await _fetch_race_ids_from_url(session, url, enc)
        if seed_ids:
            logger.info(f"race_list: {date_str} → seed {len(seed_ids)}件 ({url})")
            break

    if not seed_ids:
        logger.warning(f"race_list: {date_str} → seed IDs が見つかりませんでした")
        return []

    # ── プレフィックス(YYYYVVKKDD)ごとに R01〜R12 を全生成 ──────────────────
    # race_id 構造: YYYYVVKKDDNN
    #   VV=会場コード  KK=回(kai)  DD=日(nichi)  NN=レース番号
    # 静的HTMLには注目レースしかリンクされないため、
    # 見つかったIDのNN部分を01-12に置換して全レースを補完する。
    prefixes: list[str] = []
    seen_pfx: set[str] = set()
    for rid in seed_ids:
        if len(rid) >= 12:
            pfx = rid[:10]
            if pfx not in seen_pfx:
                seen_pfx.add(pfx)
                prefixes.append(pfx)

    all_ids: list[str] = []
    seen_all: set[str] = set()
    for pfx in prefixes:
        for race_num in range(1, 13):
            rid = f"{pfx}{race_num:02d}"
            if rid not in seen_all:
                seen_all.add(rid)
                all_ids.append(rid)

    logger.info(
        f"race_list: {date_str} → seed {len(seed_ids)}件 / "
        f"プレフィックス {len(prefixes)}会場 / 展開後 {len(all_ids)}件"
    )
    return all_ids


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
    for tag in soup.find_all(["h1", "h2", "div"], class_=re.compile(r"RaceName|race_name", re.I)):
        text = tag.get_text(strip=True)
        if text and not race_name:
            # レースナビゲーション文字列 ("1R2R3R...") を除外
            if not re.search(r"(\d+R){2,}", text):
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

    def _text(tag) -> str:
        return tag.get_text(strip=True) if tag else ""

    horses = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 10:
            continue

        # ── 馬番: CheckMark td 内の select id="mark_X" から取得 ──
        # （Umaban td はJS描画のため空）
        horse_num = None
        check_td = row.find("td", class_=re.compile(r"CheckMark", re.I))
        if check_td:
            sel = check_td.find("select")
            if sel:
                m = re.search(r"mark_(\d+)", sel.get("id", ""))
                if m:
                    horse_num = int(m.group(1))
        if horse_num is None:
            continue

        # ── 馬名 + horse_id: HorseInfo > span.HorseName > a ──
        horse_name = ""
        horse_id   = ""
        info_td = row.find("td", class_=re.compile(r"HorseInfo", re.I))
        if info_td:
            a = info_td.find("a", href=re.compile(r"/horse/"))
            if a:
                horse_name = a.get_text(strip=True)
                m = re.search(r"/horse/(\d+)", a["href"])
                if m:
                    horse_id = m.group(1)
        if not horse_name:
            continue

        # ── 騎手: td.Jockey ──
        jockey = ""
        jockey_td = row.find("td", class_=re.compile(r"^Jockey$", re.I))
        if jockey_td:
            a = jockey_td.find("a")
            jockey = _text(a) if a else _text(jockey_td)

        # ── 斤量: cells[5] (Txt_C) ──
        weight_carried = None
        try:
            weight_carried = float(_text(cells[5]))
        except (ValueError, IndexError):
            pass

        # ── 馬体重: cells[8] (Weight) - JS描画のため通常空 ──
        horse_weight_kg = None
        if len(cells) > 8:
            m = re.search(r"^(\d+)", _text(cells[8]))
            if m:
                horse_weight_kg = int(m.group(1))

        # ── オッズ: cells[9] (Txt_R Popular) - "---.-" は None ──
        odds = None
        if len(cells) > 9:
            try:
                odds = float(_text(cells[9]).replace(",", ""))
            except ValueError:
                pass

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


# ─── 単勝オッズ取得 ──────────────────────────────────────────────────────────

ODDS_JSON_API_URL = "https://race.netkeiba.com/api/api_get_jra_odds.html"


async def _fetch_odds_json(
    session: aiohttp.ClientSession,
    race_id: str,
) -> dict[int, tuple[Optional[float], Optional[int]]]:
    """
    JSON API から 馬番 → (オッズ, 人気) を返す。
    取得失敗時は {}。
    netkeiba のオッズページは JS 動的レンダリングのため、
    静的 HTML ではなく API エンドポイントを直接叩く。
    """
    url = f"{ODDS_JSON_API_URL}?race_id={race_id}&type=1&action=update"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                logger.warning(f"odds JSON API HTTP {resp.status}: {race_id}")
                return {}
            text = await resp.text(encoding="utf-8", errors="replace")
            logger.info(f"  odds JSON API raw({race_id}): HTTP200 len={len(text)} first200={text[:200]!r}")
    except Exception as e:
        logger.warning(f"odds JSON API error {race_id}: {e}")
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"odds JSON parse error {race_id}: {e} / text[:200]={text[:200]!r}")
        return {}

    # 実際のレスポンス構造（2026年3月確認）:
    # {"data": {"odds": {"1": {"01":["340.4","","15"], "02":[...], ...}, "2":{...}}}}
    # - data.odds["1"] = 単勝 (type=1)
    # - 馬番は "01","02",...のゼロ埋め2桁
    # - 値は [オッズ, 複勝下限, 人気] の3要素リスト
    try:
        inner = data.get("data", {})
        odds_by_type = inner.get("odds") or {}
        # 単勝 (type "1") を取得
        odds_raw = odds_by_type.get("1") or {}
    except AttributeError:
        logger.warning(f"odds JSON unexpected structure {race_id}: {str(data)[:200]}")
        return {}

    if not odds_raw:
        logger.warning(f"odds JSON empty odds.1 field {race_id}: data keys={list(inner.keys()) if isinstance(inner, dict) else type(inner)}")
        return {}

    odds_map: dict[int, float] = {}
    ninki_map: dict[int, int] = {}

    for horse_num_str, val in odds_raw.items():
        # 馬番: "01" → 1, "16" → 16
        try:
            horse_num = int(horse_num_str)
        except (ValueError, TypeError):
            continue

        # val = ["340.4", "", "15"] (index0=オッズ, index1=複勝下限, index2=人気)
        if isinstance(val, (list, tuple)) and len(val) >= 1:
            try:
                o = float(str(val[0]).replace(",", ""))
                if 1.0 <= o <= 9999.9:
                    odds_map[horse_num] = o
                    # 人気は index 2
                    if len(val) >= 3 and str(val[2]).strip():
                        ninki_map[horse_num] = int(str(val[2]))
            except (ValueError, TypeError):
                pass
        # 後方互換: 単純文字列
        elif isinstance(val, str):
            try:
                o = float(val.replace(",", ""))
                if 1.0 <= o <= 9999.9:
                    odds_map[horse_num] = o
            except ValueError:
                pass

    if not odds_map:
        logger.warning(f"odds JSON: no valid odds parsed for {race_id} / raw keys={list(odds_raw.keys())[:5]} / sample={list(odds_raw.items())[:2]}")
        return {}

    # 人気がAPIから取得できなかった場合はオッズ昇順で計算
    if not ninki_map:
        sorted_nums = sorted(odds_map, key=lambda n: odds_map[n])
        ninki_map = {num: rank + 1 for rank, num in enumerate(sorted_nums)}

    logger.info(f"  odds JSON API: {race_id} → {len(odds_map)}頭分取得")
    return {n: (odds_map[n], ninki_map.get(n)) for n in odds_map}


async def _fetch_odds_html(
    session: aiohttp.ClientSession,
    race_id: str,
) -> dict[int, tuple[Optional[float], Optional[int]]]:
    """
    単勝オッズ HTML ページから 馬番 → (オッズ, 人気) を返す。
    JSON API のフォールバック。取得失敗時は {}。
    """
    url = f"{ODDS_URL}?race_id={race_id}&type=b1"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                logger.debug(f"odds HTML page HTTP {resp.status}: {race_id}")
                return {}
            html = await resp.text(encoding="euc_jp", errors="replace")
    except Exception as e:
        logger.debug(f"odds HTML page error {race_id}: {e}")
        return {}

    soup = BeautifulSoup(html, "lxml")

    # 単勝テーブル候補: id/class で複数パターン試行
    table = (
        soup.find("table", id=re.compile(r"odds_tan", re.I))
        or soup.find("table", class_=re.compile(r"OddsTansho|odds_tan", re.I))
        or soup.find("table", id=re.compile(r"OddsTansho", re.I))
    )
    if table is None:
        logger.debug(f"odds HTML table not found: {race_id}")
        return {}

    odds_map: dict[int, float] = {}
    ninki_map: dict[int, int] = {}

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        # 馬番: 最初の 1-18 の整数セル
        horse_num = None
        for td in cells[:3]:
            try:
                n = int(td.get_text(strip=True))
                if 1 <= n <= 18:
                    horse_num = n
                    break
            except ValueError:
                continue
        if horse_num is None:
            continue

        # オッズ: "数字.数字" パターン、人気: 整数
        found_odds = False
        for td in cells:
            txt = td.get_text(strip=True).replace(",", "")
            if not found_odds and re.match(r"^\d+\.\d+$", txt):
                try:
                    v = float(txt)
                    if 1.0 <= v <= 9999.9:
                        odds_map[horse_num] = v
                        found_odds = True
                except ValueError:
                    pass
            elif found_odds and re.match(r"^\d+$", txt):
                try:
                    ninki_map[horse_num] = int(txt)
                    break
                except ValueError:
                    pass

    if not odds_map:
        return {}

    if not ninki_map:
        sorted_nums = sorted(odds_map, key=lambda n: odds_map[n])
        ninki_map = {num: rank + 1 for rank, num in enumerate(sorted_nums)}

    logger.info(f"  odds HTML fallback: {race_id} → {len(odds_map)}頭分取得")
    return {n: (odds_map[n], ninki_map.get(n)) for n in odds_map}


async def _fetch_odds(
    session: aiohttp.ClientSession,
    race_id: str,
) -> dict[int, tuple[Optional[float], Optional[int]]]:
    """
    単勝オッズを取得。JSON API → HTML の順で試行。
    取得失敗時は {}。
    """
    result = await _fetch_odds_json(session, race_id)
    if result:
        return result

    await asyncio.sleep(random.uniform(0.2, 0.5))
    return await _fetch_odds_html(session, race_id)


# ─── 出馬表フェッチ ───────────────────────────────────────────────────────────

async def fetch_shutuba(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    race_id: str,
) -> Optional[dict]:
    """
    PCサイトで出馬表を取得。パースできなければ SP サイト(EUC-JP)でも試みる。
    """
    async with semaphore:
        await asyncio.sleep(random.uniform(0.5, 1.0))

        # PC サイト (EUC-JP)
        url_pc = f"{SHUTUBA_URL}?race_id={race_id}"
        try:
            async with session.get(url_pc, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status == 200:
                    html = await resp.text(encoding="euc_jp", errors="replace")
                    result = _parse_shutuba(race_id, html)
                    if result:
                        logger.info(f"OK(PC) {race_id}: {result['race_name']} ({len(result['horses'])}頭)")
                        # 出馬表のオッズ列はJS描画で常に空のため、常に単勝オッズAPIで補完
                        await _fill_odds(session, race_id, result)
                        return result
                else:
                    logger.warning(f"shutuba PC failed {race_id}: HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"shutuba PC error {race_id}: {e}")

        # SP サイト フォールバック (EUC-JP)
        await asyncio.sleep(random.uniform(0.3, 0.6))
        url_sp = f"{SHUTUBA_SP_URL}?race_id={race_id}"
        try:
            async with session.get(url_sp, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status == 200:
                    html = await resp.text(encoding="euc_jp", errors="replace")
                    result = _parse_shutuba(race_id, html)
                    if result:
                        logger.info(f"OK(SP) {race_id}: {result['race_name']} ({len(result['horses'])}頭)")
                        await _fill_odds(session, race_id, result)
                        return result
                    else:
                        logger.warning(f"shutuba SP parse failed {race_id}")
                else:
                    logger.warning(f"shutuba SP failed {race_id}: HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"shutuba SP error {race_id}: {e}")

    return None


async def _fill_odds(
    session: aiohttp.ClientSession,
    race_id: str,
    result: dict,
) -> None:
    """出馬表の odds が全 None のとき単勝オッズページで補完する（in-place）。"""
    await asyncio.sleep(random.uniform(0.3, 0.6))
    odds_data = await _fetch_odds(session, race_id)
    if not odds_data:
        return
    filled = 0
    for h in result["horses"]:
        entry = odds_data.get(h["horse_num"])
        if entry:
            h["odds"], h["popularity"] = entry
            filled += 1
    if filled:
        logger.info(f"  odds補完({filled}頭): {race_id}")


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
