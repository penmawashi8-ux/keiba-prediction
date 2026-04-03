"""
スクレイパー診断スクリプト

SP版netkeibaのURLにアクセスして race_id 取得可否・HTML構造を確認し
docs/predictions/debug.json に結果を書き出す。
"""

import asyncio
import json
import pathlib
import re
import sys

import aiohttp
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://race.netkeiba.com/",
}

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT  = ROOT / "docs" / "predictions" / "debug.json"


def extract_race_ids(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    ids: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"race_id=(\d{12,})", a["href"])
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            ids.append(m.group(1))
    for tag in soup.find_all(True):
        for v in tag.attrs.values():
            if isinstance(v, str):
                for m2 in re.finditer(r"race_id=(\d{12,})", v):
                    if m2.group(1) not in seen:
                        seen.add(m2.group(1))
                        ids.append(m2.group(1))
    return ids


async def run(date_str: str) -> None:
    results: dict = {}

    async with aiohttp.ClientSession(headers=HEADERS) as s:
        # race_list URLパターンを複数試す
        for key, url in [
            ("race_list_sub",    f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"),
            ("race_list",        f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}"),
            ("race_result",      f"https://race.netkeiba.com/top/race_result.html?kaisai_date={date_str}"),
            ("race_result_sub",  f"https://race.netkeiba.com/top/race_result_sub.html?kaisai_date={date_str}"),
            ("db_race_list",     f"https://db.netkeiba.com/?pid=race_list&date={date_str}"),
        ]:
            try:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=20)) as r:
                    html = await r.text(encoding="utf-8", errors="replace")
                    ids = extract_race_ids(html)
                    results[key] = {
                        "url":      url,
                        "status":   r.status,
                        "race_ids": ids,
                        "html_head": html[:400],
                    }
                    print(f"[{r.status}] {key}: {len(ids)} race_ids → {ids[:5]}")
                    print(f"  html: {html[:150].replace(chr(10), ' ')}")
            except Exception as e:
                results[key] = {"url": url, "error": str(e)}
                print(f"ERROR {key}: {e}")

        # 既知shutuba: 中山1R 3/15 (202606020601)
        for known_id in ["202606020601", "202606020608"]:
            url2 = f"https://race.netkeiba.com/race/shutuba.html?race_id={known_id}"
            try:
                async with s.get(url2, timeout=aiohttp.ClientTimeout(total=20)) as r:
                    html2 = await r.text(encoding="utf-8", errors="replace")
                    results[f"shutuba_{known_id}"] = {
                        "url":      url2,
                        "status":   r.status,
                        "html_head": html2[:600],
                    }
                    print(f"\n[{r.status}] shutuba {known_id}")
                    print(f"  html: {html2[:300].replace(chr(10), ' ')}")
            except Exception as e:
                results[f"shutuba_{known_id}"] = {"url": url2, "error": str(e)}
                print(f"ERROR shutuba {known_id}: {e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\ndiagnostics → {OUT}")


if __name__ == "__main__":
    date_str = sys.argv[1] if len(sys.argv) > 1 else "20260315"
    asyncio.run(run(date_str))
