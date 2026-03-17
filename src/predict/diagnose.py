"""
診断スクリプト: shutuba.html の HTML 構造を解析して debug.json に保存する

Usage:
    python src/predict/diagnose.py \\
        --date 20260322 \\
        --race-list-html /tmp/netkeiba_resp.html \\
        --shutuba-html /tmp/shutuba_resp.html \\
        --race-list-status 200 \\
        --shutuba-status 200 \\
        --first-race-id 202606020809
"""

import argparse
import json
import pathlib
import re
import sys

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("beautifulsoup4 not installed")
    BeautifulSoup = None


def analyze_shutuba(html: str) -> dict:
    if not BeautifulSoup:
        return {"error": "bs4 not available"}
    soup = BeautifulSoup(html, "lxml")

    # マッチするタグを探す
    matching_tags = []
    for tag in soup.find_all(["table", "div", "section"], class_=re.compile(
        r"Shutuba|shutuba|RaceTable|HorseList|horse|ShutubaWrap", re.I
    ))[:15]:
        matching_tags.append({
            "tag": tag.name,
            "class": tag.get("class", []),
            "id": tag.get("id", ""),
        })

    # table タグを全て探す
    tables = []
    for t in soup.find_all("table")[:10]:
        tables.append({
            "class": t.get("class", []),
            "id": t.get("id", ""),
            "rows": len(t.find_all("tr")),
        })

    # 馬名リンクを探す
    horse_links = [a.get_text(strip=True) for a in soup.find_all("a", href=re.compile(r"/horse/"))][:10]

    # HorseName クラスを探す
    horse_name_tds = []
    for td in soup.find_all(["td", "div"], class_=re.compile(r"HorseName|horsename", re.I))[:5]:
        horse_name_tds.append(td.get_text(strip=True)[:50])

    # ShutubaTable の最初の2行の td クラスを調べる
    shutuba_table_cell_classes = []
    shutuba_tbl = soup.find("table", class_=re.compile(r"ShutubaTable|Shutuba_Table", re.I))
    if shutuba_tbl:
        for row in shutuba_tbl.find_all("tr")[1:4]:  # skip header
            row_classes = []
            for td in row.find_all("td"):
                cls = td.get("class", [])
                row_classes.append({"class": cls, "text": td.get_text(strip=True)[:30]})
            shutuba_table_cell_classes.append(row_classes)

    # レース名
    race_name_tags = []
    for tag in soup.find_all(["h1", "h2", "div"], class_=re.compile(r"RaceName|race_name", re.I))[:3]:
        race_name_tags.append(tag.get_text(strip=True)[:100])

    return {
        "matching_tags": matching_tags,
        "all_tables": tables,
        "horse_links": horse_links,
        "horse_name_tds": horse_name_tds,
        "race_name_tags": race_name_tags,
        "html_len": len(html),
        "shutuba_table_cell_classes": shutuba_table_cell_classes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="")
    parser.add_argument("--race-list-html", default="/tmp/netkeiba_resp.html")
    parser.add_argument("--shutuba-html", default="/tmp/shutuba_resp.html")
    parser.add_argument("--race-list-status", default="N/A")
    parser.add_argument("--race-list-bytes", default="0")
    parser.add_argument("--race-ids-found", default="0")
    parser.add_argument("--shutuba-status", default="N/A")
    parser.add_argument("--shutuba-bytes", default="0")
    parser.add_argument("--first-race-id", default="")
    args = parser.parse_args()

    shutuba_html = ""
    shutuba_path = pathlib.Path(args.shutuba_html)
    if shutuba_path.exists():
        shutuba_html = shutuba_path.read_text(errors="replace")

    analysis = analyze_shutuba(shutuba_html)

    result = {
        "date": args.date,
        "race_list_status": args.race_list_status,
        "race_list_bytes": int(args.race_list_bytes),
        "race_ids_found": int(args.race_ids_found),
        "first_race_id": args.first_race_id,
        "shutuba_status": args.shutuba_status,
        "shutuba_bytes": int(args.shutuba_bytes),
        "shutuba_analysis": analysis,
    }

    out = pathlib.Path("docs/predictions/debug.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
