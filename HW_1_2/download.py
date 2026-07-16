"""1/4  Veriyi indir.

    uv run python download.py

Kaynak: taivop/joke-dataset -> wocka.json (wocka.com arsivinden 10.019 saka,
kategori etiketli, ~7 MB JSON). Kucuk metinler + cok tekrar eden kaliplar =
BPE merge'lerinin ne ogrendigi ciplak gozle okunabilir.
"""

from __future__ import annotations

import json
import urllib.request
from collections import Counter
from pathlib import Path

URL = "https://raw.githubusercontent.com/taivop/joke-dataset/master/wocka.json"
OUT = Path(__file__).parent / "data" / "raw.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read()
    OUT.write_bytes(blob)

    data = json.loads(blob)
    print(f"{URL}\n-> {OUT}  ({len(blob):,} bayt, {len(data):,} kayit)")
    print(f"alanlar: {sorted(data[0])}")
    print("\nkategoriler:")
    for k, v in Counter(d["category"] for d in data).most_common():
        print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    main()