"""2/4  Veriyi temizle.

    uv run python clean.py
    uv run python clean.py --keep-all-categories

Yaptiklari ve nedenleri:
  * Kategori filtresi  - belirli gruplari asagilamak uzerine kurulu kategoriler
                         atiliyor. Icerik tercihi, teknik zorunluluk degil.
  * HTML unescape      - kaynak &amp; &quot; &#39; iceriyor.
  * Unicode NFKC + ASCII noktalama - egri tirnak/tire, duz muadillerine
                         cevriliyor. NFKC bunlari KENDILIGINDEN cevirmez.
                         Cevirmezsek " ile " ayri merge harciyor; 767 merge
                         butcesinde bu israf.
  * Bosluk normalizasyonu - CRLF -> LF, satir sonu bosluklari, 3+ bos satir.
  * Dedup + uzunluk filtresi - ayni saka defalarca geciyor; tekrar eden metin
                         merge istatistiklerini sisiriyor.
  * Kucuk harfe CEVIRMIYORUZ - 'The'/'the' ayri merge yiyor ama vaka bilgisi
                         veri; --lowercase ile deneyip farki olcebilirsin.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import unicodedata
from pathlib import Path

DATA = Path(__file__).parent / "data"

DROP_CATEGORIES = {
    "Yo Momma", "Yo Mama", "Blond", "Blonde", "Insults",
    "Redneck", "Gross", "Men / Women", "Religious", "News / Politics",
}

PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2013": "-", "\u2014": "--", "\u2015": "--", "\u2212": "-",
    "\u2026": "...", "\u00a0": " ", "\u2009": " ", "\u202f": " ",
    "\ufeff": "",
}


def normalise(s: str) -> str:
    s = html.unescape(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = unicodedata.normalize("NFKC", s)
    for a, b in PUNCT_MAP.items():
        s = s.replace(a, b)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-all-categories", action="store_true")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--heldout-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    raw = json.loads((DATA / "raw.json").read_text(encoding="utf-8"))
    n0 = len(raw)
    drop = set() if a.keep_all_categories else DROP_CATEGORIES

    kept, seen, counters = [], set(), {"kategori": 0, "uzunluk": 0, "tekrar": 0}
    for d in raw:
        if d["category"] in drop:
            counters["kategori"] += 1
            continue
        body = normalise(d["body"])
        if a.lowercase:
            body = body.lower()
        if not a.min_chars <= len(body) <= a.max_chars:
            counters["uzunluk"] += 1
            continue
        key = re.sub(r"\W+", "", body.lower())
        if key in seen:
            counters["tekrar"] += 1
            continue
        seen.add(key)
        kept.append(body)

    random.Random(a.seed).shuffle(kept)
    cut = int(len(kept) * (1 - a.heldout_frac))
    train, held = kept[:cut], kept[cut:]

    for name, part in (("train", train), ("heldout", held)):
        p = DATA / f"{name}{a.tag}.txt"
        p.write_text("\n\n".join(part) + "\n", encoding="utf-8")

    def stat(name: str, part: list[str]) -> None:
        b = sum(len(x.encode("utf-8")) for x in part)
        print(f"{name:8s} {len(part):6,} saka  {b:10,} bayt  "
              f"ort. {b // max(len(part), 1):4d} bayt/saka")

    print(f"girdi   {n0:6,} saka")
    for k, v in counters.items():
        print(f"  atildi ({k}): {v:,}")
    print()
    stat("train", train)
    stat("heldout", held)
    print(f"\n-> {DATA}/train{a.tag}.txt, {DATA}/heldout{a.tag}.txt")


if __name__ == "__main__":
    main()