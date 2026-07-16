"""4/4  Tokenizer'a metin ver, nasil parcaladigini gor.

    uv run python tokenize_demo.py "A horse walks into a bar."
    uv run python tokenize_demo.py --file data/heldout.txt --limit 400
    echo "merhaba" | uv run python tokenize_demo.py
    uv run python tokenize_demo.py                 # interaktif (bos satir = cik)
    uv run python tokenize_demo.py "..." --ids     # her tokenin altinda id

Byte-level gosterim notu: 'Ġ' = bosluk, 'Ċ' = satir sonu. Bunlar tokenizer'in
ic gosterimi; ekranda '.' ve '\\n' olarak gosteriliyor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).parent
DEFAULT = HERE / "out" / "tokenizer_1024.json"

# arka plan renkleri (ANSI 256), acik tonlar; siyah yazi ustune
PALETTE = [153, 157, 224, 230, 189, 216, 195, 223, 183, 194]
RESET = "\033[0m"


def paint(i: int, s: str) -> str:
    return f"\033[48;5;{PALETTE[i % len(PALETTE)]}m\033[38;5;16m{s}{RESET}"


def readable(t: str) -> str:
    return t.replace("Ġ", "·").replace("Ċ", "\\n").replace("ĉ", "\\t")


def render(tok: Tokenizer, text: str, show_ids: bool, width: int = 100) -> None:
    enc = tok.encode(text)
    n_bytes = len(text.encode("utf-8"))
    n_tok = len(enc.ids)

    cells = [readable(t) for t in enc.tokens]
    if show_ids:
        cells = [f"{c}\n{i}" for c, i in zip(cells, enc.ids)]

    # tokenlari satirlara sar
    line, col = [], 0
    lines: list[list[tuple[int, str]]] = []
    for i, c in enumerate(cells):
        w = max(len(x) for x in c.split("\n")) + 1
        if col + w > width and line:
            lines.append(line)
            line, col = [], 0
        line.append((i, c))
        col += w
    if line:
        lines.append(line)

    print()
    for ln in lines:
        rows = max(len(c.split("\n")) for _, c in ln)
        for r in range(rows):
            out = []
            for i, c in ln:
                parts = c.split("\n")
                w = max(len(x) for x in parts)
                s = (parts[r] if r < len(parts) else "").ljust(w)
                out.append(paint(i, f" {s} ") if r == 0 else f"\033[38;5;244m {s} {RESET}")
            print("".join(out))
        print()

    ok = tok.decode(enc.ids) == text
    print(f"\033[1m{n_bytes} bayt -> {n_tok} token   "
          f"{n_bytes / n_tok:.2f} bayt/token   "
          f"roundtrip: {'OK' if ok else 'BOZUK'}{RESET}")
    if not show_ids:
        print(f"\033[38;5;244midler: {enc.ids}{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--tokenizer", default=str(DEFAULT))
    ap.add_argument("--file")
    ap.add_argument("--ids", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="ilk N karakter")
    ap.add_argument("--width", type=int, default=100)
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    print(f"\033[38;5;244m{a.tokenizer}  (vocab {tok.get_vocab_size()}){RESET}")

    if a.file:
        text = Path(a.file).read_text(encoding="utf-8")
    elif a.text:
        text = " ".join(a.text)
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        while True:
            try:
                s = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not s.strip():
                break
            render(tok, s, a.ids, a.width)
        return

    if a.limit:
        text = text[:a.limit]
    render(tok, text, a.ids, a.width)


if __name__ == "__main__":
    main()