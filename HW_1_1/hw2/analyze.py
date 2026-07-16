"""Ek: analiz + gorsellestirme.  -> out/analysis.png, out/vocab_dump.txt

    uv run python analyze.py

Dort panel:
  1. bayt/token, vocab boyutuna gore (held-out uzerinde). 1024'un egrinin
     neresine dustugunu gosterir. Egri icin 257..8192 arasi ayri ayri egitir.
  2. Ogrenilen tokenlarin uzunluk histogrami.
  3. Alan disi metinlerde bayt/token. Tokenizer'in sakalara ne kadar
     ezberledigini olcer.
  4. Held-out metinde en sik 30 token.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.trainers import BpeTrainer  # noqa: E402
from tokenizers import pre_tokenizers  # noqa: E402

from train import EOT, build  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "out"
CURVE = [257, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 3072, 4096, 8192]

PROBES = {
    "saka (held-out)": None,  # dosyadan
    "Shakespeare": "To be, or not to be, that is the question: whether 'tis "
                   "nobler in the mind to suffer the slings and arrows of "
                   "outrageous fortune.",
    "Python kodu": "import numpy as np\ndef attention(q, k, v):\n    "
                   "s = q @ k.T / np.sqrt(q.shape[-1])\n    return softmax(s) @ v\n",
    "sayilar/tarih": "Q3 2024 revenue $1,247,893.55, up 12.4% YoY. Order "
                     "#A7X-99213 shipped 2024-11-03T14:22:07Z.",
    "web/argo": "lol ok so i just deployed to prod on a friday :) send help "
                "https://github.com/huggingface/tokenizers #devops",
    "Turkce": "Byte-pair encoding, sik gecen karakter ciftlerini birlestirerek "
              "bir sozluk olusturan bir sikistirma algoritmasidir. "
              "Istanbul'da yagmur yagiyor.",
}


def train_at(train_file: str, vocab_size: int, min_freq: int) -> Tokenizer:
    tok = build()
    tok.train([train_file], BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_freq, special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False))
    return tok


def bpt(tok: Tokenizer, text: str) -> float:
    n = len(tok.encode(text).ids)
    return len(text.encode("utf-8")) / n if n else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(HERE / "data/train.txt"))
    ap.add_argument("--heldout", default=str(HERE / "data/heldout.txt"))
    ap.add_argument("--tokenizer", default=str(OUT / "tokenizer_1024.json"))
    ap.add_argument("--min-frequency", type=int, default=2)
    a = ap.parse_args()

    held = Path(a.heldout).read_text(encoding="utf-8")
    tok = Tokenizer.from_file(a.tokenizer)
    probes = dict(PROBES)
    probes["saka (held-out)"] = held

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # -- 1. bayt/token vs vocab -------------------------------------------
    print("egri icin egitiliyor:", end=" ", flush=True)
    xs, ys = [], []
    for v in CURVE:
        print(v, end=" ", flush=True)
        xs.append(v)
        ys.append(bpt(train_at(a.train, v, a.min_frequency), held))
    print()
    ax = axes[0, 0]
    ax.plot(xs, ys, marker="o", ms=4, color="#1f77b4")
    ax.axvline(1024, color="crimson", ls="--", lw=1.2)
    i = xs.index(1024)
    ax.annotate(f"1024 -> {ys[i]:.2f}", (1024, ys[i]), xytext=(8, -14),
                textcoords="offset points", color="crimson", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks([257, 512, 1024, 2048, 4096, 8192])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("vocab_size")
    ax.set_ylabel("bayt / token (held-out)")
    ax.set_title("1. Sikistirma vs vocab boyutu\n(vocab 257 = 0 merge = ham bayt)",
                 fontsize=10)
    ax.grid(alpha=0.3)

    # -- 2. token uzunluk histogrami --------------------------------------
    vocab = tok.get_vocab()
    lens = [len(t) for t, i in vocab.items() if i >= 256 and t != EOT]
    ax = axes[0, 1]
    ax.hist(lens, bins=range(1, max(lens) + 2), align="left",
            color="#2ca02c", edgecolor="white")
    ax.set_xticks(range(1, max(lens) + 1))
    ax.set_xlabel("token uzunlugu (karakter)")
    ax.set_ylabel("token sayisi")
    ax.set_title(f"2. Ogrenilen {len(lens)} merge tokeninin uzunlugu\n"
                 f"ortalama {sum(lens) / len(lens):.2f}", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # -- 3. alan disi ------------------------------------------------------
    names = list(probes)
    vals = [bpt(tok, probes[n]) for n in names]
    order = sorted(range(len(names)), key=lambda i: -vals[i])
    ax = axes[1, 0]
    cols = ["#d62728" if names[i] == "saka (held-out)" else "#7f7f7f" for i in order]
    ax.barh([names[i] for i in order], [vals[i] for i in order], color=cols)
    ax.axvline(1.0, color="k", ls=":", lw=1)
    ax.text(1.02, -0.4, "ham bayt", fontsize=8, color="k")
    for y, i in enumerate(order):
        ax.text(vals[i] + 0.04, y, f"{vals[i]:.2f}", va="center", fontsize=9)
    ax.set_xlabel("bayt / token (yuksek = iyi)")
    ax.set_title("3. Alan disi bozulma (ayni vocab=1024 tokenizer)", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()

    # -- 4. en sik tokenlar ------------------------------------------------
    cnt = Counter(tok.encode(held).tokens)
    top = cnt.most_common(30)[::-1]
    ax = axes[1, 1]
    labels = [t.replace("Ġ", "·").replace("Ċ", "\\n") for t, _ in top]
    ax.barh(range(len(top)), [c for _, c in top], color="#9467bd")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8, family="monospace")
    ax.set_xlabel("held-out metinde gecis sayisi")
    ax.set_title("4. En sik 30 token  (· = bosluk)", fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(OUT / "analysis.png", dpi=150)

    # -- vocab dokumu ------------------------------------------------------
    rows = ["id\ttoken"]
    rows += [f"{i}\t{t}" for t, i in sorted(vocab.items(), key=lambda kv: kv[1])]
    (OUT / "vocab_dump.txt").write_text("\n".join(rows), encoding="utf-8")

    print(f"\n-> {OUT / 'analysis.png'}")
    print(f"-> {OUT / 'vocab_dump.txt'}")
    print(f"\nheld-out bayt/token @1024: {bpt(tok, held):.3f}")
    print("ilk ogrenilen 40 merge tokeni (id sirasi = ogrenilme sirasi):")
    inv = {i: t for t, i in vocab.items()}
    print("  " + " ".join(repr(inv[i].replace("Ġ", "·")) for i in range(257, 297)))
    print("son 20:")
    print("  " + " ".join(repr(inv[i].replace("Ġ", "·")) for i in range(1004, 1024)))


if __name__ == "__main__":
    main()