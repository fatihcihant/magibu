"""3/4  BPE tokenizer egit (HuggingFace `tokenizers`, vocab_size=1024).

    uv run python train.py
    uv run python train.py --vocab-size 4096 --tag _4096

Vocab bütcesi:  256 (byte alfabesi) + 1 (<|endoftext|>) + 767 (merge) = 1024.
`initial_alphabet=ByteLevel.alphabet()` olmazsa 256 bayt garanti girmez ve
egitim korpusunda gecmeyen baytlar UNK olur -> byte-level BPE'nin tek avantaji
(hicbir sey UNK olmaz) kaybolur.

ByteLevel pre-tokenizer `use_regex=True` (varsayilan) ile GPT-2 bolme
kalibini uygular. Bu olmadan merge'ler kelime sinirini asar ve ". The" gibi
tokenlar ogrenilir; 767 merge'lik butcede bu israftir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

HERE = Path(__file__).parent
EOT = "<|endoftext|>"


def build() -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=None, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                 use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(HERE / "data/train.txt"))
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default=str(HERE / "out"))
    a = ap.parse_args()

    tok = build()
    trainer = BpeTrainer(
        vocab_size=a.vocab_size,
        min_frequency=a.min_frequency,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train([a.train], trainer)

    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    path = out / f"tokenizer{a.tag or '_' + str(a.vocab_size)}.json"
    tok.save(str(path), pretty=False)

    vocab = tok.get_vocab()
    print(f"\nvocab: {len(vocab)}  = 256 byte + 1 ozel + {len(vocab) - 257} merge")
    print(f"kaydedildi -> {path}")

    demo = "A horse walks into a bar. The bartender says: why the long face?"
    enc = tok.encode(demo)
    print(f"\ndeneme: {demo}")
    print(f"tokenlar: {enc.tokens}")
    print(f"idler:    {enc.ids}")
    print(f"decode:   {tok.decode(enc.ids)!r}")


if __name__ == "__main__":
    main()