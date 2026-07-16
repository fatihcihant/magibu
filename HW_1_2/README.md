# vocab_size=1024 BPE tokenizer — şaka korpusu
Magibu homework_1_2 icin bir dataset secip BPE ile kendi tokenizerimizi egittik.

## Çalıştırma

```bash
uv sync
uv run python download.py       
uv run python clean.py           
uv run python train.py         
uv run python analyze.py         
```

Tokenizer'a metin verip parçalanışını görmek:

```bash
uv run python tokenize_demo.py "A horse walks into a bar."
uv run python tokenize_demo.py --file data/heldout.txt --limit 400
echo "metin" | uv run python tokenize_demo.py
uv run python tokenize_demo.py                    # interaktif 
```

## Veri

`taivop/joke-dataset` → `wocka.json`. wocka.com arşivinden 10.019 kategori
etiketli şaka. Seçim sebebi: metinler kısa, kalıplar çok tekrar ediyor
(`walks into a bar`, `What do you call a`, `Knock knock`), dolayısıyla 767
merge'in ne öğrendiği çıplak gözle okunabiliyor.

`clean.py` sonrası: 5.665 şaka / 2,6 MB train + 299 şaka / 151 KB held-out.

## Temizleme kararları

| adım | sebep |
|---|---|
| kategori filtresi | belirli grupları aşağılamak üzerine kurulu kategoriler atılıyor (`--keep-all-categories` ile kapatılır). İçerik tercihi, teknik zorunluluk değil. |
| HTML unescape | kaynak `&amp;` `&quot;` `&#39;` içeriyor |
| NFKC + ASCII noktalama | **NFKC eğri tırnakları kendiliğinden düzeltmez.** `"` ile `"` ayrı merge yer; 767 merge'lik bütçede israf |
| boşluk normalizasyonu | CRLF→LF, satır sonu boşlukları, 3+ boş satır |
| dedup | aynı şaka birden çok geçiyor, merge istatistiğini şişiriyor |
| min/max uzunluk | 40–2000 karakter |
| küçük harfe **çevrilmiyor** | `The`/`the` ayrı merge yiyor ama vaka bilgisi veri. `--lowercase` ile eğitip farkı ölç |



## Ölçülen sonuçlar

Held-out şaka metninde **2,46 bayt/token**.

`analyze.py` çıktısı (`out/analysis.png`):


İlk öğrenilen merge'ler: `·t he ·a in ·the ·s ·w ou re er nd ·o ·b ·c at ...`
Son öğrenilenler: `·minutes ·keep ·always ·water ·show ·lawy ·pull ...`
(`·` = boşluk). Sıra beklendiği gibi: önce karakter çiftleri, sonra ekler,
en sonda tam kelimeler.

## Dosyalar

| dosya | iş |
|---|---|
| `download.py` | veriyi indir |
| `clean.py` | temizle, train/held-out ayır |
| `train.py` | HF `tokenizers` ile BPE eğit |
| `tokenize_demo.py` | metin ver, renkli token dökümü gör |
| `analyze.py` | grafikler + vocab dökümü |
