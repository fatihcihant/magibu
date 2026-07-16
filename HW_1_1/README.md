# homework_1

Şehir isimlerinden oluşan bir veri kümesi kullanılarak 4 farklı model eğitilmiş ve modellerin çıktıları karşılaştırılmıştır.

## Veri Hazırlama

Ham şehir ismi verisini temizlemek için:

```bash
python3 temizle_isimler.py
```

## Eğitim

Tüm modeller için `train steps` değeri 5000 olarak ayarlanmıştır.

4 modeli tek seferde eğitmek ve çıktılarını kaydetmek için:

```bash
python run_all.py
```

## Çıktılar

Model eğitim çıktıları `training_logs` dizinine kaydedilir.
