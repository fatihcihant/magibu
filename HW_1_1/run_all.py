import subprocess
import os
from datetime import datetime

# Eğitilecek 4 modelin klasör yolları ve isimleri
models = [
    {"name": "Qwen_3.5", "script_path": "qwen3_5/train.py"},
    {"name": "Qwen_3", "script_path": "qwen3/train.py"},
    {"name": "Gemma_4", "script_path": "gemma4/train.py"},
    {"name": "DeepSeek_3", "script_path": "deepseek3/train.py"}
]

# Çıktıların kaydedileceği klasörü oluştur
log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("4 Model için toplu eğitim süreci başlıyor...\n")

for model in models:
    name = model["name"]
    script = model["script_path"]
    
    log_filename = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    print(f"[{name}] modeli eğitiliyor... Lütfen bekleyin.")
    print(f"Çıktılar buraya kaydediliyor: {log_filename}")
    
    try:
        with open(log_filename, "w", encoding="utf-8") as log_file:
            process = subprocess.run(
                ["python3", script], 
                stdout=log_file, 
                stderr=subprocess.STDOUT, 
                text=True
            )
            
        if process.returncode == 0:
            print(f"[{name}] eğitimi başarıyla tamamlandı.\n")
        else:
            print(f"[{name}] eğitiminde bir hata oluştu! Log dosyasını kontrol et.\n")
            
    except Exception as e:
        print(f"⚠️ [{name}] scripti çalıştırılamadı. Hata: {e}\n")

print("🎉 Tüm eğitimler tamamlandı! Sonuçları 'training_logs' klasöründen inceleyebilirsiniz.")