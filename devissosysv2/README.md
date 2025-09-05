# 🚀 Deviso System Dashboard

Modern ve modüler Binance Futures coin tarama ve analiz platformu.

## 📁 Proje Yapısı

```
devisosystem/
├── app.py                 # Ana Flask uygulaması
├── requirements.txt       # Python paketleri
├── README.md             # Bu dosya
├── templates/
│   └── dashboard.html    # Web arayüzü
└── modules/              # Modüller klasörü
    ├── __init__.py       # Modules paketi
    ├── config.py         # Modül yapılandırması
    ├── manager.py        # Modül yöneticisi
    ├── ema_scanner/      # Modül 1: EMA200 Scanner
    │   ├── __init__.py   # Modül paketi
    │   └── deviema.py    # EMA200 tarama algoritması
    └── rsi_macd_scanner/ # Modül 2: RSI/MACD Scanner
        ├── __init__.py   # Modül paketi
        └── devic20.py    # RSI/MACD tarama algoritması
```

## 🎯 Özellikler

### ✅ Mevcut Modüller
- **EMA200 Scanner**: Binance Futures coinlerini EMA200 crossover stratejisi ile tarar
  - **Koşul**: Ratio >= 0 (Pullback seviyesinin üstünde)
  - **Metrikler**: Score, Ratio%, Long/Short Momentum, Fiyat
- **RSI/MACD Scanner**: RSI ve MACD tabanlı sinyal tarayıcısı
  - **RSI Sinyalleri**: C20L, C10L, L20L, L20S, C20S, C10S
  - **MACD Sinyalleri**: M2L-M5L, M2S-M5S
  - **Öncelikli**: C20L, C20S, M5L, M5S

### 🔧 Teknik Özellikler
- **Modüler Yapı**: Her modül ayrı klasörde, kolay genişletilebilir
- **Dinamik Yükleme**: Modüller runtime'da yüklenir
- **Yapılandırılabilir**: Her modül için ayrı ayarlar
- **Modern UI**: Bootstrap 5 ile responsive tasarım
- **Real-time**: Canlı güncelleme ve durum takibi

## 🚀 Kurulum

### 1. Gereksinimler
```bash
pip install -r requirements.txt
```

### 2. Çalıştırma
```bash
python app.py
```

### 3. Erişim
Dashboard'a şu adresten erişin: **http://localhost:5000**

## 📊 API Endpoints

### Ana Endpoints
- `GET /` - Dashboard ana sayfası
- `GET /api/scan` - EMA200 taraması başlat
- `GET /api/scan/rsi_macd` - RSI/MACD taraması başlat
- `GET /api/results` - Tarama sonuçlarını getir
- `GET /api/status` - Sistem durumunu getir

### Modül Endpoints
- `GET /api/modules` - Yüklenen modülleri listele
- `GET /api/modules/<module_name>` - Modül bilgilerini getir

## 🔧 Modül Geliştirme

### Yeni Modül Ekleme

1. **Modül Klasörü Oluştur**:
```bash
mkdir modules/yeni_modul
```

2. **Modül Dosyaları**:
```
modules/yeni_modul/
├── __init__.py    # Modül paketi
└── main.py        # Ana modül kodu
```

3. **__init__.py Örneği**:
```python
from .main import YeniModulClass

__version__ = "1.0.0"
__description__ = "Yeni Modül Açıklaması"
__author__ = "Deviso System"

__all__ = ['YeniModulClass']
```

4. **config.py'ye Ekle**:
```python
"yeni_modul": {
    "name": "Yeni Modül",
    "description": "Yeni modül açıklaması",
    "version": "1.0.0",
    "enabled": True,
    "settings": {
        "parametre1": "değer1"
    }
}
```

### Modül Gereksinimleri
- Ana sınıf `__all__` listesinde tanımlanmalı
- `__init__.py` dosyası gerekli
- Hata yönetimi yapılmalı
- Dokümantasyon eklenmeli

## 🌐 VPS Deployment

### Production Modunda Çalıştırma
```bash
# Gunicorn ile
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Systemd service olarak
sudo systemctl enable deviso-dashboard
sudo systemctl start deviso-dashboard
```

### Nginx Konfigürasyonu
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📈 Gelecek Modüller

- [ ] **RSI Scanner**: RSI aşırı alım/satım sinyalleri
- [ ] **Volume Scanner**: Hacim analizi
- [ ] **Pattern Scanner**: Teknik formasyon tespiti
- [ ] **News Scanner**: Haber etkisi analizi
- [ ] **Portfolio Tracker**: Portföy takibi
- [ ] **Alert System**: Bildirim sistemi

## 🛠️ Geliştirme

### Kod Standartları
- Python PEP 8 standartları
- Type hints kullanımı
- Docstring'ler zorunlu
- Error handling gerekli

### Test
```bash
# Modül testi
python -m pytest tests/

# Linting
flake8 modules/
```

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Commit yapın (`git commit -am 'Yeni özellik eklendi'`)
4. Push yapın (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📞 İletişim

- **Proje**: Deviso System Dashboard
- **Versiyon**: 1.0.0
- **Güncelleme**: 24 Ağustos 2025

---

**Not**: Bu dashboard eğitim amaçlıdır. Gerçek trading kararları için profesyonel danışmanlık alın.
