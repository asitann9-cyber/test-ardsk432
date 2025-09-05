#!/usr/bin/env python3
"""
🧪 Basit Sistem Test Script
Live Bot sisteminizin çalışıp çalışmadığını kontrol eder
"""

import sys
import os

# Ana dizini Python path'ine ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("🚀 Live Bot Sistem Testi Başlatılıyor...")
print("=" * 50)

# Test 1: Modül import kontrolü
print("\n📦 Test 1: Modül Import Kontrolü")
try:
    from config import ENVIRONMENT, BASE, BINANCE_API_KEY, BINANCE_SECRET_KEY
    print("✅ Config modülü yüklendi")
    print(f"   Environment: {ENVIRONMENT}")
    print(f"   Base URL: {BASE}")
    
    if BINANCE_API_KEY:
        print(f"   API Key: {BINANCE_API_KEY[:8]}...")
    else:
        print("   ❌ API Key bulunamadı")
        
    if BINANCE_SECRET_KEY:
        print("   ✅ Secret Key bulundu")
    else:
        print("   ❌ Secret Key bulunamadı")
        
except Exception as e:
    print(f"❌ Config yüklenemedi: {e}")
    sys.exit(1)

# Test 2: API Bağlantı Testi
print("\n🌐 Test 2: API Bağlantı Testi")
try:
    from data.fetch_data import get_usdt_perp_symbols, get_current_price
    
    # Sembol listesi al
    print("   Sembol listesi alınıyor...")
    symbols = get_usdt_perp_symbols()
    
    if symbols:
        print(f"   ✅ {len(symbols)} sembol bulundu")
        print(f"   Örnek: {', '.join(symbols[:5])}")
        
        # Test fiyat al
        test_symbol = symbols[0]
        print(f"   {test_symbol} fiyatı kontrol ediliyor...")
        price = get_current_price(test_symbol)
        
        if price:
            print(f"   ✅ {test_symbol}: ${price}")
        else:
            print(f"   ❌ {test_symbol} fiyatı alınamadı")
    else:
        print("   ❌ Hiç sembol bulunamadı")
        
except Exception as e:
    print(f"   ❌ API bağlantı hatası: {e}")

# Test 3: Sinyal Analizi Testi
print("\n🤖 Test 3: AI Sinyal Analizi Testi")
try:
    from trading.analyzer import batch_analyze_with_ai
    
    print("   AI analizi başlatılıyor (bu 30-60 saniye sürer)...")
    signals = batch_analyze_with_ai("15m")
    
    if not signals.empty:
        print(f"   ✅ {len(signals)} sinyal bulundu")
        
        # En iyi 3 sinyali göster
        top_3 = signals.head(3)
        print("   🏆 En iyi 3 sinyal:")
        
        for i, (_, signal) in enumerate(top_3.iterrows(), 1):
            ai_score = signal['ai_score']
            if ai_score <= 1:
                ai_score = ai_score * 100  # 0-1 ise 0-100'e çevir
                
            print(f"      {i}. {signal['symbol']} | {signal['run_type'].upper()} | "
                  f"AI: {ai_score:.0f}% | Run: {signal['run_count']}")
    else:
        print("   ⚠️ Hiç sinyal bulunamadı")
        
except Exception as e:
    print(f"   ❌ Sinyal analizi hatası: {e}")

# Test 4: Live Trading Bağlantı Testi
print("\n🤖 Test 4: Live Trading Bağlantı Testi")
try:
    from trading.live_trader import LiveTradingBot
    
    bot = LiveTradingBot()
    print("   Live trading bot bağlantısı test ediliyor...")
    
    if bot.connect_to_binance():
        print("   ✅ Live trading bot bağlantısı başarılı")
        print(f"   💰 Hesap bakiyesi: ${bot.account_balance:.2f}")
        print(f"   📊 Tradable sembol sayısı: {len(bot.tradable_cache)}")
    else:
        print("   ❌ Live trading bot bağlanamadı")
        
except Exception as e:
    print(f"   ❌ Live trading test hatası: {e}")

# Test 5: Basit Position Size Testi
print("\n💰 Test 5: Position Size Hesaplama Testi")
try:
    from trading.live_trader import LiveTradingBot
    
    bot = LiveTradingBot()
    if bot.connect_to_binance():
        # BTCUSDT için test
        test_price = 50000.0  # Örnek fiyat
        
        quantity = bot.calculate_position_size_flexible("BTCUSDT", test_price)
        investment = quantity * test_price
        
        print(f"   Test: BTCUSDT @ ${test_price}")
        print(f"   ✅ Hesaplanan miktar: {quantity:.6f}")
        print(f"   💵 Yatırım değeri: ${investment:.2f}")
        
        if 90 <= investment <= 110:  # $100 civarı olmalı
            print("   ✅ Position size hesaplaması doğru")
        else:
            print("   ⚠️ Position size beklenenden farklı")
    else:
        print("   ❌ Bot bağlantısı kurulamadı")
        
except Exception as e:
    print(f"   ❌ Position size test hatası: {e}")

# Sonuç
print("\n" + "=" * 50)
print("🎯 TEST TAMAMLANDI")
print("\n💡 Sonuçlar:")
print("✅ = Başarılı, çalışıyor")
print("❌ = Başarısız, düzeltme gerekli") 
print("⚠️ = Kısmen çalışıyor, kontrol gerekli")

print("\n🚀 Eğer çoğu test başarılıysa, Live Bot'u başlatabilirsiniz!")
print("❌ Eğer hatalar varsa, .env dosyasını ve API anahtarlarını kontrol edin.")

input("\nEnter'a basın...")