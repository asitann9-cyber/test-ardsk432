#!/usr/bin/env python3
"""
Gerçek SL/TP Tetikleme Testi
TestBinance'de gerçek pozisyon açıp SL/TP tetikleyerek CSV kaydını test eder
"""

import os
import sys
import time
from datetime import datetime

# Ana dizin path ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(parent_dir, '.env')
    load_dotenv(env_path)
    
    from binance import Client
    from binance.enums import *
    
    API_KEY = os.getenv('BINANCE_API_KEY')
    API_SECRET = os.getenv('BINANCE_SECRET_KEY')
    
    print("🔧 GERÇEK SL/TP TETİKLEME TESTİ")
    print("=" * 50)
    
    # Client oluştur
    client = Client(
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=True
    )
    
    # Timestamp sync düzelt
    print("⏰ Sunucu saati senkronize ediliyor...")
    try:
        # Sunucu zamanını al
        server_time = client.futures_time()["serverTime"]
        local_time = int(time.time() * 1000)
        offset = int(server_time) - local_time
        client.timestamp_offset = offset
        print(f"✅ Time sync: offset={offset}ms")
    except Exception as sync_error:
        print(f"⚠️ Time sync hatası: {sync_error}")
        print("Manuel recvWindow artırımı deneniyor...")
        # Yedek çözüm: daha büyük recvWindow kullan
        pass
    
    # Bakiye kontrol (artırılmış recvWindow ile)
    account_info = client.futures_account(recvWindow=30000)  # 30 saniye window
    balance = float(account_info["totalWalletBalance"])
    print(f"💰 Bakiye: ${balance:.2f}")
    
    if balance < 50:
        print("❌ Yetersiz bakiye - test için minimum $50 gerekli")
        sys.exit(1)
    
    # Test sembol ve parametreler
    SYMBOL = 'BTCUSDT'
    TEST_QUANTITY = 0.001  # Minimum test miktarı
    
    # Güncel fiyat al (recvWindow yok, sadece market data)
    ticker = client.futures_ticker(symbol=SYMBOL)
    current_price = float(ticker['price'])
    print(f"💲 {SYMBOL} fiyat: ${current_price:.2f}")
    
    # Çok kısa SL/TP mesafeleri (hızlı tetikleme için)
    sl_distance = 0.002  # %0.2
    tp_distance = 0.001  # %0.1 (daha kısa - hemen tetiklensin)
    
    # Long pozisyon fiyatları
    sl_price = round(current_price * (1 - sl_distance), 2)
    tp_price = round(current_price * (1 + tp_distance), 2)
    
    print(f"🎯 SL: ${sl_price:.2f} (%{sl_distance*100:.1f} aşağı)")
    print(f"🎯 TP: ${tp_price:.2f} (%{tp_distance*100:.1f} yukarı)")
    
    try:
        # 1. LONG Market Emri (büyük recvWindow)
        print("\n📝 1) Long pozisyon açılıyor...")
        main_order = client.futures_create_order(
            symbol=SYMBOL,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=TEST_QUANTITY,
            recvWindow=30000  # 30 saniye
        )
        print(f"✅ Market emri: {main_order['orderId']}")
        time.sleep(2)
        
        # 2. Take Profit Emri (kısa mesafe - hızlı tetikleme)
        print("📝 2) Take Profit emri veriliyor...")
        tp_order = client.futures_create_order(
            symbol=SYMBOL,
            side=SIDE_SELL,
            type='TAKE_PROFIT_MARKET',
            quantity=TEST_QUANTITY,
            stopPrice=tp_price,
            timeInForce='GTC',
            recvWindow=30000  # 30 saniye
        )
        print(f"✅ TP emri: {tp_order['orderId']}")
        
        # 3. Stop Loss Emri
        print("📝 3) Stop Loss emri veriliyor...")
        sl_order = client.futures_create_order(
            symbol=SYMBOL,
            side=SIDE_SELL,
            type='STOP_MARKET',
            quantity=TEST_QUANTITY,
            stopPrice=sl_price,
            timeInForce='GTC',
            recvWindow=30000  # 30 saniye
        )
        print(f"✅ SL emri: {sl_order['orderId']}")
        
        print(f"\n⏰ TP tetiklenmesini bekliyor... (Max 2 dakika)")
        print(f"💡 {SYMBOL} fiyatı ${tp_price:.2f} üzerine çıkarsa TP tetiklenir")
        
        # 4. Emirleri izle
        start_time = time.time()
        tp_triggered = False
        sl_triggered = False
        
        while time.time() - start_time < 120:  # 2 dakika bekle
            try:
                # TP durumu kontrol (büyük recvWindow)
                tp_check = client.futures_get_order(
                    symbol=SYMBOL,
                    orderId=tp_order['orderId'],
                    recvWindow=30000  # 30 saniye
                )
                
                if tp_check['status'] == 'FILLED' and not tp_triggered:
                    tp_triggered = True
                    fill_price = float(tp_check.get('avgPrice', tp_price))
                    print(f"\n🎉 TAKE PROFIT TETİKLENDİ!")
                    print(f"💰 Tetiklenme fiyatı: ${fill_price:.2f}")
                    print(f"📊 Durum: {tp_check['status']}")
                    
                    # SL emrini iptal et
                    try:
                        client.futures_cancel_order(
                            symbol=SYMBOL,
                            orderId=sl_order['orderId'],
                            recvWindow=30000
                        )
                        print(f"🚫 SL emri iptal edildi")
                    except:
                        pass
                    
                    break
                
                # SL durumu kontrol (ikincil)
                sl_check = client.futures_get_order(
                    symbol=SYMBOL,
                    orderId=sl_order['orderId'],
                    recvWindow=30000  # 30 saniye
                )
                
                if sl_check['status'] == 'FILLED' and not sl_triggered:
                    sl_triggered = True
                    fill_price = float(sl_check.get('avgPrice', sl_price))
                    print(f"\n🛑 STOP LOSS TETİKLENDİ!")
                    print(f"💰 Tetiklenme fiyatı: ${fill_price:.2f}")
                    print(f"📊 Durum: {sl_check['status']}")
                    
                    # TP emrini iptal et
                    try:
                        client.futures_cancel_order(
                            symbol=SYMBOL,
                            orderId=tp_order['orderId'],
                            recvWindow=30000
                        )
                        print(f"🚫 TP emri iptal edildi")
                    except:
                        pass
                    
                    break
                
                # Güncel fiyat göster (market data - recvWindow yok)
                current_ticker = client.futures_ticker(symbol=SYMBOL)
                live_price = float(current_ticker['price'])
                elapsed = int(time.time() - start_time)
                
                if elapsed % 10 == 0:  # Her 10 saniyede bir
                    print(f"⏰ {elapsed}s - Fiyat: ${live_price:.2f} (TP: ${tp_price:.2f})")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Kontrol hatası: {e}")
                time.sleep(5)
        
        # 5. Sonuç
        if tp_triggered:
            print(f"\n✅ TEST BAŞARILI: Take Profit tetiklendi!")
            print(f"🎯 Live trader kodunda bu tetiklenme CSV'ye kaydedilmeliydi")
            print(f"📁 ai_crypto_trades.csv dosyasını kontrol edin")
        elif sl_triggered:
            print(f"\n✅ TEST BAŞARILI: Stop Loss tetiklendi!")
            print(f"🎯 Live trader kodunda bu tetiklenme CSV'ye kaydedilmeliydi") 
            print(f"📁 ai_crypto_trades.csv dosyasını kontrol edin")
        else:
            print(f"\n⏰ Zaman doldu - hiçbir emir tetiklenmedi")
            print(f"📝 Açık emirleri manuel iptal ediliyor...")
            
            # Emirleri iptal et (büyük recvWindow)
            try:
                client.futures_cancel_order(symbol=SYMBOL, orderId=tp_order['orderId'], recvWindow=30000)
                client.futures_cancel_order(symbol=SYMBOL, orderId=sl_order['orderId'], recvWindow=30000)
                print(f"🚫 Emirler iptal edildi")
                
                # Pozisyonu manuel kapat
                close_order = client.futures_create_order(
                    symbol=SYMBOL,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=TEST_QUANTITY,
                    recvWindow=30000  # 30 saniye
                )
                print(f"🔒 Pozisyon manuel kapatıldı: {close_order['orderId']}")
                
            except Exception as e:
                print(f"❌ Temizlik hatası: {e}")
        
        print(f"\n📋 TEST RAPORU:")
        print(f"- Eğer TP/SL tetiklendiyse CSV dosyasında kayıt olmalı")
        print(f"- Eğer CSV'de kayıt yoksa live_trader.py'de sorun var") 
        print(f"- Eğer kayıt varsa sorun çözülmüş")
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        
        # Temizleme (büyük recvWindow ile)
        try:
            # Tüm açık pozisyonları kapat
            positions = client.futures_position_information(recvWindow=30000)
            for pos in positions:
                if pos['symbol'] == SYMBOL and abs(float(pos['positionAmt'])) > 0:
                    side = SIDE_SELL if float(pos['positionAmt']) > 0 else SIDE_BUY
                    qty = abs(float(pos['positionAmt']))
                    
                    client.futures_create_order(
                        symbol=SYMBOL,
                        side=side,
                        type=ORDER_TYPE_MARKET,
                        quantity=qty,
                        recvWindow=30000
                    )
                    print(f"🧹 {SYMBOL} pozisyon temizlendi")
            
            # Tüm açık emirleri iptal et
            open_orders = client.futures_get_open_orders(symbol=SYMBOL, recvWindow=30000)
            for order in open_orders:
                client.futures_cancel_order(
                    symbol=SYMBOL,
                    orderId=order['orderId'],
                    recvWindow=30000
                )
            print(f"🧹 Açık emirler temizlendi")
            
        except Exception as cleanup_error:
            print(f"⚠️ Temizlik hatası: {cleanup_error}")

except Exception as e:
    print(f"❌ Import/Setup hatası: {e}")
    print("Gereksinimler:")
    print("- pip install python-binance")
    print("- .env dosyasında BINANCE_API_KEY ve BINANCE_SECRET_KEY")