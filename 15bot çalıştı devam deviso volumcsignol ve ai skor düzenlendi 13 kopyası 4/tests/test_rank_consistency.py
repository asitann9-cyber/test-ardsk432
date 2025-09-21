#!/usr/bin/env python3
"""
🔗 Binance Futures WebSocket API Test
Testnet/Mainnet UserData Stream kontrolü
"""

import os
import json
import time
import requests
from datetime import datetime

try:
    from binance import ThreadedWebsocketManager
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("❌ pip install python-binance gerekli")

# .env dosyasını ana dizinden yükle
try:
    from dotenv import load_dotenv
    import sys
    
    # Ana dizin path'i ekle (tests klasöründen çıkıp ana dizine git)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Ana dizindeki .env dosyasını yükle
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(env_path)
    print(f"📁 .env dosyası aranıyor: {env_path}")
    
except ImportError:
    print("⚠️ python-dotenv yok - manuel API key gir")

# .env dosyasından al veya buraya yaz
API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_API_KEY_HERE')
API_SECRET = os.getenv('BINANCE_SECRET_KEY', 'YOUR_SECRET_HERE')
TESTNET = True  # False = Mainnet

print(f"🔑 API Key bulundu: {'✅' if API_KEY and API_KEY != 'YOUR_API_KEY_HERE' else '❌'}")
print(f"🔐 Secret bulundu: {'✅' if API_SECRET and API_SECRET != 'YOUR_SECRET_HERE' else '❌'}")

def get_listen_key():
    """📡 UserData Stream için listenKey al"""
    try:
        if TESTNET:
            url = "https://testnet.binancefuture.com/fapi/v1/listenKey"
        else:
            url = "https://fapi.binance.com/fapi/v1/listenKey"
        
        headers = {'X-MBX-APIKEY': API_KEY}
        response = requests.post(url, headers=headers)
        
        if response.status_code == 200:
            listen_key = response.json()['listenKey']
            print(f"✅ ListenKey alındı: {listen_key[:20]}...")
            return listen_key
        else:
            print(f"❌ ListenKey hatası: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ ListenKey exception: {e}")
        return None

def handle_user_data(msg):
    """📨 WebSocket mesaj handler"""
    try:
        print(f"\n🎉 *** MESAJ GELDİ! *** 🎉")
        print(f"Raw Message: {json.dumps(msg, indent=2)}")
        
        msg_type = msg.get('e', 'unknown')
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        print(f"\n📨 [{timestamp}] Mesaj Türü: {msg_type}")
        
        if msg_type == 'executionReport':
            symbol = msg.get('s', 'N/A')
            order_type = msg.get('o', 'N/A')
            order_status = msg.get('X', 'N/A')
            order_id = msg.get('i', 'N/A')
            side = msg.get('S', 'N/A')
            
            print(f"   🏷️  Symbol: {symbol}")
            print(f"   📋 Order ID: {order_id}")
            print(f"   🔄 Type: {order_type}")
            print(f"   📊 Status: {order_status}")
            print(f"   ⬆️⬇️ Side: {side}")
            
            # ⭐ SL/TP kontrolü
            if order_status == 'FILLED' and order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
                print(f"   🎯 *** SL/TP TETİKLENDİ: {symbol} {order_type} ***")
                
        elif msg_type == 'ACCOUNT_UPDATE':
            print(f"   💰 Hesap güncellendi")
            
        elif msg_type == 'ORDER_TRADE_UPDATE':
            print(f"   📈 Trade güncellendi")
            
        else:
            print(f"   🔍 Bilinmeyen mesaj türü: {msg_type}")
            
    except Exception as e:
        print(f"   ❌ Mesaj işleme hatası: {e}")
        print(f"   Raw data: {msg}")

def test_websocket():
    """🧪 WebSocket bağlantı testi"""
    print(f"🚀 Binance Futures WebSocket Test Başlıyor...")
    print(f"🌐 Mode: {'Testnet' if TESTNET else 'Mainnet'}")
    print(f"🔑 API Key: {API_KEY[:20]}..." if API_KEY else "❌ API Key yok")
    
    if not BINANCE_AVAILABLE:
        print("❌ python-binance kütüphanesi gerekli!")
        return
        
    if not API_KEY or API_KEY == 'YOUR_API_KEY_HERE':
        print("❌ .env dosyasında BINANCE_API_KEY ayarlayın!")
        return
    
    # 1) ListenKey al
    print("\n📡 1) ListenKey alınıyor...")
    listen_key = get_listen_key()
    if not listen_key:
        return
    
    # 2) WebSocket başlat
    print("\n🔗 2) WebSocket bağlantısı kuruluyor...")
    try:
        twm = ThreadedWebsocketManager(
            api_key=API_KEY,
            api_secret=API_SECRET,
            testnet=TESTNET
        )
        
        # Start manager
        twm.start()
        print("✅ WebSocket Manager başlatıldı")
        
        # UserData stream başlat
        stream_name = twm.start_futures_user_socket(callback=handle_user_data)
        print(f"✅ UserData Stream başlatıldı: {stream_name}")
        
        print("\n👂 WebSocket dinleniyor... (30 saniye)")
        print("💡 Test için Binance'de manuel emir verin veya SL/TP tetikleyin")
        print("🛑 Çıkmak için Ctrl+C")
        print("🔍 WebSocket bağlantı testi - herhangi bir mesaj bekleniyor...")
        
        # Her 5 saniyede durum kontrolü
        for i in range(6):  # 30 saniye = 6 x 5 saniye
            print(f"⏰ Dinleme süresi: {(i+1)*5}/30 saniye")
            time.sleep(5)
        
        # Temizlik
        print("\n🧹 WebSocket kapatılıyor...")
        try:
            twm.stop()
        except Exception as stop_error:
            print(f"⚠️ Kapatma hatası (normal): {stop_error}")
        print("✅ Test tamamlandı!")
        
    except KeyboardInterrupt:
        print("\n⌨️ Kullanıcı durdurdu")
        try:
            twm.stop()
        except:
            pass
    except Exception as e:
        print(f"❌ WebSocket hatası: {e}")
        try:
            twm.stop()
        except:
            pass

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 BINANCE FUTURES WEBSOCKET TEST")
    print("=" * 50)
    test_websocket()