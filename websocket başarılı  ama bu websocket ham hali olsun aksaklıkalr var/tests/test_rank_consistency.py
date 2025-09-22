#!/usr/bin/env python3
"""
🔗 Debug WebSocket Test - Live Bot Sorun Analizi
Tablodan emtia seçip WebSocket ile izleme + Sorun tespiti
"""

import os
import json
import time
import requests
from datetime import datetime
import threading

try:
    from binance import ThreadedWebsocketManager, Client
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

# .env dosyasından al
API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_API_KEY_HERE')
API_SECRET = os.getenv('BINANCE_SECRET_KEY', 'YOUR_SECRET_HERE')
TESTNET = True  # False = Mainnet

print(f"🔑 API Key bulundu: {'✅' if API_KEY and API_KEY != 'YOUR_API_KEY_HERE' else '❌'}")
print(f"🔐 Secret bulundu: {'✅' if API_SECRET and API_SECRET != 'YOUR_SECRET_HERE' else '❌'}")

# Global değişkenler - sorun takibi için
test_orders = {}  # Order ID -> Order details
manual_orders = []  # Manuel verdiğimiz emirler
websocket_messages = []  # Gelen tüm mesajlar
debug_log = []

def log_debug(message):
    """Debug log tut"""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    log_entry = f"[{timestamp}] {message}"
    debug_log.append(log_entry)
    print(log_entry)

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
            log_debug(f"✅ ListenKey alındı: {listen_key[:20]}...")
            return listen_key
        else:
            log_debug(f"❌ ListenKey hatası: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        log_debug(f"❌ ListenKey exception: {e}")
        return None

def _sync_server_time(client, retries=3):
    """Binance sunucu saatine senkronize et"""
    import time as _t
    for i in range(retries):
        try:
            srv = client.futures_time()["serverTime"]
            loc = int(_t.time() * 1000)
            offset = int(srv) - loc
            client.timestamp_offset = offset
            log_debug(f"⏱️ Time sync: offset={offset}ms (try {i+1})")
            _t.sleep(0.2)
            return True
        except Exception as e:
            log_debug(f"⚠️ Time sync attempt {i+1} failed: {e}")
            _t.sleep(0.2)
    return False

def create_test_client():
    """Test için Binance client oluştur"""
    try:
        if TESTNET:
            client = Client(
                api_key=API_KEY,
                api_secret=API_SECRET,
                testnet=True
            )
            # Testnet URL override
            client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
            client.FUTURES_DATA_URL = "https://testnet.binancefuture.com/futures/data"
        else:
            client = Client(
                api_key=API_KEY,
                api_secret=API_SECRET
            )
        
        # 🔧 TIMESTAMP SYNC - SORUN ÇÖZÜMÜ
        log_debug("⏱️ Sunucu saati senkronize ediliyor...")
        if not _sync_server_time(client):
            log_debug("❌ Timestamp sync başarısız - devam ediliyor...")
        
        # Bağlantı testi
        client.futures_ping()
        account_info = client.futures_account(recvWindow=60000)
        balance = float(account_info["totalWalletBalance"])
        log_debug(f"✅ Binance client bağlantısı başarılı - Bakiye: ${balance:.2f}")
        return client
    except Exception as e:
        log_debug(f"❌ Binance client hatası: {e}")
        return None

def get_active_positions(client):
    """Mevcut açık pozisyonları al"""
    try:
        positions = client.futures_position_information()
        active_positions = []
        
        for pos in positions:
            position_amt = float(pos['positionAmt'])
            if abs(position_amt) > 0:
                active_positions.append({
                    'symbol': pos['symbol'],
                    'side': 'LONG' if position_amt > 0 else 'SHORT',
                    'size': abs(position_amt),
                    'entry_price': float(pos['entryPrice']),
                    'mark_price': float(pos['markPrice']),
                    'pnl': float(pos['unRealizedProfit'])
                })
        
        log_debug(f"📊 Açık pozisyonlar: {len(active_positions)} adet")
        for pos in active_positions:
            log_debug(f"   💰 {pos['symbol']}: {pos['side']} {pos['size']} @ ${pos['entry_price']:.6f} | PnL: ${pos['pnl']:.2f}")
        
        return active_positions
    except Exception as e:
        log_debug(f"❌ Pozisyon alma hatası: {e}")
        return []

def get_open_orders(client, symbol=None):
    """Açık emirleri al"""
    try:
        if symbol:
            orders = client.futures_get_open_orders(symbol=symbol, recvWindow=60000)
        else:
            orders = client.futures_get_open_orders(recvWindow=60000)
        
        log_debug(f"📋 Açık emirler: {len(orders)} adet")
        for order in orders:
            log_debug(f"   📝 {order['symbol']}: {order['type']} {order['side']} | Status: {order['status']} | ID: {order['orderId']}")
            test_orders[str(order['orderId'])] = order
        
        return orders
    except Exception as e:
        log_debug(f"❌ Açık emir alma hatası: {e}")
        return []

def handle_user_data(msg):
    """📨 WebSocket mesaj handler - Detaylı analiz"""
    try:
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        websocket_messages.append({'timestamp': timestamp, 'data': msg})
        
        msg_type = msg.get('e', 'unknown')
        
        log_debug(f"\n🎉 *** WEBSOCKET MESAJI GELDİ! *** 🎉")
        log_debug(f"📨 [{timestamp}] Mesaj Türü: {msg_type}")
        
        if msg_type == 'executionReport':
            symbol = msg.get('s', 'N/A')
            order_type = msg.get('o', 'N/A')
            order_status = msg.get('X', 'N/A')
            order_id = str(msg.get('i', 'N/A'))
            side = msg.get('S', 'N/A')
            quantity = msg.get('q', 'N/A')
            price = msg.get('p', 'N/A')
            avg_price = msg.get('ap', 'N/A')
            filled_qty = msg.get('z', 'N/A')
            
            log_debug(f"   🏷️  Symbol: {symbol}")
            log_debug(f"   📋 Order ID: {order_id}")
            log_debug(f"   🔄 Type: {order_type}")
            log_debug(f"   📊 Status: {order_status}")
            log_debug(f"   ⬆️⬇️ Side: {side}")
            log_debug(f"   💰 Quantity: {quantity}")
            log_debug(f"   💲 Price: {price}")
            log_debug(f"   📈 Avg Price: {avg_price}")
            log_debug(f"   ✅ Filled Qty: {filled_qty}")
            
            # 🔍 SORUN ANALİZİ: Bu bizim emrimiz mi?
            if order_id in test_orders:
                log_debug(f"   🎯 *** BU BİZİM EMRİMİZ! ***")
                original_order = test_orders[order_id]
                log_debug(f"   📝 Orijinal emir: {original_order['type']} {original_order['side']}")
            else:
                log_debug(f"   ❓ Bilinmeyen emir - test_orders'da yok")
            
            # ⭐ SL/TP kontrolü
            if order_status == 'FILLED' and order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
                log_debug(f"   🎯 *** SL/TP TETİKLENDİ: {symbol} {order_type} ***")
                log_debug(f"   🚨 SORUN BURADA: Bu otomatik kapatma mı?")
                
        elif msg_type == 'ACCOUNT_UPDATE':
            log_debug(f"   💰 Hesap güncellendi")
            # Hesap güncellemelerinin detayını göster
            if 'a' in msg:
                balances = msg['a'].get('B', [])
                for balance in balances:
                    if float(balance.get('wb', 0)) != 0:  # Wallet balance
                        log_debug(f"      💼 {balance.get('a', 'N/A')}: ${float(balance.get('wb', 0)):.2f}")
                
                positions = msg['a'].get('P', [])
                for pos in positions:
                    if float(pos.get('pa', 0)) != 0:  # Position amount
                        log_debug(f"      📊 {pos.get('s', 'N/A')}: {float(pos.get('pa', 0))} @ ${float(pos.get('ep', 0)):.6f}")
            
        elif msg_type == 'ORDER_TRADE_UPDATE':
            log_debug(f"   📈 Trade güncellendi")
            order_data = msg.get('o', {})
            symbol = order_data.get('s', 'N/A')
            order_id = str(order_data.get('i', 'N/A'))
            status = order_data.get('X', 'N/A')
            
            log_debug(f"      🏷️ Symbol: {symbol}")
            log_debug(f"      📋 Order ID: {order_id}")
            log_debug(f"      📊 Status: {status}")
            
            if order_id in test_orders:
                log_debug(f"      🎯 Bu bizim emrimiz!")
            
        else:
            log_debug(f"   🔍 Bilinmeyen mesaj türü: {msg_type}")
        
        # Raw data'yı da kaydet
        log_debug(f"📦 Raw Message: {json.dumps(msg, indent=2)}")
            
    except Exception as e:
        log_debug(f"   ❌ Mesaj işleme hatası: {e}")
        log_debug(f"   📦 Raw data: {msg}")

def monitor_orders_rest(client, symbol, duration=60):
    """REST API ile emirleri izle (paralel thread)"""
    log_debug(f"🔄 REST monitoring başladı: {symbol} ({duration}s)")
    
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            orders = get_open_orders(client, symbol)
            time.sleep(5)  # 5 saniyede bir kontrol
        except Exception as e:
            log_debug(f"❌ REST monitoring hatası: {e}")
        
    log_debug(f"🛑 REST monitoring bitti: {symbol}")

def test_live_bot_scenario():
    """🧪 Live bot senaryosunu test et"""
    log_debug("=" * 60)
    log_debug("🧪 LIVE BOT SORUN TESPİT TESTİ BAŞLIYOR")
    log_debug("=" * 60)
    
    if not BINANCE_AVAILABLE:
        log_debug("❌ python-binance kütüphanesi gerekli!")
        return
        
    if not API_KEY or API_KEY == 'YOUR_API_KEY_HERE':
        log_debug("❌ .env dosyasında BINANCE_API_KEY ayarlayın!")
        return
    
    # 1) Binance client oluştur
    log_debug("\n📡 1) Binance client bağlantısı...")
    client = create_test_client()
    if not client:
        return
    
    # 2) Mevcut durumu analiz et
    log_debug("\n📊 2) Mevcut durum analizi...")
    positions = get_active_positions(client)
    all_orders = get_open_orders(client)
    
    # 3) Test için sembol seç
    test_symbol = "BTCUSDT"  # En aktif coin
    log_debug(f"\n🎯 3) Test sembolü: {test_symbol}")
    
    # 4) ListenKey al
    log_debug("\n📡 4) WebSocket ListenKey alınıyor...")
    listen_key = get_listen_key()
    if not listen_key:
        return
    
    # 5) WebSocket başlat
    log_debug("\n🔗 5) WebSocket bağlantısı kuruluyor...")
    try:
        twm = ThreadedWebsocketManager(
            api_key=API_KEY,
            api_secret=API_SECRET,
            testnet=TESTNET
        )
        
        # Start manager
        twm.start()
        log_debug("✅ WebSocket Manager başlatıldı")
        
        # UserData stream başlat
        stream_name = twm.start_futures_user_socket(callback=handle_user_data)
        log_debug(f"✅ UserData Stream başlatıldı: {stream_name}")
        
        # 6) REST API monitoring başlat (paralel)
        log_debug(f"\n🔄 6) REST API monitoring başlatılıyor...")
        rest_thread = threading.Thread(
            target=monitor_orders_rest, 
            args=(client, test_symbol, 120),  # 2 dakika
            daemon=True
        )
        rest_thread.start()
        
        # 7) Test süresi
        log_debug(f"\n👂 7) WebSocket + REST dinleme başladı (2 dakika)...")
        log_debug(f"💡 Test sırasında {test_symbol} için manuel emir verin")
        log_debug(f"🎯 Özellikle SL/TP emirleri test edin")
        log_debug(f"🔍 Live bot'un erken kapatma problemini analiz ediyoruz")
        log_debug(f"🛑 Çıkmak için Ctrl+C")
        
        # Her 10 saniyede durum raporu
        for i in range(12):  # 2 dakika = 12 x 10 saniye
            try:
                log_debug(f"\n⏰ Test süresi: {(i+1)*10}/120 saniye")
                log_debug(f"📊 WebSocket mesajları: {len(websocket_messages)}")
                log_debug(f"📋 Takip edilen emirler: {len(test_orders)}")
                
                # Son 10 saniyedeki mesajları özetle
                recent_messages = [m for m in websocket_messages if 
                                 (datetime.now() - datetime.strptime(m['timestamp'], '%H:%M:%S.%f')).total_seconds() < 10]
                if recent_messages:
                    log_debug(f"📨 Son 10s'de {len(recent_messages)} WebSocket mesajı")
                
                time.sleep(10)
            except KeyboardInterrupt:
                log_debug("\n⌨️ Kullanıcı durdurdu")
                break
        
        # 8) Test sonuçları
        log_debug("\n📊 8) TEST SONUÇLARI:")
        log_debug(f"   📨 Toplam WebSocket mesajı: {len(websocket_messages)}")
        log_debug(f"   📋 Takip edilen emirler: {len(test_orders)}")
        log_debug(f"   🔄 Debug log satırları: {len(debug_log)}")
        
        # SL/TP tetikleme analizi
        sltp_triggers = [m for m in websocket_messages 
                        if m['data'].get('e') == 'executionReport' 
                        and m['data'].get('o') in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']
                        and m['data'].get('X') == 'FILLED']
        
        if sltp_triggers:
            log_debug(f"   🎯 SL/TP tetiklenmesi: {len(sltp_triggers)} adet")
            for trigger in sltp_triggers:
                data = trigger['data']
                log_debug(f"      ⚡ {trigger['timestamp']}: {data.get('s')} {data.get('o')} FILLED")
        else:
            log_debug(f"   📝 SL/TP tetiklenmesi görülmedi")
        
        # Temizlik
        log_debug("\n🧹 9) WebSocket kapatılıyor...")
        try:
            twm.stop()
        except Exception as stop_error:
            log_debug(f"⚠️ Kapatma hatası (normal): {stop_error}")
        
        # 10) Detaylı rapor
        log_debug("\n📋 10) DETAYLI RAPOR:")
        log_debug(f"Bu test Live Bot'un erken kapatma problemini analiz etti.")
        log_debug(f"Eğer test sırasında emirler otomatik kapandıysa,")
        log_debug(f"yukarıdaki WebSocket mesajlarında nedenini görebilirsiniz.")
        
        log_debug("✅ Test tamamlandı!")
        
        # Debug log'u dosyaya yaz
        try:
            with open('debug_websocket_test.log', 'w', encoding='utf-8') as f:
                for log_entry in debug_log:
                    f.write(log_entry + '\n')
            log_debug("📁 Debug log 'debug_websocket_test.log' dosyasına yazıldı")
        except Exception as e:
            log_debug(f"❌ Log dosyası yazma hatası: {e}")
        
    except KeyboardInterrupt:
        log_debug("\n⌨️ Kullanıcı durdurdu")
        try:
            twm.stop()
        except:
            pass
    except Exception as e:
        log_debug(f"❌ WebSocket test hatası: {e}")
        try:
            twm.stop()
        except:
            pass

if __name__ == "__main__":
    test_live_bot_scenario()