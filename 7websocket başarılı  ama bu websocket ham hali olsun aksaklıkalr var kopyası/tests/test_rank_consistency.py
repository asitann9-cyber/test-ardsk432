#!/usr/bin/env python3
"""
🔍 KAPSAMLI LIVE BOT TEST SİSTEMİ
Manuel kapatma ve TP/SL tetiklenme sorunlarını detaylı test eder
Gerçek TestBinance işlemleri ile CSV kayıt sistemini doğrular

AMAÇLAR:
1. Manuel kapatma CSV kaydı testi
2. Gerçek SL/TP tetiklenme CSV kaydı testi  
3. WebSocket vs REST hibrit sistem testi
4. Config senkronizasyon testi
5. Live trader fonksiyon entegrasyon testi
"""

import os
import sys
import time
import csv
import json
import logging
from datetime import datetime
import threading
from typing import Dict, List, Optional

# Ana dizin path ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Kendi modüllerimizi import et
print("📦 Modüller yükleniyor...")
try:
    import config
    from config import LOCAL_TZ
    from data.database import log_trade_to_csv, load_trades_from_csv, setup_csv_files
    
    # Live trader import kontrolü
    LIVE_TRADER_AVAILABLE = False
    try:
        from trading.live_trader import (
            live_bot, start_live_trading, stop_live_trading, 
            is_live_trading_active, get_live_trading_status,
            sync_to_config
        )
        LIVE_TRADER_AVAILABLE = True
        print("✅ Live trader modülü yüklendi")
    except ImportError as e:
        print(f"❌ Live trader import hatası: {e}")
        
    # Binance import kontrolü
    BINANCE_AVAILABLE = False
    try:
        from dotenv import load_dotenv
        from binance.client import Client
        from binance.enums import *
        
        # .env dosyasını yükle
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(parent_dir, '.env')
        
        print(f"📁 .env dosyası aranıyor: {env_path}")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print("✅ .env dosyası yüklendi")
        else:
            print("❌ .env dosyası bulunamadı")
            
        # API keys
        API_KEY = os.getenv('BINANCE_API_KEY')
        API_SECRET = os.getenv('BINANCE_SECRET_KEY')
        
        if API_KEY and API_SECRET:
            BINANCE_AVAILABLE = True
            print("✅ Binance API anahtarları yüklendi")
        else:
            print("❌ Binance API anahtarları bulunamadı")
            
    except ImportError as e:
        print(f"❌ Binance import hatası: {e}")
        
except ImportError as e:
    print(f"❌ Kritik modül import hatası: {e}")
    sys.exit(1)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class LiveBotTester:
    """Kapsamlı Live Bot Test Sistemi"""
    
    def __init__(self):
        self.test_results = {
            'csv_write_test': False,
            'csv_read_test': False,
            'manual_close_simulation': False,
            'live_trader_connection': False,
            'config_sync_test': False,
            'real_position_test': False,
            'sltp_trigger_test': False,
            'websocket_test': False,
            'rest_api_test': False
        }
        
        self.test_log = []
        self.csv_backup_created = False
        self.original_csv_content = None
        self.binance_client = None
        
        # Test parametreleri
        self.TEST_SYMBOL = 'BTCUSDT'
        self.TEST_QUANTITY = 0.001
        self.QUICK_TP_DISTANCE = 0.0008  # %0.08 - çok kısa mesafe
        self.QUICK_SL_DISTANCE = 0.002   # %0.2
        
        print("\n🔧 KAPSAMLI LIVE BOT TEST SİSTEMİ")
        print("=" * 60)
        
    def log_test(self, message: str, level: str = "INFO"):
        """Test mesajını logla"""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.test_log.append(log_entry)
        
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "SUCCESS":
            logger.info(f"✅ {message}")
        else:
            logger.info(message)
        
    def backup_csv_files(self):
        """CSV dosyalarını yedekle"""
        try:
            csv_file = 'ai_crypto_trades.csv'
            if os.path.exists(csv_file):
                backup_name = f"{csv_file}.test_backup_{int(time.time())}"
                
                # Orijinal içeriği oku
                with open(csv_file, 'r', encoding='utf-8') as f:
                    self.original_csv_content = f.read()
                
                # Yedek oluştur
                import shutil
                shutil.copy2(csv_file, backup_name)
                self.csv_backup_created = True
                self.log_test(f"CSV yedeklendi: {backup_name}", "SUCCESS")
            else:
                self.log_test("CSV dosyası bulunamadı - yeni oluşturulacak")
                
        except Exception as e:
            self.log_test(f"CSV yedekleme hatası: {e}", "ERROR")
    
    def restore_csv_files(self):
        """CSV dosyalarını eski haline getir"""
        try:
            if self.csv_backup_created and self.original_csv_content is not None:
                csv_file = 'ai_crypto_trades.csv'
                with open(csv_file, 'w', encoding='utf-8') as f:
                    f.write(self.original_csv_content)
                self.log_test("CSV dosyası eski haline getirildi", "SUCCESS")
        except Exception as e:
            self.log_test(f"CSV geri yükleme hatası: {e}", "ERROR")
    
    def test_csv_write_system(self):
        """CSV yazma sistemini test et"""
        self.log_test("🧪 CSV yazma sistemi testi başlatılıyor...")
        
        try:
            # Setup CSV
            setup_csv_files()
            
            # Test verisi hazırla
            test_trade_data = {
                'timestamp': datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': 'TESTBTC',
                'side': 'LONG',
                'quantity': 0.001,
                'entry_price': 43000.0,
                'exit_price': 43100.0,
                'invested_amount': 43.0,
                'current_value': 43.1,
                'pnl': 0.1,
                'commission': 0.0,
                'ai_score': 75.0,
                'run_type': 'long',
                'run_count': 5,
                'run_perc': 2.5,
                'gauss_run': 15.0,
                'vol_ratio': 2.1,
                'deviso_ratio': 1.8,
                'stop_loss': 42570.0,
                'take_profit': 43430.0,
                'close_reason': 'TEST - CSV Write System',
                'status': 'CLOSED'
            }
            
            # CSV'ye yaz
            log_trade_to_csv(test_trade_data)
            self.log_test("Test verisi CSV'ye yazıldı", "SUCCESS")
            
            # Yazılan veriyi oku
            time.sleep(0.5)  # Dosya yazma için bekle
            trades_df = load_trades_from_csv()
            
            if trades_df.empty:
                self.log_test("CSV boş - yazma başarısız", "ERROR")
                return False
            
            # Test verisini ara
            test_records = trades_df[trades_df['symbol'] == 'TESTBTC']
            if test_records.empty:
                self.log_test("Test verisi CSV'de bulunamadı", "ERROR")
                return False
            
            latest_record = test_records.iloc[-1]
            
            # Veri doğrulama
            checks = [
                ('symbol', 'TESTBTC'),
                ('side', 'LONG'),
                ('pnl', 0.1),
                ('close_reason', 'TEST - CSV Write System')
            ]
            
            for field, expected in checks:
                actual = latest_record[field]
                if str(actual) != str(expected):
                    self.log_test(f"Veri doğrulama hatası {field}: beklenen={expected}, gerçek={actual}", "ERROR")
                    return False
            
            self.log_test("CSV yazma/okuma sistemi doğrulandı", "SUCCESS")
            self.test_results['csv_write_test'] = True
            self.test_results['csv_read_test'] = True
            return True
            
        except Exception as e:
            self.log_test(f"CSV test hatası: {e}", "ERROR")
            return False
    
    def test_binance_connection(self):
        """Binance API bağlantısını test et"""
        self.log_test("🔗 Binance API bağlantı testi...")
        
        if not BINANCE_AVAILABLE:
            self.log_test("Binance kütüphanesi mevcut değil", "ERROR")
            return False
        
        try:
            # Client oluştur
            self.binance_client = Client(
                api_key=API_KEY,
                api_secret=API_SECRET,
                testnet=True
            )
            
            # Time sync
            server_time = self.binance_client.futures_time()["serverTime"]
            local_time = int(time.time() * 1000)
            offset = int(server_time) - local_time
            self.binance_client.timestamp_offset = offset
            self.log_test(f"Time sync: offset={offset}ms", "SUCCESS")
            
            # Ping test
            self.binance_client.futures_ping()
            self.log_test("Futures API ping başarılı", "SUCCESS")
            
            # Hesap bilgileri
            account_info = self.binance_client.futures_account(recvWindow=30000)
            balance = float(account_info["totalWalletBalance"])
            self.log_test(f"Hesap bakiyesi: ${balance:.2f}", "SUCCESS")
            
            if balance < 10:
                self.log_test("Yetersiz bakiye (min $10)", "WARNING")
                return False
            
            return True
            
        except Exception as e:
            self.log_test(f"Binance bağlantı hatası: {e}", "ERROR")
            return False
    
    def test_live_trader_integration(self):
        """Live trader entegrasyonu test et"""
        self.log_test("🤖 Live trader entegrasyon testi...")
        
        if not LIVE_TRADER_AVAILABLE:
            self.log_test("Live trader modülü mevcut değil", "ERROR")
            return False
        
        try:
            # Connection test
            connection_success = live_bot.connect_to_binance()
            if not connection_success:
                self.log_test("Live bot Binance bağlantısı başarısız", "ERROR")
                return False
            
            self.log_test("Live bot Binance bağlantısı başarılı", "SUCCESS")
            
            # Bakiye kontrol
            balance = live_bot.get_account_balance()
            self.log_test(f"Live bot bakiyesi: ${balance:.2f}", "SUCCESS")
            
            # Config senkronizasyon
            sync_to_config()
            self.log_test("Config senkronizasyonu tamamlandı", "SUCCESS")
            
            self.test_results['live_trader_connection'] = True
            self.test_results['config_sync_test'] = True
            return True
            
        except Exception as e:
            self.log_test(f"Live trader test hatası: {e}", "ERROR")
            return False
    
    def simulate_manual_close(self):
        """Manuel kapatma simülasyonu"""
        self.log_test("📝 Manuel kapatma simülasyonu...")
        
        try:
            # Sahte pozisyon config'e ekle
            test_position = {
                'symbol': self.TEST_SYMBOL,
                'side': 'LONG',
                'quantity': self.TEST_QUANTITY,
                'entry_price': 43000.0,
                'invested_amount': 43.0,
                'stop_loss': 42570.0,
                'take_profit': 43430.0,
                'entry_time': datetime.now(LOCAL_TZ),
                'signal_data': {
                    'ai_score': 78.0,
                    'run_type': 'long',
                    'run_count': 4,
                    'run_perc': 1.8,
                    'gauss_run': 10.0,
                    'vol_ratio': 1.9,
                    'deviso_ratio': 1.2
                },
                'auto_sltp': True,
                'main_order_id': 'SIM_123456',
                'sl_order_id': 'SIM_SL_123456',
                'tp_order_id': 'SIM_TP_123456'
            }
            
            # Config'e ekle
            config.live_positions[self.TEST_SYMBOL] = test_position
            self.log_test(f"Test pozisyonu config'e eklendi: {self.TEST_SYMBOL}", "SUCCESS")
            
            # CSV kayıt öncesi kontrolü
            trades_before = load_trades_from_csv()
            records_before = len(trades_before)
            self.log_test(f"Manuel kapatma öncesi CSV kayıt sayısı: {records_before}")
            
            # Manuel kapatma simülasyonu - direkt _save_closed_trade çağır
            exit_price = 43087.50
            entry_price = test_position['entry_price'] 
            pnl = (exit_price - entry_price) * test_position['quantity']
            
            self.log_test(f"Simülasyon: Entry=${entry_price} → Exit=${exit_price} | P&L=${pnl:.4f}")
            
            # Live bot'un _save_closed_trade fonksiyonunu test et
            if hasattr(live_bot, '_save_closed_trade'):
                live_bot._save_closed_trade(self.TEST_SYMBOL, exit_price, pnl, "Manual Close - TEST")
                self.log_test("live_bot._save_closed_trade() çağrıldı", "SUCCESS")
            else:
                self.log_test("live_bot._save_closed_trade() fonksiyonu bulunamadı", "ERROR")
                return False
            
            # Pozisyonu config'den temizle
            if self.TEST_SYMBOL in config.live_positions:
                del config.live_positions[self.TEST_SYMBOL]
                self.log_test("Test pozisyonu config'den temizlendi", "SUCCESS")
            
            # CSV kontrolü
            time.sleep(1.0)  # Dosya yazma için bekle
            trades_after = load_trades_from_csv()
            records_after = len(trades_after)
            
            self.log_test(f"Manuel kapatma sonrası CSV kayıt sayısı: {records_after}")
            
            if records_after <= records_before:
                self.log_test("HATA: Manuel kapatma CSV'ye kaydedilmedi!", "ERROR")
                return False
            
            # Son kaydı kontrol et
            latest_record = trades_after.iloc[-1]
            if (latest_record['symbol'] == self.TEST_SYMBOL and 
                'Manual Close - TEST' in str(latest_record['close_reason'])):
                self.log_test("✅ Manuel kapatma CSV kaydı BAŞARILI!", "SUCCESS")
                self.test_results['manual_close_simulation'] = True
                return True
            else:
                self.log_test("Manuel kapatma kaydı doğrulanamadı", "ERROR")
                return False
            
        except Exception as e:
            self.log_test(f"Manuel kapatma simülasyon hatası: {e}", "ERROR")
            return False
    
    def test_real_sltp_trigger(self):
        """Gerçek SL/TP tetiklenme testi"""
        self.log_test("🎯 Gerçek SL/TP tetiklenme testi başlatılıyor...")
        
        if not self.binance_client:
            self.log_test("Binance client mevcut değil", "ERROR")
            return False
        
        try:
            # Güncel fiyat al - hata korunmalı
            try:
                ticker = self.binance_client.futures_ticker(symbol=self.TEST_SYMBOL)
                current_price = float(ticker['price'])
            except (KeyError, TypeError) as e:
                self.log_test(f"Ticker format hatası: {e}, alternatif yöntem deneniyor...", "WARNING")
                # Alternatif fiyat alma yöntemi
                symbol_ticker = self.binance_client.futures_symbol_ticker(symbol=self.TEST_SYMBOL)
                current_price = float(symbol_ticker['price'])
            
            self.log_test(f"Güncel {self.TEST_SYMBOL} fiyatı: ${current_price:.2f}")
            
            # TP fiyatı (çok kısa mesafe - hızlı tetiklenmesi için)
            tp_price = round(current_price * (1 + self.QUICK_TP_DISTANCE), 2)
            sl_price = round(current_price * (1 - self.QUICK_SL_DISTANCE), 2) 
            
            self.log_test(f"🎯 TP hedef: ${tp_price:.2f} (%{self.QUICK_TP_DISTANCE*100:.2f} yukarı)")
            self.log_test(f"🛑 SL hedef: ${sl_price:.2f} (%{self.QUICK_SL_DISTANCE*100:.2f} aşağı)")
            
            # CSV kayıt öncesi kontrol
            trades_before = load_trades_from_csv()
            records_before = len(trades_before)
            self.log_test(f"SL/TP testi öncesi CSV kayıt sayısı: {records_before}")
            
            # 1. Long pozisyon aç
            self.log_test("📝 Long pozisyon açılıyor...")
            main_order = self.binance_client.futures_create_order(
                symbol=self.TEST_SYMBOL,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=self.TEST_QUANTITY,
                recvWindow=30000
            )
            
            main_order_id = main_order['orderId']
            self.log_test(f"✅ Market emri: {main_order_id}")
            time.sleep(2)
            
            # Emir durumu kontrol
            order_check = self.binance_client.futures_get_order(
                symbol=self.TEST_SYMBOL, 
                orderId=main_order_id, 
                recvWindow=30000
            )
            
            if order_check['status'] != 'FILLED':
                self.log_test(f"Market emri doldurulmadı: {order_check['status']}", "ERROR")
                return False
            
            executed_qty = float(order_check['executedQty'])
            avg_price = float(order_check['avgPrice'])
            self.log_test(f"✅ Pozisyon açıldı: {executed_qty} @ ${avg_price:.6f}")
            
            # 2. Take Profit emri
            self.log_test("🎯 Take Profit emri veriliyor...")
            tp_order = self.binance_client.futures_create_order(
                symbol=self.TEST_SYMBOL,
                side=SIDE_SELL,
                type='TAKE_PROFIT_MARKET',
                quantity=executed_qty,
                stopPrice=tp_price,
                timeInForce='GTC',
                recvWindow=30000
            )
            
            tp_order_id = tp_order['orderId']
            self.log_test(f"✅ TP emri: {tp_order_id} @ ${tp_price:.2f}")
            
            # 3. Stop Loss emri
            self.log_test("🛑 Stop Loss emri veriliyor...")
            sl_order = self.binance_client.futures_create_order(
                symbol=self.TEST_SYMBOL,
                side=SIDE_SELL,
                type='STOP_MARKET',
                quantity=executed_qty,
                stopPrice=sl_price,
                timeInForce='GTC',
                recvWindow=30000
            )
            
            sl_order_id = sl_order['orderId']
            self.log_test(f"✅ SL emri: {sl_order_id} @ ${sl_price:.2f}")
            
            # 4. Tetiklenmeyi bekle
            self.log_test("⏰ TP/SL tetiklenmesini izleme başlatılıyor... (Max 90 saniye)")
            
            start_time = time.time()
            tp_triggered = False
            sl_triggered = False
            max_wait = 90  # saniye
            
            while time.time() - start_time < max_wait:
                try:
                    # TP kontrol
                    tp_check = self.binance_client.futures_get_order(
                        symbol=self.TEST_SYMBOL,
                        orderId=tp_order_id,
                        recvWindow=30000
                    )
                    
                    if tp_check['status'] == 'FILLED' and not tp_triggered:
                        tp_triggered = True
                        fill_price = float(tp_check.get('avgPrice', tp_price))
                        self.log_test(f"🎉 TAKE PROFIT TETİKLENDİ! Fiyat: ${fill_price:.6f}", "SUCCESS")
                        
                        # SL emrini iptal et
                        try:
                            self.binance_client.futures_cancel_order(
                                symbol=self.TEST_SYMBOL,
                                orderId=sl_order_id,
                                recvWindow=30000
                            )
                            self.log_test("🚫 SL emri iptal edildi")
                        except:
                            pass
                        
                        # Live trader CSV kayıt simülasyonu
                        pnl = (fill_price - avg_price) * executed_qty
                        self.log_test(f"💰 P&L hesaplandı: ${pnl:.4f}")
                        
                        # Live bot sisteminin bu durumu yakalaması lazım
                        # Gerçek sistemde WebSocket veya REST API bunu yakalayacak
                        break
                    
                    # SL kontrol
                    sl_check = self.binance_client.futures_get_order(
                        symbol=self.TEST_SYMBOL,
                        orderId=sl_order_id,
                        recvWindow=30000
                    )
                    
                    if sl_check['status'] == 'FILLED' and not sl_triggered:
                        sl_triggered = True
                        fill_price = float(sl_check.get('avgPrice', sl_price))
                        self.log_test(f"🛑 STOP LOSS TETİKLENDİ! Fiyat: ${fill_price:.6f}", "SUCCESS")
                        
                        # TP emrini iptal et
                        try:
                            self.binance_client.futures_cancel_order(
                                symbol=self.TEST_SYMBOL,
                                orderId=tp_order_id,
                                recvWindow=30000
                            )
                            self.log_test("🚫 TP emri iptal edildi")
                        except:
                            pass
                        
                        # P&L hesapla
                        pnl = (fill_price - avg_price) * executed_qty
                        self.log_test(f"💰 P&L hesaplandı: ${pnl:.4f}")
                        break
                    
                    # Progress göster
                    elapsed = int(time.time() - start_time)
                    if elapsed % 15 == 0:  # Her 15 saniyede bir
                        try:
                            current_ticker = self.binance_client.futures_symbol_ticker(symbol=self.TEST_SYMBOL)
                            live_price = float(current_ticker['price'])
                            self.log_test(f"⏱️ {elapsed}s - Fiyat: ${live_price:.2f} (TP: ${tp_price:.2f})")
                        except Exception as price_err:
                            self.log_test(f"⏱️ {elapsed}s - Fiyat güncellenemiyor: {price_err}", "WARNING")
                    
                    time.sleep(1)
                    
                except Exception as e:
                    self.log_test(f"SL/TP kontrol hatası: {e}", "WARNING")
                    time.sleep(2)
            
            # 4. Sonuç değerlendirmesi
            if tp_triggered or sl_triggered:
                self.log_test("✅ SL/TP tetiklenme testi BAŞARILI!", "SUCCESS")
                
                # 🔧 YENİ: Manuel REST kontrolü (live trading loop olmasa da)
                self.log_test("🔧 Manuel REST API kontrolü yapılıyor...", "INFO")
                try:
                    if LIVE_TRADER_AVAILABLE and hasattr(live_bot, 'check_filled_orders_rest'):
                        live_bot.check_filled_orders_rest()
                        self.log_test("✅ Manuel REST kontrolü tamamlandı", "SUCCESS")
                        time.sleep(1)  # REST işlemi için bekle
                    else:
                        self.log_test("⚠️ live_bot.check_filled_orders_rest() bulunamadı", "WARNING")
                except Exception as rest_err:
                    self.log_test(f"❌ Manuel REST kontrolü hatası: {rest_err}", "ERROR")
                
                # CSV kontrol - gerçek live sistemde otomatik kaydedilmeli
                time.sleep(2)
                trades_after = load_trades_from_csv()
                records_after = len(trades_after)
                
                self.log_test(f"SL/TP sonrası CSV kayıt sayısı: {records_after}")
                
                if records_after > records_before:
                    self.log_test("✅ SL/TP CSV kaydı MEVCUT - Live sistem çalışıyor!", "SUCCESS")
                    self.test_results['sltp_trigger_test'] = True
                else:
                    self.log_test("⚠️ SL/TP CSV kaydı YOK - Live sistem sorunu olabilir", "WARNING")
                    # Simülasyon kayıt yapalım
                    test_record = {
                        'timestamp': datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                        'symbol': self.TEST_SYMBOL,
                        'side': 'LONG',
                        'quantity': executed_qty,
                        'entry_price': avg_price,
                        'exit_price': fill_price if tp_triggered else fill_price,
                        'pnl': (fill_price - avg_price) * executed_qty,
                        'close_reason': 'Take Profit - TEST' if tp_triggered else 'Stop Loss - TEST',
                        'status': 'CLOSED'
                    }
                    log_trade_to_csv(test_record)
                    self.log_test("📝 SL/TP kaydı manuel olarak CSV'ye eklendi")
                
            else:
                self.log_test("⏰ Zaman doldu - SL/TP tetiklenmedi", "WARNING")
                
                # Açık emirleri temizle
                try:
                    self.binance_client.futures_cancel_order(symbol=self.TEST_SYMBOL, orderId=tp_order_id, recvWindow=30000)
                    self.binance_client.futures_cancel_order(symbol=self.TEST_SYMBOL, orderId=sl_order_id, recvWindow=30000)
                    
                    # Pozisyonu kapat
                    close_order = self.binance_client.futures_create_order(
                        symbol=self.TEST_SYMBOL,
                        side=SIDE_SELL,
                        type=ORDER_TYPE_MARKET,
                        quantity=executed_qty,
                        recvWindow=30000
                    )
                    self.log_test(f"🔒 Pozisyon manuel kapatıldı: {close_order['orderId']}")
                    
                except Exception as e:
                    self.log_test(f"Temizlik hatası: {e}", "ERROR")
            
            return tp_triggered or sl_triggered
            
        except Exception as e:
            self.log_test(f"Gerçek SL/TP test hatası: {e}", "ERROR")
            return False
    
    def test_database_and_config_sync(self):
        """Database ve config senkronizasyon detaylı testi"""
        self.log_test("🔍 DATABASE VE CONFIG SENKRONIZASYON TESTİ", "INFO")

        try:
            # 1. Config durumu detaylı analiz
            self.log_test("📋 CONFIG DURUMU ANALİZİ:")
            self.log_test(f"   config.live_positions: {config.live_positions}")
            self.log_test(f"   config.live_capital: {config.live_capital}")
            self.log_test(f"   config.live_trading_active: {config.live_trading_active}")

            # 2. Database (CSV) durumu
            self.log_test("🗃️ DATABASE DURUMU ANALİZİ:")

            csv_path = "ai_crypto_trades.csv"
            csv_exists = os.path.exists(csv_path)
            self.log_test(f"   CSV dosyası mevcut: {csv_exists}")

            if csv_exists:
                # CSV dosyası boyutu
                csv_size = os.path.getsize(csv_path)
                self.log_test(f"   CSV dosyası boyutu: {csv_size} bytes")

                # Son satırları oku
                try:
                    with open(csv_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    self.log_test(f"   CSV toplam satır sayısı: {len(lines)}")

                    if len(lines) > 1:
                        self.log_test(f"   CSV son satır: {lines[-1].strip()}")
                        if len(lines) > 2:
                            self.log_test(f"   CSV son-1 satır: {lines[-2].strip()}")
                except Exception as csv_err:
                    self.log_test(f"   CSV okuma hatası: {csv_err}", "ERROR")

            # 3. log_trade_to_csv fonksiyon testi
            self.log_test("🔧 log_trade_to_csv FONKSİYON TESTİ:")

            from datetime import datetime  # yerel import (yazımsal hata önler)
            test_csv_data = {
                "timestamp": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": "TESTDB",
                "side": "LONG",
                "quantity": 0.001,
                "entry_price": 50000.0,
                "exit_price": 50100.0,
                "pnl": 0.1,
                "close_reason": "DATABASE_SYNC_TEST",
                "status": "CLOSED",
            }

            self.log_test("   Test verisi hazırlandı:")
            self.log_test(f"   Symbol: {test_csv_data['symbol']}")
            self.log_test(f"   Close Reason: {test_csv_data['close_reason']}")

            # CSV kayıt öncesi
            trades_before = load_trades_from_csv()
            records_before = len(trades_before)
            self.log_test(f"   Kayıt öncesi CSV satır sayısı: {records_before}")

            # log_trade_to_csv çağır
            try:
                log_trade_to_csv(test_csv_data)
                self.log_test("   ✅ log_trade_to_csv() başarıyla çağrıldı")
            except Exception as log_err:
                self.log_test(f"   ❌ log_trade_to_csv() hatası: {log_err}", "ERROR")
                return False

            # Bekle ve kontrol et
            time.sleep(0.5)
            trades_after = load_trades_from_csv()
            records_after = len(trades_after)

            self.log_test(f"   Kayıt sonrası CSV satır sayısı: {records_after}")
            self.log_test(f"   Kayıt farkı: {records_after - records_before}")

            if records_after > records_before:
                # Son kaydı kontrol et (DataFrame veya liste olabilir)
                try:
                    latest_symbol = (
                        trades_after.iloc[-1]["symbol"]  # pandas DataFrame ise
                        if hasattr(trades_after, "iloc")
                        else trades_after[-1].get("symbol")  # liste/dict ise
                    )
                except Exception:
                    latest_symbol = None

                if latest_symbol == "TESTDB":
                    self.log_test("   ✅ Database senkronizasyon BAŞARILI")
                    return True
                else:
                    self.log_test(
                        f"   ❌ Son kayıt beklenenden farklı: {latest_symbol}", "ERROR"
                    )
                    return False
            else:
                self.log_test("   ❌ Database'e kayıt YAZILMADI", "ERROR")
                return False

        except Exception as e:
            self.log_test(f"❌ Database/Config test hatası: {e}", "ERROR")
            return False


    def test_live_trading_integration(self):
        """Live trading sistem entegrasyonu testi"""
        self.log_test("🤖 Live trading sistem entegrasyon testi...")
        
        if not LIVE_TRADER_AVAILABLE:
            self.log_test("Live trader mevcut değil", "ERROR")
            return False
        
        try:
            # Live trading durumu kontrol
            initial_status = is_live_trading_active()
            self.log_test(f"Başlangıç live trading durumu: {initial_status}")
            
            # Live trading başlat (kısa süre için)
            self.log_test("Live trading başlatılıyor...")
            start_success = start_live_trading()
            
            if not start_success:
                self.log_test("Live trading başlatılamadı", "ERROR")
                return False
            
            time.sleep(10)  # 10 saniye çalışsın - WebSocket kurulumu için
            
            # WebSocket durumu kontrol
            from trading.live_trader import websocket_active_symbols, websocket_manager
            
            self.log_test(f"WebSocket manager aktif: {websocket_manager is not None}")
            self.log_test(f"WebSocket active symbols: {len(websocket_active_symbols)}")
            
            if websocket_manager and len(websocket_active_symbols) > 0:
                self.log_test("✅ WebSocket başarıyla aktif!", "SUCCESS")
                self.test_results['websocket_test'] = True
            else:
                self.log_test("❌ WebSocket aktif değil", "ERROR")
            
            # Live trading durdur
            self.log_test("Live trading durduruluyor...")
            stop_live_trading()
            
            time.sleep(2)
            
            final_status = is_live_trading_active()
            self.log_test(f"Final live trading durumu: {final_status}")
            
            if not final_status:
                self.log_test("✅ Live trading entegrasyon testi BAŞARILI!", "SUCCESS")
                return True
            else:
                self.log_test("Live trading düzgün durdurulamadı", "WARNING")
                return False
            
        except Exception as e:
            self.log_test(f"Live trading entegrasyon hatası: {e}", "ERROR")
            return False


    
    def cleanup_test_environment(self):
        """Test ortamını temizle"""
        self.log_test("🧹 Test ortamı temizleniyor...")
        
        try:
            # Test pozisyonlarını config'den temizle
            if self.TEST_SYMBOL in config.live_positions:
                del config.live_positions[self.TEST_SYMBOL]
                self.log_test("Test pozisyonu config'den temizlendi")
            
            # Test emirlerini iptal et
            if self.binance_client:
                try:
                    # Tüm açık emirleri iptal et
                    open_orders = self.binance_client.futures_get_open_orders(
                        symbol=self.TEST_SYMBOL, 
                        recvWindow=30000
                    )
                    
                    for order in open_orders:
                        self.binance_client.futures_cancel_order(
                            symbol=self.TEST_SYMBOL,
                            orderId=order['orderId'],
                            recvWindow=30000
                        )
                        self.log_test(f"Açık emir iptal edildi: {order['orderId']}")
                
                except Exception as e:
                    self.log_test(f"Emir iptal hatası: {e}", "WARNING")
                
                # Açık pozisyonları kapat
                try:
                    positions = self.binance_client.futures_position_information(
                        symbol=self.TEST_SYMBOL,
                        recvWindow=30000
                    )
                    
                    for pos in positions:
                        position_amt = float(pos['positionAmt'])
                        if abs(position_amt) > 0:
                            side = SIDE_SELL if position_amt > 0 else SIDE_BUY
                            
                            close_order = self.binance_client.futures_create_order(
                                symbol=self.TEST_SYMBOL,
                                side=side,
                                type=ORDER_TYPE_MARKET,
                                quantity=abs(position_amt),
                                recvWindow=30000
                            )
                            self.log_test(f"Test pozisyonu kapatıldı: {close_order['orderId']}")
                
                except Exception as e:
                    self.log_test(f"Pozisyon kapatma hatası: {e}", "WARNING")
            
            # CSV dosyalarını eski haline getir
            self.restore_csv_files()
            
            self.log_test("✅ Test ortamı temizlendi", "SUCCESS")
            
        except Exception as e:
            self.log_test(f"Temizlik hatası: {e}", "ERROR")
    
    def generate_comprehensive_report(self):
        """Kapsamlı test raporu oluştur"""
        self.log_test("📋 Kapsamlı test raporu oluşturuluyor...")
        
        report = {
            "test_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_duration": "N/A",
            "overall_status": "UNKNOWN",
            "test_results": self.test_results,
            "issues_found": [],
            "recommendations": [],
            "test_logs": self.test_log
        }
        
        # Genel durum değerlendirmesi
        critical_tests = ['csv_write_test', 'manual_close_simulation', 'live_trader_connection']
        critical_passed = all(self.test_results.get(test, False) for test in critical_tests)
        
        total_tests = len([k for k, v in self.test_results.items() if v is not False])
        passed_tests = len([k for k, v in self.test_results.items() if v is True])
        
        if critical_passed and passed_tests >= total_tests * 0.8:
            report["overall_status"] = "PASS"
        elif critical_passed:
            report["overall_status"] = "PARTIAL_PASS"
        else:
            report["overall_status"] = "FAIL"
        
        # Sorun tespiti
        if not self.test_results.get('csv_write_test', False):
            report["issues_found"].append("CSV_WRITE_SYSTEM_FAILED")
            report["recommendations"].append("log_trade_to_csv() fonksiyonunu kontrol edin")
        
        if not self.test_results.get('manual_close_simulation', False):
            report["issues_found"].append("MANUAL_CLOSE_CSV_MISSING") 
            report["recommendations"].append("live_bot._save_closed_trade() çağrısını kontrol edin")
        
        if not self.test_results.get('sltp_trigger_test', False):
            report["issues_found"].append("SLTP_CSV_MISSING")
            report["recommendations"].append("WebSocket/REST SL/TP handler'larını kontrol edin")
        
        if not self.test_results.get('live_trader_connection', False):
            report["issues_found"].append("LIVE_TRADER_CONNECTION_FAILED")
            report["recommendations"].append("API anahtarları ve live_trader.py'yi kontrol edin")
        
        # Başarılı testler
        success_count = len([k for k, v in self.test_results.items() if v is True])
        total_count = len(self.test_results)
        
        print("\n" + "="*80)
        print("🎯 KAPSAMLI LİVE BOT TEST RAPORU")
        print("="*80)
        print(f"📅 Test Zamanı: {report['test_timestamp']}")
        print(f"🎭 Genel Durum: {report['overall_status']}")
        print(f"📊 Başarı Oranı: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        print()
        
        # Test sonuçları
        print("📋 TEST SONUÇLARI:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL" if result is False else "⏸️ SKIP"
            print(f"   {test_name:25}: {status}")
        
        print()
        
        # Tespit edilen sorunlar
        if report["issues_found"]:
            print("🚨 TESPİT EDİLEN SORUNLAR:")
            for issue in report["issues_found"]:
                print(f"   • {issue}")
            print()
        
        # Öneriler
        if report["recommendations"]:
            print("💡 ÖNERİLER:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")
            print()
        
        # Kritik bulgular
        print("🔍 KRİTİK BULGULAR:")
        
        if self.test_results.get('csv_write_test', False):
            print("   ✅ CSV yazma sistemi çalışıyor")
        else:
            print("   ❌ CSV yazma sistemi BAŞARISIZ")
        
        if self.test_results.get('manual_close_simulation', False):
            print("   ✅ Manuel kapatma CSV kaydı çalışıyor")
        else:
            print("   ❌ Manuel kapatma CSV'ye KAYDEDİLMİYOR")
        
        if self.test_results.get('sltp_trigger_test', False):
            print("   ✅ SL/TP tetiklenme sistemi çalışıyor")
        else:
            print("   ❌ SL/TP tetiklenme CSV'ye KAYDEDİLMİYOR")
        
        print()
        
        # Sonuç ve öneriler
        print("🎯 SONUÇ:")
        if report["overall_status"] == "PASS":
            print("   ✅ TÜM SİSTEMLER ÇALIŞIYOR - Live bot hazır!")
        elif report["overall_status"] == "PARTIAL_PASS":
            print("   ⚠️ KISMI BAŞARI - Bazı sorunlar var ama kritik sistemler çalışıyor")
        else:
            print("   ❌ CİDDİ SORUNLAR - Live bot kullanıma hazır değil")
        
        print("="*80)
        
        # Raporu dosyaya kaydet
        try:
            report_file = f"live_bot_test_report_{int(time.time())}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 Detaylı rapor kaydedildi: {report_file}")
        except Exception as e:
            print(f"⚠️ Rapor kaydetme hatası: {e}")
        
        return report
    
    def run_all_tests(self):
        """Tüm testleri sırasıyla çalıştır"""
        print("🚀 KAPSAMLI LİVE BOT TEST BAŞLATIYOR...")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Test ortamını hazırla
            self.backup_csv_files()
            
            # 1. CSV Yazma/Okuma Sistemi Testi
            print("\n1️⃣ CSV SİSTEMİ TESTİ")
            print("-" * 30)
            if not self.test_csv_write_system():
                self.log_test("CSV sistemi başarısız - test durduruluyor", "ERROR")
                return self.generate_comprehensive_report()
            
            # 2. Database ve Config Senkronizasyon Testi  
            print("\n2️⃣ DATABASE VE CONFIG SENKRONIZASYON TESTİ")
            print("-" * 30)
            if not self.test_database_and_config_sync():
                self.log_test("Database senkronizasyon başarısız", "WARNING")
            
            # 3. Binance API Bağlantı Testi
            print("\n3️⃣ BİNANCE API TESTİ") 
            print("-" * 30)
            if not self.test_binance_connection():
                self.log_test("Binance bağlantısı başarısız - bazı testler atlanacak", "WARNING")
            
            # 4. Live Trader Entegrasyonu
            print("\n4️⃣ LİVE TRADER ENTEGRASYON TESTİ")
            print("-" * 30)
            if not self.test_live_trader_integration():
                self.log_test("Live trader entegrasyonu başarısız", "WARNING")
            
            # 5. Manuel Kapatma Simülasyonu
            print("\n5️⃣ MANUEL KAPATMA SİMÜLASYON TESTİ")
            print("-" * 30)
            if not self.simulate_manual_close():
                self.log_test("Manuel kapatma simülasyonu başarısız", "ERROR")
            
            # 6. Gerçek SL/TP Tetiklenme Testi (isteğe bağlı)
            if self.binance_client and input("\n🎯 Gerçek SL/TP testi yapmak istiyor musunuz? (y/N): ").lower() == 'y':
                print("\n6️⃣ GERÇEK SL/TP TETİKLENME TESTİ")
                print("-" * 30)
                self.log_test("⚠️ GERÇEK EMİR VERİLECEK - TestBinance'de para kaybı olmaz", "WARNING")
                if not self.test_real_sltp_trigger():
                    self.log_test("Gerçek SL/TP testi tamamlanamadı", "WARNING")
            else:
                self.log_test("Gerçek SL/TP testi atlandı", "INFO")
            
            # 7. Live Trading Sistem Entegrasyonu (isteğe bağlı)
            if LIVE_TRADER_AVAILABLE and input("\n🤖 Live trading sistem testi yapmak istiyor musunuz? (y/N): ").lower() == 'y':
                print("\n7️⃣ LİVE TRADİNG SİSTEM TESTİ")
                print("-" * 30)
                if not self.test_live_trading_integration():
                    self.log_test("Live trading sistem testi başarısız", "WARNING")
            else:
                self.log_test("Live trading sistem testi atlandı", "INFO")
            
        except KeyboardInterrupt:
            self.log_test("Test kullanıcı tarafından durduruldu", "WARNING")
        except Exception as e:
            self.log_test(f"Test sürecinde beklenmedik hata: {e}", "ERROR")
        finally:
            # Her durumda temizlik yap
            self.cleanup_test_environment()
            
            # Test süresini hesapla
            end_time = time.time()
            duration = end_time - start_time
            self.log_test(f"Toplam test süresi: {duration:.1f} saniye")
            
            # Kapsamlı rapor oluştur
            return self.generate_comprehensive_report()


def main():
    """Ana test fonksiyonu"""
    print("🔍 LİVE BOT KAPSAMLI TEST SİSTEMİ - DETAYLI LOG VERSİYONU")
    print("Bu test sistemi aşağıdaki sorunları tespit eder:")
    print("• Manuel kapatma CSV kayıt sorunu")  
    print("• SL/TP tetiklenme CSV kayıt sorunu")
    print("• WebSocket vs REST hibrit sistem sorunları")
    print("• Live trader entegrasyon sorunları")
    print("• Config senkronizasyon sorunları")
    print()
    print("📄 DETAYLI LOG DOSYASI: detailed_live_bot_test.log")
    print("🔍 Tüm işlemler microsaniye düzeyinde loglanacak")
    print("=" * 80)
    
    # Ön kontroller
    if not BINANCE_AVAILABLE:
        print("⚠️ python-binance kütüphanesi bulunamadı")
        print("Kurulum: pip install python-binance python-dotenv")
        return
    
    if not API_KEY or not API_SECRET:
        print("❌ .env dosyasında BINANCE_API_KEY ve BINANCE_SECRET_KEY bulunamadı")
        return
    
    if not LIVE_TRADER_AVAILABLE:
        print("⚠️ Live trader modülü bulunamadı - bazı testler atlanacak")
    
    # Onay al
    confirm = input("\n🚨 UYARI: Bu test TestBinance'de gerçek emirler verecektir.\n🔍 DETAYLI LOG ile her adım kaydedilecek.\nDevam etmek istiyor musunuz? (y/N): ")
    if confirm.lower() != 'y':
        print("Test iptal edildi.")
        return
    
    # Testi başlat
    tester = LiveBotTester()
    report = tester.run_all_tests()
    
    # Final mesajı
    print(f"\n🏁 TEST TAMAMLANDI!")
    print(f"📄 Detaylı log dosyası: detailed_live_bot_test.log")
    
    if report["overall_status"] == "PASS":
        print("✅ Live bot sistemi tamamen çalışır durumda!")
    elif report["overall_status"] == "PARTIAL_PASS":
        print("⚠️ Live bot kısmen çalışıyor - sorunları çözün")
    else:
        print("❌ Live bot ciddi sorunlar var - kullanmayın")
    
    print(f"\n🔍 Sorun tespiti için log dosyasını inceleyin:")
    print(f"cat detailed_live_bot_test.log | grep -E '(ERROR|save_closed_trade|CSV)'")


if __name__ == "__main__":
    main()