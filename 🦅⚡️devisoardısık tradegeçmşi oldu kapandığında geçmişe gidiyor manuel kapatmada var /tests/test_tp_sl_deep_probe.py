# === tests/test_websocket_live_monitor.py ========================================
# 🌐 Real-time WebSocket Crypto Analytics System
# BTC, ETH, BNB için canlı fiyat + tüm indikatörler (Deviso, C-Signal, Log Volume)
# Binance Futures WebSocket kullanarak gerçek zamanlı analiz

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import time
import math
import threading
import argparse
import pandas as pd
import websocket
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

from config import LOCAL_TZ, ENVIRONMENT
from data.fetch_data import fetch_klines, get_current_price
from core.indicators import (
    calculate_c_signal_momentum, 
    compute_consecutive_metrics,
    calculate_deviso_ratio,
    get_deviso_detailed_analysis
)
from core.ai_model import ai_model


class WebSocketLiveMonitor:
    """🌐 Real-time WebSocket Crypto Analytics Monitor"""
    
    def __init__(self, symbols: List[str], interval: str = "15m"):
        self.symbols = symbols
        self.interval = interval
        self.ws = None
        self.is_running = False
        
        # Real-time data storage
        self.live_prices = {}
        self.price_history = {symbol: deque(maxlen=100) for symbol in symbols}
        self.last_kline_data = {}
        self.indicators_cache = {}
        
        # WebSocket configuration
        self.ws_base_url = self._get_ws_url()
        self.stream_names = [f"{symbol.lower()}@ticker" for symbol in symbols]
        
        # Statistics
        self.update_count = 0
        self.last_update_time = None
        self.error_count = 0
        
        print(f"🌐 WebSocket Live Monitor başlatıldı")
        print(f"🔗 WebSocket URL: {self.ws_base_url}")
        print(f"📊 Semboller: {', '.join(symbols)}")
        print(f"⏰ Zaman aralığı: {interval}")
        print(f"🔄 Real-time price updates + Full indicators")
        print("="*80)
    
    def _get_ws_url(self) -> str:
        """Environment'a göre WebSocket URL'i belirle"""
        if ENVIRONMENT == 'testnet':
            return "wss://fstream.binancefuture.com/ws/"
        else:
            return "wss://fstream.binance.com/ws/"
    
    def _get_stream_url(self) -> str:
        """Multi-stream WebSocket URL oluştur"""
        streams = "/".join(self.stream_names)
        return f"{self.ws_base_url}{streams}"
    
    def on_message(self, ws, message):
        """WebSocket mesaj işleyici"""
        try:
            data = json.loads(message)
            
            # Multi-stream response
            if 'stream' in data and 'data' in data:
                stream = data['stream']
                ticker_data = data['data']
            else:
                # Single stream response
                ticker_data = data
                stream = None
            
            # Symbol'ü belirle
            symbol = ticker_data.get('s', '').upper()
            if symbol not in self.symbols:
                return
            
            # Fiyat verilerini güncelle
            current_price = float(ticker_data.get('c', 0))  # Current price
            if current_price > 0:
                self.live_prices[symbol] = {
                    'symbol': symbol,
                    'price': current_price,
                    'change_24h': float(ticker_data.get('P', 0)),  # 24h price change percent
                    'volume': float(ticker_data.get('v', 0)),      # Volume
                    'high_24h': float(ticker_data.get('h', 0)),    # 24h high
                    'low_24h': float(ticker_data.get('l', 0)),     # 24h low
                    'timestamp': datetime.now(LOCAL_TZ)
                }
                
                # Fiyat geçmişine ekle
                self.price_history[symbol].append({
                    'price': current_price,
                    'timestamp': datetime.now(LOCAL_TZ)
                })
                
                self.update_count += 1
                self.last_update_time = datetime.now(LOCAL_TZ)
                
                # İndikatörleri güncelle (her fiyat güncellemesinde değil, daha optimize)
                if self.update_count % 5 == 0:  # Her 5 güncellemede bir
                    self._update_indicators(symbol)
                    
        except Exception as e:
            self.error_count += 1
            print(f"❌ WebSocket mesaj hatası: {e}")
    
    def on_error(self, ws, error):
        """WebSocket hata işleyici"""
        self.error_count += 1
        print(f"❌ WebSocket hatası: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket kapatma işleyici"""
        self.is_running = False
        print(f"🔴 WebSocket bağlantısı kapandı: {close_status_code} - {close_msg}")
    
    def on_open(self, ws):
        """WebSocket açılma işleyici"""
        self.is_running = True
        print(f"🟢 WebSocket bağlantısı açıldı")
        print(f"📡 Stream'ler: {', '.join(self.stream_names)}")
        
        # İlk kline verilerini yükle
        self._load_initial_kline_data()
    
    def _load_initial_kline_data(self):
        """Başlangıçta tüm semboller için kline verilerini yükle"""
        print(f"📊 İlk kline verileri yükleniyor...")
        
        for symbol in self.symbols:
            try:
                df = fetch_klines(symbol, self.interval)
                if df is not None and not df.empty and len(df) >= 50:
                    self.last_kline_data[symbol] = df
                    self._update_indicators(symbol)
                    print(f"✅ {symbol}: {len(df)} kline yüklendi")
                else:
                    print(f"⚠️ {symbol}: Yetersiz kline verisi")
            except Exception as e:
                print(f"❌ {symbol} kline yükleme hatası: {e}")
        
        print(f"📊 İlk veri yükleme tamamlandı")
    
    def _update_indicators(self, symbol: str):
        """Sembol için tüm indikatörleri güncelle"""
        try:
            # Kline verisi kontrolü
            if symbol not in self.last_kline_data:
                return
            
            df = self.last_kline_data[symbol].copy()
            if df is None or df.empty or len(df) < 50:
                return
            
            # Real-time fiyat ile son kline'ı güncelle
            if symbol in self.live_prices:
                current_price = self.live_prices[symbol]['price']
                # Son kline'ın close price'ını güncelle (simulated)
                df.iloc[-1, df.columns.get_loc('close')] = current_price
            
            # 1. Consecutive Metrics (Run count, Deviso, Log Volume, C-Signal)
            metrics = compute_consecutive_metrics(df)
            
            # 2. Deviso detaylı analizi
            deviso_details = get_deviso_detailed_analysis(df)
            
            # 3. AI Score hesaplama
            ai_score = ai_model.predict_score(metrics) if metrics else 0.0
            
            # 4. Cache'e kaydet
            self.indicators_cache[symbol] = {
                'symbol': symbol,
                'timestamp': datetime.now(LOCAL_TZ),
                'current_price': self.live_prices.get(symbol, {}).get('price', 0),
                
                # Consecutive Metrics
                'run_type': metrics.get('run_type', 'none'),
                'run_count': metrics.get('run_count', 0),
                'run_perc': metrics.get('run_perc', 0),
                'gauss_run': metrics.get('gauss_run', 0),
                'gauss_run_perc': metrics.get('gauss_run_perc', 0),
                
                # Log Volume
                'log_volume': metrics.get('log_volume', 0),
                'log_volume_momentum': metrics.get('log_volume_momentum', 0),
                
                # Deviso
                'deviso_ratio': metrics.get('deviso_ratio', 0),
                'trend_direction': deviso_details.get('trend_direction', 'Belirsiz'),
                'trend_strength': abs(metrics.get('deviso_ratio', 0)),
                
                # C-Signal
                'c_signal_momentum': metrics.get('c_signal_momentum', 0),
                
                # AI Score
                'ai_score': ai_score,
                
                # Technical levels
                'ma_all_signals': deviso_details.get('ma_all_signals', 0),
                'ma_all_signals2': deviso_details.get('ma_all_signals2', 0),
                'long_signal': deviso_details.get('long_signal', False),
                'short_signal': deviso_details.get('short_signal', False),
                
                # Quality metrics
                'data_length': len(df),
                'last_update': datetime.now(LOCAL_TZ).strftime('%H:%M:%S')
            }
            
        except Exception as e:
            print(f"❌ {symbol} indikatör güncelleme hatası: {e}")
    
    def get_live_analysis(self, symbol: str) -> Dict:
        """Sembol için canlı analiz verilerini al"""
        live_price_data = self.live_prices.get(symbol, {})
        indicator_data = self.indicators_cache.get(symbol, {})
        
        if not live_price_data and not indicator_data:
            return {'symbol': symbol, 'error': 'No data available'}
        
        return {
            **live_price_data,
            **indicator_data,
            'ws_updates': self.update_count,
            'ws_errors': self.error_count,
        }
    
    def print_live_dashboard(self):
        """Real-time dashboard yazdır"""
        current_time = datetime.now(LOCAL_TZ).strftime('%H:%M:%S')
        
        print(f"\n{'='*20} 🌐 LIVE DASHBOARD {current_time} {'='*20}")
        print(f"📡 WebSocket: {'🟢 Connected' if self.is_running else '🔴 Disconnected'}")
        print(f"📊 Updates: {self.update_count} | Errors: {self.error_count}")
        
        if self.last_update_time:
            seconds_ago = (datetime.now(LOCAL_TZ) - self.last_update_time).total_seconds()
            print(f"⏰ Last update: {seconds_ago:.1f}s ago")
        
        print("\n" + "="*100)
        print(f"{'Symbol':<10} {'Price':<12} {'24h %':<8} {'AI':<6} {'Run':<6} {'Deviso':<10} {'C-Sig':<8} {'LogVol':<8} {'Trend':<12}")
        print("="*100)
        
        for symbol in self.symbols:
            analysis = self.get_live_analysis(symbol)
            
            if analysis.get('error'):
                print(f"{symbol:<10} {'ERROR':<12} {'N/A':<8} {'N/A':<6} {'N/A':<6} {'N/A':<10} {'N/A':<8} {'N/A':<8} {'N/A':<12}")
                continue
            
            # Format values
            price = analysis.get('price', 0)
            change_24h = analysis.get('change_24h', 0)
            ai_score = analysis.get('ai_score', 0)
            run_count = analysis.get('run_count', 0)
            deviso = analysis.get('deviso_ratio', 0)
            c_signal = analysis.get('c_signal_momentum', 0)
            log_vol = analysis.get('log_volume', 0)
            trend = analysis.get('trend_direction', 'N/A')[:10]
            
            # Color indicators
            price_color = "🟢" if change_24h > 0 else "🔴" if change_24h < 0 else "⚪"
            ai_color = "🟢" if ai_score >= 70 else "🟡" if ai_score >= 50 else "🔴"
            deviso_color = "🟢" if deviso > 1 else "🔴" if deviso < -1 else "⚪"
            
            print(f"{symbol:<10} {price:<12.4f} {change_24h:+6.2f}% {ai_color}{ai_score:<4.0f} "
                  f"{run_count:<6} {deviso_color}{deviso:<8.2f} {c_signal:<+7.2f} {log_vol:<8.2f} {trend:<12}")
        
        print("="*100)
    
    def print_detailed_analysis(self, symbol: str):
        """Detaylı sembol analizi yazdır"""
        analysis = self.get_live_analysis(symbol)
        
        if analysis.get('error'):
            print(f"❌ {symbol}: {analysis['error']}")
            return
        
        print(f"\n🔍 {symbol} - DETAYLI ANALİZ")
        print("="*60)
        
        # Price data
        print(f"💰 Fiyat Bilgileri:")
        print(f"   Current Price: ${analysis.get('price', 0):.6f}")
        print(f"   24h Change: {analysis.get('change_24h', 0):+.2f}%")
        print(f"   24h High: ${analysis.get('high_24h', 0):.6f}")
        print(f"   24h Low: ${analysis.get('low_24h', 0):.6f}")
        print(f"   Volume: {analysis.get('volume', 0):,.0f}")
        
        # Technical indicators
        print(f"\n📊 Teknik İndikatörler:")
        print(f"   Run Type: {analysis.get('run_type', 'none').upper()}")
        print(f"   Run Count: {analysis.get('run_count', 0)}")
        print(f"   Run Percentage: {analysis.get('run_perc', 0):.2f}%")
        print(f"   Gauss Run: {analysis.get('gauss_run', 0):.1f}")
        
        # Log Volume
        print(f"\n📈 Log Volume Analizi:")
        print(f"   Log Volume: {analysis.get('log_volume', 0):.4f}")
        print(f"   Log Volume Momentum: {analysis.get('log_volume_momentum', 0):+.4f}")
        
        # Deviso
        print(f"\n🎯 Deviso Analizi:")
        print(f"   Deviso Ratio: {analysis.get('deviso_ratio', 0):+.4f}")
        print(f"   Trend Direction: {analysis.get('trend_direction', 'N/A')}")
        print(f"   Trend Strength: {analysis.get('trend_strength', 0):.4f}")
        
        # C-Signal
        print(f"\n🔄 C-Signal:")
        print(f"   C-Signal Momentum: {analysis.get('c_signal_momentum', 0):+.4f}")
        
        # AI Score
        print(f"\n🤖 AI Değerlendirme:")
        print(f"   AI Score: {analysis.get('ai_score', 0):.1f}%")
        
        # Signal quality
        ai_score = analysis.get('ai_score', 0)
        if ai_score >= 80:
            quality = "🟢 Mükemmel"
        elif ai_score >= 70:
            quality = "🟡 İyi"
        elif ai_score >= 50:
            quality = "🟠 Orta"
        else:
            quality = "🔴 Zayıf"
        
        print(f"   Signal Quality: {quality}")
        
        # Deviso signals
        long_signal = analysis.get('long_signal', False)
        short_signal = analysis.get('short_signal', False)
        
        if long_signal:
            print(f"   🚀 Deviso LONG Signal Active")
        elif short_signal:
            print(f"   📉 Deviso SHORT Signal Active")
        else:
            print(f"   ➖ No Deviso Signal")
        
        print(f"\n⏰ Last Update: {analysis.get('last_update', 'N/A')}")
        print("="*60)
    
    def start_websocket(self):
        """WebSocket bağlantısını başlat"""
        try:
            stream_url = self._get_stream_url()
            print(f"🔗 WebSocket bağlanıyor: {stream_url}")
            
            self.ws = websocket.WebSocketApp(
                stream_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # WebSocket'i ayrı thread'de çalıştır
            self.ws.run_forever()
            
        except Exception as e:
            print(f"❌ WebSocket başlatma hatası: {e}")
    
    def stop_websocket(self):
        """WebSocket bağlantısını durdur"""
        self.is_running = False
        if self.ws:
            self.ws.close()
        print(f"🛑 WebSocket durduruldu")
    
    def run_live_monitor(self, duration_minutes: int = None, dashboard_interval: int = 5):
        """Live monitoring başlat"""
        print(f"🚀 Live Monitoring başlatılıyor...")
        
        if duration_minutes:
            print(f"⏰ {duration_minutes} dakika boyunca çalışacak")
        else:
            print(f"⏰ CTRL+C ile durdurun")
        
        print(f"📊 Dashboard her {dashboard_interval} saniyede güncellenecek")
        
        # WebSocket'i ayrı thread'de başlat
        ws_thread = threading.Thread(target=self.start_websocket, daemon=True)
        ws_thread.start()
        
        # WebSocket bağlantısının kurulmasını bekle
        time.sleep(3)
        
        start_time = time.time()
        iteration = 0
        
        try:
            while self.is_running:
                iteration += 1
                
                # Dashboard yazdır
                self.print_live_dashboard()
                
                # Her 20 iterasyonda detaylı analiz
                if iteration % 4 == 0:  # Her 20 saniyede bir
                    print(f"\n🔍 DETAYLI ANALİZ:")
                    for symbol in self.symbols:
                        if symbol in self.indicators_cache:
                            self.print_detailed_analysis(symbol)
                
                # Süre kontrolü
                if duration_minutes:
                    elapsed_minutes = (time.time() - start_time) / 60
                    if elapsed_minutes >= duration_minutes:
                        print(f"\n✅ {duration_minutes} dakika tamamlandı. Monitor sonlandırıldı.")
                        break
                
                # Bekle
                time.sleep(dashboard_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Kullanıcı tarafından durduruldu.")
            print(f"📊 Toplam {iteration} iterasyon, {self.update_count} WebSocket güncellemesi")
        finally:
            self.stop_websocket()


def main():
    parser = argparse.ArgumentParser(description="WebSocket Live Crypto Monitor")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,BNBUSDT", 
                       help="İzlenecek semboller (virgül ile ayırın)")
    parser.add_argument("--interval", type=str, default="15m", 
                       help="Kline zaman aralığı (1m, 5m, 15m, 1h)")
    parser.add_argument("--duration", type=int, default=None, 
                       help="Monitor süresi (dakika). Belirtilmezse süresiz çalışır")
    parser.add_argument("--dashboard", type=int, default=5, 
                       help="Dashboard güncelleme aralığı (saniye)")
    parser.add_argument("--test", action="store_true", 
                       help="Tek seferlik test modu")
    
    args = parser.parse_args()
    
    # Sembolleri parse et
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    
    if not symbols:
        print("❌ Geçerli sembol bulunamadı!")
        return
    
    print(f"🌐 WebSocket Live Monitor")
    print(f"🔗 Environment: {ENVIRONMENT}")
    print(f"📊 Semboller: {', '.join(symbols)}")
    print("="*50)
    
    # Monitor oluştur
    monitor = WebSocketLiveMonitor(symbols, args.interval)
    
    if args.test:
        # Test modu - sadece bağlantı testi
        print("🔬 WebSocket bağlantı testi yapılıyor...")
        
        def test_connection():
            monitor.start_websocket()
        
        ws_thread = threading.Thread(target=test_connection, daemon=True)
        ws_thread.start()
        
        # 10 saniye bekle
        time.sleep(10)
        
        if monitor.is_running and monitor.update_count > 0:
            print(f"✅ Test başarılı! {monitor.update_count} güncelleme alındı")
            monitor.print_live_dashboard()
        else:
            print(f"❌ Test başarısız! WebSocket bağlantısı kurulamadı")
        
        monitor.stop_websocket()
    else:
        # Normal live monitoring
        monitor.run_live_monitor(args.duration, args.dashboard)


if __name__ == "__main__":
    print("🌐 WebSocket Live Crypto Monitor")
    print("Real-time Deviso + C-Signal + Log Volume + AI Analysis")
    print("="*60)
    main()