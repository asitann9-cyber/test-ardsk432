# === tests/test_c_signal_monitor.py ==========================================
# C-Signal Momentum Test ve İzleme Sistemi
# BTC ve diğer coinlerin C-Signal değerlerini canlı takip eder

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import time
import math  # 🔥 EKLENDİ - C-Signal hesaplaması için gerekli
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List

from config import LOCAL_TZ
from data.fetch_data import fetch_klines, get_usdt_perp_symbols
from core.indicators import calculate_c_signal_momentum, compute_consecutive_metrics


class CSignalMonitor:
    """C-Signal Momentum İzleme ve Test Sistemi"""
    
    def __init__(self, symbols: List[str], interval: str = "15m"):
        self.symbols = symbols
        self.interval = interval
        self.history = {}
        print(f"🤖 C-Signal Monitor başlatıldı")
        print(f"📊 Semboller: {', '.join(symbols)}")
        print(f"⏰ Zaman aralığı: {interval}")
        print(f"🔄 Her 1 saniyede güncellenir")
        print("="*70)
    
    def calculate_c_signal_detailed(self, df: pd.DataFrame) -> Dict:
        """C-Signal hesaplamalarını detaylı şekilde yap ve ara adımları göster"""
        try:
            if df is None or df.empty or len(df) < 15:
                return {'error': 'Insufficient data'}
            
            # 1) Log Close hesaplama
            log_close = df['close'].apply(lambda x: math.log(x) if x > 0 else float('nan'))
            
            # 2) RSI hesaplama - TradingView ta.rsi ile tam uyumlu
            delta = log_close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Wilder's smoothing (alpha = 1/14)
            avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            
            # RSI calculation
            rs = avg_gain / avg_loss
            rsi_log_close = 100 - (100 / (1 + rs))
            
            # 3) RSI Change (C-Signal)
            rsi_change = rsi_log_close.diff()
            
            # Son değerler
            last_close = float(df.iloc[-1]['close'])
            last_log_close = float(log_close.iloc[-1]) if not pd.isna(log_close.iloc[-1]) else None
            last_rsi = float(rsi_log_close.iloc[-1]) if not pd.isna(rsi_log_close.iloc[-1]) else None
            prev_rsi = float(rsi_log_close.iloc[-2]) if len(rsi_log_close) > 1 and not pd.isna(rsi_log_close.iloc[-2]) else None
            c_signal = float(rsi_change.iloc[-1]) if not pd.isna(rsi_change.iloc[-1]) else None
            
            return {
                'last_close': last_close,
                'last_log_close': last_log_close,
                'last_rsi': last_rsi,
                'prev_rsi': prev_rsi,
                'c_signal': c_signal,
                'candle_time': df.iloc[-1]['close_time'],
                'data_length': len(df),
                'error': None
            }
            
        except Exception as e:
            return {'error': str(e)}

    def get_c_signal_for_symbol(self, symbol: str) -> Dict:
        """Bir sembol için C-Signal verilerini al - gelişmiş debug bilgileriyle"""
        try:
            # OHLCV verilerini çek
            df = fetch_klines(symbol, self.interval)
            if df is None or df.empty or len(df) < 20:
                return {
                    'symbol': symbol,
                    'c_signal': None,
                    'error': 'Insufficient data',
                    'timestamp': datetime.now(LOCAL_TZ)
                }
            
            # Detaylı C-Signal hesaplama
            c_signal_data = self.calculate_c_signal_detailed(df)
            
            if c_signal_data.get('error'):
                return {
                    'symbol': symbol,
                    'c_signal': None,
                    'error': c_signal_data['error'],
                    'timestamp': datetime.now(LOCAL_TZ)
                }
            
            # Ek metrikler de al (karşılaştırma için)
            metrics = compute_consecutive_metrics(df)
            
            result = {
                'symbol': symbol,
                'c_signal': c_signal_data['c_signal'],
                'last_close': c_signal_data['last_close'],
                'last_log_close': c_signal_data['last_log_close'],
                'last_rsi': c_signal_data['last_rsi'],
                'prev_rsi': c_signal_data['prev_rsi'],
                'candle_time': c_signal_data['candle_time'],
                'data_length': c_signal_data['data_length'],
                'run_type': metrics.get('run_type', 'none'),
                'run_count': metrics.get('run_count', 0),
                'deviso_ratio': metrics.get('deviso_ratio', 0),
                'timestamp': datetime.now(LOCAL_TZ),
                'error': None
            }
            
            return result
            
        except Exception as e:
            return {
                'symbol': symbol,
                'c_signal': None,
                'error': str(e),
                'timestamp': datetime.now(LOCAL_TZ)
            }
    
    def update_all_signals(self) -> Dict:
        """Tüm semboller için C-Signal'leri güncelle"""
        results = {}
        
        for symbol in self.symbols:
            result = self.get_c_signal_for_symbol(symbol)
            results[symbol] = result
            
            # Geçmişe ekle (son 10 değeri sakla)
            if symbol not in self.history:
                self.history[symbol] = []
            
            self.history[symbol].append({
                'timestamp': result['timestamp'],
                'c_signal': result['c_signal'],
                'last_price': result.get('last_price', 0)
            })
            
            # Son 10 değeri tut
            if len(self.history[symbol]) > 10:
                self.history[symbol] = self.history[symbol][-10:]
        
        return results
    
    def print_detailed_debug(self, results: Dict):
        """Detaylı debug bilgileri yazdır - TradingView ile karşılaştırma için"""
        current_time = datetime.now(LOCAL_TZ).strftime('%H:%M:%S')
        print(f"\n🔍 {current_time} - DETAYLI C-SIGNAL DEBUG")
        print("=" * 100)
        
        for symbol, data in results.items():
            if data['error']:
                print(f"❌ {symbol}: {data['error']}")
                continue
            
            print(f"\n📊 {symbol} - Detaylı Analiz:")
            print(f"   Son Fiyat: {data['last_close']:.6f}")
            print(f"   Log(Close): {data['last_log_close']:.6f}")
            print(f"   Son RSI: {data['last_rsi']:.4f}")
            print(f"   Önceki RSI: {data['prev_rsi']:.4f}")
            print(f"   C-Signal: {data['c_signal']:+7.4f}")
            print(f"   Mum Zamanı: {data['candle_time']}")
            print(f"   Veri Uzunluğu: {data['data_length']} mum")
            
            # C-Signal seviye analizi
            c_sig = data['c_signal']
            if c_sig is not None:
                if c_sig >= 15:
                    level = "🚀 C20+ Seviyesi (Çok Güçlü Al)"
                elif c_sig >= 10:
                    level = "🟢 C15 Seviyesi (Güçlü Al)"
                elif c_sig >= 8:
                    level = "📈 C10 Seviyesi (Al Sinyali)"
                elif c_sig <= -15:
                    level = "📉 C-20 Seviyesi (Çok Güçlü Sat)"
                elif c_sig <= -10:
                    level = "🔴 C-15 Seviyesi (Güçlü Sat)"
                elif c_sig <= -8:
                    level = "🔻 C-10 Seviyesi (Sat Sinyali)"
                else:
                    level = "➖ Nötr Seviye"
                
                print(f"   Seviye: {level}")
            
            print("-" * 60)

    def print_results(self, results: Dict):
        """Sonuçları konsola yazdır"""
        current_time = datetime.now(LOCAL_TZ).strftime('%H:%M:%S')
        print(f"\n🕐 {current_time} - C-Signal Güncellemesi")
        print("-" * 90)
        
        # Header
        print(f"{'Symbol':<12} {'C-Signal':<12} {'Fiyat':<12} {'RSI':<8} {'Mum Zamanı':<20} {'Durum':<15}")
        print("-" * 90)
        
        for symbol, data in results.items():
            if data['error']:
                print(f"{symbol:<12} {'ERROR':<12} {'':<12} {'':<8} {'':<20} {data['error']:<15}")
                continue
            
            c_signal = data['c_signal']
            last_close = data['last_close']
            last_rsi = data['last_rsi']
            candle_time = data['candle_time'].strftime('%H:%M:%S') if data['candle_time'] else 'N/A'
            
            # C-Signal durumu belirle
            if c_signal is None:
                status = "❓ N/A"
                c_signal_str = "N/A"
            elif c_signal > 10:
                status = "🚀 Güçlü Al"
                c_signal_str = f"{c_signal:+7.3f}"
            elif c_signal > 8:
                status = "📈 Al Sinyali"
                c_signal_str = f"{c_signal:+7.3f}"
            elif c_signal < -10:
                status = "📉 Güçlü Sat"
                c_signal_str = f"{c_signal:+7.3f}"
            elif c_signal < -8:
                status = "🔻 Sat Sinyali"
                c_signal_str = f"{c_signal:+7.3f}"
            else:
                status = "➖ Nötr"
                c_signal_str = f"{c_signal:+7.3f}"
            
            print(f"{symbol:<12} {c_signal_str:<12} {last_close:<12.2f} {last_rsi:<8.2f} "
                  f"{candle_time:<20} {status:<15}")
        
        print("-" * 90)
    
    def print_history_summary(self):
        """Geçmiş özeti yazdır"""
        print(f"\n📊 Son 5 C-Signal Değeri (Geçmiş):")
        print("-" * 60)
        
        for symbol in self.symbols:
            if symbol in self.history and len(self.history[symbol]) > 0:
                print(f"\n{symbol}:")
                recent_values = self.history[symbol][-5:]  # Son 5 değer
                
                for i, entry in enumerate(recent_values):
                    time_str = entry['timestamp'].strftime('%H:%M:%S')
                    c_sig = entry['c_signal']
                    if c_sig is not None:
                        print(f"  {time_str}: {c_sig:+7.3f}")
                    else:
                        print(f"  {time_str}: N/A")
    
    def run_continuous_monitor(self, duration_minutes: int = None, debug_mode: bool = False):
        """Sürekli izleme modu"""
        print(f"🔄 Sürekli izleme başlatıldı...")
        if duration_minutes:
            print(f"⏰ {duration_minutes} dakika boyunca çalışacak")
        else:
            print(f"⏰ CTRL+C ile durdurun")
        
        if debug_mode:
            print(f"🔍 Debug modu aktif - detaylı bilgiler gösterilecek")
        
        start_time = time.time()
        iteration = 0
        
        try:
            while True:
                iteration += 1
                print(f"\n{'='*20} İterasyon #{iteration} {'='*20}")
                
                # Tüm sinyalleri güncelle
                results = self.update_all_signals()
                
                # Sonuçları yazdır
                if debug_mode:
                    self.print_detailed_debug(results)
                else:
                    self.print_results(results)
                
                # Her 10 iterasyonda geçmiş özeti göster
                if iteration % 10 == 0:
                    self.print_history_summary()
                
                # Süre kontrolü
                if duration_minutes:
                    elapsed_minutes = (time.time() - start_time) / 60
                    if elapsed_minutes >= duration_minutes:
                        print(f"\n✅ {duration_minutes} dakika tamamlandı. Test sonlandırıldı.")
                        break
                
                # 1 saniye bekle
                time.sleep(1)
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Kullanıcı tarafından durduruldu.")
            print(f"📊 Toplam {iteration} iterasyon çalıştırıldı.")
            self.print_history_summary()


def main():
    parser = argparse.ArgumentParser(description="C-Signal Momentum Test ve İzleme")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,BNBUSDT", 
                       help="İzlenecek semboller (virgül ile ayırın)")
    parser.add_argument("--interval", type=str, default="15m", 
                       help="Zaman aralığı (1m, 5m, 15m, 1h)")
    parser.add_argument("--duration", type=int, default=None, 
                       help="Test süresi (dakika). Belirtilmezse süresiz çalışır")
    parser.add_argument("--single", action="store_true", 
                       help="Tek seferlik test (sürekli izleme yapma)")
    parser.add_argument("--debug", action="store_true", 
                       help="Debug modu - detaylı RSI hesaplama bilgileri göster")
    
    args = parser.parse_args()
    
    # Sembolleri parse et
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    
    # Geçerli sembolleri kontrol et (opsiyonel)
    print("🔍 Sembol geçerliliği kontrol ediliyor...")
    all_symbols = get_usdt_perp_symbols()
    if all_symbols:
        valid_symbols = [s for s in symbols if s in all_symbols]
        invalid_symbols = [s for s in symbols if s not in all_symbols]
        
        if invalid_symbols:
            print(f"⚠️ Geçersiz semboller: {', '.join(invalid_symbols)}")
        
        if valid_symbols:
            symbols = valid_symbols
            print(f"✅ Geçerli semboller: {', '.join(symbols)}")
        else:
            print("❌ Hiçbir geçerli sembol bulunamadı!")
            return
    else:
        print("⚠️ Sembol listesi alınamadı, verilen sembollerle devam ediliyor...")
    
    # Monitor oluştur
    monitor = CSignalMonitor(symbols, args.interval)
    
    if args.single:
        # Tek seferlik test
        print("🔬 Tek seferlik C-Signal testi yapılıyor...")
        results = monitor.update_all_signals()
        
        if args.debug:
            monitor.print_detailed_debug(results)
        else:
            monitor.print_results(results)
        
        print(f"\n📋 Test Özeti:")
        print(f"   Sembol sayısı: {len(symbols)}")
        print(f"   Zaman aralığı: {args.interval}")
        print(f"   Test zamanı: {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # TradingView karşılaştırma önerisi
        print(f"\n🔍 TradingView Karşılaştırma İpuçları:")
        print(f"   1. TradingView'da aynı sembol ve zaman dilimini açın")
        print(f"   2. C-Signal indikatörünüzü ekleyin")
        print(f"   3. Son mum kapanış zamanını karşılaştırın")
        print(f"   4. RSI değerlerini kontrol edin")
        
    else:
        # Sürekli izleme
        monitor.run_continuous_monitor(args.duration, args.debug)


if __name__ == "__main__":
    print("🤖 C-Signal Momentum Test Sistemi")
    print("=" * 50)
    main()
