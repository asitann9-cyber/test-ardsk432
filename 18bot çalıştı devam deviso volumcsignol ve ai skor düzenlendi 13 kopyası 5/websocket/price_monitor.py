"""
📊 Price Monitor - WebSocket ile Real-time Sinyal İzleme
🎯 Amaç: AI Top 10 sinyalleri + SL/TP pozisyon izleme
⚡ Real-time fiyat değişiklikleri ile dinamik sıralama
"""

import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque

import config
from config import LOCAL_TZ, MAX_OPEN_POSITIONS, STOP_LOSS_PCT, TAKE_PROFIT_PCT
from .ws_manager import (
    start_websocket_streams, get_all_realtime_prices, 
    update_monitored_symbols, get_websocket_status
)

logger = logging.getLogger("crypto-analytics")

# Global WebSocket veri yapıları
websocket_top10_data: pd.DataFrame = pd.DataFrame()
websocket_signals_history: Dict[str, deque] = {}  # Fiyat geçmişi
last_top10_update: Optional[datetime] = None
sl_tp_triggers: List[Dict] = []  # SL/TP tetikleme geçmişi


class PriceMonitor:
    """📊 Real-time Fiyat İzleme ve Sinyal Yönetimi"""
    
    def __init__(self):
        self.price_history: Dict[str, deque] = {}  # {symbol: deque([prices])}
        self.signal_scores: Dict[str, float] = {}  # AI skorları cache
        self.momentum_cache: Dict[str, Dict] = {}  # Momentum hesaplamaları
        self.last_analysis_time: Optional[datetime] = None
        
        # İstatistikler
        self.update_count: int = 0
        self.trigger_count: int = 0
        
    def update_top10_from_ai(self, ai_signals_df: pd.DataFrame) -> bool:
        """AI analizinden top 10 sinyali WebSocket'e gönder"""
        global websocket_top10_data, last_top10_update
        
        try:
            if ai_signals_df.empty:
                logger.warning("⚠️ AI sinyal DataFrame boş")
                return False
            
            # En iyi 10'u seç
            top10_df = ai_signals_df.head(10).copy()
            
            # WebSocket için sembol listesi
            new_symbols = top10_df['symbol'].tolist()
            
            logger.info(f"📡 WebSocket Top 10 güncelleniyor: {', '.join(new_symbols[:5])}{'...' if len(new_symbols) > 5 else ''}")
            
            # AI skorlarını cache'le
            for _, row in top10_df.iterrows():
                self.signal_scores[row['symbol']] = row['ai_score']
            
            # WebSocket sembolleri güncelle
            success = update_monitored_symbols(new_symbols)
            
            if success:
                websocket_top10_data = top10_df
                last_top10_update = datetime.now(LOCAL_TZ)
                logger.info(f"✅ WebSocket Top 10 başarıyla güncellendi")
                
                # Fiyat geçmişini başlat
                for symbol in new_symbols:
                    if symbol not in self.price_history:
                        self.price_history[symbol] = deque(maxlen=60)  # Son 60 güncelleme
                
                return True
            else:
                logger.error("❌ WebSocket sembol güncelleme başarısız")
                return False
                
        except Exception as e:
            logger.error(f"❌ Top 10 AI güncelleme hatası: {e}")
            return False
    
    def calculate_realtime_momentum(self, symbol: str, current_price: float) -> Dict:
        """Real-time momentum hesapla"""
        try:
            # Fiyat geçmişini güncelle
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=60)
            
            self.price_history[symbol].append({
                'price': current_price,
                'timestamp': datetime.now(LOCAL_TZ)
            })
            
            price_history = list(self.price_history[symbol])
            
            if len(price_history) < 5:
                return {'momentum_1m': 0, 'momentum_5m': 0, 'trend_strength': 0}
            
            # 1 dakikalık momentum (son 5 güncelleme)
            if len(price_history) >= 5:
                old_price_1m = price_history[-5]['price']
                momentum_1m = ((current_price - old_price_1m) / old_price_1m) * 100
            else:
                momentum_1m = 0
            
            # 5 dakikalık momentum (son 25 güncelleme)
            if len(price_history) >= 25:
                old_price_5m = price_history[-25]['price']
                momentum_5m = ((current_price - old_price_5m) / old_price_5m) * 100
            else:
                momentum_5m = 0
            
            # Trend gücü (fiyat volatilitesi)
            prices = [p['price'] for p in price_history[-10:]]
            if len(prices) >= 3:
                price_std = pd.Series(prices).std()
                avg_price = sum(prices) / len(prices)
                trend_strength = (price_std / avg_price) * 100 if avg_price > 0 else 0
            else:
                trend_strength = 0
            
            momentum_data = {
                'momentum_1m': momentum_1m,
                'momentum_5m': momentum_5m,
                'trend_strength': trend_strength,
                'price_count': len(price_history)
            }
            
            # Cache'le
            self.momentum_cache[symbol] = momentum_data
            
            return momentum_data
            
        except Exception as e:
            logger.debug(f"Momentum hesaplama hatası {symbol}: {e}")
            return {'momentum_1m': 0, 'momentum_5m': 0, 'trend_strength': 0}
    
    def create_websocket_realtime_table(self) -> pd.DataFrame:
        """Real-time WebSocket tablosu oluştur - AI + Real-time kombine"""
        global websocket_top10_data
        
        try:
            if websocket_top10_data.empty:
                return pd.DataFrame()
            
            # Real-time fiyatları al
            realtime_prices = get_all_realtime_prices()
            
            if not realtime_prices:
                logger.debug("🔄 Real-time fiyat verisi bekleniyor...")
                return websocket_top10_data.copy()
            
            # WebSocket tablosu oluştur
            ws_table = websocket_top10_data.copy()
            
            # Real-time verileri ekle
            for idx, row in ws_table.iterrows():
                symbol = row['symbol']
                
                # Real-time fiyat
                if symbol in realtime_prices:
                    rt_data = realtime_prices[symbol]
                    ws_table.at[idx, 'realtime_price'] = rt_data['price']
                    ws_table.at[idx, 'price_change_24h'] = rt_data['change_percent']
                    ws_table.at[idx, 'volume_24h'] = rt_data['volume_24h']
                    ws_table.at[idx, 'last_update'] = rt_data['timestamp']
                    
                    # Momentum hesapla
                    momentum = self.calculate_realtime_momentum(symbol, rt_data['price'])
                    ws_table.at[idx, 'momentum_1m'] = momentum['momentum_1m']
                    ws_table.at[idx, 'momentum_5m'] = momentum['momentum_5m']
                    ws_table.at[idx, 'trend_strength_rt'] = momentum['trend_strength']
                    
                    # Kombine skor hesapla (AI + Real-time)
                    ai_score = row['ai_score']
                    momentum_bonus = min(abs(momentum['momentum_1m']) * 2, 15)  # Max +15 bonus
                    trend_bonus = min(momentum['trend_strength'] * 1, 10)      # Max +10 bonus
                    
                    # Yön uyumu kontrolü
                    direction_match = False
                    if row['run_type'] == 'long' and momentum['momentum_1m'] > 0:
                        direction_match = True
                    elif row['run_type'] == 'short' and momentum['momentum_1m'] < 0:
                        direction_match = True
                    
                    direction_bonus = 5 if direction_match else -5
                    
                    combined_score = ai_score + momentum_bonus + trend_bonus + direction_bonus
                    combined_score = max(5, min(combined_score, 100))  # 5-100 arası sınırla
                    
                    ws_table.at[idx, 'combined_score'] = combined_score
                    ws_table.at[idx, 'rt_status'] = '🟢 Live' if symbol in realtime_prices else '🔴 Offline'
                    
                else:
                    # Real-time veri yok
                    ws_table.at[idx, 'realtime_price'] = row.get('last_close', 0)
                    ws_table.at[idx, 'price_change_24h'] = 0
                    ws_table.at[idx, 'volume_24h'] = 0
                    ws_table.at[idx, 'momentum_1m'] = 0
                    ws_table.at[idx, 'momentum_5m'] = 0
                    ws_table.at[idx, 'trend_strength_rt'] = 0
                    ws_table.at[idx, 'combined_score'] = row['ai_score']
                    ws_table.at[idx, 'rt_status'] = '⏳ Connecting'
                    ws_table.at[idx, 'last_update'] = datetime.now(LOCAL_TZ)
            
            # Combined score'a göre yeniden sırala
            ws_table = ws_table.sort_values(
                by=['combined_score', 'momentum_1m', 'ai_score'],
                ascending=[False, False, False]
            ).reset_index(drop=True)
            
            self.update_count += 1
            self.last_analysis_time = datetime.now(LOCAL_TZ)
            
            # Top 3 logla
            if len(ws_table) >= 3:
                top3 = ws_table.head(3)
                logger.info(f"🎯 WebSocket Top 3: {', '.join(top3['symbol'].tolist())}")
                
                # Detaylı log (debug)
                for i, (_, signal) in enumerate(top3.iterrows(), 1):
                    logger.debug(
                        f"   #{i}: {signal['symbol']} | Combined={signal['combined_score']:.0f} "
                        f"(AI={signal['ai_score']:.0f} + RT Bonus) | "
                        f"1m={signal['momentum_1m']:+.2f}% | Status={signal['rt_status']}"
                    )
            
            return ws_table
            
        except Exception as e:
            logger.error(f"❌ WebSocket tablosu oluşturma hatası: {e}")
            return websocket_top10_data.copy() if not websocket_top10_data.empty else pd.DataFrame()
    
    def monitor_sl_tp_triggers(self) -> List[Dict]:
        """SL/TP tetikleme izlemesi - Real-time pozisyonlar"""
        global sl_tp_triggers
        
        try:
            current_positions = config.live_positions.copy()
            realtime_prices = get_all_realtime_prices()
            
            triggered_positions = []
            
            for symbol, position in current_positions.items():
                if symbol not in realtime_prices:
                    continue
                    
                current_price = realtime_prices[symbol]['price']
                entry_price = position['entry_price']
                side = position['side']
                stop_loss = position.get('stop_loss', 0)
                take_profit = position.get('take_profit', 0)
                
                # SL tetikleme kontrolü
                sl_triggered = False
                tp_triggered = False
                
                if side == 'LONG':
                    if stop_loss > 0 and current_price <= stop_loss:
                        sl_triggered = True
                    if take_profit > 0 and current_price >= take_profit:
                        tp_triggered = True
                else:  # SHORT
                    if stop_loss > 0 and current_price >= stop_loss:
                        sl_triggered = True
                    if take_profit > 0 and current_price <= take_profit:
                        tp_triggered = True
                
                # Tetikleme kaydı
                if sl_triggered or tp_triggered:
                    trigger_type = "STOP_LOSS" if sl_triggered else "TAKE_PROFIT"
                    trigger_price = stop_loss if sl_triggered else take_profit
                    
                    pnl = 0
                    if side == 'LONG':
                        pnl = (current_price - entry_price) * position['quantity']
                    else:
                        pnl = (entry_price - current_price) * position['quantity']
                    
                    trigger_data = {
                        'symbol': symbol,
                        'side': side,
                        'trigger_type': trigger_type,
                        'entry_price': entry_price,
                        'trigger_price': trigger_price,
                        'current_price': current_price,
                        'pnl': pnl,
                        'quantity': position['quantity'],
                        'timestamp': datetime.now(LOCAL_TZ),
                        'position_data': position
                    }
                    
                    triggered_positions.append(trigger_data)
                    sl_tp_triggers.append(trigger_data)
                    self.trigger_count += 1
                    
                    logger.info(f"🚨 {trigger_type} TETİKLENDİ: {symbol} {side}")
                    logger.info(f"   💰 Entry: ${entry_price:.6f} → Current: ${current_price:.6f}")
                    logger.info(f"   🎯 Target: ${trigger_price:.6f} | P&L: ${pnl:.2f}")
            
            # Eski tetiklemeleri temizle (son 1 saat)
            cutoff_time = datetime.now(LOCAL_TZ) - timedelta(hours=1)
            sl_tp_triggers = [t for t in sl_tp_triggers if t['timestamp'] > cutoff_time]
            
            return triggered_positions
            
        except Exception as e:
            logger.error(f"❌ SL/TP izleme hatası: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """PriceMonitor istatistikleri"""
        return {
            'update_count': self.update_count,
            'trigger_count': self.trigger_count,
            'symbols_tracked': len(self.price_history),
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'momentum_cache_size': len(self.momentum_cache),
            'sl_tp_triggers_total': len(sl_tp_triggers)
        }


# Global PriceMonitor instance
price_monitor = PriceMonitor()


# Global fonksiyonlar
def get_websocket_data() -> pd.DataFrame:
    """WebSocket real-time tablosunu al (UI için)"""
    return price_monitor.create_websocket_realtime_table()


def update_websocket_symbols(ai_signals_df: pd.DataFrame) -> bool:
    """AI sinyallerinden WebSocket sembollerini güncelle"""
    return price_monitor.update_top10_from_ai(ai_signals_df)


def get_top10_realtime_signals() -> pd.DataFrame:
    """Live Bot için: WebSocket tablosundan en iyi 10 sinyali al"""
    return price_monitor.create_websocket_realtime_table()


def get_top3_for_live_bot() -> pd.DataFrame:
    """Live Bot için: WebSocket tablosundan en iyi 3 sinyali al"""
    ws_table = price_monitor.create_websocket_realtime_table()
    if ws_table.empty:
        return pd.DataFrame()
    
    # En iyi 3'ü al
    top3 = ws_table.head(3)
    
    logger.info(f"🤖 Live Bot için Top 3 hazırlandı:")
    for i, (_, signal) in enumerate(top3.iterrows(), 1):
        logger.info(
            f"   🥇 #{i}: {signal['symbol']} | Score={signal['combined_score']:.0f} | "
            f"1m Momentum={signal['momentum_1m']:+.2f}% | Status={signal['rt_status']}"
        )
    
    return top3


def monitor_sl_tp_positions() -> List[Dict]:
    """SL/TP pozisyon izleme (Live Bot için)"""
    return price_monitor.monitor_sl_tp_triggers()


def get_realtime_price_for_symbol(symbol: str) -> Optional[float]:
    """Belirtilen sembol için real-time fiyat al"""
    realtime_prices = get_all_realtime_prices()
    if symbol in realtime_prices:
        return realtime_prices[symbol]['price']
    return None


def get_price_monitor_status() -> Dict:
    """PriceMonitor durumu + WebSocket durumu"""
    ws_status = get_websocket_status()
    pm_stats = price_monitor.get_statistics()
    
    return {
        'websocket': ws_status,
        'price_monitor': pm_stats,
        'top10_last_update': last_top10_update.isoformat() if last_top10_update else None,
        'total_sl_tp_triggers': len(sl_tp_triggers),
        'symbols_with_realtime': len(get_all_realtime_prices())
    }


def cleanup_old_data():
    """Eski verileri temizle (memory management)"""
    global sl_tp_triggers, websocket_signals_history
    
    try:
        # 24 saat eski tetiklemeleri sil
        cutoff_time = datetime.now(LOCAL_TZ) - timedelta(hours=24)
        old_trigger_count = len(sl_tp_triggers)
        sl_tp_triggers = [t for t in sl_tp_triggers if t['timestamp'] > cutoff_time]
        
        # Eski sinyal geçmişlerini temizle
        for symbol in list(websocket_signals_history.keys()):
            if symbol not in websocket_top10_data['symbol'].values:
                del websocket_signals_history[symbol]
        
        # PriceMonitor cache temizliği
        for symbol in list(price_monitor.momentum_cache.keys()):
            if symbol not in websocket_top10_data['symbol'].values:
                del price_monitor.momentum_cache[symbol]
                if symbol in price_monitor.price_history:
                    del price_monitor.price_history[symbol]
        
        cleaned_triggers = old_trigger_count - len(sl_tp_triggers)
        if cleaned_triggers > 0:
            logger.debug(f"🧹 Eski veri temizliği: {cleaned_triggers} tetikleme silindi")
            
    except Exception as e:
        logger.error(f"❌ Veri temizliği hatası: {e}")


# Otomatik temizlik thread'i
import threading

def _auto_cleanup_thread():
    """Otomatik veri temizliği thread'i"""
    while True:
        try:
            time.sleep(3600)  # Her saat
            cleanup_old_data()
        except Exception as e:
            logger.error(f"❌ Auto cleanup hatası: {e}")
            time.sleep(3600)

_cleanup_thread = threading.Thread(target=_auto_cleanup_thread, daemon=True)
_cleanup_thread.start()
logger.debug("🧹 Auto cleanup thread başlatıldı")