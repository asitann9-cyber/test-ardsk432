"""
VP (Volume-Price) Analysis Service
Change %VP analizi ve VP nabız tespiti
"""

import numpy as np
import pandas as pd
import logging
import math
from datetime import datetime
from typing import Dict, Any, List, Optional

from .binance_service import BinanceService

logger = logging.getLogger(__name__)

class VPAnalysisService:
    """Volume-Price analizi ve nabız tespiti servisi"""
    
    @staticmethod
    def calculate_required_candles(time_range_days: int, timeframe: str) -> int:
        """Gerekli mum sayısını hesapla"""
        if timeframe == '1m':
            return min(1000, 24 * 60 * time_range_days)
        elif timeframe == '5m':
            return min(1000, 24 * 12 * time_range_days)
        elif timeframe == '15m':
            return min(1000, 24 * 4 * max(15, time_range_days))
        elif timeframe == '30m':
            return min(1000, 24 * 2 * time_range_days)
        elif timeframe == '1h':
            return min(1000, 24 * max(20, time_range_days))
        elif timeframe == '4h':
            return min(1000, 6 * max(30, time_range_days))
        else:
            return min(1000, 24 * 4 * time_range_days)
    
    @staticmethod
    def calculate_change_vp(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, 
                           volumes: np.ndarray, tw_mode: bool = True) -> Optional[Dict[str, Any]]:
        """
        Change %VP hesaplama - Volume-Price momentum analizi
        
        Args:
            closes (np.ndarray): Kapanış fiyatları
            highs (np.ndarray): En yüksek fiyatlar
            lows (np.ndarray): En düşük fiyatlar
            volumes (np.ndarray): Hacim verileri
            tw_mode (bool): TradingView modu aktif mi
            
        Returns:
            Optional[Dict[str, Any]]: VP analiz sonuçları
        """
        try:
            data_length = len(closes)
            if data_length < 2:
                return None
            
            # Eşik değerleri
            significance_threshold = 0.4
            volume_significance_threshold = 1.8
            
            # Kümülatif değişkenler
            cumulative_positive = 0.0
            cumulative_negative = 0.0
            cumulative_buy = 0.0
            cumulative_sell = 0.0
            
            # Ortalama hacim
            average_volume = np.mean(volumes) if volumes.size > 0 else 0
            
            # Yüzde dizileri
            positive_pct = np.full(data_length, 50.0)
            negative_pct = np.full(data_length, 50.0)
            buy_pct = np.full(data_length, 50.0)
            sell_pct = np.full(data_length, 50.0)
            
            # VP hesaplama döngüsü
            for i in range(1, data_length):
                # Fiyat değişimi
                price_change = closes[i] - closes[i-1]
                if price_change > 0:
                    cumulative_positive += price_change
                elif price_change < 0:
                    cumulative_negative += abs(price_change)
                
                # Fiyat aralığı
                price_range = highs[i] - lows[i]
                if price_range == 0:
                    price_range = 1e-6
                
                # Alım/Satım hacmi hesaplama
                buy_volume = ((closes[i] - lows[i]) / price_range) * volumes[i]
                sell_volume = ((highs[i] - closes[i]) / price_range) * volumes[i]
                
                cumulative_buy += buy_volume
                cumulative_sell += sell_volume
                
                # Toplam hareket ve hacim
                total_movement = cumulative_positive + cumulative_negative or 1e-6
                total_volume = cumulative_buy + cumulative_sell or 1e-6
                
                # Yüzdeleri hesapla
                positive_pct[i] = (cumulative_positive / total_movement) * 100
                negative_pct[i] = (cumulative_negative / total_movement) * 100
                buy_pct[i] = (cumulative_buy / total_volume) * 100
                sell_pct[i] = (cumulative_sell / total_volume) * 100
            
            # TradingView modu uygulamaları
            if tw_mode:
                positive_pct, negative_pct, buy_pct, sell_pct = VPAnalysisService._apply_tradingview_smoothing(
                    positive_pct, negative_pct, buy_pct, sell_pct, closes, volumes, average_volume,
                    significance_threshold, volume_significance_threshold
                )
            
            return {
                "positive_pct": positive_pct.tolist(),
                "negative_pct": negative_pct.tolist(),
                "buy_pct": buy_pct.tolist(),
                "sell_pct": sell_pct.tolist()
            }
            
        except Exception as e:
            logger.error(f"Change VP hesaplama hatası: {e}")
            return None
    
    @staticmethod
    def _apply_tradingview_smoothing(positive_pct, negative_pct, buy_pct, sell_pct, 
                                   closes, volumes, average_volume, sig_threshold, vol_threshold):
        """TradingView tarzı yumuşatma ve önemli olayları işaretleme"""
        try:
            data_length = len(positive_pct)
            
            # Önemli olayları tespit et
            significant_events = []
            if data_length > 5:
                for i in range(5, data_length):
                    previous_close = closes[i-5] or 1e-6
                    recent_price_change = abs(closes[i] - previous_close) / previous_close
                    volume_ratio = volumes[i] / average_volume if average_volume > 0 else 1.0
                    
                    if (recent_price_change > sig_threshold and volume_ratio > 1.5) or (volume_ratio > vol_threshold):
                        event_type = 'strong_buy' if closes[i] > closes[i-5] else 'strong_sell'
                        magnitude = min(3.0, 1.0 + recent_price_change * 2 + min(1.0, volume_ratio / 5))
                        
                        significant_events.append({
                            'index': i,
                            'type': event_type,
                            'magnitude': magnitude
                        })
            
            # Önemli olayların etkilerini uygula
            for event in significant_events:
                impact_range = min(20, max(10, math.floor(event['magnitude'] * 10)))
                start_index = max(0, event['index'] - impact_range)
                end_index = min(data_length - 1, event['index'] + impact_range)
                
                for j in range(start_index, end_index + 1):
                    distance = abs(j - event['index']) / impact_range if impact_range else 0
                    impact = (1 - distance) * event['magnitude']
                    
                    if event['type'] == 'strong_buy':
                        buy_pct[j] += impact * 10
                        positive_pct[j] += impact * 10
                        sell_pct[j] = max(0, sell_pct[j] - impact * 5)
                        negative_pct[j] = max(0, negative_pct[j] - impact * 5)
                    else:  # strong_sell
                        sell_pct[j] += impact * 10
                        negative_pct[j] += impact * 10
                        buy_pct[j] = max(0, buy_pct[j] - impact * 5)
                        positive_pct[j] = max(0, positive_pct[j] - impact * 5)
            
            # Değerleri 0-100 arasında sınırla
            positive_pct = np.clip(positive_pct, 0, 100)
            negative_pct = np.clip(negative_pct, 0, 100)
            buy_pct = np.clip(buy_pct, 0, 100)
            sell_pct = np.clip(sell_pct, 0, 100)
            
            # Ek yumuşatma uygula
            positive_pct, negative_pct, buy_pct, sell_pct = VPAnalysisService._smooth_data(
                positive_pct, negative_pct, buy_pct, sell_pct
            )
            
            return positive_pct, negative_pct, buy_pct, sell_pct
            
        except Exception as e:
            logger.error(f"TradingView yumuşatma hatası: {e}")
            return positive_pct, negative_pct, buy_pct, sell_pct
    
    @staticmethod
    def _smooth_data(pos, neg, buy, sell):
        """Veri yumuşatma işlemi"""
        try:
            n = len(pos)
            if n < 6:
                return pos, neg, buy, sell
            
            base = 50.0
            pull = 0.7
            
            # Hassas bölgeleri tespit et
            sensitive_ranges = []
            for i in range(5, n):
                if abs(buy[i] - buy[i-5]) > 10 or abs(sell[i] - sell[i-5]) > 10:
                    start = max(0, i - 5)
                    end = min(n - 1, i + 10)
                    
                    if sensitive_ranges and start - sensitive_ranges[-1]['end'] < 10:
                        sensitive_ranges[-1]['end'] = max(sensitive_ranges[-1]['end'], end)
                    else:
                        sensitive_ranges.append({'start': start, 'end': end})
            
            # Hassas olmayan bölgelere yumuşatma uygula
            for i in range(1, n):
                inside_sensitive = False
                for range_item in sensitive_ranges:
                    if range_item['start'] <= i <= range_item['end']:
                        inside_sensitive = True
                        break
                
                if not inside_sensitive:
                    pos[i] = base + (pos[i] - base) * (1 - pull)
                    neg[i] = base + (neg[i] - base) * (1 - pull)
                    buy[i] = base + (buy[i] - base) * (1 - pull)
                    sell[i] = base + (sell[i] - base) * (1 - pull)
            
            return pos, neg, buy, sell
            
        except Exception as e:
            logger.error(f"Veri yumuşatma hatası: {e}")
            return pos, neg, buy, sell
    
    @staticmethod
    def detect_vp_pulses(timestamps: List[int], lows: np.ndarray, highs: np.ndarray, 
                        buy_pct: List[float], sell_pct: List[float]) -> List[Dict[str, Any]]:
        """
        VP nabızlarını tespit et - ana sinyal sistemi
        
        Args:
            timestamps: Zaman damgaları (ms)
            lows: En düşük fiyatlar
            highs: En yüksek fiyatlar
            buy_pct: Alım yüzdeleri
            sell_pct: Satım yüzdeleri
            
        Returns:
            List[Dict[str, Any]]: VP nabız listesi
        """
        try:
            POWER_THRESHOLD = 70
            pulses = []
            length = len(buy_pct)
            
            if length < 2:
                return pulses
            
            for i in range(1, length):
                # Güçlü alım nabzı
                if (buy_pct[i] > POWER_THRESHOLD and 
                    buy_pct[i-1] <= POWER_THRESHOLD and 
                    buy_pct[i] > sell_pct[i]):
                    
                    power = (buy_pct[i] - POWER_THRESHOLD) / (100 - POWER_THRESHOLD) * 10
                    pulses.append({
                        'time': timestamps[i],
                        'value': lows[i] * 0.996,  # Mumun altında göster
                        'type': 'strong_buy',
                        'index': i,
                        'power': round(power, 2)
                    })
                
                # Güçlü satım nabzı
                elif (sell_pct[i] > POWER_THRESHOLD and 
                      sell_pct[i-1] <= POWER_THRESHOLD and 
                      sell_pct[i] > buy_pct[i]):
                    
                    power = (sell_pct[i] - POWER_THRESHOLD) / (100 - POWER_THRESHOLD) * 10
                    pulses.append({
                        'time': timestamps[i],
                        'value': highs[i] * 1.004,  # Mumun üstünde göster
                        'type': 'strong_sell',
                        'index': i,
                        'power': round(power, 2)
                    })
            
            logger.info(f"VP Nabız tespiti: {len(pulses)} nabız bulundu")
            return pulses
            
        except Exception as e:
            logger.error(f"VP nabız tespit hatası: {e}")
            return []
    
    @staticmethod
    def analyze_symbol_vp(symbol: str, timeframe: str = '1h', time_range: int = 7, tw_mode: bool = True) -> Optional[Dict[str, Any]]:
        """
        Tek sembol için VP analizi
        
        Args:
            symbol (str): Sembol adı
            timeframe (str): Zaman dilimi
            time_range (int): Zaman aralığı (gün)
            tw_mode (bool): TradingView modu
            
        Returns:
            Optional[Dict[str, Any]]: VP analiz sonuçları
        """
        try:
            # Gerekli mum sayısını hesapla
            candle_limit = VPAnalysisService.calculate_required_candles(time_range, timeframe)
            
            # Binance'den veri çek
            df = BinanceService.fetch_klines_data(symbol, timeframe, candle_limit)
            if df is None or len(df) < 10:
                logger.warning(f"VP analizi için yetersiz veri: {symbol}")
                return None
            
            # NumPy dizilerine çevir
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            timestamps = [int(ts.timestamp() * 1000) for ts in df['timestamp']]
            
            # Change VP hesapla
            vp_result = VPAnalysisService.calculate_change_vp(closes, highs, lows, volumes, tw_mode)
            if not vp_result:
                logger.error(f"VP hesaplama başarısız: {symbol}")
                return None
            
            # VP nabızlarını tespit et
            pulses = VPAnalysisService.detect_vp_pulses(
                timestamps, lows, highs, vp_result['buy_pct'], vp_result['sell_pct']
            )
            
            # Sonuçları derle
            result = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "time_range": time_range,
                "tw_mode": tw_mode,
                "timestamps": timestamps,
                "opens": df['open'].tolist(),
                "highs": highs.tolist(),
                "lows": lows.tolist(),
                "closes": closes.tolist(),
                "volumes": volumes.tolist(),
                "positive_pct": vp_result['positive_pct'],
                "negative_pct": vp_result['negative_pct'],
                "buy_pct": vp_result['buy_pct'],
                "sell_pct": vp_result['sell_pct'],
                "pulses": pulses,
                "last_candle_time_ms": timestamps[-1] if timestamps else None,
                "analysis_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"VP analizi tamamlandı: {symbol} - {len(pulses)} nabız")
            return result
            
        except Exception as e:
            logger.error(f"VP analiz hatası {symbol}: {e}")
            return None
    
    @staticmethod
    def get_vp_summary(pulses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        VP nabız özetini hazırla
        
        Args:
            pulses: VP nabız listesi
            
        Returns:
            Dict[str, Any]: VP özeti
        """
        try:
            if not pulses:
                return {
                    'total_pulses': 0,
                    'buy_pulses': 0,
                    'sell_pulses': 0,
                    'last_pulse': None,
                    'average_power': 0,
                    'strongest_pulse': None
                }
            
            buy_pulses = [p for p in pulses if p['type'] == 'strong_buy']
            sell_pulses = [p for p in pulses if p['type'] == 'strong_sell']
            
            # En güçlü nabzı bul
            strongest_pulse = max(pulses, key=lambda x: x.get('power', 0))
            
            # Ortalama güç
            average_power = sum(p.get('power', 0) for p in pulses) / len(pulses)
            
            return {
                'total_pulses': len(pulses),
                'buy_pulses': len(buy_pulses),
                'sell_pulses': len(sell_pulses),
                'last_pulse': pulses[-1] if pulses else None,
                'average_power': round(average_power, 2),
                'strongest_pulse': strongest_pulse
            }
            
        except Exception as e:
            logger.error(f"VP özet hatası: {e}")
            return {
                'total_pulses': 0,
                'buy_pulses': 0,
                'sell_pulses': 0,
                'last_pulse': None,
                'average_power': 0,
                'strongest_pulse': None
            }