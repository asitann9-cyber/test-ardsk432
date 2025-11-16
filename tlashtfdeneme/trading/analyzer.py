"""
🔍 Sinyal Analiz Modülü - Ultra Panel v5 Multi-HTF Sistemi
AI destekli kripto sinyal analizi ve batch processing
🔥 Heikin Ashi Multi-Timeframe analizi
🔥 Ultra Signal (3/4 HTF crossover)
🔥 Candle Power + Whale Detection
🔥 Memory Sistemi (Pine Script uyumlu)
🔥 YENİ: WebSocket Real-Time Analiz Desteği
"""

import time
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    LOCAL_TZ, MAX_WORKERS, REQ_SLEEP, DEFAULT_MIN_AI_SCORE,
    current_data, saved_signals
)
from data.fetch_data import fetch_klines, get_usdt_perp_symbols
from core.indicators import compute_ultra_metrics
from core.ai_model import ai_model

logger = logging.getLogger("crypto-analytics")

# 🔥 YENİ: WebSocket import
try:
    from websocket_stream import (
        BinanceWebSocketStream, 
        convert_ws_kline_to_dict,
        set_websocket_instance,
        get_websocket_instance,
        is_websocket_active
    )
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.warning("⚠️ WebSocket modülü bulunamadı - Real-time analiz devre dışı")


# 🔥 Global WebSocket değişkenleri
_realtime_active = False
_realtime_symbols = []
_realtime_interval = '15m'


def analyze_symbol_with_ai(symbol: str, interval: str) -> Dict:
    """
    🔥 Ultra Panel v5 - Multi-HTF sembol analizi
    
    Args:
        symbol (str): Trading sembolü
        interval (str): Zaman dilimi
        
    Returns:
        Dict: Ultra Panel analiz sonuçları
    """
    try:
        # Rate limiting
        time.sleep(REQ_SLEEP)
        
        # Veri çek
        df = fetch_klines(symbol, interval)
        if df is None or df.empty:
            logger.debug(f"❌ {symbol}: Veri çekilemedi")
            return {}
        
        if len(df) < 100:  # Ultra Panel için daha fazla veri gerekli (HTF hesabı için)
            logger.debug(f"❌ {symbol}: Yetersiz veri ({len(df)} < 100)")
            return {}
        
        # 🔥 Ultra Panel metriklerini hesapla
        metrics = compute_ultra_metrics(df, symbol)
        
        if not metrics or metrics.get('run_type') == 'none':
            logger.debug(f"❌ {symbol}: Ultra signal bulunamadı")
            return {}
        
        # Ultra Signal kontrolü
        if not metrics.get('ultra_strong_buy') and not metrics.get('ultra_strong_sell'):
            logger.debug(f"❌ {symbol}: Ultra signal yok")
            return {}
        
        # HTF count kontrolü (en az 3/4 olmalı)
        htf_count = metrics.get('htf_count', 0)
        if htf_count < 3:
            logger.debug(f"❌ {symbol}: HTF count düşük ({htf_count}/4)")
            return {}
        
        # Total Power kontrolü
        total_power = metrics.get('total_power', 0.0)
        if total_power < 5.0:
            logger.debug(f"❌ {symbol}: Power çok düşük ({total_power:.1f})")
            return {}
        
        # AI skoru hesapla (Ultra Panel bazlı)
        ai_score = ai_model.predict_score(metrics)
        
        # Minimum AI skoru kontrolü
        min_ai_threshold = DEFAULT_MIN_AI_SCORE * 100
        if ai_score < min_ai_threshold:
            logger.debug(f"❌ {symbol}: AI skoru düşük ({ai_score:.1f} < {min_ai_threshold})")
            return {}
        
        # Son fiyat ve zaman bilgisi
        last_row = df.iloc[-1]
        last_close = float(last_row['close'])
        last_update = last_row['close_time']
        
        # Başarılı Ultra Signal
        result = {
            'symbol': symbol,
            'timeframe': interval,
            'last_close': last_close,
            'run_type': metrics['run_type'],  # 'long' veya 'short'
            
            # 🔥 Ultra Panel verileri
            'ultra_strong_buy': metrics['ultra_strong_buy'],
            'ultra_strong_sell': metrics['ultra_strong_sell'],
            'bull_count': metrics['bull_count'],
            'bear_count': metrics['bear_count'],
            'htf_count': htf_count,  # 3/4 veya 4/4
            'total_power': total_power,
            'whale_active': metrics.get('whale_active', False),
            
            # Tetikleyici
            'trigger_type': 'Ultra Signal',
            
            # AI Skoru
            'ai_score': ai_score,
            
            # Zaman
            'last_update': last_update.strftime('%Y-%m-%d %H:%M:%S %Z'),
            
            # 🔥 BOT UYUMLU ALANLAR (geriye uyumluluk için)
            'run_count': htf_count,  # HTF count (3 veya 4)
            'run_perc': total_power / 5.0,  # Power normalize
            'gauss_run': total_power * 2.0,  # Power*2
            'gauss_run_perc': total_power,  # Power değeri
            'log_volume': total_power / 2.0,  # Power/2
            'log_volume_momentum': total_power / 3.0,  # Power/3
            'deviso_ratio': float(metrics.get('whale_active', False)),  # Whale flag
            'c_signal_momentum': total_power,  # Total power
            'max_zscore': 0.0,  # Deprecated - sıfır
            
            # Trend bilgileri
            'trend_direction': metrics['run_type'].upper(),
            'trend_strength': total_power
        }
        
        logger.debug(
            f"✅ {symbol}: Ultra={htf_count}/4, "
            f"Power={total_power:.1f}, AI={ai_score:.0f}%, "
            f"Whale={'YES' if metrics.get('whale_active') else 'NO'}"
        )
        return result
        
    except Exception as e:
        logger.warning(f"analyze_symbol error {symbol}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {}


# 🔥 WebSocket Real-Time Analiz Fonksiyonları
def _process_realtime_kline(symbol: str, kline_dict: Dict) -> None:
    """
    🔥 WebSocket'ten gelen kline verisini işle
    
    Args:
        symbol: Trading sembolü
        kline_dict: WebSocket kline verisi
    """
    try:
        # Yeni veri çek (HTF hesabı için)
        df = fetch_klines(symbol, _realtime_interval)
        
        if df is None or df.empty:
            logger.debug(f"⚡ {symbol}: Real-time veri çekilemedi")
            return
        
        # Ultra Panel analizi yap
        result = analyze_symbol_with_ai(symbol, _realtime_interval)
        
        if result:
            logger.info(
                f"⚡ REAL-TIME: {symbol} | "
                f"Ultra={result['htf_count']}/4 | "
                f"Power={result['total_power']:.1f} | "
                f"AI={result['ai_score']:.0f}%"
            )
            
            # Config'e ekle/güncelle (UI otomatik güncellenecek)
            _update_realtime_signal(result)
        
    except Exception as e:
        logger.error(f"⚡ Real-time analiz hatası {symbol}: {e}")


def _update_realtime_signal(result: Dict) -> None:
    """
    🔥 Real-time sinyali config'e ekle/güncelle
    
    Args:
        result: Analiz sonucu
    """
    import config
    
    if config.current_data is None:
        config.current_data = pd.DataFrame([result])
    else:
        symbol = result['symbol']
        
        # Var olan sinyali güncelle veya yeni ekle
        if symbol in config.current_data['symbol'].values:
            # Güncelle
            idx = config.current_data[config.current_data['symbol'] == symbol].index[0]
            for key, value in result.items():
                config.current_data.at[idx, key] = value
        else:
            # Yeni ekle
            config.current_data = pd.concat([
                config.current_data,
                pd.DataFrame([result])
            ], ignore_index=True)
        
        # Sıralamayı koru (AI Score > Power > HTF Count)
        config.current_data = config.current_data.sort_values(
            by=['ai_score', 'total_power', 'htf_count'],
            ascending=[False, False, False]
        )


def start_realtime_analysis(timeframe: str = '15m', symbols: Optional[List[str]] = None) -> bool:
    """
    🔥 Real-time WebSocket analizi başlat
    
    Args:
        timeframe: Zaman dilimi
        symbols: Sembol listesi (None ise tüm USDT perpetual'lar)
        
    Returns:
        bool: Başarılı mı?
    """
    global _realtime_active, _realtime_symbols, _realtime_interval
    
    if not WEBSOCKET_AVAILABLE:
        logger.error("❌ WebSocket modülü bulunamadı")
        return False
    
    if _realtime_active:
        logger.warning("⚠️ Real-time analiz zaten aktif")
        return False
    
    # Sembolleri al
    if symbols is None:
        symbols = get_usdt_perp_symbols()
    
    if not symbols:
        logger.error("❌ Sembol listesi boş")
        return False
    
    _realtime_symbols = symbols
    _realtime_interval = timeframe
    
    logger.info(f"📡 Real-time WebSocket analizi başlatılıyor...")
    logger.info(f"   Sembol sayısı: {len(symbols)}")
    logger.info(f"   Timeframe: {timeframe}")
    logger.info(f"   Mod: Ultra Panel v5 Multi-HTF")
    
    try:
        # WebSocket callback fonksiyonu
        def on_new_kline(symbol: str, kline: dict):
            """Her yeni candle kapandığında çağrılır"""
            kline_dict = convert_ws_kline_to_dict(kline)
            if kline_dict:
                _process_realtime_kline(symbol, kline_dict)
        
        # WebSocket stream'i başlat
        ws_stream = BinanceWebSocketStream(symbols, timeframe)
        ws_stream.subscribe(on_new_kline)
        
        # Global instance'ı set et
        set_websocket_instance(ws_stream)
        
        _realtime_active = True
        logger.info("✅ Real-time WebSocket analizi başlatıldı!")
        logger.info("⚡ Candle kapanışlarında otomatik Ultra Panel analizi yapılacak")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Real-time analiz başlatma hatası: {e}")
        _realtime_active = False
        return False


def stop_realtime_analysis() -> None:
    """
    🔥 Real-time analizi durdur
    """
    global _realtime_active
    
    if not _realtime_active:
        logger.info("ℹ️ Real-time analiz zaten durdurulmuş")
        return
    
    logger.info("🛑 Real-time WebSocket analizi durduruluyor...")
    
    try:
        ws_stream = get_websocket_instance()
        if ws_stream:
            ws_stream.stop()
            set_websocket_instance(None)
        
        _realtime_active = False
        logger.info("✅ Real-time analiz durduruldu")
        
    except Exception as e:
        logger.error(f"❌ Real-time analiz durdurma hatası: {e}")


def is_realtime_active() -> bool:
    """
    🔥 Real-time analiz aktif mi?
    
    Returns:
        bool: Aktif ise True
    """
    return _realtime_active and is_websocket_active()


def get_realtime_status() -> Dict:
    """
    🔥 Real-time analiz durumu
    
    Returns:
        Dict: Durum bilgileri
    """
    from websocket_stream import get_websocket_status
    
    ws_status = get_websocket_status()
    
    return {
        'active': _realtime_active,
        'symbols_count': len(_realtime_symbols),
        'interval': _realtime_interval,
        'websocket_status': ws_status
    }


# 🔥 BATCH ANALIZ (MEMORY SİSTEMİ İLE)
def batch_analyze_with_ai(interval: str) -> pd.DataFrame:
    """
    🔥 Ultra Panel v5 - Toplu analiz + Memory sistemi
    
    Args:
        interval (str): Analiz edilecek zaman dilimi
        
    Returns:
        pd.DataFrame: Analiz sonuçları
    """
    global saved_signals
    
    start_time = time.time()
    
    # Sembol listesini al
    symbols = get_usdt_perp_symbols()
    if not symbols:
        logger.error("Sembol listesi boş!")
        return pd.DataFrame()
    
    logger.info(f"🤖 {len(symbols)} sembol için Ultra Panel analizi başlatılıyor...")
    
    # Yeni analiz sonuçları
    fresh_results = []
    processed_count = 0
    ultra_success_count = 0
    
    # Paralel işleme
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(analyze_symbol_with_ai, sym, interval): sym for sym in symbols}
        
        for fut in as_completed(futures):
            symbol = futures[fut]
            processed_count += 1
            
            try:
                res = fut.result()
                if res:  # Geçerli Ultra Signal
                    fresh_results.append(res)
                    ultra_success_count += 1
                    
                    # 🔥 MEMORY SİSTEMİ: Kaydedilmiş sinyalleri güncelle
                    saved_signals[symbol] = {
                        'data': res,
                        'last_seen': datetime.now(LOCAL_TZ)
                    }
                
                # İlerleme logu
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    success_rate = (ultra_success_count / processed_count) * 100
                    logger.info(
                        f"🤖 İşlenen: {processed_count}/{len(symbols)} - Hız: {rate:.1f} s/sn - "
                        f"Ultra: {success_rate:.1f}% ({ultra_success_count})"
                    )
            
            except Exception as e:
                logger.debug(f"Future hatası {symbol}: {e}")
    
    # 🔥 MEMORY SİSTEMİ: Eski sinyalleri koruma
    current_time = datetime.now(LOCAL_TZ)
    fresh_symbols = {r['symbol'] for r in fresh_results}
    
    protected_count = 0
    for symbol, saved_info in list(saved_signals.items()):
        # Pine Script'teki barssince() mantığı
        minutes_old = (current_time - saved_info['last_seen']).total_seconds() / 60.0
        
        # 15 dakikadan eski sinyalleri sil (Pine Script'te memory sürekli tutuluyor ama biz limit koyalım)
        if minutes_old > 15:
            del saved_signals[symbol]
            continue
        
        # Yeni taramada bulunamayan ama hafızada olan sinyaller
        if symbol not in fresh_symbols:
            old_data = saved_info['data'].copy()
            original_score = old_data['ai_score']
            
            # Power ve HTF count'a göre ceza
            power = old_data.get('total_power', 0)
            htf_count = old_data.get('htf_count', 0)
            
            # Güçlü sinyal = az ceza
            if power > 20.0 and htf_count == 4:
                base_penalty = 5
            elif power > 10.0 and htf_count >= 3:
                base_penalty = 10
            else:
                base_penalty = 20
            
            # Yaşa göre ek ceza
            if minutes_old <= 3:
                penalty = base_penalty
            elif minutes_old <= 7:
                penalty = base_penalty + 10
            else:
                penalty = base_penalty + 20
            
            new_score = max(5, original_score - penalty)
            old_data['ai_score'] = new_score
            old_data['score_status'] = f"📉-{penalty}"
            
            fresh_results.append(old_data)
            protected_count += 1
            
            logger.debug(
                f"📉 {symbol}: {original_score:.0f} → {new_score:.0f} "
                f"(yaş: {minutes_old:.1f}dk, Power: {power:.1f}, HTF: {htf_count}/4)"
            )
    
    # Performans istatistikleri
    elapsed_time = time.time() - start_time
    total_rate = len(symbols) / elapsed_time if elapsed_time > 0 else 0
    
    new_signals = len(fresh_symbols)
    total_signals = len(fresh_results)
    ultra_success_rate = (ultra_success_count / len(symbols)) * 100 if len(symbols) > 0 else 0
    
    logger.info("✅ Ultra Panel Analiz tamamlandı:")
    logger.info(f"   🆕 Yeni Ultra signal: {new_signals}")
    logger.info(f"   📉 Korunan sinyal (Memory): {protected_count}")
    logger.info(f"   🎯 Toplam sinyal: {total_signals}")
    logger.info(f"   📊 Ultra başarı oranı: {ultra_success_rate:.1f}%")
    logger.info(f"   ⏱️ Süre: {elapsed_time:.1f}s - Hız: {total_rate:.1f} s/sn")
    
    if not fresh_results:
        logger.warning("⚠️ Hiç Ultra signal bulunamadı")
        return pd.DataFrame()
    
    # DataFrame oluştur ve sırala
    df = pd.DataFrame(fresh_results)
    
    # Sıralama: AI Score > Power > HTF Count
    df = df.sort_values(
        by=['ai_score', 'total_power', 'htf_count'],
        ascending=[False, False, False]
    )
    
    if len(df) > 0:
        top_signal = df.iloc[0]
        logger.info(
            f"🏆 En yüksek AI skoru: {top_signal['ai_score']:.0f}% - {top_signal['symbol']} "
            f"(Ultra: {top_signal['htf_count']}/4, "
            f"Power: {top_signal['total_power']:.1f}, "
            f"Whale: {'YES' if top_signal.get('whale_active') else 'NO'})"
        )
        
        # Run type dağılımı
        long_count = len(df[df['run_type'] == 'long'])
        short_count = len(df[df['run_type'] == 'short'])
        logger.info(f"📈 Sinyal dağılımı: LONG={long_count}, SHORT={short_count}")
        
        if protected_count > 0:
            logger.info("📉 Korunan sinyaller skor düşüşü ile aşağı kaydı (Memory System)")
        
        logger.debug("📊 Analyzer sonrası ilk 3 sinyal:")
        for idx, (i, row) in enumerate(df.head(3).iterrows(), 1):
            logger.debug(
                f"   {idx}: {row['symbol']} | AI={row['ai_score']:.0f}% | "
                f"Ultra={row['htf_count']}/4 | Power={row['total_power']:.1f}"
            )
    
    return df


def filter_signals(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    🔥 Ultra Panel bazlı sinyal filtreleme
    
    Args:
        df (pd.DataFrame): Sinyal DataFrame'i
        filters (Dict): Filtre parametreleri
        
    Returns:
        pd.DataFrame: Filtrelenmiş sinyaller
    """
    if df.empty:
        return df
    
    filtered_df = df.copy()
    original_count = len(filtered_df)
    
    # AI skoru filtresi
    if filters.get('min_ai_score', 0) > 0:
        filtered_df = filtered_df[filtered_df['ai_score'] >= filters['min_ai_score']]
    
    # 🔥 Power filtresi
    if filters.get('min_power', 0) > 0:
        filtered_df = filtered_df[filtered_df['total_power'] >= filters['min_power']]
    
    # 🔥 HTF count filtresi (3/4 veya 4/4)
    if filters.get('min_htf_count', 0) > 0:
        filtered_df = filtered_df[filtered_df['htf_count'] >= filters['min_htf_count']]
    
    # 🔥 Whale filtresi
    if filters.get('whale_only', False):
        filtered_df = filtered_df[filtered_df['whale_active'] == True]
    
    # Run type filtresi
    run_type_filter = filters.get('run_type')
    if run_type_filter and run_type_filter != 'all':
        filtered_df = filtered_df[filtered_df['run_type'] == run_type_filter]
    
    filtered_count = len(filtered_df)
    logger.info(f"🔍 Filtre sonucu: {filtered_count}/{original_count} sinyal kaldı")
    
    return filtered_df


def get_top_signals(df: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    """
    🔥 En iyi Ultra Panel sinyalleri al
    
    Args:
        df (pd.DataFrame): Sinyal DataFrame'i
        count (int): Alınacak sinyal sayısı
        
    Returns:
        pd.DataFrame: En iyi sinyaller
    """
    if df.empty:
        return df
    
    sorted_df = df.sort_values(
        by=['ai_score', 'total_power', 'htf_count'],
        ascending=[False, False, False]
    )
    
    return sorted_df.head(count)


def analyze_signal_quality(df: pd.DataFrame) -> Dict:
    """
    🔥 Ultra Panel bazlı sinyal kalitesi analizi
    
    Args:
        df (pd.DataFrame): Analiz edilecek sinyaller
        
    Returns:
        Dict: Kalite metrikleri
    """
    if df.empty:
        return {
            'total_signals': 0,
            'avg_ai_score': 0,
            'long_signals': 0,
            'short_signals': 0,
            'high_quality_signals': 0,
            'quality_distribution': {},
            'ultra_quality': {}
        }
    
    total_signals = len(df)
    avg_ai_score = df['ai_score'].mean()
    
    long_signals = len(df[df['run_type'] == 'long'])
    short_signals = len(df[df['run_type'] == 'short'])
    
    # Kalite kategorileri
    high_quality = len(df[df['ai_score'] >= 80])
    medium_quality = len(df[(df['ai_score'] >= 60) & (df['ai_score'] < 80)])
    low_quality = len(df[df['ai_score'] < 60])
    
    # 🔥 Ultra Panel kalite analizi
    ultra_quality = {}
    if 'total_power' in df.columns and 'htf_count' in df.columns:
        avg_power = df['total_power'].mean()
        perfect_ultra = len(df[df['htf_count'] == 4])  # 4/4 Ultra
        good_ultra = len(df[df['htf_count'] == 3])      # 3/4 Ultra
        whale_count = len(df[df['whale_active'] == True])
        
        ultra_quality = {
            'avg_power': avg_power,
            'perfect_ultra_4_4': perfect_ultra,
            'good_ultra_3_4': good_ultra,
            'whale_signals': whale_count
        }
    
    return {
        'total_signals': total_signals,
        'avg_ai_score': avg_ai_score,
        'long_signals': long_signals,
        'short_signals': short_signals,
        'high_quality_signals': high_quality,
        'quality_distribution': {
            'high': high_quality,
            'medium': medium_quality,
            'low': low_quality
        },
        'long_ratio': (long_signals / total_signals * 100) if total_signals > 0 else 0,
        'short_ratio': (short_signals / total_signals * 100) if total_signals > 0 else 0,
        'ultra_quality': ultra_quality
    }


def update_signal_scores():
    """
    🔥 Kayıtlı sinyallerin skorlarını güncelle (Memory System)
    """
    global saved_signals
    
    current_time = datetime.now(LOCAL_TZ)
    updated_count = 0
    removed_count = 0
    
    for symbol, saved_info in list(saved_signals.items()):
        minutes_old = (current_time - saved_info['last_seen']).total_seconds() / 60.0
        
        # 20 dakikadan eski sinyalleri sil
        if minutes_old > 20:
            del saved_signals[symbol]
            removed_count += 1
            continue
        
        # Yaşlanma cezası uygula
        if minutes_old > 3:
            original_score = saved_info['data']['ai_score']
            
            # Power ve HTF count bazlı ceza
            power = saved_info['data'].get('total_power', 0)
            htf_count = saved_info['data'].get('htf_count', 0)
            
            # Güçlü sinyal = az ceza
            if power > 20.0 and htf_count == 4:
                base_penalty = 5
            elif power > 10.0 and htf_count >= 3:
                base_penalty = 10
            else:
                base_penalty = 20
            
            # Yaşa göre ek ceza
            if minutes_old <= 7:
                penalty = base_penalty
            elif minutes_old <= 14:
                penalty = base_penalty + 15
            else:
                penalty = base_penalty + 25
            
            new_score = max(1, original_score - penalty)
            saved_info['data']['ai_score'] = new_score
            updated_count += 1
    
    if updated_count > 0 or removed_count > 0:
        logger.debug(f"🔄 Memory güncelleme: {updated_count} güncellendi, {removed_count} silindi")


def get_signal_summary() -> Dict:
    """
    🔥 Ultra Panel bazlı sinyal özeti
    
    Returns:
        Dict: Sinyal özet bilgileri
    """
    global current_data
    
    if current_data is None or current_data.empty:
        return {
            'total_signals': 0,
            'long_count': 0,
            'short_count': 0,
            'avg_ai_score': 0,
            'top_symbol': None,
            'last_update': None,
            'ultra_stats': {}
        }
    
    total_signals = len(current_data)
    long_count = len(current_data[current_data['run_type'] == 'long'])
    short_count = len(current_data[current_data['run_type'] == 'short'])
    avg_ai_score = current_data['ai_score'].mean()
    
    # En yüksek skorlu sembol
    top_signal = current_data.iloc[0] if not current_data.empty else None
    top_symbol = top_signal['symbol'] if top_signal is not None else None
    
    # 🔥 Ultra Panel istatistikleri
    ultra_stats = {}
    if 'total_power' in current_data.columns and not current_data.empty:
        ultra_stats = {
            'avg_power': current_data['total_power'].mean(),
            'max_power': current_data['total_power'].max(),
            'perfect_ultra_count': len(current_data[current_data['htf_count'] == 4]),
            'good_ultra_count': len(current_data[current_data['htf_count'] == 3]),
            'whale_count': len(current_data[current_data['whale_active'] == True])
        }
    
    return {
        'total_signals': total_signals,
        'long_count': long_count,
        'short_count': short_count,
        'avg_ai_score': avg_ai_score,
        'top_symbol': top_symbol,
        'last_update': datetime.now(LOCAL_TZ).strftime('%H:%M:%S'),
        'ultra_stats': ultra_stats
    }
