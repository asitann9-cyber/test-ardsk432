"""
🔍 Sinyal Analiz Modülü
AI destekli kripto sinyal analizi ve batch processing
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
from core.indicators import compute_consecutive_metrics
from core.ai_model import ai_model

logger = logging.getLogger("crypto-analytics")


def analyze_symbol_with_ai(symbol: str, interval: str) -> Dict:
    """
    AI skoru ile sembol analizi - DOĞRU HESAPLAMALARLA
    
    Args:
        symbol (str): Analiz edilecek sembol
        interval (str): Zaman dilimi
        
    Returns:
        Dict: Analiz sonuçları veya boş dict
    """
    try:
        # Rate limiting
        time.sleep(REQ_SLEEP)
        
        # Veri çek
        df = fetch_klines(symbol, interval)
        if df is None or df.empty:
            return {}
        
        # Teknik metrikleri hesapla
        metrics = compute_consecutive_metrics(df)
        if not metrics or metrics['run_type'] == 'none':
            return {}
        
        # Trend yönü kontrolü
        streak_type = metrics['run_type']
        deviso_ratio = metrics['deviso_ratio']
        
        # Trend uyumu kontrolü
        direction_match = False
        if streak_type == 'long' and deviso_ratio > 0:
            direction_match = True
        elif streak_type == 'short' and deviso_ratio < 0:
            direction_match = True
        
        # Trend uyumsuzluğunda sinyali reddet
        if not direction_match:
            return {}
        
        # AI skoru hesapla
        ai_score = ai_model.predict_score(metrics)
        
        # Minimum AI skoru kontrolü
        if ai_score < (DEFAULT_MIN_AI_SCORE * 100):
            return {}
        
        # Son fiyat ve zaman bilgisi
        last_row = df.iloc[-1]
        last_close = float(last_row['close'])
        last_update = last_row['close_time']
        
        return {
            'symbol': symbol,
            'timeframe': interval,
            'last_close': last_close,
            'run_type': metrics['run_type'],
            'run_count': metrics['run_count'],
            'run_perc': metrics['run_perc'],
            'gauss_run': metrics['gauss_run'],
            'gauss_run_perc': metrics['gauss_run_perc'],
            'vol_ratio': metrics['vol_ratio'],
            'hh_vol_streak': metrics['hh_vol_streak'],
            'deviso_ratio': metrics['deviso_ratio'],
            'ai_score': ai_score,  
            'last_update': last_update.strftime('%Y-%m-%d %H:%M:%S %Z'),
        }
        
    except Exception as e:
        logger.warning(f"analyze_symbol error {symbol}: {e}")
        return {}


def batch_analyze_with_ai(interval: str) -> pd.DataFrame:
    """
    🔥 GÜNCELLENMIŞ: AI skoru düşen sinyaller silinmez, sadece skoru azalır
    Sinyaller tabloda kalır ve skor değişimine göre yukarı/aşağı hareket eder
    
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
    
    logger.info(f"🤖 {len(symbols)} sembol için AI analiz başlatılıyor...")
    
    # Yeni analiz sonuçları
    fresh_results = []
    processed_count = 0
    
    # Paralel işleme
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(analyze_symbol_with_ai, sym, interval): sym for sym in symbols}
        
        for fut in as_completed(futures):
            symbol = futures[fut]
            processed_count += 1
            
            try:
                res = fut.result()
                if res:  # Geçerli sinyal
                    fresh_results.append(res)
                    
                    # Kaydedilmiş sinyalleri güncelle
                    saved_signals[symbol] = {
                        'data': res,
                        'last_seen': datetime.now(LOCAL_TZ)
                    }
                    
                # İlerleme logu
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    logger.info(f"🤖 İşlenen: {processed_count}/{len(symbols)} - Hız: {rate:.1f} s/sn - AI Onaylı: {len(fresh_results)}")
                    
            except Exception as e:
                logger.debug(f"Future hatası {symbol}: {e}")

    # Mevcut zaman
    current_time = datetime.now(LOCAL_TZ)
    fresh_symbols = {r['symbol'] for r in fresh_results} 
    
    # Eski sinyalleri koruma ve skor düşürme
    protected_count = 0
    for symbol, saved_info in list(saved_signals.items()):
        # Yaş kontrolü
        minutes_old = (current_time - saved_info['last_seen']).total_seconds() / 60.0
        if minutes_old > 10:
            del saved_signals[symbol]
            continue
        
        # Yeni analizde bulunamayan sinyaller
        if symbol not in fresh_symbols:
            old_data = saved_info['data'].copy()
            original_score = old_data['ai_score']
            
            # Yaşa göre ceza
            if minutes_old <= 2:
                penalty = 15  
            elif minutes_old <= 5:
                penalty = 30 
            else:
                penalty = 50  
            
            new_score = max(5, original_score - penalty)  # Minimum 5 puan
            old_data['ai_score'] = new_score
            old_data['score_status'] = f"📉-{penalty}"  
            
            # Eski sinyali korunmuş listesine ekle
            fresh_results.append(old_data)
            protected_count += 1
            
            logger.debug(f"📉 {symbol}: {original_score:.0f} → {new_score:.0f} (yaş: {minutes_old:.1f}dk)")

    # Performans istatistikleri
    elapsed_time = time.time() - start_time
    total_rate = len(symbols) / elapsed_time if elapsed_time > 0 else 0
    
    new_signals = len(fresh_symbols)
    total_signals = len(fresh_results)
    
    logger.info(f"✅ AI Analiz tamamlandı:")
    logger.info(f"   🆕 Yeni sinyal: {new_signals}")
    logger.info(f"   📉 Korunan sinyal: {protected_count}")
    logger.info(f"   🎯 Toplam sinyal: {total_signals}")
    logger.info(f"   ⏱️ Süre: {elapsed_time:.1f}s - Hız: {total_rate:.1f} s/sn")
    
    if not fresh_results:
        return pd.DataFrame()
        
    # DataFrame oluştur ve sırala
    df = pd.DataFrame(fresh_results)
    df = df.sort_values(by=['ai_score', 'run_perc', 'gauss_run', 'vol_ratio'], 
                       ascending=[False, False, False, False])
    
    if len(df) > 0:
        logger.info(f"🏆 En yüksek AI skoru: {df.iloc[0]['ai_score']:.0f}% - {df.iloc[0]['symbol']}")
        if protected_count > 0:
            logger.info(f"📉 Korunan sinyaller skor düşüşü ile aşağı kaydı")
    
    return df


def filter_signals(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Sinyalleri belirtilen filtrelere göre süz
    
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
    
    # AI skor filtresi
    min_ai_score = filters.get('min_ai_score', 0)
    if min_ai_score > 0:
        filtered_df = filtered_df[filtered_df['ai_score'] >= min_ai_score]
    
    # Run count filtresi
    min_run_count = filters.get('min_run_count', 0)
    if min_run_count > 0:
        filtered_df = filtered_df[filtered_df['run_count'] >= min_run_count]
    
    # Run percentage filtresi
    min_run_perc = filters.get('min_run_perc', 0)
    if min_run_perc > 0:
        filtered_df = filtered_df[filtered_df['run_perc'] >= min_run_perc]
    
    # Volume ratio filtresi
    min_vol_ratio = filters.get('min_vol_ratio', 0)
    if min_vol_ratio > 0:
        filtered_df = filtered_df[(filtered_df['vol_ratio'].isna()) | 
                                 (filtered_df['vol_ratio'] >= min_vol_ratio)]
    
    # Run type filtresi
    run_type_filter = filters.get('run_type')
    if run_type_filter and run_type_filter != 'all':
        filtered_df = filtered_df[filtered_df['run_type'] == run_type_filter]
    
    filtered_count = len(filtered_df)
    logger.info(f"🔍 Filtre sonucu: {filtered_count}/{original_count} sinyal kaldı")
    
    return filtered_df


def get_top_signals(df: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    """
    En iyi sinyalleri al
    
    Args:
        df (pd.DataFrame): Sinyal DataFrame'i
        count (int): Alınacak sinyal sayısı
        
    Returns:
        pd.DataFrame: En iyi sinyaller
    """
    if df.empty:
        return df
    
    # Sıralama: AI skoru > Run percentage > Gauss run > Volume ratio
    sorted_df = df.sort_values(
        by=['ai_score', 'run_perc', 'gauss_run', 'vol_ratio'],
        ascending=[False, False, False, False]
    )
    
    return sorted_df.head(count)


def analyze_signal_quality(df: pd.DataFrame) -> Dict:
    """
    Sinyal kalitesi analizi
    
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
            'quality_distribution': {}
        }
    
    total_signals = len(df)
    avg_ai_score = df['ai_score'].mean()
    
    long_signals = len(df[df['run_type'] == 'long'])
    short_signals = len(df[df['run_type'] == 'short'])
    
    # Kalite kategorileri
    high_quality = len(df[df['ai_score'] >= 80])
    medium_quality = len(df[(df['ai_score'] >= 60) & (df['ai_score'] < 80)])
    low_quality = len(df[df['ai_score'] < 60])
    
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
        'short_ratio': (short_signals / total_signals * 100) if total_signals > 0 else 0
    }


def update_signal_scores():
    """
    Kayıtlı sinyallerin skorlarını güncelle (yaş cezası)
    """
    global saved_signals
    
    current_time = datetime.now(LOCAL_TZ)
    updated_count = 0
    removed_count = 0
    
    for symbol, saved_info in list(saved_signals.items()):
        minutes_old = (current_time - saved_info['last_seen']).total_seconds() / 60.0
        
        # Çok eski sinyalleri sil
        if minutes_old > 15:
            del saved_signals[symbol]
            removed_count += 1
            continue
        
        # Yaşlanma cezası uygula
        if minutes_old > 2:
            original_score = saved_info['data']['ai_score']
            
            if minutes_old <= 5:
                penalty = 20
            elif minutes_old <= 10:
                penalty = 40
            else:
                penalty = 60
            
            new_score = max(1, original_score - penalty)
            saved_info['data']['ai_score'] = new_score
            updated_count += 1
    
    if updated_count > 0 or removed_count > 0:
        logger.debug(f"🔄 Sinyal güncelleme: {updated_count} güncellendi, {removed_count} silindi")


def get_signal_summary() -> Dict:
    """
    Sinyal özetini al
    
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
            'last_update': None
        }
    
    total_signals = len(current_data)
    long_count = len(current_data[current_data['run_type'] == 'long'])
    short_count = len(current_data[current_data['run_type'] == 'short'])
    avg_ai_score = current_data['ai_score'].mean()
    
    # En yüksek skorlu sembol
    top_signal = current_data.iloc[0] if not current_data.empty else None
    top_symbol = top_signal['symbol'] if top_signal is not None else None
    
    return {
        'total_signals': total_signals,
        'long_count': long_count,
        'short_count': short_count,
        'avg_ai_score': avg_ai_score,
        'top_symbol': top_symbol,
        'last_update': datetime.now(LOCAL_TZ).strftime('%H:%M:%S')
    }