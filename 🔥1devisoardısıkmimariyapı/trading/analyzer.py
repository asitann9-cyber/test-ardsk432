"""
Sinyal Analiz Modülü
AI destekli kripto sinyal analizi ve batch processing
"""

import time
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from config import (
    LOCAL_TZ, MAX_WORKERS, REQ_SLEEP, DEFAULT_MIN_AI_SCORE
)
from data.fetch_data import fetch_klines, get_usdt_perp_symbols
from core.indicators import compute_consecutive_metrics, get_deviso_detailed_analysis
from core.ai_model import ai_model

logger = logging.getLogger("crypto-analytics")


def analyze_symbol_with_ai(symbol: str, interval: str) -> Dict:
    """
    Tek sembol analizi - AI destekli
    
    Returns:
        Dict: Analiz sonuçları veya boş dict
    """
    try:
        time.sleep(REQ_SLEEP)
        
        # Veri çek
        df = fetch_klines(symbol, interval)
        if df is None or df.empty:
            return {}
        
        if len(df) < 30:
            return {}
        
        # Teknik metrikleri hesapla
        metrics = compute_consecutive_metrics(df)
        if not metrics or metrics['run_type'] == 'none':
            return {}
        
        # Deviso ratio doğrulama
        deviso_ratio = metrics.get('deviso_ratio', 0.0)
        if abs(deviso_ratio) < 0.05:
            return {}
        
        # Trend yönü kontrolü
        streak_type = metrics['run_type']
        direction_match = False
        trend_strength = abs(deviso_ratio)
        
        if streak_type == 'long' and deviso_ratio > 0.2:
            direction_match = True
        elif streak_type == 'short' and deviso_ratio < -0.2:
            direction_match = True
        
        if not direction_match:
            return {}
        
        # Kalite kontrolleri
        run_count = metrics.get('run_count', 0)
        run_perc = metrics.get('run_perc')
        
        if run_count < 1:
            return {}
            
        if run_perc is None or abs(run_perc) < 0.15:
            return {}
        
        # AI skoru hesapla
        ai_score = ai_model.predict_score(metrics)
        
        # AI skor normalizasyonu - SORUNUN ANA ÇÖZÜMÜ
        if ai_score <= 1.0:
            ai_score = ai_score * 100.0
        
        min_ai_threshold = (DEFAULT_MIN_AI_SCORE * 100) * 0.5
        if ai_score < min_ai_threshold:
            return {}
        
        # Son fiyat ve zaman bilgisi
        last_row = df.iloc[-1]
        last_close = float(last_row['close'])
        last_update = last_row['close_time']
        
        # Deviso detaylı analizi
        try:
            deviso_details = get_deviso_detailed_analysis(df)
            trend_direction = deviso_details.get('trend_direction', 'Belirsiz')
        except Exception as e:
            trend_direction = 'Belirsiz'
        
        # Başarılı sinyal
        result = {
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
            'ai_score': ai_score,  # Normalize edilmiş 0-100 aralığında
            'last_update': last_update.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
        }
        
        return result
        
    except Exception as e:
        logger.warning(f"analyze_symbol error {symbol}: {e}")
        return {}


def batch_analyze_with_ai(interval: str) -> pd.DataFrame:
    """
    Toplu analiz - AI skoru normalize edilmiş
    
    Args:
        interval (str): Analiz edilecek zaman dilimi
        
    Returns:
        pd.DataFrame: Analiz sonuçları
    """
    start_time = time.time()
    
    # Sembol listesini al
    symbols = get_usdt_perp_symbols()
    if not symbols:
        logger.error("Sembol listesi boş!")
        return pd.DataFrame()
    
    logger.info(f"AI analiz başlatılıyor: {len(symbols)} sembol")
    
    # Yeni analiz sonuçları
    fresh_results = []
    processed_count = 0
    success_count = 0
    
    # Paralel işleme
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(analyze_symbol_with_ai, sym, interval): sym for sym in symbols}
        
        for fut in as_completed(futures):
            symbol = futures[fut]
            processed_count += 1
            
            try:
                res = fut.result()
                if res:  # Geçerli sinyal
                    # AI skor normalizasyonu kontrolü
                    if res['ai_score'] <= 1.0:
                        res['ai_score'] = res['ai_score'] * 100.0
                    
                    fresh_results.append(res)
                    success_count += 1
                    
                # İlerleme logu
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    success_rate = (success_count / processed_count) * 100
                    logger.info(f"İşlenen: {processed_count}/{len(symbols)} - Başarı: {success_rate:.1f}%")
                    
            except Exception as e:
                logger.debug(f"Future hatası {symbol}: {e}")

    # Performans istatistikleri
    elapsed_time = time.time() - start_time
    total_rate = len(symbols) / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"AI Analiz tamamlandı:")
    logger.info(f"  Yeni sinyal: {success_count}")
    logger.info(f"  Süre: {elapsed_time:.1f}s - Hız: {total_rate:.1f} s/sn")
    
    if not fresh_results:
        logger.warning("Hiç sinyal bulunamadı")
        return pd.DataFrame()
    
    # DataFrame oluştur ve sırala
    df = pd.DataFrame(fresh_results)
    
    # AI skor normalizasyonu son kontrol
    if 'ai_score' in df.columns:
        if df['ai_score'].max() <= 1.0:
            df['ai_score'] = (df['ai_score'] * 100.0).clip(0, 100)
    
    # Sıralama
    df = df.sort_values(
        by=['ai_score', 'trend_strength', 'run_perc', 'gauss_run', 'vol_ratio'], 
        ascending=[False, False, False, False, False]
    )
    
    if len(df) > 0:
        top_signal = df.iloc[0]
        logger.info(f"En yüksek AI skoru: {top_signal['ai_score']:.0f}% - {top_signal['symbol']}")
    
    # Config'e güncelle
    config.current_data = df.copy()
    
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
    
    # Deviso ratio filtresi
    min_deviso_strength = filters.get('min_deviso_strength', 0)
    if min_deviso_strength > 0:
        filtered_df = filtered_df[filtered_df['trend_strength'] >= min_deviso_strength]
    
    # Run type filtresi
    run_type_filter = filters.get('run_type')
    if run_type_filter and run_type_filter != 'all':
        filtered_df = filtered_df[filtered_df['run_type'] == run_type_filter]
    
    # Trend direction filtresi
    trend_filter = filters.get('trend_direction')
    if trend_filter and trend_filter != 'all':
        filtered_df = filtered_df[filtered_df['trend_direction'] == trend_filter]
    
    filtered_count = len(filtered_df)
    logger.info(f"Filtre sonucu: {filtered_count}/{original_count} sinyal kaldı")
    
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
    
    # Sıralama: AI skoru > Trend gücü > Run percentage > Gauss run > Volume ratio
    sorted_df = df.sort_values(
        by=['ai_score', 'trend_strength', 'run_perc', 'gauss_run', 'vol_ratio'],
        ascending=[False, False, False, False, False]
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
            'quality_distribution': {},
            'deviso_quality': {}
        }
    
    total_signals = len(df)
    avg_ai_score = df['ai_score'].mean()
    
    long_signals = len(df[df['run_type'] == 'long'])
    short_signals = len(df[df['run_type'] == 'short'])
    
    # Kalite kategorileri
    high_quality = len(df[df['ai_score'] >= 80])
    medium_quality = len(df[(df['ai_score'] >= 60) & (df['ai_score'] < 80)])
    low_quality = len(df[df['ai_score'] < 60])
    
    # Deviso kalite analizi
    deviso_quality = {}
    if 'trend_strength' in df.columns:
        avg_trend_strength = df['trend_strength'].mean()
        strong_trends = len(df[df['trend_strength'] >= 2.0])
        medium_trends = len(df[(df['trend_strength'] >= 1.0) & (df['trend_strength'] < 2.0)])
        weak_trends = len(df[df['trend_strength'] < 1.0])
        
        deviso_quality = {
            'avg_trend_strength': avg_trend_strength,
            'strong_trends': strong_trends,
            'medium_trends': medium_trends,
            'weak_trends': weak_trends
        }
    
    # Trend direction dağılımı
    trend_distribution = {}
    if 'trend_direction' in df.columns:
        trend_distribution = df['trend_direction'].value_counts().to_dict()
    
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
        'deviso_quality': deviso_quality,
        'trend_distribution': trend_distribution
    }


def get_signal_summary() -> Dict:
    """
    Sinyal özetini al
    
    Returns:
        Dict: Sinyal özet bilgileri
    """
    if config.current_data is None or config.current_data.empty:
        return {
            'total_signals': 0,
            'long_count': 0,
            'short_count': 0,
            'avg_ai_score': 0,
            'top_symbol': None,
            'last_update': None,
            'deviso_stats': {}
        }
    
    df = config.current_data
    total_signals = len(df)
    long_count = len(df[df['run_type'] == 'long'])
    short_count = len(df[df['run_type'] == 'short'])
    avg_ai_score = df['ai_score'].mean()
    
    # En yüksek skorlu sembol
    top_signal = df.iloc[0] if not df.empty else None
    top_symbol = top_signal['symbol'] if top_signal is not None else None
    
    # Deviso istatistikleri
    deviso_stats = {}
    if 'deviso_ratio' in df.columns and not df.empty:
        deviso_stats = {
            'avg_deviso_ratio': df['deviso_ratio'].mean(),
            'max_deviso_ratio': df['deviso_ratio'].max(),
            'min_deviso_ratio': df['deviso_ratio'].min(),
            'positive_deviso_count': len(df[df['deviso_ratio'] > 0]),
            'negative_deviso_count': len(df[df['deviso_ratio'] < 0])
        }
    
    return {
        'total_signals': total_signals,
        'long_count': long_count,
        'short_count': short_count,
        'avg_ai_score': avg_ai_score,
        'top_symbol': top_symbol,
        'last_update': datetime.now(LOCAL_TZ).strftime('%H:%M:%S'),
        'deviso_stats': deviso_stats
    }