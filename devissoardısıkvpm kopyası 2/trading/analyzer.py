"""
🔍 Sinyal Analiz Modülü
AI destekli kripto sinyal analizi ve batch processing
🔥 YENİ DEVISO RATIO ENTEGRASYONu
🔥 YENİ: Z-Score metrikleri eklendi
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
from core.indicators import compute_consecutive_metrics, get_deviso_detailed_analysis
from core.ai_model import ai_model

logger = logging.getLogger("crypto-analytics")


def analyze_symbol_with_ai(symbol: str, interval: str) -> Dict:
    """
    🔥 GÜNCELLENDI: C-Signal momentum + Z-Score metrikleri eklendi
        
    Returns:
        Dict: Analiz sonuçları veya boş dict
    """
    try:
        # Rate limiting
        time.sleep(REQ_SLEEP)
        
        # Veri çek
        df = fetch_klines(symbol, interval)
        if df is None or df.empty:
            logger.debug(f"❌ {symbol}: Veri çekilemedi")
            return {}
        
        if len(df) < 30:  # Minimum veri kontrolü gevşetildi (50->30)
            logger.debug(f"❌ {symbol}: Yetersiz veri ({len(df)} < 30)")
            return {}
        
        # Teknik metrikleri hesapla
        metrics = compute_consecutive_metrics(df)
        if not metrics or metrics['run_type'] == 'none':
            logger.debug(f"❌ {symbol}: Run type none veya metrik yok")
            return {}
        
        # Deviso ratio doğrulama - GEVŞETILDI
        deviso_ratio = metrics.get('deviso_ratio', 0.0)
        if abs(deviso_ratio) < 0.05:  # 0.1'den 0.05'e düşürüldü
            logger.debug(f"❌ {symbol}: Deviso ratio çok küçük ({deviso_ratio:.4f})")
            return {}
        
        # Trend yönü kontrolü - GEVŞETILDI
        streak_type = metrics['run_type']
        
        # Daha gevşek trend uyumu kontrolü
        direction_match = False
        trend_strength = abs(deviso_ratio)
        
        if streak_type == 'long' and deviso_ratio > 0.2:  # 0.5'ten 0.2'ye
            direction_match = True
        elif streak_type == 'short' and deviso_ratio < -0.2:  # -0.5'ten -0.2'ye
            direction_match = True
        
        # Trend uyumsuzluğunda sinyali reddet
        if not direction_match:
            logger.debug(f"❌ {symbol}: Trend uyumsuz - {streak_type} vs deviso {deviso_ratio:.4f}")
            return {}
        
        # Ek kalite kontrolleri - GEVŞETILDI
        run_count = metrics.get('run_count', 0)
        run_perc = metrics.get('run_perc')
        vol_ratio = metrics.get('vol_ratio')
        
        # Minimum streak kontrolü - GEVŞETILDI
        if run_count < 1:  # 2'den 1'e düşürüldü
            logger.debug(f"❌ {symbol}: Run count çok düşük ({run_count})")
            return {}
            
        # Minimum hareket kontrolü - GEVŞETILDI
        if run_perc is None or abs(run_perc) < 0.15:  # 0.3'ten 0.15'e
            logger.debug(f"❌ {symbol}: Run percentage çok düşük ({run_perc})")
            return {}
        
        # AI skoru hesapla (Z-Score ceza sistemi dahil)
        ai_score = ai_model.predict_score(metrics)
        
        # Minimum AI skoru kontrolü - GEVŞETILDI
        min_ai_threshold = (DEFAULT_MIN_AI_SCORE * 100) * 0.5  # %50 daha gevşek
        if ai_score < min_ai_threshold:
            logger.debug(f"❌ {symbol}: AI skoru düşük ({ai_score:.1f} < {min_ai_threshold})")
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
            logger.debug(f"⚠️ {symbol}: Deviso detaylı analiz hatası: {e}")
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
            'log_volume': metrics.get('log_volume'),
            'log_volume_momentum': metrics.get('log_volume_momentum'),
            'deviso_ratio': metrics['deviso_ratio'],
            'vpm_score': metrics.get('vpm_score', 0.0),  # 🔥 YENİ SATIR
            'c_signal_momentum': metrics.get('c_signal_momentum', 0.0),
            # 🔥 YENİ: Z-Score metrikleri eklendi
            'volume_zscore': metrics.get('volume_zscore', 0.0),
            'price_change_zscore': metrics.get('price_change_zscore', 0.0),
            'deviso_zscore': metrics.get('deviso_zscore', 0.0),
            'max_zscore': metrics.get('max_zscore', 0.0),
            'ai_score': ai_score,  
            'last_update': last_update.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
        }
        
        # Z-Score uyarısı logla
        max_zscore = metrics.get('max_zscore', 0.0)
        if max_zscore >= 2.0:
            logger.debug(f"⚠️ {symbol}: Z-Score uyarısı - Max: {max_zscore:.2f}, AI: {ai_score:.0f}%")
        
        logger.debug(f"✅ {symbol}: Sinyal onaylandı - AI:{ai_score:.0f}%, Deviso:{deviso_ratio:.2f}, Z-Score:{max_zscore:.2f}")
        return result
        
    except Exception as e:
        logger.warning(f"analyze_symbol error {symbol}: {e}")
        return {}

def batch_analyze_with_ai(interval: str) -> pd.DataFrame:
    """
    🔥 GELİŞTİRİLMİŞ: AI skoru düşen sinyaller silinmez, sadece skoru azalır
    Yeni deviso ratio hesaplaması ile daha doğru sonuçlar
    🔥 YENİ: Z-Score metrikleri dahil
    
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
    
    logger.info(f"🤖 {len(symbols)} sembol için YENİ DEVISO + Z-Score ile AI analiz başlatılıyor...")
    
    # Yeni analiz sonuçları
    fresh_results = []
    processed_count = 0
    deviso_success_count = 0
    zscore_penalty_count = 0  # 🔥 YENİ: Z-Score ceza sayacı
    
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
                    deviso_success_count += 1
                    
                    # Z-Score ceza kontrolü
                    if res.get('max_zscore', 0.0) >= 2.0:
                        zscore_penalty_count += 1
                    
                    # Kaydedilmiş sinyalleri güncelle
                    saved_signals[symbol] = {
                        'data': res,
                        'last_seen': datetime.now(LOCAL_TZ)
                    }
                
                # İlerleme logu - İyileştirildi
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    success_rate = (deviso_success_count / processed_count) * 100
                    logger.info(
                        f"🤖 İşlenen: {processed_count}/{len(symbols)} - Hız: {rate:.1f} s/sn - "
                        f"Deviso Başarı: {success_rate:.1f}% ({deviso_success_count}) - Z-Score Ceza: {zscore_penalty_count}"
                    )
            
            except Exception as e:
                logger.debug(f"Future hatası {symbol}: {e}")
    
    # Mevcut zaman
    current_time = datetime.now(LOCAL_TZ)
    fresh_symbols = {r['symbol'] for r in fresh_results}
    
    # Eski sinyalleri koruma ve skor düşürme - İyileştirildi
    protected_count = 0
    for symbol, saved_info in list(saved_signals.items()):
        minutes_old = (current_time - saved_info['last_seen']).total_seconds() / 60.0
        if minutes_old > 10:
            del saved_signals[symbol]
            continue
        
        if symbol not in fresh_symbols:
            old_data = saved_info['data'].copy()
            original_score = old_data['ai_score']
            
            # 🔥 YENİ: Deviso ratio bazlı ceza sistemi
            deviso_ratio = old_data.get('deviso_ratio', 0)
            trend_strength = abs(deviso_ratio)
            
            if trend_strength > 2.0:
                base_penalty = 10
            elif trend_strength > 1.0:
                base_penalty = 15
            else:
                base_penalty = 25
            
            if minutes_old <= 2:
                penalty = base_penalty
            elif minutes_old <= 5:
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
                f"(yaş: {minutes_old:.1f}dk, deviso: {deviso_ratio:.2f})"
            )
    
    # Performans istatistikleri - Geliştirildi
    elapsed_time = time.time() - start_time
    total_rate = len(symbols) / elapsed_time if elapsed_time > 0 else 0
    
    new_signals = len(fresh_symbols)
    total_signals = len(fresh_results)
    deviso_success_rate = (deviso_success_count / len(symbols)) * 100 if len(symbols) > 0 else 0
    zscore_penalty_rate = (zscore_penalty_count / deviso_success_count) * 100 if deviso_success_count > 0 else 0
    
    logger.info("✅ YENİ DEVISO + Z-Score AI Analiz tamamlandı:")
    logger.info(f"   🆕 Yeni sinyal: {new_signals}")
    logger.info(f"   📉 Korunan sinyal: {protected_count}")
    logger.info(f"   🎯 Toplam sinyal: {total_signals}")
    logger.info(f"   📊 Deviso başarı oranı: {deviso_success_rate:.1f}%")
    logger.info(f"   ⚠️ Z-Score ceza oranı: {zscore_penalty_rate:.1f}% ({zscore_penalty_count}/{deviso_success_count})")
    logger.info(f"   ⏱️ Süre: {elapsed_time:.1f}s - Hız: {total_rate:.1f} s/sn")
    
    if not fresh_results:
        logger.warning("⚠️ Hiç sinyal bulunamadı - filtreleri gözden geçirin")
        return pd.DataFrame()
        
    # DataFrame oluştur ve sırala - Geliştirildi
    df = pd.DataFrame(fresh_results)
    df = df.sort_values(
        by=['ai_score', 'trend_strength', 'run_perc', 'gauss_run', 'log_volume_momentum'],
        ascending=[False, False, False, False, False]
    )
    
    if len(df) > 0:
        top_signal = df.iloc[0]
        logger.info(
            f"🏆 En yüksek AI skoru: {top_signal['ai_score']:.0f}% - {top_signal['symbol']} "
            f"(Deviso: {top_signal['deviso_ratio']:.2f}, Z-Score: {top_signal.get('max_zscore', 0):.2f})"
        )
        
        trend_counts = df['trend_direction'].value_counts()
        logger.info(f"📈 Trend dağılımı: {dict(trend_counts)}")
        
        # Z-Score istatistikleri
        high_zscore_count = len(df[df['max_zscore'] >= 2.0])
        if high_zscore_count > 0:
            logger.info(f"⚠️ Yüksek Z-Score sinyalleri: {high_zscore_count}/{len(df)}")
        
        if protected_count > 0:
            logger.info("📉 Korunan sinyaller skor düşüşü ile aşağı kaydı")
            logger.debug("📊 Analyzer sonrası ilk 5 sinyal:")
        for i, row in df.head(5).iterrows():
            logger.debug(
                f"   {i}: {row['symbol']} | AI={row['ai_score']} | "
                f"Z-Score={row.get('max_zscore', 0):.2f} | Trend={row.get('trend_strength',0)}"
            )
    
    return df


def filter_signals(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Sinyalleri belirtilen filtrelere göre süz
    🔥 YENİ: Deviso ratio + Z-Score filtreleri eklendi
    
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
    
    if filters.get('min_ai_score', 0) > 0:
        filtered_df = filtered_df[filtered_df['ai_score'] >= filters['min_ai_score']]
    
    if filters.get('min_run_count', 0) > 0:
        filtered_df = filtered_df[filtered_df['run_count'] >= filters['min_run_count']]
    
    if filters.get('min_run_perc', 0) > 0:
        filtered_df = filtered_df[filtered_df['run_perc'] >= filters['min_run_perc']]
    
    if filters.get('min_vol_ratio', 0) > 0:
        filtered_df = filtered_df[(filtered_df['vol_ratio'].isna()) | 
                                 (filtered_df['vol_ratio'] >= filters['min_vol_ratio'])]
    
    if filters.get('min_deviso_strength', 0) > 0:
        filtered_df = filtered_df[filtered_df['trend_strength'] >= filters['min_deviso_strength']]
    
    # 🔥 YENİ: Z-Score filtreleri
    if filters.get('max_zscore', 0) > 0:
        filtered_df = filtered_df[filtered_df['max_zscore'] <= filters['max_zscore']]
    
    run_type_filter = filters.get('run_type')
    if run_type_filter and run_type_filter != 'all':
        filtered_df = filtered_df[filtered_df['run_type'] == run_type_filter]
    
    trend_filter = filters.get('trend_direction')
    if trend_filter and trend_filter != 'all':
        filtered_df = filtered_df[filtered_df['trend_direction'] == trend_filter]
    
    filtered_count = len(filtered_df)
    logger.info(f"🔍 Filtre sonucu: {filtered_count}/{original_count} sinyal kaldı")
    
    return filtered_df


def get_top_signals(df: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    """
    En iyi sinyalleri al
    🔥 YENİ: Deviso ratio + Z-Score da sıralama kriterinde
    
    Args:
        df (pd.DataFrame): Sinyal DataFrame'i
        count (int): Alınacak sinyal sayısı
        
    Returns:
        pd.DataFrame: En iyi sinyaller
    """
    if df.empty:
        return df
    
    sorted_df = df.sort_values(
        by=['ai_score', 'trend_strength', 'run_perc', 'gauss_run', 'log_volume_momentum'],
        ascending=[False, False, False, False, False]
    )
    
    return sorted_df.head(count)


def analyze_signal_quality(df: pd.DataFrame) -> Dict:
    """
    Sinyal kalitesi analizi
    🔥 YENİ: Deviso ratio + Z-Score kalite metrikleri eklendi
    
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
            'deviso_quality': {},
            'zscore_quality': {}
        }
    
    total_signals = len(df)
    avg_ai_score = df['ai_score'].mean()
    
    long_signals = len(df[df['run_type'] == 'long'])
    short_signals = len(df[df['run_type'] == 'short'])
    
    # Kalite kategorileri
    high_quality = len(df[df['ai_score'] >= 80])
    medium_quality = len(df[(df['ai_score'] >= 60) & (df['ai_score'] < 80)])
    low_quality = len(df[df['ai_score'] < 60])
    
    # 🔥 YENİ: Deviso kalite analizi
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
    
    # 🔥 YENİ: Z-Score kalite analizi
    zscore_quality = {}
    if 'max_zscore' in df.columns:
        avg_zscore = df['max_zscore'].mean()
        normal_signals = len(df[df['max_zscore'] < 2.0])
        warning_signals = len(df[(df['max_zscore'] >= 2.0) & (df['max_zscore'] < 3.0)])
        high_risk_signals = len(df[df['max_zscore'] >= 3.0])
        
        zscore_quality = {
            'avg_zscore': avg_zscore,
            'normal_signals': normal_signals,
            'warning_signals': warning_signals,
            'high_risk_signals': high_risk_signals,
            'risk_ratio': (warning_signals + high_risk_signals) / total_signals * 100 if total_signals > 0 else 0
        }
    
    # 🔥 YENİ: Trend direction dağılımı
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
        'zscore_quality': zscore_quality,  # 🔥 YENİ
        'trend_distribution': trend_distribution
    }


def update_signal_scores():
    """
    Kayıtlı sinyallerin skorlarını güncelle (yaş cezası)
    🔥 YENİ: Deviso bazlı ceza sistemi
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
            
            # 🔥 YENİ: Deviso gücüne göre ceza
            deviso_ratio = saved_info['data'].get('deviso_ratio', 0)
            trend_strength = abs(deviso_ratio)
            
            if trend_strength > 2.0:
                base_penalty = 15  # Güçlü trend, az ceza
            elif trend_strength > 1.0:
                base_penalty = 25
            else:
                base_penalty = 35  # Zayıf trend, çok ceza
            
            # Yaşa göre ek ceza
            if minutes_old <= 5:
                penalty = base_penalty
            elif minutes_old <= 10:
                penalty = base_penalty + 15
            else:
                penalty = base_penalty + 25
            
            new_score = max(1, original_score - penalty)
            saved_info['data']['ai_score'] = new_score
            updated_count += 1
    
    if updated_count > 0 or removed_count > 0:
        logger.debug(f"🔄 Sinyal güncelleme: {updated_count} güncellendi, {removed_count} silindi")


def get_signal_summary() -> Dict:
    """
    Sinyal özetini al
    🔥 YENİ: Deviso + Z-Score istatistikleri eklendi
    
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
            'deviso_stats': {},
            'zscore_stats': {}
        }
    
    total_signals = len(current_data)
    long_count = len(current_data[current_data['run_type'] == 'long'])
    short_count = len(current_data[current_data['run_type'] == 'short'])
    avg_ai_score = current_data['ai_score'].mean()
    
    # En yüksek skorlu sembol
    top_signal = current_data.iloc[0] if not current_data.empty else None
    top_symbol = top_signal['symbol'] if top_signal is not None else None
    
    # 🔥 YENİ: Deviso istatistikleri
    deviso_stats = {}
    if 'deviso_ratio' in current_data.columns and not current_data.empty:
        deviso_stats = {
            'avg_deviso_ratio': current_data['deviso_ratio'].mean(),
            'max_deviso_ratio': current_data['deviso_ratio'].max(),
            'min_deviso_ratio': current_data['deviso_ratio'].min(),
            'positive_deviso_count': len(current_data[current_data['deviso_ratio'] > 0]),
            'negative_deviso_count': len(current_data[current_data['deviso_ratio'] < 0])
        }
    
    # 🔥 YENİ: Z-Score istatistikleri
    zscore_stats = {}
    if 'max_zscore' in current_data.columns and not current_data.empty:
        zscore_stats = {
            'avg_max_zscore': current_data['max_zscore'].mean(),
            'high_zscore_count': len(current_data[current_data['max_zscore'] >= 2.0]),
            'extreme_zscore_count': len(current_data[current_data['max_zscore'] >= 3.0]),
            'normal_zscore_count': len(current_data[current_data['max_zscore'] < 2.0])
        }
    
    return {
        'total_signals': total_signals,
        'long_count': long_count,
        'short_count': short_count,
        'avg_ai_score': avg_ai_score,
        'top_symbol': top_symbol,
        'last_update': datetime.now(LOCAL_TZ).strftime('%H:%M:%S'),
        'deviso_stats': deviso_stats,
        'zscore_stats': zscore_stats  # 🔥 YENİ
    }


# 🔥 YENİ: Deviso spesifik analiz fonksiyonları
def analyze_deviso_trends(df: pd.DataFrame) -> Dict:
    """
    Deviso trend analizi
    
    Args:
        df (pd.DataFrame): Sinyal verileri
        
    Returns:
        Dict: Trend analizi sonuçları
    """
    if df.empty or 'deviso_ratio' not in df.columns:
        return {}
    
    strong_bullish = len(df[df['deviso_ratio'] >= 2.0])
    moderate_bullish = len(df[(df['deviso_ratio'] >= 0.5) & (df['deviso_ratio'] < 2.0)])
    neutral = len(df[(df['deviso_ratio'] >= -0.5) & (df['deviso_ratio'] < 0.5)])
    moderate_bearish = len(df[(df['deviso_ratio'] >= -2.0) & (df['deviso_ratio'] < -0.5)])
    strong_bearish = len(df[df['deviso_ratio'] < -2.0])
    
    return {
        'strong_bullish': strong_bullish,
        'moderate_bullish': moderate_bullish,
        'neutral': neutral,
        'moderate_bearish': moderate_bearish,
        'strong_bearish': strong_bearish,
        'bullish_dominance': (strong_bullish + moderate_bullish) > (strong_bearish + moderate_bearish)
    }


def get_deviso_correlation_analysis(df: pd.DataFrame) -> Dict:
    """
    Deviso ratio ile diğer göstergeler arasındaki korelasyon analizi
    
    Args:
        df (pd.DataFrame): Sinyal verileri
        
    Returns:
        Dict: Korelasyon analizi
    """
    if df.empty or len(df) < 3:
        return {}
    
    correlations = {}
    try:
        if all(col in df.columns for col in ['deviso_ratio', 'ai_score', 'run_perc', 'log_volume', 'log_volume_momentum']):
            correlations['deviso_ai_corr'] = df['deviso_ratio'].corr(df['ai_score'])
            correlations['deviso_run_corr'] = df['deviso_ratio'].corr(df['run_perc'])

            if df['log_volume'].notna().sum() > 2:
                correlations['deviso_logvol_corr'] = df['deviso_ratio'].corr(df['log_volume'])

            if df['log_volume_momentum'].notna().sum() > 2:
                correlations['deviso_logvolmom_corr'] = df['deviso_ratio'].corr(df['log_volume_momentum'])
                
    except Exception as e:
        logger.debug(f"Korelasyon hesaplama hatası: {e}")
    
    return correlations


# 🔥 YENİ: Z-Score analiz fonksiyonları
def analyze_zscore_distribution(df: pd.DataFrame) -> Dict:
    """
    Z-Score dağılım analizi
    
    Args:
        df (pd.DataFrame): Sinyal verileri
        
    Returns:
        Dict: Z-Score analizi
    """
    if df.empty or 'max_zscore' not in df.columns:
        return {}
    
    normal_count = len(df[df['max_zscore'] < 2.0])
    warning_count = len(df[(df['max_zscore'] >= 2.0) & (df['max_zscore'] < 3.0)])
    danger_count = len(df[df['max_zscore'] >= 3.0])
    
    total = len(df)
    
    return {
        'normal_count': normal_count,
        'warning_count': warning_count,
        'danger_count': danger_count,
        'normal_percentage': (normal_count / total * 100) if total > 0 else 0,
        'warning_percentage': (warning_count / total * 100) if total > 0 else 0,
        'danger_percentage': (danger_count / total * 100) if total > 0 else 0,
        'avg_zscore': df['max_zscore'].mean(),
        'max_zscore': df['max_zscore'].max()
    }