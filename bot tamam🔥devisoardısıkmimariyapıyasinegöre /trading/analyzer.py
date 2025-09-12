"""
🔍 Sinyal Analiz Modülü
AI destekli kripto sinyal analizi ve batch processing
🔥 YENİ CVD + ROC Momentum sistemi entegrasyonu
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
from core.indicators import compute_cvd_momentum_metrics, get_deviso_detailed_analysis  # 🔥 YENİ: CVD sistemi
from core.ai_model import ai_model

logger = logging.getLogger("crypto-analytics")


def analyze_symbol_with_ai(symbol: str, interval: str) -> Dict:
    """
    CVD + ROC Momentum bazlı sembol analizi
    
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
        
        if len(df) < 30:  # Minimum veri kontrolü
            logger.debug(f"❌ {symbol}: Yetersiz veri ({len(df)} < 30)")
            return {}
        
        # 🔥 YENİ: CVD momentum metriklerini hesapla
        metrics = compute_cvd_momentum_metrics(df)
        if not metrics or metrics.get('signal_type') == 'neutral':
            logger.debug(f"❌ {symbol}: CVD sinyal neutral veya metrik yok")
            return {}
        
        # 🔥 YENİ: CVD momentum doğrulama
        cvd_roc_momentum = metrics.get('cvd_roc_momentum', 0.0)
        momentum_strength = metrics.get('momentum_strength', 0.0)
        
        if momentum_strength < 20.0:  # Minimum momentum gücü
            logger.debug(f"❌ {symbol}: CVD momentum çok zayıf ({momentum_strength:.1f})")
            return {}
        
        # 🔥 YENİ: Deviso-CVD uyum kontrolü
        deviso_cvd_harmony = metrics.get('deviso_cvd_harmony', 50.0)
        if deviso_cvd_harmony < 40.0:  # Minimum uyum skoru
            logger.debug(f"❌ {symbol}: Deviso-CVD uyumu zayıf ({deviso_cvd_harmony:.1f})")
            return {}
        
        # 🔥 YENİ: CVD yön kontrolü
        cvd_direction = metrics.get('cvd_direction', 'neutral')
        signal_type = metrics.get('signal_type', 'neutral')
        deviso_ratio = metrics.get('deviso_ratio', 0.0)
        
        # Trend uyumu kontrolü
        direction_match = False
        if ((signal_type in ['long', 'strong_long'] and deviso_ratio > 0.5) or 
            (signal_type in ['short', 'strong_short'] and deviso_ratio < -0.5)):
            direction_match = True
        
        if not direction_match:
            logger.debug(f"❌ {symbol}: Trend uyumsuz - {signal_type} vs deviso {deviso_ratio:.4f}")
            return {}
        
        # 🔥 YENİ: Volume pressure kontrolü
        buy_pressure = metrics.get('buy_pressure', 50.0)
        sell_pressure = metrics.get('sell_pressure', 50.0)
        pressure_dominance = abs(buy_pressure - sell_pressure)
        
        if pressure_dominance < 10.0:  # Minimum baskı farkı
            logger.debug(f"❌ {symbol}: Volume pressure çok zayıf ({pressure_dominance:.1f})")
            return {}
        
        # AI skoru hesapla
        ai_score = ai_model.predict_score(metrics)
        
        # Minimum AI skoru kontrolü
        min_ai_threshold = DEFAULT_MIN_AI_SCORE * 100 * 0.7  # %30 daha gevşek
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
        
        # 🔥 YENİ: CVD bazlı sonuç
        result = {
            'symbol': symbol,
            'timeframe': interval,
            'last_close': last_close,
            
            # 🔥 CVD MOMENTUM METRİKLERİ
            'cvd_roc_momentum': cvd_roc_momentum,
            'cvd_direction': cvd_direction,
            'momentum_strength': momentum_strength,
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            'deviso_cvd_harmony': deviso_cvd_harmony,
            'trend_strength': metrics.get('trend_strength', 0.0),
            'signal_type': signal_type,
            
            # 🔧 ESKİ METRİKLER (Backward compatibility)
            'run_type': signal_type,
            'run_count': int(momentum_strength / 10),  # Emülasyon
            'run_perc': abs(cvd_roc_momentum) / 5.0,  # Emülasyon
            'gauss_run': metrics.get('trend_strength', 0.0),
            'gauss_run_perc': metrics.get('trend_strength', 0.0),
            'vol_ratio': 2.0,  # Sabit değer (CVD pressure ile değiştirildi)
            'hh_vol_streak': int(pressure_dominance / 10),  # Emülasyon
            
            # 🔧 DEVİSO RATIO (Korundu)
            'deviso_ratio': deviso_ratio,
            
            # 🔥 DİĞER BİLGİLER
            'ai_score': ai_score,
            'last_update': last_update.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'trend_direction': trend_direction,
        }
        
        logger.debug(f"✅ {symbol}: CVD sinyal onaylandı - AI:{ai_score:.0f}%, CVD:{cvd_direction}, Momentum:{momentum_strength:.1f}, Harmony:{deviso_cvd_harmony:.1f}")
        return result
        
    except Exception as e:
        logger.warning(f"analyze_symbol error {symbol}: {e}")
        return {}


def batch_analyze_with_ai(interval: str) -> pd.DataFrame:
    """
    🔥 YENİ: CVD + ROC Momentum sistemi ile batch analizi
    
    Args:
        interval (str): Analiz edilecek zaman dilimi
        
    Returns:
        pd.DataFrame: CVD analiz sonuçları
    """
    global saved_signals
    
    start_time = time.time()
    
    # Sembol listesini al
    symbols = get_usdt_perp_symbols()
    if not symbols:
        logger.error("Sembol listesi boş!")
        return pd.DataFrame()
    
    logger.info(f"🔥 {len(symbols)} sembol için CVD + ROC Momentum analizi başlatılıyor...")
    
    # Yeni analiz sonuçları
    fresh_results = []
    processed_count = 0
    cvd_success_count = 0  # 🔥 CVD başarı sayacı
    
    # Paralel işleme
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(analyze_symbol_with_ai, sym, interval): sym for sym in symbols}
        
        for fut in as_completed(futures):
            symbol = futures[fut]
            processed_count += 1
            
            try:
                res = fut.result()
                if res:  # Geçerli CVD sinyal
                    fresh_results.append(res)
                    cvd_success_count += 1
                    
                    # Kaydedilmiş sinyalleri güncelle
                    saved_signals[symbol] = {
                        'data': res,
                        'last_seen': datetime.now(LOCAL_TZ)
                    }
                    
                # İlerleme logu
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    success_rate = (cvd_success_count / processed_count) * 100
                    logger.info(f"🔥 İşlenen: {processed_count}/{len(symbols)} - Hız: {rate:.1f} s/sn - CVD Başarı: {success_rate:.1f}% ({cvd_success_count})")
                    
            except Exception as e:
                logger.debug(f"Future hatası {symbol}: {e}")

    # Mevcut zaman
    current_time = datetime.now(LOCAL_TZ)
    fresh_symbols = {r['symbol'] for r in fresh_results} 
    
    # Eski sinyalleri koruma ve skor düşürme - CVD bazlı
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
            
            # 🔥 CVD momentum bazlı ceza sistemi
            momentum_strength = old_data.get('momentum_strength', 0.0)
            cvd_harmony = old_data.get('deviso_cvd_harmony', 50.0)
            
            # CVD gücüne göre ceza
            if momentum_strength > 70.0 and cvd_harmony > 80.0:
                base_penalty = 8   # Çok güçlü CVD, az ceza
            elif momentum_strength > 50.0 and cvd_harmony > 65.0:
                base_penalty = 15  # İyi CVD, orta ceza
            else:
                base_penalty = 25  # Zayıf CVD, fazla ceza
            
            # Yaşa göre ek ceza
            if minutes_old <= 2:
                penalty = base_penalty
            elif minutes_old <= 5:
                penalty = base_penalty + 10
            else:
                penalty = base_penalty + 20
            
            new_score = max(5, original_score - penalty)
            old_data['ai_score'] = new_score
            old_data['score_status'] = f"📉-{penalty}"
            
            # Eski sinyali korunmuş listesine ekle
            fresh_results.append(old_data)
            protected_count += 1
            
            logger.debug(f"📉 {symbol}: {original_score:.0f} → {new_score:.0f} (yaş: {minutes_old:.1f}dk, CVD güç: {momentum_strength:.1f})")

    # Performans istatistikleri
    elapsed_time = time.time() - start_time
    total_rate = len(symbols) / elapsed_time if elapsed_time > 0 else 0
    
    new_signals = len(fresh_symbols)
    total_signals = len(fresh_results)
    cvd_success_rate = (cvd_success_count / len(symbols)) * 100 if len(symbols) > 0 else 0
    
    logger.info(f"✅ CVD + ROC Momentum Analizi tamamlandı:")
    logger.info(f"   🆕 Yeni CVD sinyal: {new_signals}")
    logger.info(f"   📉 Korunan sinyal: {protected_count}")
    logger.info(f"   🎯 Toplam sinyal: {total_signals}")
    logger.info(f"   📊 CVD başarı oranı: {cvd_success_rate:.1f}%")
    logger.info(f"   ⏱️ Süre: {elapsed_time:.1f}s - Hız: {total_rate:.1f} s/sn")
    
    if not fresh_results:
        logger.warning("⚠️ Hiç CVD sinyal bulunamadı - filtreleri gözden geçirin")
        return pd.DataFrame()
        
    # DataFrame oluştur ve sırala
    df = pd.DataFrame(fresh_results)
    
    # 🔥 YENİ: CVD momentum bazlı sıralama
    df = df.sort_values(
        by=['ai_score', 'momentum_strength', 'deviso_cvd_harmony', 'trend_strength', 'deviso_ratio'], 
        ascending=[False, False, False, False, False]
    )
    
    if len(df) > 0:
        top_signal = df.iloc[0]
        logger.info(f"🏆 En yüksek CVD sinyal: {top_signal['ai_score']:.0f}% - {top_signal['symbol']} (CVD:{top_signal['cvd_direction']}, Momentum:{top_signal['momentum_strength']:.1f}, Harmony:{top_signal['deviso_cvd_harmony']:.1f})")
        
        # CVD trend dağılımı
        cvd_direction_counts = df['cvd_direction'].value_counts()
        logger.info(f"📈 CVD trend dağılımı: {dict(cvd_direction_counts)}")
        
        # CVD momentum dağılımı
        strong_momentum = len(df[df['momentum_strength'] >= 70])
        medium_momentum = len(df[(df['momentum_strength'] >= 40) & (df['momentum_strength'] < 70)])
        weak_momentum = len(df[df['momentum_strength'] < 40])
        logger.info(f"💪 CVD momentum dağılımı: Güçlü={strong_momentum}, Orta={medium_momentum}, Zayıf={weak_momentum}")
        
        if protected_count > 0:
            logger.info(f"📉 Korunan sinyaller CVD bazlı skor düşüşü ile aşağı kaydı")
        
        logger.debug("📊 Analyzer CVD sonrası ilk 5 sinyal:")
        for i, row in df.head(5).iterrows():
            logger.debug(f"   {i}: {row['symbol']} | AI={row['ai_score']} | CVD={row.get('cvd_direction','neutral')} | Momentum={row.get('momentum_strength',0):.1f} | Harmony={row.get('deviso_cvd_harmony',0):.1f}")

    return df


def filter_signals(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    🔥 YENİ: CVD momentum bazlı sinyal filtreleme
    
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
    
    # 🔥 YENİ: CVD momentum filtresi
    min_momentum_strength = filters.get('min_momentum_strength', 0)
    if min_momentum_strength > 0:
        filtered_df = filtered_df[filtered_df['momentum_strength'] >= min_momentum_strength]
    
    # 🔥 YENİ: CVD-Deviso harmony filtresi
    min_harmony = filters.get('min_deviso_cvd_harmony', 0)
    if min_harmony > 0:
        filtered_df = filtered_df[filtered_df['deviso_cvd_harmony'] >= min_harmony]
    
    # 🔥 YENİ: Volume pressure filtresi
    min_pressure_dominance = filters.get('min_pressure_dominance', 0)
    if min_pressure_dominance > 0:
        pressure_dominance = abs(filtered_df['buy_pressure'] - filtered_df['sell_pressure'])
        filtered_df = filtered_df[pressure_dominance >= min_pressure_dominance]
    
    # 🔥 YENİ: CVD direction filtresi
    cvd_direction_filter = filters.get('cvd_direction')
    if cvd_direction_filter and cvd_direction_filter != 'all':
        filtered_df = filtered_df[filtered_df['cvd_direction'] == cvd_direction_filter]
    
    # Signal type filtresi
    signal_type_filter = filters.get('signal_type')
    if signal_type_filter and signal_type_filter != 'all':
        filtered_df = filtered_df[filtered_df['signal_type'] == signal_type_filter]
    
    # 🔧 ESKİ ALANLAR (Backward compatibility)
    # Run count filtresi (emulated from momentum_strength)
    min_run_count = filters.get('min_run_count', 0)
    if min_run_count > 0:
        filtered_df = filtered_df[filtered_df['run_count'] >= min_run_count]
    
    # Run percentage filtresi (emulated from CVD ROC)
    min_run_perc = filters.get('min_run_perc', 0)
    if min_run_perc > 0:
        filtered_df = filtered_df[filtered_df['run_perc'] >= min_run_perc]
    
    # Volume ratio filtresi (her zaman geçer - CVD pressure ile değiştirildi)
    min_vol_ratio = filters.get('min_vol_ratio', 0)
    if min_vol_ratio > 0:
        filtered_df = filtered_df[(filtered_df['vol_ratio'].isna()) | 
                                 (filtered_df['vol_ratio'] >= min_vol_ratio)]
    
    filtered_count = len(filtered_df)
    logger.info(f"🔍 CVD filtre sonucu: {filtered_count}/{original_count} sinyal kaldı")
    
    return filtered_df


def get_top_signals(df: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    """
    🔥 YENİ: CVD momentum bazlı en iyi sinyaller
    
    Args:
        df (pd.DataFrame): Sinyal DataFrame'i
        count (int): Alınacak sinyal sayısı
        
    Returns:
        pd.DataFrame: En iyi CVD sinyaller
    """
    if df.empty:
        return df
    
    # CVD bazlı sıralama: AI skoru > CVD momentum > Harmony > Trend gücü
    sorted_df = df.sort_values(
        by=['ai_score', 'momentum_strength', 'deviso_cvd_harmony', 'trend_strength'],
        ascending=[False, False, False, False]
    )
    
    return sorted_df.head(count)


def analyze_signal_quality(df: pd.DataFrame) -> Dict:
    """
    🔥 YENİ: CVD momentum bazlı sinyal kalitesi analizi
    
    Args:
        df (pd.DataFrame): Analiz edilecek sinyaller
        
    Returns:
        Dict: CVD kalite metrikleri
    """
    if df.empty:
        return {
            'total_signals': 0,
            'avg_ai_score': 0,
            'bullish_signals': 0,
            'bearish_signals': 0,
            'neutral_signals': 0,
            'high_quality_signals': 0,
            'quality_distribution': {},
            'cvd_quality': {}
        }
    
    total_signals = len(df)
    avg_ai_score = df['ai_score'].mean()
    
    # CVD direction bazlı dağılım
    bullish_signals = len(df[df['cvd_direction'] == 'bullish'])
    bearish_signals = len(df[df['cvd_direction'] == 'bearish'])
    neutral_signals = len(df[df['cvd_direction'] == 'neutral'])
    
    # Kalite kategorileri
    high_quality = len(df[df['ai_score'] >= 80])
    medium_quality = len(df[(df['ai_score'] >= 60) & (df['ai_score'] < 80)])
    low_quality = len(df[df['ai_score'] < 60])
    
    # 🔥 YENİ: CVD kalite analizi
    cvd_quality = {}
    if 'momentum_strength' in df.columns:
        avg_momentum_strength = df['momentum_strength'].mean()
        strong_momentum = len(df[df['momentum_strength'] >= 70])
        medium_momentum = len(df[(df['momentum_strength'] >= 40) & (df['momentum_strength'] < 70)])
        weak_momentum = len(df[df['momentum_strength'] < 40])
        
        avg_harmony = df['deviso_cvd_harmony'].mean() if 'deviso_cvd_harmony' in df.columns else 0
        high_harmony = len(df[df['deviso_cvd_harmony'] >= 80]) if 'deviso_cvd_harmony' in df.columns else 0
        
        cvd_quality = {
            'avg_momentum_strength': avg_momentum_strength,
            'strong_momentum': strong_momentum,
            'medium_momentum': medium_momentum,
            'weak_momentum': weak_momentum,
            'avg_harmony': avg_harmony,
            'high_harmony': high_harmony
        }
    
    # Signal type dağılımı
    signal_type_distribution = {}
    if 'signal_type' in df.columns:
        signal_type_distribution = df['signal_type'].value_counts().to_dict()
    
    return {
        'total_signals': total_signals,
        'avg_ai_score': avg_ai_score,
        'bullish_signals': bullish_signals,
        'bearish_signals': bearish_signals,
        'neutral_signals': neutral_signals,
        'high_quality_signals': high_quality,
        'quality_distribution': {
            'high': high_quality,
            'medium': medium_quality,
            'low': low_quality
        },
        'bullish_ratio': (bullish_signals / total_signals * 100) if total_signals > 0 else 0,
        'bearish_ratio': (bearish_signals / total_signals * 100) if total_signals > 0 else 0,
        'cvd_quality': cvd_quality,
        'signal_type_distribution': signal_type_distribution
    }


def update_signal_scores():
    """
    🔥 YENİ: CVD momentum bazlı sinyal skoru güncelleme
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
            
            # CVD momentum gücüne göre ceza
            momentum_strength = saved_info['data'].get('momentum_strength', 0.0)
            cvd_harmony = saved_info['data'].get('deviso_cvd_harmony', 50.0)
            
            if momentum_strength > 70.0 and cvd_harmony > 80.0:
                base_penalty = 10  # Güçlü CVD, az ceza
            elif momentum_strength > 50.0:
                base_penalty = 20  # Orta CVD
            else:
                base_penalty = 30  # Zayıf CVD, fazla ceza
            
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
        logger.debug(f"🔄 CVD sinyal güncelleme: {updated_count} güncellendi, {removed_count} silindi")


def get_signal_summary() -> Dict:
    """
    🔥 YENİ: CVD momentum bazlı sinyal özeti
    
    Returns:
        Dict: CVD sinyal özet bilgileri
    """
    global current_data
    
    if current_data is None or current_data.empty:
        return {
            'total_signals': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'avg_ai_score': 0,
            'avg_momentum_strength': 0,
            'top_symbol': None,
            'last_update': None,
            'cvd_stats': {}
        }
    
    total_signals = len(current_data)
    
    # CVD direction bazlı sayım
    bullish_count = len(current_data[current_data['cvd_direction'] == 'bullish'])
    bearish_count = len(current_data[current_data['cvd_direction'] == 'bearish'])
    neutral_count = len(current_data[current_data['cvd_direction'] == 'neutral'])
    
    avg_ai_score = current_data['ai_score'].mean()
    avg_momentum_strength = current_data['momentum_strength'].mean() if 'momentum_strength' in current_data.columns else 0
    
    # En yüksek skorlu sembol
    top_signal = current_data.iloc[0] if not current_data.empty else None
    top_symbol = top_signal['symbol'] if top_signal is not None else None
    
    # CVD istatistikleri
    cvd_stats = {}
    if not current_data.empty:
        cvd_stats = {
            'avg_cvd_roc_momentum': current_data['cvd_roc_momentum'].mean() if 'cvd_roc_momentum' in current_data.columns else 0,
            'avg_deviso_cvd_harmony': current_data['deviso_cvd_harmony'].mean() if 'deviso_cvd_harmony' in current_data.columns else 0,
            'strong_momentum_count': len(current_data[current_data['momentum_strength'] >= 70]) if 'momentum_strength' in current_data.columns else 0,
            'high_harmony_count': len(current_data[current_data['deviso_cvd_harmony'] >= 80]) if 'deviso_cvd_harmony' in current_data.columns else 0
        }
    
    return {
        'total_signals': total_signals,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'neutral_count': neutral_count,
        'avg_ai_score': avg_ai_score,
        'avg_momentum_strength': avg_momentum_strength,
        'top_symbol': top_symbol,
        'last_update': datetime.now(LOCAL_TZ).strftime('%H:%M:%S'),
        'cvd_stats': cvd_stats
    }


# 🔥 YENİ: CVD spesifik analiz fonksiyonları
def analyze_cvd_momentum_trends(df: pd.DataFrame) -> Dict:
    """
    CVD momentum trend analizi
    
    Args:
        df (pd.DataFrame): Sinyal verileri
        
    Returns:
        Dict: CVD trend analizi sonuçları
    """
    if df.empty or 'cvd_roc_momentum' not in df.columns:
        return {}
    
    strong_bullish_momentum = len(df[(df['cvd_roc_momentum'] > 10) & (df['cvd_direction'] == 'bullish')])
    moderate_bullish_momentum = len(df[(df['cvd_roc_momentum'] > 5) & (df['cvd_roc_momentum'] <= 10) & (df['cvd_direction'] == 'bullish')])
    weak_momentum = len(df[abs(df['cvd_roc_momentum']) <= 5])
    moderate_bearish_momentum = len(df[(df['cvd_roc_momentum'] < -5) & (df['cvd_roc_momentum'] >= -10) & (df['cvd_direction'] == 'bearish')])
    strong_bearish_momentum = len(df[(df['cvd_roc_momentum'] < -10) & (df['cvd_direction'] == 'bearish')])
    
    return {
        'strong_bullish_momentum': strong_bullish_momentum,
        'moderate_bullish_momentum': moderate_bullish_momentum,
        'weak_momentum': weak_momentum,
        'moderate_bearish_momentum': moderate_bearish_momentum,
        'strong_bearish_momentum': strong_bearish_momentum,
        'bullish_dominance': (strong_bullish_momentum + moderate_bullish_momentum) > (strong_bearish_momentum + moderate_bearish_momentum)
    }


def get_cvd_correlation_analysis(df: pd.DataFrame) -> Dict:
    """
    CVD metrikleri arasındaki korelasyon analizi
    
    Args:
        df (pd.DataFrame): Sinyal verileri
        
    Returns:
        Dict: CVD korelasyon analizi
    """
    if df.empty or len(df) < 3:
        return {}
    
    correlations = {}
    try:
        cvd_columns = ['cvd_roc_momentum', 'momentum_strength', 'deviso_cvd_harmony', 'ai_score', 'deviso_ratio']
        available_columns = [col for col in cvd_columns if col in df.columns]
        
        if len(available_columns) >= 2:
            # CVD ROC ile diğer metrikler arasında korelasyon
            if 'cvd_roc_momentum' in df.columns and 'ai_score' in df.columns:
                correlations['cvd_roc_ai_corr'] = df['cvd_roc_momentum'].corr(df['ai_score'])
            
            if 'momentum_strength' in df.columns and 'ai_score' in df.columns:
                correlations['momentum_ai_corr'] = df['momentum_strength'].corr(df['ai_score'])
            
            if 'deviso_cvd_harmony' in df.columns and 'ai_score' in df.columns:
                correlations['harmony_ai_corr'] = df['deviso_cvd_harmony'].corr(df['ai_score'])
            
            if 'cvd_roc_momentum' in df.columns and 'deviso_ratio' in df.columns:
                correlations['cvd_deviso_corr'] = df['cvd_roc_momentum'].corr(df['deviso_ratio'])
                
    except Exception as e:
        logger.debug(f"CVD korelasyon hesaplama hatası: {e}")
    
    return correlations


def get_cvd_volume_analysis(df: pd.DataFrame) -> Dict:
    """
    CVD volume pressure analizi
    
    Args:
        df (pd.DataFrame): Sinyal verileri
        
    Returns:
        Dict: Volume pressure analizi
    """
    if df.empty:
        return {}
    
    volume_analysis = {}
    try:
        if 'buy_pressure' in df.columns and 'sell_pressure' in df.columns:
            # Ortalama pressure'lar
            avg_buy_pressure = df['buy_pressure'].mean()
            avg_sell_pressure = df['sell_pressure'].mean()
            
            # Baskın pressure türü
            buy_dominant_count = len(df[df['buy_pressure'] > 60])
            sell_dominant_count = len(df[df['sell_pressure'] > 60])
            neutral_pressure_count = len(df) - buy_dominant_count - sell_dominant_count
            
            # Extreme pressure durumları
            extreme_buy_count = len(df[df['buy_pressure'] > 80])
            extreme_sell_count = len(df[df['sell_pressure'] > 80])
            
            volume_analysis = {
                'avg_buy_pressure': avg_buy_pressure,
                'avg_sell_pressure': avg_sell_pressure,
                'buy_dominant_count': buy_dominant_count,
                'sell_dominant_count': sell_dominant_count,
                'neutral_pressure_count': neutral_pressure_count,
                'extreme_buy_count': extreme_buy_count,
                'extreme_sell_count': extreme_sell_count,
                'pressure_balance': avg_buy_pressure - avg_sell_pressure
            }
            
    except Exception as e:
        logger.debug(f"CVD volume analizi hatası: {e}")
    
    return volume_analysis


def get_cvd_signal_strength_distribution(df: pd.DataFrame) -> Dict:
    """
    CVD sinyal gücü dağılımı analizi
    
    Args:
        df (pd.DataFrame): Sinyal verileri
        
    Returns:
        Dict: Sinyal gücü dağılımı
    """
    if df.empty or 'signal_type' not in df.columns:
        return {}
    
    distribution = {}
    try:
        # Signal type dağılımı
        strong_long_count = len(df[df['signal_type'] == 'strong_long'])
        long_count = len(df[df['signal_type'] == 'long'])
        neutral_count = len(df[df['signal_type'] == 'neutral'])
        short_count = len(df[df['signal_type'] == 'short'])
        strong_short_count = len(df[df['signal_type'] == 'strong_short'])
        
        # Toplam sinyal gücü
        total_signals = len(df)
        strong_signals = strong_long_count + strong_short_count
        directional_signals = long_count + short_count + strong_signals
        
        distribution = {
            'strong_long': strong_long_count,
            'long': long_count,
            'neutral': neutral_count,
            'short': short_count,
            'strong_short': strong_short_count,
            'total_signals': total_signals,
            'strong_signals_ratio': (strong_signals / total_signals * 100) if total_signals > 0 else 0,
            'directional_signals_ratio': (directional_signals / total_signals * 100) if total_signals > 0 else 0,
            'bullish_bias': (strong_long_count + long_count) > (strong_short_count + short_count)
        }
        
    except Exception as e:
        logger.debug(f"CVD sinyal gücü analizi hatası: {e}")
    
    return distribution


# 🔥 YENİ: CVD sistem performans analizi
def get_cvd_system_performance(df: pd.DataFrame) -> Dict:
    """
    CVD sisteminin genel performans analizi
    
    Args:
        df (pd.DataFrame): Sinyal verileri
        
    Returns:
        Dict: Sistem performans metrikleri
    """
    if df.empty:
        return {}
    
    performance = {}
    try:
        # Temel metrikler
        total_signals = len(df)
        avg_ai_score = df['ai_score'].mean()
        
        # CVD kalite metrikleri
        if 'momentum_strength' in df.columns:
            avg_momentum = df['momentum_strength'].mean()
            high_momentum_count = len(df[df['momentum_strength'] >= 70])
            high_momentum_ratio = (high_momentum_count / total_signals * 100) if total_signals > 0 else 0
        else:
            avg_momentum = 0
            high_momentum_ratio = 0
        
        if 'deviso_cvd_harmony' in df.columns:
            avg_harmony = df['deviso_cvd_harmony'].mean()
            high_harmony_count = len(df[df['deviso_cvd_harmony'] >= 80])
            high_harmony_ratio = (high_harmony_count / total_signals * 100) if total_signals > 0 else 0
        else:
            avg_harmony = 0
            high_harmony_ratio = 0
        
        # Sistem efficiency skoru (0-100)
        efficiency_score = (
            (avg_ai_score * 0.4) +
            (avg_momentum * 0.3) +
            (avg_harmony * 0.3)
        )
        
        performance = {
            'total_signals': total_signals,
            'avg_ai_score': avg_ai_score,
            'avg_momentum_strength': avg_momentum,
            'avg_harmony_score': avg_harmony,
            'high_momentum_ratio': high_momentum_ratio,
            'high_harmony_ratio': high_harmony_ratio,
            'system_efficiency_score': efficiency_score,
            'quality_grade': get_quality_grade(efficiency_score)
        }
        
    except Exception as e:
        logger.debug(f"CVD sistem performans analizi hatası: {e}")
    
    return performance


def get_quality_grade(score: float) -> str:
    """Skor bazlı kalite notu"""
    if score >= 80:
        return 'A+'
    elif score >= 70:
        return 'A'
    elif score >= 60:
        return 'B+'
    elif score >= 50:
        return 'B'
    elif score >= 40:
        return 'C+'
    elif score >= 30:
        return 'C'
    else:
        return 'D'