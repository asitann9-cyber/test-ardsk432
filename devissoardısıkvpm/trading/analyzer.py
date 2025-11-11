"""
🔍 Sinyal Analiz Modülü - VPMV Sistemi (SADECE 4 BİLEŞEN)
AI destekli kripto sinyal analizi ve batch processing
🔥 SADECE: VPMV (Volume-Price-Momentum-Volatility)
🔥 MTF KALDIRILDI - TIME Alignment KALDIRILDI
🔥 ESKİ SİSTEM KALDIRILDI: Deviso, Z-Score, Gauss, Log Volume
"""

import time
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    LOCAL_TZ, MAX_WORKERS, REQ_SLEEP, DEFAULT_MIN_AI_SCORE,
    DEFAULT_MIN_VPMV_SCORE,
    current_data, saved_signals
)
from data.fetch_data import fetch_klines, get_usdt_perp_symbols
from core.indicators import compute_vpmv_metrics
from core.ai_model import ai_model

logger = logging.getLogger("crypto-analytics")


def analyze_symbol_with_ai(symbol: str, interval: str) -> Dict:
    """
    🔥 SADECE 4 BİLEŞEN: VPMV bazlı sembol analizi
    
    Args:
        symbol (str): Trading sembolü
        interval (str): Zaman dilimi
        
    Returns:
        Dict: Analiz sonuçları (SADECE 4 bileşen) veya boş dict
    """
    try:
        # Rate limiting
        time.sleep(REQ_SLEEP)
        
        # Veri çek
        df = fetch_klines(symbol, interval)
        if df is None or df.empty:
            logger.debug(f"❌ {symbol}: Veri çekilemedi")
            return {}
        
        if len(df) < 30:
            logger.debug(f"❌ {symbol}: Yetersiz veri ({len(df)} < 30)")
            return {}
        
        # 🔥 SADECE 4 BİLEŞEN: VPMV metriklerini hesapla
        metrics = compute_vpmv_metrics(df, symbol)
        
        if not metrics or metrics.get('run_type') == 'none':
            logger.debug(f"❌ {symbol}: VPMV hesaplanamadı")
            return {}
        
        # 🔥 VPMV skoru kontrolü
        vpmv_score = metrics.get('vpmv_score', 0.0)
        if abs(vpmv_score) < DEFAULT_MIN_VPMV_SCORE:
            logger.debug(f"❌ {symbol}: VPMV skoru düşük ({vpmv_score:.2f})")
            return {}
        
        # 🔥 Bileşen değerlerini kontrol et
        volume_comp = abs(metrics.get('volume_component', 0.0))
        price_comp = abs(metrics.get('price_component', 0.0))
        momentum_comp = abs(metrics.get('momentum_component', 0.0))
        
        # En az bir bileşen anlamlı olmalı
        if volume_comp < 5.0 and price_comp < 5.0 and momentum_comp < 5.0:
            logger.debug(f"❌ {symbol}: Tüm bileşenler çok zayıf")
            return {}
        
        # AI skoru hesapla (VPMV bazlı)
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
        
        # Başarılı sinyal (SADECE 4 BİLEŞEN)
        result = {
            'symbol': symbol,
            'timeframe': interval,
            'last_close': last_close,
            'run_type': metrics['run_type'],  # 'long' veya 'short'
            
            # 🔥 VPMV Bileşenleri (SADECE 4)
            'volume_component': metrics['volume_component'],
            'price_component': metrics['price_component'],
            'momentum_component': metrics['momentum_component'],
            'volatility_component': metrics['volatility_component'],
            'vpmv_score': metrics['vpmv_score'],
            
            # Tetikleyici
            'trigger_type': metrics.get('trigger_type', 'Yok'),
            
            # AI Skoru
            'ai_score': ai_score,
            
            # Zaman
            'last_update': last_update.strftime('%Y-%m-%d %H:%M:%S %Z'),
            
            # 🔥 ESKİ SİSTEM UYUMLULUĞU (UI için - sıfır değerler)
            'run_count': 0,
            'run_perc': 0.0,
            'gauss_run': 0.0,
            'gauss_run_perc': 0.0,
            'log_volume': 0.0,
            'log_volume_momentum': 0.0,
            'deviso_ratio': 0.0,
            'c_signal_momentum': 0.0,
            'max_zscore': 0.0,
            'trend_direction': metrics['run_type'].upper(),
            'trend_strength': abs(metrics['vpmv_score'])
        }
        
        logger.debug(
            f"✅ {symbol}: VPMV={vpmv_score:+.1f}, "
            f"AI={ai_score:.0f}%, Trigger={metrics.get('trigger_type', 'Yok')}"
        )
        return result
        
    except Exception as e:
        logger.warning(f"analyze_symbol error {symbol}: {e}")
        return {}


def batch_analyze_with_ai(interval: str) -> pd.DataFrame:
    """
    🔥 SADECE 4 BİLEŞEN: VPMV bazlı toplu analiz
    
    Args:
        interval (str): Analiz edilecek zaman dilimi
        
    Returns:
        pd.DataFrame: Analiz sonuçları (SADECE 4 bileşen)
    """
    global saved_signals
    
    start_time = time.time()
    
    # Sembol listesini al
    symbols = get_usdt_perp_symbols()
    if not symbols:
        logger.error("Sembol listesi boş!")
        return pd.DataFrame()
    
    logger.info(f"🤖 {len(symbols)} sembol için VPMV analiz başlatılıyor (SADECE 4 BİLEŞEN)...")
    
    # Yeni analiz sonuçları
    fresh_results = []
    processed_count = 0
    vpmv_success_count = 0
    
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
                    vpmv_success_count += 1
                    
                    # Kaydedilmiş sinyalleri güncelle
                    saved_signals[symbol] = {
                        'data': res,
                        'last_seen': datetime.now(LOCAL_TZ)
                    }
                
                # İlerleme logu
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    success_rate = (vpmv_success_count / processed_count) * 100
                    logger.info(
                        f"🤖 İşlenen: {processed_count}/{len(symbols)} - Hız: {rate:.1f} s/sn - "
                        f"VPMV: {success_rate:.1f}% ({vpmv_success_count})"
                    )
            
            except Exception as e:
                logger.debug(f"Future hatası {symbol}: {e}")
    
    # Mevcut zaman
    current_time = datetime.now(LOCAL_TZ)
    fresh_symbols = {r['symbol'] for r in fresh_results}
    
    # Eski sinyalleri koruma ve skor düşürme
    protected_count = 0
    for symbol, saved_info in list(saved_signals.items()):
        minutes_old = (current_time - saved_info['last_seen']).total_seconds() / 60.0
        
        # 10 dakikadan eski sinyalleri sil
        if minutes_old > 10:
            del saved_signals[symbol]
            continue
        
        if symbol not in fresh_symbols:
            old_data = saved_info['data'].copy()
            original_score = old_data['ai_score']
            
            # 🔥 VPMV gücüne göre ceza
            vpmv_score = abs(old_data.get('vpmv_score', 0))
            
            # VPMV güçlü = az ceza
            if vpmv_score > 30.0:
                base_penalty = 10
            elif vpmv_score > 20.0:
                base_penalty = 15
            else:
                base_penalty = 25
            
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
            
            fresh_results.append(old_data)
            protected_count += 1
            
            logger.debug(
                f"📉 {symbol}: {original_score:.0f} → {new_score:.0f} "
                f"(yaş: {minutes_old:.1f}dk, VPMV: {vpmv_score:.1f})"
            )
    
    # Performans istatistikleri
    elapsed_time = time.time() - start_time
    total_rate = len(symbols) / elapsed_time if elapsed_time > 0 else 0
    
    new_signals = len(fresh_symbols)
    total_signals = len(fresh_results)
    vpmv_success_rate = (vpmv_success_count / len(symbols)) * 100 if len(symbols) > 0 else 0
    
    logger.info("✅ VPMV Analiz tamamlandı (SADECE 4 BİLEŞEN):")
    logger.info(f"   🆕 Yeni sinyal: {new_signals}")
    logger.info(f"   📉 Korunan sinyal: {protected_count}")
    logger.info(f"   🎯 Toplam sinyal: {total_signals}")
    logger.info(f"   📊 VPMV başarı oranı: {vpmv_success_rate:.1f}%")
    logger.info(f"   ⏱️ Süre: {elapsed_time:.1f}s - Hız: {total_rate:.1f} s/sn")
    
    if not fresh_results:
        logger.warning("⚠️ Hiç sinyal bulunamadı - filtreleri gözden geçirin")
        return pd.DataFrame()
    
    # DataFrame oluştur ve sırala - 🔥 SADECE 4 BİLEŞEN
    df = pd.DataFrame(fresh_results)
    df = df.sort_values(
        by=['ai_score', 'vpmv_score', 'price_component', 'volatility_component'],
        ascending=[False, False, False, False]
    )
    
    if len(df) > 0:
        top_signal = df.iloc[0]
        logger.info(
            f"🏆 En yüksek AI skoru: {top_signal['ai_score']:.0f}% - {top_signal['symbol']} "
            f"(VPMV: {top_signal['vpmv_score']:+.1f}, "
            f"Trigger: {top_signal.get('trigger_type', 'Yok')})"
        )
        
        # Run type dağılımı
        long_count = len(df[df['run_type'] == 'long'])
        short_count = len(df[df['run_type'] == 'short'])
        logger.info(f"📈 Sinyal dağılımı: LONG={long_count}, SHORT={short_count}")
        
        if protected_count > 0:
            logger.info("📉 Korunan sinyaller skor düşüşü ile aşağı kaydı")
        
        logger.debug("📊 Analyzer sonrası ilk 3 sinyal:")
        for idx, (i, row) in enumerate(df.head(3).iterrows(), 1):
            logger.debug(
                f"   {idx}: {row['symbol']} | AI={row['ai_score']:.0f}% | "
                f"VPMV={row['vpmv_score']:+.1f}"
            )
    
    return df


def filter_signals(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    🔥 SADECE 4 BİLEŞEN: VPMV bazlı sinyal filtreleme
    
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
    
    # 🔥 VPMV skoru filtresi
    if filters.get('min_vpmv_score', 0) > 0:
        filtered_df = filtered_df[abs(filtered_df['vpmv_score']) >= filters['min_vpmv_score']]
    
    # 🔥 Bileşen filtreleri
    if filters.get('min_price_component', 0) > 0:
        filtered_df = filtered_df[abs(filtered_df['price_component']) >= filters['min_price_component']]
    
    if filters.get('min_volume_component', 0) > 0:
        filtered_df = filtered_df[abs(filtered_df['volume_component']) >= filters['min_volume_component']]
    
    # Run type filtresi
    run_type_filter = filters.get('run_type')
    if run_type_filter and run_type_filter != 'all':
        filtered_df = filtered_df[filtered_df['run_type'] == run_type_filter]
    
    # 🔥 Tetikleyici filtresi
    trigger_filter = filters.get('trigger_type')
    if trigger_filter and trigger_filter != 'all':
        filtered_df = filtered_df[filtered_df['trigger_type'] == trigger_filter]
    
    filtered_count = len(filtered_df)
    logger.info(f"🔍 Filtre sonucu: {filtered_count}/{original_count} sinyal kaldı")
    
    return filtered_df


def get_top_signals(df: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    """
    🔥 SADECE 4 BİLEŞEN: En iyi VPMV sinyalleri al
    
    Args:
        df (pd.DataFrame): Sinyal DataFrame'i
        count (int): Alınacak sinyal sayısı
        
    Returns:
        pd.DataFrame: En iyi sinyaller
    """
    if df.empty:
        return df
    
    sorted_df = df.sort_values(
        by=['ai_score', 'vpmv_score', 'price_component', 'volatility_component'],
        ascending=[False, False, False, False]
    )
    
    return sorted_df.head(count)


def analyze_signal_quality(df: pd.DataFrame) -> Dict:
    """
    🔥 SADECE 4 BİLEŞEN: VPMV bazlı sinyal kalitesi analizi
    
    Args:
        df (pd.DataFrame): Analiz edilecek sinyaller
        
    Returns:
        Dict: Kalite metrikleri (SADECE 4 bileşen)
    """
    if df.empty:
        return {
            'total_signals': 0,
            'avg_ai_score': 0,
            'long_signals': 0,
            'short_signals': 0,
            'high_quality_signals': 0,
            'quality_distribution': {},
            'vpmv_quality': {}
        }
    
    total_signals = len(df)
    avg_ai_score = df['ai_score'].mean()
    
    long_signals = len(df[df['run_type'] == 'long'])
    short_signals = len(df[df['run_type'] == 'short'])
    
    # Kalite kategorileri
    high_quality = len(df[df['ai_score'] >= 80])
    medium_quality = len(df[(df['ai_score'] >= 60) & (df['ai_score'] < 80)])
    low_quality = len(df[df['ai_score'] < 60])
    
    # 🔥 VPMV kalite analizi
    vpmv_quality = {}
    if 'vpmv_score' in df.columns:
        avg_vpmv = df['vpmv_score'].mean()
        strong_vpmv = len(df[abs(df['vpmv_score']) >= 30.0])
        medium_vpmv = len(df[(abs(df['vpmv_score']) >= 15.0) & (abs(df['vpmv_score']) < 30.0)])
        weak_vpmv = len(df[abs(df['vpmv_score']) < 15.0])
        
        vpmv_quality = {
            'avg_vpmv_score': avg_vpmv,
            'strong_vpmv': strong_vpmv,
            'medium_vpmv': medium_vpmv,
            'weak_vpmv': weak_vpmv
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
        'vpmv_quality': vpmv_quality
    }


def update_signal_scores():
    """
    🔥 SADECE 4 BİLEŞEN: Kayıtlı sinyallerin skorlarını güncelle (VPMV bazlı yaş cezası)
    """
    global saved_signals
    
    current_time = datetime.now(LOCAL_TZ)
    updated_count = 0
    removed_count = 0
    
    for symbol, saved_info in list(saved_signals.items()):
        minutes_old = (current_time - saved_info['last_seen']).total_seconds() / 60.0
        
        # 15 dakikadan eski sinyalleri sil
        if minutes_old > 15:
            del saved_signals[symbol]
            removed_count += 1
            continue
        
        # Yaşlanma cezası uygula
        if minutes_old > 2:
            original_score = saved_info['data']['ai_score']
            
            # 🔥 VPMV bazlı ceza
            vpmv_score = abs(saved_info['data'].get('vpmv_score', 0))
            
            # Güçlü sinyal = az ceza
            if vpmv_score > 30.0:
                base_penalty = 10
            elif vpmv_score > 20.0:
                base_penalty = 15
            else:
                base_penalty = 25
            
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
    🔥 SADECE 4 BİLEŞEN: VPMV bazlı sinyal özeti
    
    Returns:
        Dict: Sinyal özet bilgileri (SADECE 4 bileşen)
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
            'vpmv_stats': {}
        }
    
    total_signals = len(current_data)
    long_count = len(current_data[current_data['run_type'] == 'long'])
    short_count = len(current_data[current_data['run_type'] == 'short'])
    avg_ai_score = current_data['ai_score'].mean()
    
    # En yüksek skorlu sembol
    top_signal = current_data.iloc[0] if not current_data.empty else None
    top_symbol = top_signal['symbol'] if top_signal is not None else None
    
    # 🔥 VPMV istatistikleri
    vpmv_stats = {}
    if 'vpmv_score' in current_data.columns and not current_data.empty:
        vpmv_stats = {
            'avg_vpmv_score': current_data['vpmv_score'].mean(),
            'max_vpmv_score': current_data['vpmv_score'].max(),
            'min_vpmv_score': current_data['vpmv_score'].min(),
            'positive_vpmv_count': len(current_data[current_data['vpmv_score'] > 0]),
            'negative_vpmv_count': len(current_data[current_data['vpmv_score'] < 0])
        }
    
    return {
        'total_signals': total_signals,
        'long_count': long_count,
        'short_count': short_count,
        'avg_ai_score': avg_ai_score,
        'top_symbol': top_symbol,
        'last_update': datetime.now(LOCAL_TZ).strftime('%H:%M:%S'),
        'vpmv_stats': vpmv_stats
    }


# 🔥 VPMV spesifik analiz fonksiyonları (SADECE 4 BİLEŞEN)
def analyze_vpmv_components(df: pd.DataFrame) -> Dict:
    """
    VPMV bileşenlerini analiz et (SADECE 4 BİLEŞEN)
    
    Args:
        df (pd.DataFrame): Sinyal verileri
        
    Returns:
        Dict: Bileşen analizi
    """
    if df.empty:
        return {}
    
    component_analysis = {}
    
    # Volume bileşeni analizi
    if 'volume_component' in df.columns:
        component_analysis['volume'] = {
            'avg': df['volume_component'].mean(),
            'max': df['volume_component'].max(),
            'min': df['volume_component'].min(),
            'strong_count': len(df[abs(df['volume_component']) >= 30.0])
        }
    
    # Price bileşeni analizi
    if 'price_component' in df.columns:
        component_analysis['price'] = {
            'avg': df['price_component'].mean(),
            'max': df['price_component'].max(),
            'min': df['price_component'].min(),
            'strong_count': len(df[abs(df['price_component']) >= 30.0])
        }
    
    # Momentum bileşeni analizi
    if 'momentum_component' in df.columns:
        component_analysis['momentum'] = {
            'avg': df['momentum_component'].mean(),
            'max': df['momentum_component'].max(),
            'min': df['momentum_component'].min(),
            'strong_count': len(df[abs(df['momentum_component']) >= 30.0])
        }
    
    # Volatility bileşeni analizi
    if 'volatility_component' in df.columns:
        component_analysis['volatility'] = {
            'avg': df['volatility_component'].mean(),
            'max': df['volatility_component'].max(),
            'min': df['volatility_component'].min(),
            'strong_count': len(df[abs(df['volatility_component']) >= 30.0])
        }
    
    return component_analysis


def get_trigger_distribution(df: pd.DataFrame) -> Dict:
    """
    Tetikleyici dağılımını analiz et
    
    Args:
        df (pd.DataFrame): Sinyal verileri
        
    Returns:
        Dict: Tetikleyici dağılımı
    """
    if df.empty or 'trigger_type' not in df.columns:
        return {}
    
    trigger_counts = df['trigger_type'].value_counts().to_dict()
    total = len(df)
    
    trigger_distribution = {
        trigger: {
            'count': count,
            'percentage': (count / total * 100) if total > 0 else 0
        }
        for trigger, count in trigger_counts.items()
    }
    
    return trigger_distribution