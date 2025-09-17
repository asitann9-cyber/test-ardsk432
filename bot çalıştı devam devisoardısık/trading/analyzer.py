"""
🔍 Sinyal Analiz Modülü
AI destekli kripto sinyal analizi ve batch processing
🆕 RSI MOMENTUM + LOGARİTMİK HACİM SİSTEMİ ENTEGRE EDİLDİ
❌ ESKİ VOLUME SİSTEMİ KALDIRILDI
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
    🆕 GÜNCELLENMIŞ: RSI momentum + logaritmik hacim kontrolleri eklendi
    
    Args:
        symbol (str): Trading sembolü
        interval (str): Zaman dilimi
        
    Returns:
        Dict: Analiz sonuçları (yeni momentum+volume metrikleri ile)
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
        
        # 🆕 YENİ METRİKLERİ HESAPLA (RSI momentum + log volume dahil)
        metrics = compute_consecutive_metrics(df)
        if not metrics or metrics['run_type'] == 'none':
            logger.debug(f"❌ {symbol}: Run type none veya metrik yok")
            return {}
        
        # Deviso ratio doğrulama (aynı)
        deviso_ratio = metrics.get('deviso_ratio', 0.0)
        if abs(deviso_ratio) < 0.05:
            logger.debug(f"❌ {symbol}: Deviso ratio çok küçük ({deviso_ratio:.4f})")
            return {}
        
        # 🆕 RSI MOMENTUM KONTROLÜ
        rsi_momentum = metrics.get('rsi_momentum', 0.0)
        if abs(rsi_momentum) < 1.0:  # Minimum RSI momentum
            logger.debug(f"❌ {symbol}: RSI momentum çok zayıf ({rsi_momentum:.4f})")
            return {}
        
        # 🆕 LOG VOLUME KONTROLÜ
        log_volume_strength = metrics.get('log_volume_strength', 0.0)
        if log_volume_strength < 0.5:  # Minimum log volume strength
            logger.debug(f"❌ {symbol}: Log volume çok zayıf ({log_volume_strength:.4f})")
            return {}
        
        # Trend yönü kontrolü - GEVŞETILDI
        streak_type = metrics['run_type']
        direction_match = False
        trend_strength = abs(deviso_ratio)
        
        if streak_type == 'long' and deviso_ratio > 0.2:
            direction_match = True
        elif streak_type == 'short' and deviso_ratio < -0.2:
            direction_match = True
        
        # Trend uyumsuzluğunda sinyali reddet
        if not direction_match:
            logger.debug(f"❌ {symbol}: Trend uyumsuz - {streak_type} vs deviso {deviso_ratio:.4f}")
            return {}
        
        # 🆕 MOMENTUM TREND UYUMU KONTROLÜ
        # RSI momentum ile run_type uyumlu olmalı
        momentum_direction_match = False
        if streak_type == 'long' and rsi_momentum > 0:
            momentum_direction_match = True
        elif streak_type == 'short' and rsi_momentum < 0:
            momentum_direction_match = True
        
        if not momentum_direction_match:
            logger.debug(f"❌ {symbol}: Momentum trend uyumsuz - {streak_type} vs RSI Δ {rsi_momentum:.4f}")
            return {}
        
        # Ek kalite kontrolleri
        run_count = metrics.get('run_count', 0)
        run_perc = metrics.get('run_perc')
        
        # Minimum streak kontrolü
        if run_count < 1:
            logger.debug(f"❌ {symbol}: Run count çok düşük ({run_count})")
            return {}
            
        # Minimum hareket kontrolü
        if run_perc is None or abs(run_perc) < 0.15:
            logger.debug(f"❌ {symbol}: Run percentage çok düşük ({run_perc})")
            return {}
        
        # AI skoru hesapla (yeni metriklerle)
        ai_score = ai_model.predict_score(metrics)
        
        # Minimum AI skoru kontrolü
        min_ai_threshold = (DEFAULT_MIN_AI_SCORE * 100) * 0.5
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
        
        # 🆕 GÜNCELLENMIŞ SONUÇ - YENİ METRİKLER EKLENDI
        result = {
            'symbol': symbol,
            'timeframe': interval,
            'last_close': last_close,
            'run_type': metrics['run_type'],
            'run_count': metrics['run_count'],
            'run_perc': metrics['run_perc'],
            'gauss_run': metrics['gauss_run'],
            'gauss_run_perc': metrics['gauss_run_perc'],
            
            # 🆕 YENİ RSI MOMENTUM METRİKLERİ
            'rsi_momentum': metrics['rsi_momentum'],
            'momentum_score': metrics['momentum_score'],
            'momentum_strength': metrics['momentum_strength'],
            'momentum_direction': metrics['momentum_direction'],
            
            # 🆕 YENİ LOG VOLUME METRİKLERİ
            'log_volume_strength': metrics['log_volume_strength'],
            'log_volume_ratio': metrics['log_volume_ratio'],
            'log_volume_change': metrics['log_volume_change'],
            'log_volume_trend': metrics['log_volume_trend'],
            
            # ❌ KALDIRILDI: 'vol_ratio', 'hh_vol_streak'
            
            # Deviso (aynı)
            'deviso_ratio': metrics['deviso_ratio'],
            'ai_score': ai_score,  
            'last_update': last_update.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
        }
        
        logger.debug(f"✅ {symbol}: RSI+Log sinyal onaylandı - AI:{ai_score:.0f}%, RSI Δ:{rsi_momentum:.2f}, Log Vol:{log_volume_strength:.2f}")
        return result
        
    except Exception as e:
        logger.warning(f"analyze_symbol error {symbol}: {e}")
        return {}


def batch_analyze_with_ai(interval: str) -> pd.DataFrame:
    """
    🆕 GÜNCELLENMIŞ: RSI momentum + log volume ile batch analiz
    
    Args:
        interval (str): Analiz edilecek zaman dilimi
        
    Returns:
        pd.DataFrame: Analiz sonuçları (yeni metriklerle)
    """
    global saved_signals
    
    start_time = time.time()
    
    # Sembol listesini al
    symbols = get_usdt_perp_symbols()
    if not symbols:
        logger.error("Sembol listesi boş!")
        return pd.DataFrame()
    
    logger.info(f"🤖 {len(symbols)} sembol için RSI MOMENTUM + LOG VOLUME analiz başlatılıyor...")
    
    # Yeni analiz sonuçları
    fresh_results = []
    processed_count = 0
    momentum_success_count = 0  # 🆕 YENİ: Momentum+volume başarı sayacı
    
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
                    momentum_success_count += 1
                    
                    # Kaydedilmiş sinyalleri güncelle
                    saved_signals[symbol] = {
                        'data': res,
                        'last_seen': datetime.now(LOCAL_TZ)
                    }
                    
                # İlerleme logu
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    success_rate = (momentum_success_count / processed_count) * 100
                    logger.info(f"🤖 İşlenen: {processed_count}/{len(symbols)} - Hız: {rate:.1f} s/sn - RSI+Vol Başarı: {success_rate:.1f}% ({momentum_success_count})")
                    
            except Exception as e:
                logger.debug(f"Future hatası {symbol}: {e}")

    # Mevcut zaman
    current_time = datetime.now(LOCAL_TZ)
    fresh_symbols = {r['symbol'] for r in fresh_results} 
    
    # 🆕 GÜNCELLENMIŞ: RSI momentum bazlı eski sinyalleri koruma
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
            
            # 🆕 YENİ: RSI momentum + log volume bazlı ceza sistemi
            rsi_momentum = abs(old_data.get('rsi_momentum', 0))
            log_vol_strength = old_data.get('log_volume_strength', 0)
            deviso_ratio = old_data.get('deviso_ratio', 0)
            trend_strength = abs(deviso_ratio)
            
            # Güçlü momentum+volume varsa daha az ceza
            if rsi_momentum > 10 and log_vol_strength > 3.0:
                base_penalty = 8  # Çok güçlü momentum+volume
            elif rsi_momentum > 5 and log_vol_strength > 2.0:
                base_penalty = 12  # Güçlü momentum+volume
            elif rsi_momentum > 2 and log_vol_strength > 1.0:
                base_penalty = 18  # Orta momentum+volume
            else:
                base_penalty = 25  # Zayıf momentum+volume
            
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
            
            logger.debug(f"📉 {symbol}: {original_score:.0f} → {new_score:.0f} (yaş: {minutes_old:.1f}dk, RSI Δ: {rsi_momentum:.2f}, Log Vol: {log_vol_strength:.2f})")

    # Performans istatistikleri
    elapsed_time = time.time() - start_time
    total_rate = len(symbols) / elapsed_time if elapsed_time > 0 else 0
    
    new_signals = len(fresh_symbols)
    total_signals = len(fresh_results)
    momentum_success_rate = (momentum_success_count / len(symbols)) * 100 if len(symbols) > 0 else 0
    
    logger.info(f"✅ RSI MOMENTUM + LOG VOLUME Analiz tamamlandı:")
    logger.info(f"   🆕 Yeni sinyal: {new_signals}")
    logger.info(f"   📉 Korunan sinyal: {protected_count}")
    logger.info(f"   🎯 Toplam sinyal: {total_signals}")
    logger.info(f"   📊 RSI+Vol başarı oranı: {momentum_success_rate:.1f}%")
    logger.info(f"   ⏱️ Süre: {elapsed_time:.1f}s - Hız: {total_rate:.1f} s/sn")
    
    if not fresh_results:
        logger.warning("⚠️ Hiç RSI momentum+log volume sinyali bulunamadı")
        return pd.DataFrame()
        
    # DataFrame oluştur ve sırala
    df = pd.DataFrame(fresh_results)
    
    # 🆕 YENİ SIRALAMA: AI skoru > RSI momentum > Log volume > Deviso > Run perc
    df = df.sort_values(
        by=['ai_score', 'momentum_score', 'log_volume_strength', 'trend_strength', 'run_perc'], 
        ascending=[False, False, False, False, False]
    )
    
    if len(df) > 0:
        top_signal = df.iloc[0]
        logger.info(f"🏆 En yüksek AI skoru: {top_signal['ai_score']:.0f}% - {top_signal['symbol']}")
        logger.info(f"    RSI Δ: {top_signal['rsi_momentum']:.2f} | Log Vol: {top_signal['log_volume_strength']:.2f} | Deviso: {top_signal['deviso_ratio']:.2f}")
        
        # RSI momentum dağılımı
        if 'momentum_strength' in df.columns:
            momentum_dist = df['momentum_strength'].value_counts()
            logger.info(f"📈 RSI Momentum dağılımı: {dict(momentum_dist)}")
            
        if protected_count > 0:
            logger.info(f"📉 Korunan sinyaller RSI momentum+log volume bazlı skor düşüşü ile aşağı kaydı")
    
    return df


def filter_signals(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    🆕 GÜNCELLENMIŞ: RSI momentum + log volume filtreleri eklendi
    
    Args:
        df (pd.DataFrame): Sinyal DataFrame'i
        filters (Dict): Filtre parametreleri (yeni momentum+volume filtreleri dahil)
        
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
    
    # 🆕 RSI MOMENTUM FİLTRESİ
    min_rsi_momentum = filters.get('min_rsi_momentum', 0)
    if min_rsi_momentum > 0:
        filtered_df = filtered_df[abs(filtered_df['rsi_momentum']) >= min_rsi_momentum]
    
    # 🆕 LOG VOLUME FİLTRESİ
    min_log_volume = filters.get('min_log_volume_strength', 0)
    if min_log_volume > 0:
        filtered_df = filtered_df[filtered_df['log_volume_strength'] >= min_log_volume]
    
    # 🆕 MOMENTUM SCORE FİLTRESİ
    min_momentum_score = filters.get('min_momentum_score', 0)
    if min_momentum_score > 0:
        filtered_df = filtered_df[filtered_df['momentum_score'] >= min_momentum_score]
    
    # ❌ KALDIRILDI: Volume ratio filtresi
    
    # Deviso ratio filtresi (aynı)
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
    logger.info(f"🔍 RSI momentum+log volume filtre sonucu: {filtered_count}/{original_count} sinyal kaldı")
    
    return filtered_df


def get_top_signals(df: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    """
    🆕 GÜNCELLENMIŞ: RSI momentum + log volume sıralaması
    
    Args:
        df (pd.DataFrame): Sinyal DataFrame'i
        count (int): Alınacak sinyal sayısı
        
    Returns:
        pd.DataFrame: En iyi sinyaller (yeni sıralama ile)
    """
    if df.empty:
        return df
    
    # 🆕 YENİ SIRALAMA: AI skoru > Momentum skoru > Log volume gücü > Trend gücü > Run perc
    sorted_df = df.sort_values(
        by=['ai_score', 'momentum_score', 'log_volume_strength', 'trend_strength', 'run_perc'],
        ascending=[False, False, False, False, False]
    )
    
    return sorted_df.head(count)


def analyze_signal_quality(df: pd.DataFrame) -> Dict:
    """
    🆕 GÜNCELLENMIŞ: RSI momentum + log volume kalite analizi eklendi
    
    Args:
        df (pd.DataFrame): Analiz edilecek sinyaller
        
    Returns:
        Dict: Kalite metrikleri (yeni momentum+volume metrikleri dahil)
    """
    if df.empty:
        return {
            'total_signals': 0,
            'avg_ai_score': 0,
            'long_signals': 0,
            'short_signals': 0,
            'high_quality_signals': 0,
            'quality_distribution': {},
            'momentum_quality': {},  # 🆕 YENİ
            'volume_quality': {},    # 🆕 YENİ
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
    
    # 🆕 RSI MOMENTUM KALİTE ANALİZİ
    momentum_quality = {}
    if 'rsi_momentum' in df.columns and 'momentum_score' in df.columns:
        avg_rsi_momentum = abs(df['rsi_momentum']).mean()
        avg_momentum_score = df['momentum_score'].mean()
        
        strong_momentum = len(df[abs(df['rsi_momentum']) >= 10])
        medium_momentum = len(df[(abs(df['rsi_momentum']) >= 5) & (abs(df['rsi_momentum']) < 10)])
        weak_momentum = len(df[abs(df['rsi_momentum']) < 5])
        
        momentum_quality = {
            'avg_rsi_momentum': avg_rsi_momentum,
            'avg_momentum_score': avg_momentum_score,
            'strong_momentum': strong_momentum,
            'medium_momentum': medium_momentum,
            'weak_momentum': weak_momentum
        }
    
    # 🆕 LOG VOLUME KALİTE ANALİZİ
    volume_quality = {}
    if 'log_volume_strength' in df.columns:
        avg_log_volume = df['log_volume_strength'].mean()
        
        strong_volume = len(df[df['log_volume_strength'] >= 3.0])
        medium_volume = len(df[(df['log_volume_strength'] >= 1.5) & (df['log_volume_strength'] < 3.0)])
        weak_volume = len(df[df['log_volume_strength'] < 1.5])
        
        volume_quality = {
            'avg_log_volume_strength': avg_log_volume,
            'strong_volume': strong_volume,
            'medium_volume': medium_volume,
            'weak_volume': weak_volume
        }
    
    # Deviso kalite analizi (aynı)
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
        'momentum_quality': momentum_quality,  # 🆕 YENİ
        'volume_quality': volume_quality,      # 🆕 YENİ
        'deviso_quality': deviso_quality,
        'trend_distribution': trend_distribution
    }


def update_signal_scores():
    """
    🆕 GÜNCELLENMIŞ: RSI momentum + log volume bazlı yaş cezası
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
            
            # 🆕 YENİ: RSI momentum + log volume gücüne göre ceza
            rsi_momentum = abs(saved_info['data'].get('rsi_momentum', 0))
            log_vol_strength = saved_info['data'].get('log_volume_strength', 0)
            
            if rsi_momentum > 10 and log_vol_strength > 3.0:
                base_penalty = 10  # Güçlü momentum+volume, az ceza
            elif rsi_momentum > 5 and log_vol_strength > 2.0:
                base_penalty = 20  # Orta momentum+volume
            else:
                base_penalty = 35  # Zayıf momentum+volume, çok ceza
            
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
        logger.debug(f"🔄 RSI momentum+log volume sinyal güncelleme: {updated_count} güncellendi, {removed_count} silindi")


def get_signal_summary() -> Dict:
    """
    🆕 GÜNCELLENMIŞ: RSI momentum + log volume istatistikleri eklendi
    
    Returns:
        Dict: Sinyal özet bilgileri (yeni metriklerle)
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
            'momentum_stats': {},  # 🆕 YENİ
            'volume_stats': {},    # 🆕 YENİ
            'deviso_stats': {}
        }
    
    total_signals = len(current_data)
    long_count = len(current_data[current_data['run_type'] == 'long'])
    short_count = len(current_data[current_data['run_type'] == 'short'])
    avg_ai_score = current_data['ai_score'].mean()
    
    # En yüksek skorlu sembol
    top_signal = current_data.iloc[0] if not current_data.empty else None
    top_symbol = top_signal['symbol'] if top_signal is not None else None
    
    # 🆕 RSI MOMENTUM İSTATİSTİKLERİ
    momentum_stats = {}
    if 'rsi_momentum' in current_data.columns and not current_data.empty:
        momentum_stats = {
            'avg_rsi_momentum': abs(current_data['rsi_momentum']).mean(),
            'max_rsi_momentum': current_data['rsi_momentum'].max(),
            'min_rsi_momentum': current_data['rsi_momentum'].min(),
            'positive_momentum_count': len(current_data[current_data['rsi_momentum'] > 0]),
            'negative_momentum_count': len(current_data[current_data['rsi_momentum'] < 0]),
            'avg_momentum_score': current_data['momentum_score'].mean() if 'momentum_score' in current_data.columns else 0
        }
    
    # 🆕 LOG VOLUME İSTATİSTİKLERİ
    volume_stats = {}
    if 'log_volume_strength' in current_data.columns and not current_data.empty:
        volume_stats = {
            'avg_log_volume_strength': current_data['log_volume_strength'].mean(),
            'max_log_volume_strength': current_data['log_volume_strength'].max(),
            'min_log_volume_strength': current_data['log_volume_strength'].min(),
            'strong_volume_count': len(current_data[current_data['log_volume_strength'] >= 3.0]),
            'medium_volume_count': len(current_data[(current_data['log_volume_strength'] >= 1.5) & (current_data['log_volume_strength'] < 3.0)]),
            'weak_volume_count': len(current_data[current_data['log_volume_strength'] < 1.5])
        }
    
    # Deviso istatistikleri (aynı)
    deviso_stats = {}
    if 'deviso_ratio' in current_data.columns and not current_data.empty:
        deviso_stats = {
            'avg_deviso_ratio': current_data['deviso_ratio'].mean(),
            'max_deviso_ratio': current_data['deviso_ratio'].max(),
            'min_deviso_ratio': current_data['deviso_ratio'].min(),
            'positive_deviso_count': len(current_data[current_data['deviso_ratio'] > 0]),
            'negative_deviso_count': len(current_data[current_data['deviso_ratio'] < 0])
        }
    
    return {
        'total_signals': total_signals,
        'long_count': long_count,
        'short_count': short_count,
        'avg_ai_score': avg_ai_score,
        'top_symbol': top_symbol,
        'last_update': datetime.now(LOCAL_TZ).strftime('%H:%M:%S'),
        'momentum_stats': momentum_stats,  # 🆕 YENİ
        'volume_stats': volume_stats,      # 🆕 YENİ
        'deviso_stats': deviso_stats
    }


# 🆕 YENİ: RSI momentum spesifik analiz fonksiyonları
def analyze_momentum_trends(df: pd.DataFrame) -> Dict:
    """
    RSI momentum trend analizi
    
    Args:
        df (pd.DataFrame): Sinyal verileri
        
    Returns:
        Dict: Momentum trend analizi sonuçları
    """
    if df.empty or 'rsi_momentum' not in df.columns:
        return {}
    
    very_strong_bullish = len(df[df['rsi_momentum'] >= 15])
    strong_bullish = len(df[(df['rsi_momentum'] >= 10) & (df['rsi_momentum'] < 15)])
    moderate_bullish = len(df[(df['rsi_momentum'] >= 5) & (df['rsi_momentum'] < 10)])
    weak_bullish = len(df[(df['rsi_momentum'] > 0) & (df['rsi_momentum'] < 5)])
    
    weak_bearish = len(df[(df['rsi_momentum'] < 0) & (df['rsi_momentum'] > -5)])
    moderate_bearish = len(df[(df['rsi_momentum'] <= -5) & (df['rsi_momentum'] > -10)])
    strong_bearish = len(df[(df['rsi_momentum'] <= -10) & (df['rsi_momentum'] > -15)])
    very_strong_bearish = len(df[df['rsi_momentum'] <= -15])
    
    return {
        'very_strong_bullish': very_strong_bullish,
        'strong_bullish': strong_bullish,
        'moderate_bullish': moderate_bullish,
        'weak_bullish': weak_bullish,
        'weak_bearish': weak_bearish,
        'moderate_bearish': moderate_bearish,
        'strong_bearish': strong_bearish,
        'very_strong_bearish': very_strong_bearish,
        'total_bullish': very_strong_bullish + strong_bullish + moderate_bullish + weak_bullish,
        'total_bearish': weak_bearish + moderate_bearish + strong_bearish + very_strong_bearish,
        'momentum_dominance': 'bullish' if (very_strong_bullish + strong_bullish + moderate_bullish + weak_bullish) > (weak_bearish + moderate_bearish + strong_bearish + very_strong_bearish) else 'bearish'
    }


def analyze_volume_trends(df: pd.DataFrame) -> Dict:
    """
    Log volume trend analizi
    
    Args:
        df (pd.DataFrame): Sinyal verileri
        
    Returns:
        Dict: Volume trend analizi sonuçları
    """
    if df.empty or 'log_volume_strength' not in df.columns:
        return {}
    
    very_strong_volume = len(df[df['log_volume_strength'] >= 5.0])
    strong_volume = len(df[(df['log_volume_strength'] >= 3.0) & (df['log_volume_strength'] < 5.0)])
    moderate_volume = len(df[(df['log_volume_strength'] >= 1.5) & (df['log_volume_strength'] < 3.0)])
    weak_volume = len(df[(df['log_volume_strength'] >= 0.5) & (df['log_volume_strength'] < 1.5)])
    very_weak_volume = len(df[df['log_volume_strength'] < 0.5])
    
    return {
        'very_strong_volume': very_strong_volume,
        'strong_volume': strong_volume,
        'moderate_volume': moderate_volume,
        'weak_volume': weak_volume,
        'very_weak_volume': very_weak_volume,
        'avg_log_volume_strength': df['log_volume_strength'].mean(),
        'volume_quality_score': (very_strong_volume * 5 + strong_volume * 4 + moderate_volume * 3 + weak_volume * 2 + very_weak_volume * 1) / len(df) if len(df) > 0 else 0
    }


def get_momentum_volume_correlation_analysis(df: pd.DataFrame) -> Dict:
    """
    RSI momentum ile log volume arasındaki korelasyon analizi
    
    Args:
        df (pd.DataFrame): Sinyal verileri
        
    Returns:
        Dict: Korelasyon analizi
    """
    if df.empty or len(df) < 3:
        return {}
    
    correlations = {}
    try:
        required_cols = ['rsi_momentum', 'log_volume_strength', 'ai_score', 'run_perc', 'deviso_ratio']
        if all(col in df.columns for col in required_cols):
            correlations['momentum_volume_corr'] = df['rsi_momentum'].corr(df['log_volume_strength'])
            correlations['momentum_ai_corr'] = df['rsi_momentum'].corr(df['ai_score'])
            correlations['volume_ai_corr'] = df['log_volume_strength'].corr(df['ai_score'])
            correlations['momentum_deviso_corr'] = df['rsi_momentum'].corr(df['deviso_ratio'])
            correlations['volume_deviso_corr'] = df['log_volume_strength'].corr(df['deviso_ratio'])
                
    except Exception as e:
        logger.debug(f"Korelasyon hesaplama hatası: {e}")
    
    return correlations


# Deviso spesifik fonksiyonlar (aynı kalıyor)
def analyze_deviso_trends(df: pd.DataFrame) -> Dict:
    """Deviso trend analizi (değişiklik yok)"""
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
    """Deviso korelasyon analizi (güncellendi)"""
    if df.empty or len(df) < 3:
        return {}
    
    correlations = {}
    try:
        # 🆕 YENİ: RSI momentum ve log volume ile deviso korelasyonları
        required_cols = ['deviso_ratio', 'ai_score', 'run_perc', 'rsi_momentum', 'log_volume_strength']
        if all(col in df.columns for col in required_cols):
            correlations['deviso_ai_corr'] = df['deviso_ratio'].corr(df['ai_score'])
            correlations['deviso_run_corr'] = df['deviso_ratio'].corr(df['run_perc'])
            correlations['deviso_momentum_corr'] = df['deviso_ratio'].corr(df['rsi_momentum'])  # 🆕 YENİ
            correlations['deviso_volume_corr'] = df['deviso_ratio'].corr(df['log_volume_strength'])  # 🆕 YENİ
                
    except Exception as e:
        logger.debug(f"Deviso korelasyon hesaplama hatası: {e}")
    
    return correlations