"""
🔍 Sinyal Analiz Modülü - VPMV Sistemi
AI destekli kripto sinyal analizi ve batch processing
🔥 YENİ: VPMV (Volume-Price-Momentum-Volatility) + TIME Alignment
🔥 MTF (Multi-Timeframe) VPMV: 1H, 2H, 4H - TAM ENTEGRE
🔥 ESKİ SİSTEM KALDIRILDI: Deviso, Z-Score, Gauss, Log Volume
"""

import time
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    LOCAL_TZ, MAX_WORKERS, REQ_SLEEP, DEFAULT_MIN_AI_SCORE,
    DEFAULT_MIN_VPMV_SCORE, DEFAULT_MIN_TIME_MATCH,
    current_data, saved_signals
)
from data.fetch_data import fetch_klines, get_usdt_perp_symbols
from core.indicators import compute_vpmv_metrics
from core.ai_model import ai_model

logger = logging.getLogger("crypto-analytics")


def analyze_symbol_with_ai(symbol: str, interval: str) -> Dict:
    """
    🔥 GÜNCELLEME: VPMV bazlı sembol analizi + MTF VPMV + TIME Alignment
    
    Args:
        symbol (str): Trading sembolü
        interval (str): Zaman dilimi
        
    Returns:
        Dict: Analiz sonuçları (MTF VPMV dahil) veya boş dict
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
        
        # 🔥 GÜNCELLEME: VPMV metriklerini hesapla (symbol parametresi ile MTF + TIME alignment dahil)
        metrics = compute_vpmv_metrics(df, symbol)
        
        if not metrics or metrics.get('run_type') == 'none':
            logger.debug(f"❌ {symbol}: VPMV hesaplanamadı")
            return {}
        
        # 🔥 YENİ: VPMV skoru kontrolü
        vpmv_score = metrics.get('vpmv_score', 0.0)
        if abs(vpmv_score) < DEFAULT_MIN_VPMV_SCORE:
            logger.debug(f"❌ {symbol}: VPMV skoru düşük ({vpmv_score:.2f})")
            return {}
        
        # 🔥 YENİ: TIME alignment kontrolü
        time_match_count = metrics.get('time_match_count', 0)
        if time_match_count < DEFAULT_MIN_TIME_MATCH:
            logger.debug(f"❌ {symbol}: TIME uyumu yetersiz ({time_match_count}/5)")
            return {}
        
        # 🔥 YENİ: Bileşen değerlerini kontrol et
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
        
        # 🔥 YENİ: MTF VPMV verilerini al
        mtf_vpmv = metrics.get('mtf_vpmv', {})
        
        # 🔥 YENİ: MTF VPMV verilerini result'a ekle
        mtf_1h = mtf_vpmv.get('1H', {})
        mtf_2h = mtf_vpmv.get('2H', {})
        mtf_4h = mtf_vpmv.get('4H', {})
        
        # Başarılı sinyal
        result = {
            'symbol': symbol,
            'timeframe': interval,
            'last_close': last_close,
            'run_type': metrics['run_type'],  # 'long' veya 'short'
            
            # 🔥 Current Timeframe VPMV Bileşenleri
            'volume_component': metrics['volume_component'],
            'price_component': metrics['price_component'],
            'momentum_component': metrics['momentum_component'],
            'volatility_component': metrics['volatility_component'],
            'vpmv_score': metrics['vpmv_score'],
            
            # 🔥 YENİ: MTF VPMV - 1H
            'mtf_1h_volume': mtf_1h.get('volume', 0.0),
            'mtf_1h_price': mtf_1h.get('price', 0.0),
            'mtf_1h_momentum': mtf_1h.get('momentum', 0.0),
            'mtf_1h_volatility': mtf_1h.get('volatility', 0.0),
            'mtf_1h_vpmv': mtf_1h.get('vpmv_score', 0.0),
            'mtf_1h_trigger': mtf_1h.get('trigger', 'Yok'),
            
            # 🔥 YENİ: MTF VPMV - 2H
            'mtf_2h_volume': mtf_2h.get('volume', 0.0),
            'mtf_2h_price': mtf_2h.get('price', 0.0),
            'mtf_2h_momentum': mtf_2h.get('momentum', 0.0),
            'mtf_2h_volatility': mtf_2h.get('volatility', 0.0),
            'mtf_2h_vpmv': mtf_2h.get('vpmv_score', 0.0),
            'mtf_2h_trigger': mtf_2h.get('trigger', 'Yok'),
            
            # 🔥 YENİ: MTF VPMV - 4H
            'mtf_4h_volume': mtf_4h.get('volume', 0.0),
            'mtf_4h_price': mtf_4h.get('price', 0.0),
            'mtf_4h_momentum': mtf_4h.get('momentum', 0.0),
            'mtf_4h_volatility': mtf_4h.get('volatility', 0.0),
            'mtf_4h_vpmv': mtf_4h.get('vpmv_score', 0.0),
            'mtf_4h_trigger': mtf_4h.get('trigger', 'Yok'),
            
            # 🔥 TIME Alignment
            'time_match_count': metrics['time_match_count'],
            'time_directions': metrics.get('time_directions', {}),
            
            # 🔥 Current Timeframe Tetikleyici
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
            f"✅ {symbol}: VPMV={vpmv_score:+.1f}, TIME={time_match_count}/5, "
            f"AI={ai_score:.0f}%, Trigger={metrics.get('trigger_type', 'Yok')}, "
            f"MTF: 1H={mtf_1h.get('vpmv_score', 0):+.1f} 2H={mtf_2h.get('vpmv_score', 0):+.1f} 4H={mtf_4h.get('vpmv_score', 0):+.1f}"
        )
        return result
        
    except Exception as e:
        logger.warning(f"analyze_symbol error {symbol}: {e}")
        return {}


def batch_analyze_with_ai(interval: str) -> pd.DataFrame:
    """
    🔥 GÜNCELLEME: VPMV bazlı toplu analiz + MTF VPMV + TIME Alignment
    
    Args:
        interval (str): Analiz edilecek zaman dilimi
        
    Returns:
        pd.DataFrame: Analiz sonuçları (MTF VPMV dahil)
    """
    global saved_signals
    
    start_time = time.time()
    
    # Sembol listesini al
    symbols = get_usdt_perp_symbols()
    if not symbols:
        logger.error("Sembol listesi boş!")
        return pd.DataFrame()
    
    logger.info(f"🤖 {len(symbols)} sembol için VPMV + MTF + TIME Alignment analiz başlatılıyor...")
    
    # Yeni analiz sonuçları
    fresh_results = []
    processed_count = 0
    vpmv_success_count = 0
    high_time_alignment_count = 0  # TIME uyumu yüksek olanlar
    mtf_success_count = 0  # MTF verileri başarılı olanlar
    
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
                    
                    # TIME alignment yüksek olanları say
                    if res.get('time_match_count', 0) >= 4:
                        high_time_alignment_count += 1
                    
                    # 🔥 YENİ: MTF verileri mevcut mu kontrol et
                    if res.get('mtf_1h_vpmv', 0.0) != 0.0 or res.get('mtf_2h_vpmv', 0.0) != 0.0:
                        mtf_success_count += 1
                    
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
                        f"VPMV: {success_rate:.1f}% ({vpmv_success_count}) - "
                        f"Yüksek TIME: {high_time_alignment_count} - MTF: {mtf_success_count}"
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
            
            # 🔥 YENİ: VPMV gücüne göre ceza
            vpmv_score = abs(old_data.get('vpmv_score', 0))
            time_match = old_data.get('time_match_count', 0)
            
            # VPMV güçlü + TIME uyumu yüksek = az ceza
            if vpmv_score > 30.0 and time_match >= 4:
                base_penalty = 10
            elif vpmv_score > 20.0 and time_match >= 3:
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
                f"(yaş: {minutes_old:.1f}dk, VPMV: {vpmv_score:.1f}, TIME: {time_match}/5)"
            )
    
    # Performans istatistikleri
    elapsed_time = time.time() - start_time
    total_rate = len(symbols) / elapsed_time if elapsed_time > 0 else 0
    
    new_signals = len(fresh_symbols)
    total_signals = len(fresh_results)
    vpmv_success_rate = (vpmv_success_count / len(symbols)) * 100 if len(symbols) > 0 else 0
    
    logger.info("✅ VPMV + MTF + TIME Alignment Analiz tamamlandı:")
    logger.info(f"   🆕 Yeni sinyal: {new_signals}")
    logger.info(f"   📉 Korunan sinyal: {protected_count}")
    logger.info(f"   🎯 Toplam sinyal: {total_signals}")
    logger.info(f"   📊 VPMV başarı oranı: {vpmv_success_rate:.1f}%")
    logger.info(f"   ⏰ Yüksek TIME uyumu: {high_time_alignment_count}/{vpmv_success_count}")
    logger.info(f"   📈 MTF veri başarı: {mtf_success_count}/{vpmv_success_count}")
    logger.info(f"   ⏱️ Süre: {elapsed_time:.1f}s - Hız: {total_rate:.1f} s/sn")
    
    if not fresh_results:
        logger.warning("⚠️ Hiç sinyal bulunamadı - filtreleri gözden geçirin")
        return pd.DataFrame()
    
    # DataFrame oluştur ve sırala - 🔥 YENİ SIRALAMA (MTF dahil)
    df = pd.DataFrame(fresh_results)
    df = df.sort_values(
        by=['ai_score', 'vpmv_score', 'time_match_count', 'mtf_1h_vpmv', 'volatility_component'],
        ascending=[False, False, False, False, False]
    )
    
    if len(df) > 0:
        top_signal = df.iloc[0]
        logger.info(
            f"🏆 En yüksek AI skoru: {top_signal['ai_score']:.0f}% - {top_signal['symbol']} "
            f"(VPMV: {top_signal['vpmv_score']:+.1f}, TIME: {top_signal['time_match_count']}/5, "
            f"MTF: 1H={top_signal.get('mtf_1h_vpmv', 0):+.1f} 2H={top_signal.get('mtf_2h_vpmv', 0):+.1f} 4H={top_signal.get('mtf_4h_vpmv', 0):+.1f}, "
            f"Trigger: {top_signal.get('trigger_type', 'Yok')})"
        )
        
        # Run type dağılımı
        long_count = len(df[df['run_type'] == 'long'])
        short_count = len(df[df['run_type'] == 'short'])
        logger.info(f"📈 Sinyal dağılımı: LONG={long_count}, SHORT={short_count}")
        
        # TIME alignment istatistikleri
        high_time = len(df[df['time_match_count'] >= 4])
        medium_time = len(df[(df['time_match_count'] >= 3) & (df['time_match_count'] < 4)])
        low_time = len(df[df['time_match_count'] < 3])
        logger.info(f"⏰ TIME dağılımı: Yüksek(4-5)={high_time}, Orta(3)={medium_time}, Düşük(<3)={low_time}")
        
        # 🔥 YENİ: MTF istatistikleri
        mtf_positive_1h = len(df[df['mtf_1h_vpmv'] > 0])
        mtf_positive_2h = len(df[df['mtf_2h_vpmv'] > 0])
        mtf_positive_4h = len(df[df['mtf_4h_vpmv'] > 0])
        logger.info(f"📊 MTF Pozitif: 1H={mtf_positive_1h}, 2H={mtf_positive_2h}, 4H={mtf_positive_4h}")
        
        if protected_count > 0:
            logger.info("📉 Korunan sinyaller skor düşüşü ile aşağı kaydı")
        
        logger.debug("📊 Analyzer sonrası ilk 3 sinyal:")
        for idx, (i, row) in enumerate(df.head(3).iterrows(), 1):
            logger.debug(
                f"   {idx}: {row['symbol']} | AI={row['ai_score']:.0f}% | "
                f"VPMV={row['vpmv_score']:+.1f} | TIME={row['time_match_count']}/5 | "
                f"MTF: 1H={row.get('mtf_1h_vpmv', 0):+.1f} 2H={row.get('mtf_2h_vpmv', 0):+.1f} 4H={row.get('mtf_4h_vpmv', 0):+.1f}"
            )
    
    return df


def filter_signals(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    🔥 GÜNCELLEME: VPMV + MTF bazlı sinyal filtreleme
    
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
    
    # 🔥 TIME alignment filtresi
    if filters.get('min_time_match', 0) > 0:
        filtered_df = filtered_df[filtered_df['time_match_count'] >= filters['min_time_match']]
    
    # 🔥 YENİ: MTF filtreleri
    if filters.get('min_mtf_1h_vpmv', 0) > 0:
        filtered_df = filtered_df[abs(filtered_df.get('mtf_1h_vpmv', 0)) >= filters['min_mtf_1h_vpmv']]
    
    if filters.get('min_mtf_2h_vpmv', 0) > 0:
        filtered_df = filtered_df[abs(filtered_df.get('mtf_2h_vpmv', 0)) >= filters['min_mtf_2h_vpmv']]
    
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
    🔥 GÜNCELLEME: En iyi VPMV + MTF sinyalleri al
    
    Args:
        df (pd.DataFrame): Sinyal DataFrame'i
        count (int): Alınacak sinyal sayısı
        
    Returns:
        pd.DataFrame: En iyi sinyaller
    """
    if df.empty:
        return df
    
    sorted_df = df.sort_values(
        by=['ai_score', 'vpmv_score', 'time_match_count', 'mtf_1h_vpmv', 'volatility_component'],
        ascending=[False, False, False, False, False]
    )
    
    return sorted_df.head(count)


def analyze_signal_quality(df: pd.DataFrame) -> Dict:
    """
    🔥 GÜNCELLEME: VPMV + MTF bazlı sinyal kalitesi analizi
    
    Args:
        df (pd.DataFrame): Analiz edilecek sinyaller
        
    Returns:
        Dict: Kalite metrikleri (MTF dahil)
    """
    if df.empty:
        return {
            'total_signals': 0,
            'avg_ai_score': 0,
            'long_signals': 0,
            'short_signals': 0,
            'high_quality_signals': 0,
            'quality_distribution': {},
            'vpmv_quality': {},
            'time_quality': {},
            'mtf_quality': {}  # 🔥 YENİ
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
    
    # 🔥 TIME alignment kalite analizi
    time_quality = {}
    if 'time_match_count' in df.columns:
        avg_time_match = df['time_match_count'].mean()
        perfect_time = len(df[df['time_match_count'] == 5])
        high_time = len(df[df['time_match_count'] == 4])
        medium_time = len(df[df['time_match_count'] == 3])
        low_time = len(df[df['time_match_count'] <= 2])
        
        time_quality = {
            'avg_time_match': avg_time_match,
            'perfect_alignment': perfect_time,
            'high_alignment': high_time,
            'medium_alignment': medium_time,
            'low_alignment': low_time
        }
    
    # 🔥 YENİ: MTF kalite analizi
    mtf_quality = {}
    if 'mtf_1h_vpmv' in df.columns:
        mtf_quality['1h'] = {
            'avg': df['mtf_1h_vpmv'].mean(),
            'positive_count': len(df[df['mtf_1h_vpmv'] > 0]),
            'strong_count': len(df[abs(df['mtf_1h_vpmv']) >= 30.0])
        }
    if 'mtf_2h_vpmv' in df.columns:
        mtf_quality['2h'] = {
            'avg': df['mtf_2h_vpmv'].mean(),
            'positive_count': len(df[df['mtf_2h_vpmv'] > 0]),
            'strong_count': len(df[abs(df['mtf_2h_vpmv']) >= 30.0])
        }
    if 'mtf_4h_vpmv' in df.columns:
        mtf_quality['4h'] = {
            'avg': df['mtf_4h_vpmv'].mean(),
            'positive_count': len(df[df['mtf_4h_vpmv'] > 0]),
            'strong_count': len(df[abs(df['mtf_4h_vpmv']) >= 30.0])
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
        'vpmv_quality': vpmv_quality,
        'time_quality': time_quality,
        'mtf_quality': mtf_quality  # 🔥 YENİ
    }


def update_signal_scores():
    """
    🔥 YENİ: Kayıtlı sinyallerin skorlarını güncelle (VPMV + MTF bazlı yaş cezası)
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
            
            # 🔥 YENİ: VPMV ve TIME bazlı ceza
            vpmv_score = abs(saved_info['data'].get('vpmv_score', 0))
            time_match = saved_info['data'].get('time_match_count', 0)
            
            # Güçlü sinyal = az ceza
            if vpmv_score > 30.0 and time_match >= 4:
                base_penalty = 10
            elif vpmv_score > 20.0 and time_match >= 3:
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
    🔥 GÜNCELLEME: VPMV + MTF bazlı sinyal özeti
    
    Returns:
        Dict: Sinyal özet bilgileri (MTF dahil)
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
            'vpmv_stats': {},
            'time_stats': {},
            'mtf_stats': {}  # 🔥 YENİ
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
    
    # 🔥 TIME alignment istatistikleri
    time_stats = {}
    if 'time_match_count' in current_data.columns and not current_data.empty:
        time_stats = {
            'avg_time_match': current_data['time_match_count'].mean(),
            'perfect_time': len(current_data[current_data['time_match_count'] == 5]),
            'high_time': len(current_data[current_data['time_match_count'] >= 4]),
            'medium_time': len(current_data[current_data['time_match_count'] == 3]),
            'low_time': len(current_data[current_data['time_match_count'] <= 2])
        }
    
    # 🔥 YENİ: MTF istatistikleri
    mtf_stats = {}
    if 'mtf_1h_vpmv' in current_data.columns and not current_data.empty:
        mtf_stats['1h'] = {
            'avg': current_data['mtf_1h_vpmv'].mean(),
            'positive': len(current_data[current_data['mtf_1h_vpmv'] > 0])
        }
    if 'mtf_2h_vpmv' in current_data.columns and not current_data.empty:
        mtf_stats['2h'] = {
            'avg': current_data['mtf_2h_vpmv'].mean(),
            'positive': len(current_data[current_data['mtf_2h_vpmv'] > 0])
        }
    if 'mtf_4h_vpmv' in current_data.columns and not current_data.empty:
        mtf_stats['4h'] = {
            'avg': current_data['mtf_4h_vpmv'].mean(),
            'positive': len(current_data[current_data['mtf_4h_vpmv'] > 0])
        }
    
    return {
        'total_signals': total_signals,
        'long_count': long_count,
        'short_count': short_count,
        'avg_ai_score': avg_ai_score,
        'top_symbol': top_symbol,
        'last_update': datetime.now(LOCAL_TZ).strftime('%H:%M:%S'),
        'vpmv_stats': vpmv_stats,
        'time_stats': time_stats,
        'mtf_stats': mtf_stats  # 🔥 YENİ
    }


# 🔥 VPMV spesifik analiz fonksiyonları
def analyze_vpmv_components(df: pd.DataFrame) -> Dict:
    """
    VPMV bileşenlerini analiz et
    
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