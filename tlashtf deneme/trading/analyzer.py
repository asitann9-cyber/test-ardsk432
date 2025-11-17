"""
🔍 Sinyal Analiz Modülü - ULTRA PANEL v5
AI destekli kripto sinyal analizi ve batch processing
🔥 YENİ: Multi-timeframe Heikin Ashi crossover bazlı analiz
🔥 ÖZELLIKLER: HTF Count, Power, Whale detection
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
    🔥 ULTRA PANEL v5: Multi-timeframe Heikin Ashi analizi ile sinyal tespiti
    
    Args:
        symbol (str): Trading sembolü
        interval (str): Timeframe
        
    Returns:
        Dict: Ultra Panel v5 analiz sonuçları veya boş dict
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
        
        # 🔥 ULTRA PANEL v5 METRİKLERİNİ HESAPLA
        metrics = compute_consecutive_metrics(df)
        
        if not metrics:
            logger.debug(f"❌ {symbol}: Metrik hesaplanamadı")
            return {}
        
        # 🔥 ULTRA SİNYAL KONTROLÜ
        ultra_signal = metrics.get('ultra_signal', 'NONE')
        if ultra_signal == 'NONE':
            logger.debug(f"❌ {symbol}: Ultra sinyal yok")
            return {}
        
        # 🔥 HTF COUNT KONTROLÜ (minimum 3/4 olmalı)
        htf_count = metrics.get('htf_count', 0)
        if htf_count < 3:
            logger.debug(f"❌ {symbol}: HTF count yetersiz ({htf_count}/4)")
            return {}
        
        # 🔥 POWER KONTROLÜ (minimum 1.0 olmalı)
        total_power = metrics.get('total_power', 0.0)
        if total_power < 1.0:
            logger.debug(f"❌ {symbol}: Power çok düşük ({total_power:.2f})")
            return {}
        
        # 🔥 AI SKORU HESAPLA
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
        
        # Whale durumu
        whale_active = metrics.get('whale_active', False)
        
        # 🔥 BAŞARILI ULTRA SİNYAL
        result = {
            'symbol': symbol,
            'timeframe': interval,
            'last_close': last_close,
            'run_type': metrics['run_type'],  # long/short
            
            # 🔥 ULTRA PANEL v5 METRİKLERİ
            'ultra_signal': ultra_signal,  # BUY/SELL
            'htf_count': htf_count,  # 3/4 veya 4/4
            'total_power': total_power,  # Kümülatif güç
            'whale_active': whale_active,  # True/False
            
            'ai_score': ai_score,
            'last_update': last_update.strftime('%Y-%m-%d %H:%M:%S %Z'),
        }
        
        # Whale uyarısı
        if whale_active:
            logger.debug(f"🐋 {symbol}: Whale aktif!")
        
        logger.debug(
            f"✅ {symbol}: Ultra sinyal onaylandı - "
            f"{ultra_signal} | HTF:{htf_count}/4 | Power:{total_power:.1f} | AI:{ai_score:.0f}%"
        )
        return result
        
    except Exception as e:
        logger.warning(f"analyze_symbol error {symbol}: {e}")
        return {}


def batch_analyze_with_ai(interval: str) -> pd.DataFrame:
    """
    🔥 ULTRA PANEL v5: Batch analiz - Tüm sembolleri tara
    
    Args:
        interval (str): Analiz edilecek zaman dilimi
        
    Returns:
        pd.DataFrame: Ultra Panel v5 analiz sonuçları
    """
    global saved_signals
    
    start_time = time.time()
    
    # Sembol listesini al
    symbols = get_usdt_perp_symbols()
    if not symbols:
        logger.error("Sembol listesi boş!")
        return pd.DataFrame()
    
    logger.info(f"🤖 {len(symbols)} sembol için ULTRA PANEL v5 analizi başlatılıyor...")
    
    # Yeni analiz sonuçları
    fresh_results = []
    processed_count = 0
    ultra_signal_count = 0
    whale_count = 0
    htf_4_count = 0
    
    # Paralel işleme
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(analyze_symbol_with_ai, sym, interval): sym for sym in symbols}
        
        for fut in as_completed(futures):
            symbol = futures[fut]
            processed_count += 1
            
            try:
                res = fut.result()
                if res:  # Geçerli ultra sinyal
                    fresh_results.append(res)
                    ultra_signal_count += 1
                    
                    # İstatistikler
                    if res.get('whale_active', False):
                        whale_count += 1
                    if res.get('htf_count', 0) == 4:
                        htf_4_count += 1
                    
                    # Kaydedilmiş sinyalleri güncelle
                    saved_signals[symbol] = {
                        'data': res,
                        'last_seen': datetime.now(LOCAL_TZ)
                    }
                
                # İlerleme logu
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    success_rate = (ultra_signal_count / processed_count) * 100
                    logger.info(
                        f"🤖 İşlenen: {processed_count}/{len(symbols)} - Hız: {rate:.1f} s/sn - "
                        f"Ultra Sinyal: {success_rate:.1f}% ({ultra_signal_count}) | "
                        f"4/4 HTF: {htf_4_count} | Whale: {whale_count}"
                    )
            
            except Exception as e:
                logger.debug(f"Future hatası {symbol}: {e}")
    
    # Mevcut zaman
    current_time = datetime.now(LOCAL_TZ)
    fresh_symbols = {r['symbol'] for r in fresh_results}
    
    # 🔥 ESKİ SİNYALLERİ KORUMA VE SKOR DÜŞÜRME
    protected_count = 0
    for symbol, saved_info in list(saved_signals.items()):
        minutes_old = (current_time - saved_info['last_seen']).total_seconds() / 60.0
        
        # 15 dakikadan eski sinyalleri sil
        if minutes_old > 15:
            del saved_signals[symbol]
            continue
        
        if symbol not in fresh_symbols:
            old_data = saved_info['data'].copy()
            original_score = old_data['ai_score']
            
            # 🔥 ULTRA PANEL v5 BAZLI CEZA SİSTEMİ
            htf_count = old_data.get('htf_count', 0)
            total_power = old_data.get('total_power', 0.0)
            
            # HTF count ve power'a göre ceza
            if htf_count == 4 and total_power > 20:
                base_penalty = 5  # Güçlü sinyal, az ceza
            elif htf_count == 4:
                base_penalty = 10
            elif htf_count == 3 and total_power > 15:
                base_penalty = 15
            else:
                base_penalty = 25  # Zayıf sinyal, çok ceza
            
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
                f"(yaş: {minutes_old:.1f}dk, HTF:{htf_count}/4, Power:{total_power:.1f})"
            )
    
    # Performans istatistikleri
    elapsed_time = time.time() - start_time
    total_rate = len(symbols) / elapsed_time if elapsed_time > 0 else 0
    
    new_signals = len(fresh_symbols)
    total_signals = len(fresh_results)
    ultra_success_rate = (ultra_signal_count / len(symbols)) * 100 if len(symbols) > 0 else 0
    
    logger.info("✅ ULTRA PANEL v5 Analiz tamamlandı:")
    logger.info(f"   🆕 Yeni ultra sinyal: {new_signals}")
    logger.info(f"   📉 Korunan sinyal: {protected_count}")
    logger.info(f"   🎯 Toplam sinyal: {total_signals}")
    logger.info(f"   📊 Ultra başarı oranı: {ultra_success_rate:.1f}%")
    logger.info(f"   🔥 4/4 HTF: {htf_4_count} | 🐋 Whale: {whale_count}")
    logger.info(f"   ⏱️ Süre: {elapsed_time:.1f}s - Hız: {total_rate:.1f} s/sn")
    
    if not fresh_results:
        logger.warning("⚠️ Hiç ultra sinyal bulunamadı")
        return pd.DataFrame()
    
    # DataFrame oluştur ve sırala
    df = pd.DataFrame(fresh_results)
    
    # 🔥 ULTRA PANEL v5 SIRALAMA
    # 1. AI Score (en önemli)
    # 2. HTF Count (4/4 önce)
    # 3. Total Power (güç)
    # 4. Whale Active (balina varsa önce)
    df = df.sort_values(
        by=['ai_score', 'htf_count', 'total_power', 'whale_active'],
        ascending=[False, False, False, False]
    )
    
    if len(df) > 0:
        top_signal = df.iloc[0]
        whale_emoji = "🐋" if top_signal['whale_active'] else ""
        logger.info(
            f"🏆 En yüksek AI skoru: {top_signal['ai_score']:.0f}% - {top_signal['symbol']} "
            f"| {top_signal['ultra_signal']} | HTF:{top_signal['htf_count']}/4 | "
            f"Power:{top_signal['total_power']:.1f} {whale_emoji}"
        )
        
        # Sinyal tipi dağılımı
        buy_count = len(df[df['ultra_signal'] == 'BUY'])
        sell_count = len(df[df['ultra_signal'] == 'SELL'])
        logger.info(f"📈 Sinyal dağılımı: BUY:{buy_count} | SELL:{sell_count}")
        
        if protected_count > 0:
            logger.info("📉 Korunan sinyaller skor düşüşü ile aşağı kaydı")
        
        # İlk 5 sinyal
        logger.debug("📊 Analyzer sonrası ilk 5 sinyal:")
        for i, row in df.head(5).iterrows():
            whale_emoji = "🐋" if row['whale_active'] else ""
            logger.debug(
                f"   {i+1}: {row['symbol']} | AI={row['ai_score']:.0f}% | "
                f"{row['ultra_signal']} | HTF={row['htf_count']}/4 | "
                f"Power={row['total_power']:.1f} {whale_emoji}"
            )
    
    return df


def filter_signals(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    🔥 ULTRA PANEL v5: Sinyalleri filtrele
    
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
    
    # AI Score filtresi
    if filters.get('min_ai_score', 0) > 0:
        filtered_df = filtered_df[filtered_df['ai_score'] >= filters['min_ai_score']]
    
    # HTF Count filtresi
    if filters.get('min_htf_count', 0) > 0:
        filtered_df = filtered_df[filtered_df['htf_count'] >= filters['min_htf_count']]
    
    # Power filtresi
    if filters.get('min_power', 0) > 0:
        filtered_df = filtered_df[filtered_df['total_power'] >= filters['min_power']]
    
    # Whale filtresi (sadece whale olanlar)
    if filters.get('whale_only', False):
        filtered_df = filtered_df[filtered_df['whale_active'] == True]
    
    # Sinyal tipi filtresi (BUY/SELL/all)
    signal_type_filter = filters.get('signal_type')
    if signal_type_filter and signal_type_filter != 'all':
        filtered_df = filtered_df[filtered_df['ultra_signal'] == signal_type_filter]
    
    filtered_count = len(filtered_df)
    logger.info(f"🔍 Filtre sonucu: {filtered_count}/{original_count} sinyal kaldı")
    
    return filtered_df


def get_top_signals(df: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    """
    🔥 ULTRA PANEL v5: En iyi sinyalleri al
    
    Args:
        df (pd.DataFrame): Sinyal DataFrame'i
        count (int): Alınacak sinyal sayısı
        
    Returns:
        pd.DataFrame: En iyi sinyaller
    """
    if df.empty:
        return df
    
    sorted_df = df.sort_values(
        by=['ai_score', 'htf_count', 'total_power', 'whale_active'],
        ascending=[False, False, False, False]
    )
    
    return sorted_df.head(count)


def analyze_signal_quality(df: pd.DataFrame) -> Dict:
    """
    🔥 ULTRA PANEL v5: Sinyal kalitesi analizi
    
    Args:
        df (pd.DataFrame): Analiz edilecek sinyaller
        
    Returns:
        Dict: Kalite metrikleri
    """
    if df.empty:
        return {
            'total_signals': 0,
            'avg_ai_score': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'high_quality_signals': 0,
            'htf_distribution': {},
            'whale_count': 0
        }
    
    total_signals = len(df)
    avg_ai_score = df['ai_score'].mean()
    
    # Sinyal tipi dağılımı
    buy_signals = len(df[df['ultra_signal'] == 'BUY'])
    sell_signals = len(df[df['ultra_signal'] == 'SELL'])
    
    # Kalite kategorileri
    high_quality = len(df[df['ai_score'] >= 80])
    medium_quality = len(df[(df['ai_score'] >= 60) & (df['ai_score'] < 80)])
    low_quality = len(df[df['ai_score'] < 60])
    
    # HTF dağılımı
    htf_distribution = {}
    if 'htf_count' in df.columns:
        htf_distribution = {
            '4/4': len(df[df['htf_count'] == 4]),
            '3/4': len(df[df['htf_count'] == 3]),
            '2/4': len(df[df['htf_count'] == 2]),
            '1/4': len(df[df['htf_count'] == 1])
        }
    
    # Whale count
    whale_count = len(df[df['whale_active'] == True]) if 'whale_active' in df.columns else 0
    
    # Power analizi
    power_stats = {}
    if 'total_power' in df.columns:
        power_stats = {
            'avg_power': df['total_power'].mean(),
            'max_power': df['total_power'].max(),
            'high_power': len(df[df['total_power'] >= 20]),  # 20+ yüksek power
            'medium_power': len(df[(df['total_power'] >= 10) & (df['total_power'] < 20)]),
            'low_power': len(df[df['total_power'] < 10])
        }
    
    return {
        'total_signals': total_signals,
        'avg_ai_score': avg_ai_score,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'high_quality_signals': high_quality,
        'quality_distribution': {
            'high': high_quality,
            'medium': medium_quality,
            'low': low_quality
        },
        'buy_ratio': (buy_signals / total_signals * 100) if total_signals > 0 else 0,
        'sell_ratio': (sell_signals / total_signals * 100) if total_signals > 0 else 0,
        'htf_distribution': htf_distribution,
        'whale_count': whale_count,
        'power_stats': power_stats
    }


def update_signal_scores():
    """
    🔥 ULTRA PANEL v5: Kayıtlı sinyallerin skorlarını güncelle (yaş cezası)
    """
    global saved_signals
    
    current_time = datetime.now(LOCAL_TZ)
    updated_count = 0
    removed_count = 0
    
    for symbol, saved_info in list(saved_signals.items()):
        minutes_old = (current_time - saved_info['last_seen']).total_seconds() / 60.0
        
        # Çok eski sinyalleri sil (20 dakika)
        if minutes_old > 20:
            del saved_signals[symbol]
            removed_count += 1
            continue
        
        # Yaşlanma cezası uygula
        if minutes_old > 3:
            original_score = saved_info['data']['ai_score']
            
            # 🔥 HTF ve Power bazlı ceza
            htf_count = saved_info['data'].get('htf_count', 0)
            total_power = saved_info['data'].get('total_power', 0.0)
            
            if htf_count == 4 and total_power > 20:
                base_penalty = 5  # Güçlü sinyal
            elif htf_count == 4:
                base_penalty = 10
            elif htf_count == 3:
                base_penalty = 20
            else:
                base_penalty = 30  # Zayıf sinyal
            
            # Yaşa göre ek ceza
            if minutes_old <= 7:
                penalty = base_penalty
            elif minutes_old <= 15:
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
    🔥 ULTRA PANEL v5: Sinyal özetini al
    
    Returns:
        Dict: Sinyal özet bilgileri
    """
    global current_data
    
    if current_data is None or current_data.empty:
        return {
            'total_signals': 0,
            'buy_count': 0,
            'sell_count': 0,
            'avg_ai_score': 0,
            'top_symbol': None,
            'last_update': None,
            'htf_4_count': 0,
            'whale_count': 0
        }
    
    total_signals = len(current_data)
    buy_count = len(current_data[current_data['ultra_signal'] == 'BUY'])
    sell_count = len(current_data[current_data['ultra_signal'] == 'SELL'])
    avg_ai_score = current_data['ai_score'].mean()
    
    # En yüksek skorlu sembol
    top_signal = current_data.iloc[0] if not current_data.empty else None
    top_symbol = top_signal['symbol'] if top_signal is not None else None
    
    # 4/4 HTF count
    htf_4_count = len(current_data[current_data['htf_count'] == 4]) if 'htf_count' in current_data.columns else 0
    
    # Whale count
    whale_count = len(current_data[current_data['whale_active'] == True]) if 'whale_active' in current_data.columns else 0
    
    return {
        'total_signals': total_signals,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'avg_ai_score': avg_ai_score,
        'top_symbol': top_symbol,
        'last_update': datetime.now(LOCAL_TZ).strftime('%H:%M:%S'),
        'htf_4_count': htf_4_count,
        'whale_count': whale_count
    }