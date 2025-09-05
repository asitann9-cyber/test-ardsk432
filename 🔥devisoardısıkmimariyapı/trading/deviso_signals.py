"""
🎯 Deviso Signals - Timeframe Adaptive Sinyal Analizi
Multi-timeframe AI analizi ve en iyi 3 seçim algoritması
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from config import (
    LOCAL_TZ, DEVISO_TIMEFRAME_SIGNALS, DEVISO_LIVE_TEST, DEVISO_LIVE_TRADING,
    get_strategy_from_timeframe, get_adaptive_ai_threshold
)
from data.fetch_data import fetch_klines
from core.indicators import compute_consecutive_metrics
from core.ai_model import ai_model

logger = logging.getLogger("crypto-analytics")


def calculate_timeframe_adaptive_signals(symbol: str, primary_timeframe: str) -> Dict:
    """
    Seçilen timeframe'e göre adaptive sinyal analizi
    
    Args:
        symbol (str): Analiz edilecek sembol
        primary_timeframe (str): Ana timeframe (1m, 5m, 15m, 1h, 4h)
        
    Returns:
        Dict: Sinyal analizi sonucu
    """
    try:
        # Timeframe stratejisi belirleme
        strategy = get_strategy_from_timeframe(primary_timeframe)
        
        if strategy == 'scalping':
            analysis_tfs = ['1m', '5m']
            signal_types = ['EMA_CROSS', 'RSI_OVERSOLD', 'MACD_MOMENTUM']
            weights = [0.7, 0.3]  # Kısa TF ağırlığı
        elif strategy == 'swing':
            analysis_tfs = ['15m', '1h']
            signal_types = ['EMA_TREND', 'RSI_DIVERGENCE', 'MACD_SIGNAL']
            weights = [0.5, 0.5]  # Dengeli ağırlık
        else:  # position
            analysis_tfs = ['1h', '4h']
            signal_types = ['EMA_ALIGNMENT', 'RSI_ZONES', 'MACD_TREND']
            weights = [0.3, 0.7]  # Uzun TF ağırlığı
        
        # Her timeframe için analiz
        timeframe_results = {}
        total_weighted_score = 0
        
        for i, tf in enumerate(analysis_tfs):
            try:
                # Veri çek
                df = fetch_klines(symbol, tf)
                if df is None or df.empty:
                    continue
                
                # Teknik analiz
                metrics = compute_consecutive_metrics(df)
                if not metrics or metrics['run_type'] == 'none':
                    continue
                
                # AI skoru hesapla
                ai_score = ai_model.predict_score(metrics)
                
                # Timeframe sonucu
                tf_result = {
                    'timeframe': tf,
                    'metrics': metrics,
                    'ai_score': ai_score,
                    'weight': weights[i] if i < len(weights) else 0.5
                }
                
                timeframe_results[tf] = tf_result
                total_weighted_score += ai_score * tf_result['weight']
                
            except Exception as e:
                logger.debug(f"TF analiz hatası {symbol} {tf}: {e}")
                continue
        
        # Sonuç yok ise
        if not timeframe_results:
            return {}
        
        # Trend tutarlılığı kontrolü
        trend_consistency = check_trend_consistency(timeframe_results)
        
        # Final AI skoru (ağırlıklı ortalama + consistency bonus)
        final_ai_score = total_weighted_score + (trend_consistency * 10)
        final_ai_score = min(final_ai_score, 100.0)  # Max 100
        
        # Ana timeframe'den temel bilgileri al
        main_tf_data = timeframe_results.get(primary_timeframe)
        if not main_tf_data and timeframe_results:
            # Ana TF yoksa ilk TF'yi kullan
            main_tf_data = list(timeframe_results.values())[0]
        
        if not main_tf_data:
            return {}
        
        return {
            'symbol': symbol,
            'primary_timeframe': primary_timeframe,
            'strategy': strategy,
            'ai_score': final_ai_score,
            'timeframe_breakdown': timeframe_results,
            'trend_consistency': trend_consistency,
            'signal_strength': 'STRONG' if final_ai_score >= 90 else 'MEDIUM' if final_ai_score >= 70 else 'WEAK',
            'run_type': main_tf_data['metrics']['run_type'],
            'run_count': main_tf_data['metrics']['run_count'],
            'run_perc': main_tf_data['metrics']['run_perc'],
            'deviso_ratio': main_tf_data['metrics']['deviso_ratio'],
            'vol_ratio': main_tf_data['metrics']['vol_ratio'],
            'timestamp': datetime.now(LOCAL_TZ).isoformat()
        }
        
    except Exception as e:
        logger.debug(f"Adaptive sinyal analizi hatası {symbol}: {e}")
        return {}


def check_trend_consistency(timeframe_results: Dict) -> float:
    """
    Timeframe'ler arası trend tutarlılığını kontrol et
    
    Args:
        timeframe_results (Dict): Timeframe analiz sonuçları
        
    Returns:
        float: Tutarlılık skoru (0-1)
    """
    try:
        if len(timeframe_results) < 2:
            return 0.5  # Tek TF varsa nötr
        
        # Trend yönlerini topla
        trends = []
        for tf_data in timeframe_results.values():
            run_type = tf_data['metrics']['run_type']
            if run_type in ['long', 'short']:
                trends.append(run_type)
        
        if not trends:
            return 0.0
        
        # Tutarlılık hesapla
        long_count = trends.count('long')
        short_count = trends.count('short')
        total_count = len(trends)
        
        # Aynı yönde olan trend oranı
        consistency = max(long_count, short_count) / total_count
        
        return consistency
        
    except Exception as e:
        logger.debug(f"Trend tutarlılığı hatası: {e}")
        return 0.5


def calculate_ranking_score_with_timeframe(signal_info: Dict) -> float:
    """
    Timeframe bilgisi dahil sıralama skoru hesapla
    
    Args:
        signal_info (Dict): Sinyal bilgileri
        
    Returns:
        float: Ranking skoru
    """
    try:
        # Temel AI skoru
        base_score = signal_info.get('ai_score', 0)
        
        # Timeframe bonus/penalty
        primary_tf = signal_info.get('primary_timeframe', '15m')
        strategy = signal_info.get('strategy', 'swing')
        
        # Timeframe uyum bonusu
        timeframe_bonus = 0
        if strategy == 'scalping' and primary_tf in ['1m', '5m']:
            timeframe_bonus = 10  # Scalping için kısa TF bonusu
        elif strategy == 'swing' and primary_tf in ['15m', '1h']:
            timeframe_bonus = 8   # Swing için orta TF bonusu
        elif strategy == 'position' and primary_tf in ['1h', '4h']:
            timeframe_bonus = 6   # Position için uzun TF bonusu
        
        # Trend tutarlılığı bonusu
        consistency_bonus = 0
        trend_consistency = signal_info.get('trend_consistency', 0)
        if trend_consistency >= 0.8:  # %80+ tutarlılık
            consistency_bonus = 15
        elif trend_consistency >= 0.6:  # %60+ tutarlılık
            consistency_bonus = 8
        
        # Signal strength bonusu
        strength_bonus = 0
        signal_strength = signal_info.get('signal_strength', 'WEAK')
        if signal_strength == 'STRONG':
            strength_bonus = 10
        elif signal_strength == 'MEDIUM':
            strength_bonus = 5
        
        # Volume bonusu
        volume_bonus = 0
        vol_ratio = signal_info.get('vol_ratio')
        if vol_ratio and vol_ratio >= 2.0:
            volume_bonus = 5
        
        # Final skor
        final_score = base_score + timeframe_bonus + consistency_bonus + strength_bonus + volume_bonus
        
        return min(final_score, 100.0)  # Max 100
        
    except Exception as e:
        logger.debug(f"Ranking skor hesaplama hatası: {e}")
        return signal_info.get('ai_score', 0)


def select_top3_adaptive_signals(signals: List[Dict], strategy: str) -> List[Dict]:
    """
    Strateji bazlı en iyi 3 sinyal seçimi
    
    Args:
        signals (List[Dict]): Sinyal listesi
        strategy (str): Strateji tipi (scalping, swing, position)
        
    Returns:
        List[Dict]: En iyi 3 sinyal
    """
    try:
        if not signals:
            return []
        
        # Strateji eşiği
        min_ai_threshold = get_adaptive_ai_threshold(strategy)
        
        # AI eşiği filtresi
        filtered_signals = [s for s in signals if s.get('ai_score', 0) >= min_ai_threshold]
        
        if not filtered_signals:
            logger.warning(f"⚠️ {strategy} stratejisi için AI eşiği ({min_ai_threshold}%) geçen sinyal yok")
            # Eşik düşürülerek tekrar dene
            min_ai_threshold = min_ai_threshold - 10
            filtered_signals = [s for s in signals if s.get('ai_score', 0) >= min_ai_threshold]
        
        # Ranking skoruna göre sırala
        for signal in filtered_signals:
            signal['ranking_score'] = calculate_ranking_score_with_timeframe(signal)
        
        # En iyi 3'ü seç
        sorted_signals = sorted(filtered_signals, key=lambda x: x['ranking_score'], reverse=True)
        top3 = sorted_signals[:3]
        
        logger.info(f"🎯 {strategy.upper()} stratejisi - En iyi 3 seçildi:")
        for i, signal in enumerate(top3):
            logger.info(f"   {i+1}. {signal['symbol']} | AI: {signal['ai_score']:.0f}% | Rank: {signal['ranking_score']:.0f}")
        
        return top3
        
    except Exception as e:
        logger.error(f"❌ Top3 seçim hatası: {e}")
        return signals[:3] if signals else []


def analyze_multi_timeframe_batch(symbols: List[str], primary_timeframe: str) -> List[Dict]:
    """
    Toplu multi-timeframe analizi
    
    Args:
        symbols (List[str]): Analiz edilecek semboller
        primary_timeframe (str): Ana timeframe
        
    Returns:
        List[Dict]: Analiz sonuçları
    """
    results = []
    strategy = get_strategy_from_timeframe(primary_timeframe)
    
    logger.info(f"🔍 Multi-timeframe analiz başlatılıyor: {len(symbols)} sembol, {strategy} stratejisi")
    
    for symbol in symbols:
        try:
            signal_data = calculate_timeframe_adaptive_signals(symbol, primary_timeframe)
            if signal_data:
                results.append(signal_data)
                
        except Exception as e:
            logger.debug(f"Sembol analiz hatası {symbol}: {e}")
            continue
    
    logger.info(f"✅ Multi-timeframe analiz tamamlandı: {len(results)} geçerli sinyal")
    
    return results


def get_timeframe_analysis_summary(signals: List[Dict]) -> Dict:
    """
    Timeframe analiz özeti
    
    Args:
        signals (List[Dict]): Sinyal listesi
        
    Returns:
        Dict: Analiz özeti
    """
    try:
        if not signals:
            return {}
        
        # Strateji dağılımı
        strategy_counts = {}
        signal_strength_counts = {}
        avg_ai_score = 0
        avg_consistency = 0
        
        for signal in signals:
            strategy = signal.get('strategy', 'unknown')
            strength = signal.get('signal_strength', 'WEAK')
            
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            signal_strength_counts[strength] = signal_strength_counts.get(strength, 0) + 1
            
            avg_ai_score += signal.get('ai_score', 0)
            avg_consistency += signal.get('trend_consistency', 0)
        
        total_signals = len(signals)
        avg_ai_score = avg_ai_score / total_signals
        avg_consistency = avg_consistency / total_signals
        
        return {
            'total_signals': total_signals,
            'avg_ai_score': avg_ai_score,
            'avg_trend_consistency': avg_consistency,
            'strategy_distribution': strategy_counts,
            'signal_strength_distribution': signal_strength_counts,
            'top_strategies': sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True),
            'analysis_timestamp': datetime.now(LOCAL_TZ).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Timeframe analiz özeti hatası: {e}")
        return {}


def calculate_strategy_performance_metrics(signals: List[Dict], strategy: str) -> Dict:
    """
    Strateji bazlı performans metrikleri
    
    Args:
        signals (List[Dict]): Sinyal listesi  
        strategy (str): Strateji tipi
        
    Returns:
        Dict: Performans metrikleri
    """
    try:
        strategy_signals = [s for s in signals if s.get('strategy') == strategy]
        
        if not strategy_signals:
            return {
                'strategy': strategy,
                'signal_count': 0,
                'avg_ai_score': 0,
                'avg_consistency': 0,
                'strong_signals': 0,
                'recommendation': f"{strategy} için sinyal bulunamadı"
            }
        
        # Metrikleri hesapla
        total_count = len(strategy_signals)
        avg_ai = sum(s.get('ai_score', 0) for s in strategy_signals) / total_count
        avg_consistency = sum(s.get('trend_consistency', 0) for s in strategy_signals) / total_count
        strong_count = len([s for s in strategy_signals if s.get('signal_strength') == 'STRONG'])
        
        # Öneri oluştur
        recommendation = ""
        if avg_ai >= 85:
            recommendation = f"{strategy} stratejisi çok güçlü sinyaller veriyor"
        elif avg_ai >= 70:
            recommendation = f"{strategy} stratejisi iyi sinyaller veriyor"
        else:
            recommendation = f"{strategy} stratejisi zayıf sinyaller veriyor"
        
        return {
            'strategy': strategy,
            'signal_count': total_count,
            'avg_ai_score': avg_ai,
            'avg_consistency': avg_consistency,
            'strong_signals': strong_count,
            'strong_signal_ratio': (strong_count / total_count) * 100 if total_count > 0 else 0,
            'recommendation': recommendation
        }
        
    except Exception as e:
        logger.error(f"Strateji performans metrikleri hatası: {e}")
        return {}


def get_optimal_timeframe_recommendation(historical_data: Optional[Dict] = None) -> Dict:
    """
    Optimal timeframe önerisi
    
    Args:
        historical_data (Dict, optional): Geçmiş performans verileri
        
    Returns:
        Dict: Timeframe önerisi
    """
    try:
        current_hour = datetime.now(LOCAL_TZ).hour
        current_minute = datetime.now(LOCAL_TZ).minute
        
        # Zaman bazlı öneriler
        recommendations = []
        
        # Sabah açılış (08:00-10:00)
        if 8 <= current_hour <= 10:
            recommendations.append({
                'timeframe': '5m',
                'strategy': 'scalping',
                'reason': 'Sabah açılış volatilitesi için kısa vadeli scalping uygun',
                'confidence': 0.8
            })
        
        # Öğlen seansı (12:00-14:00)
        elif 12 <= current_hour <= 14:
            recommendations.append({
                'timeframe': '15m',
                'strategy': 'swing',
                'reason': 'Öğlen seansında orta vadeli swing trading optimal',
                'confidence': 0.75
            })
        
        # Akşam seansı (18:00-22:00)
        elif 18 <= current_hour <= 22:
            recommendations.append({
                'timeframe': '1h',
                'strategy': 'swing',
                'reason': 'Akşam seansında 1 saatlik timeframe güvenli',
                'confidence': 0.7
            })
        
        # Gece/Düşük volatilite
        else:
            recommendations.append({
                'timeframe': '4h',
                'strategy': 'position',
                'reason': 'Düşük volatilite döneminde uzun vadeli position trading',
                'confidence': 0.6
            })
        
        # En yüksek confidence'lı öneriyi seç
        best_recommendation = max(recommendations, key=lambda x: x['confidence']) if recommendations else None
        
        return {
            'recommended_timeframe': best_recommendation['timeframe'] if best_recommendation else '15m',
            'recommended_strategy': best_recommendation['strategy'] if best_recommendation else 'swing',
            'reason': best_recommendation['reason'] if best_recommendation else 'Varsayılan öneri',
            'confidence': best_recommendation['confidence'] if best_recommendation else 0.5,
            'current_time': datetime.now(LOCAL_TZ).strftime('%H:%M'),
            'all_recommendations': recommendations
        }
        
    except Exception as e:
        logger.error(f"Timeframe önerisi hatası: {e}")
        return {
            'recommended_timeframe': '15m',
            'recommended_strategy': 'swing',
            'reason': 'Varsayılan güvenli seçenek',
            'confidence': 0.5
        }


def validate_signal_quality(signal: Dict) -> Dict:
    """
    Sinyal kalitesini doğrula ve skorla
    
    Args:
        signal (Dict): Sinyal verisi
        
    Returns:
        Dict: Kalite değerlendirmesi
    """
    try:
        quality_score = 0
        quality_issues = []
        quality_bonuses = []
        
        # AI skoru kontrolü
        ai_score = signal.get('ai_score', 0)
        if ai_score >= 90:
            quality_bonuses.append('Çok yüksek AI skoru')
            quality_score += 25
        elif ai_score >= 70:
            quality_bonuses.append('İyi AI skoru')
            quality_score += 15
        elif ai_score < 50:
            quality_issues.append('Düşük AI skoru')
            quality_score -= 10
        
        # Trend tutarlılığı
        consistency = signal.get('trend_consistency', 0)
        if consistency >= 0.8:
            quality_bonuses.append('Yüksek trend tutarlılığı')
            quality_score += 20
        elif consistency < 0.5:
            quality_issues.append('Düşük trend tutarlılığı')
            quality_score -= 15
        
        # Volume kontrolü
        vol_ratio = signal.get('vol_ratio')
        if vol_ratio and vol_ratio >= 2.0:
            quality_bonuses.append('Yüksek volume')
            quality_score += 10
        elif vol_ratio and vol_ratio < 1.0:
            quality_issues.append('Düşük volume')
            quality_score -= 5
        
        # Deviso ratio kontrolü
        deviso_ratio = signal.get('deviso_ratio', 0)
        run_type = signal.get('run_type')
        if ((run_type == 'long' and deviso_ratio > 3) or 
            (run_type == 'short' and deviso_ratio < -3)):
            quality_bonuses.append('Güçlü Deviso sinyali')
            quality_score += 15
        
        # Final kalite skoru
        quality_score = max(0, min(100, 50 + quality_score))  # 0-100 arası
        
        # Kalite seviyesi
        if quality_score >= 80:
            quality_level = 'EXCELLENT'
        elif quality_score >= 60:
            quality_level = 'GOOD'
        elif quality_score >= 40:
            quality_level = 'FAIR'
        else:
            quality_level = 'POOR'
        
        return {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'quality_bonuses': quality_bonuses,
            'quality_issues': quality_issues,
            'is_tradeable': quality_score >= 60,
            'risk_level': 'LOW' if quality_score >= 80 else 'MEDIUM' if quality_score >= 60 else 'HIGH'
        }
        
    except Exception as e:
        logger.error(f"Sinyal kalite doğrulama hatası: {e}")
        return {
            'quality_score': 0,
            'quality_level': 'UNKNOWN',
            'is_tradeable': False,
            'risk_level': 'HIGH'
        }