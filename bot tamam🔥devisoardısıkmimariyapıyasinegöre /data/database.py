"""
📁 Veritabanı ve CSV Yönetimi - CVD + ROC Momentum
Trade geçmişi ve sermaye takibi için CSV işlemleri
🔥 YENİ: CVD field'ları ile güncellenmiş CSV header'lar
"""

import os
import csv
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

import config
from config import LOCAL_TZ, TRADES_CSV, CAPITAL_CSV
from data.fetch_data import get_current_price

logger = logging.getLogger("crypto-analytics")


def setup_csv_files():
    """🔥 YENİ: CVD field'ları ile CSV dosyalarını hazırla"""
    
    # Trades CSV dosyası - CVD field'ları eklendi
    if not os.path.exists(TRADES_CSV):
        try:
            with open(TRADES_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    # Temel trade bilgileri
                    'timestamp', 'symbol', 'side', 'quantity', 'entry_price', 'exit_price',
                    'invested_amount', 'current_value', 'pnl', 'commission', 'ai_score',
                    
                    # 🔥 YENİ CVD METRİKLERİ
                    'cvd_roc_momentum', 'cvd_direction', 'momentum_strength',
                    'buy_pressure', 'sell_pressure', 'deviso_cvd_harmony',
                    'trend_strength', 'signal_type',
                    
                    # 🔧 ESKİ METRİKLER (Backward compatibility)
                    'run_type', 'run_count', 'run_perc', 'gauss_run', 
                    'vol_ratio', 'deviso_ratio',
                    
                    # Trade sonuç bilgileri
                    'stop_loss', 'take_profit', 'close_reason', 'status'
                ])
            logger.info(f"📊 CVD Trades CSV dosyası oluşturuldu: {TRADES_CSV}")
        except Exception as e:
            logger.error(f"❌ Trades CSV oluşturma hatası: {e}")
    
    # Capital CSV dosyası (değişiklik yok)
    if not os.path.exists(CAPITAL_CSV):
        try:
            with open(CAPITAL_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'capital', 'open_positions', 'total_invested', 'unrealized_pnl'])
            logger.info(f"💰 Capital CSV dosyası oluşturuldu: {CAPITAL_CSV}")
        except Exception as e:
            logger.error(f"❌ Capital CSV oluşturma hatası: {e}")


def log_trade_to_csv(trade_data: Dict):
    """
    🔥 YENİ: CVD field'ları ile trade'i CSV'ye kaydet
    
    Args:
        trade_data (Dict): CVD field'larını içeren trade bilgileri
    """
    try:
        with open(TRADES_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                # Temel trade bilgileri
                trade_data.get('timestamp', ''),
                trade_data.get('symbol', ''),
                trade_data.get('side', ''),
                trade_data.get('quantity', 0),
                trade_data.get('entry_price', 0),
                trade_data.get('exit_price', 0),
                trade_data.get('invested_amount', 0),
                trade_data.get('current_value', 0),
                trade_data.get('pnl', 0),
                trade_data.get('commission', 0),
                trade_data.get('ai_score', 0),
                
                # 🔥 YENİ CVD METRİKLERİ
                trade_data.get('cvd_roc_momentum', 0.0),
                trade_data.get('cvd_direction', 'neutral'),
                trade_data.get('momentum_strength', 0.0),
                trade_data.get('buy_pressure', 50.0),
                trade_data.get('sell_pressure', 50.0),
                trade_data.get('deviso_cvd_harmony', 50.0),
                trade_data.get('trend_strength', 0.0),
                trade_data.get('signal_type', 'neutral'),
                
                # 🔧 ESKİ METRİKLER (Backward compatibility)
                trade_data.get('run_type', trade_data.get('signal_type', 'neutral')),
                trade_data.get('run_count', 0),
                trade_data.get('run_perc', 0.0),
                trade_data.get('gauss_run', 0.0),
                trade_data.get('vol_ratio', 0.0),
                trade_data.get('deviso_ratio', 0.0),
                
                # Trade sonuç bilgileri
                trade_data.get('stop_loss', 0),
                trade_data.get('take_profit', 0),
                trade_data.get('close_reason', ''),
                trade_data.get('status', '')
            ])
        
        # CVD bilgilerini log'la
        cvd_direction = trade_data.get('cvd_direction', 'neutral')
        momentum_strength = trade_data.get('momentum_strength', 0.0)
        logger.debug(f"📝 CVD Trade kaydedildi: {trade_data.get('symbol')} {trade_data.get('side')} (CVD: {cvd_direction}, Momentum: {momentum_strength:.1f})")
        
    except Exception as e:
        logger.error(f"❌ CVD CSV yazma hatası: {e}")


def log_capital_to_csv():
    """Config'den güncel pozisyon ve sermaye bilgilerini al (değişiklik yok)"""
    try:
        # Config'den mevcut modu al
        if config.is_live_mode():
            current_capital = config.live_capital
            open_positions = config.live_positions
            mode_tag = "LIVE"
        else:
            current_capital = config.paper_capital
            open_positions = config.paper_positions
            mode_tag = "PAPER"
        
        # Toplam yatırım hesapla
        total_invested = sum(pos.get('invested_amount', 0) for pos in open_positions.values())
        total_unrealized_pnl = 0
        
        # Gerçekleşmemiş kar/zarar hesapla
        for symbol, pos in open_positions.items():
            current_price = get_current_price(symbol)
            if current_price:
                if pos['side'] == 'LONG':
                    unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']
                else:
                    unrealized_pnl = (pos['entry_price'] - current_price) * pos['quantity']
                total_unrealized_pnl += unrealized_pnl
        
        # CSV'ye yaz
        with open(CAPITAL_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                current_capital,
                len(open_positions),
                total_invested,
                total_unrealized_pnl
            ])
        
        logger.debug(f"💰 {mode_tag} sermaye kaydedildi: ${current_capital:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Capital CSV yazma hatası: {e}")


def load_trades_from_csv() -> pd.DataFrame:
    """
    🔥 YENİ: CVD field'ları ile CSV'den trade geçmişini yükle
    
    Returns:
        pd.DataFrame: CVD field'larını içeren trade geçmişi
    """
    try:
        if os.path.exists(TRADES_CSV):
            df = pd.read_csv(TRADES_CSV)
            
            # 🔥 YENİ: CVD field'ları dahil numerik sütunları düzelt
            numeric_columns = [
                # Temel numerik sütunlar
                'quantity', 'entry_price', 'exit_price', 'invested_amount', 
                'current_value', 'pnl', 'commission', 'ai_score', 
                'stop_loss', 'take_profit',
                
                # CVD numerik field'ları
                'cvd_roc_momentum', 'momentum_strength', 'buy_pressure', 
                'sell_pressure', 'deviso_cvd_harmony', 'trend_strength',
                
                # Eski numerik field'lar
                'run_count', 'run_perc', 'gauss_run', 'vol_ratio', 'deviso_ratio'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Timestamp'i datetime'a çevir
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # CVD field'larının varlığını kontrol et
            cvd_columns = ['cvd_roc_momentum', 'cvd_direction', 'momentum_strength']
            has_cvd_data = all(col in df.columns for col in cvd_columns)
            
            logger.debug(f"📊 {len(df)} trade kaydı yüklendi (CVD data: {'✅' if has_cvd_data else '❌'})")
            return df
        else:
            logger.info("📊 Trade CSV dosyası bulunamadı - boş DataFrame döndürülüyor")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"❌ Trades CSV yükleme hatası: {e}")
        return pd.DataFrame()


def load_capital_history_from_csv() -> pd.DataFrame:
    """CSV'den sermaye geçmişini yükle (değişiklik yok)"""
    try:
        if os.path.exists(CAPITAL_CSV):
            df = pd.read_csv(CAPITAL_CSV)
            
            # Veri tiplerini düzelt
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            numeric_columns = ['capital', 'open_positions', 'total_invested', 'unrealized_pnl']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.debug(f"💰 {len(df)} sermaye kaydı yüklendi")
            return df
        else:
            logger.info("💰 Capital CSV dosyası bulunamadı - boş DataFrame döndürülüyor")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"❌ Capital CSV yükleme hatası: {e}")
        return pd.DataFrame()


def calculate_performance_metrics():
    """Config entegre performans metrikleri - Live/Paper mode destekli (değişiklik yok)"""
    try:
        # Config'den mevcut modu al
        if config.is_live_mode():
            current_capital = config.live_capital
            open_positions = config.live_positions
            initial_capital = 13000.0  # Testnet başlangıç sermayesi (tahmini)
        else:
            current_capital = config.paper_capital
            open_positions = config.paper_positions
            initial_capital = 1000.0  # Paper trading başlangıç sermayesi
        
        trades_df = load_trades_from_csv()
        
        if trades_df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'current_capital': current_capital,
                'effective_capital': current_capital,
                'total_return': 0,
                'total_loss': 0,
                'total_gain': 0,
                'active_positions': len(open_positions),
                'total_commission': 0,
                'total_invested': 0,
                'total_unrealized_pnl': 0,
                'realized_total_profit': 0,
                'mode': 'live' if config.is_live_mode() else 'paper'
            }
        
        closed_trades = trades_df[trades_df['status'] == 'CLOSED']
        
        if closed_trades.empty:
            total_pnl = 0
            win_rate = 0
            total_trades = 0
            total_loss = 0
            total_gain = 0
            total_commission = 0
            realized_total_profit = 0  
        else:
            total_pnl = closed_trades['pnl'].sum()
            win_trades = len(closed_trades[closed_trades['pnl'] > 0])
            total_trades = len(closed_trades)
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
            
            loss_trades = closed_trades[closed_trades['pnl'] < 0]
            gain_trades = closed_trades[closed_trades['pnl'] > 0]
            total_loss = abs(loss_trades['pnl'].sum()) if not loss_trades.empty else 0
            total_gain = gain_trades['pnl'].sum() if not gain_trades.empty else 0
            
            total_commission = closed_trades['commission'].sum()  
            
            # Gerçekleşen toplam kar
            realized_total_profit = closed_trades['pnl'].sum()
        
        # Açık pozisyonlar analizi - Config'den
        total_invested = 0
        for symbol, pos in open_positions.items():
            total_invested += pos.get('invested_amount', 0)
        
        # Gerçekleşmemiş kar/zarar - Config'den
        total_unrealized_pnl = 0
        for symbol, pos in open_positions.items():
            current_price = get_current_price(symbol) or pos['entry_price']
            
            if pos['side'] == 'LONG':
                unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']
            else:
                unrealized_pnl = (pos['entry_price'] - current_price) * pos['quantity']
            
            total_unrealized_pnl += unrealized_pnl
        
        effective_capital = current_capital + total_unrealized_pnl
        total_return = ((effective_capital - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'current_capital': current_capital,
            'effective_capital': effective_capital,
            'total_return': total_return,
            'total_loss': total_loss,
            'total_gain': total_gain,
            'active_positions': len(open_positions),
            'total_commission': total_commission,
            'total_invested': total_invested,
            'total_unrealized_pnl': total_unrealized_pnl,
            'realized_total_profit': realized_total_profit,
            'mode': 'live' if config.is_live_mode() else 'paper',
            'initial_capital': initial_capital
        }
        
    except Exception as e:
        logger.error(f"❌ Metrik hesaplama hatası: {e}")
        
        # Hata durumunda varsayılan değerler - Config'den
        current_capital = config.live_capital if config.is_live_mode() else config.paper_capital
        open_positions = config.live_positions if config.is_live_mode() else config.paper_positions
        
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'current_capital': current_capital,
            'effective_capital': current_capital,
            'total_return': 0,
            'total_loss': 0,
            'total_gain': 0,
            'active_positions': len(open_positions),
            'total_commission': 0,
            'total_invested': 0,
            'total_unrealized_pnl': 0,
            'realized_total_profit': 0,
            'mode': 'live' if config.is_live_mode() else 'paper'
        }


def get_trade_statistics() -> Dict:
    """
    🔥 YENİ: CVD field'ları dahil trade istatistiklerini hesapla
    
    Returns:
        Dict: CVD istatistikleri dahil trade bilgileri
    """
    try:
        trades_df = load_trades_from_csv()
        
        if trades_df.empty:
            return {
                'total_trades': 0,
                'closed_trades': 0,
                'open_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'total_commission': 0,
                'avg_ai_score': 0,
                'cvd_stats': {}  # CVD istatistikleri
            }
        
        # Temel istatistikler
        total_trades = len(trades_df)
        closed_trades = len(trades_df[trades_df['status'] == 'CLOSED'])
        open_trades = len(trades_df[trades_df['status'] == 'OPEN'])
        
        # Sadece kapalı işlemler için analiz
        closed_df = trades_df[trades_df['status'] == 'CLOSED']
        
        if closed_df.empty:
            win_rate = 0
            total_pnl = 0
            avg_pnl = 0
            best_trade = 0
            worst_trade = 0
        else:
            # Kazanma oranı
            winning_trades = len(closed_df[closed_df['pnl'] > 0])
            win_rate = (winning_trades / len(closed_df) * 100) if len(closed_df) > 0 else 0
            
            # P&L istatistikleri
            total_pnl = closed_df['pnl'].sum()
            avg_pnl = closed_df['pnl'].mean()
            best_trade = closed_df['pnl'].max()
            worst_trade = closed_df['pnl'].min()
        
        # Diğer istatistikler
        total_commission = trades_df['commission'].sum()
        avg_ai_score = trades_df['ai_score'].mean()
        
        # 🔥 YENİ: CVD istatistikleri
        cvd_stats = {}
        cvd_columns = ['cvd_direction', 'momentum_strength', 'deviso_cvd_harmony', 'signal_type']
        has_cvd_data = all(col in trades_df.columns for col in cvd_columns)
        
        if has_cvd_data and not trades_df.empty:
            cvd_stats = {
                'has_cvd_data': True,
                'avg_momentum_strength': trades_df['momentum_strength'].mean(),
                'avg_harmony_score': trades_df['deviso_cvd_harmony'].mean(),
                'bullish_trades': len(trades_df[trades_df['cvd_direction'] == 'bullish']),
                'bearish_trades': len(trades_df[trades_df['cvd_direction'] == 'bearish']),
                'neutral_trades': len(trades_df[trades_df['cvd_direction'] == 'neutral']),
                'strong_momentum_trades': len(trades_df[trades_df['momentum_strength'] >= 70]),
                'high_harmony_trades': len(trades_df[trades_df['deviso_cvd_harmony'] >= 80]),
                'signal_type_distribution': trades_df['signal_type'].value_counts().to_dict() if 'signal_type' in trades_df.columns else {}
            }
            
            # CVD performans analizi (sadece kapalı trade'ler)
            if not closed_df.empty and has_cvd_data:
                # CVD direction'a göre performans
                cvd_performance = {}
                for direction in ['bullish', 'bearish', 'neutral']:
                    direction_trades = closed_df[closed_df['cvd_direction'] == direction]
                    if not direction_trades.empty:
                        cvd_performance[f'{direction}_count'] = len(direction_trades)
                        cvd_performance[f'{direction}_avg_pnl'] = direction_trades['pnl'].mean()
                        cvd_performance[f'{direction}_win_rate'] = len(direction_trades[direction_trades['pnl'] > 0]) / len(direction_trades) * 100
                
                cvd_stats['performance_by_direction'] = cvd_performance
        else:
            cvd_stats = {'has_cvd_data': False}
        
        return {
            'total_trades': total_trades,
            'closed_trades': closed_trades,
            'open_trades': open_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'total_commission': total_commission,
            'avg_ai_score': avg_ai_score,
            'cvd_stats': cvd_stats  # 🔥 YENİ CVD istatistikleri
        }
        
    except Exception as e:
        logger.error(f"❌ İstatistik hesaplama hatası: {e}")
        return {}


def backup_csv_files(backup_suffix: str = None):
    """CSV dosyalarını yedekle (değişiklik yok)"""
    try:
        if backup_suffix is None:
            backup_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Trades dosyası yedeği
        if os.path.exists(TRADES_CSV):
            backup_trades = f"{TRADES_CSV}.{backup_suffix}.bak"
            import shutil
            shutil.copy2(TRADES_CSV, backup_trades)
            logger.info(f"📋 Trades yedeklendi: {backup_trades}")
        
        # Capital dosyası yedeği
        if os.path.exists(CAPITAL_CSV):
            backup_capital = f"{CAPITAL_CSV}.{backup_suffix}.bak"
            import shutil
            shutil.copy2(CAPITAL_CSV, backup_capital)
            logger.info(f"💰 Capital yedeklendi: {backup_capital}")
            
    except Exception as e:
        logger.error(f"❌ Yedekleme hatası: {e}")


def cleanup_old_backups(keep_count: int = 5):
    """Eski yedekleri temizle (değişiklik yok)"""
    try:
        import glob
        
        # Trades yedekleri
        trades_backups = glob.glob(f"{TRADES_CSV}.*.bak")
        trades_backups.sort(reverse=True)  # En yeniden eskiye
        
        for backup in trades_backups[keep_count:]:
            os.remove(backup)
            logger.debug(f"🗑️ Eski yedek silindi: {backup}")
        
        # Capital yedekleri
        capital_backups = glob.glob(f"{CAPITAL_CSV}.*.bak")
        capital_backups.sort(reverse=True)
        
        for backup in capital_backups[keep_count:]:
            os.remove(backup)
            logger.debug(f"🗑️ Eski yedek silindi: {backup}")
            
    except Exception as e:
        logger.error(f"❌ Yedek temizleme hatası: {e}")


def export_to_excel(filename: str = None):
    """Verileri Excel dosyasına aktar (değişiklik yok)"""
    try:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            mode = "live" if config.is_live_mode() else "paper"
            filename = f"crypto_trading_report_{mode}_{timestamp}.xlsx"
        
        # Verileri yükle
        trades_df = load_trades_from_csv()
        capital_df = load_capital_history_from_csv()
        stats = get_trade_statistics()
        performance = calculate_performance_metrics()
        
        # Excel writer oluştur
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Trades sayfası
            if not trades_df.empty:
                trades_df.to_excel(writer, sheet_name='Trades', index=False)
            
            # Capital history sayfası
            if not capital_df.empty:
                capital_df.to_excel(writer, sheet_name='Capital_History', index=False)
            
            # İstatistikler sayfası
            stats_df = pd.DataFrame([stats]).T
            stats_df.columns = ['Value']
            stats_df.to_excel(writer, sheet_name='Statistics')
            
            # Performance özeti sayfası
            performance_df = pd.DataFrame([performance]).T
            performance_df.columns = ['Value']
            performance_df.to_excel(writer, sheet_name='Performance')
        
        logger.info(f"📊 Excel raporu oluşturuldu: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"❌ Excel aktarım hatası: {e}")
        return None


# Config uyumlu yardımcı fonksiyonlar (değişiklik yok)
def get_current_trading_summary() -> Dict:
    """Mevcut trading durumunun özetini al"""
    try:
        if config.is_live_mode():
            mode = "Live Trading"
            capital = config.live_capital
            positions = config.live_positions
        else:
            mode = "Paper Trading"
            capital = config.paper_capital
            positions = config.paper_positions
        
        return {
            'mode': mode,
            'capital': capital,
            'positions_count': len(positions),
            'active_symbols': list(positions.keys()),
            'total_invested': sum(pos.get('invested_amount', 0) for pos in positions.values())
        }
        
    except Exception as e:
        logger.error(f"❌ Trading özeti hatası: {e}")
        return {
            'mode': 'Unknown',
            'capital': 0,
            'positions_count': 0,
            'active_symbols': [],
            'total_invested': 0
        }


def reset_trading_data(mode: str = 'both'):
    """Trading verilerini sıfırla"""
    try:
        if mode in ['live', 'both']:
            config.reset_live_trading()
            logger.info("🔄 Live trading verileri sıfırlandı")
        
        if mode in ['paper', 'both']:
            config.reset_paper_trading()
            logger.info("🔄 Paper trading verileri sıfırlandı")
            
    except Exception as e:
        logger.error(f"❌ Veri sıfırlama hatası: {e}")


def sync_positions_with_config():
    """Pozisyonları config ile senkronize et"""
    try:
        # Bu fonksiyon live_trader.py tarafında yapılıyor
        # Burada sadece log veriyoruz
        summary = get_current_trading_summary()
        logger.debug(f"📊 Config senkronizasyon: {summary['mode']} - {summary['positions_count']} pozisyon")
        
    except Exception as e:
        logger.error(f"❌ Config senkronizasyon hatası: {e}")


# 🔥 YENİ: CVD analiz fonksiyonları
def get_cvd_trade_analysis() -> Dict:
    """CVD trade'lerinin detaylı analizi"""
    try:
        trades_df = load_trades_from_csv()
        
        if trades_df.empty:
            return {}
        
        # CVD field'larının varlığını kontrol et
        cvd_required_columns = ['cvd_direction', 'momentum_strength', 'deviso_cvd_harmony']
        if not all(col in trades_df.columns for col in cvd_required_columns):
            return {'error': 'CVD verileri bulunamadı'}
        
        analysis = {
            'total_cvd_trades': len(trades_df),
            'direction_analysis': {},
            'momentum_analysis': {},
            'harmony_analysis': {},
            'performance_by_cvd_strength': {}
        }
        
        # CVD direction analizi
        direction_counts = trades_df['cvd_direction'].value_counts().to_dict()
        analysis['direction_analysis'] = {
            'counts': direction_counts,
            'percentages': {k: (v/len(trades_df)*100) for k, v in direction_counts.items()}
        }
        
        # Momentum strength analizi
        momentum_ranges = {
            'strong': len(trades_df[trades_df['momentum_strength'] >= 70]),
            'medium': len(trades_df[(trades_df['momentum_strength'] >= 40) & (trades_df['momentum_strength'] < 70)]),
            'weak': len(trades_df[trades_df['momentum_strength'] < 40])
        }
        analysis['momentum_analysis'] = momentum_ranges
        
        # Harmony score analizi
        harmony_ranges = {
            'excellent': len(trades_df[trades_df['deviso_cvd_harmony'] >= 80]),
            'good': len(trades_df[(trades_df['deviso_cvd_harmony'] >= 60) & (trades_df['deviso_cvd_harmony'] < 80)]),
            'fair': len(trades_df[(trades_df['deviso_cvd_harmony'] >= 40) & (trades_df['deviso_cvd_harmony'] < 60)]),
            'poor': len(trades_df[trades_df['deviso_cvd_harmony'] < 40])
        }
        analysis['harmony_analysis'] = harmony_ranges
        
        # Kapalı trade'ler için performans analizi
        closed_trades = trades_df[trades_df['status'] == 'CLOSED']
        if not closed_trades.empty:
            # CVD gücüne göre performans
            strong_cvd = closed_trades[closed_trades['momentum_strength'] >= 70]
            medium_cvd = closed_trades[(closed_trades['momentum_strength'] >= 40) & (closed_trades['momentum_strength'] < 70)]
            weak_cvd = closed_trades[closed_trades['momentum_strength'] < 40]
            
            analysis['performance_by_cvd_strength'] = {
                'strong_cvd': {
                    'count': len(strong_cvd),
                    'avg_pnl': strong_cvd['pnl'].mean() if not strong_cvd.empty else 0,
                    'win_rate': len(strong_cvd[strong_cvd['pnl'] > 0]) / len(strong_cvd) * 100 if not strong_cvd.empty else 0
                },
                'medium_cvd': {
                    'count': len(medium_cvd),
                    'avg_pnl': medium_cvd['pnl'].mean() if not medium_cvd.empty else 0,
                    'win_rate': len(medium_cvd[medium_cvd['pnl'] > 0]) / len(medium_cvd) * 100 if not medium_cvd.empty else 0
                },
                'weak_cvd': {
                    'count': len(weak_cvd),
                    'avg_pnl': weak_cvd['pnl'].mean() if not weak_cvd.empty else 0,
                    'win_rate': len(weak_cvd[weak_cvd['pnl'] > 0]) / len(weak_cvd) * 100 if not weak_cvd.empty else 0
                }
            }
        
        return analysis
        
    except Exception as e:
        logger.error(f"CVD trade analizi hatası: {e}")
        return {'error': str(e)}


def migrate_csv_to_cvd_format():
    """Eski CSV dosyasını CVD formatına dönüştür"""
    try:
        if not os.path.exists(TRADES_CSV):
            logger.info("CSV dosyası bulunamadı - yeni format ile oluşturulacak")
            return True
        
        # Mevcut dosyayı yükle
        df = pd.read_csv(TRADES_CSV)
        
        # CVD column'larının varlığını kontrol et
        cvd_columns = ['cvd_roc_momentum', 'cvd_direction', 'momentum_strength']
        has_cvd = all(col in df.columns for col in cvd_columns)
        
        if has_cvd:
            logger.info("CSV dosyası zaten CVD formatında")
            return True
        
        logger.info("CSV dosyası eski formatta - CVD formatına dönüştürülüyor...")
        
        # Yedek al
        backup_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"{TRADES_CSV}.pre_cvd_{backup_suffix}.bak"
        df.to_csv(backup_file, index=False)
        logger.info(f"Eski format yedeklendi: {backup_file}")
        
        # CVD column'ları ekle (default değerlerle)
        cvd_defaults = {
            'cvd_roc_momentum': 0.0,
            'cvd_direction': 'neutral',
            'momentum_strength': 0.0,
            'buy_pressure': 50.0,
            'sell_pressure': 50.0,
            'deviso_cvd_harmony': 50.0,
            'trend_strength': 0.0,
            'signal_type': 'neutral'
        }
        
        for col, default_val in cvd_defaults.items():
            if col not in df.columns:
                df[col] = default_val
        
        # Signal_type'ı run_type'dan dönüştür
        if 'run_type' in df.columns and 'signal_type' not in df.columns:
            df['signal_type'] = df['run_type']
        
        # Yeni formatla kaydet
        df.to_csv(TRADES_CSV, index=False)
        logger.info("CSV dosyası CVD formatına dönüştürüldü")
        
        return True
        
    except Exception as e:
        logger.error(f"CSV migrasyon hatası: {e}")
        return False