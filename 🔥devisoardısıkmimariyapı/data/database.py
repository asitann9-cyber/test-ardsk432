"""
📁 Veritabanı ve CSV Yönetimi
Trade geçmişi ve sermaye takibi için CSV işlemleri
🔥 YENİ: Config entegrasyonu ile Live/Paper trading ayrımı
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
    """CSV dosyalarını hazırla ve header'ları oluştur"""
    
    # Trades CSV dosyası
    if not os.path.exists(TRADES_CSV):
        try:
            with open(TRADES_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'entry_price', 'exit_price',
                    'invested_amount', 'current_value', 'pnl', 'commission', 'ai_score',
                    'run_type', 'run_count', 'run_perc', 'gauss_run', 'vol_ratio', 'deviso_ratio',
                    'stop_loss', 'take_profit', 'close_reason', 'status'
                ])
            logger.info(f"📊 Trades CSV dosyası oluşturuldu: {TRADES_CSV}")
        except Exception as e:
            logger.error(f"❌ Trades CSV oluşturma hatası: {e}")
    
    # Capital CSV dosyası
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
    Trade'i CSV'ye kaydet
    
    Args:
        trade_data (Dict): Trade bilgileri
    """
    try:
        with open(TRADES_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
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
                trade_data.get('run_type', ''),
                trade_data.get('run_count', 0),
                trade_data.get('run_perc', 0),
                trade_data.get('gauss_run', 0),
                trade_data.get('vol_ratio', 0),
                trade_data.get('deviso_ratio', 0),
                trade_data.get('stop_loss', 0),
                trade_data.get('take_profit', 0),
                trade_data.get('close_reason', ''),
                trade_data.get('status', '')
            ])
        
        logger.debug(f"📝 Trade kaydedildi: {trade_data.get('symbol')} {trade_data.get('side')}")
        
    except Exception as e:
        logger.error(f"❌ CSV yazma hatası: {e}")


def log_capital_to_csv():
    """🔥 YENİ: Config'den güncel pozisyon ve sermaye bilgilerini al"""
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
    CSV'den trade geçmişini yükle
    
    Returns:
        pd.DataFrame: Trade geçmişi
    """
    try:
        if os.path.exists(TRADES_CSV):
            df = pd.read_csv(TRADES_CSV)
            
            # Veri tiplerini düzelt
            numeric_columns = [
                'quantity', 'entry_price', 'exit_price', 'invested_amount', 
                'current_value', 'pnl', 'commission', 'ai_score', 'run_count', 
                'run_perc', 'gauss_run', 'vol_ratio', 'deviso_ratio', 
                'stop_loss', 'take_profit'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Timestamp'i datetime'a çevir
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            logger.debug(f"📊 {len(df)} trade kaydı yüklendi")
            return df
        else:
            logger.info("📊 Trade CSV dosyası bulunamadı - boş DataFrame döndürülüyor")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"❌ Trades CSV yükleme hatası: {e}")
        return pd.DataFrame()


def load_capital_history_from_csv() -> pd.DataFrame:
    """
    CSV'den sermaye geçmişini yükle
    
    Returns:
        pd.DataFrame: Sermaye geçmişi
    """
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
    """🔥 YENİ: Config entegre performans metrikleri - Live/Paper mode destekli"""
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
    Trade istatistiklerini hesapla
    
    Returns:
        Dict: İstatistik bilgileri
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
                'avg_ai_score': 0
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
            'avg_ai_score': avg_ai_score
        }
        
    except Exception as e:
        logger.error(f"❌ İstatistik hesaplama hatası: {e}")
        return {}


def backup_csv_files(backup_suffix: str = None):
    """
    CSV dosyalarını yedekle
    
    Args:
        backup_suffix (str): Yedek dosya soneki (varsayılan: timestamp)
    """
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
    """
    Eski yedekleri temizle
    
    Args:
        keep_count (int): Tutulacak yedek sayısı
    """
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
    """
    Verileri Excel dosyasına aktar
    
    Args:
        filename (str): Excel dosya adı (varsayılan: otomatik)
    """
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
            
            # 🔥 YENİ: Performance özeti sayfası
            performance_df = pd.DataFrame([performance]).T
            performance_df.columns = ['Value']
            performance_df.to_excel(writer, sheet_name='Performance')
        
        logger.info(f"📊 Excel raporu oluşturuldu: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"❌ Excel aktarım hatası: {e}")
        return None


# 🔥 YENİ: Config uyumlu yardımcı fonksiyonlar
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
    """
    Trading verilerini sıfırla
    
    Args:
        mode (str): 'live', 'paper', veya 'both'
    """
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