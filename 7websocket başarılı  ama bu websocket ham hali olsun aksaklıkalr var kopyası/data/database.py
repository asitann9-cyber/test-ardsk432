"""
📁 Veritabanı ve CSV Yönetimi - TAM GÜNCELLEME
Trade geçmişi ve sermaye takibi için CSV işlemleri
🔥 YENİ: CSV validation, position sync, real-time monitoring eklendi
🔧 DÜZELTME: Test sorunları için kritik fonksiyonlar eklendi
"""

import os
import csv
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List, Tuple

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


def validate_csv_write_operation(trade_data: Dict) -> bool:
    """
    🔧 YENİ: CSV yazma işlemini doğrula - test sorunları için kritik
    
    Args:
        trade_data (Dict): Trade verisi
        
    Returns:
        bool: True eğer yazma başarılıysa
    """
    try:
        # Yazma öncesi validation
        required_fields = ['symbol', 'side', 'exit_price', 'pnl', 'close_reason']
        for field in required_fields:
            if field not in trade_data or trade_data[field] is None:
                logger.error(f"❌ CSV validation hatası - eksik field: {field}")
                return False
        
        # CSV dosyası yazılabilir mi?
        if not os.access('.', os.W_OK):
            logger.error(f"❌ CSV yazma izni yok")
            return False
        
        # CSV dosyası var mı ve yazılabilir mi?
        if os.path.exists(TRADES_CSV):
            if not os.access(TRADES_CSV, os.W_OK):
                logger.error(f"❌ {TRADES_CSV} yazma izni yok")
                return False
        
        # Dosya boyutu kontrolü
        original_size = os.path.getsize(TRADES_CSV) if os.path.exists(TRADES_CSV) else 0
        
        # Test yazma işlemi
        try:
            # Gerçek yazma işlemini yap
            log_trade_to_csv(trade_data)
            
            # Yazma sonrası kontrol
            if os.path.exists(TRADES_CSV):
                new_size = os.path.getsize(TRADES_CSV)
                if new_size <= original_size:
                    logger.error(f"❌ CSV dosya boyutu artmadı: {original_size} -> {new_size}")
                    return False
                else:
                    logger.debug(f"✅ CSV yazma başarılı: {original_size} -> {new_size} bytes")
                    return True
            else:
                logger.error(f"❌ CSV dosyası oluşturulmadı")
                return False
                
        except Exception as write_err:
            logger.error(f"❌ CSV test yazma hatası: {write_err}")
            return False
        
    except Exception as e:
        logger.error(f"❌ CSV validation hatası: {e}")
        return False


def log_trade_to_csv(trade_data: Dict) -> bool:
    """
    🔧 GÜNCELLEME: Trade'i CSV'ye kaydet - validation eklendi
    
    Args:
        trade_data (Dict): Trade bilgileri
        
    Returns:
        bool: Yazma işlemi başarılı mı
    """
    try:
        # 🔥 YENİ: Pre-validation
        logger.debug(f"📝 CSV yazma başlatılıyor: {trade_data.get('symbol')} - {trade_data.get('close_reason')}")
        
        # 🔥 DÜZELTME: Eksik alanları varsayılan değerlerle doldur
        safe_trade_data = {
            'timestamp': trade_data.get('timestamp', datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S')),
            'symbol': trade_data.get('symbol', ''),
            'side': trade_data.get('side', ''),
            'quantity': float(trade_data.get('quantity', 0)),
            'entry_price': float(trade_data.get('entry_price', 0)),
            'exit_price': float(trade_data.get('exit_price', 0)),
            'invested_amount': float(trade_data.get('invested_amount', 0)),
            'current_value': float(trade_data.get('current_value', 0)),
            'pnl': float(trade_data.get('pnl', 0)),
            'commission': float(trade_data.get('commission', 0)),
            'ai_score': float(trade_data.get('ai_score', 0)),
            'run_type': trade_data.get('run_type', ''),
            'run_count': int(trade_data.get('run_count', 0)),
            'run_perc': float(trade_data.get('run_perc', 0)),
            'gauss_run': float(trade_data.get('gauss_run', 0)),
            'vol_ratio': float(trade_data.get('vol_ratio', 0)),
            'deviso_ratio': float(trade_data.get('deviso_ratio', 0)),
            'stop_loss': float(trade_data.get('stop_loss', 0)),
            'take_profit': float(trade_data.get('take_profit', 0)),
            'close_reason': trade_data.get('close_reason', ''),
            'status': trade_data.get('status', 'UNKNOWN')
        }
        
        # 🔧 YENİ: CSV dosyası boyutu kontrolü
        original_size = os.path.getsize(TRADES_CSV) if os.path.exists(TRADES_CSV) else 0
        
        # CSV'ye yaz
        with open(TRADES_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                safe_trade_data['timestamp'],
                safe_trade_data['symbol'],
                safe_trade_data['side'],
                safe_trade_data['quantity'],
                safe_trade_data['entry_price'],
                safe_trade_data['exit_price'],
                safe_trade_data['invested_amount'],
                safe_trade_data['current_value'],
                safe_trade_data['pnl'],
                safe_trade_data['commission'],
                safe_trade_data['ai_score'],
                safe_trade_data['run_type'],
                safe_trade_data['run_count'],
                safe_trade_data['run_perc'],
                safe_trade_data['gauss_run'],
                safe_trade_data['vol_ratio'],
                safe_trade_data['deviso_ratio'],
                safe_trade_data['stop_loss'],
                safe_trade_data['take_profit'],
                safe_trade_data['close_reason'],
                safe_trade_data['status']
            ])
        
        # 🔧 YENİ: Yazma sonrası validation
        new_size = os.path.getsize(TRADES_CSV) if os.path.exists(TRADES_CSV) else 0
        
        if new_size > original_size:
            logger.debug(f"✅ Trade güvenli kaydet: {safe_trade_data['symbol']} {safe_trade_data['side']} | Exit: {safe_trade_data['exit_price']} | Reason: {safe_trade_data['close_reason']}")
            logger.debug(f"📊 CSV boyut değişimi: {original_size} -> {new_size} (+{new_size - original_size} bytes)")
            return True
        else:
            logger.error(f"❌ CSV yazma başarısız - boyut değişmedi: {original_size} -> {new_size}")
            return False
        
    except Exception as e:
        logger.error(f"❌ CSV yazma hatası: {e}")
        logger.error(f"Trade data: {trade_data}")
        return False


def sync_positions_to_database():
    """
    🔧 YENİ: Config pozisyonlarını database ile senkronize et - kritik fonksiyon
    Test sorunları için gerekli
    """
    try:
        # Config'den pozisyonları al
        if config.is_live_mode():
            current_positions = config.live_positions or {}
            mode_tag = "LIVE"
        else:
            current_positions = config.paper_positions or {}
            mode_tag = "PAPER"
        
        logger.debug(f"🔄 {mode_tag} pozisyon senkronizasyonu başlatılıyor...")
        logger.debug(f"📊 Config pozisyonları: {list(current_positions.keys())}")
        
        # Her pozisyon için database durumunu kontrol et
        sync_issues = []
        fixed_issues = 0
        
        for symbol, position in current_positions.items():
            try:
                # CSV'de bu pozisyon için OPEN kaydı var mı?
                trades_df = load_trades_from_csv()
                
                if not trades_df.empty:
                    open_trades = trades_df[
                        (trades_df['symbol'] == symbol) & 
                        (trades_df['status'] == 'OPEN')
                    ]
                    
                    if open_trades.empty:
                        # Config'de var ama database'de OPEN kaydı yok
                        logger.warning(f"⚠️ {symbol} config'de var ama CSV'de OPEN kaydı yok - düzeltiliyor")
                        sync_issues.append(f"{symbol}: Config var, CSV OPEN yok")
                        
                        # OPEN kaydı oluştur
                        open_trade_data = {
                            'timestamp': datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                            'symbol': symbol,
                            'side': position.get('side', 'LONG'),
                            'quantity': position.get('quantity', 0),
                            'entry_price': position.get('entry_price', 0),
                            'exit_price': 0,  # Henüz kapanmadı
                            'invested_amount': position.get('invested_amount', 0),
                            'current_value': 0,
                            'pnl': 0,  # Henüz kapanmadı
                            'commission': 0,
                            'ai_score': position.get('signal_data', {}).get('ai_score', 0),
                            'run_type': position.get('signal_data', {}).get('run_type', ''),
                            'run_count': position.get('signal_data', {}).get('run_count', 0),
                            'run_perc': position.get('signal_data', {}).get('run_perc', 0),
                            'gauss_run': position.get('signal_data', {}).get('gauss_run', 0),
                            'vol_ratio': position.get('signal_data', {}).get('vol_ratio', 0),
                            'deviso_ratio': position.get('signal_data', {}).get('deviso_ratio', 0),
                            'stop_loss': position.get('stop_loss', 0),
                            'take_profit': position.get('take_profit', 0),
                            'close_reason': '',
                            'status': 'OPEN'
                        }
                        
                        if log_trade_to_csv(open_trade_data):
                            logger.info(f"✅ {symbol} OPEN kaydı CSV'ye eklendi")
                            fixed_issues += 1
                        else:
                            logger.error(f"❌ {symbol} OPEN kaydı eklenemedi")
                
            except Exception as pos_err:
                logger.error(f"❌ {symbol} pozisyon senkronizasyon hatası: {pos_err}")
                sync_issues.append(f"{symbol}: Hata - {pos_err}")
        
        # Database'de OPEN ama config'de yok olanları bul
        trades_df = load_trades_from_csv()
        if not trades_df.empty:
            open_trades_df = trades_df[trades_df['status'] == 'OPEN']
            
            for _, trade in open_trades_df.iterrows():
                symbol = trade['symbol']
                if symbol not in current_positions:
                    logger.warning(f"⚠️ {symbol} CSV'de OPEN ama config'de yok")
                    sync_issues.append(f"{symbol}: CSV OPEN var, Config yok")
        
        # Senkronizasyon özeti
        if sync_issues:
            logger.warning(f"⚠️ {mode_tag} senkronizasyon sorunları: {len(sync_issues)} (düzeltilen: {fixed_issues})")
            for issue in sync_issues[:5]:  # İlk 5 tanesini göster
                logger.debug(f"   • {issue}")
        else:
            logger.debug(f"✅ {mode_tag} pozisyon-database senkronizasyonu OK")
        
        return len(sync_issues) == 0
        
    except Exception as e:
        logger.error(f"❌ Pozisyon-database senkronizasyon hatası: {e}")
        return False


def validate_trade_csv_integrity() -> Dict:
    """
    🔧 YENİ: Trade CSV bütünlüğünü kontrol et - test sisteminde sorun tespiti için
    
    Returns:
        Dict: Bütünlük raporu
    """
    try:
        report = {
            'file_exists': False,
            'readable': False,
            'valid_headers': False,
            'row_count': 0,
            'open_trades': 0,
            'closed_trades': 0,
            'last_trade_time': None,
            'file_size_bytes': 0,
            'issues': []
        }
        
        # Dosya var mı?
        if not os.path.exists(TRADES_CSV):
            report['issues'].append("CSV dosyası mevcut değil")
            return report
        
        report['file_exists'] = True
        report['file_size_bytes'] = os.path.getsize(TRADES_CSV)
        
        # Okunabilir mi?
        try:
            df = pd.read_csv(TRADES_CSV)
            report['readable'] = True
            report['row_count'] = len(df)
        except Exception as read_err:
            report['issues'].append(f"CSV okunamıyor: {read_err}")
            return report
        
        if df.empty:
            report['issues'].append("CSV boş")
            return report
        
        # Header'lar doğru mu?
        required_headers = [
            'timestamp', 'symbol', 'side', 'quantity', 'entry_price', 'exit_price',
            'pnl', 'close_reason', 'status'
        ]
        
        missing_headers = [h for h in required_headers if h not in df.columns]
        if missing_headers:
            report['issues'].append(f"Eksik header'lar: {missing_headers}")
        else:
            report['valid_headers'] = True
        
        # Status kontrolü
        if 'status' in df.columns:
            report['open_trades'] = len(df[df['status'] == 'OPEN'])
            report['closed_trades'] = len(df[df['status'] == 'CLOSED'])
            
            # Unknown status kontrolü
            unknown_status = len(df[~df['status'].isin(['OPEN', 'CLOSED'])])
            if unknown_status > 0:
                report['issues'].append(f"{unknown_status} bilinmeyen status")
        
        # Timestamp kontrolü
        if 'timestamp' in df.columns and not df.empty:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                report['last_trade_time'] = df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            except Exception as ts_err:
                report['issues'].append(f"Timestamp parse hatası: {ts_err}")
        
        # Veri tutarlılığı kontrolü
        numeric_columns = ['quantity', 'entry_price', 'exit_price', 'pnl']
        for col in numeric_columns:
            if col in df.columns:
                invalid_values = df[col].isna().sum() + df[col].isin([float('inf'), float('-inf')]).sum()
                if invalid_values > 0:
                    report['issues'].append(f"{col}: {invalid_values} geçersiz değer")
        
        # Duplicate kontrolü
        if 'symbol' in df.columns and 'timestamp' in df.columns:
            duplicates = df.duplicated(subset=['symbol', 'timestamp', 'status']).sum()
            if duplicates > 0:
                report['issues'].append(f"{duplicates} duplicate kayıt")
        
        logger.debug(f"📊 CSV bütünlük raporu: {report['row_count']} satır, {len(report['issues'])} sorun")
        
        return report
        
    except Exception as e:
        logger.error(f"❌ CSV bütünlük kontrolü hatası: {e}")
        return {
            'file_exists': False,
            'readable': False,
            'valid_headers': False,
            'row_count': 0,
            'open_trades': 0,
            'closed_trades': 0,
            'last_trade_time': None,
            'file_size_bytes': 0,
            'issues': [f"Kontrol hatası: {e}"]
        }


def monitor_csv_real_time() -> Dict:
    """
    🔧 YENİ: CSV durumunu real-time izle - test sırasında sorun tespiti için
    
    Returns:
        Dict: Real-time monitoring raporu
    """
    try:
        import time
        
        monitor_report = {
            'monitoring_time': datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            'file_status': 'unknown',
            'size_changes': [],
            'recent_trades': [],
            'write_test_result': False
        }
        
        # Dosya durumu
        if os.path.exists(TRADES_CSV):
            monitor_report['file_status'] = 'exists'
            current_size = os.path.getsize(TRADES_CSV)
            monitor_report['current_size'] = current_size
        else:
            monitor_report['file_status'] = 'missing'
            return monitor_report
        
        # Son trade'leri kontrol et
        try:
            df = load_trades_from_csv()
            if not df.empty:
                # Son 5 trade
                recent_df = df.tail(5)
                for _, trade in recent_df.iterrows():
                    monitor_report['recent_trades'].append({
                        'timestamp': str(trade.get('timestamp', '')),
                        'symbol': trade.get('symbol', ''),
                        'status': trade.get('status', ''),
                        'close_reason': trade.get('close_reason', '')
                    })
        except Exception as trade_err:
            monitor_report['recent_trades'] = [f"Trade okuma hatası: {trade_err}"]
        
        # Write test
        try:
            test_data = {
                'timestamp': datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': 'MONITOR_TEST',
                'side': 'LONG',
                'quantity': 0.001,
                'entry_price': 50000.0,
                'exit_price': 50001.0,
                'pnl': 0.001,
                'close_reason': 'MONITOR_TEST',
                'status': 'CLOSED'
            }
            
            original_size = os.path.getsize(TRADES_CSV)
            write_success = log_trade_to_csv(test_data)
            new_size = os.path.getsize(TRADES_CSV)
            
            monitor_report['write_test_result'] = write_success
            monitor_report['size_changes'].append({
                'before': original_size,
                'after': new_size,
                'change': new_size - original_size
            })
            
        except Exception as write_err:
            monitor_report['write_test_result'] = False
            monitor_report['write_error'] = str(write_err)
        
        return monitor_report
        
    except Exception as e:
        logger.error(f"❌ Real-time monitoring hatası: {e}")
        return {
            'monitoring_time': datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e)
        }


def log_capital_to_csv():
    """Config'den güncel pozisyon ve sermaye bilgilerini al - HATA KORUNMALI"""
    try:
        # Config'den mevcut modu al
        if config.is_live_mode():
            current_capital = float(config.live_capital or 0)
            open_positions = config.live_positions or {}
            mode_tag = "LIVE"
        else:
            current_capital = float(config.paper_capital or 0)
            open_positions = config.paper_positions or {}
            mode_tag = "PAPER"
        
        # Toplam yatırım hesapla
        total_invested = 0
        total_unrealized_pnl = 0
        
        # Gerçekleşmemiş kar/zarar hesapla
        for symbol, pos in list(open_positions.items()):
            try:
                invested_amount = float(pos.get('invested_amount', 0))
                total_invested += invested_amount
                
                current_price = get_current_price(symbol)
                if current_price and current_price > 0:
                    entry_price = float(pos.get('entry_price', 0))
                    quantity = float(pos.get('quantity', 0))
                    side = pos.get('side', 'LONG')
                    
                    if side == 'LONG':
                        unrealized_pnl = (current_price - entry_price) * quantity
                    else:
                        unrealized_pnl = (entry_price - current_price) * quantity
                    
                    total_unrealized_pnl += unrealized_pnl
            except Exception as pos_error:
                logger.debug(f"Pozisyon hesaplama hatası {symbol}: {pos_error}")
                continue
        
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
        
        logger.debug(f"💰 {mode_tag} sermaye güvenli kaydet: ${current_capital:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Capital CSV yazma hatası: {e}")


def load_trades_from_csv() -> pd.DataFrame:
    """
    CSV'den trade geçmişini yükle - HATA KORUNMALI - Gereksiz loglar kaldırılmış
    
    Returns:
        pd.DataFrame: Trade geçmişi
    """
    try:
        if not os.path.exists(TRADES_CSV):
            return pd.DataFrame()  # Sessizce boş döndür
        
        # CSV'yi oku
        df = pd.read_csv(TRADES_CSV)
        
        if df.empty:
            return pd.DataFrame()  # Sessizce boş döndür
        
        # 🔥 DÜZELTME: Gerekli sütunları kontrol et ve ekle
        required_columns = [
            'timestamp', 'symbol', 'side', 'quantity', 'entry_price', 'exit_price',
            'invested_amount', 'current_value', 'pnl', 'commission', 'ai_score',
            'run_type', 'run_count', 'run_perc', 'gauss_run', 'vol_ratio', 'deviso_ratio',
            'stop_loss', 'take_profit', 'close_reason', 'status'
        ]
        
        # Eksik sütunları ekle
        for col in required_columns:
            if col not in df.columns:
                if col in ['quantity', 'entry_price', 'exit_price', 'invested_amount', 'current_value', 'pnl', 'commission', 'ai_score', 'run_count', 'run_perc', 'gauss_run', 'vol_ratio', 'deviso_ratio', 'stop_loss', 'take_profit']:
                    df[col] = 0.0
                else:
                    df[col] = ''
                logger.warning(f"⚠️ Eksik sütun eklendi: {col}")
        
        # Veri tiplerini düzelt
        numeric_columns = [
            'quantity', 'entry_price', 'exit_price', 'invested_amount', 
            'current_value', 'pnl', 'commission', 'ai_score', 'run_count', 
            'run_perc', 'gauss_run', 'vol_ratio', 'deviso_ratio', 
            'stop_loss', 'take_profit'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Timestamp'i datetime'a çevir
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Status sütununu temizle
        if 'status' in df.columns:
            df['status'] = df['status'].fillna('UNKNOWN')
        
        # Sadece veri varsa log
        logger.debug(f"📊 {len(df)} trade kaydı güvenli yüklendi")
        return df
        
    except Exception as e:
        logger.error(f"❌ Trades CSV yükleme hatası: {e}")
        # Hata durumunda boş DataFrame döndür
        return pd.DataFrame()


def load_capital_history_from_csv() -> pd.DataFrame:
    """
    CSV'den sermaye geçmişini yükle - HATA KORUNMALI
    
    Returns:
        pd.DataFrame: Sermaye geçmişi
    """
    try:
        if not os.path.exists(CAPITAL_CSV):
            logger.info("💰 Capital CSV dosyası bulunamadı - boş DataFrame döndürülüyor")
            return pd.DataFrame()
        
        df = pd.read_csv(CAPITAL_CSV)
        
        if df.empty:
            logger.info("💰 Capital CSV boş - boş DataFrame döndürülüyor")
            return pd.DataFrame()
        
        # Gerekli sütunları kontrol et
        required_columns = ['timestamp', 'capital', 'open_positions', 'total_invested', 'unrealized_pnl']
        for col in required_columns:
            if col not in df.columns:
                if col == 'timestamp':
                    df[col] = datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    df[col] = 0.0
                logger.warning(f"⚠️ Capital eksik sütun eklendi: {col}")
        
        # Veri tiplerini düzelt
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        numeric_columns = ['capital', 'open_positions', 'total_invested', 'unrealized_pnl']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        logger.debug(f"💰 {len(df)} sermaye kaydı güvenli yüklendi")
        return df
        
    except Exception as e:
        logger.error(f"❌ Capital CSV yükleme hatası: {e}")
        return pd.DataFrame()


def calculate_performance_metrics():
    """Config entegre performans metrikleri - HATA KORUNMALI"""
    try:
        # Config'den mevcut modu al
        if config.is_live_mode():
            current_capital = float(config.live_capital or 0)
            open_positions = config.live_positions or {}
            initial_capital = 6036.25  # Loglardan görülen testnet bakiyesi
        else:
            current_capital = float(config.paper_capital or 0)
            open_positions = config.paper_positions or {}
            initial_capital = 1000.0  # Paper trading başlangıç sermayesi
        
        trades_df = load_trades_from_csv()
        
        # Varsayılan değerler
        default_metrics = {
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
        
        if trades_df.empty:
            return default_metrics
        
        # Status sütunu güvenli kontrol
        if 'status' in trades_df.columns:
            closed_trades = trades_df[trades_df['status'] == 'CLOSED']
        else:
            # Status sütunu yoksa exit_price > 0 olanları kapalı say
            closed_trades = trades_df[trades_df['exit_price'] > 0]
        
        if closed_trades.empty:
            total_pnl = 0
            win_rate = 0
            total_trades = 0
            total_loss = 0
            total_gain = 0
            total_commission = 0
            realized_total_profit = 0  
        else:
            total_pnl = float(closed_trades['pnl'].sum() or 0)
            win_trades = len(closed_trades[closed_trades['pnl'] > 0])
            total_trades = len(closed_trades)
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
            
            loss_trades = closed_trades[closed_trades['pnl'] < 0]
            gain_trades = closed_trades[closed_trades['pnl'] > 0]
            total_loss = abs(float(loss_trades['pnl'].sum() or 0)) if not loss_trades.empty else 0
            total_gain = float(gain_trades['pnl'].sum() or 0) if not gain_trades.empty else 0
            
            total_commission = float(closed_trades['commission'].sum() or 0)
            realized_total_profit = float(closed_trades['pnl'].sum() or 0)
        
        # Açık pozisyonlar analizi - Güvenli
        total_invested = 0
        total_unrealized_pnl = 0
        
        for symbol, pos in list(open_positions.items()):
            try:
                invested_amount = float(pos.get('invested_amount', 0))
                total_invested += invested_amount
                
                current_price = get_current_price(symbol) or float(pos.get('entry_price', 0))
                entry_price = float(pos.get('entry_price', 0))
                quantity = float(pos.get('quantity', 0))
                side = pos.get('side', 'LONG')
                
                if side == 'LONG':
                    unrealized_pnl = (current_price - entry_price) * quantity
                else:
                    unrealized_pnl = (entry_price - current_price) * quantity
                
                total_unrealized_pnl += unrealized_pnl
                
            except Exception as pos_error:
                logger.debug(f"Pozisyon metrik hatası {symbol}: {pos_error}")
                continue
        
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
        logger.error(f"❌ Metrik hesaplama detaylı hata: {str(e)}")
        
        # Hata durumunda güvenli varsayılan değerler
        current_capital = float(config.live_capital or 0) if config.is_live_mode() else float(config.paper_capital or 0)
        open_positions = config.live_positions or {} if config.is_live_mode() else config.paper_positions or {}
        
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
    Trade istatistiklerini hesapla - HATA KORUNMALI
    
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
        
        # Temel istatistikler - güvenli
        total_trades = len(trades_df)
        
        if 'status' in trades_df.columns:
            closed_trades = len(trades_df[trades_df['status'] == 'CLOSED'])
            open_trades = len(trades_df[trades_df['status'] == 'OPEN'])
            closed_df = trades_df[trades_df['status'] == 'CLOSED']
        else:
            closed_trades = len(trades_df[trades_df['exit_price'] > 0])
            open_trades = len(trades_df[trades_df['exit_price'] == 0])
            closed_df = trades_df[trades_df['exit_price'] > 0]
        
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
            total_pnl = float(closed_df['pnl'].sum() or 0)
            avg_pnl = float(closed_df['pnl'].mean() or 0)
            best_trade = float(closed_df['pnl'].max() or 0)
            worst_trade = float(closed_df['pnl'].min() or 0)
        
        # Diğer istatistikler
        total_commission = float(trades_df['commission'].sum() or 0)
        avg_ai_score = float(trades_df['ai_score'].mean() or 0)
        
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


# Yardımcı ve yönetim fonksiyonları
def backup_csv_files(backup_suffix: str = None):
    """CSV dosyalarını yedekle"""
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
    """Eski yedekleri temizle"""
    try:
        import glob
        
        # Trades yedekleri
        trades_backups = glob.glob(f"{TRADES_CSV}.*.bak")
        trades_backups.sort(reverse=True)
        
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
    """Verileri Excel dosyasına aktar"""
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


# Config uyumlu yardımcı fonksiyonlar
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
    """🔧 GÜNCELLEME: Pozisyonları config ile senkronize et - geliştirildi"""
    try:
        summary = get_current_trading_summary()
        logger.debug(f"📊 Config senkronizasyon: {summary['mode']} - {summary['positions_count']} pozisyon")
        
        # Otomatik senkronizasyon çalıştır
        sync_success = sync_positions_to_database()
        
        if not sync_success:
            logger.warning("⚠️ Config-Database senkronizasyonunda sorunlar tespit edildi")
        
        return sync_success
        
    except Exception as e:
        logger.error(f"❌ Config senkronizasyon hatası: {e}")
        return False


# Test ve debugging fonksiyonları
def run_database_diagnostics() -> Dict:
    """
    🔧 YENİ: Database sisteminin tam tanısını yap - test sorunları için
    
    Returns:
        Dict: Kapsamlı tanı raporu
    """
    try:
        diagnostics = {
            'timestamp': datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            'csv_integrity': validate_trade_csv_integrity(),
            'position_sync': sync_positions_to_database(),
            'real_time_monitor': monitor_csv_real_time(),
            'file_permissions': {},
            'system_status': 'unknown'
        }
        
        # Dosya izinleri kontrolü
        for file_path in [TRADES_CSV, CAPITAL_CSV]:
            if os.path.exists(file_path):
                diagnostics['file_permissions'][file_path] = {
                    'readable': os.access(file_path, os.R_OK),
                    'writable': os.access(file_path, os.W_OK),
                    'size_bytes': os.path.getsize(file_path)
                }
            else:
                diagnostics['file_permissions'][file_path] = {
                    'exists': False
                }
        
        # Genel sistem durumu
        issues_count = len(diagnostics['csv_integrity'].get('issues', []))
        
        if issues_count == 0 and diagnostics['position_sync'] and diagnostics['real_time_monitor'].get('write_test_result', False):
            diagnostics['system_status'] = 'healthy'
        elif issues_count <= 2:
            diagnostics['system_status'] = 'warning'
        else:
            diagnostics['system_status'] = 'critical'
        
        return diagnostics
        
    except Exception as e:
        logger.error(f"❌ Database tanı hatası: {e}")
        return {
            'timestamp': datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e),
            'system_status': 'error'
        }