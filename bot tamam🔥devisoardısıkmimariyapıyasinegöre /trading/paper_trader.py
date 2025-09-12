"""
💰 Paper Trading Sistemi - CVD + ROC Momentum
Sanal trading ve pozisyon yönetimi
🔥 YENİ: CVD momentum sistemi ile uyumlu hale getirildi
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict

from config import (
    LOCAL_TZ, INITIAL_CAPITAL, MAX_OPEN_POSITIONS, STOP_LOSS_PCT, 
    TAKE_PROFIT_PCT, SCAN_INTERVAL, current_capital, open_positions, 
    trading_active, current_data, current_settings
)
from data.fetch_data import get_current_price
from data.database import log_trade_to_csv, log_capital_to_csv

logger = logging.getLogger("crypto-analytics")

# Global thread referansı
paper_trade_thread = None


def calculate_position_size(price: float) -> float:
    """
    Pozisyon büyüklüğünü hesapla - TOPLAM PARAYI 3'E BÖL
    """
    global current_capital
    
    try:
        # Toplam sermayeyi maksimum pozisyon sayısına böl
        position_value = current_capital / MAX_OPEN_POSITIONS
        
        if position_value <= 0:
            return 0.0
        
        quantity = position_value / price
        return round(quantity, 6)
        
    except Exception as e:
        logger.error(f"Pozisyon büyüklüğü hesaplama hatası: {e}")
        return 0.0


def open_position(signal: Dict) -> bool:
    """
    🔥 YENİ: CVD momentum sistemi ile paper trading pozisyon aç
    """
    global current_capital, open_positions
    
    try:
        symbol = signal['symbol']
        
        # 🔥 YENİ: CVD signal_type kullan
        signal_type = signal.get('signal_type', signal.get('run_type', 'neutral'))
        
        # Signal type'a göre side belirle
        if signal_type in ['long', 'strong_long']:
            side = 'LONG'
        elif signal_type in ['short', 'strong_short']:
            side = 'SHORT'
        else:
            logger.warning(f"⚠️ {symbol}: Geçersiz signal type: {signal_type}")
            return False
        
        # Çifte pozisyon kontrolü
        if symbol in open_positions:
            logger.warning(f"⚠️ {symbol} için zaten açık pozisyon var")
            return False
        
        # Maksimum pozisyon kontrolü
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            logger.warning(f"⚠️ Maksimum pozisyon sayısına ulaşıldı: {MAX_OPEN_POSITIONS}")
            return False
        
        # Mevcut fiyatı al
        current_price = get_current_price(symbol)
        if not current_price:
            logger.error(f"❌ {symbol} için fiyat alınamadı")
            return False
        
        # Pozisyon büyüklüğü
        quantity = calculate_position_size(current_price)
        if quantity <= 0:
            logger.error(f"❌ {symbol} için geçersiz pozisyon büyüklüğü")
            return False
        
        # Yatırım miktarı (komisyonsuz)
        invested_amount = quantity * current_price
        total_cost = invested_amount
        
        # Sermaye yeterliliği
        if current_capital < total_cost:
            logger.warning(f"⚠️ {symbol}: Yetersiz sermaye! Mevcut: ${current_capital:.2f}, Gerekli: ${total_cost:.2f}")
            return False
        
        # SL/TP
        if side == 'LONG':
            stop_loss = current_price * (1 - STOP_LOSS_PCT)
            take_profit = current_price * (1 + TAKE_PROFIT_PCT)
        else:
            stop_loss = current_price * (1 + STOP_LOSS_PCT)
            take_profit = current_price * (1 - TAKE_PROFIT_PCT)
        
        # Pozisyon verisi
        position_data = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': current_price,
            'invested_amount': invested_amount,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(LOCAL_TZ),
            'commission': 0.0,
            'signal_data': signal
        }
        
        open_positions[symbol] = position_data
        current_capital -= total_cost
        
        # CVD bilgilerini log'la
        cvd_direction = signal.get('cvd_direction', 'neutral')
        momentum_strength = signal.get('momentum_strength', 0.0)
        deviso_cvd_harmony = signal.get('deviso_cvd_harmony', 50.0)
        
        logger.info(f"✅ PAPER TRADE AÇILDI: {symbol} {side} {quantity} @ ${current_price:.6f}")
        logger.info(f"💰 Yatırılan: ${invested_amount:.2f} | Kalan sermaye: ${current_capital:.2f}")
        logger.info(f"📊 SL: ${stop_loss:.6f} | TP: ${take_profit:.6f}")
        logger.info(f"🔥 CVD: {cvd_direction} | Momentum: {momentum_strength:.1f} | Harmony: {deviso_cvd_harmony:.1f} | AI: {signal['ai_score']:.0f}%")
        
        # 🔥 YENİ: CVD field'ları ile CSV'ye kaydet
        trade_data = {
            'timestamp': position_data['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': current_price,
            'invested_amount': invested_amount,
            'commission': 0.0,
            'ai_score': signal['ai_score'],
            
            # 🔥 CVD METRİKLERİ
            'cvd_roc_momentum': signal.get('cvd_roc_momentum', 0.0),
            'cvd_direction': signal.get('cvd_direction', 'neutral'),
            'momentum_strength': signal.get('momentum_strength', 0.0),
            'buy_pressure': signal.get('buy_pressure', 50.0),
            'sell_pressure': signal.get('sell_pressure', 50.0),
            'deviso_cvd_harmony': signal.get('deviso_cvd_harmony', 50.0),
            'trend_strength': signal.get('trend_strength', 0.0),
            'signal_type': signal_type,
            
            # 🔧 ESKİ METRİKLER (Backward compatibility)
            'run_type': signal.get('run_type', signal_type),
            'run_count': signal.get('run_count', int(momentum_strength / 10)),
            'run_perc': signal.get('run_perc', abs(signal.get('cvd_roc_momentum', 0)) / 5.0),
            'gauss_run': signal.get('gauss_run', signal.get('trend_strength', 0.0)),
            'vol_ratio': signal.get('vol_ratio', 2.0),
            'deviso_ratio': signal.get('deviso_ratio', 0.0),
            
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'status': 'OPEN'
        }
        log_trade_to_csv(trade_data)
        log_capital_to_csv()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pozisyon açma hatası: {e}")
        return False


def close_position(symbol: str, close_reason: str) -> bool:
    """
    🔥 YENİ: CVD field'ları ile paper trading pozisyon kapat
    """
    global current_capital, open_positions
    
    try:
        if symbol not in open_positions:
            logger.warning(f"⚠️ {symbol} için açık pozisyon bulunamadı")
            return False
        
        position = open_positions[symbol]
        
        # Çıkış fiyatı
        current_price = get_current_price(symbol)
        if not current_price:
            logger.error(f"❌ {symbol} için çıkış fiyatı alınamadı")
            return False
        
        # Mevcut değer
        current_value = position['quantity'] * current_price
        
        # P&L
        if position['side'] == 'LONG':
            gross_pnl = (current_price - position['entry_price']) * position['quantity']
        else:
            gross_pnl = (position['entry_price'] - current_price) * position['quantity']
        
        net_pnl = gross_pnl  # komisyon yok
        
        # Sermayeye iade
        total_return = current_value
        current_capital += total_return
        
        logger.info(f"✅ PAPER TRADE KAPANDI: {symbol} {position['side']}")
        logger.info(f"💲 Giriş: ${position['entry_price']:.6f} → Çıkış: ${current_price:.6f}")
        logger.info(f"💰 P&L: ${net_pnl:.4f} (Komisyonsuz)")
        logger.info(f"🏦 Güncel sermaye: ${current_capital:.2f} | Sebep: {close_reason}")
        
        # 🔥 YENİ: CVD field'ları ile CSV kaydet
        signal_data = position['signal_data']
        trade_data = {
            'timestamp': datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'side': position['side'],
            'quantity': position['quantity'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'invested_amount': position['invested_amount'],
            'current_value': current_value,
            'pnl': net_pnl,
            'commission': 0.0,
            'ai_score': signal_data['ai_score'],
            
            # 🔥 CVD METRİKLERİ
            'cvd_roc_momentum': signal_data.get('cvd_roc_momentum', 0.0),
            'cvd_direction': signal_data.get('cvd_direction', 'neutral'),
            'momentum_strength': signal_data.get('momentum_strength', 0.0),
            'buy_pressure': signal_data.get('buy_pressure', 50.0),
            'sell_pressure': signal_data.get('sell_pressure', 50.0),
            'deviso_cvd_harmony': signal_data.get('deviso_cvd_harmony', 50.0),
            'trend_strength': signal_data.get('trend_strength', 0.0),
            'signal_type': signal_data.get('signal_type', signal_data.get('run_type', 'neutral')),
            
            # 🔧 ESKİ METRİKLER (Backward compatibility)
            'run_type': signal_data.get('run_type', signal_data.get('signal_type', 'neutral')),
            'run_count': signal_data.get('run_count', 0),
            'run_perc': signal_data.get('run_perc', 0.0),
            'gauss_run': signal_data.get('gauss_run', 0.0),
            'vol_ratio': signal_data.get('vol_ratio', 0.0),
            'deviso_ratio': signal_data.get('deviso_ratio', 0.0),
            
            'stop_loss': position['stop_loss'],
            'take_profit': position['take_profit'],
            'close_reason': close_reason,
            'status': 'CLOSED'
        }
        log_trade_to_csv(trade_data)
        log_capital_to_csv()
        
        # Pozisyonu sil
        del open_positions[symbol]
        return True
        
    except Exception as e:
        logger.error(f"❌ Pozisyon kapatma hatası {symbol}: {e}")
        return False


def monitor_positions():
    """Açık pozisyonları izle ve SL/TP kontrol et"""
    try:
        if not open_positions:
            return
        
        logger.info(f"📊 {len(open_positions)} açık pozisyon izleniyor...")
        closed_positions = []
        
        for symbol in list(open_positions.keys()):
            position = open_positions[symbol]
            
            current_price = get_current_price(symbol)
            if current_price is None:
                logger.error(f"❌ {symbol} için fiyat alınamadı - pozisyon bu döngüde atlanıyor!")
                continue
            
            should_close = False
            close_reason = ""
            
            # SL/TP
            if position['side'] == 'LONG':
                if current_price <= position['stop_loss']:
                    should_close = True
                    close_reason = "Stop Loss"
                    logger.info(f"🛑 {symbol} Long SL: ${current_price:.6f} <= ${position['stop_loss']:.6f}")
                elif current_price >= position['take_profit']:
                    should_close = True
                    close_reason = "Take Profit"
                    logger.info(f"🎯 {symbol} Long TP: ${current_price:.6f} >= ${position['take_profit']:.6f}")
            else:  # SHORT
                if current_price >= position['stop_loss']:
                    should_close = True
                    close_reason = "Stop Loss"
                    logger.info(f"🛑 {symbol} Short SL: ${current_price:.6f} >= ${position['stop_loss']:.6f}")
                elif current_price <= position['take_profit']:
                    should_close = True
                    close_reason = "Take Profit"
                    logger.info(f"🎯 {symbol} Short TP: ${current_price:.6f} <= ${position['take_profit']:.6f}")
            
            if should_close:
                success = close_position(symbol, close_reason)
                if success:
                    closed_positions.append(symbol)
        
        if closed_positions:
            new_count = len(open_positions)
            logger.info(f"🔄 {len(closed_positions)} pozisyon kapandı. Yeni durum: {new_count}/{MAX_OPEN_POSITIONS}")
            logger.info(f"ℹ️ Yeni pozisyon açma ana döngüde yapılacak - çakışma önlendi")
                
    except Exception as e:
        logger.error(f"❌ Pozisyon izleme hatası: {e}")


def fill_empty_positions():
    """
    🔥 YENİ: CVD momentum sistemli sinyal seçimi
    """
    global trading_active, current_data, current_settings

    try:
        if not trading_active:
            logger.debug("ℹ️ Trading durdurulmuş - Yeni pozisyon açılmıyor")
            return

        current_positions = len(open_positions)
        if current_positions >= MAX_OPEN_POSITIONS:
            logger.debug(f"✅ Tüm pozisyon slotları dolu: {current_positions}/{MAX_OPEN_POSITIONS}")
            return

        needed_positions = MAX_OPEN_POSITIONS - current_positions

        if current_data is None or current_data.empty:
            logger.warning("❌ TÜM SİNYALLER tablosu boş - Otomatik tarama başlatın!")
            return

        df = current_data.copy()

        # AI skoru 0–1 geldiyse 0–100'e çevir
        try:
            if df["ai_score"].max() <= 1.0:
                df["ai_score"] = (df["ai_score"] * 100.0).clip(0, 100)
        except Exception:
            pass

        # 🔥 YENİ: CVD bazlı filtreleme
        min_ai_pct = float(current_settings.get('min_ai', 30))
        trading_min_ai = max(min_ai_pct, 20.0)  # CVD için biraz daha gevşek
        
        # CVD momentum filtresi
        min_momentum_strength = 30.0  # Minimum CVD momentum gücü
        min_harmony_score = 40.0      # Minimum Deviso-CVD uyum skoru

        logger.info(f"🔥 CVD Paper Trading filtreleri: AI≥{trading_min_ai:.0f}%, Momentum≥{min_momentum_strength}, Harmony≥{min_harmony_score}")

        # CVD bazlı filtreleme uygula
        original_count = len(df)
        
        # Temel filtreler
        df = df[df['ai_score'] >= trading_min_ai]
        
        # CVD momentum filtreleri
        if 'momentum_strength' in df.columns:
            df = df[df['momentum_strength'] >= min_momentum_strength]
            
        if 'deviso_cvd_harmony' in df.columns:
            df = df[df['deviso_cvd_harmony'] >= min_harmony_score]
        
        # Signal type filtresi (neutral hariç)
        if 'signal_type' in df.columns:
            df = df[df['signal_type'] != 'neutral']

        logger.info(f"🔍 CVD filtreleme: {len(df)}/{original_count} sinyal kaldı")

        if df.empty:
            logger.warning("⚠️ CVD filtre sonrası uygun sinyal yok")
            return

        # Zaten açık pozisyonları hariç tut
        exclude_symbols = set(open_positions.keys())
        before_exclude = len(df)
        df = df[~df['symbol'].isin(exclude_symbols)]
        logger.info(f"🧮 Açık pozisyonlar hariç: {len(df)} (önce: {before_exclude})")

        if df.empty:
            logger.info("ℹ️ Uygun yeni sembol yok (hepsi açık pozisyonlarda)")
            return

        # 🔥 YENİ: CVD momentum bazlı sıralama
        sort_columns = ['ai_score']
        if 'momentum_strength' in df.columns:
            sort_columns.append('momentum_strength')
        if 'deviso_cvd_harmony' in df.columns:
            sort_columns.append('deviso_cvd_harmony')
        if 'trend_strength' in df.columns:
            sort_columns.append('trend_strength')
        
        # Fallback eski sütunlar
        for col in ['run_perc', 'gauss_run', 'vol_ratio']:
            if col in df.columns:
                sort_columns.append(col)
        
        df = df.sort_values(by=sort_columns, ascending=[False] * len(sort_columns))

        # İlk N seç
        top_n = min(3, needed_positions, len(df))
        top_signals = df.head(top_n)
        
        logger.debug("📊 CVD PaperTrader sıralama sonrası ilk 5 sinyal:")
        for i, row in df.head(5).iterrows():
            cvd_info = f"CVD={row.get('cvd_direction', 'N/A')}, Momentum={row.get('momentum_strength', 0):.1f}"
            logger.debug(f"   {i}: {row['symbol']} | AI={row['ai_score']} | {cvd_info}")

        logger.info(f"🎯 CVD EN İYİ {top_n} SİNYAL → seçilen:")
        for i, (_, s) in enumerate(top_signals.iterrows(), 1):
            cvd_direction = s.get('cvd_direction', 'neutral')
            momentum_strength = s.get('momentum_strength', 0.0)
            logger.info(f"   🔥 #{i}: {s['symbol']} | AI={s['ai_score']:.0f}% | CVD={cvd_direction} | Momentum={momentum_strength:.1f}")

        # Pozisyonları aç
        opened_count = 0
        for _, signal in top_signals.iterrows():
            if not trading_active:
                break
            if open_position(signal.to_dict()):
                opened_count += 1

        logger.info(f"🏆 CVD SONUÇ: {opened_count} yeni pozisyon açıldı | Toplam: {len(open_positions)}/{MAX_OPEN_POSITIONS}")

    except Exception as e:
        logger.error(f"❌ fill_empty_positions hatası: {e}")


def paper_trading_loop():
    """🔥 YENİ: CVD momentum sistemi ile paper trading döngüsü"""
    global trading_active
    
    logger.info("🚀 CVD Paper Trading döngüsü başlatıldı")
    logger.info("🔥 CVD + ROC Momentum sistemi ile çalışıyor!")
    loop_count = 0
    
    while trading_active:
        try:
            loop_count += 1
            loop_start = time.time()
            
            current_position_count = len(open_positions)
            logger.info(f"🔄 CVD Döngü #{loop_count} - Pozisyon: {current_position_count}/{MAX_OPEN_POSITIONS}")
            
            # Veri durumu
            if current_data is None or current_data.empty:
                logger.warning(f"⚠️ Döngü #{loop_count}: CVD sinyal tablosu boş - otomatik tarama başlatın!")
            else:
                cvd_signals = len(current_data[current_data.get('signal_type', 'neutral') != 'neutral']) if 'signal_type' in current_data.columns else 0
                logger.info(f"📊 Döngü #{loop_count}: {len(current_data)} toplam sinyal ({cvd_signals} CVD aktif)")
            
            # Boş pozisyonları CVD sinyalleri ile doldur
            fill_empty_positions()
            
            # Açık pozisyonları izle
            monitor_positions()
            
            # Sermaye durumunu kaydet
            log_capital_to_csv()
            
            # CVD performans özeti
            total_unrealized_pnl = 0.0
            cvd_position_summary = []
            
            for symbol, pos in open_positions.items():
                current_price = get_current_price(symbol) or pos['entry_price']
                
                if pos['side'] == 'LONG':
                    unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']
                else:
                    unrealized_pnl = (pos['entry_price'] - current_price) * pos['quantity']
                
                total_unrealized_pnl += unrealized_pnl
                
                # CVD bilgilerini topla
                signal_data = pos.get('signal_data', {})
                cvd_direction = signal_data.get('cvd_direction', 'N/A')
                momentum_strength = signal_data.get('momentum_strength', 0.0)
                cvd_position_summary.append(f"{symbol}({cvd_direction},{momentum_strength:.0f})")
            
            loop_time = time.time() - loop_start
            
            logger.info(f"⏱️ CVD Döngü #{loop_count}: {loop_time:.2f}s | Sermaye: ${current_capital:.2f} | P&L: ${total_unrealized_pnl:.2f}")
            
            if open_positions:
                logger.info(f"🔥 Açık CVD pozisyonlar: {', '.join(cvd_position_summary)}")
                logger.info(f"📊 Bu pozisyonlar CVD momentum sistemi ile seçildi ✅")
            
            time.sleep(SCAN_INTERVAL)
            
        except Exception as e:
            logger.error(f"❌ CVD Paper Trading döngüsü hatası: {e}")
            time.sleep(30)
    
    logger.info("ℹ️ CVD Paper Trading döngüsü sonlandırıldı")


def start_paper_trading() -> bool:
    """🔥 YENİ: CVD Paper Trading başlat"""
    global trading_active, paper_trade_thread, current_capital
    
    if trading_active:
        logger.warning("⚠️ CVD Paper Trading zaten aktif")
        return False
    
    logger.info("🚀 CVD Paper Trading başlatılıyor...")
    logger.info(f"💰 Başlangıç sermayesi: ${INITIAL_CAPITAL:.2f}")
    logger.info(f"📊 Maksimum pozisyon: {MAX_OPEN_POSITIONS} (CVD momentum sisteminden)")
    logger.info(f"⏰ Tarama aralığı: {SCAN_INTERVAL} saniye")
    logger.info(f"🛑 Stop Loss: %{STOP_LOSS_PCT*100}")
    logger.info(f"🎯 Take Profit: %{TAKE_PROFIT_PCT*100}")
    logger.info(f"💳 Komisyon: KALDIRILDI")
    logger.info(f"💰 Pozisyon Büyüklüğü: Toplam sermaye ÷ 3")
    logger.info(f"🔥 YENİ: CVD + ROC Momentum sistemi kullanıyor")
    
    # Başlangıç sermayesini sıfırla
    current_capital = INITIAL_CAPITAL
    
    # CSV dosyaları
    from data.database import setup_csv_files
    setup_csv_files()
    
    trading_active = True
    
    # Thread başlat
    paper_trade_thread = threading.Thread(target=paper_trading_loop, daemon=True)
    paper_trade_thread.start()
    
    logger.info("✅ CVD Paper Trading başlatıldı")
    return True


def stop_paper_trading():
    """🔥 YENİ: CVD field'ları ile Paper Trading durdur"""
    global trading_active, current_capital, open_positions
    
    if not trading_active:
        logger.info("💤 CVD Paper Trading zaten durdurulmuş")
        return
    
    logger.info("🛑 CVD Paper Trading durduruluyor...")
    trading_active = False
    
    if open_positions:
        position_count = len(open_positions)
        logger.info(f"📚 {position_count} CVD pozisyon toplu olarak kapatılıyor...")
        
        # Kapanış öncesi durum
        total_invested_before = sum(pos.get('invested_amount', 0) for pos in open_positions.values())
        logger.info(f"💰 Kapanış öncesi toplam yatırılan: ${total_invested_before:.2f}")
        logger.info(f"🏦 Kapanış öncesi mevcut sermaye: ${current_capital:.2f}")
        
        closed_positions = []
        total_returned = 0.0
        cvd_summary = []
        
        # Tüm pozisyonları kapat
        for symbol in list(open_positions.keys()):
            position = open_positions[symbol]
            
            current_price = get_current_price(symbol)
            if current_price is None:
                logger.error(f"❌ {symbol} için fiyat alınamadı - pozisyon atlanıyor!")
                continue
            
            current_value = position['quantity'] * current_price
            
            # P&L
            if position['side'] == 'LONG':
                gross_pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                gross_pnl = (position['entry_price'] - current_price) * position['quantity']
            
            net_pnl = gross_pnl  # komisyon yok
            total_return_when_closed = current_value  # tam değer iade
            
            current_capital += total_return_when_closed
            total_returned += total_return_when_closed
            
            # CVD pozisyon özeti
            signal_data = position.get('signal_data', {})
            cvd_direction = signal_data.get('cvd_direction', 'N/A')
            momentum_strength = signal_data.get('momentum_strength', 0.0)
            cvd_summary.append(f"{symbol}({cvd_direction},{momentum_strength:.0f},${net_pnl:.1f})")
            
            # Trade kaydı - CVD field'ları ile
            trade_data = {
                'timestamp': datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'side': position['side'],
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'invested_amount': position['invested_amount'],
                'current_value': current_value,
                'pnl': net_pnl,
                'commission': 0.0,
                'ai_score': signal_data['ai_score'],
                
                # CVD metrikler
                'cvd_roc_momentum': signal_data.get('cvd_roc_momentum', 0.0),
                'cvd_direction': signal_data.get('cvd_direction', 'neutral'),
                'momentum_strength': signal_data.get('momentum_strength', 0.0),
                'buy_pressure': signal_data.get('buy_pressure', 50.0),
                'sell_pressure': signal_data.get('sell_pressure', 50.0),
                'deviso_cvd_harmony': signal_data.get('deviso_cvd_harmony', 50.0),
                'trend_strength': signal_data.get('trend_strength', 0.0),
                'signal_type': signal_data.get('signal_type', 'neutral'),
                
                # Eski metrikler
                'run_type': signal_data.get('run_type', signal_data.get('signal_type', 'neutral')),
                'run_count': signal_data.get('run_count', 0),
                'run_perc': signal_data.get('run_perc', 0.0),
                'gauss_run': signal_data.get('gauss_run', 0.0),
                'vol_ratio': signal_data.get('vol_ratio', 0.0),
                'deviso_ratio': signal_data.get('deviso_ratio', 0.0),
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'close_reason': "CVD Trading Stopped",
                'status': 'CLOSED'
            }
            log_trade_to_csv(trade_data)
            
            closed_positions.append(symbol)
            logger.info(f"✅ {symbol} CVD pozisyon kapatıldı: ${total_return_when_closed:.2f} iade edildi")
        
        # Pozisyonları temizle
        for symbol in closed_positions:
            del open_positions[symbol]
        
        # Son durum kaydı
        log_capital_to_csv()
        
        # CVD Özet
        logger.info("✅ CVD TOPLU KAPANIŞ TAMAMLANDI:")
        logger.info(f"   📊 Kapatılan pozisyon: {len(closed_positions)}")
        logger.info(f"   💰 Toplam iade edilen: ${total_returned:.2f}")
        logger.info(f"   🏦 Final sermaye: ${current_capital:.2f}")
        logger.info(f"   📈 Net değişim: ${total_returned - total_invested_before:.2f}")
        logger.info(f"   🔥 CVD pozisyon özeti: {', '.join(cvd_summary)}")
    
    logger.info("✅ CVD Paper Trading durduruldu")


def get_position_summary() -> Dict:
    """🔥 YENİ: CVD bilgileri ile açık pozisyon özeti"""
    summary = {
        'count': len(open_positions),
        'max_positions': MAX_OPEN_POSITIONS,
        'available_slots': MAX_OPEN_POSITIONS - len(open_positions),
        'total_invested': 0.0,
        'total_unrealized_pnl': 0.0,
        'positions': [],
        'cvd_summary': {}  # CVD özet bilgileri
    }
    
    cvd_directions = {'bullish': 0, 'bearish': 0, 'neutral': 0}
    momentum_strengths = []
    harmony_scores = []
    
    for symbol, pos in open_positions.items():
        current_price = get_current_price(symbol) or pos['entry_price']
        current_value = pos['quantity'] * current_price
        
        if pos['side'] == 'LONG':
            unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']
        else:
            unrealized_pnl = (pos['entry_price'] - current_price) * pos['quantity']
        
        summary['total_invested'] += pos['invested_amount']
        summary['total_unrealized_pnl'] += unrealized_pnl
        
        # CVD bilgilerini topla
        signal_data = pos.get('signal_data', {})
        cvd_direction = signal_data.get('cvd_direction', 'neutral')
        momentum_strength = signal_data.get('momentum_strength', 0.0)
        harmony_score = signal_data.get('deviso_cvd_harmony', 50.0)
        
        cvd_directions[cvd_direction] = cvd_directions.get(cvd_direction, 0) + 1
        momentum_strengths.append(momentum_strength)
        harmony_scores.append(harmony_score)
        
        summary['positions'].append({
            'symbol': symbol,
            'side': pos['side'],
            'quantity': pos['quantity'],
            'entry_price': pos['entry_price'],
            'current_price': current_price,
            'invested_amount': pos['invested_amount'],
            'current_value': current_value,
            'unrealized_pnl': unrealized_pnl,
            'pnl_percentage': (unrealized_pnl / pos['invested_amount'] * 100) if pos['invested_amount'] > 0 else 0,
            # CVD bilgileri
            'cvd_direction': cvd_direction,
            'momentum_strength': momentum_strength,
            'harmony_score': harmony_score,
            'signal_type': signal_data.get('signal_type', 'neutral')
        })
    
    # CVD özet istatistikleri
    if momentum_strengths:
        summary['cvd_summary'] = {
            'direction_distribution': cvd_directions,
            'avg_momentum_strength': sum(momentum_strengths) / len(momentum_strengths),
            'avg_harmony_score': sum(harmony_scores) / len(harmony_scores),
            'strong_momentum_count': len([m for m in momentum_strengths if m >= 70]),
            'high_harmony_count': len([h for h in harmony_scores if h >= 80])
        }
    
    return summary


def is_trading_active() -> bool:
    """Trading durumunu kontrol et"""
    return trading_active


def get_trading_stats() -> Dict:
    """🔥 YENİ: CVD bilgileri ile trading istatistikleri"""
    global current_capital
    
    position_summary = get_position_summary()
    effective_capital = current_capital + position_summary['total_unrealized_pnl']
    total_return_pct = ((effective_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100) if INITIAL_CAPITAL > 0 else 0.0
    
    stats = {
        'is_active': trading_active,
        'current_capital': current_capital,
        'effective_capital': effective_capital,
        'total_return_pct': total_return_pct,
        'open_positions': position_summary['count'],
        'max_positions': MAX_OPEN_POSITIONS,
        'total_invested': position_summary['total_invested'],
        'unrealized_pnl': position_summary['total_unrealized_pnl'],
        'cvd_enabled': True,  # CVD sistem aktif
        'cvd_stats': position_summary.get('cvd_summary', {})
    }
    
    return stats


# 🔥 YENİ: CVD spesifik fonksiyonlar
def get_cvd_position_analysis() -> Dict:
    """CVD pozisyon analizi"""
    if not open_positions:
        return {}
    
    analysis = {
        'total_positions': len(open_positions),
        'bullish_positions': 0,
        'bearish_positions': 0,
        'strong_momentum_positions': 0,
        'high_harmony_positions': 0,
        'avg_ai_score': 0,
        'position_details': []
    }
    
    ai_scores = []
    
    for symbol, pos in open_positions.items():
        signal_data = pos.get('signal_data', {})
        cvd_direction = signal_data.get('cvd_direction', 'neutral')
        momentum_strength = signal_data.get('momentum_strength', 0.0)
        harmony_score = signal_data.get('deviso_cvd_harmony', 50.0)
        ai_score = signal_data.get('ai_score', 0.0)
        
        ai_scores.append(ai_score)
        
        if cvd_direction == 'bullish':
            analysis['bullish_positions'] += 1
        elif cvd_direction == 'bearish':
            analysis['bearish_positions'] += 1
        
        if momentum_strength >= 70:
            analysis['strong_momentum_positions'] += 1
        
        if harmony_score >= 80:
            analysis['high_harmony_positions'] += 1
        
        current_price = get_current_price(symbol) or pos['entry_price']
        if pos['side'] == 'LONG':
            unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']
        else:
            unrealized_pnl = (pos['entry_price'] - current_price) * pos['quantity']
        
        analysis['position_details'].append({
            'symbol': symbol,
            'side': pos['side'],
            'cvd_direction': cvd_direction,
            'momentum_strength': momentum_strength,
            'harmony_score': harmony_score,
            'ai_score': ai_score,
            'unrealized_pnl': unrealized_pnl,
            'entry_price': pos['entry_price'],
            'current_price': current_price
        })
    
    if ai_scores:
        analysis['avg_ai_score'] = sum(ai_scores) / len(ai_scores)
    
    return analysis