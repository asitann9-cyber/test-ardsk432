"""
💰 Paper Trading Sistemi
Sanal trading ve pozisyon yönetimi
🔥 DÜZELTME: En iyi 3 seçimi için AI skoru zorlaması kaldırıldı
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
    Paper trading pozisyon aç - KOMİSYON YOK
    """
    global current_capital, open_positions
    
    try:
        symbol = signal['symbol']
        side = signal['run_type'].upper()
        
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
        
        logger.info(f"✅ PAPER TRADE AÇILDI: {symbol} {side} {quantity} @ ${current_price:.6f}")
        logger.info(f"💰 Yatırılan: ${invested_amount:.2f} | Kalan sermaye: ${current_capital:.2f}")
        logger.info(f"📊 SL: ${stop_loss:.6f} | TP: ${take_profit:.6f} | AI: {signal['ai_score']:.0f}%")
        
        # CSV'ye kaydet
        trade_data = {
            'timestamp': position_data['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': current_price,
            'invested_amount': invested_amount,
            'commission': 0.0,
            'ai_score': signal['ai_score'],
            'run_type': signal['run_type'],
            'run_count': signal['run_count'],
            'run_perc': signal['run_perc'],
            'gauss_run': signal['gauss_run'],
            'vol_ratio': signal.get('vol_ratio', 0),
            'deviso_ratio': signal.get('deviso_ratio', 0),
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
    Paper trading pozisyon kapat - KOMİSYON YOK
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
        
        # CSV
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
            'ai_score': position['signal_data']['ai_score'],
            'run_type': position['signal_data']['run_type'],
            'run_count': position['signal_data']['run_count'],
            'run_perc': position['signal_data']['run_perc'],
            'gauss_run': position['signal_data']['gauss_run'],
            'vol_ratio': position['signal_data'].get('vol_ratio', 0),
            'deviso_ratio': position['signal_data'].get('deviso_ratio', 0),
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
    🔥 DÜZELTME: TÜM SİNYALLER tablosundaki filtrelenmiş sonuçlardan GERÇEK en iyileri seç.
    AI skoru zorlamasını kaldır - kullanıcı filtresini kullan
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

        # 👉 AI skoru 0–1 geldiyse 0–100'e çevir
        try:
            if df["ai_score"].max() <= 1.0:
                df["ai_score"] = (df["ai_score"] * 100.0).clip(0, 100)
        except Exception:
            pass

        # 🔥 DÜZELTME: Kullanıcının belirlediği filtreleri kullan (zorlamalar yok)
        min_streak = int(current_settings.get('min_streak', 3))
        min_pct    = float(current_settings.get('min_pct', 0.5))
        min_volr   = float(current_settings.get('min_volr', 1.5))
        min_ai_pct = float(current_settings.get('min_ai', 30))  # Kullanıcı ayarı

        # ❌ ESKİ: Zorla %90 AI skoru (KALDIRILDI)
        # if min_ai_pct < 90:
        #     min_ai_pct = 90.0

        # ✅ YENİ: Kullanıcı ayarını kullan (eğer çok düşükse güvenlik için %15 minimum)
        trading_min_ai = max(min_ai_pct, 15.0)  # Minimum %15 güvenlik

        logger.info(f"🎯 Paper Trading filtreleri: AI≥{trading_min_ai:.0f}%, Streak≥{min_streak}, Move≥{min_pct}%, Vol≥{min_volr}")

        # Filtreleme uygula
        original_count = len(df)
        
        df = df[
            (df['run_count'] >= min_streak) &
            (df['run_perc']  >= min_pct)    &
            (df['ai_score']  >= trading_min_ai)  # ✅ Kullanıcı ayarı
        ]

        # Volume filtresi
        if 'vol_ratio' in df.columns:
            df = df[df['vol_ratio'].fillna(0) >= min_volr]

        logger.info(f"🔍 Filtreleme: {len(df)}/{original_count} sinyal kaldı (AI≥{trading_min_ai:.0f}%)")

        if df.empty:
            logger.warning(f"⚠️ Filtre sonrası uygun sinyal yok - AI skoru {trading_min_ai:.0f}% altında hiç sinyal yok")
            return

        # Zaten açık pozisyonları hariç tut
        exclude_symbols = set(open_positions.keys())
        before_exclude = len(df)
        df = df[~df['symbol'].isin(exclude_symbols)]
        logger.info(f"🧮 Açık pozisyonlar hariç: {len(df)} (önce: {before_exclude})")
        
        if df.empty:
            logger.info("ℹ️ Uygun yeni sembol yok (hepsi açık pozisyonlarda)")
            return

        # 🔥 DÜZELTME: ÖNCE SIRALA, SONRA EN İYİ 3'Ü AL
        df = df.sort_values(
            by=['ai_score', 'run_perc', 'gauss_run', 'vol_ratio'],
            ascending=[False, False, False, False]
        )

        # --- GERÇEK en iyi 3 kuralı ---
        top_n = min(3, needed_positions, len(df))
        top_signals = df.head(top_n)

        logger.info(f"🎯 GERÇEK EN İYİ {top_n} SİNYAL (AI≥{trading_min_ai:.0f}%) → seçilen:")
        for i, (_, s) in enumerate(top_signals.iterrows(), 1):
            logger.info(f"   🥇 #{i}: {s['symbol']} | AI={s['ai_score']:.0f}% | run={s['run_count']} | move={s['run_perc']:.2f}%")

        # Pozisyonları aç
        opened_count = 0
        for _, signal in top_signals.iterrows():
            if not trading_active:
                break
            if open_position(signal.to_dict()):
                opened_count += 1

        logger.info(f"🏆 SONUÇ: {opened_count} yeni pozisyon açıldı | Toplam: {len(open_positions)}/{MAX_OPEN_POSITIONS}")

    except Exception as e:
        logger.error(f"❌ fill_empty_positions hatası: {e}")


def paper_trading_loop():
    """Ana paper trading döngüsü - SADECE current_data kullan"""
    global trading_active
    
    logger.info("🚀 Paper Trading döngüsü başlatıldı")
    logger.info("🔒 SADECE TÜM SİNYALLER tablosundan beslenecek!")
    loop_count = 0
    
    while trading_active:
        try:
            loop_count += 1
            loop_start = time.time()
            
            current_position_count = len(open_positions)
            logger.info(f"🔄 Döngü #{loop_count} - Pozisyon: {current_position_count}/{MAX_OPEN_POSITIONS}")
            
            # Veri durumu
            if current_data is None or current_data.empty:
                logger.warning(f"⚠️ Döngü #{loop_count}: TÜM SİNYALLER tablosu boş - otomatik tarama başlatın!")
            else:
                logger.info(f"📊 Döngü #{loop_count}: TÜM SİNYALLER tablosunda {len(current_data)} sinyal mevcut")
            
            # Boş pozisyonları doldur
            fill_empty_positions()
            
            # Açık pozisyonları izle
            monitor_positions()
            
            # Sermaye durumunu kaydet
            log_capital_to_csv()
            
            # Performans özeti
            total_unrealized_pnl = 0.0
            
            for symbol, pos in open_positions.items():
                current_price = get_current_price(symbol) or pos['entry_price']
                
                if pos['side'] == 'LONG':
                    unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']
                else:
                    unrealized_pnl = (pos['entry_price'] - current_price) * pos['quantity']

                
                total_unrealized_pnl += unrealized_pnl
            
            loop_time = time.time() - loop_start
            
            logger.info(f"⏱️ Döngü #{loop_count}: {loop_time:.2f}s | Sermaye: ${current_capital:.2f} | P&L: ${total_unrealized_pnl:.2f}")
            
            if open_positions:
                positions_summary = ", ".join(open_positions.keys())
                logger.info(f"🔥 Açık pozisyonlar: {positions_summary}")
                logger.info(f"📊 Bu pozisyonlar TÜM SİNYALLER tablosundan seçildi ✅")
            
            time.sleep(SCAN_INTERVAL)
            
        except Exception as e:
            logger.error(f"❌ Paper Trading döngüsü hatası: {e}")
            time.sleep(30)
    
    logger.info("ℹ️ Paper Trading döngüsü sonlandırıldı")


def start_paper_trading() -> bool:
    """Paper Trading'i başlat"""
    global trading_active, paper_trade_thread, current_capital
    
    if trading_active:
        logger.warning("⚠️ Paper Trading zaten aktif")
        return False
    
    logger.info("🚀 Paper Trading başlatılıyor...")
    logger.info(f"💰 Başlangıç sermayesi: ${INITIAL_CAPITAL:.2f}")
    logger.info(f"📊 Maksimum pozisyon: {MAX_OPEN_POSITIONS} (TÜM SİNYALLER tablosundan GERÇEK EN İYİLER)")
    logger.info(f"⏰ Tarama aralığı: {SCAN_INTERVAL} saniye")
    logger.info(f"🛑 Stop Loss: %{STOP_LOSS_PCT*100}")
    logger.info(f"🎯 Take Profit: %{TAKE_PROFIT_PCT*100}")
    logger.info(f"💳 Komisyon: KALDIRILDI")
    logger.info(f"💰 Pozisyon Büyüklüğü: Toplam sermaye ÷ 3")
    logger.info(f"🔥 DÜZELTME: AI skoru zorlaması kaldırıldı - kullanıcı ayarlarını kullanır")
    
    # Başlangıç sermayesini sıfırla
    current_capital = INITIAL_CAPITAL
    
    # CSV dosyaları
    from data.database import setup_csv_files
    setup_csv_files()
    
    trading_active = True
    
    # Thread başlat
    paper_trade_thread = threading.Thread(target=paper_trading_loop, daemon=True)
    paper_trade_thread.start()
    
    logger.info("✅ Paper Trading başlatıldı")
    return True


def stop_paper_trading():
    """Paper Trading'i durdur - KOMİSYON YOK"""
    global trading_active, current_capital, open_positions
    
    if not trading_active:
        logger.info("💤 Paper Trading zaten durdurulmuş")
        return
    
    logger.info("🛑 Paper Trading durduruluyor...")
    trading_active = False
    
    if open_positions:
        position_count = len(open_positions)
        logger.info(f"📚 {position_count} açık pozisyon toplu olarak kapatılıyor...")
        
        # Kapanış öncesi durum
        total_invested_before = sum(pos.get('invested_amount', 0) for pos in open_positions.values())
        logger.info(f"💰 Kapanış öncesi toplam yatırılan: ${total_invested_before:.2f}")
        logger.info(f"🏦 Kapanış öncesi mevcut sermaye: ${current_capital:.2f}")
        
        closed_positions = []
        total_returned = 0.0
        
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
            
            # Trade kaydı
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
                'ai_score': position['signal_data']['ai_score'],
                'run_type': position['signal_data']['run_type'],
                'run_count': position['signal_data']['run_count'],
                'run_perc': position['signal_data']['run_perc'],
                'gauss_run': position['signal_data']['gauss_run'],
                'vol_ratio': position['signal_data'].get('vol_ratio', 0),
                'deviso_ratio': position['signal_data'].get('deviso_ratio', 0),
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'close_reason': "Trading Stopped",
                'status': 'CLOSED'
            }
            log_trade_to_csv(trade_data)
            
            closed_positions.append(symbol)
            logger.info(f"✅ {symbol} kapatıldı: ${total_return_when_closed:.2f} iade edildi (komisyonsuz)")
        
        # Pozisyonları temizle
        for symbol in closed_positions:
            del open_positions[symbol]
        
        # Son durum kaydı
        log_capital_to_csv()
        
        # Özet
        logger.info("✅ TOPLU KAPANIŞ TAMAMLANDI:")
        logger.info(f"   📊 Kapatılan pozisyon: {len(closed_positions)}")
        logger.info(f"   💰 Toplam iade edilen: ${total_returned:.2f}")
        logger.info(f"   🏦 Final sermaye: ${current_capital:.2f}")
        logger.info(f"   📈 Net değişim: ${total_returned - total_invested_before:.2f}")
    
    logger.info("✅ Paper Trading durduruldu")


def get_position_summary() -> Dict:
    """Açık pozisyonların özetini al"""
    summary = {
        'count': len(open_positions),
        'max_positions': MAX_OPEN_POSITIONS,
        'available_slots': MAX_OPEN_POSITIONS - len(open_positions),
        'total_invested': 0.0,
        'total_unrealized_pnl': 0.0,
        'positions': []
    }
    
    for symbol, pos in open_positions.items():
        current_price = get_current_price(symbol) or pos['entry_price']
        current_value = pos['quantity'] * current_price
        
        if pos['side'] == 'LONG':
            unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']
        else:
            unrealized_pnl = (pos['entry_price'] - current_price) * pos['quantity']
        
        summary['total_invested'] += pos['invested_amount']
        summary['total_unrealized_pnl'] += unrealized_pnl
        
        summary['positions'].append({
            'symbol': symbol,
            'side': pos['side'],
            'quantity': pos['quantity'],
            'entry_price': pos['entry_price'],
            'current_price': current_price,
            'invested_amount': pos['invested_amount'],
            'current_value': current_value,
            'unrealized_pnl': unrealized_pnl,
            'pnl_percentage': (unrealized_pnl / pos['invested_amount'] * 100) if pos['invested_amount'] > 0 else 0
        })
    
    return summary


def is_trading_active() -> bool:
    """Trading durumunu kontrol et"""
    return trading_active


def get_trading_stats() -> Dict:
    """Trading istatistiklerini al"""
    global current_capital
    
    position_summary = get_position_summary()
    effective_capital = current_capital + position_summary['total_unrealized_pnl']
    total_return_pct = ((effective_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100) if INITIAL_CAPITAL > 0 else 0.0
    
    return {
        'is_active': trading_active,
        'current_capital': current_capital,
        'effective_capital': effective_capital,
        'total_return_pct': total_return_pct,
        'open_positions': position_summary['count'],
        'max_positions': MAX_OPEN_POSITIONS,
        'total_invested': position_summary['total_invested'],
        'unrealized_pnl': position_summary['total_unrealized_pnl']
    }