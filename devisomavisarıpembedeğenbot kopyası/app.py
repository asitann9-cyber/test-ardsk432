#!/usr/bin/env python3
"""
Supertrend + C-Signal + Ters Momentum Analiz Sistemi
Ana Flask Uygulaması - Clean Architecture Implementation
"""
import pandas as pd
import logging
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Local imports
from config import Config
from services.analysis_service import AnalysisService
from services.binance_service import BinanceService
from services.telegram_service import TelegramService
from utils.memory_storage import MemoryStorage
from utils.helpers import create_tradingview_link, format_time

# Logging setup
def setup_logging():
    """Merkezi logging konfigürasyonu"""
    logger = logging.getLogger("SupertrendSystem")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler("supertrend_system.log")
    file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

def create_app():
    """Flask uygulaması factory pattern"""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Services initialization
    memory_storage = MemoryStorage()
    binance_service = BinanceService()
    telegram_service = TelegramService()
    analysis_service = AnalysisService()
    
    # Routes
    register_routes(app, analysis_service, memory_storage, binance_service, telegram_service)
    
    logger.info("Flask uygulaması başarıyla oluşturuldu")
    return app

def register_routes(app, analysis_service, memory_storage, binance_service, telegram_service):
    """Tüm route'ları kaydet"""
    
    @app.route('/')
    def index():
        """Ana sayfa"""
        return render_template('index.html')
    
    @app.route('/api/consecutive/symbols')
    def get_all_symbols():
        """Binance sembollerini getir"""
        try:
            symbols = binance_service.fetch_symbols()
            return jsonify({
                "success": True, 
                "symbols": symbols,
                "total": len(symbols)
            })
        except Exception as e:
            logger.error(f"Sembol listesi API hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/consecutive/selected-symbols', methods=['GET'])
    def get_selected_symbols():
        """Seçili semboller listesi"""
        try:
            symbols = memory_storage.get_selected_symbols()
            return jsonify({
                "success": True, 
                "symbols": symbols, 
                "count": len(symbols)
            })
        except Exception as e:
            logger.error(f"Seçili sembol listesi API hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/consecutive/selected-symbols', methods=['POST'])
    def manage_selected_symbols():
        """Seçili sembolleri yönet (ekle/sil/temizle)"""
        try:
            data = request.get_json()
            action = data.get('action')
            symbols_to_add = data.get('symbols', [])
            symbol_to_remove = data.get('symbol_to_remove')
            
            if action == 'add':
                if symbols_to_add:
                    all_symbols = memory_storage.add_selected_symbols(symbols_to_add)
                    added_count = len(set(symbols_to_add) - set(memory_storage.get_selected_symbols()[:-len(symbols_to_add)]))
                    
                    return jsonify({
                        "success": True, 
                        "message": f"{added_count} yeni sembol eklendi. Toplam: {len(all_symbols)}",
                        "symbols": all_symbols,
                        "count": len(all_symbols)
                    })
                else:
                    return jsonify({"success": False, "error": "Eklenecek sembol bulunamadı"})
            
            elif action == 'add_all':
                all_available_symbols = binance_service.fetch_symbols()
                if all_available_symbols:
                    memory_storage.save_selected_symbols(all_available_symbols)
                    
                    return jsonify({
                        "success": True,
                        "message": f"TÜM EMTİALAR SEÇİLDİ! Toplam: {len(all_available_symbols)}",
                        "symbols": all_available_symbols,
                        "count": len(all_available_symbols)
                    })
                else:
                    return jsonify({"success": False, "error": "Sembol listesi alınamadı"})
            
            elif action == 'remove':
                if symbol_to_remove:
                    remaining_symbols = memory_storage.remove_selected_symbol(symbol_to_remove)
                    
                    return jsonify({
                        "success": True,
                        "message": f"{symbol_to_remove} silindi. Kalan: {len(remaining_symbols)}",
                        "symbols": remaining_symbols,
                        "count": len(remaining_symbols)
                    })
                else:
                    return jsonify({"success": False, "error": "Silinecek sembol bulunamadı"})
            
            elif action == 'clear':
                memory_storage.clear_selected_symbols()
                return jsonify({
                    "success": True,
                    "message": "Tüm semboller temizlendi",
                    "symbols": [],
                    "count": 0
                })
            
            else:
                return jsonify({"success": False, "error": "Geçersiz işlem"})
                
        except Exception as e:
            logger.error(f"Sembol yönetimi API hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/consecutive/analyze', methods=['POST'])
    def analyze_supertrend():
        """Supertrend analizi gerçekleştir - ÖRNEK KODDAN ALINMIŞ MANTIK"""
        try:
            data = request.get_json()
            timeframe = data.get('timeframe', '4h')
            
            selected_symbols = memory_storage.get_selected_symbols()
            if not selected_symbols:
                return jsonify({"success": False, "error": "Hiç sembol seçilmedi"})
            
            logger.info(f"{len(selected_symbols)} sembol için {timeframe} Supertrend analizi başlatılıyor...")
            
            # Paralel analiz
            results = analysis_service.analyze_multiple_symbols(selected_symbols, timeframe)
            
            if results:
                current_time = datetime.now().strftime('%H:%M')
                
                # Ratio >= 100% olan emtiaları kalıcı listeye ekle
                for result in results:
                    if analysis_service.is_high_priority_symbol(result):
                        # TradingView linkini ekle
                        result['tradingview_link'] = create_tradingview_link(result['symbol'], timeframe)
                        # Kalıcı listeye ekle
                        memory_storage.add_permanent_symbol(result)
                
                # Kalıcı listeki C-Signal'leri güncelle + Telegram bildirimi  
                permanent_symbols = memory_storage.get_permanent_symbols()
                reverse_momentum_count = 0
                
                for symbol_data in permanent_symbols:
                    # C-Signal ile güncelle (MANUEL TÜR KORUNARAK)
                    updated_symbol = analysis_service.update_symbol_with_c_signal(
                        symbol_data, 
                        symbol_data.get('timeframe', '4h'),
                        preserve_manual_type=symbol_data.get('manual_type_override', False)
                    )
                    
                    # Ters momentum varsa Telegram bildirimi gönder
                    reverse_momentum = updated_symbol.get('reverse_momentum', {})
                    if reverse_momentum.get('has_reverse_momentum', False):
                        reverse_momentum_count += 1
                        
                        # Spam önleme kontrolü
                        if telegram_service.should_send_alert(updated_symbol, reverse_momentum):
                            success = telegram_service.send_reverse_momentum_alert(updated_symbol)
                            if success:
                                updated_symbol['last_telegram_alert'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # ÖRNEK KODDAN ALINMIŞ - Sonuçları formatla (final_ratio, change_momentum dahil)
                formatted_results = []
                
                for i, result in enumerate(results, 1):
                    tradingview_link = create_tradingview_link(result['symbol'], timeframe)
                    
                    # ÖRNEK KODDAN - change_momentum ve momentum_bars_ago formatla
                    change_momentum = result.get('change_momentum', 0)
                    momentum_bars_ago = result.get('momentum_bars_ago', 0)
                    
                    if change_momentum and not pd.isna(change_momentum):
                        c_signal_display = f"C: {change_momentum:+.1f} ({int(momentum_bars_ago)} mum)"
                    else:
                        c_signal_display = "N/A"
                    
                    formatted_results.append({
                        'rank': i,
                        'symbol': result['symbol'],
                        'tradingview_link': tradingview_link,
                        'current_price': round(result['current_price'], 4),
                        'ratio_percent': result['ratio_percent'],
                        'z_score': result['z_score'],
                        'final_ratio': result.get('final_ratio', result['ratio_percent']),  # ÖRNEK KODDAN
                        'c_signal_display': c_signal_display,  # ÖRNEK KODDAN
                        'trend_direction': result.get('trend_direction', 'None'),
                        'price_vs_supertrend': result.get('price_vs_supertrend', 'None'),
                        'last_update': current_time
                    })
                
                return jsonify({
                    "success": True,
                    "results": formatted_results,
                    "count": len(formatted_results),
                    "timeframe": timeframe,
                    "message": f"{len(formatted_results)} sembol analiz edildi - Supertrend + C-Signal + Telegram bildirimleri"
                })
            else:
                return jsonify({"success": False, "error": "Analiz sonucu bulunamadı"})
                
        except Exception as e:
            logger.error(f"Supertrend analiz API hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/consecutive/permanent-high-consecutive', methods=['GET'])
    def get_permanent_high_ratio():
        """Kalıcı %100+ ratio emtialar listesi - C-Signal + Ters Momentum ile"""
        try:
            # C-Signal değerlerini güncelle + Ters momentum tespit et + Telegram bildirimi
            update_permanent_c_signals(memory_storage, analysis_service, telegram_service)
            
            permanent_symbols = memory_storage.get_permanent_symbols()
            
            # Ters momentum sayısını hesapla
            reverse_momentum_count = sum(1 for s in permanent_symbols 
                                       if s.get('reverse_momentum', {}).get('has_reverse_momentum', False))
            
            # Her kalıcı sembol için ek bilgiler ekle
            formatted_permanent = []
            for i, symbol_data in enumerate(permanent_symbols, 1):
                reverse_momentum = symbol_data.get('reverse_momentum', {})
                
                formatted_permanent.append({
                    'rank': i,
                    'symbol': symbol_data['symbol'],
                    'tradingview_link': symbol_data.get('tradingview_link', '#'),
                    'first_date': symbol_data.get('first_high_consecutive_date', 'Bilinmiyor'),
                    'max_ratio_percent': symbol_data.get('max_ratio_percent', 0),
                    'max_supertrend_type': symbol_data.get('max_supertrend_type', 'None'),
                    'max_z_score': symbol_data.get('max_z_score', 0),
                    'max_abs_ratio_percent': abs(symbol_data.get('max_ratio_percent', 0)),
                    'timeframe': symbol_data.get('timeframe', '4h'),
                    'c_signal': symbol_data.get('c_signal', 'N/A'),
                    'c_signal_update_time': symbol_data.get('c_signal_update_time', 'N/A'),
                    'add_reason': symbol_data.get('add_reason', 'Bilinmiyor'),
                    # TERS MOMENTUM BİLGİLERİ
                    'reverse_momentum': reverse_momentum.get('has_reverse_momentum', False),
                    'reverse_type': reverse_momentum.get('reverse_type', 'None'),
                    'signal_strength': reverse_momentum.get('signal_strength', 'None'),
                    'alert_message': reverse_momentum.get('alert_message', ''),
                    'last_telegram_alert': symbol_data.get('last_telegram_alert', 'Hiç gönderilmedi'),
                    # MANUEL TÜR BİLGİLERİ
                    'manual_type_override': symbol_data.get('manual_type_override', False),
                    'manual_override_date': symbol_data.get('manual_override_date', None)
                })
            
            telegram_bot_status = "✅" if telegram_service.bot_token else "❌"
            telegram_chat_status = "✅" if telegram_service.chat_id else "❌"
            
            return jsonify({
                "success": True,
                "permanent_symbols": formatted_permanent,
                "count": len(formatted_permanent),
                "reverse_momentum_count": reverse_momentum_count,
                "telegram_status": f"Bot Token: {telegram_bot_status} | Chat ID: {telegram_chat_status}",
                "message": f"Kalıcı listede {len(formatted_permanent)} emtia (%100+ Ratio + C-Signal + Ters Momentum: {reverse_momentum_count}) + Telegram bildirimleri"
            })
            
        except Exception as e:
            logger.error(f"Kalıcı liste API hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/consecutive/clear-permanent', methods=['POST'])
    def clear_permanent_high_ratio():
        """Kalıcı %100+ ratio listesini temizle"""
        try:
            old_count = memory_storage.clear_permanent_symbols()
            
            return jsonify({
                "success": True,
                "message": f"Kalıcı liste temizlendi. {old_count} emtia silindi.",
                "count": 0
            })
            
        except Exception as e:
            logger.error(f"Kalıcı liste temizleme hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # =====================================================
    # MANUEL EMTİA EKLEME ENDPOINT'İ - SUPERTREND İÇİN
    # =====================================================
    @app.route('/api/consecutive/add-to-permanent', methods=['POST'])
    def add_symbol_to_permanent():
        """Manuel olarak emtia kalıcı listeye ekleme - Supertrend için"""
        try:
            data = request.get_json()
            symbol = data.get('symbol')
            timeframe = data.get('timeframe', '4h')
            
            if not symbol:
                return jsonify({"success": False, "error": "Sembol adı gerekli"})
            
            # Mevcut kontrolü
            existing_permanent = memory_storage.get_permanent_symbol(symbol)
            if existing_permanent:
                return jsonify({
                    "success": False, 
                    "error": f"{symbol} zaten kalıcı listede mevcut"
                })
            
            # Güncel veriyi al - Supertrend analizi
            current_analysis = analysis_service.analyze_single_symbol(symbol, timeframe)
            
            if not current_analysis:
                return jsonify({
                    "success": False, 
                    "error": f"{symbol} için veri alınamadı veya Supertrend analizi yapılamadı"
                })
            
            # Manuel ekleme için sembol verisi oluştur
            manual_symbol_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'trend_direction': current_analysis.get('trend_direction', 'None'),
                'ratio_percent': current_analysis.get('ratio_percent', 0),
                'z_score': current_analysis.get('z_score', 0),
                'current_price': current_analysis.get('current_price', 0),
                'tradingview_link': create_tradingview_link(symbol, timeframe),
                'last_update': current_analysis.get('last_update', datetime.now())
            }
            
            # Kalıcı listeye ekle
            memory_storage.add_permanent_symbol(manual_symbol_data)
            
            # C-Signal ve ters momentum ile güncelle
            permanent_symbol = memory_storage.get_permanent_symbol(symbol)
            if permanent_symbol:
                updated_symbol = analysis_service.update_symbol_with_c_signal(permanent_symbol, timeframe)
                
                # Ters momentum varsa Telegram bildirimi gönder
                reverse_momentum = updated_symbol.get('reverse_momentum', {})
                if reverse_momentum.get('has_reverse_momentum', False):
                    if telegram_service.should_send_alert(updated_symbol, reverse_momentum):
                        success = telegram_service.send_reverse_momentum_alert(updated_symbol)
                        if success:
                            updated_symbol['last_telegram_alert'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            memory_storage.update_permanent_symbol(symbol, {'last_telegram_alert': updated_symbol['last_telegram_alert']})
            
            logger.info(f"✅ {symbol} manuel olarak kalıcı listeye eklendi (Supertrend)")
            
            return jsonify({
                "success": True,
                "message": f"✅ {symbol} kalıcı listeye eklendi! (Supertrend manuel ekleme + C-Signal + Ters momentum)",
                "symbol_data": {
                    'symbol': symbol,
                    'ratio_percent': manual_symbol_data['ratio_percent'],
                    'trend_direction': manual_symbol_data['trend_direction'],
                    'z_score': manual_symbol_data['z_score'],
                    'timeframe': timeframe
                }
            })
                
        except Exception as e:
            logger.error(f"Manuel emtia ekleme hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/consecutive/remove-from-permanent', methods=['POST'])
    def remove_symbol_from_permanent():
        """Kalıcı listeden emtia çıkarma"""
        try:
            data = request.get_json()
            symbol = data.get('symbol')
            
            if not symbol:
                return jsonify({"success": False, "error": "Sembol adı gerekli"})
            
            # Mevcut kontrolü
            existing_permanent = memory_storage.get_permanent_symbol(symbol)
            if not existing_permanent:
                return jsonify({
                    "success": False, 
                    "error": f"{symbol} kalıcı listede bulunamadı"
                })
            
            # Kalıcı listeden çıkar
            success = memory_storage.remove_permanent_symbol(symbol)
            
            if success:
                logger.info(f"✅ {symbol} kalıcı listeden çıkarıldı")
                return jsonify({
                    "success": True,
                    "message": f"✅ {symbol} kalıcı listeden çıkarıldı!"
                })
            else:
                return jsonify({
                    "success": False,
                    "error": f"{symbol} çıkarılırken hata oluştu"
                })
                    
        except Exception as e:
            logger.error(f"Kalıcı listeden çıkarma hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # =====================================================
    # MANUEL TÜR DEĞİŞTİRME - SUPERTREND İÇİN UYARLANMIŞ
    # =====================================================
    @app.route('/api/consecutive/update-symbol-type', methods=['POST'])
    def update_symbol_type():
        """Kalıcı listedeki sembolün trend türünü manuel güncelle (Bullish/Bearish)"""
        try:
            data = request.get_json()
            symbol = data.get('symbol')
            new_type = data.get('new_type')
            
            if not symbol or new_type not in ['Bullish', 'Bearish']:
                return jsonify({
                    "success": False, 
                    "error": "Geçersiz sembol veya tür bilgisi (Bullish/Bearish olmalı)"
                })
            
            # Kalıcı sembolü bul ve güncelle
            permanent_symbol = memory_storage.get_permanent_symbol(symbol)
            if not permanent_symbol:
                return jsonify({
                    "success": False, 
                    "error": f"{symbol} kalıcı listede bulunamadı"
                })
            
            # Eski türü kaydet
            old_type = permanent_symbol.get('max_supertrend_type', 'None')
            
            # MANUEL TÜR DEĞİŞTİRME VE KİLİTLEME
            success = memory_storage.set_manual_type_override(symbol, new_type)
            
            if not success:
                return jsonify({
                    "success": False,
                    "error": f"{symbol} manuel tür değişikliği yapılamadı"
                })
            
            # MANUEL DEĞİŞİKLİK = OTOMATIK TERS MOMENTUM TETİKLEME
            reverse_momentum_data = {
                'has_reverse_momentum': True,
                'reverse_type': 'MANUEL',
                'signal_strength': 'Strong',
                'signal_value': None,
                'alert_message': f'MANUEL TERS MOMENTUM: {old_type} → {new_type} trend değişikliği yapıldı!'
            }
            
            # Ters momentum verisini kaydet
            memory_storage.update_permanent_symbol(symbol, {
                'reverse_momentum': reverse_momentum_data,
                'last_telegram_alert': None  # Yeni bildirim için sıfırla
            })
            
            # Telegram bildirimi gönder (manuel değişiklik bildirimi)
            updated_permanent_symbol = memory_storage.get_permanent_symbol(symbol)
            if updated_permanent_symbol and telegram_service.should_send_alert(updated_permanent_symbol, reverse_momentum_data):
                success_telegram = telegram_service.send_reverse_momentum_alert({
                    'symbol': symbol,
                    'reverse_momentum': reverse_momentum_data,
                    'tradingview_link': permanent_symbol.get('tradingview_link', '#')
                })
                if success_telegram:
                    memory_storage.update_permanent_symbol(symbol, {
                        'last_telegram_alert': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            logger.info(f"🔒 {symbol} trend türü manuel olarak {old_type} -> {new_type} DEĞİŞTİRİLDİ ve KİLİTLENDİ")
            
            return jsonify({
                "success": True,
                "message": f"🔒 {symbol} trend türü {old_type} → {new_type} değiştirildi ve KİLİTLENDİ! OTOMATIK TERS MOMENTUM tetiklendi!",
                "symbol": symbol,
                "old_type": old_type,
                "new_type": new_type,
                "is_locked": True,
                "manual_reverse_momentum": True
            })
            
        except Exception as e:
            logger.error(f"Tür güncelleme hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/consecutive/unlock-symbol-type', methods=['POST'])
    def unlock_symbol_type():
        """Kalıcı listedeki sembolün manuel tür kilidini kaldır - gerçek veriye dön"""
        try:
            data = request.get_json()
            symbol = data.get('symbol')
            
            if not symbol:
                return jsonify({
                    "success": False, 
                    "error": "Sembol adı gerekli"
                })
            
            # Sembolü bul
            permanent_symbol = memory_storage.get_permanent_symbol(symbol)
            if not permanent_symbol:
                return jsonify({
                    "success": False, 
                    "error": f"{symbol} kalıcı listede bulunamadı"
                })
            
            # Manuel kilidi kontrol et
            if not permanent_symbol.get('manual_type_override', False):
                return jsonify({
                    "success": False, 
                    "error": f"{symbol} zaten manuel olarak kilitlenmemiş"
                })
            
            # Kilidi kaldır
            success = memory_storage.clear_manual_type_override(symbol)
            
            if success:
                logger.info(f"🔓 {symbol} manuel tür kilidi kaldırıldı")
                return jsonify({
                    "success": True,
                    "message": f"🔓 {symbol} manuel tür kilidi kaldırıldı! Bir sonraki güncellemede gerçek veriye dönecek."
                })
            else:
                return jsonify({
                    "success": False,
                    "error": f"{symbol} kilit kaldırma işlemi başarısız"
                })
                
        except Exception as e:
            logger.error(f"Kilit kaldırma hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/telegram/test', methods=['POST'])
    def test_telegram():
        """Telegram bot testi"""
        try:
            if not telegram_service.is_configured():
                return jsonify({
                    "success": False,
                    "error": "Telegram konfigürasyonu eksik. .env dosyasında TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID ayarlayın."
                })
            
            success = telegram_service.send_test_message()
            
            if success:
                return jsonify({
                    "success": True,
                    "message": "Test mesajı başarıyla gönderildi!"
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Test mesajı gönderilemedi. Bot token ve chat ID'yi kontrol edin."
                })
                
        except Exception as e:
            logger.error(f"Telegram test hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"success": False, "error": "Endpoint bulunamadı"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"İnternal server hatası: {error}")
        return jsonify({"success": False, "error": "Sunucu hatası"}), 500

def update_permanent_c_signals(memory_storage, analysis_service, telegram_service):
    """
    SADECE C-SIGNAL GÜNCELLE - Ratio/Z-Score güncellemesi YOK
    """
    current_time = datetime.now().strftime('%H:%M')
    permanent_symbols = memory_storage.get_permanent_symbols()
    
    for symbol_data in permanent_symbols:
        try:
            symbol = symbol_data['symbol']
            timeframe = symbol_data.get('timeframe', '4h')
            
            # Binance'den veri çek ve SADECE C-Signal hesapla
            df = BinanceService.fetch_klines_data(symbol, timeframe, limit=500)
            if df is not None and len(df) >= 50:
                c_signal = analysis_service.calculate_c_signal(df)
                symbol_data['c_signal'] = c_signal
                symbol_data['c_signal_update_time'] = current_time
                
                # Ters momentum kontrolü
                reverse_momentum = analysis_service.detect_reverse_momentum(symbol_data)
                symbol_data['reverse_momentum'] = reverse_momentum
                
                # Telegram bildirimi
                if reverse_momentum.get('has_reverse_momentum', False):
                    if telegram_service.should_send_alert(symbol_data, reverse_momentum):
                        success = telegram_service.send_reverse_momentum_alert(symbol_data)
                        if success:
                            memory_storage.update_permanent_symbol(symbol, {
                                'last_telegram_alert': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                            
        except Exception as e:
            logger.debug(f"C-Signal güncelleme hatası {symbol}: {e}")

def main():
    """Ana uygulama başlatma fonksiyonu - ÖRNEK KODDAN ALINMIŞ PARAMETRELER İLE"""
    try:
        print("\n" + "="*70)
        print("📈 Supertrend + C-Signal + Ters Momentum Analiz Sistemi")
        print("="*70)
        print("📊 ÖZELLİKLER:")
        print("   ✅ Clean Architecture Implementation")
        print("   ✅ Binance USDT Vadeli İşlem Emtia Yönetimi")
        print("   ✅ Supertrend Ratio % Analizi (ÖRNEK KODDAN ALINMIŞ)")
        print("   ✅ Z-Score Hesaplama + use_z_score parametresi")
        print("   ✅ TA-Lib ATR Hesaplama (daha güvenilir)")
        print("   ✅ 500 Mum Verisi (eskiden 200)")
        print("   ✅ Paralel Analiz (5 worker)")
        print("   ✅ Bellekte Veri Tutma (CSV yok)")
        print("   ✅ TradingView Grafik Entegrasyonu")
        print("   ✅ C-Signal + Ters Momentum Tespiti")
        print("   ✅ Telegram Bildirimleri")
        print("   ✅ Filtreleme (Hepsi / Bullish / Bearish / %100+ Ratio)")
        print("   🆕 Manuel Kalıcı Liste Ekleme")
        print("   🆕 Manuel Trend Türü Değiştirme (Bullish ↔ Bearish)")
        print("   🔒 KALICI Tür Koruma - Otomatik güncellemelerde korunur")
        print("   🚨 MANUEL TERS MOMENTUM - Trend değiştirince otomatik tetikleme")
        print("="*70)
        print("📈 SUPERTREND ANALİZİ (ÖRNEK KODDAN ALINMIŞ):")
        print("   • TA-Lib ATR bazlı Supertrend hesaplama")
        print("   • Basitleştirilmiş pullback seviyesi")
        print("   • Güvenli ratio % hesaplama")
        print("   • Z-Score istatistiksel ölçüm")
        print("   • use_z_score parametresi ile final_ratio seçimi")
        print("   • Bullish/Bearish trend tespiti")
        print("="*70)
        print("📡 C-SIGNAL + TERS MOMENTUM:")
        print("   • C-Signal = RSI(log(close), 14) değişimi")
        print("   • Ters Momentum = Supertrend trend vs C-Signal çelişkisi")
        print("   • MANUEL TERS MOMENTUM = Trend değiştirme = Otomatik ters momentum")
        print("   • Telegram bildirimleri otomatik")
        print("   • Manuel trend değiştirme ile ters momentum yeniden hesaplama")
        print("   🔒 Manuel değişiklikler KALICI olarak korunur")
        print("="*70)
        print("🎯 KALICI LİSTE KRİTERİ:")
        print("   • Supertrend Ratio >= %100 = Otomatik kalıcı liste")
        print("   • Manuel ekleme ve çıkarma imkanı")
        print("   • C-Signal + Ters momentum otomatik güncelleme")
        print("="*70)
        print("🔧 GÜNCELLENDİ - ÖRNEK KODDAN ALINDI:")
        print("   • TIMEFRAME_LIMITS: 200 → 500 mum")
        print("   • ATR hesaplama: Manuel → TA-Lib + Manuel backup")
        print("   • Pullback hesaplama: Pine Script → Basitleştirilmiş")
        print("   • Ratio hesaplama: Güvenlik kontrolleri eklendi")
        print("   • Final ratio: use_z_score parametresi ile belirleniyor")
        print("   • C-Signal momentum: Örnek koddaki mantık")
        print("="*70)
        print("🌐 Panel erişim: http://127.0.0.1:5001")
        print("⚠️  Sadece analiz amaçlıdır, yatırım tavsiyesi değildir!")
        print("📋 TA-Lib kurulumu gerekli: pip install TA-Lib==0.4.28")
        print("="*70 + "\n")
        
        logger.info("ÖRNEK KODDAN ALINMIŞ Supertrend hesaplama ile sistem başlatılıyor")
        
        app = create_app()
        app.run(debug=True, port=5001, host='127.0.0.1')
        
    except KeyboardInterrupt:
        print("\n🛑 Kullanıcı tarafından durduruldu")
    except Exception as e:
        logger.error(f"❌ Sistem başlatma hatası: {e}")
        print(f"\n❌ Hata: {e}")
    finally:
        logger.info("👋 Supertrend Analiz Sistemi kapatıldı")
        print("\n👋 Sistem kapatıldı")

if __name__ == '__main__':
    main()