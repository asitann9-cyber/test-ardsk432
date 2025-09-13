#!/usr/bin/env python3
"""
Ardışık Mum + C-Signal + Ters Momentum Analiz Sistemi
Ana Flask Uygulaması - Clean Architecture Implementation
"""

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
    logger = logging.getLogger("ConsecutiveCandlesSystem")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler("consecutive_candles_system.log")
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
    def analyze_consecutive():
        """Ardışık mum analizi gerçekleştir"""
        try:
            data = request.get_json()
            timeframe = data.get('timeframe', '4h')
            
            selected_symbols = memory_storage.get_selected_symbols()
            if not selected_symbols:
                return jsonify({"success": False, "error": "Hiç sembol seçilmedi"})
            
            logger.info(f"{len(selected_symbols)} sembol için {timeframe} ardışık mum analizi başlatılıyor...")
            
            # Paralel analiz
            results = analysis_service.analyze_multiple_symbols(selected_symbols, timeframe)
            
            if results:
                current_time = datetime.now().strftime('%H:%M')
                
                # 5+ ardışık VEYA %10+ değişim olan emtiaları kalıcı listeye ekle
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
                
                # Sonuçları formatla
                formatted_results = []
                
                for i, result in enumerate(results, 1):
                    tradingview_link = create_tradingview_link(result['symbol'], timeframe)
                    
                    formatted_results.append({
                        'rank': i,
                        'symbol': result['symbol'],
                        'tradingview_link': tradingview_link,
                        'current_price': round(result['current_price'], 4),
                        'consecutive_type': result['consecutive_type'],
                        'consecutive_count': result['consecutive_count'],
                        'percentage_change': result['percentage_change'],
                        'last_update': current_time
                    })
                
                return jsonify({
                    "success": True,
                    "results": formatted_results,
                    "count": len(formatted_results),
                    "timeframe": timeframe,
                    "message": f"{len(formatted_results)} sembol analiz edildi - Türkiye saati + Telegram bildirimleri"
                })
            else:
                return jsonify({"success": False, "error": "Analiz sonucu bulunamadı"})
                
        except Exception as e:
            logger.error(f"Ardışık analiz API hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/consecutive/permanent-high-consecutive', methods=['GET'])
    def get_permanent_high_consecutive():
        """Kalıcı 5+ ardışık VEYA %10+ değişim emtialar listesi - C-Signal + Ters Momentum ile"""
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
                    'max_consecutive_count': symbol_data.get('max_consecutive_count', 0),
                    'max_consecutive_type': symbol_data.get('max_consecutive_type', 'None'),
                    'max_percentage_change': symbol_data.get('max_percentage_change', 0),
                    'max_abs_percentage_change': symbol_data.get('max_abs_percentage_change', 0),
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
                    # 🆕 MANUEL TÜR BİLGİLERİ
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
                "message": f"Kalıcı listede {len(formatted_permanent)} emtia (5+ ardışık VEYA %10+ değişim + C-Signal + Ters Momentum: {reverse_momentum_count}) + Telegram bildirimleri"
            })
            
        except Exception as e:
            logger.error(f"Kalıcı liste API hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/consecutive/clear-permanent', methods=['POST'])
    def clear_permanent_high_consecutive():
        """Kalıcı 5+ ardışık VEYA %10+ değişim listesini temizle"""
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
    # 🆕 YENİ: MANUEL EMTİA EKLEME ENDPOINT'İ
    # =====================================================
    @app.route('/api/consecutive/add-to-permanent', methods=['POST'])
    def add_symbol_to_permanent():
        """Manuel olarak emtia kalıcı listeye ekleme"""
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
            
            # Güncel veriyi al
            current_analysis = analysis_service.analyze_single_symbol(symbol, timeframe)
            
            if not current_analysis:
                return jsonify({
                    "success": False, 
                    "error": f"{symbol} için veri alınamadı veya analiz yapılamadı"
                })
            
            # Manuel ekleme için sembol verisi oluştur
            manual_symbol_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'consecutive_type': current_analysis.get('consecutive_type', 'None'),
                'consecutive_count': current_analysis.get('consecutive_count', 0),
                'percentage_change': current_analysis.get('percentage_change', 0),
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
            
            logger.info(f"✅ {symbol} manuel olarak kalıcı listeye eklendi")
            
            return jsonify({
                "success": True,
                "message": f"✅ {symbol} kalıcı listeye eklendi! (Manuel ekleme + C-Signal + Ters momentum taraması)",
                "symbol_data": {
                    'symbol': symbol,
                    'consecutive_count': manual_symbol_data['consecutive_count'],
                    'consecutive_type': manual_symbol_data['consecutive_type'],
                    'percentage_change': manual_symbol_data['percentage_change'],
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
    # 🆕 YENİ: MANUEL TÜR DEĞİŞTİRME ENDPOINT'İ - KALICI KORUMA + OTOMATIK TERS MOMENTUM
    # =====================================================
    @app.route('/api/consecutive/update-symbol-type', methods=['POST'])
    def update_symbol_type():
        """Kalıcı listedeki sembolün türünü manuel güncelle ve KALICI OLARAK KORUMA ALTINA AL + OTOMATIK TERS MOMENTUM"""
        try:
            data = request.get_json()
            symbol = data.get('symbol')
            new_type = data.get('new_type')
            
            if not symbol or new_type not in ['Long', 'Short']:
                return jsonify({
                    "success": False, 
                    "error": "Geçersiz sembol veya tür bilgisi (Long/Short olmalı)"
                })
            
            # Kalıcı sembolü bul ve güncelle
            permanent_symbol = memory_storage.get_permanent_symbol(symbol)
            if not permanent_symbol:
                return jsonify({
                    "success": False, 
                    "error": f"{symbol} kalıcı listede bulunamadı"
                })
            
            # Eski türü kaydet
            old_type = permanent_symbol.get('max_consecutive_type', 'None')
            
            # 🔒 MANUEL TÜR DEĞİŞTİRME VE KİLİTLEME
            success = memory_storage.set_manual_type_override(symbol, new_type)
            
            if not success:
                return jsonify({
                    "success": False,
                    "error": f"{symbol} manuel tür değişikliği yapılamadı"
                })
            
            # 🆕 MANUEL DEĞİŞİKLİK = OTOMATIK TERS MOMENTUM TETİKLEME
            reverse_momentum_data = {
                'has_reverse_momentum': True,
                'reverse_type': 'MANUEL',
                'signal_strength': 'Strong',
                'signal_value': None,
                'alert_message': f'MANUEL TERS MOMENTUM: {old_type} → {new_type} yön değişikliği yapıldı!'
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
            
            # C-Signal ve ters momentum yeniden hesapla (YENİ TÜRLE)
            timeframe = permanent_symbol.get('timeframe', '4h')
            
            # Güncellenmiş sembol verisini al
            updated_permanent_symbol = memory_storage.get_permanent_symbol(symbol)
            if updated_permanent_symbol:
                # C-Signal hesapla ama türü koruyarak
                updated_symbol = analysis_service.update_symbol_with_c_signal(
                    updated_permanent_symbol, 
                    timeframe, 
                    preserve_manual_type=True
                )
                
                # Ek ters momentum varsa (C-Signal bazlı) ek bildirim gönder
                reverse_momentum = updated_symbol.get('reverse_momentum', {})
                if reverse_momentum.get('has_reverse_momentum', False) and reverse_momentum.get('reverse_type') != 'MANUEL':
                    if telegram_service.should_send_alert(updated_symbol, reverse_momentum):
                        telegram_service.send_reverse_momentum_alert(updated_symbol)
            
            logger.info(f"🔒 {symbol} türü manuel olarak {old_type} -> {new_type} DEĞİŞTİRİLDİ ve KİLİTLENDİ + OTOMATIK TERS MOMENTUM")
            
            return jsonify({
                "success": True,
                "message": f"🔒 {symbol} türü {old_type} → {new_type} değiştirildi ve KİLİTLENDİ! OTOMATIK TERS MOMENTUM tetiklendi!",
                "symbol": symbol,
                "old_type": old_type,
                "new_type": new_type,
                "is_locked": True,
                "manual_reverse_momentum": True
            })
            
        except Exception as e:
            logger.error(f"Tür güncelleme hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # 🆕 YENİ: MANUEL TÜR KİLİDİNİ KALDIRMA ENDPOINT'İ
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
    🔒 KALICI KORUMA - Kalıcı listedeki emtialar için C-Signal güncelle 
    AMA MANUEL DEĞİŞTİRİLEN TÜRLERİ KORUMA ALTINDA TUT
    """
    current_time = datetime.now().strftime('%H:%M')
    permanent_symbols = memory_storage.get_permanent_symbols()
    
    for symbol_data in permanent_symbols:
        try:
            symbol = symbol_data['symbol']
            timeframe = symbol_data.get('timeframe', '4h')
            is_manual_override = symbol_data.get('manual_type_override', False)
            
            # MANUEL DEĞİŞTİRİLEN TÜRÜ KORUMA
            if is_manual_override:
                logger.debug(f"🔒 {symbol} manuel tür koruması aktif: {symbol_data.get('manual_type_value', 'Unknown')}")
                
                # Sadece C-Signal hesapla, türü değiştirme
                updated_symbol = analysis_service.update_symbol_with_c_signal(
                    symbol_data, 
                    timeframe, 
                    preserve_manual_type=True
                )
            else:
                # Normal güncelleme (tür değişebilir)
                updated_symbol = analysis_service.update_symbol_with_c_signal(
                    symbol_data, 
                    timeframe, 
                    preserve_manual_type=False
                )
            
            # Ters momentum varsa Telegram bildirimi gönder
            reverse_momentum = updated_symbol.get('reverse_momentum', {})
            if reverse_momentum.get('has_reverse_momentum', False):
                # Spam önleme kontrolü
                if telegram_service.should_send_alert(updated_symbol, reverse_momentum):
                    success = telegram_service.send_reverse_momentum_alert(updated_symbol)
                    if success:
                        updated_symbol['last_telegram_alert'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        memory_storage.update_permanent_symbol(symbol, {'last_telegram_alert': updated_symbol['last_telegram_alert']})
                        
        except Exception as e:
            logger.debug(f"C-Signal güncelleme hatası {symbol_data['symbol']}: {e}")

def main():
    """Ana uygulama başlatma fonksiyonu"""
    try:
        print("\n" + "="*70)
        print("🔗 Ardışık Mum Yapıları Analiz Sistemi - Binance Vadeli İşlemler")
        print("="*70)
        print("📊 ÖZELLİKLER:")
        print("   ✅ Clean Architecture Implementation")
        print("   ✅ Binance USDT Vadeli İşlem Emtia Yönetimi")
        print("   ✅ Ardışık Long/Short Mum Analizi")
        print("   ✅ Yüzdesel Değişim Hesaplama")
        print("   ✅ Paralel Analiz (5 worker)")
        print("   ✅ Bellekte Veri Tutma (CSV yok)")
        print("   ✅ TradingView Grafik Entegrasyonu")
        print("   ✅ C-Signal + Ters Momentum Tespiti")
        print("   ✅ Telegram Bildirimleri")
        print("   ✅ Filtreleme (Hepsi / Long / Short / 5+ Ardışık)")
        print("   🆕 Manuel Kalıcı Liste Ekleme")
        print("   🆕 Manuel Tür Değiştirme (Long ↔ Short)")
        print("   🔒 KALICI Tür Koruma - Otomatik güncellemelerde korunur")
        print("   🚨 MANUEL TERS MOMENTUM - Yön değiştirince otomatik tetikleme")
        print("="*70)
        print("🔗 ARDIŞIK MUM ANALİZİ:")
        print("   • Ardışık Long Mum Sayısı")
        print("   • Ardışık Short Mum Sayısı")
        print("   • (Son High - İlk Low) / İlk Low * 100 (Long)")
        print("   • (İlk High - Son Low) / İlk High * 100 (Short)")
        print("="*70)
        print("📡 C-SIGNAL + TERS MOMENTUM:")
        print("   • C-Signal = RSI(log(close), 14) değişimi")
        print("   • Ters Momentum = Ardışık vs C-Signal çelişkisi")
        print("   • MANUEL TERS MOMENTUM = Yön değiştirme = Otomatik ters momentum")
        print("   • Telegram bildirimleri otomatik")
        print("   • Manuel tür değiştirme ile ters momentum yeniden hesaplama")
        print("   🔒 Manuel değişiklikler KALICI olarak korunur")
        print("="*70)
        print("🌐 Panel erişim: http://127.0.0.1:5001")
        print("⚠️  Sadece analiz amaçlıdır, yatırım tavsiyesi değildir!")
        print("="*70 + "\n")
        
        logger.info("Clean Architecture ile sistem bellekte veri tutacak, CSV kullanılmayacak")
        
        app = create_app()
        app.run(debug=True, port=5001, host='127.0.0.1')
        
    except KeyboardInterrupt:
        print("\n🛑 Kullanıcı tarafından durduruldu")
    except Exception as e:
        logger.error(f"❌ Sistem başlatma hatası: {e}")
        print(f"\n❌ Hata: {e}")
    finally:
        logger.info("👋 Ardışık Mum Analiz Sistemi kapatıldı")
        print("\n👋 Sistem kapatıldı")

if __name__ == '__main__':
    main()