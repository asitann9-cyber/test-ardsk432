#!/usr/bin/env python3
"""
Supertrend + C-Signal + VPMV Analiz Sistemi
Ana Flask Uygulaması - Clean Architecture Implementation
🆕 YENİ: VPMV (Volume-Price-Momentum-Volatility) NET POWER Sistemi
🆕 YENİ: Dinamik C-Signal Threshold - Panel'den Ayarlanabilir ±X L/S Sinyal Tespiti
🔥 YENİ: TETİKLEYİCİ ENDPOINT'LERİ - Aktif tetikleyici filtreleme ve istatistikleri
✅ FIX: Kalıcı tabloda güncel ratio gösterimi
📱 TELEGRAM: C-Signal Alert Bildirimleri Aktif!
🔍 DEBUG: Detaylı Telegram gönderim logları eklendi
🔥 FIX: WebSocket Dinamik Restart - Kalıcı listeye eklenen semboller anlık izleniyor!
"""
from flask_sock import Sock
import pandas as pd
import logging
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# WebSocket
from services.websocket_manager import WebSocketManager

# Memory
from utils.memory_storage import MemoryStorage
from utils.helpers import create_tradingview_link
memory_storage = MemoryStorage()

# Config & Services
from config import Config
from services.analysis_service import AnalysisService
from services.binance_service import BinanceService
from services.telegram_service import TelegramService

# 🔥 GLOBAL WebSocket Manager Instance
ws_manager = None

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
    app = Flask(__name__)
    app.config.from_object(Config)

    sock = Sock(app)

    
    binance_service = BinanceService()
    telegram_service = TelegramService()
    analysis_service = AnalysisService()

    register_routes(
        app,
        sock,
        analysis_service,
        memory_storage,
        binance_service,
        telegram_service
    )

    logger.info("Flask uygulaması başarıyla oluşturuldu")
    return app



def register_routes(app, sock, analysis_service, memory_storage, binance_service, telegram_service):

    """Tüm route'ları kaydet"""

    # =====================================================
    # 🟦 REAL-TIME WebSocket Endpoint
    # =====================================================
    @sock.route('/ws')
    def ws_route(ws):
        """UI WebSocket bağlantısı"""
        try:
            WebSocketManager.add_client(ws)
            logger.info("🔌 UI WebSocket client bağlandı")

            while True:
                message = ws.receive()  # UI'den mesaj bekler ama gerekmez
                if message is None:
                    break

        except Exception as e:
            logger.error(f"❌ WebSocket bağlantı hatası: {e}")
        finally:
            WebSocketManager.remove_client(ws)
            logger.info("❌ UI WebSocket client ayrıldı")

    
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
        """Supertrend + VPMV analizi gerçekleştir"""
        global ws_manager
        
        try:
            data = request.get_json()
            timeframe = data.get('timeframe', '4h')
            
            selected_symbols = memory_storage.get_selected_symbols()
            if not selected_symbols:
                return jsonify({"success": False, "error": "Hiç sembol seçilmedi"})
            
            # Mevcut threshold'ları logla
            ratio_threshold = AnalysisService.MIN_RATIO_THRESHOLD
            c_signal_threshold = Config.C_SIGNAL_ALERT_THRESHOLD
            logger.info(f"🎯 {len(selected_symbols)} sembol için {timeframe} Supertrend+VPMV analizi başlatılıyor...")
            logger.info(f"   📊 Ratio Threshold: {ratio_threshold}% | 🔔 C-Signal Threshold: ±{c_signal_threshold}")
            
            # Paralel analiz (VPMV dahil)
            results = analysis_service.analyze_multiple_symbols(selected_symbols, timeframe)
            
            if results:
                current_time = datetime.now().strftime('%H:%M')
                
                # Ratio >= threshold olan emtiaları kalıcı listeye ekle (VPMV dahil)
                high_priority_count = 0
                for result in results:
                    ratio = abs(result.get('ratio_percent', 0))
                    
                    if ratio >= ratio_threshold:
                        result['tradingview_link'] = create_tradingview_link(result['symbol'], timeframe)
                        memory_storage.add_permanent_symbol(result)
                        high_priority_count += 1
                        logger.info(f"✅ {result['symbol']} kalıcı listeye eklendi (Ratio: {ratio}% >= {ratio_threshold}%, VPMV: {result.get('vpmv_net_power', 0)})")
                
                logger.info(f"📊 {high_priority_count} sembol threshold'u geçti ve kalıcı listeye eklendi")
                
                # 🔥 WebSocket Restart - Yeni semboller eklendiğinde
                if high_priority_count > 0 and ws_manager is not None:
                    new_symbols = [s["symbol"] for s in memory_storage.get_permanent_symbols()]
                    ws_manager.symbols = new_symbols
                    ws_manager.restart()
                    logger.info(f"🔌 WebSocket restart edildi → {len(new_symbols)} sembol izleniyor")
                
                # 🆕 Kalıcı listeki analizi güncelle ve C-Signal kontrol et (DİNAMİK THRESHOLD)
                permanent_symbols = memory_storage.get_permanent_symbols()
                c_signal_alerts = []
                
                for symbol_data in permanent_symbols:
                    # Analizi güncelle (MANUEL TÜR KORUNARAK + VPMV)
                    updated_symbol = analysis_service.update_symbol_with_analysis(
                        symbol_data, 
                        symbol_data.get('timeframe', '4h'),
                        preserve_manual_type=symbol_data.get('manual_type_override', False)
                    )
                    
                    # 🆕 C-Signal güncelle ve DİNAMİK THRESHOLD kontrolü yap
                    c_signal_value = updated_symbol.get('c_signal')
                    if c_signal_value is not None:
                        signal_result = memory_storage.update_c_signal(
                            updated_symbol['symbol'], 
                            c_signal_value
                        )
                        
                        # Sinyal tetiklendiyse listeye ekle
                        if signal_result['signal_triggered']:
                            c_signal_alerts.append({
                                'symbol': updated_symbol['symbol'],
                                'signal_type': signal_result['signal_type'],
                                'c_signal_value': c_signal_value,
                                'tradingview_link': updated_symbol.get('tradingview_link', '#')
                            })
                
                # 🆕 C-Signal alertleri logla VE TELEGRAM'A GÖNDER (DEBUG LOGLU)
                if c_signal_alerts:
                    logger.info(f"🔔 {len(c_signal_alerts)} yeni C-Signal alert bulundu! (Threshold: ±{c_signal_threshold})")
                    for alert in c_signal_alerts:
                        logger.info(f"   📍 {alert['symbol']}: {alert['signal_type']} - C={alert['c_signal_value']:.2f}")
                        
                        # 🔍 DEBUG: Telegram gönderim süreci başlıyor
                        logger.info(f"🔍 DEBUG: {alert['symbol']} için Telegram kontrolü başladı")
                        
                        # 📱 TELEGRAM BİLDİRİMİ GÖNDER
                        permanent_symbol = memory_storage.get_permanent_symbol(alert['symbol'])
                        
                        if permanent_symbol:
                            logger.info(f"🔍 DEBUG: permanent_symbol bulundu: {permanent_symbol.get('symbol')}")
                            logger.info(f"🔍 DEBUG: last_c_signal_alert_time: {permanent_symbol.get('last_c_signal_alert_time')}")
                            
                            spam_check = telegram_service.should_send_c_signal_alert(permanent_symbol)
                            logger.info(f"🔍 DEBUG: Spam kontrolü sonucu: {spam_check}")
                            
                            if spam_check:
                                logger.info(f"📱 Telegram API çağrısı yapılıyor: {alert['symbol']}")
                                
                                success = telegram_service.send_c_signal_alert(
                                    symbol=alert['symbol'],
                                    signal_type=alert['signal_type'],
                                    c_signal_value=alert['c_signal_value'],
                                    tradingview_link=alert['tradingview_link'],
                                    threshold=c_signal_threshold
                                )
                                
                                logger.info(f"🔍 DEBUG: Telegram API yanıtı: {success}")
                                
                                if success:
                                    memory_storage.update_permanent_symbol(alert['symbol'], {
                                        'last_c_signal_alert_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    })
                                    memory_storage.increment_c_signal_alerts()
                                    logger.info(f"✅ {alert['symbol']} için Telegram C-Signal bildirimi gönderildi")
                                else:
                                    logger.error(f"❌ {alert['symbol']} Telegram gönderimi BAŞARISIZ!")
                            else:
                                logger.warning(f"⏳ {alert['symbol']} spam kontrolünden GEÇEMEDİ - Son alert: {permanent_symbol.get('last_c_signal_alert_time')}")
                        else:
                            logger.error(f"❌ {alert['symbol']} kalıcı listede BULUNAMADI!")
                
                # Sonuçları formatla (VPMV dahil)
                formatted_results = []
                
                for i, result in enumerate(results, 1):
                    tradingview_link = create_tradingview_link(result['symbol'], timeframe)
                    
                    # C-Signal momentum formatla
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
                        'final_ratio': result.get('final_ratio', result['ratio_percent']),
                        'c_signal_display': c_signal_display,
                        'trend_direction': result.get('trend_direction', 'None'),
                        'price_vs_supertrend': result.get('price_vs_supertrend', 'None'),
                        # 🆕 VPMV verileri
                        'vpmv_net_power': result.get('vpmv_net_power', 0),
                        'vpmv_signal': result.get('vpmv_signal', 'NEUTRAL'),
                        'last_update': current_time
                    })


                try:
                    WebSocketManager.broadcast({
                        "event": "permanent_update",
                        "data": memory_storage.get_permanent_symbols()
                    })
                    logger.info("📡 WebSocket → permanent_update broadcast edildi")
                except Exception as ws_err:
                    logger.error(f"❌ WebSocket broadcast hatası: {ws_err}")
                
                return jsonify({
                    "success": True,
                    "results": formatted_results,
                    "count": len(formatted_results),
                    "timeframe": timeframe,
                    "high_priority_count": high_priority_count,
                    "current_ratio_threshold": ratio_threshold,
                    "current_c_signal_threshold": c_signal_threshold,
                    "c_signal_alerts": c_signal_alerts,
                    "c_signal_alert_count": len(c_signal_alerts),
                    "message": f"{len(formatted_results)} sembol analiz edildi (VPMV dahil) - {high_priority_count} sembol kalıcı listeye eklendi - {len(c_signal_alerts)} C-Signal alert (±{c_signal_threshold})"
                })
            else:
                return jsonify({"success": False, "error": "Analiz sonucu bulunamadı"})
                
        except Exception as e:
            logger.error(f"Supertrend+VPMV analiz API hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/consecutive/permanent-high-consecutive', methods=['GET'])
    def get_permanent_high_ratio():
        """✅ Kalıcı %100+ ratio emtialar listesi - GÜNCEL RATIO + VPMV + TETİKLEYİCİ GÖSTERİMİ"""
        try:
            permanent_symbols = memory_storage.get_permanent_symbols()
            
            # Her kalıcı sembol için ek bilgiler ekle (VPMV + TETİKLEYİCİ dahil)
            formatted_permanent = []
            for i, symbol_data in enumerate(permanent_symbols, 1):
                # C-Signal durumunu al
                c_signal_status = memory_storage.get_c_signal_status(symbol_data['symbol'])
                
                formatted_permanent.append({
                    'rank': i,
                    'symbol': symbol_data['symbol'],
                    'tradingview_link': symbol_data.get('tradingview_link', '#'),
                    'first_date': symbol_data.get('first_high_ratio_date', 'Bilinmiyor'),
                    
                    # ✅ YENİ İSİMLER: Artık doğrudan ratio_percent ve supertrend_type kullanıyoruz
                    'ratio_percent': symbol_data.get('ratio_percent', 0),
                    'supertrend_type': symbol_data.get('supertrend_type', 'None'),
                    'z_score': symbol_data.get('z_score', 0),
                    
                    'timeframe': symbol_data.get('timeframe', '4h'),
                    'c_signal': symbol_data.get('c_signal', 'N/A'),
                    'c_signal_update_time': symbol_data.get('c_signal_update_time', 'N/A'),
                    'add_reason': symbol_data.get('add_reason', 'Bilinmiyor'),
                    'last_telegram_alert': symbol_data.get('last_telegram_alert', 'Hiç gönderilmedi'),
                    
                    # MANUEL TÜR BİLGİLERİ
                    'manual_type_override': symbol_data.get('manual_type_override', False),
                    'manual_override_date': symbol_data.get('manual_override_date', None),
                    
                    # C-SIGNAL DURUMU
                    'c_signal_status': c_signal_status['signal_type'],
                    'has_c_signal_alert': c_signal_status['has_signal'],
                    'last_c_signal_alert_time': c_signal_status['last_alert_time'],
                    
                    # 🆕 VPMV VERİLERİ
                    'vpmv_net_power': symbol_data.get('vpmv_net_power', 0),
                    'vpmv_signal': symbol_data.get('vpmv_signal', 'NEUTRAL'),
                    'vpmv_update_time': symbol_data.get('vpmv_update_time', 'N/A'),
                    
                    # 🔥 YENİ: TETİKLEYİCİ BİLGİLERİ
                    'vpmv_trigger_name': symbol_data.get('vpmv_trigger_name', 'Yok'),
                    'vpmv_trigger_active': symbol_data.get('vpmv_trigger_active', False)
                })
            
            telegram_bot_status = "✅" if telegram_service.bot_token else "❌"
            telegram_chat_status = "✅" if telegram_service.chat_id else "❌"
            
            # Aktif C-Signal sayısı
            active_c_signal_count = sum(1 for s in formatted_permanent if s['has_c_signal_alert'])
            
            # 🆕 VPMV istatistikleri
            vpmv_stats = memory_storage.get_vpmv_statistics()
            
            # 🔥 YENİ: Aktif tetikleyici sayısı
            active_trigger_count = sum(1 for s in formatted_permanent if s['vpmv_trigger_active'])
            
            return jsonify({
                "success": True,
                "permanent_symbols": formatted_permanent,
                "count": len(formatted_permanent),
                "active_c_signal_count": active_c_signal_count,
                "active_trigger_count": active_trigger_count,
                "current_c_signal_threshold": Config.C_SIGNAL_ALERT_THRESHOLD,
                "telegram_status": f"Bot Token: {telegram_bot_status} | Chat ID: {telegram_chat_status}",
                # 🆕 VPMV istatistikleri
                "vpmv_statistics": vpmv_stats,
                "message": f"Kalıcı listede {len(formatted_permanent)} emtia ({active_c_signal_count} aktif C-Signal ±{Config.C_SIGNAL_ALERT_THRESHOLD}, {active_trigger_count} aktif tetikleyici)"
            })
            
        except Exception as e:
            logger.error(f"Kalıcı liste API hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/consecutive/clear-permanent', methods=['POST'])
    def clear_permanent_high_ratio():
        """Kalıcı %100+ ratio listesini temizle"""
        global ws_manager
        
        try:
            old_count = memory_storage.clear_permanent_symbols()
            
            # 🔥 WebSocket'i durdur - Liste boşaldı
            if ws_manager is not None:
                ws_manager.stop()
                logger.info("🔌 WebSocket durduruldu - Kalıcı liste boş")
            
            return jsonify({
                "success": True,
                "message": f"Kalıcı liste temizlendi. {old_count} emtia silindi.",
                "count": 0
            })
            
        except Exception as e:
            logger.error(f"Kalıcı liste temizleme hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/consecutive/add-to-permanent', methods=['POST'])
    def add_symbol_to_permanent():
        """Manuel olarak emtia kalıcı listeye ekleme (VPMV + TETİKLEYİCİ dahil)"""
        global ws_manager
        
        try:
            data = request.get_json()
            symbol = data.get('symbol')
            timeframe = data.get('timeframe', '4h')
            
            if not symbol:
                return jsonify({"success": False, "error": "Sembol adı gerekli"})
            
            existing_permanent = memory_storage.get_permanent_symbol(symbol)
            if existing_permanent:
                return jsonify({
                    "success": False, 
                    "error": f"{symbol} zaten kalıcı listede mevcut"
                })
            
            # TAM analiz (VPMV + TETİKLEYİCİ dahil)
            current_analysis = analysis_service.analyze_single_symbol(symbol, timeframe)
            
            if not current_analysis:
                return jsonify({
                    "success": False, 
                    "error": f"{symbol} için veri alınamadı veya analiz yapılamadı"
                })
            
            manual_symbol_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'trend_direction': current_analysis.get('trend_direction', 'None'),
                'ratio_percent': current_analysis.get('ratio_percent', 0),
                'z_score': current_analysis.get('z_score', 0),
                'current_price': current_analysis.get('current_price', 0),
                'tradingview_link': create_tradingview_link(symbol, timeframe),
                # 🆕 VPMV verileri
                'vpmv_net_power': current_analysis.get('vpmv_net_power', 0),
                'vpmv_signal': current_analysis.get('vpmv_signal', 'NEUTRAL'),
                # 🔥 YENİ: Tetikleyici verileri
                'vpmv_trigger_name': current_analysis.get('vpmv_trigger_name', 'Yok'),
                'vpmv_trigger_active': current_analysis.get('vpmv_trigger_active', False),
                'last_update': current_analysis.get('last_update', datetime.now())
            }
            
            memory_storage.add_permanent_symbol(manual_symbol_data)
            
            # 🔥 WebSocket Restart - Yeni sembol eklendi
            if ws_manager is not None:
                new_symbols = [s["symbol"] for s in memory_storage.get_permanent_symbols()]
                ws_manager.symbols = new_symbols
                ws_manager.restart()
                logger.info(f"🔌 WebSocket restart edildi → {len(new_symbols)} sembol izleniyor")
            
            logger.info(f"✅ {symbol} manuel olarak kalıcı listeye eklendi (VPMV: {manual_symbol_data['vpmv_net_power']}, Tetikleyici: {manual_symbol_data['vpmv_trigger_name']})")
            
            return jsonify({
                "success": True,
                "message": f"✅ {symbol} kalıcı listeye eklendi!",
                "symbol_data": {
                    'symbol': symbol,
                    'ratio_percent': manual_symbol_data['ratio_percent'],
                    'trend_direction': manual_symbol_data['trend_direction'],
                    'z_score': manual_symbol_data['z_score'],
                    'vpmv_net_power': manual_symbol_data['vpmv_net_power'],
                    'vpmv_signal': manual_symbol_data['vpmv_signal'],
                    'vpmv_trigger_name': manual_symbol_data['vpmv_trigger_name'],
                    'vpmv_trigger_active': manual_symbol_data['vpmv_trigger_active'],
                    'timeframe': timeframe
                }
            })
                
        except Exception as e:
            logger.error(f"Manuel emtia ekleme hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/consecutive/remove-from-permanent', methods=['POST'])
    def remove_symbol_from_permanent():
        """Kalıcı listeden emtia çıkarma"""
        global ws_manager
        
        try:
            data = request.get_json()
            symbol = data.get('symbol')
            
            if not symbol:
                return jsonify({"success": False, "error": "Sembol adı gerekli"})
            
            existing_permanent = memory_storage.get_permanent_symbol(symbol)
            if not existing_permanent:
                return jsonify({
                    "success": False, 
                    "error": f"{symbol} kalıcı listede bulunamadı"
                })
            
            success = memory_storage.remove_permanent_symbol(symbol)
            
            if success:
                # 🔥 WebSocket Restart - Sembol çıkarıldı
                if ws_manager is not None:
                    new_symbols = [s["symbol"] for s in memory_storage.get_permanent_symbols()]
                    if new_symbols:
                        ws_manager.symbols = new_symbols
                        ws_manager.restart()
                        logger.info(f"🔌 WebSocket restart edildi → {len(new_symbols)} sembol izleniyor")
                    else:
                        ws_manager.stop()
                        logger.info("🔌 WebSocket durduruldu - Kalıcı liste boş")
                
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

    @app.route('/api/consecutive/update-symbol-type', methods=['POST'])
    def update_symbol_type():
        """Kalıcı listedeki sembolün trend türünü manuel güncelle"""
        try:
            data = request.get_json()
            symbol = data.get('symbol')
            new_type = data.get('new_type')
            
            if not symbol or new_type not in ['Bullish', 'Bearish']:
                return jsonify({
                    "success": False, 
                    "error": "Geçersiz sembol veya tür bilgisi (Bullish/Bearish olmalı)"
                })
            
            permanent_symbol = memory_storage.get_permanent_symbol(symbol)
            if not permanent_symbol:
                return jsonify({
                    "success": False, 
                    "error": f"{symbol} kalıcı listede bulunamadı"
                })
            
            old_type = permanent_symbol.get('supertrend_type', 'None')
            
            success = memory_storage.set_manual_type_override(symbol, new_type)
            
            if not success:
                return jsonify({
                    "success": False,
                    "error": f"{symbol} manuel tür değişikliği yapılamadı"
                })
            
            tradingview_link = permanent_symbol.get('tradingview_link', '#')
            if telegram_service.should_send_alert(permanent_symbol):
                telegram_service.send_manual_type_change_alert(symbol, old_type, new_type, tradingview_link)
                memory_storage.update_permanent_symbol(symbol, {
                    'last_telegram_alert': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            logger.info(f"🔒 {symbol} trend türü manuel olarak {old_type} -> {new_type} DEĞİŞTİRİLDİ ve KİLİTLENDİ")
            
            return jsonify({
                "success": True,
                "message": f"🔒 {symbol} trend türü {old_type} → {new_type} değiştirildi ve KİLİTLENDİ!",
                "symbol": symbol,
                "old_type": old_type,
                "new_type": new_type,
                "is_locked": True
            })
            
        except Exception as e:
            logger.error(f"Tür güncelleme hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/consecutive/unlock-symbol-type', methods=['POST'])
    def unlock_symbol_type():
        """Kalıcı listedeki sembolün manuel tür kilidini kaldır"""
        try:
            data = request.get_json()
            symbol = data.get('symbol')
            
            if not symbol:
                return jsonify({
                    "success": False, 
                    "error": "Sembol adı gerekli"
                })
            
            permanent_symbol = memory_storage.get_permanent_symbol(symbol)
            if not permanent_symbol:
                return jsonify({
                    "success": False, 
                    "error": f"{symbol} kalıcı listede bulunamadı"
                })
            
            if not permanent_symbol.get('manual_type_override', False):
                return jsonify({
                    "success": False, 
                    "error": f"{symbol} zaten manuel olarak kilitlenmemiş"
                })
            
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
    

    # =====================================================
    # RATIO THRESHOLD YÖNETIMI
    # =====================================================
    @app.route('/api/consecutive/update-threshold', methods=['POST'])
    def update_threshold():
        """Minimum ratio threshold'u güncelle"""
        try:
            data = request.get_json()
            min_ratio_threshold = data.get('min_ratio_threshold', 100.0)
            
            if not isinstance(min_ratio_threshold, (int, float)) or min_ratio_threshold < 0:
                return jsonify({
                    "success": False,
                    "error": "Geçersiz threshold değeri. 0 veya üzeri bir sayı olmalı."
                }), 400
            
            if min_ratio_threshold > 1000:
                return jsonify({
                    "success": False,
                    "error": "Threshold değeri çok yüksek (maksimum 1000%)"
                }), 400
            
            AnalysisService.MIN_RATIO_THRESHOLD = float(min_ratio_threshold)
            
            permanent_symbols = memory_storage.get_permanent_symbols()
            logger.info(f"⚙️ Ratio Threshold güncellendi: {min_ratio_threshold}%, {len(permanent_symbols)} sembol yeniden değerlendiriliyor...")
            
            symbols_to_keep = []
            symbols_removed = []
            
            for symbol_data in permanent_symbols:
                current_ratio = abs(symbol_data.get('ratio_percent', 0))
                if current_ratio >= min_ratio_threshold:
                    symbols_to_keep.append(symbol_data)
                else:
                    symbols_removed.append(symbol_data.get('symbol'))
            
            if symbols_removed:
                memory_storage.permanent_high_ratio = symbols_to_keep
                logger.info(f"🗑️ Yeni threshold nedeniyle {len(symbols_removed)} sembol kalıcı listeden çıkarıldı: {symbols_removed}")
            
            logger.info(f"⚙️ Minimum ratio threshold güncellendi: {min_ratio_threshold}%")
            
            return jsonify({
                "success": True,
                "message": f"✅ Minimum ratio threshold {min_ratio_threshold}% olarak güncellendi",
                "threshold": min_ratio_threshold,
                "symbols_kept": len(symbols_to_keep),
                "symbols_removed": len(symbols_removed),
                "removed_symbols": symbols_removed
            })
            
        except Exception as e:
            logger.error(f"Ratio threshold güncelleme hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/consecutive/get-threshold', methods=['GET'])
    def get_threshold():
        """Mevcut minimum ratio threshold'u getir"""
        try:
            current_threshold = getattr(analysis_service, 'MIN_RATIO_THRESHOLD', 100.0)
            
            return jsonify({
                "success": True,
                "threshold": current_threshold,
                "message": f"Mevcut threshold: {current_threshold}%"
            })
            
        except Exception as e:
            logger.error(f"Ratio threshold okuma hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # =====================================================
    # 🆕 C-SIGNAL THRESHOLD YÖNETIMI
    # =====================================================
    @app.route('/api/consecutive/update-c-signal-threshold', methods=['POST'])
    def update_c_signal_threshold():
        """C-Signal alert threshold'u güncelle"""
        try:
            data = request.get_json()
            c_signal_threshold = data.get('c_signal_threshold', 20.0)
            
            # Validasyon
            if not isinstance(c_signal_threshold, (int, float)) or c_signal_threshold < 0:
                return jsonify({
                    "success": False,
                    "error": "Geçersiz C-Signal threshold değeri. 0 veya üzeri bir sayı olmalı."
                }), 400
            
            if c_signal_threshold > 100:
                return jsonify({
                    "success": False,
                    "error": "C-Signal threshold değeri çok yüksek (maksimum 100)"
                }), 400
            
            # Config'de threshold'u güncelle
            success = Config.update_c_signal_threshold(c_signal_threshold)
            
            if not success:
                return jsonify({
                    "success": False,
                    "error": "C-Signal threshold güncellenemedi"
                }), 500
            
            # Mevcut permanent symbollerdeki C-Signal durumlarını yeniden değerlendir
            permanent_symbols = memory_storage.get_permanent_symbols()
            logger.info(f"⚙️ C-Signal Threshold güncellendi: ±{c_signal_threshold}, {len(permanent_symbols)} sembol yeniden değerlendiriliyor...")
            
            # Tüm sembollerdeki C-Signal alertlerini temizle ve yeniden kontrol et
            reactivated_count = 0
            cleared_count = 0
            
            for symbol_data in permanent_symbols:
                c_signal_value = symbol_data.get('c_signal')
                
                if c_signal_value is not None:
                    # Eski alert durumunu kontrol et
                    had_alert = symbol_data.get('last_c_signal_type') is not None
                    
                    # Alert'i temizle
                    memory_storage.clear_c_signal_alert(symbol_data['symbol'])
                    
                    # Yeni threshold ile yeniden kontrol et
                    signal_result = memory_storage.update_c_signal(
                        symbol_data['symbol'], 
                        c_signal_value
                    )
                    
                    # Alert durumu değiştiyse logla
                    if signal_result['signal_triggered'] and not had_alert:
                        reactivated_count += 1
                        logger.info(f"   🔔 {symbol_data['symbol']}: Yeni threshold ile alert aktif - {signal_result['signal_type']}")
                    elif had_alert and not signal_result['signal_triggered']:
                        cleared_count += 1
                        logger.info(f"   🧹 {symbol_data['symbol']}: Yeni threshold ile alert temizlendi")
            
            logger.info(f"⚙️ C-Signal threshold güncellendi: ±{c_signal_threshold}")
            logger.info(f"   🔔 {reactivated_count} yeni alert aktif | 🧹 {cleared_count} alert temizlendi")
            
            return jsonify({
                "success": True,
                "message": f"✅ C-Signal alert threshold ±{c_signal_threshold} olarak güncellendi",
                "threshold": c_signal_threshold,
                "long_threshold": Config.get_c_signal_long_threshold(),
                "short_threshold": Config.get_c_signal_short_threshold(),
                "symbols_reactivated": reactivated_count,
                "symbols_cleared": cleared_count,
                "total_symbols_checked": len(permanent_symbols)
            })
            
        except Exception as e:
            logger.error(f"C-Signal threshold güncelleme hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/consecutive/get-c-signal-threshold', methods=['GET'])
    def get_c_signal_threshold():
        """Mevcut C-Signal threshold'u getir"""
        try:
            c_signal_config = Config.get_c_signal_config()
            
            return jsonify({
                "success": True,
                "threshold": c_signal_config['alert_threshold'],
                "long_threshold": c_signal_config['long_threshold'],
                "short_threshold": c_signal_config['short_threshold'],
                "description": c_signal_config['description'],
                "message": f"Mevcut C-Signal threshold: ±{c_signal_config['alert_threshold']}"
            })
            
        except Exception as e:
            logger.error(f"C-Signal threshold okuma hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/consecutive/get-all-thresholds', methods=['GET'])
    def get_all_thresholds():
        """Tüm threshold ayarlarını getir"""
        try:
            ratio_threshold = getattr(analysis_service, 'MIN_RATIO_THRESHOLD', 100.0)
            c_signal_config = Config.get_c_signal_config()
            
            return jsonify({
                "success": True,
                "thresholds": {
                    "ratio": {
                        "value": ratio_threshold,
                        "description": f"Supertrend Ratio >= {ratio_threshold}%"
                    },
                    "c_signal": {
                        "value": c_signal_config['alert_threshold'],
                        "long_threshold": c_signal_config['long_threshold'],
                        "short_threshold": c_signal_config['short_threshold'],
                        "description": c_signal_config['description']
                    }
                },
                "message": f"Ratio: {ratio_threshold}% | C-Signal: ±{c_signal_config['alert_threshold']}"
            })
            
        except Exception as e:
            logger.error(f"Threshold okuma hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # =====================================================
    # 🆕 VPMV SPESİFİK ENDPOINT'LER
    # =====================================================
    
    @app.route('/api/vpmv/statistics', methods=['GET'])
    def get_vpmv_statistics():
        """VPMV özet istatistiklerini getir"""
        try:
            vpmv_stats = memory_storage.get_vpmv_statistics()
            
            return jsonify({
                "success": True,
                "statistics": vpmv_stats,
                "message": f"VPMV istatistikleri: {vpmv_stats['total_symbols']} sembol"
            })
            
        except Exception as e:
            logger.error(f"VPMV istatistik hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/vpmv/filter/<signal_type>', methods=['GET'])
    def filter_by_vpmv_signal(signal_type):
        """VPMV sinyaline göre sembolleri filtrele"""
        try:
            valid_signals = ['STRONG_LONG', 'LONG', 'SHORT', 'STRONG_SHORT', 'NEUTRAL']
            signal_type_formatted = signal_type.upper().replace('_', ' ')
            
            if signal_type_formatted not in ['STRONG LONG', 'LONG', 'SHORT', 'STRONG SHORT', 'NEUTRAL']:
                return jsonify({
                    "success": False,
                    "error": f"Geçersiz sinyal tipi. Geçerli değerler: {', '.join(valid_signals)}"
                }), 400
            
            filtered_symbols = memory_storage.get_symbols_by_vpmv_signal(signal_type_formatted)
            
            return jsonify({
                "success": True,
                "signal_type": signal_type_formatted,
                "symbols": filtered_symbols,
                "count": len(filtered_symbols),
                "message": f"{len(filtered_symbols)} sembol {signal_type_formatted} sinyaline sahip"
            })
            
        except Exception as e:
            logger.error(f"VPMV filtreleme hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/vpmv/top', methods=['GET'])
    def get_top_vpmv_symbols():
        """En yüksek/düşük VPMV değerine sahip sembolleri getir"""
        try:
            limit = request.args.get('limit', default=10, type=int)
            sort_by = request.args.get('sort', default='highest', type=str)
            
            if sort_by not in ['highest', 'lowest']:
                return jsonify({
                    "success": False,
                    "error": "sort parametresi 'highest' veya 'lowest' olmalı"
                }), 400
            
            if limit < 1 or limit > 50:
                return jsonify({
                    "success": False,
                    "error": "limit 1-50 arasında olmalı"
                }), 400
            
            top_symbols = memory_storage.get_top_vpmv_symbols(limit=limit, sort_by=sort_by)
            
            return jsonify({
                "success": True,
                "sort_by": sort_by,
                "limit": limit,
                "symbols": top_symbols,
                "count": len(top_symbols),
                "message": f"En {sort_by} {len(top_symbols)} VPMV sembolü"
            })
            
        except Exception as e:
            logger.error(f"VPMV top symbols hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # =====================================================
    # 🔥 YENİ: TETİKLEYİCİ ENDPOINT'LERİ
    # =====================================================
    
    @app.route('/api/vpmv/active-triggers', methods=['GET'])
    def get_active_triggers():
        """
        Aktif tetikleyicileri olan sembolleri getir
        
        Returns:
            JSON: {
                "success": bool,
                "triggers": List[Dict],
                "count": int,
                "trigger_breakdown": Dict,
                "message": str
            }
        """
        try:
            active_triggers = memory_storage.get_active_triggers()
            
            # Tetikleyici tipine göre ayır
            momentum_triggers = [t for t in active_triggers if t['trigger_name'] == 'Momentum']
            volume_triggers = [t for t in active_triggers if t['trigger_name'] == 'Hacim']
            volatility_triggers = [t for t in active_triggers if t['trigger_name'] == 'Volatilite']
            
            trigger_breakdown = {
                'momentum_count': len(momentum_triggers),
                'volume_count': len(volume_triggers),
                'volatility_count': len(volatility_triggers)
            }
            
            logger.info(f"🔥 Aktif tetikleyici listesi: {len(active_triggers)} sembol (M:{len(momentum_triggers)}, H:{len(volume_triggers)}, V:{len(volatility_triggers)})")
            
            return jsonify({
                "success": True,
                "triggers": active_triggers,
                "count": len(active_triggers),
                "trigger_breakdown": trigger_breakdown,
                "message": f"{len(active_triggers)} sembolde aktif tetikleyici var"
            })
            
        except Exception as e:
            logger.error(f"Aktif tetikleyici listesi hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/vpmv/filter-by-trigger/<trigger_type>', methods=['GET'])
    def filter_by_trigger_type(trigger_type):
        """
        Belirli tetikleyici tipine göre sembolleri filtrele
        
        Args:
            trigger_type (str): 'Momentum', 'Hacim', 'Volatilite'
        
        Returns:
            JSON: Filtrelenmiş sembol listesi
        """
        try:
            valid_triggers = ['Momentum', 'Hacim', 'Volatilite']
            
            if trigger_type not in valid_triggers:
                return jsonify({
                    "success": False,
                    "error": f"Geçersiz tetikleyici tipi. Geçerli değerler: {', '.join(valid_triggers)}"
                }), 400
            
            filtered_symbols = memory_storage.get_symbols_by_trigger_type(trigger_type)
            
            logger.info(f"🔍 {trigger_type} tetikleyicisi filtresi: {len(filtered_symbols)} sembol bulundu")
            
            return jsonify({
                "success": True,
                "trigger_type": trigger_type,
                "symbols": filtered_symbols,
                "count": len(filtered_symbols),
                "message": f"{len(filtered_symbols)} sembol {trigger_type} tetikleyicisine sahip"
            })
            
        except Exception as e:
            logger.error(f"Tetikleyici filtreleme hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/vpmv/trigger-statistics', methods=['GET'])
    def get_trigger_statistics():
        """
        Tetikleyici istatistiklerini getir
        
        Returns:
            JSON: Detaylı tetikleyici istatistikleri
        """
        try:
            vpmv_stats = memory_storage.get_vpmv_statistics()
            
            trigger_stats = {
                'total_active_triggers': vpmv_stats['trigger_active_count'],
                'momentum_triggers': vpmv_stats['momentum_trigger_count'],
                'volume_triggers': vpmv_stats['volume_trigger_count'],
                'volatility_triggers': vpmv_stats['volatility_trigger_count'],
                'trigger_distribution': {
                    'Momentum': vpmv_stats['momentum_trigger_count'],
                    'Hacim': vpmv_stats['volume_trigger_count'],
                    'Volatilite': vpmv_stats['volatility_trigger_count']
                },
                'total_symbols': vpmv_stats['total_symbols']
            }
            
            return jsonify({
                "success": True,
                "statistics": trigger_stats,
                "message": f"{trigger_stats['total_active_triggers']} aktif tetikleyici"
            })
            
        except Exception as e:
            logger.error(f"Tetikleyici istatistik hatası: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"success": False, "error": "Endpoint bulunamadı"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"İnternal server hatası: {error}")
        return jsonify({"success": False, "error": "Sunucu hatası"}), 500

# 🔥 GLOBAL DEĞİŞKEN - app.py EN ÜSTE EKLE (main() fonksiyonundan önce)
last_analysis_times = {}  # {symbol: timestamp}

def handle_kline_closed(symbol, close_price, is_kline_closed):
    """
    🔥 TRADİNGVİEW BALANCED MOD + THROTTLING
    
    - KAPANAN MUMLAR: Her zaman tam analiz (15dk'da bir)
    - AÇIK MUMLAR: 5 saniyede bir tam analiz (TradingView benzeri)
    
    Bu sayede:
    ✅ VPMV, C-Signal, Tetikleyici sürekli güncel
    ✅ CPU kullanımı optimize
    ✅ TradingView gibi responsive
    """
    
    # ✅ YENİ: None kontrolü ekle
    if symbol is None or close_price is None:
        logger.warning(f"⚠️ Geçersiz veri: symbol={symbol}, price={close_price}")
        return
    
    # 🔥 HER CALLBACK'TE LOG (None-safe)
    try:
        symbol_str = str(symbol) if symbol else "UNKNOWN"
        price_str = f"{close_price:.6f}" if close_price is not None else "N/A"
        status_str = 'KAPANAN 🟦' if is_kline_closed else 'AÇIK ⚡'
        
        logger.info(f"╔{'=' * 78}╗")
        logger.info(f"║ 🔥 CALLBACK: {symbol_str:<64}║")
        logger.info(f"║   Price: {price_str:<65}║")
        logger.info(f"║   Type: {status_str:<63}║")
        logger.info(f"╚{'=' * 78}╝")
    except Exception as log_err:
        logger.error(f"❌ Log formatı hatası: {log_err}")
        return
    
    try:
        # ✅ Kalıcı listede mi?
        permanent_symbol = memory_storage.get_permanent_symbol(symbol)
        if not permanent_symbol:
            logger.warning(f"⚠️ {symbol} kalıcı listede YOK")
            return
        
        current_time = datetime.now()
        time_str = current_time.strftime('%H:%M:%S.%f')[:-3]
        
        # 🔥 ANALİZ KARARI: Tam analiz mi? Sadece fiyat mı?
        should_analyze = False
        reason = ""
        
        if is_kline_closed:
            # ✅ KAPANAN MUM: HER ZAMAN TAM ANALİZ
            should_analyze = True
            reason = "KAPANAN MUM 🟦"
        else:
            # ⚡ AÇIK MUM: 5 SANİYEDE BİR TAM ANALİZ
            last_time = last_analysis_times.get(symbol)
            
            if last_time is None:
                # İlk açık mum
                should_analyze = True
                reason = "İLK AÇIK MUM ⚡"
            else:
                elapsed = (current_time - last_time).total_seconds()
                
                if elapsed >= 5:
                    # 5 saniye geçti, tam analiz yap
                    should_analyze = True
                    reason = f"AÇIK MUM ({elapsed:.1f}s geçti) ⚡"
                else:
                    # Henüz 5 saniye olmadı, sadece fiyat
                    should_analyze = False
                    reason = f"AÇIK MUM (Throttled: {elapsed:.1f}s/5s) ⏳"
        
        # 🔥🔥🔥 TAM ANALİZ veya SADECE FİYAT? 🔥🔥🔥
        if should_analyze:
            # ✅✅✅ TAM ANALİZ ✅✅✅
            logger.info(f"🔄 {reason} → TAM ANALİZ BAŞLIYOR: {symbol}")
            
            # Tam analiz yap (Supertrend + VPMV + C-Signal + Tetikleyici)
            updated = AnalysisService.update_symbol_with_analysis(
                permanent_symbol,
                permanent_symbol.get("timeframe", "15m"),
                preserve_manual_type=permanent_symbol.get("manual_type_override", False)
            )
            
            # Fiyat ve zaman ekle
            updated['current_price'] = close_price
            updated['last_live_update'] = time_str
            
            # Memory'ye kaydet
            memory_storage.update_permanent_symbol(symbol, updated)
            
            # Son analiz zamanını kaydet
            last_analysis_times[symbol] = current_time
            
            # Frontend'e broadcast
            WebSocketManager.broadcast({
                "event": "live_price_update",
                "symbol": symbol,
                "price": close_price,
                "time": time_str,
                "all_data": memory_storage.get_permanent_symbols()
            })
            
            # 🔥 DETAYLI SONUÇ LOGU (None-safe)
            try:
                price_display = f"{close_price:.6f}" if close_price is not None else "N/A"
                ratio_val = updated.get('ratio_percent', 0)
                ratio_display = f"{ratio_val:.2f}" if ratio_val is not None else "N/A"
                vpmv_val = updated.get('vpmv_net_power', 0)
                vpmv_display = f"{vpmv_val:.2f}" if vpmv_val is not None else "N/A"
                vpmv_signal = str(updated.get('vpmv_signal', 'N/A'))
                c_signal_val = updated.get('c_signal')
                c_signal_display = f"{c_signal_val:.2f}" if c_signal_val is not None else "N/A"
                trigger_name = str(updated.get('vpmv_trigger_name', 'Yok'))
                trigger_active = updated.get('vpmv_trigger_active', False)
                
                logger.info(f"╔{'=' * 78}╗")
                logger.info(f"║ ✅ TAM ANALİZ TAMAMLANDI: {symbol:<50}║")
                logger.info(f"║   💰 Price: {price_display:<62}║")
                logger.info(f"║   📊 Ratio: {ratio_display:<61}%║")
                logger.info(f"║   🎯 VPMV NET: {vpmv_display:<57}║")
                logger.info(f"║   📊 VPMV Signal: {vpmv_signal:<55}║")
                logger.info(f"║   💡 C-Signal: {c_signal_display:<58}║")
                logger.info(f"║   🔥 Tetikleyici: {trigger_name:<55}║")
                logger.info(f"║   ✅ Aktif: {str(trigger_active):<61}║")
                logger.info(f"║   ⏰ Zaman: {time_str:<63}║")
                logger.info(f"╚{'=' * 78}╝")
            except Exception as log_err:
                logger.error(f"❌ Detaylı log hatası: {log_err}")
            
        else:
            # ⚡⚡⚡ SADECE FİYAT GÜNCELLE ⚡⚡⚡
            logger.info(f"⚡ {reason} → Sadece fiyat güncelleniyor: {symbol}")
            
            memory_storage.update_permanent_symbol(symbol, {
                'current_price': close_price,
                'last_live_update': time_str
            })
            
            WebSocketManager.broadcast({
                "event": "live_price_update",
                "symbol": symbol,
                "price": close_price,
                "time": time_str,
                "all_data": memory_storage.get_permanent_symbols()
            })
            
            price_display = f"{close_price:.6f}" if close_price is not None else "N/A"
            logger.info(f"✅ Sadece fiyat güncellendi: {symbol} = {price_display}")
        
    except Exception as e:
        logger.error(f"❌ handle_kline_closed hatası ({symbol}): {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Ana uygulama başlatma fonksiyonu"""
    global ws_manager
    
    try:
        print("\n" + "="*70)
        print("🎯 Supertrend + C-Signal + VPMV + TETİKLEYİCİ Analiz Sistemi")
        print("="*70)
        print("🌐 Panel erişim: http://127.0.0.1:5001")
        print("⚠️  Sadece analiz amaçlıdır, yatırım tavsiyesi değildir!")
        print("="*70 + "\n")
        
        logger.info("Sistem başlatılıyor")
        
        # ----------------------------------
        # 1) Flask uygulaması oluştur
        # ----------------------------------
        app = create_app()

        # ----------------------------------
        # 2) WebSocket başlat - HER DURUMDA
        # ----------------------------------
        logger.info("🔍 Kalıcı liste yükleniyor...")
        persistent_symbols = [s["symbol"] for s in memory_storage.get_permanent_symbols()]

        # 🔥 WebSocket'i her durumda başlat (boş liste bile olsa)
        logger.info(f"🔌 WebSocket başlatılıyor...")
        ws_manager = WebSocketManager(
            symbols=persistent_symbols,  # Boş liste bile olabilir
            interval="1m",
            on_kline_closed=handle_kline_closed
        )
        ws_manager.start()
        
        if persistent_symbols:
            logger.info(f"✅ WebSocket aktif → {len(persistent_symbols)} sembol izleniyor")
        else:
            logger.info("⚠️ WebSocket aktif → Henüz sembol yok (dinamik eklenebilir)")

        # ----------------------------------
        # 3) Flask server çalıştır
        # ----------------------------------
        app.run(debug=False, port=5001, host='127.0.0.1', use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n🛑 Kullanıcı tarafından durduruldu")
        if ws_manager:
            ws_manager.stop()
    except Exception as e:
        logger.error(f"❌ Sistem başlatma hatası: {e}")
        print(f"\n❌ Hata: {e}")
    finally:
        if ws_manager:
            ws_manager.stop()
        logger.info("👋 Analiz Sistemi kapatıldı")
        print("\n👋 Sistem kapatıldı")


if __name__ == '__main__':
    main()