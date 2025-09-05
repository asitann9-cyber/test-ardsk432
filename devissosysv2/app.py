from flask import Flask, render_template, jsonify, request
import threading
import time
from datetime import datetime
import json
from modules.manager import module_manager

app = Flask(__name__)

# Modülleri yükle
module_manager.load_all_modules()

# Global değişkenler
scan_results = []
last_scan_time = None
is_scanning = False

# Scanner-specific results and status
ema_scan_results = []
ema_last_scan_time = None
ema_is_scanning = False

rsi_macd_scan_results = []
rsi_macd_last_scan_time = None
rsi_macd_is_scanning = False

super_signal_scanner = None
pro_signal_scanner = None
live_test_scanner = None
live_trading_scanner = None

@app.route('/')
def dashboard():
    """Ana dashboard sayfası"""
    return render_template('dashboard.html')

@app.route('/api/scan')
def start_scan():
    """Yeni tarama başlat"""
    global is_scanning, scan_results, last_scan_time
    
    if is_scanning:
        return jsonify({'status': 'error', 'message': 'Tarama zaten devam ediyor!'})
    
    max_coins = request.args.get('max_coins', 480, type=int)
    timeframe = request.args.get('timeframe', '5m')
    
    def scan_thread():
        global is_scanning, scan_results, last_scan_time
        is_scanning = True
        try:
            # EMA Scanner modülünü kullan
            scanner = module_manager.get_module_instance('ema_scanner')
            if scanner:
                scan_results = scanner.scan_coins(max_coins=max_coins, timeframe=timeframe)
                last_scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            else:
                print("EMA Scanner modülü bulunamadı!")
        except Exception as e:
            print(f"Tarama hatası: {e}")
        finally:
            is_scanning = False
    
    thread = threading.Thread(target=scan_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'Tarama başlatıldı!'})

@app.route('/api/results')
def get_results():
    """Tarama sonuçlarını getir"""
    global scan_results, last_scan_time, is_scanning
    
    # Verileri JSON serializable hale getir
    serializable_results = []
    if scan_results:
        for result in scan_results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, datetime):
                    serializable_result[key] = value.isoformat()
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
    
    return jsonify({
        'results': serializable_results,
        'last_scan_time': last_scan_time,
        'is_scanning': is_scanning,
        'total_coins': len(serializable_results)
    })

@app.route('/api/results/ema_scanner')
def get_ema_results():
    """EMA Scanner sonuçlarını getir"""
    global ema_scan_results, ema_last_scan_time, ema_is_scanning
    
    # Verileri JSON serializable hale getir
    serializable_results = []
    if ema_scan_results:
        for result in ema_scan_results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, datetime):
                    serializable_result[key] = value.isoformat()
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
    
    return jsonify({
        'results': serializable_results,
        'last_scan_time': ema_last_scan_time,
        'is_scanning': ema_is_scanning,
        'total_coins': len(serializable_results)
    })

@app.route('/api/results/rsi_macd_scanner')
def get_rsi_macd_results():
    """RSI/MACD Scanner sonuçlarını getir"""
    global rsi_macd_scan_results, rsi_macd_last_scan_time, rsi_macd_is_scanning
    
    # Verileri JSON serializable hale getir
    serializable_results = []
    if rsi_macd_scan_results:
        for result in rsi_macd_scan_results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, datetime):
                    serializable_result[key] = value.isoformat()
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
    
    return jsonify({
        'results': serializable_results,
        'last_scan_time': rsi_macd_last_scan_time,
        'is_scanning': rsi_macd_is_scanning,
        'total_coins': len(serializable_results)
    })

@app.route('/api/status')
def get_status():
    """Sistem durumunu getir"""
    global is_scanning, last_scan_time
    
    return jsonify({
        'is_scanning': is_scanning,
        'last_scan_time': last_scan_time,
        'uptime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/modules')
def get_modules():
    """Yüklenen modülleri listele"""
    loaded_modules = module_manager.list_loaded_modules()
    modules_info = {}
    
    for module_name in loaded_modules:
        modules_info[module_name] = module_manager.get_module_info(module_name)
    
    return jsonify({
        'loaded_modules': loaded_modules,
        'modules_info': modules_info,
        'total_modules': len(loaded_modules)
    })

@app.route('/api/modules/<module_name>')
def get_module_info(module_name):
    """Belirli bir modülün bilgilerini getir"""
    module_info = module_manager.get_module_info(module_name)
    return jsonify(module_info)

@app.route('/api/scan/ema_scanner')
def start_ema_scan():
    """EMA Scanner taraması başlat"""
    global ema_is_scanning, ema_scan_results, ema_last_scan_time
    
    if ema_is_scanning:
        return jsonify({'status': 'error', 'message': 'EMA Tarama zaten devam ediyor!'})
    
    max_coins = request.args.get('max_coins', 30, type=int)
    timeframe = request.args.get('timeframe', '5m')
    
    def ema_scan_thread():
        global ema_is_scanning, ema_scan_results, ema_last_scan_time
        ema_is_scanning = True
        try:
            # EMA Scanner modülünü kullan
            scanner = module_manager.get_module_instance('ema_scanner')
            if scanner:
                results = scanner.scan_coins(max_coins=max_coins, timeframe=timeframe)
                # Sonuçları düzleştir
                ema_scan_results = []
                for signal in results:
                    ema_scan_results.append({
                        'symbol': signal['symbol'],
                        'signal_type': signal['signal_type'],
                        'entry_price': round(signal['entry_price'], 6),
                        'stop_loss': round(signal['stop_loss'], 6),
                        'take_profit': round(signal['take_profit'], 6),
                        'ratio': signal['ratio'],
                        'timestamp': signal['timestamp'].isoformat(),
                        'timeframe': signal['timeframe'],
                        'ranking_score': signal.get('ranking_score', 0),
                        'ranking_details': signal.get('ranking_details', {})
                    })
                ema_last_scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            else:
                print("EMA Scanner modülü bulunamadı!")
        except Exception as e:
            print(f"EMA tarama hatası: {e}")
        finally:
            ema_is_scanning = False
    
    thread = threading.Thread(target=ema_scan_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'EMA Scanner taraması başlatıldı!'})

@app.route('/api/scan/rsi_macd_scanner')
def start_rsi_macd_scan():
    """RSI/MACD Scanner taraması başlat"""
    global rsi_macd_is_scanning, rsi_macd_scan_results, rsi_macd_last_scan_time
    
    if rsi_macd_is_scanning:
        return jsonify({'status': 'error', 'message': 'RSI/MACD Tarama zaten devam ediyor!'})
    
    max_coins = request.args.get('max_coins', 30, type=int)
    timeframe = request.args.get('timeframe', '5m')
    
    def rsi_macd_scan_thread():
        global rsi_macd_is_scanning, rsi_macd_scan_results, rsi_macd_last_scan_time
        rsi_macd_is_scanning = True
        try:
            # RSI/MACD Scanner modülünü kullan
            scanner = module_manager.get_module_instance('rsi_macd_scanner')
            if scanner:
                results = scanner.scan_coins(max_coins=max_coins, timeframe=timeframe)
                # Sonuçları düzleştir
                rsi_macd_scan_results = []
                for signal in results:
                    rsi_macd_scan_results.append({
                        'symbol': signal['symbol'],
                        'signal_type': signal['signal_type'],
                        'entry_price': round(signal['entry_price'], 6),
                        'rsi': signal['rsi'],
                        'macd': signal['macd'],
                        'macd_signal': signal['macd_signal'],
                        'macd_histogram': signal['macd_histogram'],
                        'timestamp': signal['timestamp'].isoformat(),
                        'timeframe': signal['timeframe'],
                        'ranking_score': signal.get('ranking_score', 0),
                        'ranking_details': signal.get('ranking_details', {})
                    })
                rsi_macd_last_scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            else:
                print("RSI/MACD Scanner modülü bulunamadı!")
        except Exception as e:
            print(f"RSI/MACD tarama hatası: {e}")
        finally:
            rsi_macd_is_scanning = False
    
    thread = threading.Thread(target=rsi_macd_scan_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'RSI/MACD Scanner taraması başlatıldı!'})

@app.route('/api/scan/super_signal')
def start_super_signal_scan():
    """Super Signal taraması başlat"""
    global super_signal_scanner, is_scanning
    
    if is_scanning:
        return jsonify({'status': 'error', 'message': 'Tarama zaten devam ediyor!'})
    
    max_coins = request.args.get('max_coins', 50, type=int)
    timeframe = request.args.get('timeframe', '5m')
    
    def super_signal_scan_thread():
        global super_signal_scanner, is_scanning
        is_scanning = True
        try:
            # Super Signal Scanner modülünü kullan
            scanner = module_manager.get_module_instance('super_signal_scanner')
            if scanner:
                super_signal_scanner = scanner
                scanner.start_super_signal_scan(max_coins=max_coins, timeframe=timeframe)
            else:
                print("Super Signal Scanner modülü bulunamadı!")
        except Exception as e:
            print(f"Super Signal tarama hatası: {e}")
        finally:
            is_scanning = False
    
    thread = threading.Thread(target=super_signal_scan_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'Super Signal taraması başlatıldı!'})

@app.route('/api/scan/super_signal/stop')
def stop_super_signal_scan():
    """Super Signal taramasını durdur"""
    global super_signal_scanner, is_scanning
    
    if super_signal_scanner:
        super_signal_scanner.stop_scan()
        is_scanning = False
        return jsonify({'status': 'success', 'message': 'Super Signal taraması durduruldu!'})
    else:
        return jsonify({'status': 'error', 'message': 'Aktif Super Signal taraması bulunamadı!'})

@app.route('/api/results/super_signal')
def get_super_signal_results():
    """Super Signal sonuçlarını getir"""
    global super_signal_scanner
    
    if super_signal_scanner:
        try:
            latest_signals = super_signal_scanner.get_latest_signals(limit=100)
            stats = super_signal_scanner.get_statistics()
            
            return jsonify({
                'signals': latest_signals,
                'statistics': stats,
                'is_running': super_signal_scanner.is_running,
                'current_cycle': super_signal_scanner.current_cycle
            })
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'signals': [], 'statistics': {}, 'is_running': False, 'current_cycle': 0})

@app.route('/api/scan/pro_signal')
def start_pro_signal_scan():
    """Pro Signal taraması başlat"""
    global pro_signal_scanner, is_scanning
    
    if is_scanning:
        return jsonify({'status': 'error', 'message': 'Tarama zaten devam ediyor!'})
    
    max_coins = request.args.get('max_coins', 50, type=int)
    
    def pro_signal_scan_thread():
        global pro_signal_scanner, is_scanning
        is_scanning = True
        try:
            # Pro Signal Scanner modülünü kullan
            scanner = module_manager.get_module_instance('pro_signal_scanner')
            if scanner:
                pro_signal_scanner = scanner
                scanner.start_pro_signal_scan(max_coins=max_coins)
            else:
                print("Pro Signal Scanner modülü bulunamadı!")
        except Exception as e:
            print(f"Pro Signal tarama hatası: {e}")
        finally:
            is_scanning = False
    
    thread = threading.Thread(target=pro_signal_scan_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'Pro Signal taraması başlatıldı!'})

@app.route('/api/scan/pro_signal/stop')
def stop_pro_signal_scan():
    """Pro Signal taramasını durdur"""
    global pro_signal_scanner, is_scanning
    
    if pro_signal_scanner:
        pro_signal_scanner.stop_scan()
        is_scanning = False
        return jsonify({'status': 'success', 'message': 'Pro Signal taraması durduruldu!'})
    else:
        return jsonify({'status': 'error', 'message': 'Aktif Pro Signal taraması bulunamadı!'})

@app.route('/api/results/pro_signal')
def get_pro_signal_results():
    """Pro Signal sonuçlarını getir"""
    global pro_signal_scanner
    
    if pro_signal_scanner:
        try:
            latest_signals = pro_signal_scanner.get_latest_signals(limit=100)
            stats = pro_signal_scanner.get_statistics()
            
            return jsonify({
                'signals': latest_signals,
                'statistics': stats,
                'is_running': pro_signal_scanner.is_running,
                'current_cycle': pro_signal_scanner.current_cycle
            })
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'signals': [], 'statistics': {}, 'is_running': False, 'current_cycle': 0})

@app.route('/api/scan/live_test')
def start_live_test():
    """Live Test başlat"""
    global live_test_scanner, is_scanning
    
    if is_scanning:
        return jsonify({'status': 'error', 'message': 'Test zaten devam ediyor!'})
    
    max_coins = request.args.get('max_coins', 480, type=int)
    max_trades = request.args.get('max_trades', 5, type=int)
    
    def live_test_thread():
        global live_test_scanner, is_scanning
        is_scanning = True
        try:
            # Live Test Scanner modülünü kullan
            scanner = module_manager.get_module_instance('live_test_scanner')
            if scanner:
                live_test_scanner = scanner
                scanner.start_live_test(max_coins=max_coins, max_trades=max_trades)
            else:
                print("❌ Live Test Scanner modülü yüklenemedi!")
        except Exception as e:
            print(f"❌ Live Test başlatma hatası: {e}")
        finally:
            is_scanning = False
    
    thread = threading.Thread(target=live_test_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'Live Test başlatıldı!'})

@app.route('/api/scan/live_test/stop')
def stop_live_test():
    """Live Test durdur"""
    global live_test_scanner
    if live_test_scanner:
        live_test_scanner.stop_live_test()
        return jsonify({'status': 'success', 'message': 'Live Test durduruldu!'})
    return jsonify({'status': 'error', 'message': 'Live Test çalışmıyor!'})

@app.route('/api/results/live_test')
def get_live_test_results():
    """Live Test sonuçlarını getir"""
    global live_test_scanner
    try:
        if live_test_scanner:
            active_trades = live_test_scanner.get_active_trades()
            completed_trades = live_test_scanner.get_completed_trades()
            performance = live_test_scanner.get_performance_summary()
            
            return jsonify({
                'active_trades': active_trades,
                'completed_trades': completed_trades,
                'performance': performance,
                'is_running': live_test_scanner.is_running
            })
        else:
            return jsonify({
                'active_trades': [],
                'completed_trades': [],
                'performance': {},
                'is_running': False
            })
    except Exception as e:
        print(f"Live Test results error: {e}")
        return jsonify({
            'active_trades': [],
            'completed_trades': [],
            'performance': {},
            'is_running': False,
            'error': str(e)
        })

@app.route('/api/scan/live_trading')
def start_live_trading():
    """Live Trading başlat"""
    global live_trading_scanner, is_scanning
    
    if is_scanning:
        return jsonify({'status': 'error', 'message': 'Trading zaten devam ediyor!'})
    
    max_coins = request.args.get('max_coins', 480, type=int)
    max_trades = request.args.get('max_trades', 5, type=int)
    
    def live_trading_thread():
        global live_trading_scanner, is_scanning
        is_scanning = True
        try:
            # Live Trading Scanner modülünü kullan
            scanner = module_manager.get_module_instance('live_trading_scanner')
            if scanner:
                live_trading_scanner = scanner
                scanner.start_live_trading(max_coins=max_coins, max_trades=max_trades)
            else:
                print("❌ Live Trading Scanner modülü yüklenemedi!")
        except Exception as e:
            print(f"❌ Live Trading başlatma hatası: {e}")
        finally:
            is_scanning = False
    
    thread = threading.Thread(target=live_trading_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'Live Trading başlatıldı!'})

@app.route('/api/scan/live_trading/stop')
def stop_live_trading():
    """Live Trading durdur"""
    global live_trading_scanner
    if live_trading_scanner:
        live_trading_scanner.stop_live_trading()
        return jsonify({'status': 'success', 'message': 'Live Trading durduruldu!'})
    return jsonify({'status': 'error', 'message': 'Live Trading çalışmıyor!'})

@app.route('/api/results/live_trading')
def get_live_trading_results():
    """Live Trading sonuçlarını getir"""
    global live_trading_scanner
    if live_trading_scanner:
        active_trades = live_trading_scanner.get_active_trades()
        completed_trades = live_trading_scanner.get_completed_trades()
        performance = live_trading_scanner.get_performance_summary()
        
        return jsonify({
            'active_trades': active_trades,
            'completed_trades': completed_trades,
            'performance': performance,
            'is_running': live_trading_scanner.is_running
        })
    return jsonify({
        'active_trades': [],
        'completed_trades': [],
        'performance': {},
        'is_running': False
    })

@app.route('/api/futures/balance')
def get_futures_balance():
    """Futures hesap bakiyesini getir"""
    global live_trading_scanner
    if live_trading_scanner:
        try:
            account_info = live_trading_scanner.get_futures_account_info(force_update=True)
            return jsonify({
                'status': 'success',
                'account_info': account_info,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e),
                'account_info': {},
                'timestamp': datetime.now().isoformat()
            })
    return jsonify({
        'status': 'error',
        'message': 'Live Trading Scanner aktif değil',
        'account_info': {},
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/live-trading/test-trade', methods=['POST'])
def test_manual_trade():
    """Test için manuel işlem aç"""
    global live_trading_scanner
    
    if not live_trading_scanner:
        return jsonify({
            'status': 'error',
            'message': 'Live Trading Scanner aktif değil!'
        })
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTCUSDT')
        signal_type = data.get('signal_type', 'LONG')
        
        success = live_trading_scanner.test_manual_trade(symbol, signal_type)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Test işlemi başarıyla açıldı: {symbol} {signal_type}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Test işlemi açılamadı: {symbol} {signal_type}'
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Test işlemi hatası: {str(e)}'
        })

@app.route('/api/live-trading/close-all-trades', methods=['POST'])
def close_all_trades():
    """Tüm aktif işlemleri kapat"""
    global live_trading_scanner
    
    if not live_trading_scanner:
        return jsonify({
            'status': 'error',
            'message': 'Live Trading Scanner aktif değil!'
        })
    
    try:
        success = live_trading_scanner.close_all_trades()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Tüm işlemler başarıyla kapatıldı',
                'closed_count': len(live_trading_scanner.get_active_trades())
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'İşlemler kapatılırken hata oluştu'
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'İşlem kapatma hatası: {str(e)}'
        })

if __name__ == '__main__':
    print("🚀 Deviso System Dashboard Başlatılıyor...")
    print("📊 Modül 1: Binance Futures Scanner aktif")
    print("🌐 Dashboard: http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
