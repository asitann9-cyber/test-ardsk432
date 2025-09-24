"""
🚀 Ana Uygulama - Kripto AI Trading Sistemi - DÜZELTME
🔧 BUTON SORUNLARI ÇÖZÜLDü: Otomatik durdur ve Live bot durdur
🔧 WEBSOCKET TUTARSIZLIĞI ÇÖZÜLDü: Manuel kapatmada temizleniyor
🔧 THREAD YÖNETİMİ DÜZELTME: Proper cleanup ve state management
"""

import threading
import time
import pandas as pd
from datetime import datetime

import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State

# Kendi modüllerimizi import et
import config
from config import (
    initialize, DASH_CONFIG, TABLE_REFRESH_INTERVAL,
    DEFAULT_TIMEFRAME, DEFAULT_MIN_STREAK, DEFAULT_MIN_PCT, 
    DEFAULT_MIN_VOLR, DEFAULT_MIN_AI_SCORE, LOCAL_TZ,
    MAX_OPEN_POSITIONS
)
from ui.components import create_layout, create_ai_score_bar
from trading.analyzer import batch_analyze_with_ai
from data.fetch_data import get_current_price
from data.database import (
    setup_csv_files, load_trades_from_csv, 
    calculate_performance_metrics
)

# Live Trading import
try:
    from trading.live_trader import (
        start_live_trading, stop_live_trading, is_live_trading_active,
        get_live_trading_status, get_live_bot_status_for_symbol, get_auto_sltp_count,
        websocket_active_symbols, websocket_manager, stop_websocket
    )
    LIVE_TRADER_AVAILABLE = True
except ImportError:
    LIVE_TRADER_AVAILABLE = False
    websocket_active_symbols = set()
    print("⚠️ Live trader modülü bulunamadı - sadece analiz modu")

# Sistem başlatma
logger, session = initialize()

# 🔧 DÜZELTME: Thread-safe global değişkenler
auto_scan_active = False
auto_scan_thread = None
auto_scan_lock = threading.Lock()

current_settings = {
    'timeframe': DEFAULT_TIMEFRAME,
    'min_streak': DEFAULT_MIN_STREAK,
    'min_pct': DEFAULT_MIN_PCT,
    'min_volr': DEFAULT_MIN_VOLR,
    'min_ai': DEFAULT_MIN_AI_SCORE * 100
}


def get_websocket_rest_counts(df_length: int) -> tuple:
    """🔧 DÜZELTME: Gerçek WebSocket/REST sayılarını hesapla"""
    try:
        if (LIVE_TRADER_AVAILABLE and 
            websocket_active_symbols and 
            len(websocket_active_symbols) > 0):
            
            websocket_count = len(websocket_active_symbols)
            rest_count = max(0, df_length - websocket_count)
            logger.debug(f"📊 Gerçek WebSocket: {websocket_count}, REST: {rest_count}")
        else:
            websocket_count = 0
            rest_count = df_length
            logger.debug(f"📱 WebSocket kapalı - tümü REST: {rest_count}")
        
        return websocket_count, rest_count
        
    except Exception as e:
        logger.debug(f"WebSocket/REST hesaplama hatası: {e}")
        try:
            live_positions_count = len(config.live_positions) if config.live_positions else 0
            return live_positions_count, max(0, df_length - live_positions_count)
        except:
            return 0, df_length


def auto_scan_worker():
    """🔧 DÜZELTME: Thread-safe otomatik tarama işleyicisi"""
    global auto_scan_active
    
    logger.info("🚀 Otomatik tarama thread başlatıldı")
    
    while auto_scan_active:
        try:
            if not auto_scan_active:  # Double check
                break
                
            logger.info("🔄 Otomatik tarama başlatılıyor...")
            result_data = batch_analyze_with_ai(current_settings['timeframe'])
            
            # Thread-safe config güncelleme
            with auto_scan_lock:
                if auto_scan_active:  # Hala aktifse güncelle
                    config.current_data = result_data
                    logger.info(f"✅ Otomatik tarama tamamlandı - {len(result_data)} sinyal bulundu")
                
            # Interruptible sleep
            for i in range(10):  # 1 saniye = 10 x 0.1 saniye
                if not auto_scan_active:
                    break
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Otomatik tarama hatası: {e}")
            # Hata durumunda daha uzun bekle
            for i in range(50):  # 5 saniye
                if not auto_scan_active:
                    break
                time.sleep(0.1)
    
    logger.info("⛔ Otomatik tarama thread sonlandırıldı")


def start_auto_scan():
    """🔧 DÜZELTME: Thread-safe otomatik tarama başlatma"""
    global auto_scan_active, auto_scan_thread
    
    with auto_scan_lock:
        if auto_scan_active:
            logger.warning("⚠️ Otomatik tarama zaten aktif")
            return False
            
        # Eski thread temizle
        if auto_scan_thread and auto_scan_thread.is_alive():
            logger.info("🔧 Eski thread temizleniyor...")
            auto_scan_active = False
            auto_scan_thread.join(timeout=2)
        
        # Yeni thread başlat
        auto_scan_active = True
        auto_scan_thread = threading.Thread(target=auto_scan_worker, daemon=True)
        auto_scan_thread.start()
        
        logger.info("🚀 Otomatik tarama başlatıldı - 1 saniye aralıklarla")
        return True


def stop_auto_scan():
    """🔧 DÜZELTME: Thread-safe otomatik tarama durdurma"""
    global auto_scan_active, auto_scan_thread
    
    with auto_scan_lock:
        if not auto_scan_active:
            logger.info("💤 Otomatik tarama zaten durdurulmuş")
            return True
            
        logger.info("⛔ Otomatik tarama durduruluyor...")
        auto_scan_active = False
        
        # Thread'in bitmesini bekle
        if auto_scan_thread and auto_scan_thread.is_alive():
            auto_scan_thread.join(timeout=3)
            
            if auto_scan_thread.is_alive():
                logger.warning("⚠️ Thread zorla sonlandırıldı")
            else:
                logger.info("✅ Thread temiz şekilde sonlandırıldı")
        
        auto_scan_thread = None
        logger.info("⛔ Otomatik tarama durduruldu")
        return True


# Dash uygulaması oluştur
app = dash.Dash(__name__)
app.title = DASH_CONFIG['title']

# CSS stillerini ekle
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
            }
            
            .Select-control, .Select-menu-outer, .Select-option {
                background-color: #2a2a2a !important;
                color: #ffffff !important;
                border-color: #404040 !important;
            }
            .Select-option:hover {
                background-color: #404040 !important;
                color: #ffffff !important;
            }
            .Select-value-label, .Select-placeholder {
                color: #ffffff !important;
            }
            .Select-arrow-zone {
                color: #ffffff !important;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 1rem;
                margin-bottom: 1.5rem;
            }
            
            .stat-card {
                background: linear-gradient(135deg, #2a2a2a 0%, #363636 100%);
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
                border: 1px solid #404040;
            }
            
            .stat-value {
                font-size: 1.5rem;
                font-weight: bold;
                color: #22c55e;
                margin-bottom: 0.25rem;
            }
            
            .stat-label {
                font-size: 0.875rem;
                color: #cccccc;
            }
            
            .control-button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                padding: 12px 24px;
                color: white;
                font-weight: 600;
                cursor: pointer;
                border-radius: 6px;
                font-size: 14px;
                transition: all 0.3s ease;
                margin: 5px;
            }
            
            .control-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            
            .auto-button {
                background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
            }
            
            .auto-button.active {
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            }
            
            .live-button {
                background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            }
            
            .live-button.active {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout'u oluştur
app.layout = create_layout()

# 🔧 DÜZELTME: Otomatik tarama callback
@app.callback(
    [Output('btn-auto', 'children'),
     Output('btn-auto', 'className')],
    Input('btn-auto', 'n_clicks'),
    prevent_initial_call=False
)
def toggle_auto_scan(n_clicks):
    """🔧 DÜZELTME: Thread-safe otomatik tarama toggle"""
    
    # İlk yükleme kontrolü
    if not n_clicks or n_clicks == 0:
        if auto_scan_active:
            return "⛔ Otomatik Durdur", "control-button auto-button active"
        else:
            return "🔄 Otomatik Başlat", "control-button auto-button"
    
    # Button tıklandığında
    try:
        with auto_scan_lock:
            current_status = auto_scan_active
        
        if current_status:
            # Şu anda aktif - durdur
            success = stop_auto_scan()
            if success:
                logger.info("✅ Otomatik tarama kullanıcı tarafından durduruldu")
                return "🔄 Otomatik Başlat", "control-button auto-button"
            else:
                logger.error("❌ Otomatik tarama durdurulamadı")
                return "⛔ Otomatik Durdur", "control-button auto-button active"
        else:
            # Şu anda durdurulmuş - başlat
            success = start_auto_scan()
            if success:
                logger.info("✅ Otomatik tarama kullanıcı tarafından başlatıldı")
                return "⛔ Otomatik Durdur", "control-button auto-button active"
            else:
                logger.error("❌ Otomatik tarama başlatılamadı")
                return "🔄 Otomatik Başlat", "control-button auto-button"
                
    except Exception as e:
        logger.error(f"❌ Otomatik tarama toggle hatası: {e}")
        return "🔄 Otomatik Başlat", "control-button auto-button"


# 🔧 DÜZELTME: Live trading callback
@app.callback(
    Output('trading-status', 'children'),
    [Input('btn-start-live-trading', 'n_clicks'),
     Input('btn-stop-live-trading', 'n_clicks')],
    prevent_initial_call=False
)
def control_live_trading(start_clicks, stop_clicks):
    """🔧 DÜZELTME: Live Trading kontrolü - WebSocket temizleme dahil"""
    
    if not LIVE_TRADER_AVAILABLE:
        return "❌ Live trader modülü bulunamadı - .env dosyasını kontrol edin"
    
    ctx = dash.callback_context
    
    # İlk yükleme kontrolü
    if not ctx.triggered or not ctx.triggered[0]['prop_id']:
        current_status = is_live_trading_active()
        if current_status:
            return "🟢 Live Trading Aktif ✅ (WebSocket + REST hibrit)"
        else:
            return "🔴 Live Trading Durduruldu"
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'btn-start-live-trading' and start_clicks and start_clicks > 0:
        logger.info("🔥 LIVE TRADING BAŞLATMA komutu alındı")
        
        try:
            # Zaten aktifse uyarı ver
            if is_live_trading_active():
                logger.warning("⚠️ Live trading zaten aktif")
                return "🟢 Live Trading Zaten Aktif ✅"
            
            # Live trading başlat
            success = start_live_trading()
            
            if success:
                logger.info("✅ Live Trading başarıyla başlatıldı")
                return "🟢 Live Trading Başarıyla Başlatıldı ✅ (WebSocket + REST hibrit)"
            else:
                logger.error("❌ Live Trading başlatılamadı")
                return "❌ Live Trading Başlatılamadı - API anahtarlarını kontrol edin"
                
        except Exception as e:
            logger.error(f"❌ Live Trading başlatma hatası: {e}")
            return f"❌ Live Trading başlatma hatası: {str(e)[:100]}"
    
    elif trigger_id == 'btn-stop-live-trading' and stop_clicks and stop_clicks > 0:
        logger.info("🔥 LIVE TRADING DURDURMA komutu alındı")
        
        try:
            # Zaten durdurulmuşsa uyarı ver
            if not is_live_trading_active():
                logger.warning("⚠️ Live trading zaten durdurulmuş")
                return "🔴 Live Trading Zaten Durdurulmuş"
            
            # 🔧 KRITIK DÜZELTME: ZORLA durdurma işlemi
            logger.info("🛑 ZORLA Live Trading durduruluyor - TÜM POZİSYONLAR KAPANACAK")
            
            # Live trading durdur (pozisyonları kapat ve CSV'ye kaydet)
            stop_live_trading()
            
            # 🔧 YENİ: WebSocket'i de zorla temizle
            try:
                if LIVE_TRADER_AVAILABLE:
                    stop_websocket()
                    websocket_active_symbols.clear()
                    logger.info("🧹 WebSocket zorla temizlendi")
            except Exception as ws_err:
                logger.error(f"❌ WebSocket temizleme hatası: {ws_err}")
            
            # Config temizle
            try:
                config.live_positions.clear()
                logger.info("🧹 Config pozisyonları temizlendi")
            except Exception as config_err:
                logger.error(f"❌ Config temizleme hatası: {config_err}")
            
            # Durum kontrolü
            time.sleep(1)  # Kısa bekle
            
            if not is_live_trading_active():
                logger.info("✅ Live Trading başarıyla durduruldu")
                return "🔴 Live Trading Durduruldu - Tüm pozisyonlar kapatıldı ve CSV'ye kaydedildi"
            else:
                logger.warning("⚠️ Live Trading tam olarak durdurulamadı")
                return "⚠️ Live Trading durduruluyor... (Lütfen bekleyin)"
                
        except Exception as e:
            logger.error(f"❌ Live Trading durdurma hatası: {e}")
            return f"❌ Live Trading durdurma hatası: {str(e)[:100]}"
    
    # Default durum
    current_status = is_live_trading_active()
    if current_status:
        return "🟢 Live Trading Aktif ✅"
    else:
        return "🔴 Live Trading Durduruldu"


@app.callback(
    Output('api-connection-status', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_api_connection_status(n_intervals):
    """🔥 API bağlantı durumu göstergesi + WebSocket bilgisi"""
    if not LIVE_TRADER_AVAILABLE:
        return html.Div([
            html.Span("❌ Live Trader Modülü Yüklenmedi", style={'color': '#ef4444'}),
            html.Br(),
            html.Small("python-binance kurulu değil veya .env dosyası eksik", style={'color': '#9ca3af'})
        ])
    
    try:
        status = get_live_trading_status()
        
        if status['api_connected']:
            websocket_count, _ = get_websocket_rest_counts(0)
            
            return html.Div([
                html.Span("🟢 API Bağlantısı Aktif", style={'color': '#22c55e', 'fontWeight': 'bold'}),
                html.Br(),
                html.Small(f"Environment: {status['environment']}", style={'color': '#9ca3af'}),
                html.Br(),
                html.Small(f"Bakiye: ${status['balance']:.2f} USDT", style={'color': '#22c55e'}),
                html.Br(),
                html.Small(f"🤖 WebSocket: {websocket_count} coin, 📱 REST: Geri kalan", 
                          style={'color': '#3b82f6'}),
                html.Br(),
                html.Small(f"Otomatik SL/TP: {'Aktif' if status.get('auto_sltp_enabled', False) else 'Kapalı'}", 
                          style={'color': '#22c55e' if status.get('auto_sltp_enabled', False) else '#f59e0b'})
            ])
        else:
            return html.Div([
                html.Span("🔴 API Bağlantısı Yok", style={'color': '#ef4444', 'fontWeight': 'bold'}),
                html.Br(),
                html.Small("WebSocket ve REST API kapalı", style={'color': '#9ca3af'})
            ])
            
    except Exception as e:
        return html.Div([
            html.Span("⚠️ API Durumu Kontrol Edilemiyor", style={'color': '#f59e0b', 'fontWeight': 'bold'}),
            html.Br(),
            html.Small(f"Hata: {str(e)[:50]}...", style={'color': '#9ca3af'})
        ])


@app.callback(
    Output('performance-metrics', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_performance_metrics(n_intervals):
    """Performance metrikleri güncelleme"""
    try:
        if config.is_live_mode():
            current_capital = config.live_capital
            open_positions = config.live_positions.copy()  
            mode_info = "🤖 Live"
        else:
            current_capital = config.paper_capital
            open_positions = config.paper_positions.copy()  
            mode_info = "📝 Paper"
        
        # Gerçekleşmemiş P&L hesapla
        total_unrealized_pnl = 0.0
        
        for symbol, position in list(open_positions.items()):
            current_price = get_current_price(symbol)
            if current_price is None:
                current_price = position['entry_price']
            
            if position['side'] == 'LONG':
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
            
            total_unrealized_pnl += unrealized_pnl
        
        # Trade geçmişi analizi
        metrics = calculate_performance_metrics()
        
        return html.Div([
            html.Div([
                html.H4(f"💰 {mode_info} Sermaye", style={'margin': '0', 'fontSize': '14px', 'color': '#22c55e'}),
                html.H2(f"${current_capital:.2f}", style={'color': '#22c55e', 'margin': '5px 0', 'fontSize': '18px'})
            ], style={
                'width': '24%', 'display': 'inline-block', 'textAlign': 'center', 
                'margin': '0.5%', 'padding': '15px', 'border': '2px solid #22c55e', 
                'borderRadius': '10px', 'backgroundColor': '#f8fff8'
            }),
            
            html.Div([
                html.H4("📈 Toplam Kar", style={'margin': '0', 'fontSize': '14px', 'color': '#10b981'}),
                html.H2(f"${metrics['total_gain']:.2f}", style={'color': '#10b981', 'margin': '5px 0', 'fontSize': '18px'})
            ], style={
                'width': '24%', 'display': 'inline-block', 'textAlign': 'center', 
                'margin': '0.5%', 'padding': '15px', 'border': '2px solid #10b981', 
                'borderRadius': '10px', 'backgroundColor': '#f0fdf4'
            }),
            
            html.Div([
                html.H4("📉 Toplam Zarar", style={'margin': '0', 'fontSize': '14px', 'color': '#ef4444'}),
                html.H2(f"${metrics['total_loss']:.2f}", style={'color': '#ef4444', 'margin': '5px 0', 'fontSize': '18px'})
            ], style={
                'width': '24%', 'display': 'inline-block', 'textAlign': 'center', 
                'margin': '0.5%', 'padding': '15px', 'border': '2px solid #ef4444', 
                'borderRadius': '10px', 'backgroundColor': '#fef2f2'
            }),
            
            html.Div([
                html.H4("🎯 Gerçekleşmemiş P&L", style={'margin': '0', 'fontSize': '14px', 'color': '#8b5cf6'}),
                html.H2(f"${total_unrealized_pnl:.2f}", style={
                    'color': '#22c55e' if total_unrealized_pnl >= 0 else '#ef4444', 
                    'margin': '5px 0', 'fontSize': '18px'
                })
            ], style={
                'width': '24%', 'display': 'inline-block', 'textAlign': 'center', 
                'margin': '0.5%', 'padding': '15px', 'border': '2px solid #8b5cf6', 
                'borderRadius': '10px', 'backgroundColor': '#faf5ff'
            })
        ])
        
    except Exception as e:
        logger.error(f"❌ Performans metrikleri hatası: {e}")
        return html.Div("Performans verileri yüklenemiyor...")


@app.callback(
    [Output('positions-table', 'data'),
     Output('trades-table', 'data')],
    [Input('interval-component', 'n_intervals')]
)
def update_trading_tables(n_intervals):
    """Trading tablolarını güncelle"""
    positions_data = []
    
    # Config'den mevcut pozisyonları al
    if config.is_live_mode():
        current_positions = config.live_positions.copy()
        mode_info = "Live"
    else:
        current_positions = config.paper_positions.copy()
        mode_info = "Paper"
    
    # Açık pozisyonlar
    try:
        for symbol, position in list(current_positions.items()):
            current_price = get_current_price(symbol)
            if current_price is None:
                current_price = position['entry_price']
            
            current_value = position['quantity'] * current_price
            
            if position['side'] == 'LONG':
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
            
            # WebSocket/REST + Otomatik SL/TP durumu göster
            if LIVE_TRADER_AVAILABLE:
                live_status = get_live_bot_status_for_symbol(symbol)
                if "🤖" in live_status:
                    control_type = "WebSocket"
                elif "📱" in live_status:
                    control_type = "REST API"
                else:
                    control_type = "Unknown"
            else:
                control_type = "Manual"
                
            auto_sltp_status = f"🤖 Auto ({control_type})" if position.get('auto_sltp', False) else f"📱 Manual ({control_type})"
            
            positions_data.append({
                'symbol': symbol,
                'side': position['side'],
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'current_price': current_price,
                'invested_amount': position['invested_amount'],
                'current_value': current_value,
                'unrealized_pnl': unrealized_pnl,
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'ai_score': position['signal_data']['ai_score'],
                'sltp_mode': auto_sltp_status
            })
    except Exception as e:
        logger.error(f"❌ Pozisyon tablosu güncelleme hatası: {e}")
    
    # Trade geçmişi
    try:
        trades_df = load_trades_from_csv()
        if not trades_df.empty:
            trades_data = []
            for _, row in trades_df.sort_values('timestamp', ascending=False).head(50).iterrows():
                trades_data.append({
                    'timestamp': row.get('timestamp', ''),
                    'symbol': row.get('symbol', ''),
                    'side': row.get('side', ''),
                    'quantity': row.get('quantity', 0),
                    'entry_price': row.get('entry_price', 0),
                    'exit_price': row.get('exit_price', 0),
                    'pnl': row.get('pnl', 0),
                    'ai_score': row.get('ai_score', 0),
                    'close_reason': row.get('close_reason', ''),
                    'status': row.get('status', '')
                })
        else:
            trades_data = []
    except Exception as e:
        logger.error(f"❌ Trade tablosu güncelleme hatası: {e}")
        trades_data = []
    
    return positions_data, trades_data


@app.callback(
    [Output('signals-table', 'data'),
     Output('status-text', 'children'),
     Output('stats-panel', 'children')],
    [Input('interval-component', 'n_intervals')],  
    [State('dd-timeframe', 'value'),
     State('inp-min-streak', 'value'),
     State('inp-min-pct', 'value'),
     State('inp-min-volr', 'value'),
     State('inp-min-ai', 'value')]
)
def update_signals(n_intervals, tf, min_streak, min_pct, min_volr, min_ai_pct):  
    """🔥 Config entegre sinyal güncelleme + WebSocket/REST hesaplama"""
    global current_settings
    
    tf = tf or DEFAULT_TIMEFRAME
    min_streak = int(min_streak or DEFAULT_MIN_STREAK)
    min_pct = float(min_pct or DEFAULT_MIN_PCT)
    min_volr = float(min_volr or DEFAULT_MIN_VOLR)
    min_ai_score = float(min_ai_pct or 30) / 100
    
    current_settings.update({
        'timeframe': tf,
        'min_streak': min_streak,
        'min_pct': min_pct,
        'min_volr': min_volr,
        'min_ai': min_ai_pct
    })
    
    # Config.current_data kullan
    if config.current_data is None or config.current_data.empty:
        status_text = "Otomatik tarama aktif - Veri bekleniyor..." if auto_scan_active else "Otomatik tarama başlatın"
        return [], status_text, []
    
    df = config.current_data.copy()
    original_count = len(df)
    
    # Filtreleme
    if min_ai_pct > 70:
        df = df[df['ai_score'] >= min_ai_pct]
        
    if min_streak > 6:
        df = df[df['run_count'] >= min_streak]
        
    if min_pct > 3.0:
        df = df[df['run_perc'] >= min_pct]

    # Vol_ratio yerine log_volume kullan
    if min_volr > 3.0:
        df = df[(df['log_volume'].isna()) | (df['log_volume'] >= min_volr)]
    
    # Sıralama sütunları
    sort_columns = ['ai_score']
    if 'trend_strength' in df.columns:
        sort_columns.append('trend_strength')
    sort_columns.extend(['run_perc', 'gauss_run', 'log_volume_momentum'])
    
    df = df.sort_values(by=sort_columns, ascending=[False] * len(sort_columns))
    
    # Live Bot ile karşılaştırma için log
    logger.info(f"📊 UI'nin gösterdiği ilk 3: {df.head(3)['symbol'].tolist()}")
    
    if df.empty:
        status_text = f"Filtreler çok sıkı! {original_count} sinyal var."
        if auto_scan_active:
            status_text += f" | Otomatik aktif"
        return [], status_text, []
    
    table_data = []
    for i, (_, row) in enumerate(df.iterrows()):
        ai_score_unicode = create_ai_score_bar(row['ai_score'])
        
        # Live bot durumu + WebSocket/REST ayrımı
        live_status = "⭐"
        if LIVE_TRADER_AVAILABLE:
            live_status = get_live_bot_status_for_symbol(row['symbol'])
            
            # Gerçek WebSocket durumuna göre simge seçimi
            current_symbol = row['symbol']
            try:
                if LIVE_TRADER_AVAILABLE and websocket_active_symbols and current_symbol in websocket_active_symbols:
                    if "✅" in live_status:
                        live_status = "✅🤖"  # Açık pozisyon + WebSocket
                    else:
                        live_status = "⭐🤖"  # WebSocket hazır
                else:
                    if "✅" in live_status:
                        live_status = "✅📱"  # Açık pozisyon + REST
                    else:
                        live_status = "⭐📱"  # REST hazır
            except:
                # Fallback durumu
                live_status = "⭐📱"  # REST hazır
        
        table_data.append({
            'symbol': row['symbol'],
            'timeframe': row['timeframe'].upper(),
            'side': row['run_type'].upper(),
            'run_count': row['run_count'],
            'run_perc': row['run_perc'],
            'gauss_run': row['gauss_run'],
            'gauss_run_perc': row['gauss_run_perc'] if row['gauss_run_perc'] is not None else 0,
            'log_volume': row['log_volume'] if row['log_volume'] is not None else 0,
            'log_volume_momentum': row['log_volume_momentum'] if row['log_volume_momentum'] is not None else 0,
            'deviso_ratio': row['deviso_ratio'],
            'c_signal_momentum': row.get('c_signal_momentum', 0.0),
            'ai_score': ai_score_unicode,
            'live_status': live_status,
            'timestamp': datetime.now(LOCAL_TZ).strftime('%H:%M')
        })
    
    long_count = len(df[df['run_type'] == 'long'])
    short_count = len(df[df['run_type'] == 'short'])
    
    top_3_df = df.head(3)
    top_3_info = f"(En iyi 3: {', '.join(top_3_df['symbol'].tolist())})" if len(top_3_df) > 0 else ""
    
    # Gerçek WebSocket/REST sayılarını hesapla
    websocket_count, rest_count = get_websocket_rest_counts(len(df))
    
    stats_panel = html.Div([
        html.Div([
            html.Div(f"{len(df)}", className="stat-value"),
            html.Div("Toplam Sinyal", className="stat-label")
        ], className="stat-card"),
        html.Div([
            html.Div(f"{websocket_count}", className="stat-value", style={'color': '#3b82f6'}), 
            html.Div("🤖 WebSocket", className="stat-label")
        ], className="stat-card"),
        html.Div([
            html.Div(f"{rest_count}", className="stat-value", style={'color': '#f59e0b'}),  
            html.Div("📱 REST API", className="stat-label")
        ], className="stat-card"),
        html.Div([
            html.Div(f"{long_count}/{short_count}", className="stat-value", style={'color': '#8b5cf6'}),  
            html.Div("Long/Short", className="stat-label")
        ], className="stat-card"),
    ], className="stats-grid")
    
    current_positions = config.live_positions if config.is_live_mode() else config.paper_positions
    current_mode = "Live" if config.is_live_mode() else "Paper"
    
    status = f"{len(df)}/{original_count} SİNYAL: (L:{long_count}, S:{short_count}) | {datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}"
    if auto_scan_active:
        status += f" | 🔄 Otomatik aktif"
    
    if LIVE_TRADER_AVAILABLE and is_live_trading_active():
        auto_sltp_count = get_auto_sltp_count() if LIVE_TRADER_AVAILABLE else 0
        
        # Gerçek WebSocket/REST sayıları
        try:
            real_ws_count, real_rest_count = get_websocket_rest_counts(len(df))
            status += f" | 🤖 {current_mode} Bot aktif ({len(current_positions)}/{MAX_OPEN_POSITIONS})"
            status += f" [🤖WS:{real_ws_count}, 📱REST:{real_rest_count}, SL/TP:{auto_sltp_count}]"
        except Exception as e:
            status += f" | 🤖 {current_mode} Bot aktif ({len(current_positions)}/{MAX_OPEN_POSITIONS}) [Auto SL/TP: {auto_sltp_count}]"
            logger.debug(f"Status hesaplama hatası: {e}")
            
    elif len(current_positions) > 0:
        status += f" | 📝 {current_mode} Trading ({len(current_positions)}/{MAX_OPEN_POSITIONS})"
    
    status += f" {top_3_info}"
    
    return table_data, status, stats_panel


# Ana çalıştırma
if __name__ == "__main__":
    setup_csv_files()
    logger.info("🚀 AI Crypto Analytics + Live Trading başlatılıyor...")
    
    # Live trader kontrolü
    if LIVE_TRADER_AVAILABLE:
        logger.info("✅ Live trading modülü hazır")
        logger.info("🔑 .env dosyasındaki API anahtarları ile otomatik bağlantı yapılacak")
        logger.info("🤖 Hibrit sistem: Gerçek WebSocket/REST hesaplama ile")
        logger.info("🚀 WebSocket ile anında SL/TP algılama aktif")
        logger.info("📱 REST API ile backup kontrol sistemi aktif")
        logger.info("🎯 Otomatik SL/TP emirleri Binance sunucu tarafında yönetilecek")
        logger.info("🔧 DÜZELTME: Buton sorunları ve WebSocket tutarsızlığı çözüldü")
    else:
        logger.warning("⚠️ Live trading modülü bulunamadı - sadece analiz modu")
        logger.warning("⚠️ Live trading için: pip install python-binance")
        logger.warning("⚠️ .env dosyasına BINANCE_API_KEY ve BINANCE_SECRET_KEY ekleyin")
    
    # Dash uygulamasını başlat
    app.run(
        debug=DASH_CONFIG['debug'],
        host=DASH_CONFIG['host'],
        port=DASH_CONFIG['port']
    )