"""
🚀 Ana Uygulama - Kripto AI Trading Sistemi
Modüler yapıda organize edilmiş Dash web uygulaması + Live Trading Bot
🔥 LIVE TRADER ENTEGRASYONu TAMAMLANDI + Config Senkronizasyonu
🆕 RSI MOMENTUM + LOGARİTMİK HACİM SİSTEMİ ENTEGRE EDİLDİ
❌ ESKİ VOLUME SİSTEMİ TAMAMEN KALDIRILDI
"""

import threading
import pandas as pd
from datetime import datetime

import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State

# Kendi modüllerimizi import et
import config  # 🔥 YENİ: config modülünü direkt import et
from config import (
    initialize, DASH_CONFIG, TABLE_REFRESH_INTERVAL,
    DEFAULT_TIMEFRAME, DEFAULT_MIN_STREAK, DEFAULT_MIN_PCT, 
    DEFAULT_MIN_AI_SCORE, LOCAL_TZ, MAX_OPEN_POSITIONS
)
from ui.components import create_layout, create_ai_score_bar
from trading.analyzer import batch_analyze_with_ai
from data.fetch_data import get_current_price
from data.database import (
    setup_csv_files, load_trades_from_csv, 
    calculate_performance_metrics
)

# 🔥 LIVE TRADING IMPORT - GÜNCELLENMIŞ
try:
    from trading.live_trader import (
        start_live_trading, stop_live_trading, is_live_trading_active,
        get_live_trading_status, get_live_bot_status_for_symbol, get_auto_sltp_count
    )
    LIVE_TRADER_AVAILABLE = True
except ImportError:
    LIVE_TRADER_AVAILABLE = False
    print("⚠️ Live trader modülü bulunamadı - sadece analiz modu")

# Sistem başlatma
logger, session = initialize()

# Global değişkenler - 🆕 GÜNCELLENMIŞ
auto_scan_active = False
current_settings = {
    'timeframe': DEFAULT_TIMEFRAME,
    'min_streak': DEFAULT_MIN_STREAK,
    'min_pct': DEFAULT_MIN_PCT,
    'min_rsi_momentum': 2.0,          # 🆕 YENİ: RSI momentum filtresi
    'min_log_volume': 1.0,            # 🆕 YENİ: Log volume filtresi
    'min_momentum_score': 40,         # 🆕 YENİ: Momentum score filtresi
    'min_ai': DEFAULT_MIN_AI_SCORE * 100
    # ❌ KALDIRILDI: 'min_volr': DEFAULT_MIN_VOLR
}


def auto_scan_worker():
    """🔥 DÜZELTME: Otomatik tarama işleyicisi - config.current_data'yı günceller"""
    global auto_scan_active
    
    while auto_scan_active:
        try:
            logger.info("🔄 RSI momentum + log volume tarama başlatılıyor...")
            # 🔥 YENİ: config.current_data'yı direkt güncelle
            result_data = batch_analyze_with_ai(current_settings['timeframe'])
            config.current_data = result_data  # GLOBAL GÜNCELLEME
            logger.info(f"✅ RSI momentum + log volume tarama tamamlandı - {len(result_data)} sinyal bulundu")
            logger.info(f"🔥 config.current_data güncellendi: {len(config.current_data)} satır")
            
            # 1 saniye bekle
            import time
            for i in range(1):  # 1 saniye
                if not auto_scan_active:  
                    return
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"RSI momentum + log volume tarama hatası: {e}")
            import time
            time.sleep(5)


def start_auto_scan():
    """Otomatik taramayı başlat"""
    global auto_scan_active
    
    if not auto_scan_active:
        auto_scan_active = True
        thread = threading.Thread(target=auto_scan_worker, daemon=True)
        thread.start()
        logger.info(f"🚀 RSI momentum + log volume otomatik tarama başlatıldı - 1 saniye aralıklarla")


def stop_auto_scan():
    """Otomatik taramayı durdur"""
    global auto_scan_active
    auto_scan_active = False
    logger.info("⛔ RSI momentum + log volume otomatik tarama durduruldu")


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


# Callbacks
@app.callback(
    [Output('btn-auto', 'children'),
     Output('btn-auto', 'className')],
    Input('btn-auto', 'n_clicks')
)
def toggle_auto_scan(n_clicks):
    """Otomatik tarama butonunu kontrol et"""
    global auto_scan_active
    
    if n_clicks > 0:
        if auto_scan_active:
            stop_auto_scan()
            return "🔄 Otomatik Başlat", "control-button auto-button"
        else:
            start_auto_scan()
            return "⛔ Otomatik Durdur", "control-button auto-button active"
    
    return "🔄 Otomatik Başlat", "control-button auto-button"


@app.callback(
    Output('trading-status', 'children'),
    [Input('btn-start-live-trading', 'n_clicks'),
     Input('btn-stop-live-trading', 'n_clicks')]
)
def control_live_trading(start_clicks, stop_clicks):
    """🔥 Live Trading kontrolü"""
    if not LIVE_TRADER_AVAILABLE:
        return "❌ Live trader modülü bulunamadı - .env dosyasını kontrol edin"
    
    ctx = dash.callback_context
    
    if not ctx.triggered:
        status = "🔴 Live Trading Durduruldu" if not is_live_trading_active() else "🟢 Live Trading Aktif"
        return status
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'btn-start-live-trading' and start_clicks > 0:
        success = start_live_trading()
        if success:
            return "🟢 Live Trading Başarıyla Başlatıldı ✅ "
        else:
            return "❌ Live Trading Başlatılamadı - API anahtarlarını kontrol edin"
    
    elif trigger_id == 'btn-stop-live-trading' and stop_clicks > 0:
        stop_live_trading()
        return "🔴 Live Trading Durduruldu - Tüm pozisyonlar kapatıldı"
    
    return "🔴 Live Trading Durduruldu" if not is_live_trading_active() else "🟢 Live Trading Aktif"


@app.callback(
    Output('api-status', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_api_status(n_intervals):
    """🔥 API bağlantı durumunu göster"""
    if not LIVE_TRADER_AVAILABLE:
        return html.Div("❌ Live Trader Yok", style={'color': '#ef4444'})
    
    try:
        status = get_live_trading_status()
        if status['api_connected']:
            return html.Div([
                html.Span("✅ API Bağlı", style={'color': '#22c55e'}),
                html.Br(),
                html.Small(f"Bakiye: ${status['balance']:.2f}", style={'color': '#cccccc'})
            ])
        else:
            return html.Div("❌ API Bağlantısı Yok", style={'color': '#ef4444'})
    except:
        return html.Div("⚠️ API Durumu Bilinmiyor", style={'color': '#f59e0b'})


@app.callback(
    Output('api-connection-status', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_api_connection_status(n_intervals):
    """🔥 YENİ: API bağlantı durumu göstergesi"""
    if not LIVE_TRADER_AVAILABLE:
        return html.Div([
            html.Span("❌ Live Trader Modülü Yüklenmedi", style={'color': '#ef4444'}),
            html.Br(),
            html.Small("python-binance kurulu değil veya .env dosyası eksik", style={'color': '#9ca3af'})
        ])
    
    try:
        status = get_live_trading_status()
        
        if status['api_connected']:
            return html.Div([
                html.Span("🟢 API Bağlantısı Aktif", style={'color': '#22c55e', 'fontWeight': 'bold'}),
                html.Br(),
                html.Small(f"Environment: {status['environment']}", style={'color': '#9ca3af'}),
                html.Br(),
                html.Small(f"Bakiye: ${status['balance']:.2f} USDT", style={'color': '#22c55e'}),
                html.Br(),
                html.Small(f"Otomatik SL/TP: {'Aktif' if status.get('auto_sltp_enabled', False) else 'Kapalı'}", 
                          style={'color': '#22c55e' if status.get('auto_sltp_enabled', False) else '#f59e0b'})
            ])
        else:
            return html.Div([
                html.Span("🔴 API Bağlantısı Yok", style={'color': '#ef4444', 'fontWeight': 'bold'}),
                html.Br(),
                html.Small("", style={'color': '#9ca3af'})
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
    """🔥 GÜNCELLEME: Kazanma oranı kaldırıldı, 4 kart eşit genişlikte"""
    try:
        # Config'den mevcut modu al
        if config.is_live_mode():
            current_capital = config.live_capital
            open_positions = config.live_positions.copy()  
            mode_info = "🤖 Live"
        else:
            current_capital = config.paper_capital
            open_positions = config.paper_positions.copy()  
            mode_info = "📝 Paper"
        
        # Gerçekleşmemiş P&L hesapla (açık pozisyonlarla)
        total_unrealized_pnl = 0.0
        
        # 🔥 DÜZELTME: list() ile güvenli iteration
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
        
        # Auto SL/TP sayısı - 🔥 GÜVENLI
        auto_sltp_count = sum(1 for p in list(open_positions.values()) if p.get("auto_sltp", False))
        
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
            
            # 🔥 KAZANMA ORANI KALDIRILDI - Artık sadece 4 kart var
            
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
    """🔥 DÜZELTME: Dictionary iteration problemi çözüldü"""
    positions_data = []
    
    # Config'den mevcut pozisyonları al
    if config.is_live_mode():
        current_positions = config.live_positions.copy()  # 🔥 KOPYA AL
        mode_info = "Live"
    else:
        current_positions = config.paper_positions.copy()  # 🔥 KOPYA AL
        mode_info = "Paper"
    
    # Açık pozisyonlar - 🔥 GÜVENLI ITERATION
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
            
            # 🔥 YENİ: Otomatik SL/TP durumu göster
            auto_sltp_status = f"🤖 Auto ({mode_info})" if position.get('auto_sltp', False) else f"📱 Manual ({mode_info})"
            
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
     State('inp-min-rsi-momentum', 'value'),
     State('inp-min-ai', 'value')]
)
def update_signals(n_intervals, tf, min_streak, min_pct, min_rsi_momentum, min_ai_pct):
    """🆕 GÜNCELLENMIŞ: RSI momentum + log volume sinyal güncelleme - hata korumalı"""
    global current_settings
    
    tf = tf or DEFAULT_TIMEFRAME
    min_streak = int(min_streak or DEFAULT_MIN_STREAK)
    min_pct = float(min_pct or DEFAULT_MIN_PCT)
    min_rsi_momentum = float(min_rsi_momentum or 2.0)
    min_ai_score = float(min_ai_pct or 30) / 100
    
    current_settings.update({
        'timeframe': tf,
        'min_streak': min_streak,
        'min_pct': min_pct,
        'min_rsi_momentum': min_rsi_momentum,
        'min_ai': min_ai_pct
    })
    
    if config.current_data is None or config.current_data.empty:
        status_text = "RSI momentum + log volume tarama aktif - Veri bekleniyor..." if auto_scan_active else "RSI momentum + log volume tarama başlatın"
        return [], status_text, []
    
    df = config.current_data.copy()
    original_count = len(df)
    
    # 🔍 DEBUG: Hangi sütunlar var kontrol et
    available_columns = list(df.columns)
    logger.debug(f"Mevcut sütunlar: {available_columns}")
    
    # Filtreleme - güvenli kontroller
    if min_ai_pct > 70:
        df = df[df['ai_score'] >= min_ai_pct]
        
    if min_streak > 6:
        df = df[df['run_count'] >= min_streak]
        
    if min_pct > 3.0:
        df = df[df['run_perc'] >= min_pct]

    # 🔧 RSI momentum filtresi - güvenli kontrol
    if min_rsi_momentum > 0 and 'rsi_momentum' in df.columns:
        df = df[abs(df['rsi_momentum']) >= min_rsi_momentum]
        logger.debug(f"RSI momentum filtresi uygulandı: >= {min_rsi_momentum}")
    elif min_rsi_momentum > 0:
        logger.warning("RSI momentum filtresi istendi ama 'rsi_momentum' sütunu bulunamadı")
    
    # 🔧 Güvenli sıralama - önce hangi sütunların var olduğunu kontrol et
    sort_columns = ['ai_score']
    
    # Yeni momentum sütunlarını kontrol et ve ekle
    if 'momentum_score' in df.columns:
        sort_columns.append('momentum_score')
    if 'log_volume_strength' in df.columns:
        sort_columns.append('log_volume_strength')
    
    # Varsayılan sıralama sütunlarını ekle
    if 'run_perc' in df.columns:
        sort_columns.append('run_perc')
    if 'gauss_run' in df.columns:
        sort_columns.append('gauss_run')
    
    logger.debug(f"Sıralama sütunları: {sort_columns}")
    
    try:
        df = df.sort_values(by=sort_columns, ascending=[False] * len(sort_columns))
    except Exception as e:
        logger.error(f"Sıralama hatası: {e}")
        # Hata durumunda sadece AI score'a göre sırala
        df = df.sort_values(by=['ai_score'], ascending=[False])
    
    if df.empty:
        status_text = f"RSI momentum + log volume filtreleri çok sıkı! {original_count} sinyal var."
        if auto_scan_active:
            status_text += " | Otomatik aktif"
        return [], status_text, []
    
    # 🔧 Tablo verisi oluşturma - güvenli veri alma
    table_data = []
    for _, row in df.iterrows():
        try:
            ai_score_unicode = create_ai_score_bar(row['ai_score'])
            
            live_status = "⭐"
            if LIVE_TRADER_AVAILABLE:
                live_status = get_live_bot_status_for_symbol(row['symbol'])
            
            table_data.append({
                'symbol': row.get('symbol', ''),
                'timeframe': str(row.get('timeframe', '')).upper(),
                'side': str(row.get('run_type', '')).upper(),
                'run_count': int(row.get('run_count', 0)),
                'run_perc': float(row.get('run_perc', 0.0)),
                'gauss_run': float(row.get('gauss_run', 0.0)),
                'gauss_run_perc': float(row.get('gauss_run_perc', 0.0) or 0.0),
                
                # 🆕 YENİ SÜTUNLAR - güvenli veri alma
                'rsi_momentum': float(row.get('rsi_momentum', 0.0) or 0.0),
                'log_volume_strength': float(row.get('log_volume_strength', 0.0) or 0.0),
                'momentum_score': int(row.get('momentum_score', 0) or 0),
                
                'deviso_ratio': float(row.get('deviso_ratio', 0.0)),
                'ai_score': ai_score_unicode,
                'live_status': live_status,
                'timestamp': datetime.now(LOCAL_TZ).strftime('%H:%M')
            })
        except Exception as e:
            logger.error(f"Tablo verisi oluşturma hatası {row.get('symbol', 'UNKNOWN')}: {e}")
            continue
    
    # İstatistikler
    long_count = len(df[df['run_type'] == 'long']) if 'run_type' in df.columns else 0
    short_count = len(df[df['run_type'] == 'short']) if 'run_type' in df.columns else 0
    
    stats_panel = html.Div([
        html.Div([
            html.Div(f"{len(df)}", className="stat-value"),
            html.Div("RSI+Vol Sinyal", className="stat-label")
        ], className="stat-card"),
        html.Div([
            html.Div(f"{long_count}", className="stat-value", style={'color': '#22c55e'}), 
            html.Div("Long", className="stat-label")
        ], className="stat-card"),
        html.Div([
            html.Div(f"{short_count}", className="stat-value", style={'color': '#ef4444'}),  
            html.Div("Short", className="stat-label")
        ], className="stat-card"),
    ], className="stats-grid")
    
    # Pozisyon bilgileri
    current_positions = config.live_positions if config.is_live_mode() else config.paper_positions
    current_mode = "Live" if config.is_live_mode() else "Paper"
    
    status = f"{len(df)}/{original_count} RSI+VOL SİNYAL: (L:{long_count}, S:{short_count}) | {datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}"
    if auto_scan_active:
        status += " | Otomatik aktif"
    
    if LIVE_TRADER_AVAILABLE and is_live_trading_active():
        auto_sltp_count = get_auto_sltp_count() if LIVE_TRADER_AVAILABLE else 0
        status += f" | {current_mode} Bot aktif ({len(current_positions)}/{MAX_OPEN_POSITIONS})"
    
    # 🔍 DEBUG bilgileri
    if len(df) > 0:
        sample_row = df.iloc[0]
        logger.debug(f"İlk sinyalden örnek veriler:")
        logger.debug(f"  RSI momentum: {sample_row.get('rsi_momentum', 'YOK')}")
        logger.debug(f"  Log volume: {sample_row.get('log_volume_strength', 'YOK')}")
        logger.debug(f"  Momentum score: {sample_row.get('momentum_score', 'YOK')}")
    
    return table_data, status, stats_panel


# Ana çalıştırma
if __name__ == "__main__":
    setup_csv_files()
    logger.info("🚀 AI Crypto Analytics + Live Trading başlatılıyor...")
    logger.info("🆕 RSI MOMENTUM + LOGARİTMİK HACİM sistemi aktif")
    logger.info("❌ Eski basit volume sistemi tamamen kaldırıldı")
    
    # Live trader kontrolü
    if LIVE_TRADER_AVAILABLE:
        logger.info("✅ Live trading modülü hazır")
        logger.info("🔑 .env dosyasındaki API anahtarları ile otomatik bağlantı yapılacak")
        logger.info("🤖 Otomatik SL/TP emirleri Binance sunucu tarafında yönetilecek")
        logger.info("🆕 RSI momentum + log volume sinyalleri ile trading yapılacak")
    else:
        logger.warning("⚠️ Live trading modülü bulunamadı - sadece RSI momentum + log volume analiz modu")
        logger.warning("⚠️ Live trading için: pip install python-binance")
        logger.warning("⚠️ .env dosyasına BINANCE_API_KEY ve BINANCE_SECRET_KEY ekleyin")
    
    # Dash uygulamasını başlat
    app.run(
        debug=DASH_CONFIG['debug'],
        host=DASH_CONFIG['host'],
        port=DASH_CONFIG['port']
    )