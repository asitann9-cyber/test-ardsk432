"""
🚀 Ana Uygulama - ULTRA PANEL v5 Kripto AI Trading Sistemi
🔥 SADECE ULTRA ETİKET SİSTEMİ: Pine Script "ULTRA 3/4" mantığı
📊 Bars Ago tracking + Basitleştirilmiş filtreleme
"""

import threading
import pandas as pd
from datetime import datetime

import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State

# Kendi modüllerimizi import et
import config
from config import (
    initialize, DASH_CONFIG, TABLE_REFRESH_INTERVAL,
    DEFAULT_TIMEFRAME, DEFAULT_MIN_AI_SCORE, LOCAL_TZ,
    MAX_OPEN_POSITIONS
)
from ui.components import (
    create_layout, create_ai_score_bar
)
from trading.analyzer import batch_analyze_with_ai
from data.fetch_data import get_current_price
from data.database import (
    setup_csv_files, load_trades_from_csv, 
    calculate_performance_metrics
)

# 🔥 LIVE TRADING IMPORT
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

# Global değişkenler
auto_scan_active = False
current_settings = {
    'timeframe': DEFAULT_TIMEFRAME,
    'min_ai': DEFAULT_MIN_AI_SCORE * 100,
    'max_bars_ago': 15  # 🔥 YENİ: Maximum kaç mum önce
}


def auto_scan_worker():
    """Otomatik tarama işleyicisi"""
    global auto_scan_active
    
    while auto_scan_active:
        try:
            logger.info("🔄 ULTRA PANEL v5 otomatik tarama başlatılıyor...")
            result_data = batch_analyze_with_ai(current_settings['timeframe'])
            config.current_data = result_data
            
            if not result_data.empty:
                logger.info(f"✅ Ultra analiz tamamlandı - {len(result_data)} ultra sinyal bulundu")
                
                # 🔥 YENİ: Bars ago istatistikleri
                fresh_signals = len(result_data[result_data['bars_ago'] == 0])
                old_signals = len(result_data[result_data['bars_ago'] > 0])
                logger.info(f"   🆕 Yeni: {fresh_signals} | 📉 Eski: {old_signals}")
            else:
                logger.info("⚠️ Hiç ultra sinyal bulunamadı")
            
            # 1 saniye bekle
            import time
            for i in range(1):
                if not auto_scan_active:  
                    return
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Otomatik tarama hatası: {e}")
            import time
            time.sleep(5)


def start_auto_scan():
    """Otomatik taramayı başlat"""
    global auto_scan_active
    
    if not auto_scan_active:
        auto_scan_active = True
        thread = threading.Thread(target=auto_scan_worker, daemon=True)
        thread.start()
        logger.info(f"🚀 ULTRA PANEL v5 otomatik tarama başlatıldı")


def stop_auto_scan():
    """Otomatik taramayı durdur"""
    global auto_scan_active
    auto_scan_active = False
    logger.info("⛔ Otomatik tarama durduruldu")


# Dash uygulaması oluştur
app = dash.Dash(__name__)
app.title = "🔥 ULTRA PANEL v5 - AI Crypto Trading"

# CSS stilleri
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
    """Live Trading kontrolü"""
    if not LIVE_TRADER_AVAILABLE:
        return "❌ Live trader modülü bulunamadı"
    
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return "🔴 Durduruldu" if not is_live_trading_active() else "🟢 Aktif"
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'btn-start-live-trading' and start_clicks > 0:
        success = start_live_trading()
        return "🟢 Live Trading Başlatıldı ✅" if success else "❌ Başlatılamadı"
    
    elif trigger_id == 'btn-stop-live-trading' and stop_clicks > 0:
        stop_live_trading()
        return "🔴 Live Trading Durduruldu"
    
    return "🔴 Durduruldu" if not is_live_trading_active() else "🟢 Aktif"


@app.callback(
    Output('api-status', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_api_status(n_intervals):
    """API bağlantı durumunu göster"""
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
    """API bağlantı durumu göstergesi"""
    if not LIVE_TRADER_AVAILABLE:
        return html.Div([
            html.Span("❌ Live Trader Modülü Yüklenmedi", style={'color': '#ef4444'}),
            html.Br(),
            html.Small("python-binance kurulu değil", style={'color': '#9ca3af'})
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
                html.Span("🔴 API Bağlantısı Yok", style={'color': '#ef4444', 'fontWeight': 'bold'})
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
    """Performans metrikleri"""
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
    """Trading tabloları güncelleme"""
    positions_data = []
    
    if config.is_live_mode():
        current_positions = config.live_positions.copy()
        mode_info = "Live"
    else:
        current_positions = config.paper_positions.copy()
        mode_info = "Paper"
    
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
        logger.error(f"❌ Pozisyon tablosu hatası: {e}")
    
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
        logger.error(f"❌ Trade tablosu hatası: {e}")
        trades_data = []
    
    return positions_data, trades_data


@app.callback(
    [Output('signals-table', 'data'),
     Output('status-text', 'children'),
     Output('stats-panel', 'children')],
    [Input('interval-component', 'n_intervals')],  
    [State('dd-timeframe', 'value'),
     State('inp-min-ai', 'value'),
     State('inp-max-bars-ago', 'value')]  # 🔥 YENİ: Max bars ago filtresi
)
def update_signals(n_intervals, tf, min_ai_pct, max_bars_ago):
    """🔥 ULTRA ETİKET SİSTEMİ - Sadece Pine Script mantığı"""
    global current_settings
    
    tf = tf or DEFAULT_TIMEFRAME
    min_ai_score = float(min_ai_pct or 30) / 100
    max_bars_ago = int(max_bars_ago or 15)  # 🔥 YENİ
    
    current_settings.update({
        'timeframe': tf,
        'min_ai': min_ai_pct,
        'max_bars_ago': max_bars_ago  # 🔥 YENİ
    })
    
    if config.current_data is None or config.current_data.empty:
        status_text = "Otomatik tarama aktif - Veri bekleniyor..." if auto_scan_active else "Otomatik tarama başlatın"
        return [], status_text, []
    
    df = config.current_data.copy()
    original_count = len(df)
    
    # 🔥 BASİTLEŞTİRİLMİŞ FİLTRELEME (sadece 2 filtre)
    if min_ai_pct > 30:
        df = df[df['ai_score'] >= min_ai_pct]
    
    if max_bars_ago < 15:
        df = df[df['bars_ago'] <= max_bars_ago]
    
    # 🔥 SIRALAMA: AI Score → Bars Ago (en yeni önce)
    df = df.sort_values(
        by=['ai_score', 'bars_ago'],
        ascending=[False, True]  # bars_ago küçük olanı önce
    )
    
    if df.empty:
        status_text = f"Filtreler çok sıkı! {original_count} sinyal var."
        if auto_scan_active:
            status_text += " | 🔄 Otomatik aktif"
        return [], status_text, []
    
    # 🔥 BASİTLEŞTİRİLMİŞ TABLO VERİSİ
    table_data = []
    for _, row in df.iterrows():
        # AI Score bar
        ai_score_unicode = create_ai_score_bar(row['ai_score'])
        
        # 🔥 ULTRA Signal formatı (Pine Script gibi)
        ultra_signal = row['ultra_signal']  # BUY veya SELL
        htf_count = row['htf_count']  # 3 veya 4
        ultra_text = f"🔥 {ultra_signal} {htf_count}/4"
        
        # 🔥 Power formatı
        power_value = row['total_power']
        power_text = f"{power_value:.1f}"
        
        # 🔥 Bars Ago formatı
        bars_ago = row['bars_ago']
        if bars_ago == 0:
            bars_ago_text = "🆕 Şimdi"
        else:
            bars_ago_text = f"📉 {bars_ago} mum"
        
        # Live status
        live_status = "⭐"
        if LIVE_TRADER_AVAILABLE:
            live_status = get_live_bot_status_for_symbol(row['symbol'])
        
        table_data.append({
            'symbol': row['symbol'],
            'ultra_signal': ultra_text,  # 🔥 "🔥 BUY 3/4" formatı
            'power': power_text,
            'bars_ago': bars_ago_text,  # 🔥 YENİ SÜTUN
            'ai_score': ai_score_unicode,
            'live_status': live_status,
            'timestamp': datetime.now(LOCAL_TZ).strftime('%H:%M')
        })
    
    # 🔥 BASİTLEŞTİRİLMİŞ İSTATİSTİKLER
    buy_count = len(df[df['ultra_signal'] == 'BUY'])
    sell_count = len(df[df['ultra_signal'] == 'SELL'])
    htf_4_count = len(df[df['htf_count'] == 4])
    fresh_signals = len(df[df['bars_ago'] == 0])  # 🔥 YENİ
    
    stats_panel = html.Div([
        html.Div([
            html.Div(f"{len(df)}", className="stat-value"),
            html.Div("Ultra Sinyal", className="stat-label")
        ], className="stat-card"),
        html.Div([
            html.Div(f"{buy_count}", className="stat-value", style={'color': '#22c55e'}), 
            html.Div("BUY", className="stat-label")
        ], className="stat-card"),
        html.Div([
            html.Div(f"{sell_count}", className="stat-value", style={'color': '#ef4444'}),  
            html.Div("SELL", className="stat-label")
        ], className="stat-card"),
        html.Div([
            html.Div(f"{htf_4_count}", className="stat-value", style={'color': '#f59e0b'}),  
            html.Div("4/4 HTF", className="stat-label")
        ], className="stat-card"),
        html.Div([
            html.Div(f"{fresh_signals}", className="stat-value", style={'color': '#06b6d4'}),  
            html.Div("🆕 Yeni", className="stat-label")  # 🔥 YENİ
        ], className="stat-card"),
    ], className="stats-grid")
    
    current_positions = config.live_positions if config.is_live_mode() else config.paper_positions
    current_mode = "Live" if config.is_live_mode() else "Paper"
    
    top_3 = df.head(3)['symbol'].tolist() if len(df) >= 3 else df['symbol'].tolist()
    top_3_str = f"(Top 3: {', '.join(top_3)})" if top_3 else ""
    
    # 🔥 BASİTLEŞTİRİLMİŞ STATUS TEXT
    status = f"🔥 {len(df)}/{original_count} ULTRA | BUY:{buy_count} SELL:{sell_count} | 4/4:{htf_4_count} | 🆕:{fresh_signals}"
    
    if auto_scan_active:
        status += " | 🔄 Auto"
    
    if LIVE_TRADER_AVAILABLE and is_live_trading_active():
        auto_sltp = get_auto_sltp_count() if LIVE_TRADER_AVAILABLE else 0
        status += f" | 🤖 {current_mode} ({len(current_positions)}/{MAX_OPEN_POSITIONS})"
    elif len(current_positions) > 0:
        status += f" | 📝 {current_mode} ({len(current_positions)}/{MAX_OPEN_POSITIONS})"
    
    status += f" {top_3_str} | {datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}"
    
    return table_data, status, stats_panel


# Ana çalıştırma
if __name__ == "__main__":
    setup_csv_files()
    logger.info("🔥 ULTRA PANEL v5 Başlatılıyor...")
    logger.info("📊 Pine Script 'ULTRA 3/4' etiket sistemi aktif")
    logger.info("🎯 Hedef: Sadece ultra sinyalleri tarama + Bars ago tracking")
    
    if LIVE_TRADER_AVAILABLE:
        logger.info("✅ Live trading hazır")
    else:
        logger.warning("⚠️ Live trading yok - sadece analiz")
    
    app.run(
        debug=DASH_CONFIG['debug'],
        host=DASH_CONFIG['host'],
        port=DASH_CONFIG['port']
    )
