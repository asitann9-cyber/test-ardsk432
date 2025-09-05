"""
🚀 Ana Uygulama - Kripto AI Trading Sistemi
Modüler yapıda organize edilmiş Dash web uygulaması + Live Trading Bot
"""

import threading
import pandas as pd
from datetime import datetime

import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State

# Kendi modüllerimizi import et
from config import (
    initialize, DASH_CONFIG, TABLE_REFRESH_INTERVAL,
    DEFAULT_TIMEFRAME, DEFAULT_MIN_STREAK, DEFAULT_MIN_PCT, 
    DEFAULT_MIN_VOLR, DEFAULT_MIN_AI_SCORE, LOCAL_TZ,
    current_capital, open_positions, MAX_OPEN_POSITIONS
)
from ui.components import create_layout, create_ai_score_bar
from trading.analyzer import batch_analyze_with_ai
from data.fetch_data import get_current_price
from data.database import (
    setup_csv_files, load_trades_from_csv, 
    calculate_performance_metrics
)

# 🔥 LIVE TRADING IMPORT (Paper trader kaldırıldı)
try:
    from trading.live_trader import (
        start_live_trading, stop_live_trading, is_live_trading_active,
        get_live_trading_status
    )
    LIVE_TRADER_AVAILABLE = True
except ImportError:
    LIVE_TRADER_AVAILABLE = False
    print("⚠️ Live trader modülü bulunamadı - sadece analiz modu")

# Sistem başlatma
logger, session = initialize()

# Global değişkenler
auto_scan_active = False
current_data = pd.DataFrame()
current_settings = {
    'timeframe': DEFAULT_TIMEFRAME,
    'min_streak': DEFAULT_MIN_STREAK,
    'min_pct': DEFAULT_MIN_PCT,
    'min_volr': DEFAULT_MIN_VOLR,
    'min_ai': DEFAULT_MIN_AI_SCORE * 100
}


def auto_scan_worker():
    """Otomatik tarama işleyicisi - 1 saniyede bir çalışır"""
    global auto_scan_active, current_data
    
    while auto_scan_active:
        try:
            logger.info("🔄 Otomatik tarama başlatılıyor...")
            current_data = batch_analyze_with_ai(current_settings['timeframe'])
            logger.info(f"✅ Otomatik tarama tamamlandı - {len(current_data)} sinyal bulundu")
            
            # 1 saniye bekle
            import time
            for i in range(1):  # 1 saniye
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
        logger.info(f"🚀 Otomatik tarama başlatıldı - 1 saniye aralıklarla")


def stop_auto_scan():
    """Otomatik taramayı durdur"""
    global auto_scan_active
    auto_scan_active = False
    logger.info("⛔ Otomatik tarama durduruldu")


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
            return "🟢 Live Trading Başarıyla Başlatıldı ✅ (Gerçek Para)"
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
    Output('performance-metrics', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_performance_metrics(n_intervals):
    """Performans metriklerini güncelle"""
    metrics = calculate_performance_metrics()
    
    return html.Div([
        html.Div([
            html.H4("💰 Mevcut Sermaye", style={'margin': '0', 'fontSize': '14px', 'color': '#22c55e'}),
            html.H2(f"${metrics['current_capital']:.2f}", style={'color': '#22c55e', 'margin': '5px 0', 'fontSize': '18px'})
        ], style={
            'width': '16%', 'display': 'inline-block', 'textAlign': 'center', 
            'margin': '0.5%', 'padding': '15px', 'border': '2px solid #22c55e', 
            'borderRadius': '10px', 'backgroundColor': '#f8fff8'
        }),
        
        html.Div([
            html.H4("📈 Efektif Sermaye", style={'margin': '0', 'fontSize': '14px', 'color': '#3b82f6'}),
            html.H2(f"${metrics['effective_capital']:.2f}", style={'color': '#3b82f6', 'margin': '5px 0', 'fontSize': '18px'})
        ], style={
            'width': '16%', 'display': 'inline-block', 'textAlign': 'center', 
            'margin': '0.5%', 'padding': '15px', 'border': '2px solid #3b82f6', 
            'borderRadius': '10px', 'backgroundColor': '#f0f8ff'
        }),
        
        html.Div([
            html.H4("🎯 Toplam Kar", style={'margin': '0', 'fontSize': '14px', 'color': '#8b5cf6'}),
            html.H2(f"${metrics['realized_total_profit']:.2f}", style={
                'color': '#22c55e' if metrics['realized_total_profit'] >= 0 else '#ef4444', 
                'margin': '5px 0', 'fontSize': '18px'
            })
        ], style={
            'width': '16%', 'display': 'inline-block', 'textAlign': 'center', 
            'margin': '0.5%', 'padding': '15px', 'border': '2px solid #8b5cf6', 
            'borderRadius': '10px', 'backgroundColor': '#faf5ff'
        }),
        
        html.Div([
            html.H4("📋 Toplam İşlem", style={'margin': '0', 'fontSize': '14px', 'color': '#6b7280'}),
            html.H2(f"{metrics['total_trades']}", style={'color': '#6b7280', 'margin': '5px 0', 'fontSize': '18px'})
        ], style={
            'width': '16%', 'display': 'inline-block', 'textAlign': 'center', 
            'margin': '0.5%', 'padding': '15px', 'border': '2px solid #6b7280', 
            'borderRadius': '10px', 'backgroundColor': '#f9fafb'
        }),
        
        html.Div([
            html.H4("🏆 Kazanma Oranı", style={'margin': '0', 'fontSize': '14px', 'color': '#10b981'}),
            html.H2(f"{metrics['win_rate']:.1f}%", style={'color': '#10b981', 'margin': '5px 0', 'fontSize': '18px'})
        ], style={
            'width': '16%', 'display': 'inline-block', 'textAlign': 'center', 
            'margin': '0.5%', 'padding': '15px', 'border': '2px solid #10b981', 
            'borderRadius': '10px', 'backgroundColor': '#f0fdf4'
        })
    ])


@app.callback(
    [Output('positions-table', 'data'),
     Output('trades-table', 'data')],
    [Input('interval-component', 'n_intervals')]
)
def update_trading_tables(n_intervals):
    """Trading tablolarını güncelle"""
    positions_data = []
    
    # Açık pozisyonlar
    try:
        for symbol, position in open_positions.items():
            current_price = get_current_price(symbol)
            if current_price is None:
                current_price = position['entry_price']
            
            current_value = position['quantity'] * current_price
            
            if position['side'] == 'LONG':
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
            
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
                'ai_score': position['signal_data']['ai_score']
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
    global current_data, current_settings
    
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
    
    if current_data is None or current_data.empty:
        status_text = "Otomatik tarama aktif - Veri bekleniyor..." if auto_scan_active else "Otomatik tarama başlatın"
        return [], status_text, []
    
    df = current_data.copy()
    original_count = len(df)
    
    # Filtreleme
    if min_ai_pct > 70:
        df = df[df['ai_score'] >= min_ai_pct]
        
    if min_streak > 6:
        df = df[df['run_count'] >= min_streak]
        
    if min_pct > 3.0:
        df = df[df['run_perc'] >= min_pct]

    if min_volr > 3.0:
        df = df[(df['vol_ratio'].isna()) | (df['vol_ratio'] >= min_volr)]
    
    df = df.sort_values(by=['ai_score', 'run_perc', 'gauss_run', 'vol_ratio'], 
                       ascending=[False, False, False, False])
    
    if df.empty:
        status_text = f"Filtreler çok sıkı! {original_count} sinyal var."
        if auto_scan_active:
            status_text += f" | Otomatik aktif"
        return [], status_text, []
    
    table_data = []
    for _, row in df.iterrows():
        ai_score_unicode = create_ai_score_bar(row['ai_score'])
        
        table_data.append({
            'symbol': row['symbol'],
            'timeframe': row['timeframe'].upper(),
            'side': row['run_type'].upper(),
            'run_count': row['run_count'],
            'run_perc': row['run_perc'],
            'gauss_run': row['gauss_run'],
            'gauss_run_perc': row['gauss_run_perc'] if row['gauss_run_perc'] is not None else 0,
            'vol_ratio': row['vol_ratio'] if row['vol_ratio'] is not None else 0,
            'hh_vol_streak': row['hh_vol_streak'],
            'deviso_ratio': row['deviso_ratio'],
            'ai_score': ai_score_unicode,
            'timestamp': datetime.now(LOCAL_TZ).strftime('%H:%M')
        })
    
    long_count = len(df[df['run_type'] == 'long'])
    short_count = len(df[df['run_type'] == 'short'])
    
    top_3_df = df.head(3)
    top_3_info = f"(En iyi 3: {', '.join(top_3_df['symbol'].tolist())})" if len(top_3_df) > 0 else ""
    
    stats_panel = html.Div([
        html.Div([
            html.Div(f"{len(df)}", className="stat-value"),
            html.Div("Sinyal", className="stat-label")
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
    
    status = f"{len(df)}/{original_count} SİNYAL: (L:{long_count}, S:{short_count}) | {datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}"
    if auto_scan_active:
        status += f" | 🔄 Otomatik aktif"
    
    # 🔥 Live Trading durumu
    if LIVE_TRADER_AVAILABLE and is_live_trading_active():
        status += f" | 🤖 Live Bot aktif ({len(open_positions)}/{MAX_OPEN_POSITIONS})"
    
    status += f" {top_3_info}"
    
    return table_data, status, stats_panel


# Ana çalıştırma
if __name__ == "__main__":
    setup_csv_files()
    logger.info("🚀 AI Crypto Analytics + Live Trading başlatılıyor...")
    
    # Live trader kontrolü
    if LIVE_TRADER_AVAILABLE:
        logger.info("✅ Live trading modülü hazır")
    else:
        logger.warning("⚠️ Live trading modülü bulunamadı - sadece analiz modu")
    
    # Dash uygulamasını başlat
    app.run(
        debug=DASH_CONFIG['debug'],
        host=DASH_CONFIG['host'],
        port=DASH_CONFIG['port']
    )