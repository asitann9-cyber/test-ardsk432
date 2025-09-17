"""
🎨 UI Bileşenleri
Dash arayüz bileşenleri ve layout fonksiyonları
"""

from dash import html, dcc, dash_table
from config import (
    DEFAULT_TIMEFRAME, DEFAULT_MIN_STREAK, DEFAULT_MIN_PCT, 
    DEFAULT_MIN_VOLR, DEFAULT_MIN_AI_SCORE, TABLE_REFRESH_INTERVAL
)


def create_ai_score_bar(score):
    """Unicode karakterlerle progress bar - dolu kısım renkli"""
    score = float(score)
    
    bar_length = 10
    filled_length = int(bar_length * score / 100)
    
    if score >= 70:
        filled_char = '🟩'  
        emoji = '🟢'
    elif score >= 50:
        filled_char = '🟨' 
        emoji = '🟡'
    elif score >= 30:
        filled_char = '🟧' 
        emoji = '🟠'
    else:
        filled_char = '🟥' 
        emoji = '🔴'
    
    empty_char = '⬜'  
    
    bar = filled_char * filled_length + empty_char * (bar_length - filled_length)
    
    return f"{emoji} {bar} {score:.0f}%"


def create_header():
    """Header bölümünü oluştur"""
    return html.Div([
        html.H1("🤖 AI Crypto Analytics + Live Trading Bot", style={
            'textAlign': 'center',
            'margin': '0 0 0.5rem 0',
            'fontSize': '2rem',
            'fontWeight': '700'
        }),
        html.P("⚠️ UYARI: Eğitim amaçlıdır, yatırım tavsiyesi değildir.", style={
            'textAlign': 'center',
            'margin': '0 0 2rem 0',
            'color': '#ff4d4d',
            'fontWeight': '600'
        })
    ], style={'padding': '2rem 0 1rem 0'})




def create_control_panel():
    """🔥 Kontrol paneli bölümünü oluştur - Live Trading butonları"""
    return html.Div([
        html.Div([
            html.Label("Timeframe"),
            dcc.Dropdown(
                id='dd-timeframe',
                options=[
                    {'label': '1m', 'value': '1m'},
                    {'label': '5m', 'value': '5m'},
                    {'label': '15m', 'value': '15m'},
                    {'label': '1h', 'value': '1h'},
                    {'label': '4h', 'value': '4h'},
                ],
                value=DEFAULT_TIMEFRAME,
                clearable=False
            )
        ], style={'minWidth': '120px'}),
        
        html.Div([
            html.Label("Min Streak"),
            dcc.Input(
                id='inp-min-streak',
                type='number',
                value=DEFAULT_MIN_STREAK,
                min=1,
                style={'width': '80px', 'backgroundColor': '#2a2a2a', 'border': '1px solid #404040', 'color': '#ffffff', 'padding': '8px'}
            )
        ], style={'minWidth': '100px'}),
        
        html.Div([
            html.Label("Min Move %"),
            dcc.Input(
                id='inp-min-pct',
                type='number',
                value=DEFAULT_MIN_PCT,
                step=0.1,
                min=0,
                style={'width': '80px', 'backgroundColor': '#2a2a2a', 'border': '1px solid #404040', 'color': '#ffffff', 'padding': '8px'}
            )
        ], style={'minWidth': '100px'}),
        
        html.Div([
            html.Label("Min Vol Ratio"),
            dcc.Input(
                id='inp-min-volr',
                type='number',
                value=DEFAULT_MIN_VOLR,
                step=0.1,
                min=0,
                style={'width': '80px', 'backgroundColor': '#2a2a2a', 'border': '1px solid #404040', 'color': '#ffffff', 'padding': '8px'}
            )
        ], style={'minWidth': '100px'}),
        
        html.Div([
            html.Label("Min AI %"),
            dcc.Input(
                id='inp-min-ai',
                type='number',
                value=DEFAULT_MIN_AI_SCORE * 100,
                min=0,
                max=100,
                style={'width': '80px', 'backgroundColor': '#2a2a2a', 'border': '1px solid #404040', 'color': '#ffffff', 'padding': '8px'}
            )
        ], style={'minWidth': '100px'}),
        
        # 🔥 LIVE TRADING BUTONLARI (Paper trading butonları kaldırıldı)
        html.Button("🔄 Otomatik Başlat", id='btn-auto', n_clicks=0, className='control-button auto-button'),
        html.Button("🤖 Live Bot Başlat", id='btn-start-live-trading', n_clicks=0, className='control-button live-button'),
        html.Button("🛑 Live Bot Durdur", id='btn-stop-live-trading', n_clicks=0, className='control-button live-button'),
    ], style={
        'display': 'flex',
        'gap': '1rem',
        'alignItems': 'end',
        'justifyContent': 'center',
        'marginBottom': '2rem',
        'flexWrap': 'wrap'
    })


def create_status_section():
    """🔥 Durum gösterge bölümünü oluştur - API durumu eklendi"""
    return html.Div([
        html.Div([
            html.Span(id='trading-status', style={'fontSize': '1.1rem', 'fontWeight': 'bold'}),
            html.Br(),
            html.Div(id='api-status', style={'fontSize': '0.9rem', 'marginTop': '0.5rem'})
        ])
    ], style={
        'textAlign': 'center',
        'marginBottom': '1rem',
        'padding': '1rem',
        'backgroundColor': '#2a2a2a',
        'borderRadius': '6px',
        'border': '1px solid #404040'
    })


def create_performance_section():
    """Performans metrikleri bölümünü oluştur"""
    return html.Div([
        html.H3("📊 Live Trading Performance", style={'textAlign': 'center', 'margin': '0 0 1rem 0'}),
        html.Div(id='performance-metrics', style={'marginBottom': '2rem'})
    ])


def create_positions_table():
    """Açık pozisyonlar tablosunu oluştur"""
    return html.Div([
        html.H3("🤖 Açık Live Pozisyonlar", style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='positions-table',
            columns=[
                {"name": "Symbol", "id": "symbol"},
                {"name": "Side", "id": "side"},
                {"name": "Quantity", "id": "quantity", "type": "numeric", "format": {"specifier": ".6f"}},
                {"name": "Entry Price", "id": "entry_price", "type": "numeric", "format": {"specifier": ".6f"}},
                {"name": "Current Price", "id": "current_price", "type": "numeric", "format": {"specifier": ".6f"}},
                {"name": "Invested $", "id": "invested_amount", "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "Current Value $", "id": "current_value", "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "Unrealized P&L $", "id": "unrealized_pnl", "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "Stop Loss", "id": "stop_loss", "type": "numeric", "format": {"specifier": ".6f"}},
                {"name": "Take Profit", "id": "take_profit", "type": "numeric", "format": {"specifier": ".6f"}},
                {"name": "Score", "id": "ai_score", "type": "numeric", "format": {"specifier": ".0f"}},
            ],
            style_cell={'textAlign': 'center', 'padding': '10px', 'fontSize': '12px', 'backgroundColor': '#2a2a2a', 'color': '#ffffff', 'border': '1px solid #404040'},
            style_header={'backgroundColor': '#1a1a1a', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'column_id': 'unrealized_pnl', 'filter_query': '{unrealized_pnl} > 0'},
                    'backgroundColor': '#22c55e',
                    'color': '#000000',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'unrealized_pnl', 'filter_query': '{unrealized_pnl} < 0'},
                    'backgroundColor': '#ef4444',
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'side', 'filter_query': '{side} = LONG'},
                    'backgroundColor': '#059669',
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'side', 'filter_query': '{side} = SHORT'},
                    'backgroundColor': '#dc2626',
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                }
            ]
        )
    ], style={'marginBottom': '2rem'})


def create_warning_section():
    """Uyarı bölümünü oluştur - Eğitim amaçlı uyarısı"""
    return html.Div([
        html.P([
            "⚠️ ",
            html.Strong("UYARI: "),
            "Eğitim amaçlıdır, yatırım tavsiyesi değildir."
        ], style={
            'textAlign': 'center',
            'fontSize': '14px',
            'color': '#fbbf24',
            'margin': '20px 0',
            'padding': '15px',
            'backgroundColor': '#2a2a2a',
            'borderRadius': '6px',
            'border': '1px solid #f59e0b'
        })
    ])


def create_stats_panel():
    """İstatistik paneli bölümünü oluştur"""
    return html.Div(id='stats-panel', style={'marginBottom': '2rem'})


def create_status_text():
    """Durum metni bölümünü oluştur"""
    return html.Div([
        html.Span(id='status-text', style={'fontSize': '0.9rem', 'color': '#cccccc'})
    ], style={
        'textAlign': 'center',
        'marginBottom': '2rem',
        'padding': '1rem',
        'backgroundColor': '#2a2a2a',
        'borderRadius': '6px',
        'border': '1px solid #404040'
    })


def create_signals_table():
    """🔥 Sinyaller tablosunu oluştur - Live Bot sütunu eklendi"""
    return html.Div([
        html.H3("📊 TÜM SİNYALLER Paneli", style={
            'textAlign': 'center',
            'margin': '0 0 1rem 0',
            'fontSize': '1.3rem',
            'fontWeight': '600'
        }),
        dash_table.DataTable(
            id='signals-table',
            columns=[
                {"name": "Coin", "id": "symbol"},
                {"name": "Side", "id": "side"},
                {"name": "Ardışık(n)", "id": "run_count", 'type': 'numeric'},
                {"name": "Ardışık %", "id": "run_perc", 'type': 'numeric', 'format': {"specifier": ".2f"}},
                {"name": "Gauss(n)", "id": "gauss_run", 'type': 'numeric', 'format': {"specifier": ".1f"}},
                {"name": "Gauss(%)", "id": "gauss_run_perc", 'type': 'numeric', 'format': {"specifier": ".1f"}},
                {"name": "Vol/SMA", "id": "vol_ratio", 'type': 'numeric', 'format': {"specifier": ".2f"}},
                {"name": "HH Vol", "id": "hh_vol_streak", 'type': 'numeric'},
                {"name": "Deviso %", "id": "deviso_ratio", 'type': 'numeric', 'format': {"specifier": ".2f"}},
                {"name": "AI Skor", "id": "ai_score"},  
                {"name": "🤖", "id": "live_status"},  # 🔥 YENİ SÜTUN: Live bot durumu
                {"name": "Time", "id": "timestamp"}
            ],
            data=[],
            page_size=20,
            sort_action='native',
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': '#1a1a1a',
                'color': '#ffffff',
                'fontWeight': '600',
                'fontSize': '0.85rem',
                'textAlign': 'center',
                'border': '1px solid #404040'
            },
            style_cell={
                'backgroundColor': '#2a2a2a',
                'color': '#ffffff',
                'border': '1px solid #404040',
                'textAlign': 'center',
                'fontSize': '0.8rem',
                'padding': '6px'
            },
            style_data_conditional=[
                # Side renklendirme
                {
                    'if': {'column_id': 'side', 'filter_query': '{side} = LONG'},
                    'backgroundColor': '#059669',
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'side', 'filter_query': '{side} = SHORT'},
                    'backgroundColor': '#dc2626',
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                },
                # Deviso ratio renklendirme
                {
                    'if': {'column_id': 'deviso_ratio', 'filter_query': '{deviso_ratio} > 0'},
                    'backgroundColor': '#059669',
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'deviso_ratio', 'filter_query': '{deviso_ratio} < 0'},
                    'backgroundColor': '#dc2626',
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                },
                # 🔥 Live bot status renklendirme
                {
                    'if': {'column_id': 'live_status', 'filter_query': '{live_status} contains ✅'},
                    'backgroundColor': '#22c55e',
                    'color': '#000000',
                    'fontWeight': 'bold'
                }
            ]
        )
    ], style={'marginBottom': '2rem'})


def create_trades_table():
    """🔥 Live Trading geçmişi tablosunu oluştur"""
    return html.Div([
        html.H3("📋 Live Trading Geçmişi", style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='trades-table',
            columns=[
                {"name": "Time", "id": "timestamp"},
                {"name": "Symbol", "id": "symbol"},
                {"name": "Side", "id": "side"},
                {"name": "Quantity", "id": "quantity", "type": "numeric", "format": {"specifier": ".6f"}},
                {"name": "Entry Price", "id": "entry_price", "type": "numeric", "format": {"specifier": ".6f"}},
                {"name": "Exit Price", "id": "exit_price", "type": "numeric", "format": {"specifier": ".6f"}},
                {"name": "P&L $", "id": "pnl", "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "AI Score", "id": "ai_score", "type": "numeric", "format": {"specifier": ".0f"}},
                {"name": "Close Reason", "id": "close_reason"},
                {"name": "Status", "id": "status"}
            ],
            data=[],
            page_size=20,
            sort_action='native',
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': '#1a1a1a',
                'color': '#ffffff',
                'fontWeight': '600',
                'fontSize': '0.85rem',
                'textAlign': 'center',
                'border': '1px solid #404040'
            },
            style_cell={
                'backgroundColor': '#2a2a2a',
                'color': '#ffffff',
                'border': '1px solid #404040',
                'textAlign': 'center',
                'fontSize': '0.8rem',
                'padding': '6px'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'pnl', 'filter_query': '{pnl} > 0'},
                    'backgroundColor': '#22c55e',
                    'color': '#000000',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'pnl', 'filter_query': '{pnl} < 0'},
                    'backgroundColor': '#ef4444',
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'status', 'filter_query': '{status} = OPEN'},
                    'backgroundColor': '#3b82f6',
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'status', 'filter_query': '{status} = CLOSED'},
                    'backgroundColor': '#6b7280',
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                }
            ]
        )
    ])


def create_interval_component():
    """Interval component'i oluştur"""
    return dcc.Interval(
        id='interval-component',
        interval=TABLE_REFRESH_INTERVAL,  
        n_intervals=0
    )


# 🔥 YENİ FONKSİYON: Live trading için sinyal durumu göster
def get_live_signal_status(symbol):
    """Sinyalin live trading'de kullanılıp kullanılmadığını göster"""
    # Bu fonksiyon live_trader.py'dan gelecek veriyle çalışacak
    # Şimdilik placeholder
    return "⭐"  # ⭐ = Beklemede, ✅ = Açık pozisyon, ❌ = Reddedildi


# 🔥 YENİ FONKSİYON: API bağlantı durumu göster  
def create_api_connection_indicator():
    """API bağlantı durumu göstergesi"""
    return html.Div([
        html.H4("🔗 API Bağlantı Durumu", style={'textAlign': 'center', 'margin': '0 0 1rem 0'}),
        html.Div(id='api-connection-status', style={'textAlign': 'center'})
    ], style={'marginBottom': '2rem'})


def create_layout():
    """🔥 Ana layout'u oluştur - Live trading güncellemeleri"""
    return html.Div([
        # Header
        create_header(),
        
        # Kontrol paneli
        create_control_panel(),
        
        # Trading durumu + API durumu
        create_status_section(),
        
        # API bağlantı göstergesi
        create_api_connection_indicator(),
        
        # Performans metrikleri
        create_performance_section(),
        
        # Açık pozisyonlar
        create_positions_table(),

        # Uyarı
        create_warning_section(),
        
        # İstatistik paneli
        create_stats_panel(),
        
        # Durum metni
        create_status_text(),

        # Sinyaller tablosu
        create_signals_table(),
        
        # Live Trading geçmişi
        create_trades_table(),

        # Interval Component
        create_interval_component()
        
    ], style={
        'maxWidth': '1400px',
        'margin': '0 auto',
        'padding': '0 1rem 2rem 1rem',
        'backgroundColor': '#1a1a1a',
        'minHeight': '100vh'
    })


def get_app_styles():
    """🔥 CSS stillerini döndür - Live trading buton stilleri eklendi"""
    return """
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
    """