"""
🎨 UI Bileşenleri - ULTRA PANEL v5
Dash arayüz bileşenleri ve layout fonksiyonları
🔥 YENİ: HTF Count, Power, Whale göstergeleri
"""

from dash import html, dcc, dash_table
from config import (
    DEFAULT_TIMEFRAME, DEFAULT_MIN_AI_SCORE, TABLE_REFRESH_INTERVAL
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


def create_htf_indicator(htf_count):
    """🔥 YENİ: HTF Count göstergesi (3/4 veya 4/4)"""
    try:
        htf_count = int(htf_count)
        
        if htf_count == 4:
            return "🔥 4/4"  # Mükemmel
        elif htf_count == 3:
            return "✅ 3/4"  # İyi
        elif htf_count == 2:
            return "⚠️ 2/4"  # Zayıf
        else:
            return f"❌ {htf_count}/4"  # Çok zayıf
    except:
        return "N/A"


def create_power_indicator(power):
    """🔥 YENİ: Power göstergesi"""
    try:
        power = float(power)
        
        if power >= 30:
            return f"🔥 {power:.1f}"  # Çok yüksek
        elif power >= 20:
            return f"💪 {power:.1f}"  # Yüksek
        elif power >= 10:
            return f"⚡ {power:.1f}"  # Orta
        elif power >= 5:
            return f"📊 {power:.1f}"  # Düşük
        else:
            return f"📉 {power:.1f}"  # Çok düşük
    except:
        return "N/A"


def create_whale_indicator(whale_active):
    """🔥 YENİ: Whale göstergesi"""
    try:
        if whale_active:
            return "🐋 YES"
        else:
            return "—"
    except:
        return "—"


def create_header():
    """Header bölümünü oluştur"""
    return html.Div([
        html.H1("🔥 ULTRA PANEL v5 - AI Crypto Analytics + Live Trading", style={
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
    """🔥 ULTRA PANEL v5 - Kontrol paneli"""
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
            html.Label("Min HTF Count"),
            dcc.Input(
                id='inp-min-htf',
                type='number',
                value=3,
                min=0,
                max=4,
                style={'width': '80px', 'backgroundColor': '#2a2a2a', 'border': '1px solid #404040', 'color': '#ffffff', 'padding': '8px'}
            )
        ], style={'minWidth': '120px'}),
        
        html.Div([
            html.Label("Min Power"),
            dcc.Input(
                id='inp-min-power',
                type='number',
                value=5.0,
                step=1.0,
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
        
        html.Div([
            html.Label("Whale Only"),
            dcc.Checklist(
                id='chk-whale-only',
                options=[{'label': '🐋', 'value': 'whale'}],
                value=[],
                style={'color': '#ffffff', 'marginTop': '8px'}
            )
        ], style={'minWidth': '100px'}),
        
        # 🔥 LIVE TRADING BUTONLARI
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
    """Durum gösterge bölümü"""
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
    """Performans metrikleri bölümü"""
    return html.Div([
        html.H3("📊 Live Trading Performance", style={'textAlign': 'center', 'margin': '0 0 1rem 0'}),
        html.Div(id='performance-metrics', style={'marginBottom': '2rem'})
    ])


def create_positions_table():
    """Açık pozisyonlar tablosu"""
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
                {"name": "SL/TP Mode", "id": "sltp_mode"}
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
                },
                {
                    'if': {'column_id': 'sltp_mode', 'filter_query': '{sltp_mode} contains Auto'},
                    'backgroundColor': '#10b981',
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'sltp_mode', 'filter_query': '{sltp_mode} contains Manual'},
                    'backgroundColor': '#6b7280',
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                }
            ]
        )
    ], style={'marginBottom': '2rem'})


def create_warning_section():
    """Uyarı bölümü"""
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
    """İstatistik paneli"""
    return html.Div(id='stats-panel', style={'marginBottom': '2rem'})


def create_status_text():
    """Durum metni"""
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
    """🔥 ULTRA PANEL v5 - Sinyaller tablosu"""
    return html.Div([
        html.H3("🔥 ULTRA PANEL v5 - SİNYALLER", style={
            'textAlign': 'center',
            'margin': '0 0 1rem 0',
            'fontSize': '1.3rem',
            'fontWeight': '600'
        }),
        dash_table.DataTable(
            id='signals-table',
            columns=[
                {"name": "Coin", "id": "symbol"},
                {"name": "🔥 ULTRA", "id": "ultra_signal"},  # 🔥 YENİ
                {"name": "HTF", "id": "htf_count"},  # 🔥 YENİ
                {"name": "⚡ POWER", "id": "total_power", 'type': 'numeric', 'format': {"specifier": ".1f"}},  # 🔥 YENİ
                {"name": "🐋 WHALE", "id": "whale_active"},  # 🔥 YENİ
                {"name": "AI Skor", "id": "ai_score"},
                {"name": "🤖", "id": "live_status"},
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
                # 🔥 ULTRA SIGNAL renklendirme
                {
                    'if': {'column_id': 'ultra_signal', 'filter_query': '{ultra_signal} = BUY'},
                    'backgroundColor': '#059669',
                    'color': '#ffffff',
                    'fontWeight': 'bold',
                    'fontSize': '0.9rem'
                },
                {
                    'if': {'column_id': 'ultra_signal', 'filter_query': '{ultra_signal} = SELL'},
                    'backgroundColor': '#dc2626',
                    'color': '#ffffff',
                    'fontWeight': 'bold',
                    'fontSize': '0.9rem'
                },
                # 🔥 HTF COUNT renklendirme
                {
                    'if': {'column_id': 'htf_count', 'filter_query': '{htf_count} contains 4/4'},
                    'backgroundColor': '#f59e0b',
                    'color': '#000000',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'htf_count', 'filter_query': '{htf_count} contains 3/4'},
                    'backgroundColor': '#eab308',
                    'color': '#000000',
                    'fontWeight': 'bold'
                },
                # 🔥 POWER renklendirme
                {
                    'if': {'column_id': 'total_power', 'filter_query': '{total_power} > 30'},
                    'backgroundColor': '#dc2626',  # Çok yüksek
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'total_power', 'filter_query': '{total_power} > 20'},
                    'backgroundColor': '#f59e0b',  # Yüksek
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'total_power', 'filter_query': '{total_power} > 10'},
                    'backgroundColor': '#eab308',  # Orta
                    'color': '#000000',
                    'fontWeight': 'bold'
                },
                # 🔥 WHALE renklendirme
                {
                    'if': {'column_id': 'whale_active', 'filter_query': '{whale_active} contains YES'},
                    'backgroundColor': '#06b6d4',  # Cyan
                    'color': '#ffffff',
                    'fontWeight': 'bold'
                },
                # Live bot status
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
    """Live Trading geçmişi tablosu"""
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
    """Interval component"""
    return dcc.Interval(
        id='interval-component',
        interval=TABLE_REFRESH_INTERVAL,  
        n_intervals=0
    )


def create_api_connection_indicator():
    """API bağlantı durumu göstergesi"""
    return html.Div([
        html.H4("🔗 API Bağlantı Durumu", style={'textAlign': 'center', 'margin': '0 0 1rem 0'}),
        html.Div(id='api-connection-status', style={'textAlign': 'center'})
    ], style={'marginBottom': '2rem'})


def create_layout():
    """🔥 ULTRA PANEL v5 - Ana layout"""
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

        # 🔥 ULTRA PANEL v5 Sinyaller tablosu
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
    """CSS stilleri"""
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