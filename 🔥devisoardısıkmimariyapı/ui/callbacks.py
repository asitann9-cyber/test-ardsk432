"""
🔄 UI Callback Fonksiyonları
Dash callback'leri ve interaktif özellikler
"""

from datetime import datetime
import pandas as pd
import logging

from dash import html
from dash.dependencies import Input, Output, State
import dash

from config import (
    LOCAL_TZ, DEFAULT_TIMEFRAME, DEFAULT_MIN_STREAK, DEFAULT_MIN_PCT, 
    DEFAULT_MIN_VOLR, DEFAULT_MIN_AI_SCORE, MAX_OPEN_POSITIONS,
    open_positions, current_data, current_settings
)
from ui.components import create_ai_score_bar
from trading.paper_trader import (
    start_paper_trading, stop_paper_trading, is_trading_active
)
from data.fetch_data import get_current_price
from data.database import load_trades_from_csv, calculate_performance_metrics

logger = logging.getLogger("crypto-analytics")


def register_callbacks(app, auto_scan_control):
    """
    Tüm callback'leri kaydet
    
    Args:
        app: Dash uygulaması
        auto_scan_control: Otomatik tarama kontrol fonksiyonları (dict)
    """
    
    @app.callback(
        [Output('btn-auto', 'children'),
         Output('btn-auto', 'className')],
        Input('btn-auto', 'n_clicks')
    )
    def toggle_auto_scan(n_clicks):
        """Otomatik tarama butonunu kontrol et"""
        if n_clicks > 0:
            if auto_scan_control['is_active']():
                auto_scan_control['stop']()
                return "🔄 Otomatik Başlat", "control-button auto-button"
            else:
                auto_scan_control['start']()
                return "⛔ Otomatik Durdur", "control-button auto-button active"
        
        return "🔄 Otomatik Başlat", "control-button auto-button"

    
    @app.callback(
        Output('trading-status', 'children'),
        [Input('btn-start-trading', 'n_clicks'),
         Input('btn-stop-trading', 'n_clicks')]
    )
    def control_paper_trading(start_clicks, stop_clicks):
        """Paper Trading kontrolü"""
        ctx = dash.callback_context
        
        if not ctx.triggered:
            status = "🔴 Paper Trading Durduruldu" if not is_trading_active() else "🟢 Paper Trading Aktif"
            return status
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'btn-start-trading' and start_clicks > 0:
            success = start_paper_trading()
            if success:
                return "🟢 Paper Trading Başarıyla Başlatıldı ✅ (Komisyonsuz)"
            else:
                return "❌ Paper Trading Başlatılamadı"
        
        elif trigger_id == 'btn-stop-trading' and stop_clicks > 0:
            stop_paper_trading()
            return "🔴 Paper Trading Durduruldu - Tüm pozisyonlar kapatıldı (Komisyonsuz)"
        
        return "🔴 Paper Trading Durduruldu" if not is_trading_active() else "🟢 Paper Trading Aktif"

    
    @app.callback(
        Output('performance-metrics', 'children'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_performance_metrics(n_intervals):
        """Performans metriklerini güncelle"""
        try:
            metrics = calculate_performance_metrics()
            
            return html.Div([
                html.Div([
                    html.H4("💰 Mevcut Sermaye", style={'margin': '0', 'fontSize': '14px', 'color': '#22c55e'}),
                    html.H2(f"${metrics['current_capital']:.2f}", style={'color': '#22c55e', 'margin': '5px 0', 'fontSize': '18px'})
                ], style={
                    'width': '19%', 'display': 'inline-block', 'textAlign': 'center', 
                    'margin': '0.5%', 'padding': '15px', 'border': '2px solid #22c55e', 
                    'borderRadius': '10px', 'backgroundColor': '#f8fff8'
                }),
                
                html.Div([
                    html.H4("📈 Efektif Sermaye", style={'margin': '0', 'fontSize': '14px', 'color': '#3b82f6'}),
                    html.H2(f"${metrics['effective_capital']:.2f}", style={'color': '#3b82f6', 'margin': '5px 0', 'fontSize': '18px'})
                ], style={
                    'width': '19%', 'display': 'inline-block', 'textAlign': 'center', 
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
                    'width': '19%', 'display': 'inline-block', 'textAlign': 'center', 
                    'margin': '0.5%', 'padding': '15px', 'border': '2px solid #8b5cf6', 
                    'borderRadius': '10px', 'backgroundColor': '#faf5ff'
                }),
                
                html.Div([
                    html.H4("📋 Toplam İşlem", style={'margin': '0', 'fontSize': '14px', 'color': '#6b7280'}),
                    html.H2(f"{metrics['total_trades']}", style={'color': '#6b7280', 'margin': '5px 0', 'fontSize': '18px'})
                ], style={
                    'width': '19%', 'display': 'inline-block', 'textAlign': 'center', 
                    'margin': '0.5%', 'padding': '15px', 'border': '2px solid #6b7280', 
                    'borderRadius': '10px', 'backgroundColor': '#f9fafb'
                }),
                
                html.Div([
                    html.H4("🏆 Kazanma Oranı", style={'margin': '0', 'fontSize': '14px', 'color': '#10b981'}),
                    html.H2(f"{metrics['win_rate']:.1f}%", style={'color': '#10b981', 'margin': '5px 0', 'fontSize': '18px'})
                ], style={
                    'width': '19%', 'display': 'inline-block', 'textAlign': 'center', 
                    'margin': '0.5%', 'padding': '15px', 'border': '2px solid #10b981', 
                    'borderRadius': '10px', 'backgroundColor': '#f0fdf4'
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
        """Sinyalleri ve istatistikleri güncelle"""
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
            status_text = "Otomatik tarama aktif - Veri bekleniyor..." if auto_scan_control['is_active']() else "Otomatik tarama başlatın"
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
            if auto_scan_control['is_active']():
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
        if auto_scan_control['is_active']():
            status += f" | 🔄 Otomatik aktif"
        if is_trading_active():
            status += f" | 💰 Trading aktif ({len(open_positions)}/{MAX_OPEN_POSITIONS})"
        status += f" {top_3_info}"
        
        return table_data, status, stats_panel


def get_callback_functions():
    """Callback fonksiyonlarını al (standalone kullanım için)"""
    return {
        'register_callbacks': register_callbacks
    }