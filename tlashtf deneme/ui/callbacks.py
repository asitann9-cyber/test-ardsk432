"""
🔄 UI Callback Fonksiyonları - TAMAMEN GÜNCELLENMİŞ
Dash callback'leri ve interaktif özellikler
🔥 YENİ: Z-Score filtreleri + Live Trading entegrasyonu
🔧 DÜZELTME: Config uyumluluğu + Live bot fonksiyonları
🔥 Z-SCORE DÜZELTMESİ: Sadece max_zscore kullanılıyor, karmaşık hesaplama kaldırıldı
🔥 YENİ: VPM (Volume-Price-Momentum) sütunu eklendi
"""

from datetime import datetime
import pandas as pd
import logging

from dash import html
from dash.dependencies import Input, Output, State
import dash

import config
from config import (
    LOCAL_TZ, DEFAULT_TIMEFRAME, DEFAULT_MIN_STREAK, DEFAULT_MIN_PCT, 
    DEFAULT_MIN_VOLR, DEFAULT_MIN_AI_SCORE, MAX_OPEN_POSITIONS
)
from ui.components import create_ai_score_bar, create_zscore_indicator
from data.fetch_data import get_current_price
from data.database import load_trades_from_csv, calculate_performance_metrics

# 🔥 LIVE TRADING IMPORT - GÜVENLI
try:
    from trading.live_trader import (
        start_live_trading, stop_live_trading, is_live_trading_active,
        get_live_trading_status, get_live_bot_status_for_symbol, get_auto_sltp_count
    )
    LIVE_TRADER_AVAILABLE = True
except ImportError:
    LIVE_TRADER_AVAILABLE = False

logger = logging.getLogger("crypto-analytics")


def register_callbacks(app, auto_scan_control):
    """
    🔥 TAMAMEN GÜNCELLENMİŞ: Tüm callback'leri kaydet
    
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
        [Input('btn-start-live-trading', 'n_clicks'),
         Input('btn-stop-live-trading', 'n_clicks')]
    )
    def control_live_trading(start_clicks, stop_clicks):
        """🔥 Live Trading kontrolü - Paper trading kaldırıldı"""
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
        """🔥 GÜNCELLEME: Config entegre performans metrikleri"""
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
        """🔥 DÜZELTME: Config tabanlı trading tabloları"""
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
         State('inp-min-volr', 'value'),
         State('inp-min-ai', 'value'),
         State('inp-max-zscore', 'value')]  # 🔥 YENİ: Z-Score filtresi
    )
    def update_signals(n_intervals, tf, min_streak, min_pct, min_volr, min_ai_pct, max_zscore):  
        """🔥 YENİ: Z-Score filtresi dahil sinyal güncelleme - BASITLEŞTIRILDI + VPM"""
        tf = tf or DEFAULT_TIMEFRAME
        min_streak = int(min_streak or DEFAULT_MIN_STREAK)
        min_pct = float(min_pct or DEFAULT_MIN_PCT)
        min_volr = float(min_volr or DEFAULT_MIN_VOLR)
        min_ai_score = float(min_ai_pct or 30) / 100
        max_zscore = float(max_zscore or 5.0)  # 🔥 YENİ
        
        config.current_settings.update({
            'timeframe': tf,
            'min_streak': min_streak,
            'min_pct': min_pct,
            'min_volr': min_volr,
            'min_ai': min_ai_pct,
            'max_zscore': max_zscore  # 🔥 YENİ
        })
        
        # 🔥 DÜZELTME: config.current_data kullan
        if config.current_data is None or config.current_data.empty:
            status_text = "Otomatik tarama aktif - Veri bekleniyor..." if auto_scan_control['is_active']() else "Otomatik tarama başlatın"
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

        # 🔥 vol_ratio yerine log_volume kullan
        if min_volr > 3.0:
            df = df[(df['log_volume'].isna()) | (df['log_volume'] >= min_volr)]
        
        # 🔥 YENİ: Z-Score filtresi - MUTLAK DEĞER KONTROLÜ
        if max_zscore < 5.0:
            if 'max_zscore' in df.columns:
                df = df[abs(df['max_zscore']) <= max_zscore]
        
        # 🔧 Sıralama sütunları - VPM eklendi
        sort_columns = ['ai_score']
        if 'vpm_score' in df.columns:
            sort_columns.append('vpm_score')  # 🔥 YENİ: VPM sıralamada
        if 'trend_strength' in df.columns:
            sort_columns.append('trend_strength')
        sort_columns.extend(['run_perc', 'gauss_run', 'log_volume_momentum'])
        
        df = df.sort_values(by=sort_columns, ascending=[False] * len(sort_columns))
        
        # 🔥 YENİ LOG: Live Bot ile karşılaştırma için
        logger.info(f"📊 UI'nin gösterdiği ilk 3: {df.head(3)['symbol'].tolist()}")
        
        if df.empty:
            status_text = f"Filtreler çok sıkı! {original_count} sinyal var."
            if auto_scan_control['is_active']():
                status_text += f" | Otomatik aktif"
            return [], status_text, []
        
        table_data = []
        for _, row in df.iterrows():
            ai_score_unicode = create_ai_score_bar(row['ai_score'])
            
            # 🔥 BASİTLEŞTİRİLDİ: Direkt max_zscore kullan
            zscore_indicator = create_zscore_indicator(row.get('max_zscore', 0.0))
            
            # 🔥 YENİ: VPM gösterimi
            vpm_value = row.get('vpm_score', 0.0)
            vpm_display = f"{vpm_value:+.1f}"  # +25.3 veya -15.2 gibi
            
            # 🔥 YENİ: Live bot durumu
            live_status = "⭐"
            if LIVE_TRADER_AVAILABLE:
                live_status = get_live_bot_status_for_symbol(row['symbol'])
            
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
                'vpm_score': vpm_display,  # 🔥 YENİ SÜTUN
                'c_signal_momentum': row.get('c_signal_momentum', 0.0),
                'max_zscore': zscore_indicator,  # 🔥 BASİTLEŞTİRİLDİ
                'ai_score': ai_score_unicode,
                'live_status': live_status,
                'timestamp': datetime.now(LOCAL_TZ).strftime('%H:%M')
            })
        
        long_count = len(df[df['run_type'] == 'long'])
        short_count = len(df[df['run_type'] == 'short'])
        
        # 🔥 YENİ: Z-Score istatistikleri - MUTLAK DEĞER
        high_zscore_count = 0
        if 'max_zscore' in df.columns:
            high_zscore_count = len(df[abs(df['max_zscore']) >= 2.0])
        
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
            # 🔥 YENİ: Z-Score istatistiği
            html.Div([
                html.Div(f"{high_zscore_count}", className="stat-value", style={'color': '#f59e0b'}),  
                html.Div("Yüksek Z-Score", className="stat-label")
            ], className="stat-card"),
        ], className="stats-grid")
        
        current_positions = config.live_positions if config.is_live_mode() else config.paper_positions
        current_mode = "Live" if config.is_live_mode() else "Paper"
        
        status = f"{len(df)}/{original_count} SİNYAL: (L:{long_count}, S:{short_count}) | {datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}"
        if auto_scan_control['is_active']():
            status += f" | 🔄 Otomatik aktif"
        
        # 🔥 YENİ: Z-Score uyarı bilgisi ekle
        if high_zscore_count > 0:
            status += f" | ⚠️ Yüksek Z-Score: {high_zscore_count}"
        
        if LIVE_TRADER_AVAILABLE and is_live_trading_active():
            auto_sltp_count = get_auto_sltp_count() if LIVE_TRADER_AVAILABLE else 0
            status += f" | 🤖 {current_mode} Bot aktif ({len(current_positions)}/{MAX_OPEN_POSITIONS}) [Auto SL/TP: {auto_sltp_count}]"
        elif len(current_positions) > 0:
            status += f" | 📝 {current_mode} Trading ({len(current_positions)}/{MAX_OPEN_POSITIONS})"
        
        status += f" {top_3_info}"
        
        return table_data, status, stats_panel


def get_callback_functions():
    """Callback fonksiyonlarını al (standalone kullanım için)"""
    return {
        'register_callbacks': register_callbacks
    }