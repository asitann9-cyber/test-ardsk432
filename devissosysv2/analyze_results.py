import sqlite3
import pandas as pd
from datetime import datetime

def analyze_live_test_results():
    """Live test sonuçlarını analiz et"""
    try:
        conn = sqlite3.connect('live_test.db')
        
        # Genel istatistikler
        print("=== LIVE TEST SONUÇLARI ANALİZİ ===")
        print()
        
        # Tamamlanan işlemler
        df_completed = pd.read_sql_query('SELECT * FROM completed_trades', conn)
        
        if len(df_completed) == 0:
            print("❌ Tamamlanan işlem bulunamadı!")
            return
        
        print(f"📊 Toplam İşlem Sayısı: {len(df_completed)}")
        
        # Kazanan/Kaybeden işlemler
        winning_trades = df_completed[df_completed['pnl'] > 0]
        losing_trades = df_completed[df_completed['pnl'] < 0]
        
        win_rate = (len(winning_trades) / len(df_completed)) * 100
        
        print(f"✅ Kazanan İşlemler: {len(winning_trades)}")
        print(f"❌ Kaybeden İşlemler: {len(losing_trades)}")
        print(f"📈 Win Rate: {win_rate:.2f}%")
        print()
        
        # PnL analizi
        total_pnl = df_completed['pnl'].sum()
        avg_win = winning_trades['pnl_percentage'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_percentage'].mean() if len(losing_trades) > 0 else 0
        
        print(f"💰 Toplam PnL: {total_pnl:.2f} USDT")
        print(f"📈 Ortalama Kazanç: {avg_win:.2f}%")
        print(f"📉 Ortalama Kayıp: {avg_loss:.2f}%")
        print()
        
        # Sinyal kategorilerine göre analiz
        print("=== SİNYAL KATEGORİLERİNE GÖRE ANALİZ ===")
        category_stats = df_completed.groupby('signal_category').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_percentage': 'mean'
        }).round(2)
        
        print(category_stats)
        print()
        
        # Çıkış nedenlerine göre analiz
        print("=== ÇIKIŞ NEDENLERİNE GÖRE ANALİZ ===")
        exit_reason_stats = df_completed.groupby('exit_reason').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_percentage': 'mean'
        }).round(2)
        
        print(exit_reason_stats)
        print()
        
        # En iyi ve en kötü işlemler
        print("=== EN İYİ 5 İŞLEM ===")
        best_trades = df_completed.nlargest(5, 'pnl')[['symbol', 'signal_type', 'signal_category', 'pnl', 'pnl_percentage', 'exit_reason']]
        print(best_trades)
        print()
        
        print("=== EN KÖTÜ 5 İŞLEM ===")
        worst_trades = df_completed.nsmallest(5, 'pnl')[['symbol', 'signal_type', 'signal_category', 'pnl', 'pnl_percentage', 'exit_reason']]
        print(worst_trades)
        print()
        
        # Sinyal skorlarına göre analiz
        print("=== SİNYAL SKORLARINA GÖRE ANALİZ ===")
        df_completed['score_range'] = pd.cut(df_completed['signal_score'], 
                                           bins=[0, 8, 12, 15, 25], 
                                           labels=['Düşük (0-8)', 'Orta (8-12)', 'Yüksek (12-15)', 'Çok Yüksek (15+)'])
        
        score_stats = df_completed.groupby('score_range').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_percentage': 'mean'
        }).round(2)
        
        print(score_stats)
        print()
        
        # Zaman analizi
        print("=== ZAMAN ANALİZİ ===")
        df_completed['entry_time'] = pd.to_datetime(df_completed['entry_time'])
        df_completed['hour'] = df_completed['entry_time'].dt.hour
        
        hourly_stats = df_completed.groupby('hour').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_percentage': 'mean'
        }).round(2)
        
        print("Saatlik Performans:")
        print(hourly_stats)
        print()
        
        # İşlem süreleri analizi
        print("=== İŞLEM SÜRELERİ ANALİZİ ===")
        duration_stats = df_completed.groupby(pd.cut(df_completed['duration_minutes'], 
                                                   bins=[0, 30, 60, 120, 240, 1000], 
                                                   labels=['0-30dk', '30-60dk', '1-2sa', '2-4sa', '4sa+'])).agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_percentage': 'mean'
        }).round(2)
        
        print(duration_stats)
        print()
        
        # Problem tespiti
        print("=== PROBLEM TESPİTİ ===")
        
        if win_rate < 40:
            print(f"⚠️ Win rate çok düşük: {win_rate:.2f}%")
            print("   - Sinyal kalitesi artırılmalı")
            print("   - Giriş kriterleri sıkılaştırılmalı")
            print("   - Stop loss seviyeleri gözden geçirilmeli")
        
        if avg_loss > abs(avg_win):
            print(f"⚠️ Ortalama kayıp, ortalama kazancı aşıyor: {avg_loss:.2f}% vs {avg_win:.2f}%")
            print("   - Risk/ödül oranı iyileştirilmeli")
            print("   - Take profit seviyeleri artırılmalı")
            print("   - Stop loss seviyeleri sıkılaştırılmalı")
        
        # En çok kayıp veren kategoriler
        losing_categories = df_completed[df_completed['pnl'] < 0].groupby('signal_category')['pnl'].sum().sort_values()
        if len(losing_categories) > 0:
            print(f"📉 En çok kayıp veren kategori: {losing_categories.index[0]} ({losing_categories.iloc[0]:.2f} USDT)")
        
        # En çok kayıp veren çıkış nedeni
        losing_reasons = df_completed[df_completed['pnl'] < 0].groupby('exit_reason')['pnl'].sum().sort_values()
        if len(losing_reasons) > 0:
            print(f"📉 En çok kayıp veren çıkış nedeni: {losing_reasons.index[0]} ({losing_reasons.iloc[0]:.2f} USDT)")
        
        conn.close()
        
    except Exception as e:
        print(f"Analiz hatası: {e}")

if __name__ == "__main__":
    analyze_live_test_results()
