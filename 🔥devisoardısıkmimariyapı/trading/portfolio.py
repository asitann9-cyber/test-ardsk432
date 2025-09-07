"""
📊 Portföy Yönetimi Modülü
Pozisyon analizi, risk yönetimi ve performans takibi
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

from config import (
    LOCAL_TZ, INITIAL_CAPITAL, MAX_OPEN_POSITIONS, 
    open_positions, current_capital
)
from data.fetch_data import get_current_price
from data.database import load_trades_from_csv, load_capital_history_from_csv

logger = logging.getLogger("crypto-analytics")


class PortfolioManager:
    """Portföy yönetimi ve analiz sınıfı"""
    
    def __init__(self):
        self.initial_capital = INITIAL_CAPITAL
        self.max_positions = MAX_OPEN_POSITIONS
    
    def get_current_portfolio_status(self) -> Dict:
        """
        Mevcut portföy durumunu al
        
        Returns:
            Dict: Portföy durum bilgileri
        """
        global current_capital, open_positions
        
        portfolio = {
            'timestamp': datetime.now(LOCAL_TZ).isoformat(),
            'cash': current_capital,
            'positions_count': len(open_positions),
            'max_positions': self.max_positions,
            'available_slots': self.max_positions - len(open_positions),
            'total_invested': 0,
            'total_market_value': 0,
            'total_unrealized_pnl': 0,
            'positions': []
        }
        
        for symbol, position in open_positions.items():
            current_price = get_current_price(symbol)
            if current_price is None:
                current_price = position['entry_price']
            
            market_value = position['quantity'] * current_price
            
            # P&L hesaplama
            if position['side'] == 'LONG':
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
            
            pnl_percentage = (unrealized_pnl / position['invested_amount'] * 100) if position['invested_amount'] > 0 else 0
            
            position_info = {
                'symbol': symbol,
                'side': position['side'],
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'current_price': current_price,
                'invested_amount': position['invested_amount'],
                'market_value': market_value,
                'unrealized_pnl': unrealized_pnl,
                'pnl_percentage': pnl_percentage,
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'entry_time': position['entry_time'].isoformat(),
                'ai_score': position['signal_data']['ai_score'],
                'days_held': (datetime.now(LOCAL_TZ) - position['entry_time']).days
            }
            
            portfolio['positions'].append(position_info)
            portfolio['total_invested'] += position['invested_amount']
            portfolio['total_market_value'] += market_value
            portfolio['total_unrealized_pnl'] += unrealized_pnl
        
        # Toplam portföy değeri
        portfolio['total_portfolio_value'] = portfolio['cash'] + portfolio['total_market_value']
        portfolio['total_return'] = portfolio['total_portfolio_value'] - self.initial_capital
        portfolio['total_return_pct'] = (portfolio['total_return'] / self.initial_capital * 100) if self.initial_capital > 0 else 0
        
        return portfolio
    
    def calculate_risk_metrics(self) -> Dict:
        """
        Risk metriklerini hesapla
        
        Returns:
            Dict: Risk analizi
        """
        portfolio = self.get_current_portfolio_status()
        trades_df = load_trades_from_csv()
        
        risk_metrics = {
            'portfolio_concentration': 0,
            'max_position_risk': 0,
            'total_exposure': 0,
            'cash_ratio': 0,
            'volatility': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'win_rate': 0,
            'avg_holding_period': 0
        }
        
        try:
            # Portföy konsantrasyonu
            if portfolio['total_portfolio_value'] > 0:
                risk_metrics['cash_ratio'] = portfolio['cash'] / portfolio['total_portfolio_value'] * 100
                risk_metrics['total_exposure'] = portfolio['total_invested'] / portfolio['total_portfolio_value'] * 100
                
                # En büyük pozisyon riski
                if portfolio['positions']:
                    max_position_value = max(pos['invested_amount'] for pos in portfolio['positions'])
                    risk_metrics['max_position_risk'] = max_position_value / portfolio['total_portfolio_value'] * 100
                    
                    # Konsantrasyon riski (Herfindahl Index)
                    weights_squared = sum((pos['invested_amount'] / portfolio['total_invested'])**2 
                                        for pos in portfolio['positions'] if portfolio['total_invested'] > 0)
                    risk_metrics['portfolio_concentration'] = weights_squared * 100
            
            # Trade bazlı metrikler
            if not trades_df.empty:
                closed_trades = trades_df[trades_df['status'] == 'CLOSED']
                
                if not closed_trades.empty:
                    # Kazanma oranı
                    winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
                    risk_metrics['win_rate'] = (winning_trades / len(closed_trades)) * 100
                    
                    # Ortalama tutma süresi (eğer timestamp varsa)
                    if 'timestamp' in closed_trades.columns:
                        try:
                            closed_trades['timestamp'] = pd.to_datetime(closed_trades['timestamp'])
                            # Basitleştirilmiş holding period (1 gün varsayımı)
                            risk_metrics['avg_holding_period'] = 1.0  # Placeholder
                        except:
                            risk_metrics['avg_holding_period'] = 0
            
            # Sermaye geçmişi bazlı metrikler
            capital_df = load_capital_history_from_csv()
            if not capital_df.empty and len(capital_df) > 1:
                try:
                    # Basit volatilite hesaplama
                    returns = capital_df['capital'].pct_change().dropna()
                    if len(returns) > 1:
                        risk_metrics['volatility'] = returns.std() * 100
                    
                    # Max drawdown
                    rolling_max = capital_df['capital'].expanding().max()
                    drawdown = (capital_df['capital'] - rolling_max) / rolling_max * 100
                    risk_metrics['max_drawdown'] = abs(drawdown.min()) if not drawdown.empty else 0
                    
                except Exception as e:
                    logger.debug(f"Risk metrikleri hesaplama hatası: {e}")
        
        except Exception as e:
            logger.error(f"Risk analizi hatası: {e}")
        
        return risk_metrics
    
    def get_performance_summary(self, period_days: int = 30) -> Dict:
        """
        Belirtilen dönem için performans özeti
        
        Args:
            period_days (int): Analiz dönemi (gün)
            
        Returns:
            Dict: Performans özeti
        """
        end_date = datetime.now(LOCAL_TZ)
        start_date = end_date - timedelta(days=period_days)
        
        # Trade verilerini yükle
        trades_df = load_trades_from_csv()
        capital_df = load_capital_history_from_csv()
        
        performance = {
            'period_days': period_days,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'capital_growth': 0,
            'return_pct': 0
        }
        
        try:
            if not trades_df.empty:
                # Dönem filtresi
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                period_trades = trades_df[
                    (trades_df['timestamp'] >= start_date) & 
                    (trades_df['timestamp'] <= end_date) &
                    (trades_df['status'] == 'CLOSED')
                ]
                
                if not period_trades.empty:
                    performance['total_trades'] = len(period_trades)
                    performance['total_pnl'] = period_trades['pnl'].sum()
                    
                    winning_trades = period_trades[period_trades['pnl'] > 0]
                    losing_trades = period_trades[period_trades['pnl'] < 0]
                    
                    performance['winning_trades'] = len(winning_trades)
                    performance['losing_trades'] = len(losing_trades)
                    performance['win_rate'] = (len(winning_trades) / len(period_trades) * 100) if len(period_trades) > 0 else 0
                    
                    if not winning_trades.empty:
                        performance['avg_win'] = winning_trades['pnl'].mean()
                        performance['best_trade'] = winning_trades['pnl'].max()
                    
                    if not losing_trades.empty:
                        performance['avg_loss'] = abs(losing_trades['pnl'].mean())
                        performance['worst_trade'] = losing_trades['pnl'].min()
                    
                    # Profit factor
                    total_wins = winning_trades['pnl'].sum() if not winning_trades.empty else 0
                    total_losses = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
                    performance['profit_factor'] = (total_wins / total_losses) if total_losses > 0 else float('inf') if total_wins > 0 else 0
            
            # Sermaye büyümesi
            if not capital_df.empty:
                capital_df['timestamp'] = pd.to_datetime(capital_df['timestamp'])
                period_capital = capital_df[
                    (capital_df['timestamp'] >= start_date) & 
                    (capital_df['timestamp'] <= end_date)
                ]
                
                if len(period_capital) > 1:
                    start_capital = period_capital.iloc[0]['capital']
                    end_capital = period_capital.iloc[-1]['capital']
                    performance['capital_growth'] = end_capital - start_capital
                    performance['return_pct'] = (performance['capital_growth'] / start_capital * 100) if start_capital > 0 else 0
        
        except Exception as e:
            logger.error(f"Performans özeti hatası: {e}")
        
        return performance
    
    def get_position_recommendations(self) -> List[Dict]:
        """
        Mevcut pozisyonlar için öneriler
        
        Returns:
            List[Dict]: Pozisyon önerileri
        """
        recommendations = []
        portfolio = self.get_current_portfolio_status()
        
        for position in portfolio['positions']:
            symbol = position['symbol']
            recommendation = {
                'symbol': symbol,
                'action': 'HOLD',
                'reason': 'Normal pozisyon',
                'priority': 'LOW',
                'risk_level': 'NORMAL'
            }
            
            # P&L bazlı öneriler
            pnl_pct = position['pnl_percentage']
            
            if pnl_pct <= -15:  # %15 zarar
                recommendation['action'] = 'REVIEW_STOP_LOSS'
                recommendation['reason'] = f'Yüksek zarar: {pnl_pct:.1f}%'
                recommendation['priority'] = 'HIGH'
                recommendation['risk_level'] = 'HIGH'
                
            elif pnl_pct >= 20:  # %20 kar
                recommendation['action'] = 'CONSIDER_PARTIAL_CLOSE'
                recommendation['reason'] = f'İyi kar: {pnl_pct:.1f}%'
                recommendation['priority'] = 'MEDIUM'
                recommendation['risk_level'] = 'LOW'
            
            # Tutma süresi bazlı öneriler
            days_held = position['days_held']
            if days_held >= 7:  # 1 haftadan uzun
                recommendation['action'] = 'REVIEW_POSITION'
                recommendation['reason'] += f' | Uzun tutma: {days_held} gün'
                recommendation['priority'] = 'MEDIUM'
            
            # Stop loss mesafesi
            current_price = position['current_price']
            stop_loss = position['stop_loss']
            
            if position['side'] == 'LONG':
                stop_distance = ((current_price - stop_loss) / current_price) * 100
            else:
                stop_distance = ((stop_loss - current_price) / current_price) * 100
            
            if stop_distance < 1:  # Stop loss çok yakın
                recommendation['action'] = 'ADJUST_STOP_LOSS'
                recommendation['reason'] += f' | SL çok yakın: {stop_distance:.1f}%'
                recommendation['priority'] = 'HIGH'
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def get_allocation_analysis(self) -> Dict:
        """
        Portföy tahsis analizi
        
        Returns:
            Dict: Tahsis analizi
        """
        portfolio = self.get_current_portfolio_status()
        
        allocation = {
            'cash_allocation': 0,
            'crypto_allocation': 0,
            'long_allocation': 0,
            'short_allocation': 0,
            'position_sizes': [],
            'recommendations': []
        }
        
        if portfolio['total_portfolio_value'] > 0:
            allocation['cash_allocation'] = (portfolio['cash'] / portfolio['total_portfolio_value']) * 100
            allocation['crypto_allocation'] = (portfolio['total_market_value'] / portfolio['total_portfolio_value']) * 100
            
            # Long/Short dağılımı
            long_value = sum(pos['market_value'] for pos in portfolio['positions'] if pos['side'] == 'LONG')
            short_value = sum(pos['market_value'] for pos in portfolio['positions'] if pos['side'] == 'SHORT')
            
            allocation['long_allocation'] = (long_value / portfolio['total_portfolio_value']) * 100
            allocation['short_allocation'] = (short_value / portfolio['total_portfolio_value']) * 100
            
            # Pozisyon büyüklükleri
            for pos in portfolio['positions']:
                allocation['position_sizes'].append({
                    'symbol': pos['symbol'],
                    'allocation_pct': (pos['market_value'] / portfolio['total_portfolio_value']) * 100,
                    'side': pos['side']
                })
            
            # Tahsis önerileri
            if allocation['cash_allocation'] > 70:
                allocation['recommendations'].append({
                    'type': 'CASH_HIGH',
                    'message': f'Yüksek nakit oranı: {allocation["cash_allocation"]:.1f}%',
                    'suggestion': 'Daha fazla pozisyon açmayı düşünün'
                })
            
            if allocation['crypto_allocation'] > 90:
                allocation['recommendations'].append({
                    'type': 'EXPOSURE_HIGH',
                    'message': f'Yüksek kripto maruziyeti: {allocation["crypto_allocation"]:.1f}%',
                    'suggestion': 'Risk yönetimi için nakit tutmayı düşünün'
                })
            
            # Pozisyon konsantrasyonu
            max_position_pct = max((pos['market_value'] / portfolio['total_portfolio_value']) * 100 
                                 for pos in portfolio['positions']) if portfolio['positions'] else 0
            
            if max_position_pct > 50:
                allocation['recommendations'].append({
                    'type': 'CONCENTRATION_HIGH',
                    'message': f'Yüksek konsantrasyon: {max_position_pct:.1f}%',
                    'suggestion': 'Portföyü daha fazla diversifiye edin'
                })
        
        return allocation
    
    def export_portfolio_report(self) -> Dict:
        """
        Kapsamlı portföy raporu oluştur
        
        Returns:
            Dict: Tam portföy raporu
        """
        report = {
            'generated_at': datetime.now(LOCAL_TZ).isoformat(),
            'portfolio_status': self.get_current_portfolio_status(),
            'risk_metrics': self.calculate_risk_metrics(),
            'performance_30d': self.get_performance_summary(30),
            'performance_7d': self.get_performance_summary(7),
            'allocation_analysis': self.get_allocation_analysis(),
            'position_recommendations': self.get_position_recommendations(),
            'summary': {}
        }
        
        # Özet bilgiler
        portfolio = report['portfolio_status']
        risk = report['risk_metrics']
        perf_30d = report['performance_30d']
        
        report['summary'] = {
            'total_value': portfolio['total_portfolio_value'],
            'total_return': portfolio['total_return'],
            'return_pct': portfolio['total_return_pct'],
            'open_positions': portfolio['positions_count'],
            'cash_ratio': risk['cash_ratio'],
            'win_rate_30d': perf_30d['win_rate'],
            'max_drawdown': risk['max_drawdown'],
            'volatility': risk['volatility'],
            'recommendations_count': len(report['position_recommendations'])
        }
        
        return report


# Global portföy manager instance
portfolio_manager = PortfolioManager()


def get_portfolio_status() -> Dict:
    """Portföy durumunu al"""
    return portfolio_manager.get_current_portfolio_status()


def get_risk_analysis() -> Dict:
    """Risk analizini al"""
    return portfolio_manager.calculate_risk_metrics()


def get_performance_report(days: int = 30) -> Dict:
    """Performans raporunu al"""
    return portfolio_manager.get_performance_summary(days)


def get_position_recommendations() -> List[Dict]:
    """Pozisyon önerilerini al"""
    return portfolio_manager.get_position_recommendations()


def get_allocation_breakdown() -> Dict:
    """Tahsis dağılımını al"""
    return portfolio_manager.get_allocation_analysis()


def generate_full_report() -> Dict:
    """Tam portföy raporunu oluştur"""
    return portfolio_manager.export_portfolio_report()


def calculate_portfolio_beta() -> float:
    """
    Portföy beta'sını hesapla (basitleştirilmiş)
    
    Returns:
        float: Beta değeri
    """
    try:
        # Basitleştirilmiş beta hesaplama
        # Gerçek uygulamada piyasa endeksi ile korelasyon hesaplanmalı
        portfolio = get_portfolio_status()
        
        if not portfolio['positions']:
            return 0.0
        
        # Pozisyon sayısına göre risk faktörü
        position_count = len(portfolio['positions'])
        diversification_factor = min(1.0, position_count / MAX_OPEN_POSITIONS)
        
        # Ortalama leverage (eğer varsa)
        avg_leverage = 1.0  # Spot trading için
        
        # Beta tahmini
        beta = avg_leverage * (1 + (1 - diversification_factor) * 0.5)
        
        return min(beta, 2.0)  # Maksimum 2.0 beta
        
    except Exception as e:
        logger.error(f"Beta hesaplama hatası: {e}")
        return 1.0


def get_position_health_check() -> Dict:
    """
    Pozisyon sağlık kontrolü
    
    Returns:
        Dict: Sağlık durumu
    """
    portfolio = get_portfolio_status()
    
    health_check = {
        'overall_health': 'GOOD',
        'issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    try:
        # Zarar durumu kontrolü
        losing_positions = [pos for pos in portfolio['positions'] if pos['unrealized_pnl'] < 0]
        if len(losing_positions) > len(portfolio['positions']) * 0.6:  # %60'ından fazlası zararda
            health_check['overall_health'] = 'POOR'
            health_check['issues'].append('Pozisyonların çoğu zararda')
        
        # Büyük zarar kontrolü
        big_losses = [pos for pos in portfolio['positions'] if pos['pnl_percentage'] < -10]
        if big_losses:
            health_check['overall_health'] = 'FAIR' if health_check['overall_health'] == 'GOOD' else health_check['overall_health']
            health_check['warnings'].append(f'{len(big_losses)} pozisyon %10+ zararda')
        
        # Konsantrasyon riski
        if portfolio['positions']:
            max_position_pct = max(pos['invested_amount'] / portfolio['total_invested'] * 100 
                                 for pos in portfolio['positions']) if portfolio['total_invested'] > 0 else 0
            if max_position_pct > 40:
                health_check['warnings'].append(f'Yüksek konsantrasyon riski: {max_position_pct:.1f}%')
        
        # Öneriler
        if portfolio['available_slots'] > 0 and portfolio['cash'] > portfolio['total_invested'] * 0.3:
            health_check['recommendations'].append('Mevcut nakit ile yeni pozisyon açabilirsiniz')
        
        if len(losing_positions) > 0:
            health_check['recommendations'].append('Zarardaki pozisyonları gözden geçirin')
    
    except Exception as e:
        logger.error(f"Sağlık kontrolü hatası: {e}")
        health_check['overall_health'] = 'UNKNOWN'
        health_check['issues'].append('Analiz hatası oluştu')
    
    return health_check