"""
Telegram Bot servisleri
Telegram API ile mesaj gönderme ve bildirim yönetimi
"""

import os
import requests
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TelegramService:
    """Telegram bot işlemleri için service sınıfı"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
    
    def is_configured(self) -> bool:
        """
        Telegram bot konfigürasyonunun tamamlanıp tamamlanmadığını kontrol et
        
        Returns:
            bool: Konfigürasyon tamamsa True
        """
        return bool(self.bot_token and self.chat_id)
    
    def get_status(self) -> Dict[str, str]:
        """
        Telegram bot durumunu getir
        
        Returns:
            Dict[str, str]: Bot durumu bilgileri
        """
        return {
            'bot_token_status': '✅' if self.bot_token else '❌',
            'chat_id_status': '✅' if self.chat_id else '❌',
            'overall_status': 'Aktif' if self.is_configured() else 'Yapılandırılmamış'
        }
    
    def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """
        Telegram'a mesaj gönder
        
        Args:
            message (str): Gönderilecek mesaj
            parse_mode (str): Mesaj formatı (HTML, Markdown)
            
        Returns:
            bool: Mesaj başarıyla gönderildi ise True
        """
        if not self.is_configured():
            logger.warning("⚠️ Telegram konfigürasyonu eksik")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                logger.info("📱 Telegram mesajı başarıyla gönderildi")
                return True
            else:
                logger.error(f"❌ Telegram mesaj gönderme hatası: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("❌ Telegram API timeout")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Telegram API bağlantı hatası: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Telegram mesaj gönderme hatası: {e}")
            return False
    
    def send_test_message(self) -> bool:
        """
        Test mesajı gönder
        
        Returns:
            bool: Test mesajı başarıyla gönderildi ise True
        """
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        test_message = f"""
🧪 <b>TELEGRAM BOT TEST</b> 🧪

✅ Bot başarıyla çalışıyor!
⏰ Test zamanı: {current_time}
🔗 Ardışık Mum + C-Signal + Ters Momentum Sistemi

#Test #TelegramBot
        """.strip()
        
        return self.send_message(test_message)
    
    def send_reverse_momentum_alert(self, symbol_data: Dict[str, Any]) -> bool:
        """
        Ters momentum uyarısı gönder
        
        Args:
            symbol_data (Dict[str, Any]): Sembol verileri
            
        Returns:
            bool: Uyarı başarıyla gönderildi ise True
        """
        if not self.is_configured():
            return False
        
        try:
            symbol = symbol_data.get('symbol', 'UNKNOWN')
            reverse_momentum_data = symbol_data.get('reverse_momentum', {})
            tradingview_link = symbol_data.get('tradingview_link', '#')
            
            message = self._format_reverse_momentum_message(
                symbol, reverse_momentum_data, tradingview_link
            )
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Ters momentum uyarısı formatı hatası: {e}")
            return False
    
    def _format_reverse_momentum_message(self, symbol: str, reverse_momentum_data: Dict[str, Any], tradingview_link: str) -> str:
        """
        Ters momentum mesajını formatla
        
        Args:
            symbol (str): Sembol adı
            reverse_momentum_data (Dict[str, Any]): Ters momentum verileri
            tradingview_link (str): TradingView linki
            
        Returns:
            str: Formatlanmış mesaj
        """
        try:
            reverse_type = reverse_momentum_data.get('reverse_type', 'Unknown')
            signal_strength = reverse_momentum_data.get('signal_strength', 'Unknown')
            c_signal_value = reverse_momentum_data.get('signal_value', 0)
            
            # Emoji ve sinyal metni
            if reverse_type == 'C↓':
                trend_emoji = "📻"
                signal_text = "DÜŞÜŞ SİNYALİ"
            elif reverse_type == 'C↑':
                trend_emoji = "📺" 
                signal_text = "YÜKSELİŞ SİNYALİ"
            else:
                trend_emoji = "⚠️"
                signal_text = "TERS MOMENTUM"
            
            # Güç seviyesi emojisi
            if signal_strength == 'Strong':
                strength_emoji = "🔥🔥🔥"
            elif signal_strength == 'Medium':
                strength_emoji = "🔥🔥"
            elif signal_strength == 'Weak':
                strength_emoji = "🔥"
            else:
                strength_emoji = "⚡"
            
            # Mesaj formatı
            current_time = datetime.now().strftime('%H:%M:%S')
            
            message = f"""
🚨 <b>TERS MOMENTUM ALARMI</b> 🚨

{trend_emoji} <b>{symbol}</b>
📊 {signal_text}
{strength_emoji} Güç: <b>{signal_strength}</b>
📈 C-Signal: <b>{c_signal_value}</b>
⏰ Saat: <b>{current_time}</b>

<a href="{tradingview_link}">📊 TradingView'da İncele</a>

#TersMomentum #{symbol} #CSignal
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Telegram mesaj formatı hatası: {e}")
            return f"🚨 TERS MOMENTUM: {symbol} - {reverse_momentum_data.get('alert_message', 'Bilinmeyen sinyal')}"
    
    def send_analysis_summary(self, analysis_data: Dict[str, Any]) -> bool:
        """
        Analiz özeti gönder
        
        Args:
            analysis_data (Dict[str, Any]): Analiz verileri
            
        Returns:
            bool: Özet başarıyla gönderildi ise True
        """
        if not self.is_configured():
            return False
        
        try:
            total_symbols = analysis_data.get('total_symbols', 0)
            high_consecutive_count = analysis_data.get('high_consecutive_count', 0)
            reverse_momentum_count = analysis_data.get('reverse_momentum_count', 0)
            timeframe = analysis_data.get('timeframe', '4h')
            current_time = datetime.now().strftime('%H:%M:%S')
            
            message = f"""
📊 <b>ANALİZ ÖZETİ</b> 📊

⏰ Saat: <b>{current_time}</b>
📈 Timeframe: <b>{timeframe}</b>
💰 Toplam Sembol: <b>{total_symbols}</b>
🔥 5+ Ardışık: <b>{high_consecutive_count}</b>
🚨 Ters Momentum: <b>{reverse_momentum_count}</b>

#AnalizÖzeti #ArdışıkMum
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Analiz özeti formatı hatası: {e}")
            return False
    
    def should_send_alert(self, symbol_data: Dict[str, Any], reverse_momentum: Dict[str, Any], min_interval_minutes: int = 5) -> bool:
        """
        Telegram uyarısı gönderilmeli mi kontrol et (spam önleme)
        
        Args:
            symbol_data (Dict[str, Any]): Sembol verileri
            reverse_momentum (Dict[str, Any]): Ters momentum verileri
            min_interval_minutes (int): Minimum bekleme süresi (dakika)
            
        Returns:
            bool: Uyarı gönderilmeli ise True
        """
        if not reverse_momentum.get('has_reverse_momentum', False):
            return False
        
        # Son bildirim zamanını kontrol et
        last_alert = symbol_data.get('last_telegram_alert')
        if last_alert:
            try:
                last_alert_time = datetime.strptime(last_alert, '%Y-%m-%d %H:%M:%S')
                time_diff = datetime.now() - last_alert_time
                if time_diff.total_seconds() < (min_interval_minutes * 60):
                    return False
            except Exception:
                pass  # Tarih parse hatası durumunda bildirimi gönder
        
        return True