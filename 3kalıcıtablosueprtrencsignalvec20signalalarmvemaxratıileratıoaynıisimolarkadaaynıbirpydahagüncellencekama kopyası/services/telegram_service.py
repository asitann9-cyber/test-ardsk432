"""
Telegram Bot servisleri
Telegram API ile mesaj gönderme
🆕 YENİ: C-Signal Alert Bildirimleri
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
🔗 Supertrend + C-Signal Analiz Sistemi

#Test #TelegramBot
        """.strip()
        
        return self.send_message(test_message)
    
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
            high_ratio_count = analysis_data.get('high_ratio_count', 0)
            timeframe = analysis_data.get('timeframe', '4h')
            current_time = datetime.now().strftime('%H:%M:%S')
            
            message = f"""
📊 <b>ANALİZ ÖZETİ</b> 📊

⏰ Saat: <b>{current_time}</b>
📈 Timeframe: <b>{timeframe}</b>
💰 Toplam Sembol: <b>{total_symbols}</b>
🔥 %100+ Ratio: <b>{high_ratio_count}</b>

#AnalizÖzeti #Supertrend
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Analiz özeti formatı hatası: {e}")
            return False
    
    def send_manual_type_change_alert(self, symbol: str, old_type: str, new_type: str, tradingview_link: str) -> bool:
        """
        Manuel tür değişikliği bildirimi gönder
        
        Args:
            symbol (str): Sembol adı
            old_type (str): Eski tür
            new_type (str): Yeni tür
            tradingview_link (str): TradingView linki
            
        Returns:
            bool: Bildirim başarıyla gönderildi ise True
        """
        if not self.is_configured():
            return False
        
        try:
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Emoji seçimi
            if new_type == 'Bullish':
                type_emoji = "🟢"
            elif new_type == 'Bearish':
                type_emoji = "🔴"
            else:
                type_emoji = "⚪"
            
            message = f"""
🔒 <b>MANUEL TÜR DEĞİŞİKLİĞİ</b> 🔒

{type_emoji} <b>{symbol}</b>
📊 Eski Tür: <b>{old_type}</b> → Yeni Tür: <b>{new_type}</b>
⏰ Saat: <b>{current_time}</b>

⚠️ <i>Tür manuel olarak değiştirildi ve kilitlendi!</i>
💡 Bir sonraki güncellemeye kadar bu tür korunacak.

<a href="{tradingview_link}">📊 TradingView'da İncele</a>

#ManuelDeğişiklik #{symbol} #{new_type}
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Manuel tür değişikliği bildirimi hatası: {e}")
            return False
    
    # 🆕 C-SIGNAL ALERT BİLDİRİMİ
    def send_c_signal_alert(self, symbol: str, signal_type: str, c_signal_value: float, 
                           tradingview_link: str, threshold: float) -> bool:
        """
        C-Signal alert bildirimi gönder
        
        Args:
            symbol (str): Sembol adı
            signal_type (str): Sinyal tipi ('L' = LONG, 'S' = SHORT)
            c_signal_value (float): C-Signal değeri
            tradingview_link (str): TradingView linki
            threshold (float): Kullanılan threshold değeri
            
        Returns:
            bool: Bildirim başarıyla gönderildi ise True
        """
        if not self.is_configured():
            return False
        
        try:
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Sinyal tipine göre emoji ve mesaj
            if signal_type == 'L':
                signal_emoji = "🟢"
                signal_name = "LONG"
                signal_description = f"C-Signal <b>+{threshold:.0f}</b> eşiğini geçti!"
            elif signal_type == 'S':
                signal_emoji = "🔴"
                signal_name = "SHORT"
                signal_description = f"C-Signal <b>-{threshold:.0f}</b> eşiğini geçti!"
            else:
                signal_emoji = "⚪"
                signal_name = "UNKNOWN"
                signal_description = "Bilinmeyen sinyal"
            
            message = f"""
🔔 <b>C-SIGNAL ALERT!</b> 🔔

{signal_emoji} <b>{symbol}</b> - {signal_name}
📊 C-Signal Değeri: <b>{c_signal_value:+.2f}</b>
🎯 Threshold: <b>±{threshold:.0f}</b>
⏰ Saat: <b>{current_time}</b>

💡 {signal_description}

<a href="{tradingview_link}">📊 TradingView'da İncele</a>

#CSignalAlert #{symbol} #{signal_name}
            """.strip()
            
            success = self.send_message(message)
            
            if success:
                logger.info(f"📱 C-Signal alert gönderildi: {symbol} - {signal_name} (C={c_signal_value:+.2f})")
            
            return success
            
        except Exception as e:
            logger.error(f"C-Signal alert bildirimi hatası: {e}")
            return False
    
    def should_send_alert(self, symbol_data: Dict[str, Any], min_interval_minutes: int = 5) -> bool:
        """
        Bildirim gönderilmeli mi kontrol et (spam önleme)
        
        Args:
            symbol_data (Dict[str, Any]): Sembol verileri
            min_interval_minutes (int): Minimum bekleme süresi (dakika)
            
        Returns:
            bool: Bildirim gönderilmeli ise True
        """
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
    
    # 🆕 C-SIGNAL İÇİN ÖZEL SPAM ÖNLEME
    def should_send_c_signal_alert(self, symbol_data: Dict[str, Any], min_interval_minutes: int = 5) -> bool:
        """
        C-Signal alert gönderilmeli mi kontrol et (spam önleme)
        
        Args:
            symbol_data (Dict[str, Any]): Sembol verileri
            min_interval_minutes (int): Minimum bekleme süresi (dakika)
            
        Returns:
            bool: C-Signal alert gönderilmeli ise True
        """
        # Son C-Signal alert zamanını kontrol et
        last_c_signal_alert = symbol_data.get('last_c_signal_alert_time')
        if last_c_signal_alert:
            try:
                last_alert_time = datetime.strptime(last_c_signal_alert, '%Y-%m-%d %H:%M:%S')
                time_diff = datetime.now() - last_alert_time
                if time_diff.total_seconds() < (min_interval_minutes * 60):
                    logger.debug(f"⏳ {symbol_data.get('symbol')} C-Signal spam önleme: {time_diff.total_seconds():.0f}s < {min_interval_minutes*60}s")
                    return False
            except Exception:
                pass  # Tarih parse hatası durumunda bildirimi gönder
        
        return True
    