"""
Telegram Bot servisleri
Telegram API ile mesaj gönderme ve Devis'So çizgi temas bildirimleri
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
🔗 Supertrend + C-Signal + Devis'So Trend Sistemi

#Test #TelegramBot #DevisoTrend
        """.strip()
        
        return self.send_message(test_message)
    
    def send_deviso_line_contact_alert(self, symbol_data: Dict[str, Any]) -> bool:
        """
        Devis'So çizgi temas uyarısı gönder - LONG trend için
        
        Args:
            symbol_data (Dict[str, Any]): Sembol verileri
            
        Returns:
            bool: Uyarı başarıyla gönderildi ise True
        """
        if not self.is_configured():
            return False
        
        try:
            symbol = symbol_data.get('symbol', 'UNKNOWN')
            deviso_status = symbol_data.get('deviso_status', {})
            tradingview_link = symbol_data.get('tradingview_link', '#')
            deviso_lines = symbol_data.get('deviso_lines', {})
            
            message = self._format_deviso_contact_message(
                symbol, deviso_status, tradingview_link, deviso_lines
            )
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Devis'So çizgi temas uyarısı formatı hatası: {e}")
            return False
    
    def _format_deviso_contact_message(self, symbol: str, deviso_status: Dict[str, Any], 
                                     tradingview_link: str, deviso_lines: Dict[str, Any]) -> str:
        """
        Devis'So çizgi temas mesajını formatla
        
        Args:
            symbol (str): Sembol adı
            deviso_status (Dict[str, Any]): Devis'So temas durumu
            tradingview_link (str): TradingView linki
            deviso_lines (Dict[str, Any]): Çizgi değerleri
            
        Returns:
            str: Formatlanmış mesaj
        """
        try:
            contact_level = deviso_status.get('contact_level', 'Unknown')
            signal_strength = deviso_status.get('signal_strength', 'Unknown')
            alert_message = deviso_status.get('alert_message', '')
            line_values = deviso_status.get('line_values', {})
            
            # Seviye bazlı emoji ve mesaj
            if contact_level == 'Level_1_Blue':
                level_emoji = "🔵"
                level_text = "SEVİYE 1 - MAVİ ÇİZGİ"
                level_description = "İlk temas noktası"
            elif contact_level == 'Level_2_Yellow':
                level_emoji = "🟡"
                level_text = "SEVİYE 2 - SARI ÇİZGİ"
                level_description = "Mavi çizgi delindi"
            elif contact_level == 'Level_3_Purple':
                level_emoji = "🟣"
                level_text = "SEVİYE 3 - PEMBE ÇİZGİ"
                level_description = "Sarı çizgi delindi - KRİTİK!"
            elif contact_level == 'Manual_Reset':
                level_emoji = "🚨"
                level_text = "MANUEL DEĞİŞİKLİK"
                level_description = "Trend manuel olarak değiştirildi"
            else:
                level_emoji = "⚠️"
                level_text = "DEVIS'SO SİNYAL"
                level_description = "Bilinmeyen seviye"
            
            # Güç seviyesi emojisi
            if signal_strength == 'Strong':
                strength_emoji = "🔥🔥🔥"
            elif signal_strength == 'Medium':
                strength_emoji = "🔥🔥"
            elif signal_strength == 'Critical':
                strength_emoji = "🚨🔥🔥🔥"
            else:
                strength_emoji = "⚡"
            
            # Çizgi değerleri formatla
            line_info = ""
            if line_values:
                ma_blue = line_values.get('ma_blue', 0)
                ma_yellow = line_values.get('ma_yellow', 0)
                ma_purple = line_values.get('ma_purple', 0)
                
                line_info = f"""
📊 <b>Çizgi Değerleri (USDT):</b>
🔵 Mavi: <code>{ma_blue:.5f}</code>
🟡 Sarı: <code>{ma_yellow:.5f}</code>
🟣 Pembe: <code>{ma_purple:.5f}</code>
                """.strip()
            
            # Mesaj formatı
            current_time = datetime.now().strftime('%H:%M:%S')
            
            message = f"""
🎯 <b>DEVIS'SO TREND ALARMI</b> 🎯

{level_emoji} <b>{symbol}</b> - LONG TREND
📈 {level_text}
💡 {level_description}
{strength_emoji} Güç: <b>{signal_strength}</b>
⏰ Saat: <b>{current_time}</b>

{line_info}

📝 <b>Detay:</b> {alert_message}

<a href="{tradingview_link}">📊 TradingView'da İncele</a>

#DevisoTrend #{symbol} #LongTrend #{contact_level}
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Telegram mesaj formatı hatası: {e}")
            return f"🎯 DEVIS'SO TREND: {symbol} - {deviso_status.get('alert_message', 'Bilinmeyen sinyal')}"
    
    def send_analysis_summary(self, analysis_data: Dict[str, Any]) -> bool:
        """
        Analiz özeti gönder - Devis'So entegreli
        
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
            deviso_contact_count = analysis_data.get('deviso_contact_count', 0)
            timeframe = analysis_data.get('timeframe', '4h')
            current_time = datetime.now().strftime('%H:%M:%S')
            
            message = f"""
📊 <b>ANALİZ ÖZETİ</b> 📊

⏰ Saat: <b>{current_time}</b>
📈 Timeframe: <b>{timeframe}</b>
💰 Toplam Sembol: <b>{total_symbols}</b>
🔥 %100+ Ratio: <b>{high_ratio_count}</b>
🎯 Devis'So Temas: <b>{deviso_contact_count}</b>

#AnalizÖzeti #Supertrend #DevisoTrend
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Analiz özeti formatı hatası: {e}")
            return False
    
    def should_send_deviso_alert(self, symbol_data: Dict[str, Any], deviso_status: Dict[str, Any], 
                                min_interval_minutes: int = 5) -> bool:
        """
        Devis'So çizgi temas uyarısı gönderilmeli mi kontrol et (spam önleme)
        
        Args:
            symbol_data (Dict[str, Any]): Sembol verileri
            deviso_status (Dict[str, Any]): Devis'So temas durumu
            min_interval_minutes (int): Minimum bekleme süresi (dakika)
            
        Returns:
            bool: Uyarı gönderilmeli ise True
        """
        if not deviso_status.get('has_line_contact', False):
            return False
        
        # Sadece LONG trend için bildirim gönder
        trend_direction = symbol_data.get('max_supertrend_type', 'None')
        if trend_direction != 'Bullish':
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
    
    # =====================================================
    # BACKWARD COMPATIBILITY METHODS
    # =====================================================
    
    def send_reverse_momentum_alert(self, symbol_data: Dict[str, Any]) -> bool:
        """
        DEPRECATED: Geriye uyumluluk için - Devis'So çizgi temas uyarısı gönder
        
        Args:
            symbol_data (Dict[str, Any]): Sembol verileri
            
        Returns:
            bool: Uyarı başarıyla gönderildi ise True
        """
        logger.warning("send_reverse_momentum_alert() DEPRECATED - send_deviso_line_contact_alert() kullanın")
        return self.send_deviso_line_contact_alert(symbol_data)
    
    def _format_reverse_momentum_message(self, symbol: str, reverse_momentum_data: Dict[str, Any], 
                                       tradingview_link: str) -> str:
        """
        DEPRECATED: Geriye uyumluluk için - Eski format mesaj
        
        Args:
            symbol (str): Sembol adı
            reverse_momentum_data (Dict[str, Any]): Ters momentum verileri (deprecated)
            tradingview_link (str): TradingView linki
            
        Returns:
            str: Formatlanmış mesaj
        """
        logger.warning("_format_reverse_momentum_message() DEPRECATED - _format_deviso_contact_message() kullanın")
        
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
⚠️ <i>DEPRECATED - Devis'So Trend kullanın</i>

{trend_emoji} <b>{symbol}</b>
📊 {signal_text}
{strength_emoji} Güç: <b>{signal_strength}</b>
📈 C-Signal: <b>{c_signal_value}</b>
⏰ Saat: <b>{current_time}</b>

<a href="{tradingview_link}">📊 TradingView'da İncele</a>

#TersMomentum #{symbol} #CSignal #DEPRECATED
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Telegram mesaj formatı hatası: {e}")
            return f"🚨 TERS MOMENTUM: {symbol} - {reverse_momentum_data.get('alert_message', 'Bilinmeyen sinyal')}"
    
    def should_send_alert(self, symbol_data: Dict[str, Any], reverse_momentum: Dict[str, Any], 
                         min_interval_minutes: int = 5) -> bool:
        """
        DEPRECATED: Geriye uyumluluk için - Devis'So alert kontrolü yap
        
        Args:
            symbol_data (Dict[str, Any]): Sembol verileri
            reverse_momentum (Dict[str, Any]): Ters momentum verileri (deprecated)
            min_interval_minutes (int): Minimum bekleme süresi (dakika)
            
        Returns:
            bool: Uyarı gönderilmeli ise True
        """
        logger.warning("should_send_alert() DEPRECATED - should_send_deviso_alert() kullanın")
        
        # Eski formatı yeni formata dönüştür
        deviso_status = {
            'has_line_contact': reverse_momentum.get('has_reverse_momentum', False),
            'contact_level': 'Manual_Reset' if reverse_momentum.get('reverse_type') != 'None' else 'None',
            'signal_strength': reverse_momentum.get('signal_strength', 'None'),
            'alert_message': reverse_momentum.get('alert_message', ''),
            'line_values': {}
        }
        
        return self.should_send_deviso_alert(symbol_data, deviso_status, min_interval_minutes)