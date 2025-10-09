"""
Telegram Bot servisleri
Telegram API ile mesaj gönderme
🆕 YENİ: C-Signal ±20 L/S Alert Bildirimi Eklendi
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
    
    # 🆕 YENİ FONKSİYON: C-Signal ±20 Alert
    def send_c_signal_alert(self, symbol: str, signal_type: str, c_signal_value: float, 
                           tradingview_link: str, ratio_percent: float = 0, 
                           supertrend_type: str = 'None') -> bool:
        """
        C-Signal ±20 alert bildirimi gönder
        
        Args:
            symbol (str): Sembol adı
            signal_type (str): Sinyal tipi ('L' = LONG, 'S' = SHORT)
            c_signal_value (float): C-Signal değeri
            tradingview_link (str): TradingView linki
            ratio_percent (float): Supertrend ratio yüzdesi
            supertrend_type (str): Supertrend trend tipi
            
        Returns:
            bool: Bildirim başarıyla gönderildi ise True
        """
        if not self.is_configured():
            return False
        
        try:
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Sinyal tipine göre emoji ve açıklama
            if signal_type == 'L':
                signal_emoji = "🟢"
                signal_name = "LONG"
                signal_desc = "C-Signal +20 seviyesini YUKARI geçti!"
                action = "🚀 LONG pozisyon için uygun sinyal"
            elif signal_type == 'S':
                signal_emoji = "🔴"
                signal_name = "SHORT"
                signal_desc = "C-Signal -20 seviyesini AŞAĞI geçti!"
                action = "📉 SHORT pozisyon için uygun sinyal"
            else:
                signal_emoji = "⚪"
                signal_name = "BİLİNMEYEN"
                signal_desc = "Bilinmeyen sinyal tipi"
                action = "⚠️ Dikkatli olun"
            
            # Supertrend trend emoji
            trend_emoji = "🟢" if supertrend_type == 'Bullish' else "🔴" if supertrend_type == 'Bearish' else "⚪"
            
            message = f"""
🔔 <b>C-SIGNAL ±20 ALERT!</b> 🔔

{signal_emoji} <b>{symbol}</b> - <b>{signal_name} SİNYALİ</b>
━━━━━━━━━━━━━━━━━━━━━
📊 <b>C-Signal Değeri:</b> {c_signal_value:+.2f}
{signal_desc}

📈 <b>Supertrend Bilgileri:</b>
   • Trend: {trend_emoji} {supertrend_type}
   • Ratio: {abs(ratio_percent):.2f}%

⏰ <b>Zaman:</b> {current_time}

💡 <b>ÖNERİ:</b> {action}

<a href="{tradingview_link}">📊 TradingView'da İncele</a>

#CSignalAlert #{symbol} #{signal_name} #Trading
            """.strip()
            
            success = self.send_message(message)
            
            if success:
                logger.info(f"📱 C-Signal alert gönderildi: {symbol} - {signal_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"C-Signal alert bildirimi hatası: {e}")
            return False
    
    # 🆕 YENİ FONKSİYON: Toplu C-Signal Alert
    def send_batch_c_signal_alerts(self, alerts: list) -> int:
        """
        Birden fazla C-Signal alert'ini tek mesajda gönder
        
        Args:
            alerts (list): Alert listesi, her biri dict formatında
                         {'symbol', 'signal_type', 'c_signal_value', 'tradingview_link'}
            
        Returns:
            int: Gönderilen mesaj sayısı
        """
        if not self.is_configured() or not alerts:
            return 0
        
        try:
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Alert'leri LONG ve SHORT olarak grupla
            long_alerts = [a for a in alerts if a['signal_type'] == 'L']
            short_alerts = [a for a in alerts if a['signal_type'] == 'S']
            
            message = f"""
🔔 <b>TOPLU C-SIGNAL ALERT!</b> 🔔
━━━━━━━━━━━━━━━━━━━━━
⏰ <b>Zaman:</b> {current_time}
📊 <b>Toplam Alert:</b> {len(alerts)} adet
"""
            
            # LONG sinyalleri
            if long_alerts:
                message += f"\n\n🟢 <b>LONG SİNYALLERİ ({len(long_alerts)} adet):</b>\n"
                for alert in long_alerts[:5]:  # Maksimum 5 göster
                    c_val = alert['c_signal_value']
                    message += f"   • <b>{alert['symbol']}</b>: C={c_val:+.2f}\n"
                
                if len(long_alerts) > 5:
                    message += f"   ... ve {len(long_alerts) - 5} sembol daha\n"
            
            # SHORT sinyalleri
            if short_alerts:
                message += f"\n🔴 <b>SHORT SİNYALLERİ ({len(short_alerts)} adet):</b>\n"
                for alert in short_alerts[:5]:  # Maksimum 5 göster
                    c_val = alert['c_signal_value']
                    message += f"   • <b>{alert['symbol']}</b>: C={c_val:+.2f}\n"
                
                if len(short_alerts) > 5:
                    message += f"   ... ve {len(short_alerts) - 5} sembol daha\n"
            
            message += "\n💡 Detaylar için panel'i kontrol edin!\n\n#CSignalAlert #BatchAlert #Trading"
            
            success = self.send_message(message)
            
            if success:
                logger.info(f"📱 Toplu C-Signal alert gönderildi: {len(alerts)} alert")
                return len(alerts)
            
            return 0
            
        except Exception as e:
            logger.error(f"Toplu C-Signal alert hatası: {e}")
            return 0
    
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
    
    # 🆕 YENİ FONKSİYON: C-Signal için spam kontrolü
    def should_send_c_signal_alert(self, symbol_data: Dict[str, Any], min_interval_minutes: int = 30) -> bool:
        """
        C-Signal alert'i için spam kontrolü (daha uzun interval)
        
        Args:
            symbol_data (Dict[str, Any]): Sembol verileri
            min_interval_minutes (int): Minimum bekleme süresi (dakika) - default 30 dakika
            
        Returns:
            bool: Alert gönderilmeli ise True
        """
        # Son C-Signal alert zamanını kontrol et
        last_alert_time = symbol_data.get('last_c_signal_alert_time')
        if last_alert_time:
            try:
                last_alert_dt = datetime.strptime(last_alert_time, '%Y-%m-%d %H:%M:%S')
                time_diff = datetime.now() - last_alert_dt
                if time_diff.total_seconds() < (min_interval_minutes * 60):
                    logger.debug(f"C-Signal spam koruması: {symbol_data.get('symbol')} - Son alert: {time_diff.total_seconds()//60:.0f} dakika önce")
                    return False
            except Exception:
                pass
        
        return True