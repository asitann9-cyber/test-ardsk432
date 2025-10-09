"""
Telegram Bot servisleri
Telegram API ile mesaj gönderme
🆕 YENİ: C-Signal ±20 L/S Alert Bildirimi Eklendi
✅ FIX: Spam kontrolü düzeltildi - C-Signal alertleri çalışıyor!
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
        
        # ✅ FIX: Başlatma logları
        if self.is_configured():
            logger.info("✅ Telegram servisi başlatıldı - Bot Token ve Chat ID mevcut")
        else:
            logger.warning("⚠️ Telegram servisi başlatıldı ama konfigürasyon eksik!")
            logger.warning(f"   Bot Token: {'✅ Mevcut' if self.bot_token else '❌ Eksik'}")
            logger.warning(f"   Chat ID: {'✅ Mevcut' if self.chat_id else '❌ Eksik'}")
    
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
            logger.warning("⚠️ Telegram konfigürasyonu eksik - mesaj gönderilemedi")
            logger.warning(f"   Bot Token: {'✅' if self.bot_token else '❌'}")
            logger.warning(f"   Chat ID: {'✅' if self.chat_id else '❌'}")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            logger.debug(f"📤 Telegram mesajı gönderiliyor... (URL: {url})")
            logger.debug(f"   Chat ID: {self.chat_id}")
            logger.debug(f"   Mesaj uzunluğu: {len(message)} karakter")
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                logger.info("✅ Telegram mesajı başarıyla gönderildi")
                return True
            else:
                logger.error(f"❌ Telegram mesaj gönderme hatası: HTTP {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("❌ Telegram API timeout (10 saniye)")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Telegram API bağlantı hatası: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Telegram mesaj gönderme hatası: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
        
        logger.info("📤 Test mesajı gönderiliyor...")
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
            
            logger.info(f"📤 Manuel tür değişikliği bildirimi gönderiliyor: {symbol} ({old_type} → {new_type})")
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
            logger.warning(f"⚠️ C-Signal alert gönderilemedi ({symbol}) - Telegram konfigürasyonu eksik")
            return False
        
        try:
            logger.info(f"📤 C-Signal alert hazırlanıyor: {symbol} - {signal_type}")
            
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
            
            logger.info(f"📤 C-Signal alert gönderiliyor: {symbol} - {signal_name} (C={c_signal_value:+.2f})")
            success = self.send_message(message)
            
            if success:
                logger.info(f"✅ C-Signal alert başarıyla gönderildi: {symbol} - {signal_name}")
            else:
                logger.error(f"❌ C-Signal alert gönderilemedi: {symbol} - {signal_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ C-Signal alert bildirimi hatası ({symbol}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    # 🆕 YENİ FONKSİYON: Toplu C-Signal Alert
    def send_batch_c_signal_alerts(self, alerts: list) -> int:
        """
        Birden fazla C-Signal alert'ini tek mesajda gönder
        
        Args:
            alerts (list): Alert listesi, her biri dict formatında
                         {'symbol', 'signal_type', 'c_signal_value', 'tradingview_link'}
            
        Returns:
            int: Gönderilen alert sayısı (başarılı ise)
        """
        if not self.is_configured():
            logger.warning(f"⚠️ Toplu C-Signal alert gönderilemedi - Telegram konfigürasyonu eksik")
            return 0
        
        if not alerts:
            logger.warning("⚠️ Toplu C-Signal alert listesi boş")
            return 0
        
        try:
            logger.info(f"📤 Toplu C-Signal alert hazırlanıyor: {len(alerts)} alert")
            
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Alert'leri LONG ve SHORT olarak grupla
            long_alerts = [a for a in alerts if a['signal_type'] == 'L']
            short_alerts = [a for a in alerts if a['signal_type'] == 'S']
            
            logger.info(f"   🟢 LONG: {len(long_alerts)} alert")
            logger.info(f"   🔴 SHORT: {len(short_alerts)} alert")
            
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
            
            logger.info(f"📤 Toplu C-Signal alert gönderiliyor: {len(alerts)} alert")
            success = self.send_message(message)
            
            if success:
                logger.info(f"✅ Toplu C-Signal alert başarıyla gönderildi: {len(alerts)} alert")
                return len(alerts)
            else:
                logger.error(f"❌ Toplu C-Signal alert gönderilemedi")
                return 0
            
        except Exception as e:
            logger.error(f"❌ Toplu C-Signal alert hatası: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
                elapsed_minutes = time_diff.total_seconds() / 60
                
                if time_diff.total_seconds() < (min_interval_minutes * 60):
                    logger.debug(f"⏳ Spam koruması: {symbol_data.get('symbol')} - Son alert: {elapsed_minutes:.1f} dakika önce (min: {min_interval_minutes} dk)")
                    return False
                else:
                    logger.debug(f"✅ Spam kontrolü geçildi: {symbol_data.get('symbol')} - Son alert: {elapsed_minutes:.1f} dakika önce")
            except Exception as e:
                logger.warning(f"⚠️ Tarih parse hatası (spam kontrolü atlandı): {e}")
                pass  # Tarih parse hatası durumunda bildirimi gönder
        else:
            logger.debug(f"✅ İlk alert: {symbol_data.get('symbol')} - spam kontrolü yok")
        
        return True
    
    # ✅ FIX: C-Signal için spam kontrolü düzeltildi
    def should_send_c_signal_alert(self, symbol_data: Dict[str, Any], min_interval_minutes: int = 30) -> bool:
        """
        C-Signal alert'i için spam kontrolü (daha uzun interval)
        
        Args:
            symbol_data (Dict[str, Any]): Sembol verileri
            min_interval_minutes (int): Minimum bekleme süresi (dakika) - default 30 dakika
            
        Returns:
            bool: Alert gönderilmeli ise True
        """
        symbol = symbol_data.get('symbol', 'UNKNOWN')
        
        # ✅ FIX: Son C-Signal alert zamanını kontrol et
        last_alert_time_str = symbol_data.get('last_c_signal_alert_time')
        
        logger.debug(f"🔍 C-Signal spam kontrolü: {symbol}")
        logger.debug(f"   Son alert zamanı: {last_alert_time_str}")
        logger.debug(f"   Min interval: {min_interval_minutes} dakika")
        
        if last_alert_time_str and last_alert_time_str != 'Hiç gönderilmedi':
            try:
                last_alert_dt = datetime.strptime(last_alert_time_str, '%Y-%m-%d %H:%M:%S')
                time_diff = datetime.now() - last_alert_dt
                elapsed_seconds = time_diff.total_seconds()
                elapsed_minutes = elapsed_seconds / 60
                
                logger.debug(f"   Geçen süre: {elapsed_minutes:.1f} dakika ({elapsed_seconds:.0f} saniye)")
                
                if elapsed_seconds < (min_interval_minutes * 60):
                    logger.info(f"⏳ C-Signal spam koruması AKTİF: {symbol} - Son alert: {elapsed_minutes:.1f} dakika önce (min: {min_interval_minutes} dk)")
                    return False
                else:
                    logger.info(f"✅ C-Signal spam kontrolü GEÇİLDİ: {symbol} - Son alert: {elapsed_minutes:.1f} dakika önce")
                    return True
                    
            except Exception as e:
                logger.warning(f"⚠️ C-Signal tarih parse hatası ({symbol}): {e} - Alert gönderilecek")
                return True  # Parse hatası durumunda gönder
        else:
            logger.info(f"✅ İLK C-Signal alert: {symbol} - spam kontrolü YOK")
            return True