/**
 * Ardışık Mum + C-Signal + Ters Momentum Analiz Sistemi
 * Frontend JavaScript Application
 */

class AnalysisApp {
  constructor() {
    this.currentResults = null;
    this.allSymbols = [];
    this.selectedSymbols = [];
    this.currentFilter = 'all';
    this.allResults = [];
    this.autoUpdateInterval = null;
    this.isAutoUpdateActive = false;
    this.updateCounter = 0;
    
    // Moment.js konfigürasyonu
    if (typeof moment !== 'undefined') {
      moment.locale('tr');
    }
  }

  // =====================================================
  // UTILITY FUNCTIONS
  // =====================================================
  
  showStatus(elementId, message, type = 'info', showLoader = false) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.className = 'status-message';
    
    switch(type) {
      case 'success': element.classList.add('status-success'); break;
      case 'error': element.classList.add('status-error'); break;
      case 'warning': element.classList.add('status-warning'); break;
      default: element.classList.add('status-info'); break;
    }
    
    const loader = showLoader ? '<div class="loader"></div>' : '';
    element.innerHTML = loader + message;
  }

  formatTime(timestamp) {
    if (typeof moment !== 'undefined') {
      return moment.utc(timestamp).local().format('DD.MM.YYYY HH:mm:ss');
    }
    return new Date(timestamp).toLocaleString('tr-TR');
  }

  updateSystemStatus() {
    const lastUpdateElement = document.getElementById('last-update');
    if (lastUpdateElement) {
      lastUpdateElement.textContent = new Date().toLocaleTimeString('tr-TR');
    }
  }

  // =====================================================
  // FORMATTING FUNCTIONS
  // =====================================================

  formatCSignal(cSignal) {
    if (cSignal === null || cSignal === undefined || cSignal === 'N/A') {
      return '<span class="c-signal-neutral">N/A</span>';
    }
    
    const value = parseFloat(cSignal);
    if (isNaN(value)) {
      return '<span class="c-signal-neutral">N/A</span>';
    }
    
    if (value > 0) {
      return `<span class="c-signal-positive">+${value.toFixed(2)}</span>`;
    } else if (value < 0) {
      return `<span class="c-signal-negative">${value.toFixed(2)}</span>`;
    } else {
      return '<span class="c-signal-neutral">0.00</span>';
    }
  }

  formatCSignalWithTime(cSignal, updateTime) {
    const formattedSignal = this.formatCSignal(cSignal);
    if (updateTime && updateTime !== 'N/A') {
      return `${formattedSignal}<br><small style="color: var(--text-muted); font-size: 10px;">${updateTime}</small>`;
    }
    return formattedSignal;
  }

  formatReverseMomentum(reverseMomentum, reverseType, signalStrength) {
    if (!reverseMomentum || reverseType === 'None' || reverseType === 'Normal') {
      return '<span style="color: var(--text-muted); font-size: 11px;">-</span>';
    }
    
    let strengthClass = '';
    switch(signalStrength) {
      case 'Strong': strengthClass = 'signal-strength-strong'; break;
      case 'Medium': strengthClass = 'signal-strength-medium'; break;
      case 'Weak': strengthClass = 'signal-strength-weak'; break;
      default: strengthClass = '';
    }
    
    if (reverseType === 'C↓') {
      return `<div class="reverse-momentum-indicator reverse-momentum-c-down ${strengthClass}">
        <span>📻 C↓</span>
        <small>${signalStrength}</small>
      </div>`;
    } else if (reverseType === 'C↑') {
      return `<div class="reverse-momentum-indicator reverse-momentum-c-up ${strengthClass}">
        <span>📺 C↑</span>
        <small>${signalStrength}</small>
      </div>`;
    }
    
    return '<span style="color: var(--text-muted); font-size: 11px;">-</span>';
  }

  // =====================================================
  // SYMBOL MANAGEMENT FUNCTIONS
  // =====================================================

  async loadAllSymbols() {
    try {
      this.showStatus('symbol-status', '📊 Emtia listesi yükleniyor...', 'info', true);
      
      const response = await fetch('/api/consecutive/symbols');
      const data = await response.json();
      
      if (data.success) {
        this.allSymbols = data.symbols;
        const select = document.getElementById('available-symbols');
        select.innerHTML = '';
        
        data.symbols.forEach(symbol => {
          const option = document.createElement('option');
          option.value = symbol;
          option.textContent = symbol;
          select.appendChild(option);
        });
        
        this.showStatus('symbol-status', `✅ ${data.symbols.length} Binance emtiası yüklendi`, 'success');
      } else {
        this.showStatus('symbol-status', `❌ ${data.error}`, 'error');
      }
    } catch (error) {
      console.error('Emtia listesi yükleme hatası:', error);
      this.showStatus('symbol-status', `❌ Bağlantı hatası: ${error.message}`, 'error');
    }
  }

  async loadSelectedSymbols() {
    try {
      const response = await fetch('/api/consecutive/selected-symbols');
      const data = await response.json();
      
      if (data.success) {
        this.selectedSymbols = data.symbols;
        this.updateSelectedSymbolsList(data.symbols);
        this.showStatus('symbol-status', `📊 ${data.count} seçili emtia yüklendi`, 'info');
      }
    } catch (error) {
      console.error('Seçili emtia listesi yükleme hatası:', error);
    }
  }

  updateSelectedSymbolsList(symbols) {
    const container = document.getElementById('selected-symbols-list');
    
    if (symbols.length === 0) {
      container.innerHTML = '<div style="color: var(--text-muted); font-style: italic;">Henüz emtia seçilmedi</div>';
      return;
    }
    
    container.innerHTML = '';
    symbols.forEach(symbol => {
      const item = document.createElement('div');
      item.className = 'symbol-item';
      
      item.innerHTML = `
        <span class="symbol-name">₿ ${symbol}</span>
        <button class="remove-symbol-btn" onclick="app.removeSymbol('${symbol}')">❌</button>
      `;
      container.appendChild(item);
    });
  }

  async removeSymbol(symbol) {
    try {
      const response = await fetch('/api/consecutive/selected-symbols', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'remove', symbol_to_remove: symbol })
      });
      
      const data = await response.json();
      if (data.success) {
        this.selectedSymbols = data.symbols;
        this.updateSelectedSymbolsList(data.symbols);
        this.showStatus('symbol-status', data.message, 'success');
      } else {
        this.showStatus('symbol-status', `❌ ${data.error}`, 'error');
      }
    } catch (error) {
      this.showStatus('symbol-status', `❌ ${error.message}`, 'error');
    }
  }

  async addSelectedSymbols() {
    const select = document.getElementById('available-symbols');
    const selectedSymbols = Array.from(select.selectedOptions).map(option => option.value);
    
    if (selectedSymbols.length === 0) {
      this.showStatus('symbol-status', '⚠️ Eklenecek emtia seçilmedi', 'warning');
      return;
    }
    
    try {
      const response = await fetch('/api/consecutive/selected-symbols', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'add', symbols: selectedSymbols })
      });
      
      const data = await response.json();
      if (data.success) {
        this.selectedSymbols = data.symbols;
        this.updateSelectedSymbolsList(data.symbols);
        this.showStatus('symbol-status', data.message, 'success');
      } else {
        this.showStatus('symbol-status', `❌ ${data.error}`, 'error');
      }
    } catch (error) {
      this.showStatus('symbol-status', `❌ ${error.message}`, 'error');
    }
  }

  async selectAllSymbols() {
    try {
      this.showStatus('symbol-status', '📊 Tüm emtialar seçiliyor...', 'info', true);
      
      const response = await fetch('/api/consecutive/selected-symbols', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'add_all' })
      });
      
      const data = await response.json();
      if (data.success) {
        this.selectedSymbols = data.symbols;
        this.updateSelectedSymbolsList(data.symbols);
        this.showStatus('symbol-status', data.message, 'success');
      } else {
        this.showStatus('symbol-status', `❌ ${data.error}`, 'error');
      }
    } catch (error) {
      this.showStatus('symbol-status', `❌ ${error.message}`, 'error');
    }
  }

  async clearAllSymbols() {
    if (!confirm('Tüm seçili emtiaları silmek istediğinizden emin misiniz?')) return;
    
    try {
      const response = await fetch('/api/consecutive/selected-symbols', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'clear' })
      });
      
      const data = await response.json();
      if (data.success) {
        this.selectedSymbols = data.symbols;
        this.updateSelectedSymbolsList(data.symbols);
        this.showStatus('symbol-status', data.message, 'success');
      } else {
        this.showStatus('symbol-status', `❌ ${data.error}`, 'error');
      }
    } catch (error) {
      this.showStatus('symbol-status', `❌ ${error.message}`, 'error');
    }
  }

  // =====================================================
  // FILTER FUNCTIONS
  // =====================================================

  setActiveFilter(filterType) {
    document.querySelectorAll('.btn-filter').forEach(btn => {
      btn.classList.remove('active');
    });
    
    document.getElementById(`filter-${filterType}`).classList.add('active');
    this.currentFilter = filterType;
  }

  filterResults(filterType) {
    if (!this.allResults || this.allResults.length === 0) return;
    
    let filteredResults = [];
    
    switch(filterType) {
      case 'long':
        filteredResults = this.allResults.filter(result => result.consecutive_type === 'Long');
        break;
      case 'short':
        filteredResults = this.allResults.filter(result => result.consecutive_type === 'Short');
        break;
      case 'high-count':
        filteredResults = this.allResults.filter(result => result.consecutive_count >= 5);
        break;
      default: // 'all'
        filteredResults = this.allResults;
        break;
    }
    
    // Ardışık sayıya göre tekrar sırala
    filteredResults.sort((a, b) => b.consecutive_count - a.consecutive_count);
    
    filteredResults.forEach((result, index) => {
      result.filtered_rank = index + 1;
    });
    
    this.updateConsecutiveTable(filteredResults);
    document.getElementById('filter-count').textContent = `${filteredResults.length} sonuç`;
  }

  updateConsecutiveTable(results) {
    const tbody = document.getElementById('consecutive-results-tbody');
    
    if (results.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" style="padding: 20px; text-align: center; color: var(--text-muted);">📊 Bu filtrede sonuç bulunamadı</td></tr>';
      return;
    }
    
    tbody.innerHTML = '';
    results.forEach(result => {
      const row = document.createElement('tr');
      
      // Consecutive type styling
      let consecutiveClass = '';
      let consecutiveIcon = '';
      if (result.consecutive_type === 'Long') {
        consecutiveClass = 'consecutive-long';
        consecutiveIcon = '🟢';
      } else if (result.consecutive_type === 'Short') {
        consecutiveClass = 'consecutive-short';
        consecutiveIcon = '🔴';
      } else {
        consecutiveClass = '';
        consecutiveIcon = '⚪';
      }
      
      // Percentage styling
      const percentageClass = result.percentage_change > 0 ? 'percentage-positive' : 'percentage-negative';
      
      // Count highlighting for high consecutive counts
      const countClass = result.consecutive_count >= 5 ? 'count-highlight' : '';
      
      row.innerHTML = `
        <td>${result.filtered_rank || result.rank}</td>
        <td class="symbol-clickable" onclick="window.open('${result.tradingview_link}', '_blank')" 
            title="TradingView'da aç">${result.symbol}</td>
        <td>${result.current_price}</td>
        <td class="${consecutiveClass}">${consecutiveIcon} ${result.consecutive_type}</td>
        <td class="${countClass}">${result.consecutive_count}</td>
        <td class="${percentageClass}">${result.percentage_change}%</td>
        <td style="font-size: 11px; color: var(--text-muted);">${result.last_update || '-'}</td>
      `;
      tbody.appendChild(row);
    });
  }

  // =====================================================
  // PERMANENT LIST FUNCTIONS
  // =====================================================

  displayReverseMomentumAlerts(permanentSymbols) {
    const alertsContainer = document.getElementById('reverse-momentum-alerts');
    const alertContent = document.getElementById('reverse-alert-content');
    const countSpan = document.getElementById('reverse-momentum-count');
    
    const reverseMomentumSymbols = permanentSymbols.filter(s => s.reverse_momentum);
    
    if (reverseMomentumSymbols.length === 0) {
      alertsContainer.classList.remove('show');
      return;
    }
    
    countSpan.textContent = reverseMomentumSymbols.length;
    
    let alertHtml = '';
    reverseMomentumSymbols.forEach(symbol => {
      const alertType = symbol.reverse_type === 'C↓' ? '📻' : '📺';
      const alertClass = symbol.reverse_type === 'C↓' ? 'reverse-momentum-c-down' : 'reverse-momentum-c-up';
      
      alertHtml += `
        <div class="reverse-alert-item">
          <strong onclick="window.open('${symbol.tradingview_link}', '_blank')" 
                  style="cursor: pointer; color: var(--primary-color); text-decoration: underline;">
            ${symbol.symbol}
          </strong>
          <span class="reverse-momentum-indicator ${alertClass}" style="margin: 0 8px;">
            ${alertType} ${symbol.reverse_type} - ${symbol.signal_strength}
          </span>
          <span style="color: var(--text-muted); font-size: 11px;">
            ${symbol.alert_message}
          </span>
        </div>
      `;
    });
    
    alertContent.innerHTML = alertHtml;
    alertsContainer.classList.add('show');
  }

  async loadPermanentHighConsecutive() {
    try {
      const response = await fetch('/api/consecutive/permanent-high-consecutive');
      const data = await response.json();
      
      if (data.success) {
        this.updatePermanentTable(data.permanent_symbols);
        this.displayReverseMomentumAlerts(data.permanent_symbols);
        
        const reverseMomentumCount = data.reverse_momentum_count || 0;
        const statusMessage = reverseMomentumCount > 0 
          ? `🏆 ${data.count} kalıcı emtia + 🚨 ${reverseMomentumCount} TERS MOMENTUM tespit edildi!`
          : `🏆 ${data.count} kalıcı emtia yüklendi + C-Signal hesaplandı`;
        
        this.showStatus('permanent-status', statusMessage, reverseMomentumCount > 0 ? 'warning' : 'success');
        
        // Telegram durumunu güncelle
        const telegramStatus = document.getElementById('telegram-status');
        if (telegramStatus) {
          telegramStatus.textContent = data.telegram_status || 'Bilinmiyor';
        }
      } else {
        this.showStatus('permanent-status', `❌ ${data.error}`, 'error');
      }
    } catch (error) {
      console.error('Kalıcı liste yükleme hatası:', error);
      this.showStatus('permanent-status', `❌ Bağlantı hatası: ${error.message}`, 'error');
    }
  }

  updatePermanentTable(permanentSymbols) {
    const tbody = document.getElementById('permanent-results-tbody');
    
    if (permanentSymbols.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" style="padding: 20px; text-align: center; color: var(--text-muted);">🏆 Henüz 5+ ardışık olan emtia yok</td></tr>';
      return;
    }
    
    tbody.innerHTML = '';
    permanentSymbols.forEach(symbol => {
      const row = document.createElement('tr');
      
      // TERS MOMENTUM KONTROLÜ - SATIR VURGULAMA
      if (symbol.reverse_momentum) {
        row.classList.add('reverse-momentum-alert');
      }
      
      // Type styling
      let typeClass = '';
      let typeIcon = '';
      if (symbol.max_consecutive_type === 'Long') {
        typeClass = 'consecutive-long';
        typeIcon = '🟢';
      } else if (symbol.max_consecutive_type === 'Short') {
        typeClass = 'consecutive-short';
        typeIcon = '🔴';
      }
      
      // Percentage styling
      const percentageClass = symbol.max_percentage_change > 0 ? 'percentage-positive' : 'percentage-negative';
      
      // Highlight high consecutive counts
      const countClass = symbol.max_consecutive_count >= 7 ? 'count-highlight' : '';
      
      const cSignalWithTime = this.formatCSignalWithTime(symbol.c_signal, symbol.c_signal_update_time);
      const reverseMomentumFormatted = this.formatReverseMomentum(symbol.reverse_momentum, symbol.reverse_type, symbol.signal_strength);
      
      row.innerHTML = `
        <td>${symbol.rank}</td>
        <td class="symbol-clickable" onclick="window.open('${symbol.tradingview_link}', '_blank')" 
            title="TradingView'da aç">🏆 ${symbol.symbol}</td>
        <td style="font-size: 11px; color: var(--text-muted);">${symbol.first_date}</td>
        <td class="${countClass}">${symbol.max_consecutive_count}</td>
        <td class="${typeClass}">${typeIcon} ${symbol.max_consecutive_type}</td>
        <td class="${percentageClass}">${symbol.max_percentage_change}%</td>
        <td style="font-size: 11px; color: var(--text-muted);">${symbol.timeframe}</td>
        <td>${cSignalWithTime}</td>
        <td>${reverseMomentumFormatted}</td>
      `;
      tbody.appendChild(row);
    });
  }

  async clearPermanentList() {
    if (!confirm('Kalıcı 5+ ardışık listesini temizlemek istediğinizden emin misiniz?')) return;
    
    try {
      this.showStatus('permanent-status', '🗑️ Kalıcı liste temizleniyor...', 'warning', true);
      
      const response = await fetch('/api/consecutive/clear-permanent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      if (data.success) {
        this.updatePermanentTable([]);
        document.getElementById('reverse-momentum-alerts').classList.remove('show');
        this.showStatus('permanent-status', data.message, 'success');
      } else {
        this.showStatus('permanent-status', `❌ ${data.error}`, 'error');
      }
    } catch (error) {
      this.showStatus('permanent-status', `❌ ${error.message}`, 'error');
    }
  }

  // =====================================================
  // ANALYSIS FUNCTIONS
  // =====================================================

  async startConsecutiveAnalysis(isAutoUpdate = false) {
    if (this.selectedSymbols.length === 0 && !isAutoUpdate) {
      this.showStatus('analysis-status', '⚠️ Analiz için emtia seçmelisiniz', 'warning');
      return;
    }
    
    if (isAutoUpdate && this.selectedSymbols.length === 0) {
      console.log('Auto-update: Seçili emtia yok, analiz atlanıyor');
      return;
    }
    
    const timeframe = document.getElementById('timeframe').value;
    
    try {
      if (!isAutoUpdate) {
        this.showStatus('analysis-status', `🎯 ${this.selectedSymbols.length} emtia için ${timeframe} ardışık mum analizi başlatılıyor...`, 'info', true);
      } else {
        this.updateCounter++;
        this.showStatus('analysis-status', `🔄 Otomatik güncelleme #${this.updateCounter} - ${this.selectedSymbols.length} emtia analiz ediliyor + C-Signal + Ters momentum...`, 'info', true);
      }
      
      const response = await fetch('/api/consecutive/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ timeframe: timeframe })
      });
      
      const data = await response.json();
      if (data.success) {
        console.log(`📊 API'den ${data.results.length} sonuç alındı`);
        
        this.allResults = data.results;
        this.currentResults = data.results;
        
        // Mevcut filtreyi koru
        this.filterResults(this.currentFilter);
        
        // Kalıcı listeyi de güncelle
        this.loadPermanentHighConsecutive();
        
        // Sistem durumunu güncelle
        this.updateSystemStatus();
        
        if (!isAutoUpdate) {
          // İlk analiz tamamlandı, otomatik döngüyü başlat
          this.startAutoUpdateLoop();
          const now = new Date().toLocaleTimeString('tr-TR');
          this.showStatus('analysis-status', `✅ Analiz başlatıldı - Her 30 saniyede otomatik güncellenecek + Ters momentum tespiti (${now})`, 'success');
        } else {
          const now = new Date().toLocaleTimeString('tr-TR');
          this.showStatus('analysis-status', `🔄 Otomatik güncelleme aktif - Son: ${now} (${data.results.length} emtia + Ters momentum)`, 'info');
        }
      } else {
        this.showStatus('analysis-status', `❌ ${data.error}`, 'error');
      }
    } catch (error) {
      console.error('Ardışık analiz hatası:', error);
      if (!isAutoUpdate) {
        this.showStatus('analysis-status', `❌ ${error.message}`, 'error');
      } else {
        this.showStatus('analysis-status', `❌ Otomatik güncelleme hatası: ${error.message}`, 'error');
      }
    }
  }

  startAutoUpdateLoop() {
    if (this.isAutoUpdateActive) return;
    
    this.isAutoUpdateActive = true;
    this.updateCounter = 0;
    
    // 30 saniyede bir güncelleme döngüsü
    this.autoUpdateInterval = setInterval(() => {
      if (this.selectedSymbols.length > 0) {
        this.startConsecutiveAnalysis(true);
      }
    }, 30000); // 30 saniye
    
    // Buton durumlarını güncelle
    document.getElementById('start-analysis').disabled = true;
    document.getElementById('stop-auto-update').disabled = false;
  }

  stopAutoUpdate() {
    if (!this.isAutoUpdateActive) return;
    
    this.isAutoUpdateActive = false;
    
    if (this.autoUpdateInterval) {
      clearInterval(this.autoUpdateInterval);
      this.autoUpdateInterval = null;
    }
    
    // Buton durumlarını güncelle
    document.getElementById('start-analysis').disabled = false;
    document.getElementById('stop-auto-update').disabled = true;
    
    this.showStatus('analysis-status', `⏹️ Otomatik güncelleme durduruldu (Toplam ${this.updateCounter} güncelleme yapıldı)`, 'warning');
  }

  // =====================================================
  // TELEGRAM FUNCTIONS
  // =====================================================

  async testTelegram() {
    try {
      this.showStatus('analysis-status', '🧪 Telegram bot test ediliyor...', 'info', true);
      
      const response = await fetch('/api/telegram/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      if (data.success) {
        this.showStatus('analysis-status', `✅ ${data.message}`, 'success');
      } else {
        this.showStatus('analysis-status', `❌ ${data.error}`, 'error');
      }
    } catch (error) {
      this.showStatus('analysis-status', `❌ Telegram test hatası: ${error.message}`, 'error');
    }
  }

  // =====================================================
  // EVENT LISTENERS
  // =====================================================

  bindEventListeners() {
    // Symbol management
    document.getElementById('refresh-symbols').addEventListener('click', () => this.loadAllSymbols());
    document.getElementById('add-symbols').addEventListener('click', () => this.addSelectedSymbols());
    document.getElementById('select-all').addEventListener('click', () => this.selectAllSymbols());
    document.getElementById('clear-symbols').addEventListener('click', () => this.clearAllSymbols());
    
    // Analysis
    document.getElementById('start-analysis').addEventListener('click', () => this.startConsecutiveAnalysis(false));
    document.getElementById('stop-auto-update').addEventListener('click', () => this.stopAutoUpdate());
    document.getElementById('test-telegram').addEventListener('click', () => this.testTelegram());
    
    // Permanent list
    document.getElementById('refresh-permanent').addEventListener('click', () => this.loadPermanentHighConsecutive());
    document.getElementById('clear-permanent').addEventListener('click', () => this.clearPermanentList());
    
    // Filter buttons
    document.getElementById('filter-all').addEventListener('click', () => {
      this.setActiveFilter('all');
      this.filterResults('all');
    });
    
    document.getElementById('filter-long').addEventListener('click', () => {
      this.setActiveFilter('long');
      this.filterResults('long');
    });
    
    document.getElementById('filter-short').addEventListener('click', () => {
      this.setActiveFilter('short');
      this.filterResults('short');
    });
    
    document.getElementById('filter-high-count').addEventListener('click', () => {
      this.setActiveFilter('high-count');
      this.filterResults('high-count');
    });
    
    // Sayfa kapatılırken otomatik güncellemeyi durdur
    window.addEventListener('beforeunload', () => {
      if (this.isAutoUpdateActive) {
        this.stopAutoUpdate();
      }
    });
  }

  // =====================================================
  // INITIALIZATION
  // =====================================================

  async init() {
    console.log('🚨 Consecutive Candles + C-Signal + Reverse Momentum System initializing...');
    
    try {
      // Paralel yükleme
      await Promise.all([
        this.loadAllSymbols(),
        this.loadSelectedSymbols(),
        this.loadPermanentHighConsecutive()
      ]);
      
      this.bindEventListeners();
      this.updateSystemStatus();
      
      console.log('✅ Consecutive Candles + C-Signal + Reverse Momentum System initialized');
      
    } catch (error) {
      console.error('❌ Sistem başlatma hatası:', error);
    }
  }
}

// Global app instance
let app;

// DOM yüklendiğinde uygulamayı başlat
document.addEventListener('DOMContentLoaded', () => {
  app = new AnalysisApp();
  app.init();
});

// Global fonksiyonlar (HTML'den çağrılabilir)
window.app = app;