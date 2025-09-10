/**
 * Ardışık Mum + C-Signal + Ters Momentum Analiz Sistemi
 * Frontend JavaScript Application - SYNTAX HATALARI DÜZELTİLDİ
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
  // ARAMA FONKSİYONU - TAMAMEN YENİDEN YAZILDI
  // =====================================================
  
  searchSymbols(searchTerm) {
    console.log('🔍 Arama yapılıyor:', searchTerm);
    
    // Arama terimini temizle
    const cleanSearchTerm = (searchTerm || '').trim().toUpperCase();
    
    // Boş arama - tüm sonuçları göster
    if (!cleanSearchTerm) {
      console.log('📊 Boş arama - tüm sonuçlar gösteriliyor');
      this.filterResults(this.currentFilter);
      return;
    }
    
    // Mevcut sonuçlar yoksa çık
    if (!this.allResults || this.allResults.length === 0) {
      console.log('⚠️ Arama için veri yok');
      return;
    }
    
    // ARAMA ALGORITMASI
    const searchResults = [];
    const nonMatchResults = [];
    
    this.allResults.forEach(result => {
      const symbol = (result.symbol || '').toUpperCase();
      
      // Eşleşme kontrolü - sembol içinde arama terimi var mı?
      if (symbol.includes(cleanSearchTerm)) {
        // Eşleşen sonuçlar - öncelikli liste
        result.shouldHighlight = true; // Vurgulama için işaret
        searchResults.push(result);
        console.log('✅ Eşleşme bulundu:', symbol);
      } else {
        // Eşleşmeyen sonuçlar - arkada tutulacak
        result.shouldHighlight = false;
        nonMatchResults.push(result);
      }
    });
    
    // SIRALAMA: Önce eşleşenler, sonra eşleşmeyenler
    // Eşleşenler kendi içinde ardışık sayıya göre sıralanır
    searchResults.sort((a, b) => b.consecutive_count - a.consecutive_count);
    nonMatchResults.sort((a, b) => b.consecutive_count - a.consecutive_count);
    
    // Birleştir: Eşleşenler önce, geri kalanlar arkada
    const finalResults = [...searchResults, ...nonMatchResults];
    
    // Rank'ları yeniden ayarla
    finalResults.forEach((result, index) => {
      result.filtered_rank = index + 1;
    });
    
    // Tabloyu güncelle
    this.updateConsecutiveTable(finalResults);
    
    // Sonuç sayısını güncelle
    const matchCount = searchResults.length;
    const totalCount = finalResults.length;
    const filterCountElement = document.getElementById('filter-count');
    if (filterCountElement) {
      filterCountElement.textContent = matchCount > 0 
        ? `${matchCount} eşleşme / ${totalCount} toplam (arama: "${cleanSearchTerm}")` 
        : `Arama bulunamadı: "${cleanSearchTerm}" / ${totalCount} toplam`;
    }
    
    console.log(`🎯 Arama tamamlandı: ${matchCount} eşleşme bulundu`);
  }

  // =====================================================
  // UTILITY FUNCTIONS
  // =====================================================
  
  showStatus(elementId, message, type = 'info', showLoader = false) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.className = 'status-message';
    
    switch(type) {
      case 'success': 
        element.classList.add('status-success'); 
        break;
      case 'error': 
        element.classList.add('status-error'); 
        break;
      case 'warning': 
        element.classList.add('status-warning'); 
        break;
      default: 
        element.classList.add('status-info'); 
        break;
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
      case 'Strong': 
        strengthClass = 'signal-strength-strong'; 
        break;
      case 'Medium': 
        strengthClass = 'signal-strength-medium'; 
        break;
      case 'Weak': 
        strengthClass = 'signal-strength-weak'; 
        break;
      default: 
        strengthClass = '';
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
        if (select) {
          select.innerHTML = '';
          
          data.symbols.forEach(symbol => {
            const option = document.createElement('option');
            option.value = symbol;
            option.textContent = symbol;
            select.appendChild(option);
          });
        }
        
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
    if (!container) return;
    
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
    if (!select) return;
    
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
  // MANUEL KALICI LISTE EKLEME/ÇIKARMA FONKSİYONLARI
  // =====================================================

  async addToPermanentList(symbol, timeframe = '4h') {
    try {
      // Butonu disable et
      const addButton = document.querySelector(`[data-symbol="${symbol}"]`);
      if (addButton) {
        addButton.disabled = true;
        addButton.innerHTML = '⏳ Ekleniyor...';
      }
      
      const response = await fetch('/api/consecutive/add-to-permanent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          symbol: symbol, 
          timeframe: timeframe 
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Başarı mesajı göster
        this.showStatus('analysis-status', data.message, 'success');
        
        // Kalıcı listeyi yenile
        this.loadPermanentHighConsecutive();
        
        // Butonu güncelle
        if (addButton) {
          addButton.innerHTML = '✅ Eklendi!';
          addButton.classList.add('btn-success');
          addButton.classList.remove('btn-warning');
          
          // 3 saniye sonra buton metnini geri değiştir
          setTimeout(() => {
            addButton.innerHTML = '🏆 Kalıcı Listede';
            addButton.disabled = true;
          }, 3000);
        }
        
      } else {
        this.showStatus('analysis-status', `❌ ${data.error}`, 'error');
        
        // Buton durumunu eski haline getir
        if (addButton) {
          addButton.disabled = false;
          addButton.innerHTML = '🏆 Kalıcı Listeye Ekle';
        }
      }
      
    } catch (error) {
      console.error('Manuel ekleme hatası:', error);
      this.showStatus('analysis-status', `❌ Ekleme hatası: ${error.message}`, 'error');
      
      // Buton durumunu eski haline getir
      const addButton = document.querySelector(`[data-symbol="${symbol}"]`);
      if (addButton) {
        addButton.disabled = false;
        addButton.innerHTML = '🏆 Kalıcı Listeye Ekle';
      }
    }
  }

  async removePermanentSymbol(symbol) {
    try {
      if (!confirm(`${symbol} emtiasını kalıcı listeden çıkarmak istediğinizden emin misiniz?`)) {
        return;
      }

      const response = await fetch('/api/consecutive/remove-from-permanent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: symbol })
      });

      const data = await response.json();

      if (data.success) {
        this.showStatus('permanent-status', data.message, 'success');
        this.loadPermanentHighConsecutive();
      } else {
        this.showStatus('permanent-status', `❌ ${data.error}`, 'error');
      }

    } catch (error) {
      console.error('Kalıcı listeden çıkarma hatası:', error);
      this.showStatus('permanent-status', `❌ Çıkarma hatası: ${error.message}`, 'error');
    }
  }

  // =====================================================
  // FILTER FUNCTIONS
  // =====================================================

  setActiveFilter(filterType) {
    document.querySelectorAll('.btn-filter').forEach(btn => {
      btn.classList.remove('active');
    });
    
    const filterButton = document.getElementById(`filter-${filterType}`);
    if (filterButton) {
      filterButton.classList.add('active');
    }
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
      // Arama vurgulamasını koru ama reset et
      result.shouldHighlight = false;
    });
    
    this.updateConsecutiveTable(filteredResults);
    const filterCountElement = document.getElementById('filter-count');
    if (filterCountElement) {
      filterCountElement.textContent = `${filteredResults.length} sonuç`;
    }
  }

  updateConsecutiveTable(results) {
    const tbody = document.getElementById('consecutive-results-tbody');
    if (!tbody) return;
    
    if (results.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" style="padding: 20px; text-align: center; color: var(--text-muted);">📊 Bu filtrede sonuç bulunamadı</td></tr>';
      return;
    }
    
    tbody.innerHTML = '';
    results.forEach(result => {
      const row = document.createElement('tr');
      
      // VURGULAMA - Arama sonucu varsa sarı arka plan
      if (result.shouldHighlight) {
        row.style.backgroundColor = 'rgba(255, 235, 59, 0.3)'; // Sarı vurgu
        row.style.border = '2px solid #ffeb3b';
        row.style.boxShadow = '0 0 10px rgba(255, 235, 59, 0.5)';
      }
      
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
      
      // Timeframe element'ini güvenli bir şekilde al
      const timeframeElement = document.getElementById('timeframe');
      const currentTimeframe = timeframeElement ? timeframeElement.value : '4h';
      
      // MANUEL EKLEME BUTONU
      const addButtonHtml = `
        <button class="btn btn-warning" 
                style="font-size: 10px; padding: 4px 8px;" 
                data-symbol="${result.symbol}"
                onclick="app.addToPermanentList('${result.symbol}', '${currentTimeframe}')"
                title="Bu emtiayı kalıcı listeye manuel ekle">
          🏆 Kalıcı Listeye Ekle
        </button>
      `;
      
      row.innerHTML = `
        <td>${result.filtered_rank || result.rank}</td>
        <td class="symbol-clickable" onclick="window.open('${result.tradingview_link}', '_blank')" 
            title="TradingView'da aç">${result.symbol}</td>
        <td>${result.current_price}</td>
        <td class="${consecutiveClass}">${consecutiveIcon} ${result.consecutive_type}</td>
        <td class="${countClass}">${result.consecutive_count}</td>
        <td class="${percentageClass}">${result.percentage_change}%</td>
        <td style="font-size: 11px; color: var(--text-muted);">${result.last_update || '-'}</td>
        <td style="text-align: center;">${addButtonHtml}</td>
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
    
    if (!alertsContainer || !alertContent || !countSpan) return;
    
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
    if (!tbody) return;
    
    if (permanentSymbols.length === 0) {
      tbody.innerHTML = '<tr><td colspan="10" style="padding: 20px; text-align: center; color: var(--text-muted);">🏆 Henüz 5+ ardışık olan emtia yok</td></tr>';
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
        <td style="text-align: center;">
          <button class="btn btn-danger" 
                  style="font-size: 9px; padding: 2px 6px;" 
                  onclick="app.removePermanentSymbol('${symbol.symbol}')"
                  title="Bu emtiayı kalıcı listeden çıkar">
            🗑️ Çıkar
          </button>
        </td>
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
        const alertsContainer = document.getElementById('reverse-momentum-alerts');
        if (alertsContainer) {
          alertsContainer.classList.remove('show');
        }
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
    
    const timeframeElement = document.getElementById('timeframe');
    const timeframe = timeframeElement ? timeframeElement.value : '4h';
    
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
    const startButton = document.getElementById('start-analysis');
    const stopButton = document.getElementById('stop-auto-update');
    if (startButton) startButton.disabled = true;
    if (stopButton) stopButton.disabled = false;
  }

  stopAutoUpdate() {
    if (!this.isAutoUpdateActive) return;
    
    this.isAutoUpdateActive = false;
    
    if (this.autoUpdateInterval) {
      clearInterval(this.autoUpdateInterval);
      this.autoUpdateInterval = null;
    }
    
    // Buton durumlarını güncelle
    const startButton = document.getElementById('start-analysis');
    const stopButton = document.getElementById('stop-auto-update');
    if (startButton) startButton.disabled = false;
    if (stopButton) stopButton.disabled = true;
    
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
    const refreshSymbolsBtn = document.getElementById('refresh-symbols');
    if (refreshSymbolsBtn) {
      refreshSymbolsBtn.addEventListener('click', () => this.loadAllSymbols());
    }

    const addSymbolsBtn = document.getElementById('add-symbols');
    if (addSymbolsBtn) {
      addSymbolsBtn.addEventListener('click', () => this.addSelectedSymbols());
    }

    const selectAllBtn = document.getElementById('select-all');
    if (selectAllBtn) {
      selectAllBtn.addEventListener('click', () => this.selectAllSymbols());
    }

    const clearSymbolsBtn = document.getElementById('clear-symbols');
    if (clearSymbolsBtn) {
      clearSymbolsBtn.addEventListener('click', () => this.clearAllSymbols());
    }
    
    // Analysis
    const startAnalysisBtn = document.getElementById('start-analysis');
    if (startAnalysisBtn) {
      startAnalysisBtn.addEventListener('click', () => this.startConsecutiveAnalysis(false));
    }

    const stopAutoUpdateBtn = document.getElementById('stop-auto-update');
    if (stopAutoUpdateBtn) {
      stopAutoUpdateBtn.addEventListener('click', () => this.stopAutoUpdate());
    }
    
    // Permanent list
    const refreshPermanentBtn = document.getElementById('refresh-permanent');
    if (refreshPermanentBtn) {
      refreshPermanentBtn.addEventListener('click', () => this.loadPermanentHighConsecutive());
    }

    const clearPermanentBtn = document.getElementById('clear-permanent');
    if (clearPermanentBtn) {
      clearPermanentBtn.addEventListener('click', () => this.clearPermanentList());
    }
    
    // Filter buttons
    const filterAllBtn = document.getElementById('filter-all');
    if (filterAllBtn) {
      filterAllBtn.addEventListener('click', () => {
        this.setActiveFilter('all');
        this.filterResults('all');
      });
    }
    
    const filterLongBtn = document.getElementById('filter-long');
    if (filterLongBtn) {
      filterLongBtn.addEventListener('click', () => {
        this.setActiveFilter('long');
        this.filterResults('long');
      });
    }
    
    const filterShortBtn = document.getElementById('filter-short');
    if (filterShortBtn) {
      filterShortBtn.addEventListener('click', () => {
        this.setActiveFilter('short');
        this.filterResults('short');
      });
    }
    
    const filterHighCountBtn = document.getElementById('filter-high-count');
    if (filterHighCountBtn) {
      filterHighCountBtn.addEventListener('click', () => {
        this.setActiveFilter('high-count');
        this.filterResults('high-count');
      });
    }
    
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
      
      // ARAMA KUTUSU EVENT LISTENER - DOM hazır olduktan sonra
      setTimeout(() => {
        const searchInput = document.getElementById('symbol-search');
        if (searchInput) {
          console.log('🔍 Arama kutusu bulundu, event listener ekleniyor...');
          
          // Gerçek zamanlı arama - her karakter girildiğinde
          searchInput.addEventListener('input', (e) => {
            const searchValue = e.target.value;
            console.log('🔍 Arama yapılıyor:', searchValue);
            this.searchSymbols(searchValue);
          });
          
          // Enter tuşu ile arama
          searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
              console.log('🔍 Enter ile arama:', e.target.value);
              this.searchSymbols(e.target.value);
            }
          });
          
          // Focus/blur events
          searchInput.addEventListener('focus', () => {
            console.log('🔍 Arama kutusu odaklandı');
          });
          
          searchInput.addEventListener('blur', () => {
            console.log('🔍 Arama kutusu odağı kaybetti');
          });
          
          console.log('✅ Arama kutusu event listener\'ları başarıyla eklendi');
        } else {
          console.error('❌ Arama kutusu bulunamadı! symbol-search id\'li element mevcut değil');
        }
      }, 500); // 500ms bekle ki DOM tamamen hazır olsun
      
    } catch (error) {
      console.error('❌ Sistem başlatma hatası:', error);
    }
  }
}

// Global app instance
let app;

// DOM yüklendiğinde uygulamayı başlat
document.addEventListener('DOMContentLoaded', () => {
  console.log('🚀 DOM yüklendi, app başlatılıyor...');
  app = new AnalysisApp();
  app.init();
});

// Global fonksiyonlar (HTML'den çağrılabilir)
if (typeof window !== 'undefined') {
  window.app = app;
}