/**
 * Supertrend + C-Signal Analiz Sistemi
 * Frontend JavaScript Application
 * 🆕 YENİ: Threshold Yönetimi
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
    
    // 🆕 Minimum ratio threshold
    this.minRatioThreshold = 100; // Varsayılan
    
    // Moment.js konfigürasyonu
    if (typeof moment !== 'undefined') {
      moment.locale('tr');
    }
    
    // Ayarları yükle
    this.loadSettings();
  }

  // =====================================================
  // 🆕 SETTINGS MANAGEMENT
  // =====================================================
  
  loadSettings() {
    /**
     * LocalStorage'dan ayarları yükle
     */
    try {
      const savedThreshold = localStorage.getItem('minRatioThreshold');
      if (savedThreshold) {
        this.minRatioThreshold = parseFloat(savedThreshold);
        
        // Input ve display'leri güncelle
        const inputElement = document.getElementById('min-ratio-threshold');
        const displayElement = document.getElementById('current-threshold-display');
        
        if (inputElement) {
          inputElement.value = this.minRatioThreshold;
        }
        
        if (displayElement) {
          displayElement.textContent = this.minRatioThreshold;
        }
        
        this.updateSettingsStatus();
        
        console.log(`⚙️ Ayarlar yüklendi: Min Ratio = ${this.minRatioThreshold}%`);
      }
    } catch (error) {
      console.error('Ayar yükleme hatası:', error);
      this.minRatioThreshold = 100; // Varsayılana dön
    }
  }
  
  saveSettings() {
    /**
     * Ayarları LocalStorage'a kaydet
     */
    try {
      const inputElement = document.getElementById('min-ratio-threshold');
      if (!inputElement) return;
      
      const newThreshold = parseFloat(inputElement.value);
      
      // Validasyon
      if (isNaN(newThreshold) || newThreshold < 0) {
        this.showStatus('settings-status', '❌ Geçersiz değer! 0 veya üzeri bir sayı girin.', 'error');
        return;
      }
      
      // Çok yüksek değer kontrolü
      if (newThreshold > 1000) {
        this.showStatus('settings-status', '❌ Çok yüksek değer! Maksimum 1000% olmalı.', 'error');
        return;
      }
      
      // Kaydet
      this.minRatioThreshold = newThreshold;
      localStorage.setItem('minRatioThreshold', newThreshold.toString());
      
      // Display'i güncelle
      const displayElement = document.getElementById('current-threshold-display');
      if (displayElement) {
        displayElement.textContent = newThreshold;
      }
      
      this.updateSettingsStatus();
      this.showStatus('settings-status', 
        `✅ Ayarlar kaydedildi! Yeni minimum ratio: ${newThreshold}%`, 
        'success');
      
      console.log(`💾 Ayarlar kaydedildi: Min Ratio = ${newThreshold}%`);
      
      // Sunucuya bildir
      this.updateServerThreshold(newThreshold);
      
    } catch (error) {
      console.error('Ayar kaydetme hatası:', error);
      this.showStatus('settings-status', `❌ Kaydetme hatası: ${error.message}`, 'error');
    }
  }
  
  updateSettingsStatus() {
    /**
     * Ayarlar status mesajını güncelle
     */
    const statusElement = document.getElementById('settings-status');
    if (statusElement) {
      statusElement.innerHTML = `⚙️ Mevcut minimum ratio eşiği: <strong>${this.minRatioThreshold}%</strong> 
        <br><small style="color: var(--text-muted);">Bu değerin üzerindeki emtialar otomatik kalıcı listeye eklenir</small>`;
    }
  }
  
  async updateServerThreshold(threshold) {
    /**
     * Sunucuya yeni threshold değerini gönder
     */
    try {
      const response = await fetch('/api/consecutive/update-threshold', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ min_ratio_threshold: threshold })
      });
      
      const data = await response.json();
      
      if (data.success) {
        console.log('✅ Sunucu threshold güncellendi:', data.threshold);
      } else {
        console.warn('⚠️ Sunucu threshold güncellenemedi:', data.error);
        this.showStatus('settings-status', `⚠️ Sunucu hatası: ${data.error}`, 'warning');
      }
    } catch (error) {
      console.warn('⚠️ Sunucu threshold güncelleme hatası:', error);
    }
  }
  
  getMinRatioThreshold() {
    /**
     * Mevcut minimum ratio threshold'u döndür
     */
    return this.minRatioThreshold;
  }

  // =====================================================
  // ARAMA FONKSİYONU - SIRALAMA KORUYARAK VURGULAMA VE KAYDIRMA
  // =====================================================
  
  searchSymbols(searchTerm) {
    console.log('🔍 Arama yapılıyor:', searchTerm);
    
    const cleanSearchTerm = (searchTerm || '').trim().toUpperCase();
    
    if (!cleanSearchTerm) {
      console.log('📊 Boş arama - tüm sonuçlar gösteriliyor');
      this.filterResults(this.currentFilter);
      return;
    }
    
    if (!this.allResults || this.allResults.length === 0) {
      console.log('⚠️ Arama için veri yok');
      return;
    }
    
    let matchCount = 0;
    let firstMatchIndex = -1;
    let filteredResults = [];
    
    switch(this.currentFilter) {
      case 'bullish':
        filteredResults = this.allResults.filter(result => result.trend_direction === 'Bullish');
        break;
      case 'bearish':
        filteredResults = this.allResults.filter(result => result.trend_direction === 'Bearish');
        break;
      case 'high-ratio':
        // 🆕 Dinamik threshold kullanımı
        filteredResults = this.allResults.filter(result => Math.abs(result.ratio_percent) >= this.minRatioThreshold);
        break;
      default:
        filteredResults = [...this.allResults];
        break;
    }
    
    filteredResults.forEach((result, index) => {
      const symbol = (result.symbol || '').toUpperCase();
      
      if (symbol.includes(cleanSearchTerm)) {
        result.shouldHighlight = true;
        matchCount++;
        if (firstMatchIndex === -1) {
          firstMatchIndex = index;
        }
        console.log('✅ Eşleşme bulundu:', symbol, 'sıra:', index + 1);
      } else {
        result.shouldHighlight = false;
      }
      
      result.filtered_rank = index + 1;
    });
    
    this.updateSupertrendTable(filteredResults);
    
    if (firstMatchIndex !== -1) {
      setTimeout(() => {
        const tbody = document.getElementById('consecutive-results-tbody');
        if (tbody && tbody.children.length > firstMatchIndex) {
          const targetRow = tbody.children[firstMatchIndex];
          if (targetRow) {
            targetRow.scrollIntoView({
              behavior: 'smooth',
              block: 'center'
            });
            console.log(`📍 İlk eşleşmeye kaydırıldı: sıra ${firstMatchIndex + 1}`);
          }
        }
      }, 100);
    }
    
    const totalCount = filteredResults.length;
    const filterCountElement = document.getElementById('filter-count');
    if (filterCountElement) {
      filterCountElement.textContent = matchCount > 0 
        ? `${matchCount} eşleşme / ${totalCount} toplam (arama: "${cleanSearchTerm}")` 
        : `Arama bulunamadı: "${cleanSearchTerm}" / ${totalCount} toplam`;
    }
    
    console.log(`🎯 Arama tamamlandı: ${matchCount} eşleşme bulundu, sıralama korundu`);
  }

  // =====================================================
  // ARAMA ÖNERİLERİ VE OTOMATIK TAMAMLAMA
  // =====================================================

  showSearchSuggestions(searchTerm) {
    if (!this.allResults || this.allResults.length === 0) return;
    
    const searchValue = searchTerm.toUpperCase();
    const dropdown = document.getElementById('search-dropdown');
    
    if (!dropdown) return;
    
    const matches = this.allResults
      .filter(result => result.symbol.toUpperCase().startsWith(searchValue))
      .slice(0, 10);
    
    if (matches.length === 0) {
      this.hideSearchDropdown();
      return;
    }
    
    let dropdownHTML = '';
    matches.forEach(result => {
      const highlightedSymbol = result.symbol.replace(
        new RegExp(`(${searchValue})`, 'gi'), 
        '<strong style="color: #ffeb3b;">$1</strong>'
      );
      
      const trendIcon = result.trend_direction === 'Bullish' ? '🟢' : 
                       result.trend_direction === 'Bearish' ? '🔴' : '⚪';
      
      dropdownHTML += `
        <div class="search-suggestion-item" 
             onclick="app.selectSearchSuggestion('${result.symbol}')"
             style="padding: 8px 12px; cursor: pointer; border-bottom: 1px solid var(--border-color); transition: all 0.2s ease; display: flex; justify-content: space-between; align-items: center;">
          <div>
            <div style="font-weight: bold; color: var(--primary-color); font-size: 14px;">${highlightedSymbol}</div>
            <div style="font-size: 11px; color: var(--text-muted);">
              ${trendIcon} ${result.trend_direction} - Ratio: ${result.ratio_percent}% - Z: ${result.z_score}
            </div>
          </div>
          <div style="font-size: 10px; color: var(--text-muted);">
            ${result.current_price}
          </div>
        </div>
      `;
    });
    
    dropdown.innerHTML = dropdownHTML;
    dropdown.style.display = 'block';
    
    dropdown.querySelectorAll('.search-suggestion-item').forEach(item => {
      item.addEventListener('mouseenter', () => {
        item.style.backgroundColor = 'rgba(41, 98, 255, 0.1)';
      });
      item.addEventListener('mouseleave', () => {
        item.style.backgroundColor = 'transparent';
      });
    });
  }

  selectSearchSuggestion(symbol) {
    const searchInput = document.getElementById('symbol-search');
    if (searchInput) {
      searchInput.value = symbol;
      this.searchSymbols(symbol);
      this.hideSearchDropdown();
    }
  }

  hideSearchDropdown() {
    const dropdown = document.getElementById('search-dropdown');
    if (dropdown) {
      dropdown.style.display = 'none';
      dropdown.innerHTML = '';
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
  // MANUEL KALICI LISTE EKLEME/ÇIKARMA
  // =====================================================

  async addToPermanentList(symbol, timeframe = '4h') {
    try {
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
        this.showStatus('analysis-status', data.message, 'success');
        this.loadPermanentHighRatio();
        
        if (addButton) {
          addButton.innerHTML = '✅ Eklendi!';
          addButton.classList.add('btn-success');
          addButton.classList.remove('btn-warning');
          
          setTimeout(() => {
            addButton.innerHTML = '🏆 Kalıcı Listede';
            addButton.disabled = true;
          }, 3000);
        }
      } else {
        this.showStatus('analysis-status', `❌ ${data.error}`, 'error');
        
        if (addButton) {
          addButton.disabled = false;
          addButton.innerHTML = '🏆 Kalıcı Listeye Ekle';
        }
      }
    } catch (error) {
      console.error('Manuel ekleme hatası:', error);
      this.showStatus('analysis-status', `❌ Ekleme hatası: ${error.message}`, 'error');
      
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
        this.loadPermanentHighRatio();
      } else {
        this.showStatus('permanent-status', `❌ ${data.error}`, 'error');
      }
    } catch (error) {
      console.error('Kalıcı listeden çıkarma hatası:', error);
      this.showStatus('permanent-status', `❌ Çıkarma hatası: ${error.message}`, 'error');
    }
  }

  // =====================================================
  // MANUEL TÜR DEĞİŞTİRME
  // =====================================================

  async updateSymbolType(symbol, newType) {
    try {
      console.log(`🔄 ${symbol} türü ${newType} olarak değiştiriliyor...`);
      
      const response = await fetch('/api/consecutive/update-symbol-type', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          symbol: symbol, 
          new_type: newType 
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        this.showStatus('permanent-status', `✅ ${data.message}`, 'success');
        this.loadPermanentHighRatio();
      } else {
        this.showStatus('permanent-status', `❌ ${data.error}`, 'error');
        this.loadPermanentHighRatio();
      }
    } catch (error) {
      console.error('Tür güncelleme hatası:', error);
      this.showStatus('permanent-status', `❌ Güncelleme hatası: ${error.message}`, 'error');
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
      case 'bullish':
        filteredResults = this.allResults.filter(result => result.trend_direction === 'Bullish');
        break;
      case 'bearish':
        filteredResults = this.allResults.filter(result => result.trend_direction === 'Bearish');
        break;
      case 'high-ratio':
        // 🆕 Dinamik threshold kullanımı
        filteredResults = this.allResults.filter(result => Math.abs(result.ratio_percent) >= this.minRatioThreshold);
        break;
      default:
        filteredResults = this.allResults;
        break;
    }
    
    filteredResults.sort((a, b) => Math.abs(b.ratio_percent) - Math.abs(a.ratio_percent));
    
    filteredResults.forEach((result, index) => {
      result.filtered_rank = index + 1;
      result.shouldHighlight = false;
    });
    
    this.updateSupertrendTable(filteredResults);
    const filterCountElement = document.getElementById('filter-count');
    if (filterCountElement) {
      filterCountElement.textContent = `${filteredResults.length} sonuç`;
    }
  }

  updateSupertrendTable(results) {
    const tbody = document.getElementById('consecutive-results-tbody');
    if (!tbody) return;
    
    if (results.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" style="padding: 20px; text-align: center; color: var(--text-muted);">📊 Bu filtrede sonuç bulunamadı</td></tr>';
      return;
    }
    
    tbody.innerHTML = '';
    results.forEach(result => {
      const row = document.createElement('tr');
      
      if (result.shouldHighlight) {
        row.style.backgroundColor = 'rgba(255, 235, 59, 0.3)';
        row.style.border = '2px solid #ffeb3b';
        row.style.boxShadow = '0 0 10px rgba(255, 235, 59, 0.5)';
      }
      
      const ratioClass = result.trend_direction === 'Bullish' ? 'percentage-positive' : 'percentage-negative';
      const zScoreClass = Math.abs(result.z_score) > 2 ? 'count-highlight' : '';
      
      const timeframeElement = document.getElementById('timeframe');
      const currentTimeframe = timeframeElement ? timeframeElement.value : '4h';
      
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
        <td class="${ratioClass}">${result.ratio_percent}%</td>
        <td class="${zScoreClass}">${result.z_score}</td>
        <td style="text-align: center;">${addButtonHtml}</td>
      `;
      tbody.appendChild(row);
    });
  }

  // =====================================================
  // PERMANENT LIST FUNCTIONS
  // =====================================================

  async loadPermanentHighRatio() {
    try {
      const response = await fetch('/api/consecutive/permanent-high-consecutive');
      const data = await response.json();
      
      if (data.success) {
        this.updatePermanentTable(data.permanent_symbols);
        
        const statusMessage = `🏆 ${data.count} kalıcı emtia yüklendi`;
        this.showStatus('permanent-status', statusMessage, 'success');
        
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
      tbody.innerHTML = '<tr><td colspan="7" style="padding: 20px; text-align: center; color: var(--text-muted);">🏆 Henüz kalıcı listede emtia yok</td></tr>';
      return;
    }
    
    tbody.innerHTML = '';
    permanentSymbols.forEach(symbol => {
      const row = document.createElement('tr');
      
      const ratioHighlight = Math.abs(symbol.max_ratio_percent) >= 200 ? 'count-highlight' : '';
      const cSignalWithTime = this.formatCSignalWithTime(symbol.c_signal, symbol.c_signal_update_time);
      
      const typeDropdownHtml = `
        <select onchange="app.updateSymbolType('${symbol.symbol}', this.value)" 
                style="background: var(--darker-bg); color: var(--text-primary); border: 1px solid var(--border-color); padding: 2px 6px; border-radius: 4px; font-size: 11px;">
          <option value="Bullish" ${symbol.max_supertrend_type === 'Bullish' ? 'selected' : ''}>🟢 Bullish</option>
          <option value="Bearish" ${symbol.max_supertrend_type === 'Bearish' ? 'selected' : ''}>🔴 Bearish</option>
        </select>
      `;
      
      row.innerHTML = `
        <td>${symbol.rank}</td>
        <td class="symbol-clickable" onclick="window.open('${symbol.tradingview_link}', '_blank')" 
            title="TradingView'da aç">🏆 ${symbol.symbol}</td>
        <td style="font-size: 11px; color: var(--text-muted);">${symbol.first_date}</td>
        <td class="${ratioHighlight}">${Math.abs(symbol.max_ratio_percent)}%</td>
        <td>${typeDropdownHtml}</td>
        <td style="font-size: 11px;">${cSignalWithTime}</td>
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
    if (!confirm('Kalıcı ratio listesini temizlemek istediğinizden emin misiniz?')) return;
    
    try {
      this.showStatus('permanent-status', '🗑️ Kalıcı liste temizleniyor...', 'warning', true);
      
      const response = await fetch('/api/consecutive/clear-permanent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      if (data.success) {
        this.updatePermanentTable([]);
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

  async startSupertrendAnalysis(isAutoUpdate = false) {
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
        this.showStatus('analysis-status', `🎯 ${this.selectedSymbols.length} emtia için ${timeframe} Supertrend analizi başlatılıyor...`, 'info', true);
      } else {
        this.updateCounter++;
        this.showStatus('analysis-status', `🔄 Otomatik güncelleme #${this.updateCounter} - ${this.selectedSymbols.length} emtia analiz ediliyor...`, 'info', true);
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
        
        this.filterResults(this.currentFilter);
        this.loadPermanentHighRatio();
        this.updateSystemStatus();
        
        if (!isAutoUpdate) {
          this.startAutoUpdateLoop();
          const now = new Date().toLocaleTimeString('tr-TR');
          this.showStatus('analysis-status', `✅ Analiz başlatıldı - Her 30 saniyede otomatik güncellenecek (${now})`, 'success');
        } else {
          const now = new Date().toLocaleTimeString('tr-TR');
          this.showStatus('analysis-status', `🔄 Otomatik güncelleme aktif - Son: ${now} (${data.results.length} emtia)`, 'info');
        }
      } else {
        this.showStatus('analysis-status', `❌ ${data.error}`, 'error');
      }
    } catch (error) {
      console.error('Supertrend analiz hatası:', error);
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
    
    this.autoUpdateInterval = setInterval(() => {
      if (this.selectedSymbols.length > 0) {
        this.startSupertrendAnalysis(true);
      }
    }, 30000);
    
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
    
    // 🆕 Settings kaydet butonu
    const saveSettingsBtn = document.getElementById('save-settings');
    if (saveSettingsBtn) {
      saveSettingsBtn.addEventListener('click', () => this.saveSettings());
    }
    
    // 🆕 Threshold input değişiminde display'i güncelle
    const thresholdInput = document.getElementById('min-ratio-threshold');
    if (thresholdInput) {
      thresholdInput.addEventListener('input', (e) => {
        const displayElement = document.getElementById('current-threshold-display');
        if (displayElement) {
          displayElement.textContent = e.target.value;
        }
      });
    }
    
    const startAnalysisBtn = document.getElementById('start-analysis');
    if (startAnalysisBtn) {
      startAnalysisBtn.addEventListener('click', () => this.startSupertrendAnalysis(false));
    }

    const stopAutoUpdateBtn = document.getElementById('stop-auto-update');
    if (stopAutoUpdateBtn) {
      stopAutoUpdateBtn.addEventListener('click', () => this.stopAutoUpdate());
    }
    
    const refreshPermanentBtn = document.getElementById('refresh-permanent');
    if (refreshPermanentBtn) {
      refreshPermanentBtn.addEventListener('click', () => this.loadPermanentHighRatio());
    }

    const clearPermanentBtn = document.getElementById('clear-permanent');
    if (clearPermanentBtn) {
      clearPermanentBtn.addEventListener('click', () => this.clearPermanentList());
    }
    
    const filterAllBtn = document.getElementById('filter-all');
    if (filterAllBtn) {
      filterAllBtn.addEventListener('click', () => {
        this.setActiveFilter('all');
        this.filterResults('all');
      });
    }
    
    const filterBullishBtn = document.getElementById('filter-bullish');
    if (filterBullishBtn) {
      filterBullishBtn.addEventListener('click', () => {
        this.setActiveFilter('bullish');
        this.filterResults('bullish');
      });
    }
    
    const filterBearishBtn = document.getElementById('filter-bearish');
    if (filterBearishBtn) {
      filterBearishBtn.addEventListener('click', () => {
        this.setActiveFilter('bearish');
        this.filterResults('bearish');
      });
    }
    
    const filterHighRatioBtn = document.getElementById('filter-high-ratio');
    if (filterHighRatioBtn) {
      filterHighRatioBtn.addEventListener('click', () => {
        this.setActiveFilter('high-ratio');
        this.filterResults('high-ratio');
      });
    }
    
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
    console.log('🎯 Supertrend + C-Signal + Threshold System initializing...');
    
    try {
      await Promise.all([
        this.loadAllSymbols(),
        this.loadSelectedSymbols(),
        this.loadPermanentHighRatio()
      ]);
      
      this.bindEventListeners();
      this.updateSystemStatus();
      
      console.log('✅ Supertrend + C-Signal + Threshold System initialized');
      console.log(`⚙️ Mevcut threshold: ${this.minRatioThreshold}%`);
      
      setTimeout(() => {
        const searchInput = document.getElementById('symbol-search');
        if (searchInput) {
          console.log('🔍 Arama kutusu bulundu, event listener ekleniyor...');
          
          searchInput.addEventListener('input', (e) => {
            const searchValue = e.target.value;
            
            if (searchValue.length >= 1) {
              this.showSearchSuggestions(searchValue);
            } else {
              this.hideSearchDropdown();
            }
            
            if (searchValue.length >= 2) {
              this.searchSymbols(searchValue);
            } else if (searchValue.length === 0) {
              this.filterResults(this.currentFilter);
            }
          });
          
          searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
              this.searchSymbols(e.target.value);
              this.hideSearchDropdown();
            }
          });
          
          searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
              this.hideSearchDropdown();
            }
          });
          
          searchInput.addEventListener('focus', () => {
            const value = searchInput.value;
            if (value.length >= 1) {
              this.showSearchSuggestions(value);
            }
          });
          
          searchInput.addEventListener('blur', () => {
            setTimeout(() => {
              this.hideSearchDropdown();
            }, 150);
          });
          
          console.log('✅ Arama kutusu event listener\'ları başarıyla eklendi');
        }
      }, 500);
      
    } catch (error) {
      console.error('❌ Sistem başlatma hatası:', error);
    }
  }
}

// Global app instance
let app;

// DOM yüklendiğinde uygulamayı başlat
document.addEventListener('DOMContentLoaded', () => {
  console.log('🚀 DOM yüklendi, sistem başlatılıyor...');
  app = new AnalysisApp();
  app.init();
});

// Global fonksiyonlar
if (typeof window !== 'undefined') {
  window.app = app;
}