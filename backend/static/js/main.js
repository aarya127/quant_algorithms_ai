// Global state
let currentStock = null;
let notifications = [];
let charts = {};
let stockDataCache = {}; // Cache for individual stock data
let currentSection = 'dashboard'; // Track current section
let userWatchlist = []; // User's custom watchlist

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize AOS (Animate On Scroll)
    AOS.init({
        duration: 800,
        easing: 'ease-in-out',
        once: true,
        offset: 100
    });
    
    loadUserWatchlist(); // Load saved watchlist from localStorage
    showSection('dashboard'); // Show dashboard by default
    loadDashboard();
    loadEarningsCalendar();
    setupEventListeners();
    setupNavigationListeners();
    startNotificationPolling();
    updateMarketStatus();
    loadWatchlistPrices(); // Load prices for all watchlist stocks
});

// Setup navigation listeners
function setupNavigationListeners() {
    document.getElementById('navDashboard').addEventListener('click', function(e) {
        e.preventDefault();
        showSection('dashboard');
        setActiveNav(this);
    });
    
    document.getElementById('navNews').addEventListener('click', function(e) {
        e.preventDefault();
        showSection('news');
        setActiveNav(this);
    });
    
    document.getElementById('navCalendar').addEventListener('click', function(e) {
        e.preventDefault();
        showSection('calendar');
        setActiveNav(this);
    });
    
    document.getElementById('navQuant').addEventListener('click', function(e) {
        e.preventDefault();
        showSection('quant');
        setActiveNav(this);
    });
}

// Show specific section
function showSection(section) {
    // Hide all sections
    document.querySelectorAll('.content-section').forEach(sec => {
        sec.style.display = 'none';
    });
    
    // Show requested section
    const sectionElement = document.getElementById('section' + section.charAt(0).toUpperCase() + section.slice(1));
    if (sectionElement) {
        sectionElement.style.display = 'block';
        currentSection = section;
        
        // Load section-specific data
        if (section === 'news') {
            loadMarketNews();
        } else if (section === 'calendar') {
            loadEarningsCalendar();
        } else if (section === 'quant') {
            loadQuantAnalysis();
        }
    }
}

// Set active navigation item
function setActiveNav(element) {
    document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
        link.classList.remove('active');
    });
    element.classList.add('active');
}

// Setup event listeners
function setupEventListeners() {
    // Search form with live results
    const searchInput = document.getElementById('searchInput');
    const searchForm = document.getElementById('searchForm');
    
    searchInput.addEventListener('input', debounce(function() {
        const query = searchInput.value.trim();
        if (query.length >= 1) {
            performSearch(query);
        } else {
            hideSearchResults();
        }
    }, 300));
    
    searchForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const query = searchInput.value.trim();
        if (query) {
            performSearch(query);
        }
    });
    
    // Watchlist items
    document.querySelectorAll('.stock-item').forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const symbol = this.dataset.symbol;
            showSection('dashboard'); // Always show in dashboard
            setActiveNav(document.getElementById('navDashboard'));
            loadStockDetails(symbol);
        });
    });
    
    // Notifications button
    document.getElementById('notificationsBtn').addEventListener('click', function() {
        showNotifications();
    });
    
    // Timeframe selector
    const timeframeSelect = document.getElementById('timeframeSelect');
    if (timeframeSelect) {
        timeframeSelect.addEventListener('change', function() {
            if (currentStock) {
                loadScenarios(currentStock, this.value);
            }
        });
    }
    
    // Tab change listeners
    document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(e) {
            const target = e.target.getAttribute('data-bs-target');
            if (currentStock) {
                handleTabChange(target, currentStock);
            }
        });
    });
}

// Load real-time prices for watchlist stocks
async function loadWatchlistPrices() {
    const watchlistItems = document.querySelectorAll('.stock-item');
    
    watchlistItems.forEach(async (item) => {
        const symbol = item.dataset.symbol;
        const badge = item.querySelector('.badge');
        
        try {
            console.log(`📊 Loading price for ${symbol}`);
            const response = await fetch(`/api/stock/${symbol}`);
            const data = await response.json();
            
            if (data.success && data.quote) {
                const price = data.quote.c;
                const change = data.quote.d;
                const changePercent = data.quote.dp;
                
                badge.className = change >= 0 ? 'badge bg-success' : 'badge bg-danger';
                badge.textContent = `$${price.toFixed(2)} (${change >= 0 ? '+' : ''}${changePercent.toFixed(2)}%)`;
                console.log(`✓ ${symbol}: $${price.toFixed(2)}`);
            }
        } catch (error) {
            console.error(`✗ Error loading price for ${symbol}:`, error);
            badge.textContent = 'Error';
            badge.className = 'badge bg-secondary';
        }
    });
}

// Load dashboard with trending news
async function loadDashboard() {
    try {
        const response = await fetch('/api/dashboard');
        const data = await response.json();
        
        const container = document.getElementById('trendingNews');
        container.innerHTML = '';
        
        if (data.trending_news && data.trending_news.length > 0) {
            data.trending_news.forEach((news, index) => {
                const newsCard = createNewsCard(news, index);
                container.appendChild(newsCard);
            });
        } else {
            container.innerHTML = '<div class="col-12"><p class="text-muted">No trending news available</p></div>';
        }
        
        updateLastUpdated();
    } catch (error) {
        console.error('Error loading dashboard:', error);
        document.getElementById('trendingNews').innerHTML = 
            '<div class="col-12"><div class="alert alert-danger">Failed to load trending news</div></div>';
    }
}

// Create news card element
function createNewsCard(news, index) {
    const col = document.createElement('div');
    col.className = 'col-md-6 col-lg-4 fade-in';
    col.style.animationDelay = `${index * 0.1}s`;
    
    const sentiment = news.sentiment || 'neutral';
    const sentimentClass = sentiment === 'positive' ? 'success' : sentiment === 'negative' ? 'danger' : 'secondary';
    
    col.innerHTML = `
        <div class="card news-card h-100">
            ${news.image ? `<img src="${news.image}" class="card-img-top" alt="News image">` : ''}
            <span class="badge bg-${sentimentClass} news-badge">${sentiment.toUpperCase()}</span>
            <div class="card-body">
                <h5 class="card-title">${news.headline || news.title || 'No title'}</h5>
                <p class="card-text text-muted">${truncateText(news.summary || '', 150)}</p>
                <div class="d-flex justify-content-between align-items-center">
                    <small class="text-muted">
                        <i class="far fa-clock"></i> ${formatDate(news.datetime || news.time_published)}
                    </small>
                    ${news.url ? `<a href="${news.url}" target="_blank" class="btn btn-sm btn-outline-primary">Read More</a>` : ''}
                </div>
            </div>
        </div>
    `;
    
    return col;
}

// Load stock details
async function loadStockDetails(symbol) {
    currentStock = symbol;
    
    console.log(`\n========================================`);
    console.log(`LOADING DATA FOR: ${symbol}`);
    console.log(`========================================\n`);
    
    // Show stock details section
    document.getElementById('stockDetails').style.display = 'block';
    document.getElementById('stockSymbol').textContent = symbol;
    
    // Show loading state
    document.getElementById('stockPrice').textContent = 'Loading...';
    document.getElementById('priceChange').textContent = '...';
    
    // Scroll to stock details
    document.getElementById('stockDetails').scrollIntoView({ behavior: 'smooth' });
    
    // Load overview tab by default
    await loadStockOverview(symbol);
    
    // Update watchlist active state
    document.querySelectorAll('.stock-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.symbol === symbol) {
            item.classList.add('active');
        }
    });
}

// Load stock overview with dedicated API call
async function loadStockOverview(symbol) {
    console.log(`📊 API CALL: /api/stock/${symbol}`);
    
    try {
        // Dedicated API call for this specific stock
        const startTime = Date.now();
        const response = await fetch(`/api/stock/${symbol}`);
        const data = await response.json();
        const endTime = Date.now();
        
        console.log(`✓ ${symbol} overview loaded in ${endTime - startTime}ms`);
        
        // Cache the data for this specific symbol
        stockDataCache[symbol] = {
            timestamp: Date.now(),
            data: data
        };
        
        // Get currency symbol
        const currencySymbol = data.currency === 'CAD' ? 'C$' : '$';
        
        // Update price display
        if (data.quote) {
            document.getElementById('stockPrice').textContent = `${currencySymbol}${data.quote.c.toFixed(2)}`;
            const change = data.quote.d;
            const changePercent = data.quote.dp;
            const changeClass = change >= 0 ? 'bg-success' : 'bg-danger';
            document.getElementById('priceChange').className = `badge ${changeClass}`;
            document.getElementById('priceChange').textContent = 
                `${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePercent.toFixed(2)}%)`;
        }
        
        // Update company info
        const companyInfo = document.getElementById('companyInfo');
        if (data.company) {
            let companyHTML = `<div class="company-details">`;
            
            // Show loading placeholder for AI overview
            companyHTML += `
                <div class="alert alert-light mb-3" id="aiOverviewContainer">
                    <h6><i class="fas fa-robot"></i> AI-Generated Overview</h6>
                    <div class="spinner-border spinner-border-sm text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <small class="text-muted ms-2">Generating AI overview...</small>
                </div>
            `;
            
            // Business Description
            if (data.company.longBusinessSummary) {
                companyHTML += `
                    <div class="alert alert-info mb-3">
                        <h6><i class="fas fa-building"></i> About ${data.company.name}</h6>
                        <p class="mb-0 small">${data.company.longBusinessSummary}</p>
                    </div>
                `;
            }
            
            // Company Overview Section
            companyHTML += `<h6 class="mt-3 mb-2"><i class="fas fa-info-circle"></i> Company Overview</h6>`;
            companyHTML += `<div class="row">`;
            
            // Left column
            companyHTML += `<div class="col-md-6">`;
            companyHTML += `<p class="mb-2"><strong>Sector:</strong> ${data.company.sector || 'N/A'}</p>`;
            companyHTML += `<p class="mb-2"><strong>Industry:</strong> ${data.company.finnhubIndustry || 'N/A'}</p>`;
            companyHTML += `<p class="mb-2"><strong>Exchange:</strong> ${data.company.exchange || 'N/A'}</p>`;
            companyHTML += `<p class="mb-2"><strong>Currency:</strong> ${data.currency || 'USD'}</p>`;
            if (data.company.city || data.company.state) {
                companyHTML += `<p class="mb-2"><strong>Location:</strong> ${data.company.city || ''}${data.company.city && data.company.state ? ', ' : ''}${data.company.state || ''}, ${data.company.country_full || data.company.country || 'N/A'}</p>`;
            }
            if (data.company.fullTimeEmployees && data.company.fullTimeEmployees > 0) {
                companyHTML += `<p class="mb-2"><strong>Employees:</strong> ${data.company.fullTimeEmployees.toLocaleString()}</p>`;
            }
            companyHTML += `</div>`;
            
            // Right column
            companyHTML += `<div class="col-md-6">`;
            companyHTML += `<p class="mb-2"><strong>Market Cap:</strong> ${currencySymbol}${((data.company.marketCapitalization || 0) / 1e9).toFixed(2)}B</p>`;
            if (data.company.fiftyTwoWeekHigh && data.company.fiftyTwoWeekLow) {
                companyHTML += `<p class="mb-2"><strong>52-Week Range:</strong> ${currencySymbol}${data.company.fiftyTwoWeekLow.toFixed(2)} - ${currencySymbol}${data.company.fiftyTwoWeekHigh.toFixed(2)}</p>`;
            }
            if (data.company.dividendYield && data.company.dividendYield > 0) {
                companyHTML += `<p class="mb-2"><strong>Dividend Yield:</strong> ${(data.company.dividendYield * 100).toFixed(2)}%</p>`;
            }
            if (data.company.trailingPE && data.company.trailingPE > 0) {
                companyHTML += `<p class="mb-2"><strong>P/E Ratio:</strong> ${data.company.trailingPE.toFixed(2)}</p>`;
            }
            if (data.company.priceToBook && data.company.priceToBook > 0) {
                companyHTML += `<p class="mb-2"><strong>Price/Book:</strong> ${data.company.priceToBook.toFixed(2)}</p>`;
            }
            if (data.company.beta && data.company.beta > 0) {
                companyHTML += `<p class="mb-2"><strong>Beta:</strong> ${data.company.beta.toFixed(2)}</p>`;
            }
            companyHTML += `</div>`;
            
            companyHTML += `</div>`; // Close row
            
            // Website link
            if (data.company.weburl) {
                companyHTML += `<p class="mt-3"><a href="${data.company.weburl}" target="_blank" class="btn btn-sm btn-outline-primary"><i class="fas fa-external-link-alt"></i> Company Website</a></p>`;
            }
            
            companyHTML += `</div>`;
            
            companyInfo.innerHTML = companyHTML;
            
            // Load AI overview in background (non-blocking)
            loadAIOverview(symbol);
        } else {
            companyInfo.innerHTML = '<p class="text-muted">Company information not available</p>';
        }
        
        // Update recent news with enhanced visual design
        const newsContainer = document.getElementById('recentNews');
        if (data.news && data.news.length > 0) {
            console.log(`📰 Loading ${data.news.length} news articles for ${symbol}`);
            console.log(`First headline: ${data.news[0].headline}`);
            
            newsContainer.innerHTML = `
                <div class="mb-3">
                    <span class="badge bg-primary">
                        <i class="fas fa-newspaper"></i> ${symbol} Specific News (${data.news.length} articles)
                    </span>
                    <small class="text-muted ms-2">
                        Source: Finnhub API
                    </small>
                </div>
            `;
            
            data.news.slice(0, 5).forEach((news, index) => {
                const newsCard = document.createElement('div');
                newsCard.className = 'news-card mb-3 fade-in';
                newsCard.style.animationDelay = `${index * 0.1}s`;
                
                // Extract domain from URL for source display
                let source = 'Unknown';
                try {
                    const urlObj = new URL(news.url);
                    source = urlObj.hostname.replace('www.', '');
                } catch (e) {
                    source = 'Unknown Source';
                }
                
                // Format date
                const newsDate = formatDate(news.datetime);
                
                newsCard.innerHTML = `
                    <div class="card h-100 shadow-sm hover-shadow">
                        ${news.image ? `
                            <img src="${news.image}" class="card-img-top" alt="News thumbnail" 
                                 style="height: 200px; object-fit: cover;" 
                                 onerror="this.style.display='none'">
                        ` : ''}
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <span class="badge bg-primary">
                                    <i class="fas fa-globe"></i> ${source}
                                </span>
                                <small class="text-muted">
                                    <i class="far fa-clock"></i> ${newsDate}
                                </small>
                            </div>
                            <h6 class="card-title mb-2">${news.headline}</h6>
                            ${news.summary ? `
                                <p class="card-text text-muted small mb-3">
                                    ${truncateText(news.summary, 150)}
                                </p>
                            ` : ''}
                            <a href="${news.url}" target="_blank" class="btn btn-sm btn-outline-primary">
                                Read Full Article <i class="fas fa-external-link-alt ms-1"></i>
                            </a>
                        </div>
                    </div>
                `;
                
                newsContainer.appendChild(newsCard);
            });
        } else {
            newsContainer.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i> No recent news available
                </div>
            `;
        }
        
    } catch (error) {
        console.error(`✗ Error loading overview for ${symbol}:`, error);
        document.getElementById('companyInfo').innerHTML = 
            '<div class="alert alert-danger">Failed to load stock data. Please try again.</div>';
    }
}

// Load AI overview separately (non-blocking)
async function loadAIOverview(symbol) {
    console.log(`🤖 Loading AI overview for ${symbol}...`);
    
    try {
        const response = await fetch(`/api/ai-overview/${symbol}`);
        const data = await response.json();
        
        const container = document.getElementById('aiOverviewContainer');
        if (!container) return; // User might have navigated away
        
        if (data.success && data.ai_overview) {
            container.className = 'alert alert-info mb-3 fade-in';
            container.innerHTML = `
                <h6><i class="fas fa-robot"></i> AI-Generated Overview</h6>
                <p class="mb-0" style="white-space: pre-wrap;">${data.ai_overview}</p>
                <small class="text-muted">Powered by NVIDIA Llama 3.1 70B</small>
            `;
            console.log(`✓ AI overview loaded for ${symbol}`);
        } else {
            // AI overview is disabled - hide the section entirely
            container.style.display = 'none';
            console.log(`ℹ️ AI overview disabled for faster loading`);
        }
    } catch (error) {
        console.error(`✗ Error loading AI overview for ${symbol}:`, error);
        const container = document.getElementById('aiOverviewContainer');
        if (container) {
            // Hide on error instead of showing error message
            container.style.display = 'none';
        }
    }
}

// Handle tab changes with separate API calls
function handleTabChange(target, symbol) {
    console.log(`\n🔄 TAB CHANGE: ${target} for ${symbol}`);
    
    switch(target) {
        case '#statistics':
            loadCompanyStatistics(symbol);
            break;
        case '#charts':
            loadCharts(symbol, currentPeriod, currentInterval);
            break;
        case '#sentiment':
            // Don't auto-load sentiment - user must click button
            setupSentimentButton(symbol);
            break;
        case '#scenarios':
            loadScenarios(symbol, '1M');
            break;
        case '#metrics':
            loadMetrics(symbol);
            break;
        case '#recommendations':
            loadRecommendations(symbol);
            break;
    }
}

// Setup sentiment button to load on demand
function setupSentimentButton(symbol) {
    const button = document.getElementById('analyzeSentimentBtn');
    const container = document.getElementById('sentimentAnalysis');
    
    if (button) {
        // Remove any existing listeners
        const newButton = button.cloneNode(true);
        button.parentNode.replaceChild(newButton, button);
        
        // Add click handler
        newButton.addEventListener('click', async function() {
            newButton.disabled = true;
            newButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            
            await loadSentiment(symbol);
            
            newButton.disabled = false;
            newButton.innerHTML = '<i class="fas fa-brain"></i> Refresh Analysis';
        });
    }
}

// Load sentiment analysis with dedicated API call
async function loadSentiment(symbol) {
    console.log(`🧠 API CALL: /api/sentiment/${symbol}`);
    
    const container = document.getElementById('sentimentAnalysis');
    container.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3">Analyzing sentiment from multiple sources...</p>
            <p class="text-muted"><small>This may take 20-30 seconds as we analyze FinBERT, Alpha Vantage, and Insider Trading data</small></p>
        </div>
    `;
    
    try {
        // Dedicated sentiment API call for this stock
        const startTime = Date.now();
        const response = await fetch(`/api/sentiment/${symbol}`);
        const data = await response.json();
        const endTime = Date.now();
        
        console.log(`✓ ${symbol} sentiment loaded in ${(endTime - startTime)/1000}s`);
        console.log(`  Sources:`, data.sources?.map(s => s.name).join(', '));
        
        if (!data.success) {
            container.innerHTML = `<div class="alert alert-warning">${data.error || 'Failed to load sentiment'}</div>`;
            return;
        }
        
        const overall = data.overall_sentiment;
        const sentimentClass = overall === 'positive' ? 'positive' : overall === 'negative' ? 'negative' : 'neutral';
        
        // Build sources comparison HTML
        let sourcesHTML = '';
        if (data.sources && data.sources.length > 0) {
            sourcesHTML = `
                <div class="mt-4">
                    <h6>📊 Source Comparison</h6>
                    <div class="row">
            `;
            
            data.sources.forEach(source => {
                const sourceClass = source.sentiment === 'positive' ? 'success' : 
                                  source.sentiment === 'negative' ? 'danger' : 'secondary';
                sourcesHTML += `
                    <div class="col-md-6 mb-3">
                        <div class="card">
                            <div class="card-body">
                                <h6 class="card-title">${source.provider}</h6>
                                <div class="d-flex justify-content-between align-items-center">
                                    <span class="badge bg-${sourceClass}">${source.sentiment.toUpperCase()}</span>
                                    <strong>${source.score.toFixed(1)}/100</strong>
                                </div>
                                <p class="mt-2 mb-0"><small>${source.articles_analyzed || 0} articles analyzed</small></p>
                                ${source.confidence ? `<p class="mb-0"><small>Confidence: ${(source.confidence * 100).toFixed(1)}%</small></p>` : ''}
                                ${source.mspr !== undefined ? `<p class="mb-0"><small>MSPR: ${source.mspr.toFixed(2)} (${source.insider_signal})</small></p>` : ''}
                            </div>
                        </div>
                    </div>
                `;
            });
            
            sourcesHTML += '</div></div>';
        }
        
        container.innerHTML = `
            <div class="sentiment-container">
                <div class="sentiment-score ${sentimentClass}">
                    <div>
                        <small>Overall Consensus</small>
                        <h2>${overall.toUpperCase()}</h2>
                        <p>Confidence: ${(data.confidence * 100).toFixed(1)}%</p>
                        <p>Agreement: ${data.agreement_level.toUpperCase()}</p>
                    </div>
                </div>
                
                <div class="alert alert-info">
                    <strong>Multi-Source Analysis:</strong> ${data.summary}
                </div>
                
                ${sourcesHTML}
                
                <div class="sentiment-breakdown">
                    <div class="sentiment-item">
                        <i class="fas fa-smile text-success"></i>
                        <h4>${(data.positive_ratio * 100).toFixed(1)}%</h4>
                        <small>Positive Sources</small>
                    </div>
                    <div class="sentiment-item">
                        <i class="fas fa-meh text-secondary"></i>
                        <h4>${(data.neutral_ratio * 100).toFixed(1)}%</h4>
                        <small>Neutral Sources</small>
                    </div>
                    <div class="sentiment-item">
                        <i class="fas fa-frown text-danger"></i>
                        <h4>${(data.negative_ratio * 100).toFixed(1)}%</h4>
                        <small>Negative Sources</small>
                    </div>
                </div>
                
                <div class="mt-4">
                    <small class="text-muted">
                        Total articles: ${data.articles_analyzed || 0} | 
                        Consensus score: ${data.consensus_score.toFixed(1)}/100 |
                        Variance: ${data.score_variance.toFixed(1)}
                    </small>
                </div>
            </div>
        `;
        
    } catch (error) {
        console.error(`✗ Error loading sentiment for ${symbol}:`, error);
        container.innerHTML = '<div class="alert alert-danger">Failed to load sentiment analysis. Please try again.</div>';
    }
}

// Load scenarios with dedicated API call
async function loadScenarios(symbol, timeframe) {
    console.log(`🎯 API CALL: /api/scenarios/${symbol}?timeframe=${timeframe}`);
    
    const container = document.getElementById('scenarioAnalysis');
    container.innerHTML = '<div class="spinner-border text-primary" role="status"></div><p class="mt-2">Generating scenarios using multi-source data...</p>';
    
    try {
        // Dedicated scenarios API call
        const startTime = Date.now();
        const response = await fetch(`/api/scenarios/${symbol}?timeframe=${timeframe}`);
        const data = await response.json();
        const endTime = Date.now();
        
        console.log(`✓ ${symbol} scenarios loaded in ${endTime - startTime}ms`);
        
        if (!data.success) {
            container.innerHTML = `<div class="alert alert-warning">${data.error || 'Failed to load scenarios'}</div>`;
            return;
        }
        
        container.innerHTML = `
            <div class="alert alert-info mb-3">
                <strong>Current Price:</strong> $${data.current_price.toFixed(2)} | 
                <strong>Sentiment Score:</strong> ${data.sentiment_score.toFixed(1)}/100 | 
                <strong>EPS Growth:</strong> ${data.eps_growth.toFixed(1)}%
                <br><small>Using data from: ${data.data_sources.join(', ')}</small>
            </div>
            <div class="scenario-container">
                ${createScenarioCard('bull', data.bull_case)}
                ${createScenarioCard('base', data.base_case)}
                ${createScenarioCard('bear', data.bear_case)}
            </div>
        `;
        
    } catch (error) {
        console.error(`✗ Error loading scenarios for ${symbol}:`, error);
        container.innerHTML = '<div class="alert alert-danger">Failed to load scenarios. Please try again.</div>';
    }
}

// Create scenario card
function createScenarioCard(type, scenario) {
    const icons = {
        bull: 'fa-arrow-trend-up',
        base: 'fa-minus',
        bear: 'fa-arrow-trend-down'
    };
    
    const colors = {
        bull: 'success',
        base: 'warning',
        bear: 'danger'
    };
    
    return `
        <div class="scenario-card ${type}">
            <div class="scenario-title">
                <i class="fas ${icons[type]} text-${colors[type]}"></i>
                ${type.toUpperCase()} CASE
                <span class="ms-auto badge bg-${colors[type]}">${scenario.probability}%</span>
            </div>
            <div class="price-target">
                Target: $${scenario.price_target.toFixed(2)}
                <small class="text-muted">(${scenario.return > 0 ? '+' : ''}${scenario.return.toFixed(2)}%)</small>
            </div>
            <hr>
            <h6>Key Factors:</h6>
            <ul class="scenario-factors">
                ${scenario.factors.map(factor => `<li><i class="fas fa-check-circle text-${colors[type]}"></i> ${factor}</li>`).join('')}
            </ul>
            <div class="mt-3">
                <strong>Rationale:</strong>
                <p>${scenario.rationale}</p>
            </div>
        </div>
    `;
}

// Load metrics and grading with dedicated API call
async function loadMetrics(symbol) {
    console.log(`⭐ API CALL: /api/metrics/${symbol}`);
    
    const container = document.getElementById('metricsGrading');
    container.innerHTML = '<div class="spinner-border text-primary" role="status"></div><p class="mt-2">Calculating comprehensive metrics...</p>';
    
    try {
        // Dedicated metrics API call
        const startTime = Date.now();
        const response = await fetch(`/api/metrics/${symbol}`);
        const data = await response.json();
        const endTime = Date.now();
        
        console.log(`✓ ${symbol} metrics loaded in ${endTime - startTime}ms`);
        console.log(`  Overall Grade: ${data.overall_grade} (${data.average_score.toFixed(1)}/100)`);
        
        if (!data.success) {
            container.innerHTML = `<div class="alert alert-warning">${data.error || 'Failed to load metrics'}</div>`;
            return;
        }
        
        // Update overall grade
        document.getElementById('overallGrade').innerHTML = `
            <div class="grade-badge grade-${data.overall_grade}">
                ${data.overall_grade}
            </div>
        `;
        
        container.innerHTML = `
            <div class="row mb-4">
                <div class="col-md-12">
                    <div class="alert alert-info">
                        <h5>Overall Grade: ${data.overall_grade}</h5>
                        <p>${getGradeDescription(data.overall_grade)}</p>
                        <strong>Average Score: ${data.average_score.toFixed(1)}/100</strong>
                    </div>
                </div>
            </div>
            
            <div class="metrics-grid">
                ${createMetricCard('Valuation', data.metrics.valuation)}
                ${createMetricCard('Profitability', data.metrics.profitability)}
                ${createMetricCard('Growth', data.metrics.growth)}
                ${createMetricCard('Financial Health', data.metrics.financial_health)}
            </div>
        `;
        
    } catch (error) {
        console.error(`✗ Error loading metrics for ${symbol}:`, error);
        container.innerHTML = '<div class="alert alert-danger">Failed to load metrics. Please try again.</div>';
    }
}

// Create metric card
function createMetricCard(title, metric) {
    return `
        <div class="metric-card">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h5 class="metric-title mb-0">${title}</h5>
                <div class="grade-badge grade-${metric.grade}">
                    ${metric.grade}
                </div>
            </div>
            <div class="metric-value">${metric.score}/100</div>
            <div class="progress mb-2">
                <div class="progress-bar ${getProgressBarClass(metric.grade)}" 
                     style="width: ${metric.score}%"></div>
            </div>
            <p class="metric-description">${metric.description}</p>
            <small class="text-muted">Based on ${metric.factors.length} factors</small>
        </div>
    `;
}

// Load recommendations with dedicated API call
async function loadRecommendations(symbol) {
    console.log(`💡 API CALL: /api/recommendations/${symbol}`);
    
    const container = document.getElementById('timeRecommendations');
    container.innerHTML = '<div class="spinner-border text-primary" role="status"></div><p class="mt-2">Generating time-based recommendations...</p>';
    
    try {
        // Dedicated recommendations API call
        const startTime = Date.now();
        const response = await fetch(`/api/recommendations/${symbol}`);
        const data = await response.json();
        const endTime = Date.now();
        
        console.log(`✓ ${symbol} recommendations loaded in ${(endTime - startTime)/1000}s`);
        
        if (!data.success) {
            container.innerHTML = `<div class="alert alert-warning">${data.error || 'Failed to load recommendations'}</div>`;
            return;
        }
        
        container.innerHTML = `
            <div class="alert alert-info mb-3">
                <strong>Analysis Base:</strong> Sentiment Score: ${data.sentiment_score.toFixed(1)}/100 | 
                Grade: ${data.overall_grade}
            </div>
            <div class="recommendations-container">
                ${Object.entries(data.recommendations).map(([timeframe, rec]) => 
                    createRecommendationCard(timeframe, rec)
                ).join('')}
            </div>
        `;
        
    } catch (error) {
        console.error(`✗ Error loading recommendations for ${symbol}:`, error);
        container.innerHTML = '<div class="alert alert-danger">Failed to load recommendations. Please try again.</div>';
    }
}

// Create recommendation card
function createRecommendationCard(timeframe, rec) {
    const actionColors = {
        'Strong Buy': 'success',
        'Buy': 'info',
        'Hold': 'warning',
        'Sell': 'danger',
        'Strong Sell': 'dark'
    };
    
    return `
        <div class="recommendation-card">
            <div class="recommendation-header">
                <h5><i class="far fa-clock"></i> ${timeframe}</h5>
                <span class="timeframe-badge">${rec.action}</span>
            </div>
            <div class="recommendation-action">
                ${rec.action}
            </div>
            <p>${rec.reasoning}</p>
            <div>
                <small>Confidence Level</small>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${rec.confidence * 100}%"></div>
                </div>
                <small class="mt-1 d-block">${(rec.confidence * 100).toFixed(1)}%</small>
            </div>
        </div>
    `;
}

// Load comprehensive calendar (earnings, dividends, macro events)
async function loadEarningsCalendar() {
    const container = document.getElementById('earningsCalendar');
    
    try {
        const response = await fetch('/api/calendar');
        const data = await response.json();
        
        if (data.events && data.events.length > 0) {
            // Group events by month and date
            const eventsByMonth = {};
            data.events.forEach(event => {
                const dateObj = new Date(event.date + 'T00:00:00');
                const monthKey = dateObj.toLocaleDateString('en-US', { year: 'numeric', month: 'long' });
                
                if (!eventsByMonth[monthKey]) {
                    eventsByMonth[monthKey] = {};
                }
                
                if (!eventsByMonth[monthKey][event.date]) {
                    eventsByMonth[monthKey][event.date] = [];
                }
                
                eventsByMonth[monthKey][event.date].push(event);
            });
            
            // Get sorted months
            const sortedMonths = Object.keys(eventsByMonth).sort((a, b) => {
                return new Date(a) - new Date(b);
            });
            
            // Count total events by type
            const eventCounts = {
                'Earnings': 0,
                'Dividend': 0,
                'FOMC Meeting': 0,
                'Economic Data': 0,
                'Election': 0,
                'Holiday': 0
            };
            
            data.events.forEach(event => {
                if (eventCounts[event.type] !== undefined) {
                    eventCounts[event.type]++;
                }
            });
            
            let calendarHTML = `
                <div class="mb-4">
                    <div class="row g-3">
                        <div class="col-md-12">
                            <div class="alert alert-primary">
                                <h5 class="mb-3"><i class="fas fa-calendar-alt"></i> 2026 Financial Calendar</h5>
                                <p class="mb-2">Tracking ${data.events.length} important dates for your watchlist</p>
                                <div class="d-flex flex-wrap gap-3 mt-3">
                                    <span class="badge bg-primary"><i class="fas fa-chart-line"></i> ${eventCounts['Earnings']} Earnings</span>
                                    <span class="badge bg-success"><i class="fas fa-dollar-sign"></i> ${eventCounts['Dividend']} Dividends</span>
                                    <span class="badge bg-danger"><i class="fas fa-university"></i> ${eventCounts['FOMC Meeting']} FOMC</span>
                                    <span class="badge bg-info"><i class="fas fa-chart-bar"></i> ${eventCounts['Economic Data']} Economic Data</span>
                                    <span class="badge bg-warning text-dark"><i class="fas fa-vote-yea"></i> ${eventCounts['Election']} Elections</span>
                                    <span class="badge bg-secondary"><i class="fas fa-calendar-times"></i> ${eventCounts['Holiday']} Holidays</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Filter Controls -->
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label class="form-label"><i class="fas fa-filter"></i> Filter by Type:</label>
                            <select class="form-select" id="calendarTypeFilter" onchange="filterCalendarEvents()">
                                <option value="all">All Events</option>
                                <option value="Earnings">Earnings Reports</option>
                                <option value="Dividend">Dividends</option>
                                <option value="FOMC Meeting">FOMC Meetings</option>
                                <option value="Economic Data">Economic Data</option>
                                <option value="Election">Elections</option>
                                <option value="Holiday">Holidays</option>
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label"><i class="fas fa-fire"></i> Filter by Importance:</label>
                            <select class="form-select" id="calendarImportanceFilter" onchange="filterCalendarEvents()">
                                <option value="all">All Levels</option>
                                <option value="high">High Priority Only</option>
                                <option value="medium">Medium & High</option>
                            </select>
                        </div>
                    </div>
                </div>
                <div class="calendar-list" id="calendarEventsList">
            `;
            
            // Store events globally for filtering
            window.allCalendarEvents = data.events;
            
            sortedMonths.forEach(monthKey => {
                const dates = Object.keys(eventsByMonth[monthKey]).sort();
                
                calendarHTML += `
                    <div class="calendar-month-section mb-4">
                        <h4 class="calendar-month-header">
                            <i class="fas fa-calendar"></i> ${monthKey}
                            <span class="badge bg-secondary ms-2">${Object.values(eventsByMonth[monthKey]).flat().length} events</span>
                        </h4>
                `;
                
                dates.forEach(date => {
                    const dateObj = new Date(date + 'T00:00:00');
                    const formattedDate = dateObj.toLocaleDateString('en-US', { 
                        weekday: 'short', 
                        month: 'short', 
                        day: 'numeric' 
                    });
                    
                    // Check if date is today
                    const today = new Date();
                    const isToday = dateObj.getDate() === today.getDate() && 
                                   dateObj.getMonth() === today.getMonth() && 
                                   dateObj.getFullYear() === today.getFullYear();
                    
                    calendarHTML += `
                        <div class="calendar-date-group mb-3 ${isToday ? 'today-highlight' : ''}" data-date="${date}">
                            <h6 class="calendar-date-header ${isToday ? 'text-primary fw-bold' : ''}">
                                <i class="far fa-calendar"></i> ${formattedDate}
                                ${isToday ? '<span class="badge bg-primary ms-2">TODAY</span>' : ''}
                            </h6>
                            <div class="list-group">
                    `;
                    
                    eventsByMonth[monthKey][date].forEach(event => {
                        let badgeClass = 'secondary';
                        let icon = 'fa-info-circle';
                        
                        if (event.type === 'Earnings') {
                            badgeClass = 'primary';
                            icon = 'fa-chart-line';
                        } else if (event.type === 'Dividend') {
                            badgeClass = 'success';
                            icon = 'fa-dollar-sign';
                        } else if (event.type === 'FOMC Meeting') {
                            badgeClass = 'danger';
                            icon = 'fa-university';
                        } else if (event.type === 'Election') {
                            badgeClass = 'warning text-dark';
                            icon = 'fa-vote-yea';
                        } else if (event.type === 'Economic Data') {
                            badgeClass = 'info';
                            icon = 'fa-chart-bar';
                        } else if (event.type === 'Holiday') {
                            badgeClass = 'secondary';
                            icon = 'fa-calendar-times';
                        }
                        
                        const importanceIcon = event.importance === 'high' ? '🔥' : 
                                              event.importance === 'medium' ? '⚡' : '📌';
                        
                        calendarHTML += `
                            <div class="list-group-item calendar-event-item" data-type="${event.type}" data-importance="${event.importance}">
                                <div class="d-flex justify-content-between align-items-start">
                                    <div class="flex-grow-1">
                                        <div class="mb-1">
                                            <span class="badge bg-${badgeClass} me-2">
                                                <i class="fas ${icon}"></i> ${event.type}
                                            </span>
                                            ${event.symbol ? `<span class="badge bg-dark me-2">${event.symbol}</span>` : ''}
                                            <span class="importance-badge">${importanceIcon}</span>
                                        </div>
                                        <div class="event-description">
                                            ${event.description}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                    
                    calendarHTML += `
                            </div>
                        </div>
                    `;
                });
                
                calendarHTML += `
                    </div>
                `;
            });
            
            calendarHTML += '</div>';
            container.innerHTML = calendarHTML;
        } else {
            container.innerHTML = '<p class="text-muted">No upcoming events</p>';
        }
    } catch (error) {
        console.error('Error loading calendar:', error);
        container.innerHTML = '<div class="alert alert-danger">Failed to load calendar</div>';
    }
}

// Load Market News with Twitter and Alpaca integration
let allNewsItems = [];  // Store all news for filtering

async function loadMarketNews() {
    const container = document.getElementById('marketNews');
    const countFilter = document.getElementById('newsCountFilter');
    const count = countFilter ? countFilter.value : 30;
    
    container.innerHTML = '<div class="col-12 text-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div><p class="mt-2">Loading news from Twitter, Alpaca, and Finnhub...</p></div>';
    
    try {
        // Fetch combined news from all sources
        const response = await fetch(`/api/news/combined?count=${count}`);
        const data = await response.json();
        
        if (data.success && data.news && data.news.length > 0) {
            allNewsItems = data.news;  // Store for filtering
            
            // Show warning if Twitter API failed
            if (data.warning) {
                const warningDiv = document.createElement('div');
                warningDiv.className = 'col-12 mb-3';
                warningDiv.innerHTML = `
                    <div class="alert alert-warning alert-dismissible fade show" role="alert">
                        <i class="fas fa-exclamation-triangle"></i> <strong>Twitter/X API Notice:</strong> ${data.warning}
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>
                `;
                container.innerHTML = '';
                container.appendChild(warningDiv);
            }
            
            displayNews(allNewsItems);
        } else {
            container.innerHTML = '<div class="col-12"><div class="alert alert-warning">No market news available at the moment</div></div>';
        }
        
        document.getElementById('newsUpdated').textContent = 
            `Updated: ${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
    } catch (error) {
        console.error('Error loading market news:', error);
        container.innerHTML = '<div class="col-12"><div class="alert alert-danger">Failed to load market news. Please try again.</div></div>';
    }
}

// Filter news based on current filter selections
function filterNews() {
    const sourceFilter = document.getElementById('newsSourceFilter').value;
    const symbolFilter = document.getElementById('newsSymbolFilter').value.toUpperCase().trim();
    
    let filteredNews = allNewsItems;
    
    // Filter by source
    if (sourceFilter !== 'all') {
        filteredNews = filteredNews.filter(item => {
            const source = item.source.toLowerCase();
            if (sourceFilter === 'twitter') return item.type === 'tweet';
            if (sourceFilter === 'alpaca') return source.includes('alpaca');
            if (sourceFilter === 'finnhub') return source.includes('finnhub');
            return true;
        });
    }
    
    // Filter by symbol
    if (symbolFilter) {
        filteredNews = filteredNews.filter(item => {
            const symbols = item.symbols || [];
            return symbols.some(s => s.toUpperCase().includes(symbolFilter));
        });
    }
    
    displayNews(filteredNews);
}

// Display news items
function displayNews(newsItems) {
    const container = document.getElementById('marketNews');
    container.innerHTML = '';
    
    if (newsItems.length === 0) {
        container.innerHTML = '<div class="col-12"><div class="alert alert-info">No news items match your filters</div></div>';
        return;
    }
    
    newsItems.forEach((item, index) => {
        const newsCard = document.createElement('div');
        newsCard.className = 'col-md-6 col-lg-4 mb-3 fade-in';
        newsCard.style.animationDelay = `${index * 0.05}s`;
        
        const newsDate = new Date(item.created_at).toLocaleString([], {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
        
        // Different styling for tweets vs articles
        if (item.type === 'tweet') {
            const verified = item.author.verified ? '<i class="fas fa-check-circle text-primary ms-1"></i>' : '';
            const profileImg = item.author.profile_image || 'https://via.placeholder.com/50';
            
            newsCard.innerHTML = `
                <div class="card h-100 shadow-sm hover-shadow news-card border-start border-info border-4">
                    <div class="card-body">
                        <div class="d-flex align-items-center mb-2">
                            <img src="${profileImg}" class="rounded-circle me-2" width="40" height="40" alt="${item.author.name}">
                            <div class="flex-grow-1">
                                <span class="badge bg-info mb-1">
                                    <i class="fab fa-twitter"></i> Twitter
                                </span>
                                <div class="fw-bold">${item.author.name}${verified}</div>
                                <small class="text-muted">@${item.author.username}</small>
                            </div>
                        </div>
                        <small class="text-muted d-block mb-2">
                            <i class="far fa-clock"></i> ${newsDate}
                        </small>
                        <p class="card-text">${item.summary}</p>
                        ${item.symbols && item.symbols.length > 0 ? `
                            <div class="mb-2">
                                ${item.symbols.map(s => `<span class="badge bg-secondary me-1">$${s}</span>`).join('')}
                            </div>
                        ` : ''}
                        <div class="d-flex justify-content-between align-items-center">
                            <div class="text-muted small">
                                <i class="fas fa-heart"></i> ${item.metrics.likes}
                                <i class="fas fa-retweet ms-2"></i> ${item.metrics.retweets}
                            </div>
                            <a href="${item.url}" target="_blank" class="btn btn-sm btn-outline-primary">
                                View <i class="fas fa-external-link-alt"></i>
                            </a>
                        </div>
                    </div>
                </div>
            `;
        } else {
            // Article (Finnhub or Alpaca)
            const sourceColor = item.source.includes('Alpaca') ? 'success' : 'primary';
            const sourceIcon = item.source.includes('Alpaca') ? 'fa-bolt' : 'fa-newspaper';
            
            newsCard.innerHTML = `
                <div class="card h-100 shadow-sm hover-shadow news-card">
                    <div class="card-body">
                        <span class="badge bg-${sourceColor} mb-2">
                            <i class="fas ${sourceIcon}"></i> ${item.source}
                        </span>
                        <small class="text-muted d-block mb-2">
                            <i class="far fa-clock"></i> ${newsDate}
                        </small>
                        ${item.author ? `<small class="text-muted d-block mb-2"><i class="fas fa-user"></i> ${item.author}</small>` : ''}
                        <h6 class="card-title">${item.headline}</h6>
                        <p class="card-text text-muted">${truncateText(item.summary, 150)}</p>
                        ${item.symbols && item.symbols.length > 0 ? `
                            <div class="mb-2">
                                ${item.symbols.map(s => `<span class="badge bg-secondary me-1">$${s}</span>`).join('')}
                            </div>
                        ` : ''}
                        ${item.url ? `
                            <a href="${item.url}" target="_blank" class="btn btn-sm btn-outline-primary">
                                Read More <i class="fas fa-external-link-alt"></i>
                            </a>
                        ` : ''}
                    </div>
                </div>
            `;
        }
        
        container.appendChild(newsCard);
    });
}

// Load Quant Analysis
async function loadQuantAnalysis() {
    // Refresh AOS animations when switching to quant section
    setTimeout(() => {
        AOS.refresh();
    }, 100);
}

// View Research Paper
function viewResearch(type) {
    // Show loading indicator
    const loadingModal = document.createElement('div');
    loadingModal.className = 'modal fade';
    loadingModal.innerHTML = `
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-body text-center p-5">
                    <div class="spinner-border text-primary mb-3" role="status" style="width: 3rem; height: 3rem;">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <h5>Compiling LaTeX document...</h5>
                    <p class="text-muted">This may take a few seconds</p>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(loadingModal);
    const bsLoadingModal = new bootstrap.Modal(loadingModal);
    bsLoadingModal.show();
    
    if (type === 'advanced_trading') {
        // Handle markdown files
        fetch(`/api/research/${type}/markdown`)
            .then(response => response.json())
            .then(data => {
                bsLoadingModal.hide();
                loadingModal.remove();
                
                if (data.success) {
                    showMarkdownModal('Advanced Trading Theory', data.content);
                } else {
                    alert('Failed to load research paper: ' + data.error);
                }
            })
            .catch(error => {
                bsLoadingModal.hide();
                loadingModal.remove();
                console.error('Error loading markdown:', error);
                alert('Failed to load research paper');
            });
    } else {
        // Handle LaTeX PDFs - dynamically compile and open
        fetch(`/api/research/${type}`)
            .then(response => {
                bsLoadingModal.hide();
                loadingModal.remove();
                
                if (response.ok) {
                    return response.blob();
                } else {
                    throw new Error('Failed to compile PDF');
                }
            })
            .then(blob => {
                // Create a URL for the blob and open it in a new tab
                const url = window.URL.createObjectURL(blob);
                window.open(url, '_blank');
                // Clean up the URL after a short delay
                setTimeout(() => window.URL.revokeObjectURL(url), 100);
            })
            .catch(error => {
                console.error('Error compiling LaTeX:', error);
                alert('Failed to generate PDF. Please try again.');
            });
    }
}

// Show markdown in modal
function showMarkdownModal(title, markdown) {
    // Create a simple modal to display markdown
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.innerHTML = `
        <div class="modal-dialog modal-xl modal-dialog-scrollable">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">${title}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <pre style="white-space: pre-wrap; font-family: Georgia, serif; line-height: 1.6;">${markdown}</pre>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
    
    modal.addEventListener('hidden.bs.modal', function() {
        modal.remove();
    });
}

// Download Research Paper
function downloadResearch(type) {
    // Show loading indicator
    const loadingModal = document.createElement('div');
    loadingModal.className = 'modal fade';
    loadingModal.innerHTML = `
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-body text-center p-5">
                    <div class="spinner-border text-primary mb-3" role="status" style="width: 3rem; height: 3rem;">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <h5>Compiling LaTeX document...</h5>
                    <p class="text-muted">Preparing your download</p>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(loadingModal);
    const bsLoadingModal = new bootstrap.Modal(loadingModal);
    bsLoadingModal.show();
    
    // Generate and download the PDF
    fetch(`/api/research/${type}`)
        .then(response => {
            bsLoadingModal.hide();
            loadingModal.remove();
            
            if (response.ok) {
                return response.blob();
            } else {
                throw new Error('Failed to compile PDF');
            }
        })
        .then(blob => {
            // Create a download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${type}_model.pdf`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        })
        .catch(error => {
            console.error('Error downloading PDF:', error);
            alert('Failed to generate PDF for download. Please try again.');
        });
}

// Create New Algorithm
function createNewAlgorithm() {
    alert('Algorithm builder coming soon! This will allow you to create custom trading algorithms.');
}

// Run Backtest
function runBacktest() {
    const strategy = document.getElementById('backtestStrategy').value;
    const asset = document.getElementById('backtestAsset').value;
    const startDate = document.getElementById('backtestStartDate').value;
    const endDate = document.getElementById('backtestEndDate').value;
    
    if (!strategy || strategy === 'Select a strategy...') {
        alert('Please select a strategy');
        return;
    }
    
    if (!asset) {
        alert('Please enter an asset symbol');
        return;
    }
    
    if (!startDate || !endDate) {
        alert('Please select start and end dates');
        return;
    }
    
    // Show loading state
    const resultsDiv = document.getElementById('backtestResults');
    resultsDiv.innerHTML = '<div class="spinner-border text-primary" role="status"></div><p class="mt-2">Running backtest...</p>';
    
    // Simulate backtest (replace with actual API call)
    setTimeout(() => {
        const mockResults = {
            totalReturn: (Math.random() * 40 - 10).toFixed(2),
            sharpeRatio: (Math.random() * 2 + 0.5).toFixed(2),
            maxDrawdown: (Math.random() * 30 + 5).toFixed(2),
            winRate: (Math.random() * 30 + 40).toFixed(2),
            trades: Math.floor(Math.random() * 100 + 50)
        };
        
        resultsDiv.innerHTML = `
            <div class="metric-item">
                <strong>Total Return:</strong>
                <span class="${mockResults.totalReturn > 0 ? 'text-success' : 'text-danger'}">
                    ${mockResults.totalReturn}%
                </span>
            </div>
            <div class="metric-item">
                <strong>Sharpe Ratio:</strong>
                <span>${mockResults.sharpeRatio}</span>
            </div>
            <div class="metric-item">
                <strong>Max Drawdown:</strong>
                <span class="text-danger">-${mockResults.maxDrawdown}%</span>
            </div>
            <div class="metric-item">
                <strong>Win Rate:</strong>
                <span>${mockResults.winRate}%</span>
            </div>
            <div class="metric-item">
                <strong>Total Trades:</strong>
                <span>${mockResults.trades}</span>
            </div>
        `;
        
        // Show chart
        document.getElementById('backtestChart').style.display = 'block';
        createBacktestChart();
    }, 2000);
}

// Create Backtest Chart
function createBacktestChart() {
    const ctx = document.getElementById('backtestChartCanvas');
    
    if (charts.backtest) {
        charts.backtest.destroy();
    }
    
    // Generate mock equity curve data
    const labels = [];
    const data = [];
    let value = 10000;
    
    for (let i = 0; i < 100; i++) {
        labels.push(`Day ${i + 1}`);
        value += (Math.random() - 0.48) * 200;
        data.push(value);
    }
    
    charts.backtest = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Portfolio Value',
                data: data,
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Equity Curve',
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(0);
                        }
                    }
                }
            }
        }
    });
}

// Explore Model
function exploreModel(modelType) {
    let message = '';
    
    switch(modelType) {
        case 'ml':
            message = 'Machine Learning models for price prediction are coming soon! This will include LSTM networks, Random Forest, and XGBoost implementations.';
            break;
        case 'stochastic':
            message = 'Explore our research papers on Heston and SABR models in the Research Library tab!';
            break;
        case 'statistical':
            message = 'Statistical models including ARIMA, GARCH, and VAR are in development.';
            break;
    }
    
    alert(message);
}

// Search functionality
async function performSearch() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) return;
    
    try {
        const response = await fetch(`/api/search?query=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        if (data.results && data.results.length > 0) {
            // If exact match, load that stock
            const exactMatch = data.results.find(r => 
                r.symbol.toUpperCase() === query.toUpperCase()
            );
            
            if (exactMatch) {
                loadStockDetails(exactMatch.symbol);
            } else {
                // Show search results
                showSearchResults(data.results);
            }
        } else {
            alert('No stocks found matching your query');
        }
    } catch (error) {
        console.error('Error searching:', error);
        alert('Search failed. Please try again.');
    }
}

// Show search results
function showSearchResults(results) {
    // TODO: Implement search results modal
    console.log('Search results:', results);
}

// Notifications
function startNotificationPolling() {
    // Simulate notifications (in production, this would poll a backend endpoint)
    setInterval(() => {
        checkForNotifications();
    }, 60000); // Check every minute
}

function checkForNotifications() {
    // Simulate notification check
    // In production, fetch from /api/notifications
}

function showNotifications() {
    const modal = new bootstrap.Modal(document.getElementById('notificationsModal'));
    modal.show();
}

// Update market status
function updateMarketStatus() {
    const now = new Date();
    const hours = now.getHours();
    const day = now.getDay();
    
    const marketStatus = document.getElementById('marketStatus');
    
    // Simple market hours check (9:30 AM - 4:00 PM ET, Mon-Fri)
    if (day >= 1 && day <= 5 && hours >= 9 && hours < 16) {
        marketStatus.innerHTML = '<span class="text-success">● Open</span>';
    } else {
        marketStatus.innerHTML = '<span class="text-danger">● Closed</span>';
    }
}

// Utility functions
function formatDate(timestamp) {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

function truncateText(text, maxLength) {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

function updateLastUpdated() {
    const now = new Date();
    document.getElementById('lastUpdated').textContent = 
        `Updated: ${now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
}

function getGradeDescription(grade) {
    const descriptions = {
        'A': 'Excellent - Strong fundamentals across all metrics',
        'B': 'Good - Above average performance with minor concerns',
        'C': 'Average - Mixed performance, neutral outlook',
        'D': 'Below Average - Multiple areas of concern',
        'F': 'Poor - Significant fundamental issues'
    };
    return descriptions[grade] || 'No description available';
}

function getProgressBarClass(grade) {
    const classes = {
        'A': 'bg-success',
        'B': 'bg-info',
        'C': 'bg-warning',
        'D': 'bg-orange',
        'F': 'bg-danger'
    };
    return classes[grade] || 'bg-secondary';
}

// ============================================
// CHARTS FUNCTIONALITY
// ============================================

let priceChartInstance = null;
let volumeChartInstance = null;
let rsiChartInstance = null;
let macdChartInstance = null;
let currentPeriod = '1d';
let currentInterval = '5m';

function setupChartEventListeners() {
    // Timeframe button listeners
    document.querySelectorAll('#timeframeButtons button').forEach(button => {
        button.addEventListener('click', function() {
            // Update active state
            document.querySelectorAll('#timeframeButtons button').forEach(btn => {
                btn.classList.remove('active');
            });
            this.classList.add('active');
            
            // Load chart with new period
            const period = this.dataset.period;
            const interval = this.dataset.interval;
            currentPeriod = period;
            currentInterval = interval;
            
            if (currentStock) {
                loadCharts(currentStock, period, interval);
            }
        });
    });
}

function loadCharts(symbol, period = '1d', interval = '5m') {
    console.log(`📊 Loading charts for ${symbol} (${period}, ${interval})...`);
    
    // Load basic price chart
    fetch(`/api/charts/${symbol}?period=${period}&interval=${interval}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                renderPriceChart(data);
                renderVolumeChart(data);
                updateChartStats(data);
                console.log(`✓ Charts loaded: ${data.data_points} data points`);
            } else {
                console.error('Failed to load chart data:', data.error);
                showChartError(data.error);
            }
        })
        .catch(error => {
            console.error('Error loading charts:', error);
            showChartError('Failed to load chart data');
        });
    
    // Load technical indicators
    fetch(`/api/charts/${symbol}/indicators?period=${period}&interval=${interval}`)
        .then(response => response.json())
        .then(data => {
            console.log('📊 Technical indicators response:', data);
            if (data.success && data.indicators) {
                console.log('✓ RSI data points:', data.indicators.rsi ? data.indicators.rsi.length : 0);
                console.log('✓ MACD data points:', data.indicators.macd ? data.indicators.macd.macd.length : 0);
                renderRSIChart(data);
                renderMACDChart(data);
                renderPriceChartWithIndicators(data);
                console.log(`✓ Technical indicators loaded`);
            } else {
                console.warn('⚠️ No technical indicators data:', data);
            }
        })
        .catch(error => {
            console.error('Error loading technical indicators:', error);
        });
}

function renderPriceChart(data) {
    const ctx = document.getElementById('priceChart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (priceChartInstance) {
        priceChartInstance.destroy();
    }
    
    // Prepare data
    const labels = data.dates;
    const prices = data.close;
    
    // Determine color based on overall trend
    const firstPrice = prices[0];
    const lastPrice = prices[prices.length - 1];
    const isPositive = lastPrice >= firstPrice;
    const lineColor = isPositive ? 'rgba(34, 197, 94, 1)' : 'rgba(239, 68, 68, 1)';
    const backgroundColor = isPositive ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)';
    
    // Create new chart
    priceChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: `${data.symbol} Price`,
                data: prices,
                borderColor: lineColor,
                backgroundColor: backgroundColor,
                borderWidth: 2,
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                pointHoverRadius: 5,
                pointHoverBackgroundColor: lineColor,
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14
                    },
                    bodyFont: {
                        size: 13
                    },
                    callbacks: {
                        label: function(context) {
                            return `Price: $${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 8,
                        autoSkip: true
                    }
                },
                y: {
                    display: true,
                    position: 'right',
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// Filter calendar events
function filterCalendarEvents() {
    const typeFilter = document.getElementById('calendarTypeFilter').value;
    const importanceFilter = document.getElementById('calendarImportanceFilter').value;
    
    // Get all event items
    const eventItems = document.querySelectorAll('.calendar-event-item');
    const dateGroups = document.querySelectorAll('.calendar-date-group');
    const monthSections = document.querySelectorAll('.calendar-month-section');
    
    eventItems.forEach(item => {
        const itemType = item.getAttribute('data-type');
        const itemImportance = item.getAttribute('data-importance');
        
        let showItem = true;
        
        // Filter by type
        if (typeFilter !== 'all' && itemType !== typeFilter) {
            showItem = false;
        }
        
        // Filter by importance
        if (importanceFilter === 'high' && itemImportance !== 'high') {
            showItem = false;
        } else if (importanceFilter === 'medium' && itemImportance === 'low') {
            showItem = false;
        }
        
        item.style.display = showItem ? '' : 'none';
    });
    
    // Hide empty date groups
    dateGroups.forEach(group => {
        const visibleEvents = group.querySelectorAll('.calendar-event-item:not([style*="display: none"])');
        group.style.display = visibleEvents.length > 0 ? '' : 'none';
    });
    
    // Hide empty month sections
    monthSections.forEach(section => {
        const visibleGroups = section.querySelectorAll('.calendar-date-group:not([style*="display: none"])');
        section.style.display = visibleGroups.length > 0 ? '' : 'none';
    });
}

function renderVolumeChart(data) {
    const ctx = document.getElementById('volumeChart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (volumeChartInstance) {
        volumeChartInstance.destroy();
    }
    
    // Prepare data
    const labels = data.dates;
    const volumes = data.volume;
    
    // Create color array based on price movement
    const colors = data.close.map((price, index) => {
        if (index === 0) return 'rgba(156, 163, 175, 0.5)';
        return price >= data.close[index - 1] ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)';
    });
    
    // Create new chart
    volumeChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Volume',
                data: volumes,
                backgroundColor: colors,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return `Volume: ${formatVolume(context.parsed.y)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    display: true,
                    position: 'right',
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return formatVolume(value);
                        }
                    }
                }
            }
        }
    });
}

function updateChartStats(data) {
    const high = Math.max(...data.high);
    const low = Math.min(...data.low);
    const avgVolume = data.volume.reduce((a, b) => a + b, 0) / data.volume.length;
    const firstPrice = data.close[0];
    const lastPrice = data.close[data.close.length - 1];
    const change = lastPrice - firstPrice;
    const changePercent = (change / firstPrice * 100).toFixed(2);
    
    document.getElementById('chartHigh').textContent = `$${high.toFixed(2)}`;
    document.getElementById('chartLow').textContent = `$${low.toFixed(2)}`;
    document.getElementById('chartAvgVolume').textContent = formatVolume(avgVolume);
    
    const changeEl = document.getElementById('chartChange');
    changeEl.textContent = `${change >= 0 ? '+' : ''}${changePercent}%`;
    changeEl.className = change >= 0 ? 'text-success' : 'text-danger';
}

function formatVolume(volume) {
    if (volume >= 1000000000) {
        return (volume / 1000000000).toFixed(2) + 'B';
    } else if (volume >= 1000000) {
        return (volume / 1000000).toFixed(2) + 'M';
    } else if (volume >= 1000) {
        return (volume / 1000).toFixed(2) + 'K';
    }
    return volume.toFixed(0);
}

function showChartError(message) {
    const priceCanvas = document.getElementById('priceChart');
    const volumeCanvas = document.getElementById('volumeChart');
    
    if (priceCanvas && priceCanvas.parentElement) {
        priceCanvas.parentElement.innerHTML = `
            <div class="alert alert-warning" role="alert">
                <i class="fas fa-exclamation-triangle"></i> ${message}
            </div>
        `;
    }
    
    if (volumeCanvas && volumeCanvas.parentElement) {
        volumeCanvas.style.display = 'none';
    }
}

// Initialize chart event listeners when page loads
document.addEventListener('DOMContentLoaded', function() {
    setupChartEventListeners();
});

// ============ SEARCH & WATCHLIST MANAGEMENT ============

// Debounce function for search input
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Perform stock search
async function performSearch(query) {
    if (!query || query.length < 1) return;
    
    try {
        const response = await fetch(`/api/search/${encodeURIComponent(query)}`);
        const data = await response.json();
        
        if (data.success && data.results.length > 0) {
            displaySearchResults(data.results);
        } else {
            displaySearchResults([]);
        }
    } catch (error) {
        console.error('Search error:', error);
        displaySearchResults([]);
    }
}

// Display search results
function displaySearchResults(results) {
    // Create or get search results container
    let resultsContainer = document.getElementById('searchResults');
    
    if (!resultsContainer) {
        resultsContainer = document.createElement('div');
        resultsContainer.id = 'searchResults';
        resultsContainer.className = 'search-results-dropdown';
        document.getElementById('searchForm').appendChild(resultsContainer);
    }
    
    if (results.length === 0) {
        resultsContainer.innerHTML = `
            <div class="search-result-item">
                <i class="fas fa-info-circle text-muted"></i>
                <span class="ms-2">No stocks found</span>
            </div>
        `;
        resultsContainer.style.display = 'block';
        return;
    }
    
    resultsContainer.innerHTML = results.map(stock => `
        <div class="search-result-item" data-symbol="${stock.symbol}">
            <div class="d-flex justify-content-between align-items-center w-100">
                <div>
                    <strong>${stock.symbol}</strong>
                    <div class="small text-muted">${stock.name}</div>
                    <div class="small">
                        <span class="badge bg-secondary">${stock.exchange}</span>
                        ${stock.sector ? `<span class="badge bg-info ms-1">${stock.sector}</span>` : ''}
                    </div>
                </div>
                <div class="d-flex gap-2">
                    <button class="btn btn-sm btn-primary view-stock-btn" data-symbol="${stock.symbol}">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn btn-sm btn-success add-to-watchlist-btn" data-symbol="${stock.symbol}" data-name="${stock.name}">
                        <i class="fas fa-plus"></i>
                    </button>
                </div>
            </div>
        </div>
    `).join('');
    
    resultsContainer.style.display = 'block';
    
    // Add event listeners to buttons
    resultsContainer.querySelectorAll('.view-stock-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const symbol = this.dataset.symbol;
            hideSearchResults();
            showSection('dashboard');
            setActiveNav(document.getElementById('navDashboard'));
            loadStockDetails(symbol);
        });
    });
    
    resultsContainer.querySelectorAll('.add-to-watchlist-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const symbol = this.dataset.symbol;
            const name = this.dataset.name;
            addToWatchlist(symbol, name);
        });
    });
}

// Hide search results
function hideSearchResults() {
    const resultsContainer = document.getElementById('searchResults');
    if (resultsContainer) {
        resultsContainer.style.display = 'none';
    }
}

// Close search results when clicking outside
document.addEventListener('click', function(e) {
    const searchForm = document.getElementById('searchForm');
    const resultsContainer = document.getElementById('searchResults');
    
    if (resultsContainer && !searchForm.contains(e.target)) {
        hideSearchResults();
    }
});

// Load user watchlist from localStorage
function loadUserWatchlist() {
    const saved = localStorage.getItem('userWatchlist');
    if (saved) {
        try {
            userWatchlist = JSON.parse(saved);
        } catch (e) {
            console.error('Error loading watchlist:', e);
            userWatchlist = [];
        }
    }
    
    // If watchlist is empty, initialize with default stocks
    if (userWatchlist.length === 0) {
        const watchlistContainer = document.getElementById('watchlist');
        const defaultStocks = Array.from(watchlistContainer.querySelectorAll('.stock-item'))
            .map(item => item.dataset.symbol);
        
        userWatchlist = defaultStocks.map(symbol => ({
            symbol: symbol,
            name: '',
            addedAt: new Date().toISOString()
        }));
        saveUserWatchlist();
    }
    
    renderWatchlist();
}

// Save watchlist to localStorage
function saveUserWatchlist() {
    localStorage.setItem('userWatchlist', JSON.stringify(userWatchlist));
}

// Add stock to watchlist
function addToWatchlist(symbol, name) {
    // Check if already in watchlist
    if (userWatchlist.find(item => item.symbol === symbol)) {
        alert(`${symbol} is already in your watchlist`);
        return;
    }
    
    userWatchlist.push({
        symbol: symbol,
        name: name,
        addedAt: new Date().toISOString()
    });
    
    saveUserWatchlist();
    renderWatchlist();
    
    // Show success message
    showNotification(`Added ${symbol} to watchlist`, 'success');
    
    // Load price for new stock
    loadWatchlistPrices();
}

// Remove stock from watchlist
function removeFromWatchlist(symbol) {
    userWatchlist = userWatchlist.filter(item => item.symbol !== symbol);
    saveUserWatchlist();
    renderWatchlist();
    showNotification(`Removed ${symbol} from watchlist`, 'info');
}

// Render watchlist
function renderWatchlist() {
    const watchlistContainer = document.getElementById('watchlist');
    
    // Clear and rebuild
    watchlistContainer.innerHTML = '';
    
    // Add all stocks from userWatchlist (which now includes default + custom)
    userWatchlist.forEach(stock => {
        addWatchlistItem(watchlistContainer, stock.symbol, stock.name);
    });
    
    // Reload prices
    loadWatchlistPrices();
}

// Add watchlist item to DOM
function addWatchlistItem(container, symbol, name) {
    const item = document.createElement('a');
    item.href = '#';
    item.className = 'list-group-item list-group-item-action stock-item';
    item.dataset.symbol = symbol;
    
    item.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
            <div class="flex-grow-1">
                <strong>${symbol}</strong>
                ${name ? `<div class="small text-muted">${name.substring(0, 25)}${name.length > 25 ? '...' : ''}</div>` : ''}
            </div>
            <div class="d-flex align-items-center gap-2">
                <span class="badge bg-primary">...</span>
                <button class="btn btn-sm btn-link text-danger p-0 remove-from-watchlist" data-symbol="${symbol}" title="Remove from watchlist" style="font-size: 0.9rem;">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </div>
    `;
    
    // Click to view
    item.addEventListener('click', function(e) {
        if (!e.target.closest('.remove-from-watchlist')) {
            e.preventDefault();
            showSection('dashboard');
            setActiveNav(document.getElementById('navDashboard'));
            loadStockDetails(symbol);
        }
    });
    
    // Remove button
    const removeBtn = item.querySelector('.remove-from-watchlist');
    removeBtn.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        if (confirm(`Remove ${symbol} from watchlist?`)) {
            removeFromWatchlist(symbol);
        }
    });
    
    container.appendChild(item);
}

// Show notification
function showNotification(message, type = 'info') {
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} position-fixed`;
    toast.style.cssText = 'top: 80px; right: 20px; z-index: 9999; min-width: 250px;';
    toast.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i>
        ${message}
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Load and display company statistics
async function loadCompanyStatistics(symbol) {
    console.log(`📊 API CALL: /api/statistics/${symbol}`);
    
    try {
        const startTime = Date.now();
        const response = await fetch(`/api/statistics/${symbol}`);
        const data = await response.json();
        const endTime = Date.now();
        
        console.log(`✓ ${symbol} statistics loaded in ${endTime - startTime}ms`);
        
        const container = document.getElementById('companyStatistics');
        
        if (!data.success || !data.statistics) {
            container.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    Unable to load statistics for ${symbol}. Please try again later.
                </div>
            `;
            return;
        }
        
        const stats = data.statistics;
        
        // Build statistics HTML
        let html = '<div class="row">';
        
        // Profile Section
        if (stats.profile && Object.keys(stats.profile).length > 0) {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <h6 class="mb-0"><i class="fas fa-building"></i> Profile</h6>
                        </div>
                        <div class="card-body">
                            <table class="table table-sm mb-0">
                                ${formatStatRow('Market Cap', stats.profile['Market Cap'])}
                                ${formatStatRow('Enterprise Value', stats.profile['Enterprise Value'])}
                                ${formatStatRow('Shares Outstanding', stats.profile['Shares Outstanding'])}
                                ${formatStatRow('Revenue (TTM)', stats.profile['Revenue (TTM)'])}
                                ${formatStatRow('Employees', stats.profile['Employees'])}
                            </table>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Margins Section
        if (stats.margins && Object.keys(stats.margins).length > 0) {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header bg-success text-white">
                            <h6 class="mb-0"><i class="fas fa-percentage"></i> Margins</h6>
                        </div>
                        <div class="card-body">
                            <table class="table table-sm mb-0">
                                ${formatStatRow('Gross Margin', stats.margins['Gross'])}
                                ${formatStatRow('EBITDA Margin', stats.margins['EBITDA'])}
                                ${formatStatRow('Operating Margin', stats.margins['Operating'])}
                                ${formatStatRow('Pre-Tax Margin', stats.margins['Pre-Tax'])}
                                ${formatStatRow('Net Margin', stats.margins['Net'])}
                                ${formatStatRow('FCF Margin', stats.margins['FCF'])}
                            </table>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Returns Section
        if (stats.returns && Object.keys(stats.returns).length > 0) {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header bg-info text-white">
                            <h6 class="mb-0"><i class="fas fa-chart-line"></i> Returns (5Y Avg)</h6>
                        </div>
                        <div class="card-body">
                            <table class="table table-sm mb-0">
                                ${formatStatRow('Return on Assets (ROA)', stats.returns['ROA'])}
                                ${formatStatRow('Return on Total Assets (ROTA)', stats.returns['ROTA'])}
                                ${formatStatRow('Return on Equity (ROE)', stats.returns['ROE'])}
                                ${formatStatRow('Return on Capital Employed (ROCE)', stats.returns['ROCE'])}
                                ${formatStatRow('Return on Invested Capital (ROIC)', stats.returns['ROIC'])}
                            </table>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Valuation TTM Section
        if (stats.valuation_ttm && Object.keys(stats.valuation_ttm).length > 0) {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header bg-warning text-dark">
                            <h6 class="mb-0"><i class="fas fa-calculator"></i> Valuation (TTM)</h6>
                        </div>
                        <div class="card-body">
                            <table class="table table-sm mb-0">
                                ${formatStatRow('P/E Ratio', stats.valuation_ttm['P/E'])}
                                ${formatStatRow('P/B Ratio', stats.valuation_ttm['P/B'])}
                                ${formatStatRow('P/S Ratio', stats.valuation_ttm['P/S'])}
                                ${formatStatRow('EV/Sales', stats.valuation_ttm['EV/Sales'])}
                                ${formatStatRow('EV/EBITDA', stats.valuation_ttm['EV/EBITDA'])}
                                ${formatStatRow('P/FCF', stats.valuation_ttm['P/FCF'])}
                                ${formatStatRow('EV/Gross Profit', stats.valuation_ttm['EV/Gross Profit'])}
                            </table>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Valuation Forward Section
        if (stats.valuation_forward && Object.keys(stats.valuation_forward).length > 0) {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header bg-secondary text-white">
                            <h6 class="mb-0"><i class="fas fa-forward"></i> Valuation (Forward)</h6>
                        </div>
                        <div class="card-body">
                            <table class="table table-sm mb-0">
                                ${formatStatRow('Price Target', stats.valuation_forward['Price Target'])}
                                ${formatStatRow('Forward P/E', stats.valuation_forward['Forward P/E'])}
                                ${formatStatRow('PEG Ratio', stats.valuation_forward['PEG'])}
                                ${formatStatRow('Forward EV/Sales', stats.valuation_forward['Forward EV/Sales'])}
                                ${formatStatRow('Forward EV/EBITDA', stats.valuation_forward['Forward EV/EBITDA'])}
                                ${formatStatRow('Forward P/FCF', stats.valuation_forward['Forward P/FCF'])}
                            </table>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Financial Health Section
        if (stats.financial_health && Object.keys(stats.financial_health).length > 0) {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header bg-danger text-white">
                            <h6 class="mb-0"><i class="fas fa-heartbeat"></i> Financial Health</h6>
                        </div>
                        <div class="card-body">
                            <table class="table table-sm mb-0">
                                ${formatStatRow('Cash', stats.financial_health['Cash'])}
                                ${formatStatRow('Total Debt', stats.financial_health['Total Debt'])}
                                ${formatStatRow('Net Debt', stats.financial_health['Net Debt'])}
                                ${formatStatRow('Debt/Equity', stats.financial_health['Debt/Equity'])}
                                ${formatStatRow('Current Ratio', stats.financial_health['Current Ratio'])}
                                ${formatStatRow('Quick Ratio', stats.financial_health['Quick Ratio'])}
                                ${formatStatRow('EBIT/Interest', stats.financial_health['EBIT/Interest'])}
                            </table>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Growth Section
        if (stats.growth && Object.keys(stats.growth).length > 0) {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header bg-dark text-white">
                            <h6 class="mb-0"><i class="fas fa-seedling"></i> Growth</h6>
                        </div>
                        <div class="card-body">
                            <table class="table table-sm mb-0">
                                ${formatStatRow('Revenue Growth (3Y)', stats.growth['Revenue 3Yr'])}
                                ${formatStatRow('Revenue Growth (5Y)', stats.growth['Revenue 5Yr'])}
                                ${formatStatRow('Revenue Growth (10Y)', stats.growth['Revenue 10Yr'])}
                                ${formatStatRow('EPS Growth (3Y)', stats.growth['EPS 3Yr'])}
                                ${formatStatRow('EPS Growth (5Y)', stats.growth['EPS 5Yr'])}
                                ${formatStatRow('EPS Growth (10Y)', stats.growth['EPS 10Yr'])}
                                ${formatStatRow('Revenue Fwd (2Y)', stats.growth['Revenue Fwd 2Yr'])}
                                ${formatStatRow('EBITDA Fwd (2Y)', stats.growth['EBITDA Fwd 2Yr'])}
                                ${formatStatRow('EPS Fwd (2Y)', stats.growth['EPS Fwd 2Yr'])}
                                ${formatStatRow('LT Growth Est', stats.growth['EPS LT Growth Est'])}
                            </table>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Dividends Section
        if (stats.dividends && Object.keys(stats.dividends).length > 0) {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header" style="background-color: #6f42c1; color: white;">
                            <h6 class="mb-0"><i class="fas fa-coins"></i> Dividends</h6>
                        </div>
                        <div class="card-body">
                            <table class="table table-sm mb-0">
                                ${formatStatRow('Dividend Yield', stats.dividends['Yield'])}
                                ${formatStatRow('Payout Ratio', stats.dividends['Payout Ratio'])}
                                ${formatStatRow('DPS (Annual)', stats.dividends['DPS'])}
                                ${formatStatRow('Ex-Dividend Date', stats.dividends['Ex-Dividend Date'])}
                                ${formatStatRow('DPS Growth (3Y)', stats.dividends['DPS Growth 3Yr'])}
                                ${formatStatRow('DPS Growth (5Y)', stats.dividends['DPS Growth 5Yr'])}
                            </table>
                        </div>
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Error loading statistics:', error);
        document.getElementById('companyStatistics').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle"></i>
                Error loading statistics: ${error.message}
            </div>
        `;
    }
}

// Helper function to format statistics rows
function formatStatRow(label, value) {
    if (value === null || value === undefined || value === '—' || value === 'N/A') {
        return `<tr><td class="text-muted">${label}</td><td class="text-end text-muted">—</td></tr>`;
    }
    return `<tr><td>${label}</td><td class="text-end"><strong>${value}</strong></td></tr>`;
}

// Render RSI Chart
function renderRSIChart(data) {
    console.log('🎨 renderRSIChart called with data:', data);
    const ctx = document.getElementById('rsiChart');
    console.log('📊 RSI canvas element:', ctx);
    if (!ctx || !data.indicators || !data.indicators.rsi) {
        console.warn('⚠️ Cannot render RSI chart:', {
            hasCanvas: !!ctx,
            hasIndicators: !!data.indicators,
            hasRSI: !!(data.indicators && data.indicators.rsi)
        });
        return;
    }
    
    // Destroy existing chart
    if (rsiChartInstance) {
        rsiChartInstance.destroy();
    }
    
    const labels = data.dates;
    const rsiData = data.indicators.rsi;
    
    // Filter out null/NaN values and count valid data points
    const validCount = rsiData.filter(x => x !== null && x !== undefined && !isNaN(x)).length;
    console.log(`📊 RSI valid data points: ${validCount} / ${rsiData.length}`);
    
    // Skip rendering if too many NaN values
    if (validCount < 5) {
        console.warn('⚠️ Insufficient RSI data points for rendering');
        if (ctx.parentElement) {
            ctx.parentElement.innerHTML = '<p class="text-muted small">Insufficient data for RSI calculation. Try a longer time period.</p>';
        }
        return;
    }
    
    // Create horizontal line datasets for overbought/oversold levels
    const overboughtLine = new Array(labels.length).fill(70);
    const oversoldLine = new Array(labels.length).fill(30);
    
    rsiChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'RSI',
                    data: rsiData,
                    borderColor: 'rgba(139, 92, 246, 1)',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    order: 1
                },
                {
                    label: 'Overbought (70)',
                    data: overboughtLine,
                    borderColor: 'rgba(239, 68, 68, 0.5)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0,
                    order: 2
                },
                {
                    label: 'Oversold (30)',
                    data: oversoldLine,
                    borderColor: 'rgba(34, 197, 94, 0.5)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0,
                    order: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 10,
                        font: {
                            size: 11
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.datasetIndex === 0) {
                                return `RSI: ${context.parsed.y.toFixed(2)}`;
                            }
                            return context.dataset.label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 8
                    }
                },
                y: {
                    display: true,
                    position: 'right',
                    min: 0,
                    max: 100,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            }
        }
    });
}

// Render MACD Chart
function renderMACDChart(data) {
    console.log('🎨 renderMACDChart called with data:', data);
    const ctx = document.getElementById('macdChart');
    console.log('📊 MACD canvas element:', ctx);
    if (!ctx || !data.indicators || !data.indicators.macd) {
        console.warn('⚠️ Cannot render MACD chart:', {
            hasCanvas: !!ctx,
            hasIndicators: !!data.indicators,
            hasMACD: !!(data.indicators && data.indicators.macd)
        });
        return;
    }
    
    // Destroy existing chart
    if (macdChartInstance) {
        macdChartInstance.destroy();
    }
    
    const labels = data.dates;
    const macdData = data.indicators.macd;
    
    // Filter out null/NaN values and count valid data points
    const validCount = macdData.macd.filter(x => x !== null && x !== undefined && !isNaN(x)).length;
    console.log(`📊 MACD valid data points: ${validCount} / ${macdData.macd.length}`);
    
    // Skip rendering if too many NaN values
    if (validCount < 10) {
        console.warn('⚠️ Insufficient MACD data points for rendering');
        const parent = ctx.closest('.card-body');
        if (parent) {
            parent.innerHTML = '<p class="text-muted small text-center py-3">Insufficient data for MACD calculation. Try a longer time period (1M, 3M, or 1Y).</p>';
        }
        return;
    }
    
    macdChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'MACD',
                    data: macdData.macd,
                    borderColor: 'rgba(59, 130, 246, 1)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'Signal',
                    data: macdData.signal,
                    borderColor: 'rgba(239, 68, 68, 1)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'Histogram',
                    data: macdData.histogram,
                    backgroundColor: macdData.histogram.map(v => v >= 0 ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)'),
                    borderWidth: 0,
                    type: 'bar',
                    yAxisID: 'y'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 8
                    }
                },
                y: {
                    display: true,
                    position: 'right',
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            }
        }
    });
}

// Enhanced price chart with moving averages and Bollinger Bands
function renderPriceChartWithIndicators(data) {
    const ctx = document.getElementById('priceChart');
    if (!ctx || !data.indicators) return;
    
    // Destroy existing chart
    if (priceChartInstance) {
        priceChartInstance.destroy();
    }
    
    const labels = data.dates;
    const prices = data.price.close;
    const firstPrice = prices[0];
    const lastPrice = prices[prices.length - 1];
    const isPositive = lastPrice >= firstPrice;
    
    const datasets = [
        {
            label: `${data.symbol} Price`,
            data: prices,
            borderColor: isPositive ? 'rgba(34, 197, 94, 1)' : 'rgba(239, 68, 68, 1)',
            backgroundColor: isPositive ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
            borderWidth: 2,
            fill: false,
            tension: 0.1,
            pointRadius: 0,
            order: 1
        }
    ];
    
    // Add moving averages
    if (data.indicators.sma_20) {
        datasets.push({
            label: 'SMA 20',
            data: data.indicators.sma_20,
            borderColor: 'rgba(59, 130, 246, 1)',
            borderWidth: 1,
            fill: false,
            tension: 0.1,
            pointRadius: 0,
            borderDash: [5, 5],
            order: 2
        });
    }
    
    if (data.indicators.sma_50) {
        datasets.push({
            label: 'SMA 50',
            data: data.indicators.sma_50,
            borderColor: 'rgba(168, 85, 247, 1)',
            borderWidth: 1,
            fill: false,
            tension: 0.1,
            pointRadius: 0,
            borderDash: [5, 5],
            order: 2
        });
    }
    
    // Add Bollinger Bands
    if (data.indicators.bollinger) {
        datasets.push({
            label: 'BB Upper',
            data: data.indicators.bollinger.upper,
            borderColor: 'rgba(156, 163, 175, 0.5)',
            borderWidth: 1,
            fill: false,
            tension: 0.1,
            pointRadius: 0,
            order: 3
        });
        
        datasets.push({
            label: 'BB Lower',
            data: data.indicators.bollinger.lower,
            borderColor: 'rgba(156, 163, 175, 0.5)',
            borderWidth: 1,
            fill: '-1',
            backgroundColor: 'rgba(156, 163, 175, 0.1)',
            tension: 0.1,
            pointRadius: 0,
            order: 3
        });
    }
    
    priceChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: $${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 8
                    }
                },
                y: {
                    display: true,
                    position: 'right',
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

