// Global state
let currentStock = null;
let notifications = [];
let charts = {};
let stockDataCache = {}; // Cache for individual stock data

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadDashboard();
    loadEarningsCalendar();
    setupEventListeners();
    startNotificationPolling();
    updateMarketStatus();
    loadWatchlistPrices(); // Load prices for all watchlist stocks
});

// Setup event listeners
function setupEventListeners() {
    // Search form
    document.getElementById('searchForm').addEventListener('submit', function(e) {
        e.preventDefault();
        performSearch();
    });
    
    // Watchlist items
    document.querySelectorAll('.stock-item').forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const symbol = this.dataset.symbol;
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
        
        // Update price display
        if (data.quote) {
            document.getElementById('stockPrice').textContent = `$${data.quote.c.toFixed(2)}`;
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
            companyInfo.innerHTML = `
                <div class="company-details">
                    <p><strong>Name:</strong> ${data.company.name || 'N/A'}</p>
                    <p><strong>Industry:</strong> ${data.company.finnhubIndustry || 'N/A'}</p>
                    <p><strong>Market Cap:</strong> $${(data.company.marketCapitalization || 0).toFixed(2)}B</p>
                    <p><strong>Country:</strong> ${data.company.country || 'N/A'}</p>
                    ${data.company.weburl ? `<p><a href="${data.company.weburl}" target="_blank">Company Website</a></p>` : ''}
                </div>
            `;
        } else {
            companyInfo.innerHTML = '<p class="text-muted">Company information not available</p>';
        }
        
        // Update recent news
        const newsContainer = document.getElementById('recentNews');
        if (data.news && data.news.length > 0) {
            newsContainer.innerHTML = '<div class="list-group">';
            data.news.slice(0, 5).forEach(news => {
                newsContainer.innerHTML += `
                    <a href="${news.url}" target="_blank" class="list-group-item list-group-item-action">
                        <div class="d-flex w-100 justify-content-between">
                            <h6 class="mb-1">${truncateText(news.headline, 80)}</h6>
                            <small>${formatDate(news.datetime)}</small>
                        </div>
                    </a>
                `;
            });
            newsContainer.innerHTML += '</div>';
        } else {
            newsContainer.innerHTML = '<p class="text-muted">No recent news available</p>';
        }
        
    } catch (error) {
        console.error(`✗ Error loading overview for ${symbol}:`, error);
        document.getElementById('companyInfo').innerHTML = 
            '<div class="alert alert-danger">Failed to load stock data. Please try again.</div>';
    }
}

// Handle tab changes with separate API calls
function handleTabChange(target, symbol) {
    console.log(`\n🔄 TAB CHANGE: ${target} for ${symbol}`);
    
    switch(target) {
        case '#sentiment':
            loadSentiment(symbol);
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

// Load earnings calendar
async function loadEarningsCalendar() {
    const container = document.getElementById('earningsCalendar');
    
    try {
        const response = await fetch('/api/calendar');
        const data = await response.json();
        
        if (data.earnings && data.earnings.length > 0) {
            container.innerHTML = `
                <div class="calendar-grid">
                    ${data.earnings.map(event => `
                        <div class="calendar-item">
                            <div class="calendar-date">
                                <i class="far fa-calendar"></i> ${event.date}
                            </div>
                            <h6 class="mt-2">${event.symbol}</h6>
                            <div class="calendar-company">${event.company || 'N/A'}</div>
                            ${event.epsEstimate ? `<small class="text-muted">Est. EPS: $${event.epsEstimate}</small>` : ''}
                        </div>
                    `).join('')}
                </div>
            `;
        } else {
            container.innerHTML = '<p class="text-muted">No upcoming earnings events</p>';
        }
    } catch (error) {
        console.error('Error loading calendar:', error);
        container.innerHTML = '<div class="alert alert-danger">Failed to load earnings calendar</div>';
    }
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
