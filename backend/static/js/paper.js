// ─── Paper trading ──────────────────────────────────────────────────────────
// The account lives in this browser (localStorage); the server only prices
// quotes and decides fills (/api/paper/*), from bars that start after the order
// was placed. No commission, 5 bps market slippage. The account is in USD:
// other listings (e.g. TSX in CAD) convert at the server's live FX rate.
// Shorting is allowed; gross exposure is capped at 2× equity, like a margin
// account. Depends on esc() from main.js.

var PAPER_KEY = 'investai.paper.v1';
var PAPER_START_CASH = 100000;
var PAPER_POLL_MS = 20000;
var PAPER_LEVERAGE = 2;     // max gross exposure / equity

var _paper = null;          // account state, see _paperNew()
var _paperQuotes = {};      // symbol → {price, prev_close, ...}
var _paperTimer = null;

function _paperNew() {
    return { cash: PAPER_START_CASH, start_cash: PAPER_START_CASH, positions: {},
             orders: [], realized: 0, created_at: new Date().toISOString() };
}

function _paperLoad() {
    try { _paper = JSON.parse(localStorage.getItem(PAPER_KEY)) || _paperNew(); }
    catch (e) { _paper = _paperNew(); }
    // v1 positions were long-only USD {qty, cost}
    Object.keys(_paper.positions).forEach(function (s) {
        var p = _paper.positions[s];
        if (p.cost != null) _paper.positions[s] = { qty: p.qty, avg_usd: p.cost / p.qty, currency: 'USD' };
    });
}

function _paperSave() {
    try { localStorage.setItem(PAPER_KEY, JSON.stringify(_paper)); } catch (e) {}
}

function _paperFmt(n, d) {
    return (n == null || isNaN(n)) ? '—'
        : Number(n).toLocaleString('en-US', { minimumFractionDigits: d == null ? 2 : d,
                                              maximumFractionDigits: d == null ? 2 : d });
}

// Last price in USD (falls back to the entry price until a quote arrives)
function _paperLastUSD(sym) {
    var q = _paperQuotes[sym];
    return q && q.price ? q.price * q.fx_usd : _paper.positions[sym].avg_usd;
}

// Equity and gross exposure, optionally as if `sym` held `qty` at `pxUsd`
function _paperBook(sym, qty, pxUsd, cash) {
    var eq = cash == null ? _paper.cash : cash, gross = 0, seen = false;
    Object.keys(_paper.positions).forEach(function (s) {
        var q = _paper.positions[s].qty, px = _paperLastUSD(s);
        if (s === sym) { q = qty; px = pxUsd; seen = true; }
        eq += q * px;
        gross += Math.abs(q * px);
    });
    if (sym && !seen) { eq += qty * pxUsd; gross += Math.abs(qty * pxUsd); }
    return { equity: eq, gross: gross };
}

function _paperEquity() { return _paperBook().equity; }

function _paperBuyingPower() {
    var b = _paperBook();
    return PAPER_LEVERAGE * b.equity - b.gross;
}

// ── Orders ─────────────────────────────────────────────────────────────────

// Returns an error string, or null when the order was queued.
function paperPlaceOrder(o) {
    if (!_paper) _paperLoad();
    var sym = String(o.symbol || '').trim().toUpperCase();
    var qty = Number(o.qty), limit = o.type === 'limit' ? Number(o.limit) : null;
    if (!/^[A-Z0-9][A-Z0-9.\-]{0,9}$/.test(sym)) return 'Enter a valid symbol.';
    if (!(qty > 0) || Math.floor(qty) !== qty) return 'Quantity must be a whole number above 0.';
    if (o.type === 'limit' && !(limit > 0)) return 'Enter a limit price.';
    var q = _paperQuotes[sym];
    if (q) {   // estimate now; the fill re-checks at its actual price
        var px = (limit || q.price) * q.fx_usd, held = _paper.positions[sym] ? _paper.positions[sym].qty : 0;
        var dq = o.side === 'buy' ? qty : -qty;
        var after = _paperBook(sym, held + dq, px, _paper.cash - dq * px);
        if (after.gross > PAPER_LEVERAGE * after.equity) return 'Not enough buying power (gross exposure is capped at 2× equity).';
    }
    _paper.orders.unshift({
        id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
        symbol: sym, side: o.side, qty: qty, type: o.type, limit: limit,
        placed_at: new Date().toISOString(), status: 'open', source: o.source || 'manual'
    });
    _paperSave();
    paperRender();
    paperSync();
    return null;
}

function paperCancel(id) {
    _paper.orders.forEach(function (o) { if (o.id === id && o.status === 'open') o.status = 'cancelled'; });
    _paperSave();
    paperRender();
}

function paperReset() {
    if (!confirm('Reset your paper account to $' + _paperFmt(PAPER_START_CASH, 0) + ' and clear all orders?')) return;
    _paper = _paperNew();
    _paperSave();
    paperRender();
}

// Apply a server fill. Positions are signed (short < 0) with a USD average entry
// price; reducing a position realizes P&L against that average, and an order
// that crosses zero opens the remainder at the fill price.
function _paperApplyFill(o, r) {
    var px = r.price * r.fx_usd;
    var pos = _paper.positions[o.symbol] || { qty: 0, avg_usd: 0, currency: r.currency };
    var dq = o.side === 'buy' ? o.qty : -o.qty;
    var after = _paperBook(o.symbol, pos.qty + dq, px, _paper.cash - dq * px);
    if (after.gross > PAPER_LEVERAGE * after.equity + 1e-6) {
        o.status = 'rejected'; o.reason = 'Not enough buying power at fill'; return;
    }
    if (pos.qty === 0 || (pos.qty > 0) === (dq > 0)) {          // open or add
        pos.avg_usd = (Math.abs(pos.qty) * pos.avg_usd + Math.abs(dq) * px) / (Math.abs(pos.qty) + Math.abs(dq));
    } else {                                                      // reduce, close or flip
        var closing = Math.min(Math.abs(dq), Math.abs(pos.qty));
        o.realized = closing * (px - pos.avg_usd) * (pos.qty > 0 ? 1 : -1);
        _paper.realized += o.realized;
        if (Math.abs(dq) > Math.abs(pos.qty)) pos.avg_usd = px;
    }
    pos.qty += dq;
    _paper.cash -= dq * px;
    if (pos.qty === 0) delete _paper.positions[o.symbol];
    else _paper.positions[o.symbol] = pos;
    o.status = 'filled';
    o.price = r.price;
    o.currency = r.currency;
    o.filled_at = r.filled_at;
}

// Ask the server to fill open orders, and refresh quotes for what's on screen.
function paperSync() {
    if (!_paper) _paperLoad();
    var open = _paper.orders.filter(function (o) { return o.status === 'open'; });
    var done = Promise.resolve();
    if (open.length) {
        done = fetch('/api/paper/fill', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ orders: open.slice(0, 50).map(function (o) {
                return { id: o.id, symbol: o.symbol, side: o.side, qty: o.qty,
                         type: o.type, limit: o.limit, placed_at: o.placed_at };
            }) })
        }).then(function (r) { return r.json(); }).then(function (d) {
            (d.results || []).forEach(function (r) {
                var o = _paper.orders.find(function (x) { return x.id === r.id; });
                if (!o || o.status !== 'open') return;
                if (r.status === 'filled') _paperApplyFill(o, r);
                else if (r.status === 'rejected') { o.status = 'rejected'; o.reason = r.reason; }
            });
            _paperSave();
        }).catch(function () {});
    }
    // quotes for open orders too, so a position opened by this sync has a price
    var syms = Object.keys(_paper.positions);
    open.forEach(function (o) { if (syms.indexOf(o.symbol) < 0) syms.push(o.symbol); });
    var tick = (document.getElementById('paperSymbol') || {}).value;
    if (tick && syms.indexOf(tick.toUpperCase()) < 0) syms.push(tick.toUpperCase());
    var quotes = syms.length ? fetch('/api/paper/quote?symbols=' + encodeURIComponent(syms.join(',')))
        .then(function (r) { return r.json(); })
        .then(function (d) {
            Object.keys(d.quotes || {}).forEach(function (s) {
                if (!d.quotes[s].error) _paperQuotes[s] = d.quotes[s];
            });
        }).catch(function () {}) : Promise.resolve();
    return Promise.all([done, quotes]).then(function () { paperRender(); paperRenderMarkers(); });
}

// Poll only while the Trading tab is on screen
function paperStartPolling() {
    if (_paperTimer) return;
    paperSync();
    _paperTimer = setInterval(function () {
        if (currentSection === 'trading') paperSync();
    }, PAPER_POLL_MS);
}

// ── Rendering ──────────────────────────────────────────────────────────────

function paperRender() {
    if (!_paper || !document.getElementById('paperPanel')) return;
    var eq = _paperEquity();
    var ret = (eq / _paper.start_cash - 1) * 100;
    document.getElementById('paperEquity').textContent = '$' + _paperFmt(eq);
    var retEl = document.getElementById('paperReturn');
    retEl.textContent = (ret >= 0 ? '+' : '') + _paperFmt(ret) + '%';
    retEl.style.color = ret >= 0 ? '#26c281' : '#f87171';
    document.getElementById('paperCash').textContent = '$' + _paperFmt(_paper.cash);
    document.getElementById('paperBP').textContent = '$' + _paperFmt(Math.max(0, _paperBuyingPower()));

    // Positions
    var rows = Object.keys(_paper.positions).sort().map(function (s) {
        var p = _paper.positions[s], q = _paperQuotes[s];
        var last = q ? q.price * q.fx_usd : null, avg = p.avg_usd;
        var pnl = last != null ? (last - avg) * p.qty : null;
        var col = pnl == null ? '#94a3b8' : (pnl >= 0 ? '#26c281' : '#f87171');
        return '<tr class="paper-row" data-symbol="' + esc(s) + '">' +
            '<td>' + esc(s) + (p.currency && p.currency !== 'USD' ? ' <span class="text-secondary">' + esc(p.currency) + '</span>' : '') +
            (p.qty < 0 ? ' <span style="color:#f87171">short</span>' : '') + '</td>' +
            '<td class="text-end">' + _paperFmt(p.qty, 0) + '</td>' +
            '<td class="text-end">' + _paperFmt(avg) + '</td>' +
            '<td class="text-end">' + _paperFmt(last) + '</td>' +
            '<td class="text-end" style="color:' + col + '">' + (pnl == null ? '—' : (pnl >= 0 ? '+' : '') + _paperFmt(pnl)) + '</td></tr>';
    }).join('');
    document.getElementById('paperPositions').innerHTML = rows ||
        '<tr><td colspan="5" class="text-secondary">No positions yet.</td></tr>';

    // Orders: open first, then the most recent others
    var open = _paper.orders.filter(function (o) { return o.status === 'open'; });
    var rest = _paper.orders.filter(function (o) { return o.status !== 'open'; }).slice(0, 12);
    var badge = { open: '#eab308', filled: '#26c281', cancelled: '#64748b', rejected: '#f87171' };
    document.getElementById('paperOrders').innerHTML = open.concat(rest).map(function (o) {
        var what = (o.side === 'buy' ? 'Buy ' : 'Sell ') + _paperFmt(o.qty, 0) + ' ' + esc(o.symbol) +
                   (o.type === 'limit' ? ' @ ' + _paperFmt(o.limit) + ' lmt' : ' @ mkt');
        var detail = o.status === 'filled' ? 'filled ' + _paperFmt(o.price) +
                         (o.currency && o.currency !== 'USD' ? ' ' + esc(o.currency) : '') +
                         (o.realized != null ? ' · P&amp;L ' + (o.realized >= 0 ? '+' : '') + _paperFmt(o.realized) : '')
                   : o.status === 'rejected' ? esc(o.reason || 'rejected')
                   : o.status === 'open' ? 'waiting for the next price after ' + esc(new Date(o.placed_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
                   : 'cancelled';
        return '<div class="paper-order">' +
            '<span style="color:' + badge[o.status] + '">●</span> ' + what +
            (o.source === 'chat' ? ' <span class="text-secondary">(from chat)</span>' : '') +
            '<div class="text-secondary">' + detail +
            (o.status === 'open' ? ' · <a href="#" onclick="paperCancel(\'' + esc(o.id) + '\'); return false;">cancel</a>' : '') +
            '</div></div>';
    }).join('') || '<div class="text-secondary">No orders yet.</div>';

    _paperRenderEstimate();
}

function _paperRenderEstimate() {
    var sym = document.getElementById('paperSymbol').value.trim().toUpperCase();
    var qty = Number(document.getElementById('paperQty').value);
    var type = document.getElementById('paperType').value;
    var q = _paperQuotes[sym];
    var px = type === 'limit' ? Number(document.getElementById('paperLimit').value) : (q ? q.price : null);
    var cur = q ? q.currency : 'USD';
    document.getElementById('paperEstimate').textContent = !(px && qty > 0) ? '' :
        '≈ ' + (cur === 'USD' ? '$' + _paperFmt(px * qty)
                              : _paperFmt(px * qty) + ' ' + cur + ' (≈ $' + _paperFmt(px * qty * q.fx_usd) + ')') +
        (type === 'market' ? ' at last price' : '');
}

// Buy/sell arrows on the trading chart for this symbol's fills
function paperRenderMarkers() {
    if (typeof _lwPriceSeries === 'undefined' || !_lwPriceSeries || !_tradingChartData) return;
    var sym = String(_tradingChartData.ticker || '').toUpperCase();
    var times = _tradingChartData.bars.map(function (b) { return b.time; });
    var markers = [];
    _paper.orders.forEach(function (o) {
        if (o.status !== 'filled' || o.symbol !== sym) return;
        var t = Math.floor(new Date(o.filled_at).getTime() / 1000);
        var bar = null;                       // the bar the fill falls in
        for (var i = times.length - 1; i >= 0; i--) { if (times[i] <= t) { bar = times[i]; break; } }
        if (bar == null) return;
        var buy = o.side === 'buy';
        markers.push({ time: bar, position: buy ? 'belowBar' : 'aboveBar',
                       color: buy ? '#26c281' : '#f87171', shape: buy ? 'arrowUp' : 'arrowDown',
                       text: (buy ? 'B ' : 'S ') + o.qty });
    });
    markers.sort(function (a, b) { return a.time - b.time; });
    try { _lwPriceSeries.setMarkers(markers); } catch (e) {}
}

// Follow the charted ticker in the order ticket
function paperOnChartLoaded(ticker) {
    var el = document.getElementById('paperSymbol');
    if (el) el.value = ticker;
    paperSync();
}

// ── Ticket ─────────────────────────────────────────────────────────────────

function paperSubmit(side) {
    var err = paperPlaceOrder({
        symbol: document.getElementById('paperSymbol').value,
        side: side,
        qty: document.getElementById('paperQty').value,
        type: document.getElementById('paperType').value,
        limit: document.getElementById('paperLimit').value
    });
    var msg = document.getElementById('paperTicketMsg');
    msg.textContent = err || 'Order placed. It fills from the next price after now.';
    msg.style.color = err ? '#f87171' : '#94a3b8';
}

// ── Model account ──────────────────────────────────────────────────────────

var _paperModelChart = null;

function paperLoadModelAccount() {
    var box = document.getElementById('paperModel');
    fetch('/api/paper/model?ticker=NVDA').then(function (r) { return r.json(); }).then(function (d) {
        if (!d.success) { box.innerHTML = '<div class="text-secondary">' + esc(d.error) + '</div>'; return; }
        var last = d.equity[d.equity.length - 1];
        var mret = (last.equity / d.start_cash - 1) * 100, bret = (last.benchmark / d.start_cash - 1) * 100;
        var sig = d.last_signal || {};
        var col = function (v) { return v >= 0 ? '#26c281' : '#f87171'; };
        box.innerHTML =
            '<div class="d-flex justify-content-between"><span>' + esc(d.ticker) + ' model</span>' +
            '<span style="color:' + col(mret) + '">' + (mret >= 0 ? '+' : '') + _paperFmt(mret) + '%</span></div>' +
            '<div class="d-flex justify-content-between text-secondary"><span>buy &amp; hold</span>' +
            '<span style="color:' + col(bret) + '">' + (bret >= 0 ? '+' : '') + _paperFmt(bret) + '%</span></div>' +
            '<div class="text-secondary mt-1">Signal ' + esc(sig.signal || '—') + ' (' + esc(sig.date || '') + ') · ' +
            (d.shares > 0 ? 'holding ' + _paperFmt(d.shares, 0) + ' sh' : 'in cash') +
            (d.pending ? ' · ' + esc(d.pending.side) + ' at next open' : '') + '</div>' +
            '<div id="paperModelChart" style="height:110px; margin-top:6px;"></div>' +
            '<div class="text-secondary" style="font-size:.7rem">Since ' + esc(d.equity[0].date) + ' · ' +
            d.trades.length + ' trade' + (d.trades.length === 1 ? '' : 's') +
            ' · decides at the close, fills at the next open</div>';
        if (_paperModelChart) { try { _paperModelChart.remove(); } catch (e) {} }
        _paperModelChart = LightweightCharts.createChart(document.getElementById('paperModelChart'), {
            autoSize: true, layout: { background: { type: 'solid', color: '#0a0a0a' }, textColor: '#64748b' },
            grid: { vertLines: { visible: false }, horzLines: { color: '#151515' } },
            timeScale: { visible: false }, rightPriceScale: { borderVisible: false },
            handleScroll: false, handleScale: false
        });
        var pts = function (k) { return d.equity.map(function (e) { return { time: e.date, value: e[k] }; }); };
        _paperModelChart.addLineSeries({ color: '#00c9a7', lineWidth: 2, title: 'model' }).setData(pts('equity'));
        _paperModelChart.addLineSeries({ color: '#64748b', lineWidth: 1, title: 'B&H' }).setData(pts('benchmark'));
    }).catch(function () { box.innerHTML = '<div class="text-secondary">Model account unavailable.</div>'; });
}

// ── For the AI chat ────────────────────────────────────────────────────────

// Compact portfolio snapshot for the chat's SCREEN block
function paperSummary() {
    if (!_paper) _paperLoad();
    return {
        currency: 'USD',
        cash: Math.round(_paper.cash * 100) / 100,
        equity: Math.round(_paperEquity() * 100) / 100,
        buying_power: Math.round(_paperBuyingPower() * 100) / 100,
        start_cash: _paper.start_cash,
        positions: Object.keys(_paper.positions).map(function (s) {
            var p = _paper.positions[s], q = _paperQuotes[s];
            return { symbol: s, qty: p.qty, avg_cost_usd: Math.round(p.avg_usd * 100) / 100,
                     last_usd: q ? Math.round(q.price * q.fx_usd * 100) / 100 : null,
                     listing_currency: p.currency };
        }),
        open_orders: _paper.orders.filter(function (o) { return o.status === 'open'; }).map(function (o) {
            return { side: o.side, qty: o.qty, symbol: o.symbol, type: o.type, limit: o.limit };
        })
    };
}

document.addEventListener('DOMContentLoaded', function () {
    _paperLoad();
    if (!document.getElementById('paperPanel')) return;
    document.getElementById('paperType').addEventListener('change', function () {
        document.getElementById('paperLimit').style.display = this.value === 'limit' ? '' : 'none';
        _paperRenderEstimate();
    });
    ['paperQty', 'paperLimit'].forEach(function (id) {
        document.getElementById(id).addEventListener('input', _paperRenderEstimate);
    });
    document.getElementById('paperSymbol').addEventListener('change', paperSync);
    document.getElementById('paperPositions').addEventListener('click', function (e) {
        var row = e.target.closest('.paper-row');
        if (!row) return;
        document.getElementById('tradingSymbol').value = row.dataset.symbol;
        loadTradingChart();
    });
    document.getElementById('navTrading').addEventListener('click', function () {
        paperStartPolling();
        paperLoadModelAccount();
    });
    paperRender();
});
