// tests/js/test_paper_accounting.js — paper account bookkeeping (static/js/paper.js):
// long/short/flip P&L, FX conversion, the 2× buying-power cap, v1 migration.
// Run: node tests/js/test_paper_accounting.js
const path = require('path');
const fs = require('fs');
global.document = { addEventListener(){}, getElementById(){ return null; } };
global.localStorage = { getItem(){ return null; }, setItem(){} };
global.esc = x => String(x);
eval(fs.readFileSync(path.join(__dirname, '../../backend/static/js/paper.js'), 'utf8') + '\n;global.T={_paperLoad,_paperApplyFill,_paperEquity,_paperBuyingPower,paperPlaceOrder,get p(){return _paper}, q:_paperQuotes};');
const near = (a,b) => Math.abs(a-b) < 1e-6;
const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) process.exitCode = 1; };
T._paperLoad();
const fill = (side, qty, sym, price, fx=1, cur='USD') => { const o={side, qty, symbol:sym, status:'open'}; T._paperApplyFill(o,{price, fx_usd:fx, currency:cur, filled_at:'x'}); return o; };
T.q.NVDA = {price:100, fx_usd:1, currency:'USD'};
fill('buy', 100, 'NVDA', 100);
ok(T.p.positions.NVDA.qty===100 && near(T.p.cash, 90000), 'long open: qty 100, cash 90k');
fill('sell', 40, 'NVDA', 110); 
ok(T.p.positions.NVDA.qty===60 && near(T.p.realized, 400), 'partial sell realizes +400 at avg 100');
let o = fill('sell', 100, 'NVDA', 120);   // flip: close 60 (+1200), open short 40 @120
ok(T.p.positions.NVDA.qty===-40 && near(T.p.positions.NVDA.avg_usd,120) && near(o.realized,1200), 'flip to short 40 @120, realized +1200');
T.q.NVDA.price = 100;
ok(near(T.p.positions.NVDA.qty*(100-T.p.positions.NVDA.avg_usd), 800), 'short gains +800 when price drops to 100');
const eqBefore = T._paperEquity();
o = fill('buy', 40, 'NVDA', 100);
ok(!T.p.positions.NVDA && near(o.realized, 800) && near(T._paperEquity(), eqBefore), 'cover short: realized +800, equity unchanged by the trade');
ok(near(T._paperEquity(), 100000+400+1200+800), 'equity = start + all realized (2400)');
// CAD listing converts at fx
T.q['TD.TO'] = {price:160, fx_usd:0.7, currency:'CAD'};
fill('buy', 10, 'TD.TO', 160, 0.7, 'CAD');
ok(near(T.p.positions['TD.TO'].avg_usd, 112) && T.p.positions['TD.TO'].currency==='CAD', 'CAD fill: avg $112 = 160 CAD × 0.7');
// buying power: 2x equity cap
const r = fill('buy', 2100, 'NVDA', 100);   // $210k > 2 × ~$102.4k equity
ok(r.status==='rejected' && /buying power/.test(r.reason), 'buy beyond 2x equity rejected at fill');
ok(T.paperPlaceOrder({symbol:'NVDA', side:'sell', qty:3000, type:'market'})!==null, 'short beyond 2x equity rejected at placement');
ok(T.paperPlaceOrder({symbol:'NVDA', side:'sell', qty:10, type:'market'})===null, 'small short allowed at placement');
// v1 migration
global.localStorage.getItem = () => JSON.stringify({cash:50000,start_cash:100000,positions:{AAPL:{qty:10,cost:3000}},orders:[],realized:0});
T._paperLoad();
ok(near(T.p.positions.AAPL.avg_usd,300) && T.p.positions.AAPL.currency==='USD', 'v1 account migrated (avg 300)');
