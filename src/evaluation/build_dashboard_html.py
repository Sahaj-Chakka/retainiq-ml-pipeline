"""
RetainIQ · src/evaluation/build_dashboard_html.py
--------------------------------------------------
Reads dashboard/dashboard_data.json and writes a single self-contained
dashboard/index.html (data inlined) — works on GitHub Pages and via file://.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "dashboard"


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RetainIQ · Hotel Booking Analytics</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#0e1116; --panel:#161b22; --panel2:#1c232d; --line:#2a323d;
    --ink:#e6edf3; --muted:#8b949e; --accent:#E85D04; --accent2:#4aa8ff;
    --good:#3fb950; --warn:#d29922;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--ink);font-family:'IBM Plex Sans',sans-serif;
       line-height:1.5;-webkit-font-smoothing:antialiased;
       background-image:radial-gradient(circle at 15% -10%,rgba(232,93,4,.10),transparent 40%),
                        radial-gradient(circle at 90% 0%,rgba(74,168,255,.08),transparent 35%);}
  .wrap{max-width:1180px;margin:0 auto;padding:48px 28px 80px}
  header{border-bottom:1px solid var(--line);padding-bottom:26px;margin-bottom:8px}
  .eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.22em;
           text-transform:uppercase;color:var(--accent);margin-bottom:10px}
  h1{font-family:'Fraunces',serif;font-weight:600;font-size:clamp(34px,5vw,54px);
     line-height:1.02;letter-spacing:-.01em}
  h1 em{font-style:italic;color:var(--accent)}
  .sub{color:var(--muted);max-width:680px;margin-top:14px;font-size:15px}
  .synthtag{display:inline-block;margin-top:16px;font-family:'IBM Plex Mono',monospace;
            font-size:11px;color:var(--warn);border:1px solid var(--warn);
            border-radius:999px;padding:5px 12px;letter-spacing:.05em}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:34px 0 10px}
  .kpi{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px 18px 16px;
       position:relative;overflow:hidden;opacity:0;transform:translateY(12px);
       animation:rise .6s cubic-bezier(.2,.7,.2,1) forwards}
  .kpi:before{content:"";position:absolute;inset:0 auto 0 0;width:3px;background:var(--accent)}
  .kpi .label{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}
  .kpi .val{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:30px;margin-top:8px}
  .kpi .val small{font-size:14px;color:var(--muted)}
  @keyframes rise{to{opacity:1;transform:none}}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:22px}
  .card.wide{grid-column:1 / -1}
  .card h3{font-family:'Fraunces',serif;font-weight:600;font-size:19px;margin-bottom:4px}
  .card p.cap{color:var(--muted);font-size:13px;margin-bottom:16px}
  .chartbox{position:relative;height:300px}
  .chartbox.tall{height:340px}
  table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
  th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}
  th{color:var(--muted);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.08em}
  td.num{font-family:'IBM Plex Mono',monospace;text-align:right}
  tr.best td{color:var(--accent);font-weight:600}
  .pill{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--good)}
  footer{margin-top:40px;padding-top:22px;border-top:1px solid var(--line);
         color:var(--muted);font-size:13px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px}
  a{color:var(--accent2);text-decoration:none}
  @media(max-width:760px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow">RetainIQ · Synthetic Hotel Analytics</div>
    <h1>Booking <em>cancellations</em>, revenue &amp; <em>forecasting</em></h1>
    <p class="sub">An end-to-end ML pipeline on 70,000 synthetic hotel bookings, modeled on a real
       (confidential) hotel group project. Every figure below is a real model output — not hand-set.</p>
    <span class="synthtag">⚠ SYNTHETIC DATA · patterns illustrate methodology, not a real hotel</span>
  </header>

  <section class="kpis" id="kpis"></section>

  <div class="grid">
    <div class="card wide">
      <h3>Monthly revenue</h3><p class="cap">Honoured bookings, 24 months — clear seasonality (Apr/Sep peaks, May/Jun troughs)</p>
      <div class="chartbox"><canvas id="revLine"></canvas></div>
    </div>

    <div class="card">
      <h3>Cancellation rate by source</h3><p class="cap">Online channels cancel far more than walk-ins/corporate</p>
      <div class="chartbox"><canvas id="cancelBar"></canvas></div>
    </div>

    <div class="card">
      <h3>Room-type demand</h3><p class="cap">Deluxe dominates — a real pattern from the source project</p>
      <div class="chartbox"><canvas id="roomDonut"></canvas></div>
    </div>

    <div class="card">
      <h3>Revenue concentration (Pareto)</h3><p class="cap">Top 18% of guests drive the majority of revenue</p>
      <div class="chartbox"><canvas id="pareto"></canvas></div>
    </div>

    <div class="card">
      <h3>Revenue share by source</h3><p class="cap">Corporate punches above its booking volume</p>
      <div class="chartbox"><canvas id="revSource"></canvas></div>
    </div>

    <div class="card wide">
      <h3>Cancellation model benchmark <span class="pill">AUC</span></h3>
      <p class="cap">9 algorithms, identical splits — gradient boosting leads; XGBoost selected for production</p>
      <table id="benchTable"><thead><tr><th>Model</th><th style="text-align:right">AUC</th>
        <th style="text-align:right">Accuracy</th><th style="text-align:right">F1</th></tr></thead><tbody></tbody></table>
    </div>
  </div>

  <footer>
    <span>RetainIQ · built by Sahaj Chakka · synthetic data, real pipeline</span>
    <span><a href="https://github.com/Sahaj-Chakka/RetainIQ">github.com/Sahaj-Chakka/RetainIQ</a></span>
  </footer>
</div>

<script>
const DATA = __DATA__;
const C = {ink:'#e6edf3',muted:'#8b949e',accent:'#E85D04',accent2:'#4aa8ff',line:'#2a323d',
           good:'#3fb950',navy:'#1f6feb'};
Chart.defaults.color = C.muted;
Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";
Chart.defaults.borderColor = C.line;
const inr = v => '₹' + Intl.NumberFormat('en-IN',{notation:'compact',maximumFractionDigits:1}).format(v);

// KPIs
const k = DATA.kpis;
const cards = [
  ['Bookings', Intl.NumberFormat('en-IN').format(k.bookings)],
  ['Cancellation rate', (k.cancellation_rate*100).toFixed(1)+'<small>%</small>'],
  ['Total revenue', inr(k.total_revenue)],
  ['Avg booking', '₹'+Intl.NumberFormat('en-IN').format(k.avg_booking_value)],
  ['Best cancel AUC', k.best_cancel_auc.toFixed(3)+'<small> '+k.best_cancel_model.split(' ')[0]+'</small>'],
  ['Revenue model R²', k.best_revenue_r2.toFixed(3)],
  ['Forecast vs baseline', '−'+k.forecast_improvement.toFixed(0)+'<small>% err</small>'],
  ['Top 18% guests', (k.top18_revenue_share*100).toFixed(0)+'<small>% rev</small>'],
];
document.getElementById('kpis').innerHTML = cards.map((c,i)=>
  `<div class="kpi" style="animation-delay:${i*60}ms"><div class="label">${c[0]}</div><div class="val">${c[1]}</div></div>`).join('');

// Monthly revenue line
new Chart(revLine,{type:'line',data:{labels:DATA.monthly_revenue.map(d=>d.month),
  datasets:[{data:DATA.monthly_revenue.map(d=>d.revenue),borderColor:C.accent,
    backgroundColor:'rgba(232,93,4,.12)',fill:true,tension:.35,pointRadius:0,borderWidth:2}]},
  options:{plugins:{legend:{display:false}},scales:{y:{ticks:{callback:inr}},x:{ticks:{maxTicksLimit:12}}},
    maintainAspectRatio:false}});

// Cancellation by source
new Chart(cancelBar,{type:'bar',data:{labels:DATA.cancellation_by_source.map(d=>d.booking_source),
  datasets:[{data:DATA.cancellation_by_source.map(d=>+(d.cancellation_rate*100).toFixed(1)),
    backgroundColor:C.accent,borderRadius:6}]},
  options:{indexAxis:'y',plugins:{legend:{display:false}},
    scales:{x:{ticks:{callback:v=>v+'%'}}},maintainAspectRatio:false}});

// Room donut
new Chart(roomDonut,{type:'doughnut',data:{labels:DATA.room_mix.map(d=>d.room_type),
  datasets:[{data:DATA.room_mix.map(d=>d.pct),backgroundColor:[C.accent,C.navy],borderColor:'#161b22',borderWidth:3}]},
  options:{cutout:'62%',plugins:{legend:{position:'bottom'}},maintainAspectRatio:false}});

// Pareto
new Chart(pareto,{type:'line',data:{labels:DATA.pareto.map(d=>(d.x*100).toFixed(0)),
  datasets:[{data:DATA.pareto.map(d=>+(d.y*100).toFixed(1)),borderColor:C.accent2,
    backgroundColor:'rgba(74,168,255,.10)',fill:true,pointRadius:0,borderWidth:2,tension:.2}]},
  options:{plugins:{legend:{display:false}},
    scales:{y:{ticks:{callback:v=>v+'%'},title:{display:true,text:'cum. revenue'}},
            x:{ticks:{maxTicksLimit:6,callback:v=>v+'%'},title:{display:true,text:'cum. guests'}}},
    maintainAspectRatio:false}});

// Revenue by source
new Chart(revSource,{type:'bar',data:{labels:DATA.revenue_by_source.map(d=>d.booking_source),
  datasets:[{data:DATA.revenue_by_source.map(d=>d.pct_of_revenue),backgroundColor:C.good,borderRadius:6}]},
  options:{plugins:{legend:{display:false}},scales:{y:{ticks:{callback:v=>v+'%'}}},maintainAspectRatio:false}});

// Benchmark table
const tb = document.querySelector('#benchTable tbody');
const sorted = [...DATA.cancel_benchmark].sort((a,b)=>b.AUC-a.AUC);
tb.innerHTML = sorted.map((r,i)=>
  `<tr class="${i===0?'best':''}"><td>${r.Model}</td><td class="num">${r.AUC.toFixed(3)}</td>
   <td class="num">${r.Accuracy.toFixed(3)}</td><td class="num">${(r.F1).toFixed(3)}</td></tr>`).join('');
</script>
</body>
</html>
"""


def main():
    data = json.loads((DASH / "dashboard_data.json").read_text())
    html = HTML.replace("__DATA__", json.dumps(data))
    (DASH / "index.html").write_text(html)
    print("Wrote", DASH / "index.html", f"({len(html):,} bytes)")


if __name__ == "__main__":
    main()
