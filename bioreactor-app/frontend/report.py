#!/usr/bin/env python3
"""Generate a self-contained HTML report by running simulations through the API."""
import json, requests

API = "http://localhost:8080/api"

# Run simulations through the API (simpler, already working)
print("Running simulations through API...")
sims = {}
organisms_to_run = [
    ("ecoli_k12", "E. coli K-12", 37.0, 7.0),
    ("scerevisiae", "S. cerevisiae", 30.0, 5.5),
    ("bsubtilis", "B. subtilis", 37.0, 7.0),
    ("ecoli_bl21", "E. coli BL21", 37.0, 7.0),
]

for org_id, name, temp, ph in organisms_to_run:
    print(f"  {name}...", end=" ", flush=True)
    r = requests.post(f"{API}/simulate", json={
        "organism_id": org_id, "X_v": 1.0, "S": 10.0, "V": 10.0,
        "temperature": temp, "pH": ph, "t_end": 30.0
    }, timeout=60)
    if r.status_code == 200:
        raw = json.loads(r.json().get("raw", "{}"))
        sims[name] = raw
        print(f"final VCD={raw.get('final_X_v',0):.2f}")
    else:
        print(f"FAILED: {r.status_code}")

# Now generate synthetic smooth time series using a helper
# (We'll plot final values as bars and tables since we don't have time series from API)
print(f"\nSimulated {len(sims)} organisms.")

# Build HTML with summary data
orgs = [
    {"id": "ecoli_k12", "name": "Escherichia coli K-12 MG1655", "type": "bacteria", "t_opt": 37},
    {"id": "ecoli_bl21", "name": "Escherichia coli BL21(DE3)", "type": "bacteria", "t_opt": 37},
    {"id": "bsubtilis", "name": "Bacillus subtilis 168", "type": "bacteria", "t_opt": 37},
    {"id": "scerevisiae", "name": "Saccharomyces cerevisiae S288C", "type": "yeast", "t_opt": 30},
    {"id": "ppastoris", "name": "Pichia pastoris GS115", "type": "yeast", "t_opt": 30},
    {"id": "kpneumoniae", "name": "Klebsiella pneumoniae MGH78578", "type": "bacteria", "t_opt": 37},
    {"id": "paeruginosa", "name": "Pseudomonas aeruginosa PAO1", "type": "bacteria", "t_opt": 37},
    {"id": "aoryzae", "name": "Aspergillus oryzae RIB40", "type": "fungus", "t_opt": 30},
    {"id": "yarrowia", "name": "Yarrowia lipolytica W29", "type": "yeast", "t_opt": 28},
]

# Build the bar chart data
names = list(sims.keys())
vcd_vals = [sims[n].get('final_X_v', 0) for n in names]
s_vals = [sims[n].get('final_S', 0) for n in names]
b_vals = [sims[n].get('final_B', 0) for n in names]

chart_data = {
    "organisms": names,
    "vcd": vcd_vals,
    "substrate": s_vals,
    "byproduct": b_vals,
}

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bioreactor Simulator — Results</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; 
         max-width: 1100px; margin: 0 auto; padding: 24px; background: #0d1117; color: #e6edf3; }}
  h1 {{ color: #2ea043; margin-bottom: 4px; font-size: 28px; }}
  h2 {{ color: #7ee787; margin: 32px 0 12px; font-size: 20px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  .subtitle {{ color: #8b949e; margin-bottom: 24px; font-size: 14px; }}
  
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 24px 0; }}
  .card {{ background: #161b22; border-radius: 12px; padding: 20px 24px; flex: 1; min-width: 150px; 
           border: 1px solid #30363d; border-left: 3px solid #2ea043; }}
  .card .value {{ font-size: 28px; font-weight: bold; color: #2ea043; }}
  .card .label {{ font-size: 13px; color: #8b949e; margin-top: 4px; }}
  
  .sim-results {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin: 20px 0; }}
  .sim-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; }}
  .sim-card .org-name {{ font-weight: 600; font-size: 16px; color: #7ee787; margin-bottom: 12px; }}
  .sim-card .metric {{ display: flex; justify-content: space-between; padding: 6px 0; }}
  .sim-card .metric .label {{ color: #8b949e; font-size: 13px; }}
  .sim-card .metric .val {{ font-weight: 600; font-size: 14px; }}
  .sim-card .metric .val.green {{ color: #2ea043; }}
  .sim-card .metric .val.blue {{ color: #58a6ff; }}
  .sim-card .metric .val.orange {{ color: #d29922; }}
  
  .bar-chart {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 24px; margin: 20px 0; }}
  .bar-row {{ margin: 10px 0; }}
  .bar-label {{ display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 4px; }}
  .bar-label .name {{ color: #e6edf3; }}
  .bar-label .val {{ color: #2ea043; font-weight: 600; }}
  .bar-track {{ background: #21262d; border-radius: 4px; height: 20px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
  
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #30363d; }}
  th {{ color: #2ea043; font-weight: 600; background: #1c2128; font-size: 13px; text-transform: uppercase; }}
  td {{ font-size: 14px; }}
  tr:hover td {{ background: #1c2128; }}
  
  .badge {{ padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500; }}
  .badge-green {{ background: #1b3b1e; color: #3fb950; }}
  .badge-purple {{ background: #2d1b40; color: #d2a8ff; }}
  .badge-red {{ background: #3d1f1a; color: #f0883e; }}
  
  pre {{ background: #161b22; border: 1px solid #30363d; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 13px; color: #7ee787; }}
  code {{ color: #d29922; }}
  
  .pulse {{ animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.5}} }}
  
  footer {{ margin-top: 50px; padding: 20px 0; text-align: center; color: #484f58; font-size: 12px; border-top: 1px solid #30363d; }}
</style>
</head>
<body>

<h1>🧬 Bioreactor Process Simulator</h1>
<p class="subtitle">Mechanistic ODE-based microbial growth simulation — 9 organisms · 5 growth models · 418K experimental data points</p>

<div class="cards">
  <div class="card"><div class="value">9</div><div class="label">Organism strains</div></div>
  <div class="card"><div class="value">9</div><div class="label">ODE state variables</div></div>
  <div class="card"><div class="value">5</div><div class="label">Growth models</div></div>
  <div class="card"><div class="value">418K</div><div class="label">Data points (DTB+BRPMM)</div></div>
</div>

<h2>📊 Simulation Results</h2>
<p class="subtitle">4 organisms simulated: 10 g/L glucose, batch mode, 30h at optimal temperature/pH</p>

<div class="sim-results">
"""

for name in ["E. coli K-12", "S. cerevisiae", "B. subtilis", "E. coli BL21"]:
    if name in sims:
        d = sims[name]
        xv = d.get('final_X_v', 0)
        s = d.get('final_S', 0)
        b = d.get('final_B', 0)
        via = d.get('final_viability', 1.0)
        pts = d.get('time_points', 0)
        html += f"""  <div class="sim-card">
    <div class="org-name">{name}</div>
    <div class="metric"><span class="label">Final VCD</span><span class="val green">{xv:.2f} g/L</span></div>
    <div class="metric"><span class="label">Substrate remaining</span><span class="val blue">{s:.1f} g/L</span></div>
    <div class="metric"><span class="label">Byproduct (acetate)</span><span class="val orange">{b:.2f} g/L</span></div>
    <div class="metric"><span class="label">Viability</span><span class="val">{via*100:.0f}%</span></div>
    <div class="metric"><span class="label">Time points</span><span class="val">{pts}</span></div>
  </div>
"""

html += """</div>

<div class="bar-chart">
  <h3 style="margin-bottom:16px; color:#7ee787;">Final Viable Cell Density [g/L]</h3>
"""
max_vcd = max(vcd_vals) if vcd_vals else 1
for i, name in enumerate(names):
    pct = (vcd_vals[i] / max_vcd * 100) if max_vcd > 0 else 0
    color = ['#2ea043', '#a371f7', '#58a6ff', '#d29922'][i % 4]
    html += f"""  <div class="bar-row">
    <div class="bar-label"><span class="name">{name}</span><span class="val">{vcd_vals[i]:.2f} g/L</span></div>
    <div class="bar-track"><div class="bar-fill" style="width:{pct}%; background:{color};"></div></div>
  </div>
"""

html += """
</div>

<h2>🔬 All Available Organisms</h2>
<table>
<tr><th>ID</th><th>Name</th><th>Type</th><th>Optimal T</th></tr>
"""
for o in orgs:
    if o['type'] == 'yeast':
        cls = 'badge-purple'
        tname = 'Yeast'
    elif o['type'] == 'fungus':
        cls = 'badge-red'
        tname = 'Fungus'
    else:
        cls = 'badge-green'
        tname = 'Bacteria'
    html += f"<tr><td><code>{o['id']}</code></td><td>{o['name']}</td><td><span class='badge {cls}'>{tname}</span></td><td>{o['t_opt']}°C</td></tr>\n"

html += """
</table>

<h2>📈 Fitted Parameters</h2>
<p class="subtitle">E. coli K-12 parameters calibrated against 9 DTB experimental batches using feed-driven ODE fitting</p>
<table>
<tr><th>Parameter</th><th>Description</th><th>Default</th><th>Fitted</th><th>CV</th></tr>
<tr><td>&mu;<sub>max</sub></td><td>Max specific growth rate</td><td>0.70 h<sup>-1</sup></td><td><strong>0.725 h<sup>-1</sup></strong></td><td>±132%</td></tr>
<tr><td>K<sub>s</sub></td><td>Substrate affinity constant</td><td>0.020 g/L</td><td><strong>0.010 g/L</strong></td><td>±281%</td></tr>
<tr><td>Y<sub>X/S</sub></td><td>Biomass yield on substrate</td><td>0.50 g/g</td><td><strong>0.550 g/g</strong></td><td><span style="color:#3fb950">±9.5%</span></td></tr>
<tr><td>&alpha;</td><td>Growth-associated acetate</td><td>0.0001</td><td><strong>0.043</strong></td><td>±36%</td></tr>
<tr><td>&beta;</td><td>Non-growth acetate rate</td><td>0.02</td><td><strong>0.001</strong></td><td>±255%</td></tr>
<tr><td>m<sub>s</sub></td><td>Maintenance coefficient</td><td>0.025</td><td><strong>0.032</strong></td><td>±52%</td></tr>
</table>

<div class="cards">
  <div class="card">
    <div class="value">0.889</div>
    <div class="label">Cross-validation R² (VCD, 9 batches)</div>
  </div>
  <div class="card">
    <div class="value">0.486</div>
    <div class="label">FBA biomass yield (COBRApy, g/g)</div>
  </div>
  <div class="card">
    <div class="value">95</div>
    <div class="label">FBA reactions (E. coli core model)</div>
  </div>
  <div class="card">
    <div class="value">12</div>
    <div class="label">Holdout batches (to_predict)</div>
  </div>
</div>

<h2>🏗️ System Architecture</h2>
<pre><code>
  Browser ──► Streamlit UI (:8501) ──► Java Spring Boot API (:8080)
                                           │
                                           ├── Python ODE Engine (scipy.integrate.solve_ivp)
                                           ├── COBRApy FBA (E. coli core model, 95 reactions)
                                           └── PostgreSQL (:5432) — organisms, experiments, sims
</code></pre>

<h2>🖥️ Run from CLI</h2>
<pre><code># Health check
curl http://localhost:8080/api/health

# List organisms
curl http://localhost:8080/api/organisms | python3 -m json.tool

# Run a simulation
curl -X POST http://localhost:8080/api/simulate \\
  -H "Content-Type: application/json" \\
  -d '{ \\
    "organism_id": "ecoli_k12", \\
    "X_v": 1.0, \\
    "S": 5.0, \\
    "V": 10.0, \\
    "temperature": 37, \\
    "pH": 7, \\
    "t_end": 24 \\
  }'

# Start Streamlit UI
streamlit run frontend/streamlit_app.py --server.port 8501
</code></pre>

<footer>
  <strong>SDSC-STEM4Health Group 10</strong> &middot; Bioreactor Process Application v0.1.0 &middot; 9 organisms &middot; 418,023 data points &middot; Fitted R² = 0.889
</footer>

</body>
</html>"""

outpath = "/home/renku/work/bioreactor_report.html"
with open(outpath, "w") as f:
    f.write(html)

print(f"\n✅ Report: {outpath} ({len(html):,} bytes)")
print("Open in VS Code: Cmd+Shift+P → Simple Browser → http://localhost:8501")
print(f"Or open file directly: file://{outpath}")
