"""
generate_timeline.py
====================
Reads df_topics.csv and generates an interactive HTML timeline.

Layout:
  Top half    : Side-by-side Gantt (LDA left, BERTopic right)
  Bottom half : Side-by-side bubble scatter (same pairing)

Click any bar or bubble → modal with weekly breakdown.

Usage
-----
    python generate_timeline.py \
        --csv  clustered_output/df_topics.csv \
        --output clustered_output/timeline.html \
        --min_events 3
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
import numpy as np

LDA_PALETTE = [
    "#e05c5c","#e07c3a","#e0b83a","#8fd44e","#4ec98f",
    "#4eb8e0","#4e72e0","#a04ee0","#e04eb8","#e04e72",
    "#c0392b","#d35400","#f39c12","#27ae60","#16a085",
    "#2980b9","#8e44ad","#c0392b","#7f8c8d","#2c3e50",
    "#e74c3c","#e67e22","#f1c40f","#2ecc71","#1abc9c",
    "#3498db","#9b59b6","#e91e63","#00bcd4","#ff5722",
]
BERT_PALETTE = [
    "#ff6b6b","#ffa94d","#ffe066","#69db7c","#38d9a9",
    "#74c0fc","#a9e34b","#f783ac","#da77f2","#63e6be",
]

def hex_rgba(h, a=0.75):
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"


def build_topic_data(df, id_col, label_col, palette, min_events,
                     exclude_oversized=True):
    mask = pd.Series([True]*len(df), index=df.index)
    if exclude_oversized and "bert_topic_oversized" in df.columns \
            and id_col == "bert_topic_id":
        mask &= df["bert_topic_oversized"] == 0
    if id_col == "bert_topic_id":
        mask &= df["bert_is_outlier"] == 0
    if id_col == "lda_topic_id" and "lda_topic_singleton" in df.columns:
        mask &= df["lda_topic_singleton"] == 0

    sub = df[mask].copy()
    out = []

    for tid, grp in sub.groupby(id_col):
        if len(grp) < min_events:
            continue
        dates = pd.to_datetime(grp["date"], errors="coerce").dropna().sort_values()
        if dates.empty:
            continue

        label = str(grp[label_col].iloc[0])
        start = dates.min()
        end   = dates.max()
        span  = max((end - start).days, 1)

        weekly = dates.dt.to_period("W").value_counts().sort_index()
        weekly_data = [{"week": str(p.start_time)[:10], "count": int(c)}
                       for p, c in weekly.items()]

        price_col = "price_change_t_minus_7_to_t_plus_7"
        impact = float(grp[price_col].abs().mean()) \
                 if price_col in grp.columns else 0.0

        color = palette[int(tid) % len(palette)]
        out.append({
            "id":        int(tid),
            "label":     label[:55],
            "count":     len(grp),
            "start":     str(start)[:10],
            "end":       str(end)[:10],
            "span_days": span,
            "color":     color,
            "color_a":   hex_rgba(color, 0.8),
            "impact":    round(impact, 2),
            "weekly":    weekly_data,
            "lifespan":  "macro" if span>=30 else "medium" if span>=7 else "micro",
        })

    out.sort(key=lambda x: x["start"])
    return out


def generate_html(lda_topics, bert_topics, out_path):
    lda_js  = json.dumps(lda_topics,  ensure_ascii=False)
    bert_js = json.dumps(bert_topics, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Topic Timeline</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Playfair+Display:wght@900&display=swap');
:root {{
  --bg:#080b10; --surf:#111520; --bdr:#1a2035;
  --txt:#b0bcd4; --muted:#3a4560; --white:#e8eef8;
  --lda:#e05c5c; --bert:#74c0fc;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;overflow:hidden}}
body{{background:var(--bg);color:var(--txt);font-family:'IBM Plex Mono',monospace;font-size:12px;display:flex;flex-direction:column}}

/* ── Header ── */
header{{padding:0.8rem 1.5rem;border-bottom:1px solid var(--bdr);display:flex;align-items:center;gap:1.5rem;flex-shrink:0}}
h1{{font-family:'Playfair Display',serif;font-size:1.25rem;color:var(--white)}}
.leg{{display:flex;gap:1.2rem;font-size:0.65rem;color:var(--muted)}}
.ldot{{display:inline-block;width:7px;height:7px;border-radius:2px;margin-right:4px;vertical-align:middle}}

/* ── Controls ── */
.ctrl{{padding:0.5rem 1.5rem;border-bottom:1px solid var(--bdr);display:flex;gap:0.5rem;align-items:center;flex-shrink:0;font-size:0.65rem}}
.cl{{color:var(--muted);margin-right:0.2rem}}
.btn{{background:var(--surf);border:1px solid var(--bdr);color:var(--txt);padding:0.2rem 0.6rem;border-radius:3px;cursor:pointer;font-family:'IBM Plex Mono',monospace;font-size:0.62rem;transition:all .12s}}
.btn:hover{{border-color:var(--white);color:var(--white)}}
.btn.on{{background:var(--bdr);color:var(--white);border-color:var(--muted)}}
.sep{{width:1px;height:16px;background:var(--bdr);margin:0 0.3rem}}

/* ── Main grid: 2 cols, 2 rows ── */
.grid{{flex:1;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:1px;background:var(--bdr);overflow:hidden}}
.cell{{background:var(--bg);display:flex;flex-direction:column;overflow:hidden}}

/* ── Panel header ── */
.ph{{padding:0.4rem 1rem;border-bottom:1px solid var(--bdr);display:flex;align-items:center;gap:0.5rem;flex-shrink:0;font-size:0.62rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase}}
.pdot{{width:5px;height:5px;border-radius:50%}}
.pc{{color:var(--muted);margin-left:auto;font-weight:400}}

/* ── Gantt ── */
.gantt-scroll{{flex:1;overflow-y:auto;overflow-x:hidden}}
.axis{{display:flex;height:18px;margin-left:170px;padding-right:12px;position:relative;border-bottom:1px solid var(--bdr);flex-shrink:0}}
.atick{{position:absolute;font-size:0.58rem;color:var(--muted);transform:translateX(-50%);white-space:nowrap;top:3px}}
.grow{{display:flex;align-items:center;padding:1px 12px 1px 0;border-bottom:1px solid #0d111a;min-height:28px}}
.grow:hover{{background:rgba(255,255,255,0.02)}}
.glbl{{width:165px;min-width:165px;font-size:0.58rem;color:var(--muted);text-align:right;padding-right:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:default}}
.glbl:hover{{color:var(--txt)}}
.gtrk{{flex:1;position:relative;height:26px}}
.gbar{{position:absolute;top:50%;transform:translateY(-50%);border-radius:3px;cursor:pointer;min-width:3px;display:flex;align-items:center;justify-content:center;font-size:0.55rem;color:rgba(255,255,255,.9);overflow:hidden;white-space:nowrap;padding:0 3px;transition:filter .12s,transform .12s}}
.gbar:hover{{filter:brightness(1.5);transform:translateY(-50%) scaleY(1.3);z-index:5}}
.ggrid{{position:absolute;top:0;bottom:0;width:1px;background:var(--bdr);pointer-events:none}}

/* ── Bubble canvas ── */
.bwrap{{flex:1;position:relative;overflow:hidden}}
canvas{{display:block;width:100%;height:100%}}

/* ── Modal ── */
#ov{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:100;backdrop-filter:blur(4px)}}
#modal{{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--surf);border:1px solid var(--bdr);border-radius:8px;padding:1.6rem;max-width:480px;width:90vw;max-height:72vh;overflow-y:auto;z-index:101}}
#modal h2{{font-family:'Playfair Display',serif;font-size:.95rem;color:var(--white);margin-bottom:.6rem}}
.mtags{{display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.8rem}}
.mtag{{font-size:.6rem;padding:.15rem .5rem;border-radius:3px;border:1px solid var(--bdr);color:var(--muted)}}
.mtag.hi{{color:var(--white)}}
#mbody{{font-size:.68rem;line-height:1.8;color:var(--txt)}}
#mcls{{position:absolute;top:.8rem;right:1rem;background:none;border:none;color:var(--muted);font-size:1.1rem;cursor:pointer}}
#mcls:hover{{color:var(--white)}}
::-webkit-scrollbar{{width:3px;height:3px}}
::-webkit-scrollbar-thumb{{background:var(--bdr);border-radius:2px}}
</style>
</head>
<body>

<header>
  <h1>Topic Timeline</h1>
  <div class="leg">
    <span><span class="ldot" style="background:var(--lda)"></span>LDA word-based</span>
    <span><span class="ldot" style="background:var(--bert)"></span>BERTopic semantic</span>
    <span style="color:var(--muted)">bar width=lifespan · height=events · bubble=weekly volume</span>
  </div>
</header>

<div class="ctrl">
  <span class="cl">filter:</span>
  <button class="btn on" onclick="setFilter('all',this)">all</button>
  <button class="btn" onclick="setFilter('macro',this)">macro 30d+</button>
  <button class="btn" onclick="setFilter('medium',this)">medium 7–30d</button>
  <button class="btn" onclick="setFilter('micro',this)">micro &lt;7d</button>
  <div class="sep"></div>
  <span class="cl">sort:</span>
  <button class="btn on" onclick="setSort('date',this)">date</button>
  <button class="btn" onclick="setSort('size',this)">size</button>
  <button class="btn" onclick="setSort('span',this)">lifespan</button>
</div>

<div class="grid">
  <!-- LDA Gantt (top-left) -->
  <div class="cell" id="c-lda-gantt">
    <div class="ph"><span class="pdot" style="background:var(--lda)"></span>LDA — Gantt<span class="pc" id="lda-gc"></span></div>
    <div class="axis" id="lda-ax"></div>
    <div class="gantt-scroll" id="lda-gs"></div>
  </div>

  <!-- BERTopic Gantt (top-right) -->
  <div class="cell" id="c-bert-gantt">
    <div class="ph"><span class="pdot" style="background:var(--bert)"></span>BERTopic — Gantt<span class="pc" id="bert-gc"></span></div>
    <div class="axis" id="bert-ax"></div>
    <div class="gantt-scroll" id="bert-gs"></div>
  </div>

  <!-- LDA Bubbles (bottom-left) -->
  <div class="cell" id="c-lda-bub">
    <div class="ph"><span class="pdot" style="background:var(--lda)"></span>LDA — Weekly events</div>
    <div class="bwrap"><canvas id="lda-cv"></canvas></div>
  </div>

  <!-- BERTopic Bubbles (bottom-right) -->
  <div class="cell" id="c-bert-bub">
    <div class="ph"><span class="pdot" style="background:var(--bert)"></span>BERTopic — Weekly events</div>
    <div class="bwrap"><canvas id="bert-cv"></canvas></div>
  </div>
</div>

<!-- Modal -->
<div id="ov" onclick="closeM()">
  <div id="modal" onclick="event.stopPropagation()">
    <button id="mcls" onclick="closeM()">×</button>
    <h2 id="mt"></h2>
    <div class="mtags" id="mm"></div>
    <div id="mbody"></div>
  </div>
</div>

<script>
const LDA  = {lda_js};
const BERT = {bert_js};

let filt='all', srt='date';

// ── Date range across both datasets ──────────────────────────────────────────
function range(sets) {{
  const all = sets.flat();
  const mn  = Math.min(...all.map(d=>+new Date(d.start)));
  const mx  = Math.max(...all.map(d=>+new Date(d.end)));
  return {{mn, mx, span: mx-mn||1}};
}}
const R = range([LDA, BERT]);

function xpct(ms) {{ return (ms - R.mn) / R.span * 100; }}

// ── Filter + sort ─────────────────────────────────────────────────────────────
function applyFS(data) {{
  let d = filt==='all' ? [...data] : data.filter(x=>x.lifespan===filt);
  if (srt==='date') d.sort((a,b)=>a.start.localeCompare(b.start));
  if (srt==='size') d.sort((a,b)=>b.count-a.count);
  if (srt==='span') d.sort((a,b)=>b.span_days-a.span_days);
  return d;
}}

// ── Axis ──────────────────────────────────────────────────────────────────────
function buildAxis(id) {{
  const el=document.getElementById(id); el.innerHTML='';
  for(let i=0;i<=6;i++) {{
    const ms=R.mn+R.span*i/6;
    const t=document.createElement('div');
    t.className='atick';
    t.style.left=(i/6*100)+'%';
    t.textContent=new Date(ms).toLocaleDateString('en-GB',{{month:'short',year:'2-digit'}});
    el.appendChild(t);
  }}
}}

// ── Gantt ─────────────────────────────────────────────────────────────────────
function buildGantt(scrollId, countId, data) {{
  const wrap=document.getElementById(scrollId);
  const cnt =document.getElementById(countId);
  wrap.innerHTML='';
  const items=applyFS(data);
  if(cnt) cnt.textContent=items.length+' topics';
  const mx=Math.max(...items.map(d=>d.count));

  // grid lines
  const gw=document.createElement('div');
  gw.style.cssText='position:relative;margin-left:170px;pointer-events:none;height:0';
  for(let i=0;i<=6;i++) {{
    const gl=document.createElement('div');
    gl.className='ggrid';
    gl.style.left=(i/6*100)+'%';
    gl.style.height=(items.length*30)+'px';
    gw.appendChild(gl);
  }}
  wrap.appendChild(gw);

  items.forEach(d => {{
    const row=document.createElement('div'); row.className='grow';
    const lbl=document.createElement('div'); lbl.className='glbl';
    lbl.textContent=d.label; lbl.title=d.label;

    const trk=document.createElement('div'); trk.className='gtrk';

    const sMs=+new Date(d.start), eMs=+new Date(d.end);
    const left=xpct(sMs).toFixed(3);
    const wid =Math.max(((eMs-sMs)/R.span*100),0.5).toFixed(3);
    const h   =8+(d.count/mx)*18;

    const bar=document.createElement('div'); bar.className='gbar';
    bar.style.cssText=`left:${{left}}%;width:${{wid}}%;height:${{h}}px;
      background:${{d.color_a}};border:1px solid ${{d.color}};
      box-shadow:0 0 8px ${{d.color}}33;`;
    if(parseFloat(wid)>3) bar.textContent=d.count;
    bar.onclick=()=>openM(d);

    trk.appendChild(bar);
    row.appendChild(lbl); row.appendChild(trk);
    wrap.appendChild(row);
  }});
}}

// ── Bubble canvas ─────────────────────────────────────────────────────────────
function buildBubble(canvasId, data) {{
  const cv  = document.getElementById(canvasId);
  const wrap = cv.parentElement;
  const W = wrap.clientWidth  || 600;
  const H = wrap.clientHeight || 300;
  cv.width  = W * devicePixelRatio;
  cv.height = H * devicePixelRatio;
  cv.style.width  = W+'px';
  cv.style.height = H+'px';
  const ctx = cv.getContext('2d');
  ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.clearRect(0,0,W,H);

  const items = applyFS(data);
  if(!items.length) return;

  const PL=175, PR=15, PT=12, PB=24;
  const CW=W-PL-PR, CH=H-PT-PB;
  const yStep = CH / (items.length+1);

  const allW = items.flatMap(d=>d.weekly.map(w=>w.count));
  const maxW = Math.max(...allW,1);

  // grid lines
  ctx.strokeStyle='#1a2035'; ctx.lineWidth=1;
  for(let i=0;i<=6;i++) {{
    const x=PL+CW*i/6;
    ctx.beginPath(); ctx.moveTo(x,PT); ctx.lineTo(x,PT+CH); ctx.stroke();
  }}

  items.forEach((d,i) => {{
    const y = PT+(i+1)*yStep;

    // dashed row line
    ctx.strokeStyle='#1a2035'; ctx.lineWidth=1; ctx.setLineDash([2,5]);
    ctx.beginPath(); ctx.moveTo(PL,y); ctx.lineTo(PL+CW,y); ctx.stroke();
    ctx.setLineDash([]);

    // label
    ctx.fillStyle='#3a4560'; ctx.font='9.5px IBM Plex Mono';
    ctx.textAlign='right';
    ctx.fillText(d.label.slice(0,22), PL-6, y+3.5);

    // bubbles
    d.weekly.forEach(w => {{
      const wMs = +new Date(w.week);
      const x   = PL + (wMs-R.mn)/R.span*CW;
      if(x<PL-5||x>PL+CW+5) return;
      const r = 3+(w.count/maxW)*15;

      // glow
      const grd=ctx.createRadialGradient(x,y,0,x,y,r*1.8);
      grd.addColorStop(0,d.color+'99'); grd.addColorStop(1,d.color+'00');
      ctx.beginPath(); ctx.arc(x,y,r*1.8,0,Math.PI*2);
      ctx.fillStyle=grd; ctx.fill();

      // bubble
      ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
      ctx.fillStyle=d.color_a; ctx.strokeStyle=d.color; ctx.lineWidth=1;
      ctx.fill(); ctx.stroke();

      if(r>8){{
        ctx.fillStyle='rgba(255,255,255,.9)';
        ctx.font='8px IBM Plex Mono'; ctx.textAlign='center';
        ctx.fillText(w.count,x,y+3);
      }}
    }});
  }});

  // x-axis labels
  ctx.fillStyle='#3a4560'; ctx.font='9px IBM Plex Mono'; ctx.textAlign='center';
  for(let i=0;i<=6;i++) {{
    const ms=R.mn+R.span*i/6;
    ctx.fillText(new Date(ms).toLocaleDateString('en-GB',{{month:'short',year:'2-digit'}}),
                 PL+CW*i/6, H-6);
  }}

  // click handler — rebuild each render so remove old listener first
  cv._clickHandler && cv.removeEventListener('click', cv._clickHandler);
  cv._clickHandler = (e) => {{
    const rect=cv.getBoundingClientRect();
    const mx=e.clientX-rect.left, my=e.clientY-rect.top;
    items.forEach((d,i) => {{
      const y=PT+(i+1)*yStep;
      d.weekly.forEach(w => {{
        const wMs=+new Date(w.week);
        const x=PL+(wMs-R.mn)/R.span*CW;
        const r=3+(w.count/maxW)*15;
        if(Math.hypot(mx-x,my-y)<r+4) openM(d,w);
      }});
    }});
  }};
  cv.addEventListener('click', cv._clickHandler);
}}

// ── Modal ─────────────────────────────────────────────────────────────────────
function fmtD(s){{return new Date(s).toLocaleDateString('en-GB',{{day:'2-digit',month:'short',year:'numeric'}})}}

function openM(d,week) {{
  document.getElementById('mt').textContent=d.label;
  const mm=document.getElementById('mm');
  mm.innerHTML=`
    <span class="mtag hi" style="border-color:${{d.color}};color:${{d.color}}">${{d.lifespan}}</span>
    <span class="mtag">${{fmtD(d.start)}} → ${{fmtD(d.end)}}</span>
    <span class="mtag">${{d.span_days}}d</span>
    <span class="mtag">${{d.count}} events</span>
    ${{d.impact>0?`<span class="mtag">±${{d.impact.toFixed(1)}}% price</span>`:''}}
  `;
  const mb=document.getElementById('mbody');
  if(week){{
    mb.innerHTML=`<p style="color:var(--muted);margin-bottom:.4rem">Week of ${{fmtD(week.week)}}</p>
    <p><strong style="color:var(--white);font-size:1.1rem">${{week.count}}</strong> events this week</p>`;
  }} else {{
    mb.innerHTML='<p style="color:var(--muted);margin-bottom:.5rem;font-size:.6rem">WEEKLY BREAKDOWN</p>'
      + d.weekly.map(w=>`<div style="display:flex;gap:.8rem;margin-bottom:.25rem">
          <span style="color:var(--muted);width:80px;flex-shrink:0">${{fmtD(w.week)}}</span>
          <span style="color:var(--white)">${{'█'.repeat(Math.min(w.count*2,28))}} ${{w.count}}</span>
        </div>`).join('');
  }}
  document.getElementById('ov').style.display='block';
}}
function closeM(){{document.getElementById('ov').style.display='none'}}
document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeM()}});

// ── Controls ──────────────────────────────────────────────────────────────────
function setFilter(f,btn) {{
  filt=f;
  document.querySelectorAll('.ctrl .btn').slice(0,4).forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  render();
}}
function setSort(s,btn) {{
  srt=s;
  document.querySelectorAll('.ctrl .btn').forEach(b=>{{
    if(['date','size','lifespan'].includes(b.textContent)) b.classList.remove('on');
  }});
  btn.classList.add('on');
  render();
}}

// ── Render ────────────────────────────────────────────────────────────────────
function render() {{
  buildAxis('lda-ax');
  buildAxis('bert-ax');
  buildGantt('lda-gs',  'lda-gc',  LDA);
  buildGantt('bert-gs', 'bert-gc', BERT);
  buildBubble('lda-cv',  LDA);
  buildBubble('bert-cv', BERT);
}}

render();
window.addEventListener('resize', render);
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"[html] written -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv",    required=True)
    ap.add_argument("--output", default="clustered_output/timeline.html")
    ap.add_argument("--min_events", type=int, default=3)
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        sys.exit(f"ERROR: not found -- {csv_path}")

    print(f"[load] {csv_path.name} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[load] {len(df)} rows")

    required = {"date","lda_topic_id","lda_topic_label","bert_topic_id","bert_topic_label"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: missing columns {missing} — run theme_cluster_pipeline.py first")

    print("[build] LDA topics ...")
    lda  = build_topic_data(df,"lda_topic_id", "lda_topic_label", LDA_PALETTE,  args.min_events)
    print("[build] BERTopic topics ...")
    bert = build_topic_data(df,"bert_topic_id","bert_topic_label", BERT_PALETTE, args.min_events)

    print(f"[build] LDA={len(lda)} topics  BERTopic={len(bert)} topics")
    generate_html(lda, bert, out_path)

    print(f"\n{'='*50}")
    print(f"  LDA topics    : {len(lda)}")
    print(f"  BERTopic topics: {len(bert)}")
    print(f"  output -> {out_path}")
    print(f"  open timeline.html in your browser")
    print(f"{'='*50}")

if __name__=="__main__":
    main()