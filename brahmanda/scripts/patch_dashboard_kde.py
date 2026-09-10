"""
Patches redflag_dashboard_v5.html:
  1. drawKDE() no longer fetches /kde/{event_id} and overwrites the live
     forecast_cone numbers from /analyze with the precomputed offline
     file -- that was silently defeating tonight's whole point.
  2. Adds a new "Forecast Cone" chart rendering d.forecast_cone (p5 /
     median / p95 across days 1-15), which is the actual Step 3/4
     deliverable made visible for the first time.

Idempotent -- safe to rerun; skips if already patched.

    python scripts/patch_dashboard_kde.py
"""
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = SCRIPT_DIR / "redflag_dashboard_v5.html"

OLD_DRAWKDE = '''async function drawKDE(d){
  let gx=null, gy=null, src="mixture from analogues";
  if(S.live && S.eventId){
    try{
      const r = await fetch(`${API}/kde/${S.eventId}`,{signal:AbortSignal.timeout(9000)});
      if(r.ok){
        const k = await r.json();
        if(k.x_grid && k.density){ gx=k.x_grid; gy=k.density; src="precomputed KDE v3"; }
        const ps = k.posterior_stats||{};
        d.kde_p5 = ps.kde_p5 ?? ps.p5 ?? d.kde_p5;
        d.kde_median = ps.kde_median ?? ps.median ?? ps.p50 ?? d.kde_median;
        d.kde_p95 = ps.kde_p95 ?? ps.p95 ?? d.kde_p95;
        d.is_bimodal = (ps.kde_is_bimodal===1) || ps.is_bimodal || d.is_bimodal;
        d.tail_weight = ps.kde_tail_weight ?? ps.tail_weight ?? d.tail_weight;
      }
    }catch(e){}
  }
  if(!gx){'''

NEW_DRAWKDE = '''async function drawKDE(d){
  // Live forecast_cone from /analyze (path_kde.py, real three-rank
  // mixture computed from actual analogue dates) is the source of
  // truth. The precomputed /kde/{event_id} endpoint is no longer
  // consulted here -- it was silently overwriting live numbers with a
  // static offline file whenever the selected event happened to have
  // one, which defeated the whole point of computing this live.
  let gx=null, gy=null, src="live mixture (path_kde.py)";
  const cone = d.forecast_cone || {};
  const horizonKey = "15";
  if(cone[horizonKey]){
    const t = cone[horizonKey];
    d.kde_p5 = t.p5 ?? d.kde_p5;
    d.kde_median = t.median ?? d.kde_median;
    d.kde_p95 = t.p95 ?? d.kde_p95;
    d.is_bimodal = t.is_bimodal ?? d.is_bimodal;
    d.tail_weight = t.tail_weight ?? d.tail_weight;
  }
  drawForecastCone(d);
  if(!gx){'''

CONE_CHART_JS = '''

/* ── forecast cone: the evolving posterior across t+1..t+15 ──
   This is the actual Step 3/4 deliverable -- a day-by-day cone built
   from real analogue price paths (path_kde.py), not a single static
   terminal-horizon number. */
function drawForecastCone(d){
  const cone = d.forecast_cone || {};
  const days = Object.keys(cone).map(Number).sort((a,b)=>a-b);
  const valid = days.filter(dd => cone[String(dd)] != null);
  if(!valid.length){
    Plotly.purge("coneChart");
    return;
  }
  const p5s   = valid.map(dd => cone[String(dd)].p5);
  const meds  = valid.map(dd => cone[String(dd)].median);
  const p95s  = valid.map(dd => cone[String(dd)].p95);
  const bimodalDays = valid.filter(dd => cone[String(dd)].is_bimodal);

  const traces = [
    { x:valid, y:p95s, type:"scatter", mode:"lines",
      line:{color:"rgba(76,175,125,0.5)", width:1, dash:"dot"},
      name:"p95", hoverinfo:"skip" },
    { x:valid, y:p5s, type:"scatter", mode:"lines",
      line:{color:"rgba(217,87,87,0.5)", width:1, dash:"dot"},
      fill:"tonexty", fillcolor:"rgba(226,154,60,0.08)",
      name:"p5", hoverinfo:"skip" },
    { x:valid, y:meds, type:"scatter", mode:"lines+markers",
      line:{color:"#E29A3C", width:2.4},
      marker:{size:5, color: valid.map(dd => cone[String(dd)].is_bimodal ? "#9B7FD4" : "#E29A3C")},
      name:"median",
      hovertemplate: "day %{x}<br>median %{y:.2f}%<extra></extra>" },
  ];

  Plotly.newPlot("coneChart", traces, {
    ...PLOT_BASE, height:220,
    margin:{l:44,r:14,t:8,b:30},
    xaxis:{ title:{text:"days from event", font:{size:9}},
            gridcolor:"#1D2836", zeroline:true, zerolinecolor:"#2A3848" },
    yaxis:{ title:{text:"return %", font:{size:9}},
            gridcolor:"#1D2836", zeroline:true, zerolinecolor:"#2A3848" },
    showlegend:false,
  }, PLOT_CFG);

  const bm = document.getElementById("coneMeta");
  if(bm){
    bm.textContent = bimodalDays.length
      ? `bimodal from day ${bimodalDays[0]} onward (${bimodalDays.length}/${valid.length} days)`
      : "unimodal across full window";
  }
}
'''

CONE_PANEL_HTML = '''
    <div class="panel" id="conePanel" style="margin-top:14px">
      <div class="phdr">Forecast Cone -- evolving posterior, t+1 to t+15
        <span class="tag" id="coneMeta"></span></div>
      <div class="pbody">
        <div id="coneChart" style="min-height:0"></div>
      </div>
    </div>
'''


def main():
    if not DASHBOARD_PATH.exists():
        print(f"ERROR: {DASHBOARD_PATH} not found.")
        return

    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    if 'id="coneChart"' in html:
        print("Already patched -- skipping (delete coneChart/coneMeta manually to re-patch).")
        return

    if OLD_DRAWKDE not in html:
        print("ERROR: could not find the expected drawKDE() function text to replace.")
        print("The dashboard file may have been edited since this patch was written --")
        print("apply the change manually: see OLD_DRAWKDE / NEW_DRAWKDE in this script.")
        return

    backup_path = DASHBOARD_PATH.with_suffix(".html.bak2")
    backup_path.write_text(html, encoding="utf-8")

    html = html.replace(OLD_DRAWKDE, NEW_DRAWKDE)
    html = html.replace(
        '<div class="panel" id="postPanel"',
        CONE_PANEL_HTML + '\n    <div class="panel" id="postPanel"'
    )
    html = html.replace("boot();", CONE_CHART_JS + "\n\nboot();")

    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    print(f"Backup saved to {backup_path}")
    print("Patched drawKDE() and added the Forecast Cone panel.")


if __name__ == "__main__":
    main()
