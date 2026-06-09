/*
 * Stock Forecast UI — React + TypeScript (no build step).
 * Transpiled in-browser by Babel standalone (see index.html).
 *
 * Data source: the existing FastAPI service (same origin).
 *   GET /models                 -> populate the symbol dropdown
 *   GET /recommendation/{symbol} -> next-day 5-class signal + probabilities + sentiment
 *
 * The "Share Price Forecast" cone is a MODEL-IMPLIED projection derived from the
 * signal probabilities (the model itself only predicts the next day). It is a
 * visualization, not a price target — see the disclaimer banner.
 */

const { useState, useEffect, useMemo, useCallback, useRef } = React;

// ── Types ───────────────────────────────────────────────
interface Probabilities {
  strong_sell: number; sell: number; hold: number; buy: number; strong_buy: number;
}
interface Sentiment {
  symbol: string; score: number; label: string; confidence: number; news_count: number;
}
interface Recommendation {
  symbol: string;
  signal: string;
  signal_class: number;
  confidence: number;
  probabilities: Probabilities;
  model_used: string;
  sentiment: Sentiment | null;
  top_features: Record<string, number>;
  explanation: string;
  timestamp: string;
  data_as_of: string;
}

type Horizon = "day" | "month" | "year";

// ── Constants ───────────────────────────────────────────
// Representative daily returns per class = midpoints of the config label bands.
const CLASS_RETURN: Record<string, number> = {
  strong_sell: -0.025, sell: -0.0125, hold: 0.0, buy: 0.0125, strong_buy: 0.025,
};
// Trading days per horizon (≈ calendar Year ~252 trading days).
const HORIZON_DAYS: Record<Horizon, number> = { day: 7, month: 21, year: 252 };
const HORIZON_LABEL: Record<Horizon, string> = { day: "7 days", month: "1 month", year: "1 year" };

const CLASS_ORDER = ["strong_buy", "buy", "hold", "sell", "strong_sell"];
const CLASS_TEXT: Record<string, string> = {
  strong_buy: "Strong Buy", buy: "Buy", hold: "Hold", sell: "Sell", strong_sell: "Strong Sell",
};
const CLASS_COLOR: Record<string, string> = {
  strong_buy: "#2e7d32", buy: "#66bb6a", hold: "#9aa3b2", sell: "#ef9a9a", strong_sell: "#c62828",
};

const FALLBACK_SYMBOLS = ["INFY.NS", "IRCTC.NS", "TATAPOWER.NS", "TMPV.NS", "TRIVENI.NS", "VBL.NS"];

// ── Helpers ─────────────────────────────────────────────
function pct(x: number): string { return (x * 100).toFixed(1) + "%"; }
function signed(x: number): string { return (x >= 0 ? "+" : "") + (x * 100).toFixed(1) + "%"; }

// Expected per-day return (mu) and std (sigma) from the probability distribution.
function muSigma(p: Probabilities): { mu: number; sigma: number } {
  let mu = 0;
  const keys = ["strong_sell", "sell", "hold", "buy", "strong_buy"];
  for (const k of keys) mu += (p as any)[k] * CLASS_RETURN[k];
  let varSum = 0;
  for (const k of keys) {
    const d = CLASS_RETURN[k] - mu;
    varSum += (p as any)[k] * d * d;
  }
  return { mu, sigma: Math.sqrt(varSum) };
}

// Build the indexed (start=100) High/Mean/Low projection over `days`.
interface ConePoint { t: number; mean: number; high: number; low: number; }
function buildCone(p: Probabilities, days: number): ConePoint[] {
  const { mu, sigma } = muSigma(p);
  const z = 1.645; // ~90% band
  const steps = 40;
  const pts: ConePoint[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = (days * i) / steps;
    const drift = mu * t;
    const spread = z * sigma * Math.sqrt(t);
    pts.push({
      t,
      mean: 100 * Math.exp(drift),
      high: 100 * Math.exp(drift + spread),
      low: 100 * Math.exp(drift - spread),
    });
  }
  return pts;
}

// ── API ─────────────────────────────────────────────────
function fetchJSON(url: string): Promise<any> {
  return fetch(url).then(function (r) {
    if (!r.ok) {
      return r.text().then(function (t) {
        let detail = t;
        try { detail = JSON.parse(t).detail || t; } catch (e) { /* keep raw */ }
        throw new Error("HTTP " + r.status + ": " + detail);
      });
    }
    return r.json();
  });
}

// ── Cone chart (hand-rolled SVG) ─────────────────────────
function ConeChart(props: { points: ConePoint[] }) {
  const pts = props.points;
  const W = 360, H = 210, PAD_L = 8, PAD_R = 8, PAD_T = 14, PAD_B = 22;
  const plotW = W - PAD_L - PAD_R, plotH = H - PAD_T - PAD_B;

  let lo = Infinity, hi = -Infinity;
  for (const p of pts) { lo = Math.min(lo, p.low); hi = Math.max(hi, p.high); }
  const padY = (hi - lo) * 0.08 || 1;
  lo -= padY; hi += padY;
  const maxT = pts[pts.length - 1].t || 1;

  const xOf = function (t: number): number { return PAD_L + (t / maxT) * plotW; };
  const yOf = function (v: number): number { return PAD_T + (1 - (v - lo) / (hi - lo)) * plotH; };

  const line = function (sel: (p: ConePoint) => number): string {
    return pts.map(function (p, i) { return (i === 0 ? "M" : "L") + xOf(p.t).toFixed(1) + " " + yOf(sel(p)).toFixed(1); }).join(" ");
  };
  const meanLine = line(function (p) { return p.mean; });
  const highLine = line(function (p) { return p.high; });
  const lowLine = line(function (p) { return p.low; });

  // Filled areas: high→mean (green), mean→low (red)
  const upperArea =
    "M" + pts.map(function (p) { return xOf(p.t).toFixed(1) + " " + yOf(p.high).toFixed(1); }).join(" L") +
    " L" + pts.slice().reverse().map(function (p) { return xOf(p.t).toFixed(1) + " " + yOf(p.mean).toFixed(1); }).join(" L") + " Z";
  const lowerArea =
    "M" + pts.map(function (p) { return xOf(p.t).toFixed(1) + " " + yOf(p.mean).toFixed(1); }).join(" L") +
    " L" + pts.slice().reverse().map(function (p) { return xOf(p.t).toFixed(1) + " " + yOf(p.low).toFixed(1); }).join(" L") + " Z";

  const baselineY = yOf(100);

  return (
    <svg className="cone-chart" viewBox={"0 0 " + W + " " + H} preserveAspectRatio="xMidYMid meet">
      {/* baseline at indexed 100 */}
      <line x1={PAD_L} y1={baselineY} x2={W - PAD_R} y2={baselineY} stroke="#e3e8ef" strokeWidth="1" strokeDasharray="3 3" />
      <path d={upperArea} fill="#2e9e5b" opacity="0.16" />
      <path d={lowerArea} fill="#d8453a" opacity="0.14" />
      <path d={highLine} fill="none" stroke="#2e9e5b" strokeWidth="1.5" />
      <path d={lowLine} fill="none" stroke="#d8453a" strokeWidth="1.5" />
      <path d={meanLine} fill="none" stroke="#8a94a6" strokeWidth="1.5" strokeDasharray="4 3" />
      {/* end markers */}
      <circle cx={xOf(maxT)} cy={yOf(pts[pts.length - 1].high)} r="3" fill="#2e9e5b" />
      <circle cx={xOf(maxT)} cy={yOf(pts[pts.length - 1].low)} r="3" fill="#d8453a" />
      <text x={PAD_L} y={H - 6} fontSize="10" fill="#8a94a6">today</text>
      <text x={W - PAD_R} y={H - 6} fontSize="10" fill="#8a94a6" textAnchor="end">horizon</text>
    </svg>
  );
}

// ── Signal donut (hand-rolled SVG) ───────────────────────
function SignalDonut(props: { rec: Recommendation }) {
  const rec = props.rec;
  const r = 58, cx = 75, cy = 75, sw = 18;
  const C = 2 * Math.PI * r;
  let offset = 0;
  const segs = CLASS_ORDER.map(function (k) {
    const frac = (rec.probabilities as any)[k] as number;
    const seg = (
      <circle
        key={k}
        cx={cx} cy={cy} r={r}
        fill="none"
        stroke={CLASS_COLOR[k]}
        strokeWidth={sw}
        strokeDasharray={(frac * C).toFixed(2) + " " + ((1 - frac) * C).toFixed(2)}
        strokeDashoffset={(-offset * C).toFixed(2)}
        transform={"rotate(-90 " + cx + " " + cy + ")"}
      />
    );
    offset += frac;
    return seg;
  });

  return (
    <div className="donut">
      <svg viewBox="0 0 150 150" width="150" height="150">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#eef1f6" strokeWidth={sw} />
        {segs}
      </svg>
      <div className="center">
        <div className="sig" style={{ color: CLASS_COLOR[keyForClass(rec.signal_class)] }}>{rec.signal}</div>
        <div className="conf">{pct(rec.confidence)} conf.</div>
      </div>
    </div>
  );
}

function keyForClass(c: number): string {
  // signal_class: 0 strong_sell .. 4 strong_buy
  const map: Record<number, string> = { 0: "strong_sell", 1: "sell", 2: "hold", 3: "buy", 4: "strong_buy" };
  return map[c] || "hold";
}

// ── Advanced chart (TradingView Lightweight Charts on OUR data) ──────
interface Candle { time: string; open: number; high: number; low: number; close: number; volume: number; }
type ChartRange = "1m" | "3m" | "6m" | "1y" | "max";
const RANGE_DAYS: Record<ChartRange, number> = { "1m": 21, "3m": 63, "6m": 126, "1y": 252, "max": 100000 };

function AdvancedChart(props: { symbol: string }) {
  const elRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<any>(null);
  const candleRef = useRef<any>(null);
  const volRef = useRef<any>(null);
  const dataRef = useRef<Candle[]>([]);
  const [range, setRange] = useState<ChartRange>("6m");
  const [status, setStatus] = useState<string>("loading");

  // Create the chart once.
  useEffect(function () {
    const el = elRef.current;
    if (!el || !(window as any).LightweightCharts) { setStatus("nolib"); return; }
    const LC = (window as any).LightweightCharts;
    const chart = LC.createChart(el, {
      autoSize: true,
      layout: { background: { color: "#131722" }, textColor: "#b2b9c9" },
      grid: { vertLines: { color: "#1e222d" }, horzLines: { color: "#1e222d" } },
      timeScale: { borderColor: "#2a2e39", rightOffset: 4 },
      rightPriceScale: { borderColor: "#2a2e39" },
      crosshair: { mode: LC.CrosshairMode ? LC.CrosshairMode.Normal : 0 },
    });
    const candle = chart.addCandlestickSeries({
      upColor: "#26a69a", downColor: "#ef5350",
      wickUpColor: "#26a69a", wickDownColor: "#ef5350", borderVisible: false,
    });
    const vol = chart.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "" });
    vol.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    chartRef.current = chart; candleRef.current = candle; volRef.current = vol;
    return function () { chart.remove(); chartRef.current = null; };
  }, []);

  // Re-fetch when the symbol changes.
  useEffect(function () {
    let alive = true;
    setStatus("loading");
    fetchJSON("/history/" + encodeURIComponent(props.symbol))
      .then(function (d: any) {
        if (!alive) return;
        dataRef.current = d.candles || [];
        setStatus(dataRef.current.length ? "ok" : "empty");
      })
      .catch(function () { if (alive) setStatus("error"); });
    return function () { alive = false; };
  }, [props.symbol]);

  // Apply the visible slice whenever range or data changes.
  useEffect(function () {
    const all = dataRef.current;
    if (status !== "ok" || !all.length || !candleRef.current) return;
    const n = RANGE_DAYS[range];
    const slice = all.slice(Math.max(0, all.length - n));
    candleRef.current.setData(slice.map(function (c) {
      return { time: c.time, open: c.open, high: c.high, low: c.low, close: c.close };
    }));
    volRef.current.setData(slice.map(function (c) {
      return { time: c.time, value: c.volume, color: c.close >= c.open ? "rgba(38,166,154,0.5)" : "rgba(239,83,80,0.5)" };
    }));
    if (chartRef.current) chartRef.current.timeScale().fitContent();
  }, [range, status]);

  const last = dataRef.current.length ? dataRef.current[dataRef.current.length - 1] : null;

  return (
    <div>
      <div className="chart-range">
        {(["1m", "3m", "6m", "1y", "max"] as ChartRange[]).map(function (r) {
          return <button key={r} className={range === r ? "active" : ""} onClick={function () { setRange(r); }}>{r.toUpperCase()}</button>;
        })}
        <span className="chart-status">
          {status === "loading" ? "loading…" :
            status === "error" ? "history unavailable" :
              status === "empty" ? "no data" :
                status === "nolib" ? "chart library failed to load" :
                  last ? ("Last close ₹" + last.close.toFixed(2) + " · " + last.time) : ""}
        </span>
      </div>
      <div className="tv-host" ref={elRef}></div>
    </div>
  );
}

// ── App ──────────────────────────────────────────────────
function App() {
  const [symbols, setSymbols] = useState<string[]>(FALLBACK_SYMBOLS);
  const [symbol, setSymbol] = useState<string>(FALLBACK_SYMBOLS[0]);
  const [horizon, setHorizon] = useState<Horizon>("month");
  const [rec, setRec] = useState<Recommendation | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  // Load available symbols once.
  useEffect(function () {
    fetchJSON("/models")
      .then(function (d) {
        const uniq: string[] = [];
        (d.models || []).forEach(function (m: any) { if (uniq.indexOf(m.symbol) < 0) uniq.push(m.symbol); });
        if (uniq.length) { setSymbols(uniq); setSymbol(uniq[0]); }
      })
      .catch(function () { /* keep fallback list */ });
  }, []);

  const load = useCallback(function (sym: string) {
    setLoading(true); setError("");
    fetchJSON("/recommendation/" + encodeURIComponent(sym))
      .then(function (d: Recommendation) { setRec(d); })
      .catch(function (e: Error) { setError(e.message); setRec(null); })
      .then(function () { setLoading(false); });
  }, []);

  useEffect(function () { load(symbol); }, [symbol, load]);

  const cone = useMemo(function () {
    if (!rec) return [];
    return buildCone(rec.probabilities, HORIZON_DAYS[horizon]);
  }, [rec, horizon]);

  const last = cone.length ? cone[cone.length - 1] : null;
  const sent = rec && rec.sentiment ? rec.sentiment : null;
  const feats = rec ? Object.keys(rec.top_features || {}) : [];

  return (
    <div className="wrap">
      <div className="topbar">
        <h1 className="title">FORECAST</h1>
        <span className="badge-new">new</span>
        <div className="spacer"></div>
        {rec ? <span className="asof">Data as of {rec.data_as_of}</span> : null}
      </div>

      <div className="controls">
        <select className="symbol-select" value={symbol} onChange={function (e) { setSymbol(e.target.value); }}>
          {symbols.map(function (s) { return <option key={s} value={s}>{s}</option>; })}
        </select>
        <div className="seg">
          {(["day", "month", "year"] as Horizon[]).map(function (h) {
            return (
              <button key={h} className={horizon === h ? "active" : ""} onClick={function () { setHorizon(h); }}>
                {h === "day" ? "Days" : h === "month" ? "Month" : "Year"}
              </button>
            );
          })}
        </div>
        <button className="refresh" onClick={function () { load(symbol); }} disabled={loading}>
          {loading ? "Loading…" : "↻ Refresh"}
        </button>
      </div>

      <div className="divider"></div>

      <div className="disclaimer">
        <strong>Model-implied projection.</strong> The model predicts a <em>next-day</em> 5-class signal only.
        The price-forecast cone extrapolates that signal's probability distribution over the selected horizon
        (Days / Month / Year), indexed to 100 today. It is a visualization, <strong>not a price target</strong>.
      </div>

      {error ? <div className="state error">⚠ {error}</div> : null}
      {!error && !rec && loading ? <div className="state">Loading forecast…</div> : null}

      {rec ? (
        <div className="grid">
          {/* ── Panel 1: Share Price Forecast ── */}
          <div className="card">
            <h3>Share Price Forecast</h3>
            <div className="cone-wrap">
              <ConeChart points={cone} />
              <div className="cone-legend">
                <div className="lvl high"><div className="k">HIGH</div><div className="v">{last ? signed(last.high / 100 - 1) : "—"}</div></div>
                <div className="lvl mean"><div className="k">MEAN</div><div className="v">{last ? signed(last.mean / 100 - 1) : "—"}</div></div>
                <div className="lvl low"><div className="k">LOW</div><div className="v">{last ? signed(last.low / 100 - 1) : "—"}</div></div>
              </div>
            </div>
            <div className="cone-foot">Projected over {HORIZON_LABEL[horizon]} · indexed to 100 today</div>
          </div>

          {/* ── Panel 2: Model Signal Rating ── */}
          <div className="card">
            <h3>Model Rating · {rec.signal}</h3>
            <div className="donut-row">
              <SignalDonut rec={rec} />
              <div className="rating-list">
                {CLASS_ORDER.map(function (k) {
                  const v = (rec.probabilities as any)[k] as number;
                  return (
                    <div className="rating-row" key={k}>
                      <span className="lbl">{CLASS_TEXT[k]}</span>
                      <span className="track"><span className="fill" style={{ width: pct(v), background: CLASS_COLOR[k] }}></span></span>
                      <span className="pct">{Math.round(v * 100)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* ── Panel 3: Sentiment & Outlook ── */}
          <div className="card">
            <h3>Sentiment &amp; Outlook</h3>
            <div className="kv">
              <span className="k">News sentiment</span>
              <span className="v">
                <span className={"pill " + (sent ? sent.label : "neutral")}>{sent ? sent.label : "neutral"}</span>
              </span>
            </div>
            <div className="kv"><span className="k">Sentiment score</span><span className="v">{sent ? signed(sent.score) : "0.0%"}</span></div>
            <div className="kv"><span className="k">Headlines (7d)</span><span className="v">{sent ? sent.news_count : 0}</span></div>
            <div className="kv"><span className="k">Model</span><span className="v">{rec.model_used}</span></div>

            {feats.length ? (
              <div style={{ marginTop: "10px" }}>
                <div className="explain h">Top drivers</div>
                {feats.slice(0, 5).map(function (f) {
                  return <div className="feat" key={f}><span>{f}</span><span>{(rec.top_features[f]).toFixed(3)}</span></div>;
                })}
              </div>
            ) : <div className="muted-note" style={{ marginTop: "10px" }}>Top drivers unavailable (SHAP not installed).</div>}

            <div className="explain">
              <div className="h">Explanation</div>
              {rec.explanation}
            </div>
          </div>
        </div>
      ) : null}

      {/* ── Advanced price chart (Lightweight Charts on our OHLCV) — below the forecast ── */}
      <div className="card chart-card">
        <h3 className="chart-h">Advanced Chart · {symbol}</h3>
        <AdvancedChart symbol={symbol} />
      </div>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
