import { useEffect, useState } from "react";

const DEFAULT_SEQS = `EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDLGRRGYFDYWGQGTLVTVSS
DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK
GSHMKEIAALKEKIAALKEKIAALKE`;

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

function AdaptyvLogo({ height = 26 }) {
  return (
    <span className="brand" aria-label="Adaptyv">
      <svg className="brand__mark" style={{ height, width: height }} viewBox="0 0 32 32" fill="none" role="img" aria-hidden="true">
        <path d="M16 2.4 27.6 9v14L16 29.6 4.4 23V9L16 2.4Z" fill="none" stroke="#17293b" strokeWidth="2.1" strokeLinejoin="round"/>
        <path d="M11 20.5c0-4 2.6-7 6-7 2.2 0 3.8 1.4 3.8 3.4 0 2.2-1.8 3.6-4 3.6-2.5 0-4.2-1.7-4.2-4.2 0-3 2.3-5.3 5.4-5.3" fill="none" stroke="#5bb2ef" strokeWidth="2.1" strokeLinecap="round"/>
      </svg>
      <span className="brand__word">Adaptyv</span>
    </span>
  );
}

function badgeClass(v) {
  const s = String(v).toLowerCase();
  if (["strong", "high"].includes(s)) return "badge badge--good";
  if (["medium", "weak", "low"].includes(s)) return "badge badge--mid";
  return "badge badge--none";
}

function LineChart({ tested, informed, random, total }) {
  const W = 600, H = 300, pad = { l: 44, r: 16, t: 16, b: 34 };
  const xmax = tested[tested.length - 1];
  const ymax = Math.ceil(total / 10) * 10;
  const sx = (x) => pad.l + (x / xmax) * (W - pad.l - pad.r);
  const sy = (y) => H - pad.b - (y / ymax) * (H - pad.t - pad.b);
  const path = (arr) => arr.map((y, i) => `${i ? "L" : "M"}${sx(tested[i]).toFixed(1)},${sy(y).toFixed(1)}`).join(" ");
  const yticks = [0, ymax / 2, ymax];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Active learning discovery curve">
      {yticks.map((t, i) => (
        <g key={i}>
          <line className="grid" x1={pad.l} y1={sy(t)} x2={W - pad.r} y2={sy(t)} />
          <text className="lbl" x={pad.l - 8} y={sy(t) + 4} textAnchor="end">{t}</text>
        </g>
      ))}
      <line className="axis" x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} />
      {[0, xmax / 2, xmax].map((t, i) => (
        <text key={i} className="lbl" x={sx(t)} y={H - pad.b + 20} textAnchor="middle">{Math.round(t)}</text>
      ))}
      <path className="line-random" d={path(random)} />
      <path className="line-informed" d={path(informed)} />
    </svg>
  );
}

export default function App() {
  const [stats, setStats] = useState(null);
  const [binder, setBinder] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [bt, setBt] = useState(null);
  const [agentOn, setAgentOn] = useState(false);

  // rank tool
  const [seqs, setSeqs] = useState(DEFAULT_SEQS);
  const [budget, setBudget] = useState(2);
  const [ranked, setRanked] = useState(null);
  const [ranking, setRanking] = useState(false);

  // agent
  const [msg, setMsg] = useState(
    "I have budget for 2 tests. Rank these and explain the trade-off:\n" + DEFAULT_SEQS
  );
  const [reply, setReply] = useState("");
  const [thinking, setThinking] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    getJSON("/api/stats").then(setStats).catch(() => {});
    getJSON("/api/example-binder").then(setBinder).catch(() => {});
    getJSON("/api/metrics").then(setMetrics).catch(() => {});
    getJSON("/api/backtest").then(setBt).catch(() => {});
    getJSON("/api/agent/available").then((d) => setAgentOn(d.available)).catch(() => {});
  }, []);

  async function runRank() {
    setRanking(true); setErr("");
    try {
      const d = await postJSON("/api/rank", {
        sequences: seqs.split("\n").map((s) => s.trim()).filter(Boolean),
        budget: Number(budget),
      });
      setRanked(d.ranked);
    } catch (e) { setErr(String(e.message)); }
    setRanking(false);
  }

  async function runAgent() {
    setThinking(true); setErr(""); setReply("");
    try {
      const d = await postJSON("/api/agent", { message: msg });
      setReply(d.reply);
    } catch (e) { setErr(String(e.message)); }
    setThinking(false);
  }

  const s100 = bt?.summary_at_100;

  return (
    <>
      <nav className="nav">
        <AdaptyvLogo />
        <div className="nav__links">
          <a href="#workflow">Workflow</a>
          <a href="#results">Results</a>
          <a href="#rank">Rank</a>
          <a href="#agent">Agent</a>
        </div>
        <div className="nav__meta">EGFR · POC</div>
      </nav>

      {/* HERO */}
      <section className="section section--dark section--hero">
        <div className="wrap hero">
          <div>
            <p className="eyebrow">Active learning for the cloud lab</p>
            <h1>Test fewer designs.<br />Find more binders.</h1>
            <p className="lead" style={{ marginTop: 24 }}>
              An active-learning + agent layer over Adaptyv Bio's public EGFR
              competition data. Rank designs so the real binders surface early,
              same experimental budget, more hits.
            </p>
            <div className="hero__cta">
              <a className="btn btn--primary" href="#results">See the result</a>
              <a className="btn btn--ghost" href="#agent">Try the agent</a>
            </div>
          </div>

          <div className="card card--float">
            <div className="datacard__head">
              <span className="datacard__id">
                {binder ? binder.id : "binder_——"} <span style={{ opacity: 0.5 }}>· R{binder?.round ?? "—"}</span>
              </span>
              <span className="datacard__dot" />
            </div>
            <div className="datacard__metric">
              <div className="datacard__label">Binding affinity (K<sub>D</sub>)</div>
              <div className="datacard__value">
                {binder ? binder.kd_nM : "—"}<small>nM</small>
              </div>
            </div>
            <div className="datacard__row">
              <span className={badgeClass(binder?.binding_strength)}>
                {binder?.binding_strength || "—"}
              </span>
              <span className={badgeClass(binder?.expression)}>
                Expr · {binder?.expression || "—"}
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* WORKFLOW */}
      <section id="workflow" className="section section--light">
        <div className="wrap">
          <p className="eyebrow">How it plugs in</p>
          <h2>A round advisor for the design → test loop</h2>
          <div className="steps">
            <div className="step">
              <div className="step__num mono">001</div>
              <h3>Upload designs</h3>
              <p>Submit candidate sequences. The model scores each for binder probability and estimated affinity.</p>
            </div>
            <div className="step">
              <div className="step__num mono">002</div>
              <h3>Rank &amp; run</h3>
              <p>An acquisition function picks the batch worth testing under your budget, then submits it to the lab API.</p>
            </div>
            <div className="step">
              <div className="step__num mono">003</div>
              <h3>Plan next round</h3>
              <p>Results feed back in. The advisor updates and proposes the next batch, closing the active-learning loop.</p>
            </div>
          </div>
        </div>
      </section>

      {/* RESULTS */}
      <section id="results" className="section section--dark">
        <div className="wrap">
          <p className="eyebrow">Cross-validated, on real data</p>
          <h2>What the decision layer buys you</h2>
          <div className="tiles">
            <div className="tile">
              <div className="tile__value">{stats?.designs ?? "—"}</div>
              <div className="tile__label">Designs pooled</div>
            </div>
            <div className="tile">
              <div className="tile__value">{stats?.binders ?? "—"}</div>
              <div className="tile__label">Real binders</div>
            </div>
            <div className="tile">
              <div className="tile__value">{metrics?.roc_auc ?? "—"}</div>
              <div className="tile__label">Classifier ROC-AUC</div>
            </div>
            <div className="tile">
              <div className="tile__value">
                {s100 ? `${s100.lift.toFixed(1)}` : "—"}<small>×</small>
              </div>
              <div className="tile__label">Binders vs random @100</div>
            </div>
          </div>
        </div>
      </section>

      {/* ACTIVE LEARNING */}
      <section className="section section--light">
        <div className="wrap">
          <p className="eyebrow">The backtest</p>
          <h2>Informed selection vs random</h2>
          <div className="chartrow">
            <div>
              {bt && (
                <LineChart tested={bt.tested} informed={bt.informed} random={bt.random} total={bt.total_hits} />
              )}
              <div className="legend">
                <span><i style={{ background: "var(--accent)" }} /> Informed (UCB acquisition)</span>
                <span><i style={{ background: "#b8b8b0" }} /> Random selection</span>
              </div>
            </div>
            <div className="callout">
              <div className="callout__big">
                {s100 ? `${s100.informed_found.toFixed(0)}` : "—"} <span>vs {s100 ? s100.random_found.toFixed(0) : "—"}</span>
              </div>
              <p>
                binders found at a 100-test budget, informed vs random.
                That's {s100 ? `${(s100.informed_recall * 100).toFixed(0)}%` : "—"} of all
                {" "}{bt?.total_hits ?? "—"} binders, versus {s100 ? `${(s100.random_recall * 100).toFixed(0)}%` : "—"}.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* RANK */}
      <section id="rank" className="section section--paper">
        <div className="wrap">
          <p className="eyebrow">Try it, no key needed</p>
          <h2>Score &amp; rank candidate sequences</h2>
          <div className="tool">
            <div>
              <label className="field">Sequences (one per line)</label>
              <textarea rows={6} value={seqs} onChange={(e) => setSeqs(e.target.value)} />
            </div>
            <div className="row">
              <div>
                <label className="field">Budget</label>
                <input type="number" min={1} max={100} value={budget}
                  onChange={(e) => setBudget(e.target.value)} style={{ width: 110 }} />
              </div>
              <button className="btn btn--dark" onClick={runRank} disabled={ranking}>
                {ranking ? "Ranking…" : "Rank candidates"}
              </button>
            </div>
          </div>

          {ranked && (
            <div className="rank">
              <div className="rank__head">
                <div>#</div><div>Sequence</div><div>P(bind)</div>
                <div className="rank__col--p50">Uncert.</div><div>Selected</div>
              </div>
              {ranked.map((r, i) => (
                <div key={i} className={"rank__row" + (r.selected ? " is-selected" : "")}>
                  <div className="rank__idx mono">{String(i + 1).padStart(2, "0")}</div>
                  <div className="rank__seq">{r.sequence}</div>
                  <div className="rank__num">{(r.binder_probability * 100).toFixed(0)}%</div>
                  <div className="rank__col--p50 mono" style={{ color: "var(--ink-soft)" }}>
                    ±{(r.binder_probability_std * 100).toFixed(0)}
                  </div>
                  <div>
                    <span className={r.selected ? "badge badge--good" : "badge badge--none"}>
                      {r.selected ? "Test" : "Skip"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* AGENT */}
      <section id="agent" className="section section--dark">
        <div className="wrap">
          <p className="eyebrow">Natural-language orchestration</p>
          <h2>Ask the advisor</h2>
          {!agentOn && (
            <p className="hint" style={{ marginTop: 16 }}>
              Agent offline, set <span className="mono">OPENROUTER_API_KEY</span> in <span className="mono">.env</span> and restart the API.
            </p>
          )}
          <div className="tool agentbox">
            <div>
              <label className="field">Request</label>
              <textarea rows={5} value={msg} onChange={(e) => setMsg(e.target.value)} disabled={!agentOn} />
            </div>
            <div className="row">
              <button className="btn btn--primary" onClick={runAgent} disabled={!agentOn || thinking}>
                {thinking ? "Thinking…" : "Run agent"}
              </button>
              {thinking && <span className="spinner">calling tools…</span>}
            </div>
          </div>
          {reply && <div className="reply">{reply}</div>}
        </div>
      </section>

      {err && (
        <div style={{ position: "fixed", bottom: 20, left: "50%", transform: "translateX(-50%)",
          background: "#3a1414", color: "#ffbcbc", padding: "10px 18px", borderRadius: 10,
          fontFamily: "var(--mono)", fontSize: 13, zIndex: 50 }}>
          {err}
        </div>
      )}

      <footer className="section section--dark footer">
        <div className="wrap">
          <p>
            Data: Adaptyv Bio EGFR Protein Design Competition (rounds 1–2), licensed
            ODbL. The API schema and simulated lab results are a hypothesis, not
            Adaptyv's real system. POC, model is illustrative, not production.
          </p>
          <p className="mono">Adaptyv Foundry · EGFR · v0.1</p>
        </div>
      </footer>
    </>
  );
}
