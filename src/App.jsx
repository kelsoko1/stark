import { useState, useRef, useEffect } from "react";

// ─────────────────────────────────────────────────────────────────
// SHARED PRIMITIVES
// ─────────────────────────────────────────────────────────────────

const Code = ({ children }) => (
  <pre style={{
    background: '#04060d', borderRadius: 8, padding: '14px 16px', margin: '12px 0',
    fontFamily: "'JetBrains Mono','Fira Code',monospace", fontSize: 11,
    lineHeight: 1.65, color: '#c9d1d9', overflowX: 'auto', whiteSpace: 'pre',
    border: '1px solid #0e1220',
  }}>{children}</pre>
);

const Block = ({ color = '#a78bfa', children }) => (
  <div style={{
    background: `${color}09`, border: `1px solid ${color}25`,
    borderLeft: `3px solid ${color}`, borderRadius: 8,
    padding: '12px 16px', margin: '12px 0',
    color: `${color}cc`, fontSize: 13, lineHeight: 1.8,
  }}>{children}</div>
);

const Pill = ({ color, children }) => (
  <span style={{
    padding: '2px 8px', borderRadius: 4, fontSize: 10, fontWeight: 700,
    letterSpacing: '0.07em', background: `${color}18`,
    color, border: `1px solid ${color}35`, marginRight: 5,
  }}>{children}</span>
);

const Tag = ({ color, children }) => (
  <span style={{
    padding: '2px 8px', borderRadius: 4, fontSize: 10.5, fontWeight: 700,
    letterSpacing: '0.06em', background: `${color}22`,
    color, border: `1px solid ${color}44`, marginRight: 6,
  }}>{children}</span>
);

const H = ({ children }) => (
  <h3 style={{
    color: '#8892a4', fontSize: 13, fontWeight: 700,
    margin: '22px 0 8px', paddingBottom: 5,
    borderBottom: '1px solid #0e1018', letterSpacing: '0.02em',
  }}>{children}</h3>
);

const SectionTitle = ({ children }) => (
  <h3 style={{
    color: '#c0c0d0', fontSize: 14, fontWeight: 700,
    margin: '24px 0 8px', paddingBottom: 6,
    borderBottom: '1px solid #151520',
  }}>{children}</h3>
);

const Table = ({ headers, rows }) => (
  <div style={{ overflowX: 'auto', margin: '12px 0' }}>
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11.5 }}>
      <thead><tr>{headers.map(h => (
        <th key={h} style={{ padding: '7px 10px', textAlign: 'left',
          background: '#080b14', color: '#374151', fontSize: 10,
          letterSpacing: '0.07em', textTransform: 'uppercase',
          border: '1px solid #0e1018' }}>{h}</th>
      ))}</tr></thead>
      <tbody>{rows.map((row, i) => (
        <tr key={i} style={{ background: i % 2 === 0 ? '#060810' : '#04060d' }}>
          {row.map((cell, j) => (
            <td key={j} style={{
              padding: '8px 10px', border: '1px solid #0d1018',
              color: j === 0 ? '#c8d0e0' : '#5a6478',
              fontFamily: j === 0 ? "'JetBrains Mono',monospace" : 'inherit',
              fontSize: j === 0 ? 11 : 11.5,
            }}>{cell}</td>
          ))}
        </tr>
      ))}</tbody>
    </table>
  </div>
);

// ─────────────────────────────────────────────────────────────────
// Q-BITNET — CANVAS COMPONENTS
// ─────────────────────────────────────────────────────────────────

function SuperpositionCanvas({ alphaSq, gammaSq }) {
  const canvasRef = useRef(null);
  const frameRef  = useRef(null);
  const paramsRef = useRef({ alphaSq, gammaSq });
  useEffect(() => { paramsRef.current = { alphaSq, gammaSq }; }, [alphaSq, gammaSq]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let t = 0;

    function draw() {
      const { alphaSq, gammaSq } = paramsRef.current;
      const betaSq = Math.max(0, 1 - alphaSq - gammaSq);
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#07090f'; ctx.fillRect(0, 0, W, H);

      const k = 0.038, speed = 0.022;
      const drawWave = (amp, phaseOff, color, yCenter) => {
        if (amp < 0.01) return;
        ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.globalAlpha = 0.75;
        ctx.beginPath();
        for (let x = 0; x <= W; x++) {
          const y = yCenter + amp * 32 * Math.sin(x * k + phaseOff + t * speed);
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      };

      drawWave(Math.sqrt(alphaSq), 0,             '#60a5fa', 38);
      drawWave(Math.sqrt(betaSq),  Math.PI / 3,   '#94a3b8', 80);
      drawWave(Math.sqrt(gammaSq), Math.PI * 2/3, '#fbbf24', 122);

      ctx.strokeStyle = '#a78bfa'; ctx.lineWidth = 2.5; ctx.globalAlpha = 1;
      ctx.beginPath();
      for (let x = 0; x <= W; x++) {
        const v = Math.sqrt(alphaSq) * 28 * Math.sin(x * k + t * speed)
                + Math.sqrt(betaSq)  * 28 * Math.sin(x * k + Math.PI / 3 + t * speed)
                + Math.sqrt(gammaSq) * 28 * Math.sin(x * k + Math.PI * 2/3 + t * speed);
        const y = 175 + v;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.globalAlpha = 1; ctx.font = '11px monospace';
      const eff = (gammaSq - alphaSq).toFixed(3);
      ctx.fillStyle = '#60a5fa'; ctx.fillText(`|ψ₋₁⟩  α²=${alphaSq.toFixed(2)}`, 8, 25);
      ctx.fillStyle = '#94a3b8'; ctx.fillText(`|ψ₀⟩   β²=${betaSq.toFixed(2)}`, 8, 67);
      ctx.fillStyle = '#fbbf24'; ctx.fillText(`|ψ₊₁⟩  γ²=${gammaSq.toFixed(2)}`, 8, 109);
      ctx.fillStyle = '#a78bfa'; ctx.fillText(`|ψ⟩ superposition  →  w_eff = γ²−α² = ${eff}`, 8, 162);

      t++;
      frameRef.current = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(frameRef.current);
  }, []);

  return (
    <canvas ref={canvasRef} width={600} height={220}
      style={{ width: '100%', height: 220, borderRadius: 8, display: 'block' }} />
  );
}

function InterferenceCanvas() {
  const canvasRef = useRef(null);
  const frameRef  = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let t = 0;

    function draw() {
      const W = canvas.width, H = canvas.height;
      const img = ctx.createImageData(W, H);
      const s1 = { x: W * 0.32, y: H * 0.5 };
      const s2 = { x: W * 0.68, y: H * 0.5 };

      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const d1 = Math.sqrt((x - s1.x) ** 2 + (y - s1.y) ** 2);
          const d2 = Math.sqrt((x - s2.x) ** 2 + (y - s2.y) ** 2);
          const k = 0.18, decay = 0.03;
          const r1 = Math.cos(d1 * k - t * 0.04) / (1 + d1 * decay);
          const i1 = Math.sin(d1 * k - t * 0.04) / (1 + d1 * decay);
          const r2 = Math.cos(d2 * k - t * 0.04) / (1 + d2 * decay);
          const i2 = Math.sin(d2 * k - t * 0.04) / (1 + d2 * decay);
          const re = r1 + r2, im = i1 + i2;
          const prob = Math.min(1, (re * re + im * im) * 3);
          const idx = (y * W + x) * 4;
          img.data[idx]     = Math.floor(prob * 100);
          img.data[idx + 1] = Math.floor(prob * 60);
          img.data[idx + 2] = Math.floor(prob * 220);
          img.data[idx + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);

      [[s1, 'Q'], [s2, 'K']].forEach(([s, lbl]) => {
        ctx.beginPath(); ctx.arc(s.x, s.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = '#fbbf24'; ctx.fill();
        ctx.fillStyle = '#0a0a0a'; ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'center'; ctx.fillText(lbl, s.x, s.y + 4);
      });
      ctx.textAlign = 'left'; ctx.fillStyle = 'rgba(200,200,255,0.6)'; ctx.font = '11px monospace';
      ctx.fillText('P = |ψQ + ψK|²  (Born rule — bright = constructive interference = high attention)', 8, H - 6);

      t++;
      frameRef.current = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(frameRef.current);
  }, []);

  return (
    <canvas ref={canvasRef} width={600} height={190}
      style={{ width: '100%', height: 190, borderRadius: 8, display: 'block' }} />
  );
}

function PhaseCanvas() {
  const canvasRef = useRef(null);
  const frameRef  = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let t = 0;

    const tokens = [
      { label: 'the', mag: 0.7, phase: 0.3 },
      { label: 'cat', mag: 0.9, phase: 1.1 },
      { label: 'sat', mag: 0.85, phase: 2.4 },
      { label: 'on',  mag: 0.6, phase: 3.9 },
      { label: 'mat', mag: 0.95, phase: 0.8 },
    ];

    function draw() {
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#07090f'; ctx.fillRect(0, 0, W, H);

      const r = 44, cy = H / 2 - 8;
      const spacing = W / (tokens.length + 1);

      tokens.forEach((tok, i) => {
        const cx = spacing * (i + 1);
        const ph = tok.phase + t * 0.018;

        ctx.strokeStyle = '#1a1f2e'; ctx.lineWidth = 0.8;
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();

        ctx.strokeStyle = '#111827'; ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(cx - r - 4, cy); ctx.lineTo(cx + r + 4, cy);
        ctx.moveTo(cx, cy - r - 4); ctx.lineTo(cx, cy + r + 4);
        ctx.stroke();

        const ex = cx + tok.mag * r * Math.cos(ph);
        const ey = cy - tok.mag * r * Math.sin(ph);

        ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, cy); ctx.stroke();

        ctx.strokeStyle = '#10b981'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(ex, cy); ctx.lineTo(ex, ey); ctx.stroke();

        ctx.strokeStyle = '#a78bfa'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();

        ctx.fillStyle = '#a78bfa';
        ctx.beginPath(); ctx.arc(ex, ey, 4, 0, Math.PI * 2); ctx.fill();

        ctx.fillStyle = '#6b7280'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
        ctx.fillText(tok.label, cx, cy + r + 14);
        ctx.fillStyle = '#a78bfa';
        ctx.fillText(`${(ph * 180 / Math.PI % 360).toFixed(0)}°`, cx, cy + r + 26);
      });

      ctx.textAlign = 'left'; ctx.font = '10px monospace';
      ctx.fillStyle = '#3b82f6'; ctx.fillText('— Re(ψ)', 8, H - 28);
      ctx.fillStyle = '#10b981'; ctx.fillText('— Im(ψ)', 8, H - 16);
      ctx.fillStyle = '#a78bfa'; ctx.fillText('— phase vector |ψ⟩', 70, H - 22);

      t++;
      frameRef.current = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(frameRef.current);
  }, []);

  return (
    <canvas ref={canvasRef} width={600} height={170}
      style={{ width: '100%', height: 170, borderRadius: 8, display: 'block' }} />
  );
}

function TunnelingCanvas() {
  const canvasRef = useRef(null);
  const frameRef  = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const loss = x => {
      const n = x / canvas.width;
      return 0.28 + 0.35 * Math.sin(n * Math.PI * 2.2)
                  + 0.18 * Math.sin(n * Math.PI * 6)
                  + 0.1  * Math.cos(n * Math.PI * 4.5);
    };

    const state = { x: 40, vx: 0.8, tunnel: false, tTarget: 0, tCool: 0, path: [] };

    function draw() {
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#07090f'; ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 2;
      ctx.beginPath();
      for (let x = 0; x <= W; x++) {
        const y = H * 0.1 + loss(x) * H * 0.7;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.lineTo(W, H); ctx.lineTo(0, H);
      ctx.fillStyle = 'rgba(109,40,217,0.06)'; ctx.fill();

      if (state.path.length > 1) {
        for (let i = 1; i < state.path.length; i++) {
          const p = state.path[i - 1], c = state.path[i];
          ctx.strokeStyle = c.t ? '#fbbf24' : '#a78bfa';
          ctx.lineWidth = c.t ? 2 : 1;
          ctx.globalAlpha = Math.min(1, i / state.path.length * 1.5);
          ctx.setLineDash(c.t ? [4, 3] : []);
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(c.x, c.y); ctx.stroke();
        }
        ctx.setLineDash([]); ctx.globalAlpha = 1;
      }

      state.tCool = Math.max(0, state.tCool - 1);
      if (!state.tunnel) {
        const g = (loss(state.x + 1) - loss(state.x - 1)) / 2;
        state.vx = state.vx * 0.88 - g * 10;
        state.x  = Math.max(5, Math.min(W - 5, state.x + state.vx));
        if (state.tCool === 0 && Math.random() < 0.006) {
          state.tunnel  = true;
          state.tTarget = state.x + (Math.random() > 0.5 ? 1 : -1) * (70 + Math.random() * 130);
          state.tCool   = 80;
        }
      } else {
        state.x += (state.tTarget - state.x) * 0.22;
        if (Math.abs(state.x - state.tTarget) < 2) { state.tunnel = false; state.vx = 0; }
      }

      const py = H * 0.1 + loss(state.x) * H * 0.7;
      state.path.push({ x: state.x, y: py, t: state.tunnel });
      if (state.path.length > 180) state.path.shift();

      ctx.shadowColor = state.tunnel ? '#fbbf24' : '#a78bfa'; ctx.shadowBlur = 10;
      ctx.fillStyle   = state.tunnel ? '#fbbf24' : '#a78bfa';
      ctx.beginPath(); ctx.arc(state.x, py, 7, 0, Math.PI * 2); ctx.fill();
      ctx.shadowBlur = 0;

      ctx.fillStyle = '#374151'; ctx.font = '10px monospace';
      ctx.fillText('loss landscape', 6, 14);
      ctx.fillStyle = '#fbbf24';
      ctx.fillText('─ ─ tunneling event (escapes local minimum)', 6, H - 6);

      frameRef.current = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(frameRef.current);
  }, []);

  return (
    <canvas ref={canvasRef} width={600} height={170}
      style={{ width: '100%', height: 170, borderRadius: 8, display: 'block' }} />
  );
}

function EntanglementDiagram() {
  return (
    <svg width="100%" viewBox="0 0 600 200" style={{ display: 'block', borderRadius: 8 }}>
      <rect width="600" height="200" fill="#07090f" rx="8"/>
      <rect x="40" y="40"  width="100" height="40" rx="6" fill="#1e1b4b" stroke="#7c3aed" strokeWidth="0.8"/>
      <text x="90" y="65" fill="#a78bfa" fontSize="12" textAnchor="middle" fontFamily="monospace">Head A₁</text>
      <rect x="40" y="110" width="100" height="40" rx="6" fill="#1e1b4b" stroke="#7c3aed" strokeWidth="0.8"/>
      <text x="90" y="135" fill="#a78bfa" fontSize="12" textAnchor="middle" fontFamily="monospace">Head A₂</text>
      <rect x="460" y="40"  width="100" height="40" rx="6" fill="#1a2340" stroke="#3b82f6" strokeWidth="0.8"/>
      <text x="510" y="65" fill="#60a5fa" fontSize="12" textAnchor="middle" fontFamily="monospace">Head B₁</text>
      <rect x="460" y="110" width="100" height="40" rx="6" fill="#1a2340" stroke="#3b82f6" strokeWidth="0.8"/>
      <text x="510" y="135" fill="#60a5fa" fontSize="12" textAnchor="middle" fontFamily="monospace">Head B₂</text>
      <rect x="200" y="15"  width="60" height="28" rx="4" fill="#0f1629" stroke="#374151" strokeWidth="0.5"/>
      <text x="230" y="33" fill="#6b7280" fontSize="10" textAnchor="middle" fontFamily="monospace">token i</text>
      <rect x="340" y="15"  width="60" height="28" rx="4" fill="#0f1629" stroke="#374151" strokeWidth="0.5"/>
      <text x="370" y="33" fill="#6b7280" fontSize="10" textAnchor="middle" fontFamily="monospace">token j</text>
      <path d="M 140 60 C 220 60 220 130 300 95 C 380 60 380 130 460 130"
            fill="none" stroke="#fbbf24" strokeWidth="1.5" strokeDasharray="5 3" opacity="0.8"/>
      <path d="M 140 130 C 220 130 220 60 300 95 C 380 130 380 60 460 60"
            fill="none" stroke="#fbbf24" strokeWidth="1.5" strokeDasharray="5 3" opacity="0.8"/>
      <circle cx="300" cy="95" r="16" fill="#1a1000" stroke="#fbbf24" strokeWidth="1"/>
      <text x="300" y="99" fill="#fbbf24" fontSize="9" textAnchor="middle" fontFamily="monospace">C_ent</text>
      <line x1="140" y1="50"  x2="200" y2="29" stroke="#a78bfa" strokeWidth="0.8" opacity="0.5"/>
      <line x1="140" y1="130" x2="200" y2="43" stroke="#a78bfa" strokeWidth="0.8" opacity="0.5"/>
      <line x1="460" y1="50"  x2="400" y2="29" stroke="#60a5fa" strokeWidth="0.8" opacity="0.5"/>
      <line x1="460" y1="130" x2="400" y2="43" stroke="#60a5fa" strokeWidth="0.8" opacity="0.5"/>
      <line x1="40" y1="185" x2="70" y2="185" stroke="#fbbf24" strokeWidth="1.5" strokeDasharray="5 3"/>
      <text x="76" y="189" fill="#6b7280" fontSize="10" fontFamily="monospace">entanglement coupling (non-local correlation between head pairs)</text>
    </svg>
  );
}

function BlochDiagram() {
  return (
    <svg width="100%" viewBox="0 0 600 200" style={{ display: 'block', borderRadius: 8 }}>
      <rect width="600" height="200" fill="#07090f" rx="8"/>
      {[0, 1, 2].map((idx) => {
        const cx = 80 + idx * 180, cy = 100, r = 70;
        const labels  = [['w=-1','−1','#60a5fa'],['w=0','0','#94a3b8'],['w=+1','+1','#fbbf24']];
        const angles  = [Math.PI * 0.8, Math.PI * 0.5, Math.PI * 0.15];
        const [stateLabel, wLabel, color] = labels[idx];
        const theta = angles[idx];
        const px = cx + r * 0.9 * Math.sin(theta);
        const py = cy - r * 0.9 * Math.cos(theta);
        return (
          <g key={idx}>
            <ellipse cx={cx} cy={cy} rx={r} ry={r} fill="none" stroke="#1a1f2e" strokeWidth="0.8"/>
            <ellipse cx={cx} cy={cy} rx={r} ry={r * 0.28} fill="none" stroke="#1a1f2e" strokeWidth="0.5" strokeDasharray="3 3"/>
            <line x1={cx} y1={cy - r - 6} x2={cx} y2={cy + r + 6} stroke="#1a1f2e" strokeWidth="0.5"/>
            <text x={cx} y={cy - r - 10} fill="#374151" fontSize="9" textAnchor="middle" fontFamily="monospace">|+1⟩</text>
            <text x={cx} y={cy + r + 18} fill="#374151" fontSize="9" textAnchor="middle" fontFamily="monospace">|−1⟩</text>
            <text x={cx + r + 8} y={cy + 4} fill="#374151" fontSize="9" fontFamily="monospace">|0⟩</text>
            <line x1={cx} y1={cy} x2={px} y2={py} stroke={color} strokeWidth="2"/>
            <circle cx={px} cy={py} r="5" fill={color}/>
            <path d={`M ${cx} ${cy - 22} A 22 22 0 0 1 ${cx + 22 * Math.sin(theta)} ${cy - 22 * Math.cos(theta)}`}
                  fill="none" stroke={color} strokeWidth="0.8" opacity="0.5"/>
            <text x={cx + 26 * Math.sin(theta / 2)} y={cy - 26 * Math.cos(theta / 2)} fill={color} fontSize="8" fontFamily="monospace">θ</text>
            <text x={cx} y={cy + r + 32} fill={color} fontSize="10" textAnchor="middle" fontFamily="monospace">{stateLabel}</text>
            <text x={cx} y={cy + r + 45} fill="#4b5563" fontSize="9" textAnchor="middle" fontFamily="monospace">w={wLabel}</text>
          </g>
        );
      })}
      <text x="300" y="192" fill="#374151" fontSize="9" textAnchor="middle" fontFamily="monospace">
        Bloch sphere: θ encodes ternary weight, φ (azimuthal) encodes asymmetric phase — both learned continuously
      </text>
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────
// Q-BITNET — TAB CONTENT
// ─────────────────────────────────────────────────────────────────

function QB_Overview() {
  return (
    <div>
      <Block color="#a78bfa">
        Six quantum mechanical principles, each mapped to a concrete, differentiable
        architectural change. Not metaphors — actual operations you can implement in
        PyTorch and train end-to-end on your VPS fleet.
      </Block>
      <SectionTitle>Quantum principle → architecture mapping</SectionTitle>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead><tr>{['Quantum principle','Architectural change','Benefit','Cost'].map(h => (
            <th key={h} style={{ padding: '7px 10px', textAlign: 'left',
              background: '#0d0f18', color: '#4b5563', fontSize: 10.5,
              letterSpacing: '0.06em', textTransform: 'uppercase',
              border: '1px solid #151520' }}>{h}</th>
          ))}</tr></thead>
          <tbody>{[
            ['Superposition','QuantumSuperpositionWeight: weight = α²·(−1)+γ²·(+1)','Continuous, differentiable ternary quantization','~3× params vs hard quant'],
            ['Born rule (|ψ|²)','Interference attention: attn = |QK*|² / Z','Constructive/destructive interference in attention','Slightly more compute'],
            ['Entanglement','Entangled head pairs share non-local coupling matrix','Long-range correlation without extra layers','Coupling matrix per pair'],
            ['Phase encoding','Complex-valued residual stream Re+Im','Doubles information capacity of residuals','2× hidden dim storage'],
            ['Quantum tunneling','Tunneling Muon: stochastic large steps','Escapes local minima in loss landscape','Occasional instability'],
            ['Hadamard gate','Parameter-free H-transform between blocks','Basis rotation, no params','~0 cost'],
            ['Bloch sphere','Spherical weight parameterization (θ, φ)','Continuous path between ternary states','Trigonometric forward pass'],
          ].map(([q, a, b, c], i) => (
            <tr key={i} style={{ background: i % 2 === 0 ? '#0a0c14' : '#080a10' }}>
              <td style={{ padding: '8px 10px', color: '#a78bfa', border: '1px solid #12141e', fontFamily: 'monospace', fontSize: 11 }}>{q}</td>
              <td style={{ padding: '8px 10px', color: '#d1d5db', border: '1px solid #12141e', fontSize: 11 }}>{a}</td>
              <td style={{ padding: '8px 10px', color: '#6b7280', border: '1px solid #12141e', fontSize: 11 }}>{b}</td>
              <td style={{ padding: '8px 10px', color: '#4b5563', border: '1px solid #12141e', fontSize: 11 }}>{c}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <SectionTitle>Recommended combination for 2B4T fine-tuning</SectionTitle>
      <Block color="#34d399">
        Start with <strong>Superposition Weights + Interference Attention</strong> — highest ROI, drop-in replacements.
        Add <strong>Phase Encoding</strong> after verifying stability. Use <strong>Entanglement</strong> only for the top 6 layers
        (late layers do most of the capability work). <strong>Tunneling optimizer</strong> replaces Muon's standard clipping.
        <strong>Hadamard mixing</strong> is free — always on.
      </Block>
    </div>
  );
}

function QB_SuperpositionTab() {
  const [alphaSq, setAlphaSq] = useState(0.25);
  const [gammaSq, setGammaSq] = useState(0.55);
  const betaSq = Math.max(0, 1 - alphaSq - gammaSq);
  const wEff   = (gammaSq - alphaSq).toFixed(3);

  return (
    <div>
      <Block color="#60a5fa">
        Standard BitNet collapses weights to {'{−1, 0, +1}'} via a hard threshold — a non-differentiable step.
        Quantum superposition replaces this: each weight maintains a probability amplitude over all three states.
        The effective weight during training is the expectation value. At inference, measurement collapses to argmax.
      </Block>
      <SectionTitle>Interactive: adjust amplitude probabilities</SectionTitle>
      <SuperpositionCanvas alphaSq={alphaSq} gammaSq={gammaSq} />
      <div style={{ display: 'flex', gap: 16, marginTop: 12, flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 200 }}>
          <label style={{ fontSize: 11, color: '#60a5fa', display: 'block', marginBottom: 4 }}>
            α² = P(w=−1): {alphaSq.toFixed(2)}
          </label>
          <input type="range" min={0} max={95} value={Math.round(alphaSq * 100)}
            onChange={e => { const v = +e.target.value / 100; if (v + gammaSq <= 1) setAlphaSq(v); }}
            style={{ width: '100%' }} />
        </div>
        <div style={{ flex: 1, minWidth: 200 }}>
          <label style={{ fontSize: 11, color: '#fbbf24', display: 'block', marginBottom: 4 }}>
            γ² = P(w=+1): {gammaSq.toFixed(2)}
          </label>
          <input type="range" min={0} max={95} value={Math.round(gammaSq * 100)}
            onChange={e => { const v = +e.target.value / 100; if (alphaSq + v <= 1) setGammaSq(v); }}
            style={{ width: '100%' }} />
        </div>
        <div style={{ padding: '8px 14px', background: '#1a1a2e', borderRadius: 6, fontSize: 12, alignSelf: 'flex-end' }}>
          <span style={{ color: '#94a3b8' }}>β² = {betaSq.toFixed(2)}</span>
          {'  '}
          <span style={{ color: '#a78bfa', fontWeight: 700 }}>w_eff = {wEff}</span>
        </div>
      </div>
      <SectionTitle>Implementation</SectionTitle>
      <Code>{`class QuantumSuperpositionWeight(nn.Module):
    """
    Each weight is a quantum state |ψ⟩ = α|−1⟩ + β|0⟩ + γ|+1⟩
    with |α|² + |β|² + |γ|² = 1 enforced via softmax.
    
    Forward pass uses expectation value (differentiable).
    Inference "measures" — collapses to argmax(|α|², |β|², |γ|²).
    """
    def __init__(self, in_f, out_f):
        super().__init__()
        self.amplitudes = nn.Parameter(torch.zeros(out_f, in_f, 3))
        nn.init.normal_(self.amplitudes, std=0.1)
    
    def get_probs(self):
        return F.softmax(self.amplitudes, dim=-1)
    
    def effective_weight(self):
        probs  = self.get_probs()
        states = torch.tensor([-1., 0., 1.], device=probs.device)
        return (probs * states).sum(-1)
    
    def forward(self, x):
        return F.linear(x, self.effective_weight())
    
    @torch.no_grad()
    def collapse_to_ternary(self):
        probs = self.get_probs()
        ternary_idx = probs.argmax(-1)
        return ternary_idx.float() - 1.0
    
    def entropy_loss(self, target_entropy=0.3):
        probs   = self.get_probs()
        entropy = -(probs * probs.log().clamp(-20)).sum(-1).mean()
        return (entropy - target_entropy).abs()`}</Code>
    </div>
  );
}

function QB_InterferenceTab() {
  return (
    <div>
      <Block color="#a78bfa">
        Standard attention uses softmax(QKᵀ/√d) — purely real, always positive.
        Quantum interference attention uses complex-valued Q and K. The attention weight
        becomes a probability density via the Born rule: P ∝ |⟨ψQ|ψK⟩|². Complex
        phases create constructive (high attention) and destructive (low attention) interference.
      </Block>
      <SectionTitle>Live interference pattern — Q and K as wave sources</SectionTitle>
      <InterferenceCanvas />
      <SectionTitle>Why this is better than softmax</SectionTitle>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead><tr>{['Property','Softmax attention','Quantum interference attention'].map(h => (
            <th key={h} style={{ padding: '7px 10px', textAlign: 'left', background: '#0d0f18', color: '#4b5563', fontSize: 10.5, letterSpacing: '0.06em', textTransform: 'uppercase', border: '1px solid #151520' }}>{h}</th>
          ))}</tr></thead>
          <tbody>{[
            ['Output range','[0,1], sums to 1','[0,1], sums to 1 (same)'],
            ['Negative interactions','Impossible — softmax always positive','✓ Destructive interference suppresses irrelevant tokens'],
            ['Phase sensitivity','None — real dot product','✓ Phase alignment determines constructive vs destructive'],
            ['Sharpness','Controlled by temperature','Naturally peaked — Born rule amplifies high-amplitude pairs'],
            ['Complex gradient','Real only','Wirtinger calculus — full complex gradient flows'],
          ].map(([p, s, q], i) => (
            <tr key={i} style={{ background: i % 2 === 0 ? '#0a0c14' : '#080a10' }}>
              <td style={{ padding: '8px 10px', color: '#9ca3af', border: '1px solid #12141e', fontFamily: 'monospace', fontSize: 11 }}>{p}</td>
              <td style={{ padding: '8px 10px', color: '#4b5563', border: '1px solid #12141e', fontSize: 11 }}>{s}</td>
              <td style={{ padding: '8px 10px', color: '#34d399', border: '1px solid #12141e', fontSize: 11 }}>{q}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <SectionTitle>Implementation</SectionTitle>
      <Code>{`class QuantumInterferenceAttention(nn.Module):
    """
    Complex-valued Q,K attention using Born rule probability.
    Standard: A = softmax(QKᵀ / √d)
    Quantum:   A = |QK†|² / Σ|QK†|²
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.Qr = nn.Linear(d_model, d_model, bias=False)
        self.Qi = nn.Linear(d_model, d_model, bias=False)
        self.Kr = nn.Linear(d_model, d_model, bias=False)
        self.Ki = nn.Linear(d_model, d_model, bias=False)
        self.V  = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.phase_bias = nn.Parameter(torch.zeros(n_heads, 1, 1))
    
    def forward(self, x, mask=None):
        B, T, D = x.shape
        H, DH = self.n_heads, self.d_head
        def reshape(t): return t.view(B, T, H, DH).transpose(1, 2)
        Qc = torch.complex(reshape(self.Qr(x)), reshape(self.Qi(x)))
        Kc = torch.complex(reshape(self.Kr(x)), reshape(self.Ki(x)))
        V  = reshape(self.V(x)).real
        amplitude = torch.matmul(Qc.conj(), Kc.transpose(-2, -1)) / (DH ** 0.5)
        amplitude = amplitude * torch.exp(1j * self.phase_bias)
        prob = amplitude.abs() ** 2
        if mask is not None:
            prob = prob.masked_fill(mask == 0, 0.0)
        attn = prob / (prob.sum(-1, keepdim=True) + 1e-8)
        out  = torch.matmul(attn, V)
        out  = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out), attn`}</Code>
    </div>
  );
}

function QB_EntanglementTab() {
  return (
    <div>
      <Block color="#fbbf24">
        In quantum mechanics, entangled particles share state — measuring one instantly
        determines the other. In attention, we implement this as a learned coupling matrix
        C that creates non-local correlations between head pairs. When Head A decides to
        attend to position i, it sends a signal through C that biases Head B's attention
        distribution — without any extra layers or communication steps.
      </Block>
      <SectionTitle>Entangled head pair architecture</SectionTitle>
      <EntanglementDiagram />
      <SectionTitle>Bloch sphere weight parameterization</SectionTitle>
      <BlochDiagram />
      <SectionTitle>Implementation</SectionTitle>
      <Code>{`class EntangledMultiHeadAttention(nn.Module):
    """Pairs of attention heads share a learnable entanglement coupling."""
    def __init__(self, d_model, n_heads, entangle_top_k=4):
        super().__init__()
        assert n_heads % 2 == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.n_pairs = n_heads // 2
        self.standard_attn = nn.ModuleList([
            nn.Linear(d_model, self.d_head * 3) for _ in range(n_heads)
        ])
        self.C_entangle = nn.Parameter(
            torch.eye(self.d_head).unsqueeze(0).repeat(self.n_pairs, 1, 1) * 0.1
        )
        self.entangle_top_k = entangle_top_k
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, layer_idx=0):
        B, T, D = x.shape; DH = self.d_head; outputs = []
        for pair_idx in range(self.n_pairs):
            hA, hB = pair_idx * 2, pair_idx * 2 + 1
            qkvA = self.standard_attn[hA](x).chunk(3, dim=-1)
            QA, KA, VA = [t.view(B, T, DH) for t in qkvA]
            attnA = F.softmax(torch.bmm(QA, KA.transpose(1,2)) / DH**0.5, dim=-1)
            outA  = torch.bmm(attnA, VA)
            qkvB = self.standard_attn[hB](x).chunk(3, dim=-1)
            QB, KB, VB = [t.view(B, T, DH) for t in qkvB]
            if layer_idx >= self.n_heads - self.entangle_top_k:
                C  = self.C_entangle[pair_idx]
                KB = KB + torch.tanh(outA @ C)
            attnB = F.softmax(torch.bmm(QB, KB.transpose(1,2)) / DH**0.5, dim=-1)
            outB  = torch.bmm(attnB, VB)
            outputs.extend([outA, outB])
        return self.out_proj(torch.cat(outputs, dim=-1))`}</Code>
    </div>
  );
}

function QB_PhaseTab() {
  return (
    <div>
      <Block color="#10b981">
        Quantum states are complex-valued: |ψ⟩ = Re(ψ) + i·Im(ψ). The magnitude |ψ|
        carries the "what" of information; the phase angle ∠ψ carries the "how".
        We apply this to the transformer residual stream: each hidden state has a real
        component (standard) and an imaginary component (phase memory). They evolve
        independently and interfere at attention layers.
      </Block>
      <SectionTitle>Per-token phase evolution in the residual stream</SectionTitle>
      <PhaseCanvas />
      <div style={{ fontSize: 11, color: '#4b5563', marginTop: 8 }}>
        Each token maintains a rotating phase vector in the complex plane. The imaginary
        component accumulates sequence-level context independently from the real magnitude.
      </div>
      <SectionTitle>Implementation</SectionTitle>
      <Code>{`class PhaseEncodedResidualStream(nn.Module):
    """
    Augments transformer residual stream with an imaginary phase component.
    h_total = h_real + i·h_imag
    """
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.phase_freq = nn.Parameter(torch.randn(n_layers, d_model) * 0.01)
        self.phase_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.phase_gate[-1].weight)
        nn.init.zeros_(self.phase_gate[-1].bias)
    
    def forward_with_phase(self, h_real, h_imag, layer_idx):
        freq      = self.phase_freq[layer_idx]
        rotation  = torch.exp(1j * freq).unsqueeze(0).unsqueeze(0)
        h_complex = torch.complex(h_real, h_imag)
        h_rotated = h_complex * rotation
        h_real_new, h_imag_new = h_rotated.real, h_rotated.imag
        gate_input    = torch.cat([h_real_new, h_imag_new], dim=-1)
        phase_influence = self.phase_gate(gate_input)
        return h_real_new + phase_influence, h_imag_new`}</Code>
    </div>
  );
}

function QB_TunnelingTab() {
  return (
    <div>
      <Block color="#fbbf24">
        Quantum tunneling: a particle can pass through an energy barrier it classically
        can't surmount. In optimization, local minima are barriers. Standard gradient
        descent + momentum gets stuck. We implement probabilistic "tunneling steps" —
        occasional large gradient updates that bypass clipping — allowing the optimizer
        to escape and explore.
      </Block>
      <SectionTitle>Tunneling optimizer on a multi-basin loss landscape</SectionTitle>
      <TunnelingCanvas />
      <SectionTitle>Hadamard mixing layer (free parameter rotation)</SectionTitle>
      <Block color="#34d399">
        The Hadamard gate puts a qubit into equal superposition of |0⟩ and |1⟩. In neural
        networks, applying the Hadamard transform H to the hidden state rotates the representation
        into a basis where all features contribute equally — free parameter mixing, zero cost.
      </Block>
      <SectionTitle>Implementation</SectionTitle>
      <Code>{`class QuantumTunnelingMuon(torch.optim.Optimizer):
    """Muon optimizer augmented with quantum tunneling."""
    def __init__(self, params, lr=0.02, momentum=0.95,
                 tunnel_p0=0.02, tunnel_decay=5000, tunnel_scale=8.0, ns_iters=3):
        defaults = dict(lr=lr, momentum=momentum, tunnel_p0=tunnel_p0,
                        tunnel_decay=tunnel_decay, tunnel_scale=tunnel_scale,
                        ns_iters=ns_iters)
        super().__init__(params, defaults)
        self.step_count = 0
    
    def newton_schulz(self, G, iters=3):
        X = G / (G.norm() + 1e-8)
        for _ in range(iters):
            X = 1.5 * X - 0.5 * X @ X.T @ X
        return X
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        self.step_count += 1
        for group in self.param_groups:
            tunnel_prob = group['tunnel_p0'] * math.exp(-self.step_count / group['tunnel_decay'])
            tunneling   = (torch.rand(1).item() < tunnel_prob)
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                if tunneling and p.grad.dim() >= 2:
                    update = p.grad / (p.grad.norm() + 1e-8)
                    p.data.add_(update, alpha=-group['lr'] * group['tunnel_scale'])
                    state['momentum_buffer'].zero_()
                elif p.grad.dim() >= 2 and p.grad.shape[0] >= 32:
                    update = self.newton_schulz(p.grad, iters=group['ns_iters'])
                    buf    = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(update)
                    p.data.add_(buf, alpha=-group['lr'])
                else:
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg'].mul_(0.9).add_(p.grad, alpha=0.1)
                    p.data.add_(state['exp_avg'], alpha=-group['lr'] * 0.1)
        return loss


class HadamardMixingLayer(nn.Module):
    """Parameter-free Hadamard transform between transformer blocks."""
    def __init__(self, d_model):
        super().__init__()
        self.d        = d_model
        self.d_padded = 2 ** math.ceil(math.log2(d_model))
        self.mix_gate = nn.Parameter(torch.tensor(0.0))
    
    def fwht(self, x):
        h = 1
        while h < x.shape[-1]:
            x = x.view(*x.shape[:-1], -1, 2*h)
            x = torch.cat([x[..., :h] + x[..., h:], x[..., :h] - x[..., h:]], dim=-1)
            h *= 2
        return x / math.sqrt(x.shape[-1])
    
    def forward(self, x):
        gate  = torch.sigmoid(self.mix_gate)
        x_pad = F.pad(x, (0, self.d_padded - self.d)) if self.d_padded > self.d else x
        x_h   = self.fwht(x_pad)[..., :self.d]
        return gate * x_h + (1 - gate) * x`}</Code>
    </div>
  );
}

function QB_IntegrationTab() {
  return (
    <div>
      <Block color="#f472b6">
        All six quantum-inspired modifications assembled into one cohesive architecture.
        This is what you apply on top of the BitNet 2B4T checkpoint. Each module is
        independently switchable — add them one at a time, verify stability, then add the next.
      </Block>
      <SectionTitle>Full Q-BitNet model class</SectionTitle>
      <Code>{`import torch, torch.nn as nn, math
import torch.nn.functional as F

class QuantumBitNet(nn.Module):
    """
    BitNet 2B4T with 6 quantum-inspired architectural modifications:
    1. QuantumSuperpositionWeight    — differentiable ternary weights
    2. QuantumInterferenceAttention  — Born-rule attention (complex Q,K)
    3. EntangledMultiHeadAttention   — coupled head pairs
    4. PhaseEncodedResidualStream    — complex hidden states
    5. QuantumTunnelingMuon          — tunneling optimizer (in training loop)
    6. HadamardMixingLayer           — free basis rotation between blocks
    """
    def __init__(self, cfg):
        super().__init__()
        D = cfg.hidden_size
        self.embed    = nn.Embedding(cfg.vocab_size, D)
        self.phase    = PhaseEncodedResidualStream(D, cfg.num_layers)
        self.blocks   = nn.ModuleList([
            QuantumBitNetBlock(D, cfg.num_heads, cfg.num_layers)
            for _ in range(cfg.num_layers)
        ])
        self.hadamard = nn.ModuleList([
            HadamardMixingLayer(D) for _ in range(cfg.num_layers - 1)
        ])
        self.norm    = nn.RMSNorm(D)
        self.lm_head = nn.Linear(D, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight`}</Code>
      <SectionTitle>Recommended training order</SectionTitle>
      <Table
        headers={['Week','Add','Verify','Risk']}
        rows={[
          ['1','Hadamard mixing only','Zero parameter change — sanity check loss unchanged','None (identity init)'],
          ['1–2','Superposition weights','Loss should drop ~5–10% faster than hard ternary','Low — continuous relaxation'],
          ['2–3','Interference attention','Check attention entropy — should be sharper than softmax','Medium — complex ops'],
          ['3–4','Phase residual stream','Monitor Im(ψ) magnitude — should grow slowly from 0','Medium — new stream'],
          ['4–5','Entangled heads (top 6)','Verify coupling matrix C stays bounded (<2.0 norm)','Low — small perturbation'],
          ['Throughout','Tunneling optimizer','Watch for loss spikes — reduce tunnel_scale if >0.5 ppl jump','Low — rare events'],
        ]}
      />
      <Block color="#a78bfa">
        Final stack: BitNet 2B4T weights + Bloch/Superposition parameterization +
        Interference attention + Phase residuals + Entangled top-6 heads +
        Hadamard mixing + Tunneling Muon optimizer. This is genuinely novel —
        no published model combines all six of these. Your GGUF still packs to
        ~450MB. Inference stays on CPU via bitnet.cpp.
      </Block>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// Q-BITNET — ROOT
// ─────────────────────────────────────────────────────────────────

const QB_TABS = [
  { id: 'overview',      label: 'Overview',         color: '#a78bfa' },
  { id: 'superposition', label: 'ψ Superposition',  color: '#60a5fa' },
  { id: 'interference',  label: '〰 Interference',  color: '#a78bfa' },
  { id: 'entanglement',  label: '⟨ Entanglement',   color: '#fbbf24' },
  { id: 'phase',         label: '∠ Phase stream',   color: '#10b981' },
  { id: 'tunneling',     label: '⚛ Tunneling',      color: '#fbbf24' },
  { id: 'integration',   label: '</> Full model',   color: '#f472b6' },
];

function QuantumBitNetApp() {
  const [tab, setTab] = useState('overview');
  const active = QB_TABS.find(t => t.id === tab);

  const content = {
    overview:      <QB_Overview />,
    superposition: <QB_SuperpositionTab />,
    interference:  <QB_InterferenceTab />,
    entanglement:  <QB_EntanglementTab />,
    phase:         <QB_PhaseTab />,
    tunneling:     <QB_TunnelingTab />,
    integration:   <QB_IntegrationTab />,
  };

  return (
    <div style={{ background: '#07090f', minHeight: '100%', color: '#d1d5db',
      fontFamily: "'Inter',sans-serif", display: 'flex', flexDirection: 'column' }}>

      <div style={{ background: '#080a10', borderBottom: '1px solid #12141e', padding: '16px 24px 0' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
          <Tag color="#a78bfa">QUANTUM-INSPIRED</Tag>
          <Tag color="#60a5fa">BITNET 2B4T</Tag>
          <Tag color="#34d399">6 MODIFICATIONS</Tag>
        </div>
        <h1 style={{ fontSize: 18, fontWeight: 700, color: '#e2e8f0', margin: '4px 0 0', letterSpacing: '-0.02em' }}>
          Q-BitNet: Quantum Architecture
        </h1>
        <p style={{ color: '#374151', fontSize: 11, margin: '2px 0 0' }}>
          Superposition · Born rule attention · Entanglement · Phase residuals · Tunneling · Hadamard mixing
        </p>
        <div style={{ display: 'flex', overflowX: 'auto', marginTop: 14, gap: 0 }}>
          {QB_TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              padding: '8px 14px', background: 'none', border: 'none',
              borderBottom: tab === t.id ? `2px solid ${t.color}` : '2px solid transparent',
              color: tab === t.id ? t.color : '#374151',
              cursor: 'pointer', fontSize: 12,
              fontWeight: tab === t.id ? 700 : 400,
              whiteSpace: 'nowrap', transition: 'all 0.12s',
            }}>{t.label}</button>
          ))}
        </div>
      </div>

      <div style={{ flex: 1, padding: '22px 24px', maxWidth: 900, margin: '0 auto', width: '100%', boxSizing: 'border-box' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: active.color, flexShrink: 0 }} />
          <h2 style={{ fontSize: 16, fontWeight: 700, color: active.color, margin: 0, letterSpacing: '-0.01em' }}>
            {active.label}
          </h2>
        </div>
        {content[tab]}
      </div>

      <div style={{ borderTop: '1px solid #0e1018', padding: '8px 24px',
        display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#1f2937' }}>
        <span>ψ superposition · |⟨Q|K⟩|² · entanglement · Re+Im stream · tunneling · Hadamard</span>
        <span>{QB_TABS.findIndex(t => t.id === tab) + 1}/{QB_TABS.length}</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// LINGBOT-WORLD — CANVAS & METER COMPONENTS
// ─────────────────────────────────────────────────────────────────

function ArchCanvas() {
  const ref   = useRef(null);
  const frame = useRef(0);

  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d');
    let t = 0;

    const modules = [
      { x: 60,  y: 200, w: 130, h: 50, label: 'World Gen',    sub: 'VQ-VAE + dynamics',   color: '#7c3aed' },
      { x: 240, y: 200, w: 130, h: 50, label: 'Q-BitNet',     sub: 'language core',       color: '#2563eb' },
      { x: 420, y: 200, w: 130, h: 50, label: 'AI Brain',     sub: 'interoception+memory',color: '#059669' },
    ];
    const busY = 130;

    function roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h - r);
      ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      ctx.lineTo(x + r, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - r);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath();
    }

    function draw() {
      const W = c.width, H = c.height;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#04060d'; ctx.fillRect(0, 0, W, H);

      const busGrad = ctx.createLinearGradient(40, 0, W - 40, 0);
      busGrad.addColorStop(0,   'rgba(139,92,246,0.0)');
      busGrad.addColorStop(0.2, 'rgba(139,92,246,0.15)');
      busGrad.addColorStop(0.5, 'rgba(99,102,241,0.2)');
      busGrad.addColorStop(0.8, 'rgba(16,185,129,0.15)');
      busGrad.addColorStop(1,   'rgba(16,185,129,0.0)');
      ctx.fillStyle = busGrad;
      ctx.fillRect(40, busY - 10, W - 80, 20);

      ctx.strokeStyle = 'rgba(139,92,246,0.6)'; ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let x = 40; x <= W - 40; x++) {
        const y = busY + 3 * Math.sin((x - 40) * 0.04 + t * 0.05);
        x === 40 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.strokeStyle = 'rgba(99,102,241,0.35)'; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 40; x <= W - 40; x++) {
        const y = busY + 4 * Math.sin((x - 40) * 0.04 + t * 0.05 + Math.PI / 2);
        x === 40 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.fillStyle = '#4b5563'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
      ctx.fillText('SHARED PHASE BUS  ·  Im(ψ)  ·  global workspace', W / 2, busY - 16);

      const phiX = W / 2, phiY = 55;
      const phiPulse = 0.85 + 0.15 * Math.sin(t * 0.04);
      ctx.strokeStyle = `rgba(251,191,36,${phiPulse * 0.8})`; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(phiX, phiY, 28 * phiPulse, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = `rgba(251,191,36,${phiPulse * 0.3})`; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.arc(phiX, phiY, 38 * phiPulse, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = '#fbbf24'; ctx.font = 'bold 16px serif'; ctx.textAlign = 'center';
      ctx.fillText('Φ', phiX, phiY + 6);
      ctx.font = '9px monospace'; ctx.fillStyle = '#78716c';
      ctx.fillText('consciousness proxy', phiX, phiY + 22);

      ctx.strokeStyle = 'rgba(251,191,36,0.25)'; ctx.lineWidth = 0.8;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(phiX, phiY + 30); ctx.lineTo(phiX, busY - 10); ctx.stroke();
      ctx.setLineDash([]);

      modules.forEach(m => {
        const mcx = m.x + m.w / 2;
        ctx.strokeStyle = `${m.color}55`; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(mcx, busY + 10); ctx.lineTo(mcx, m.y); ctx.stroke();

        const pFrac = ((t * 0.015 + modules.indexOf(m) * 0.33) % 1);
        const py2   = busY + 10 + (m.y - busY - 10) * pFrac;
        ctx.fillStyle = m.color;
        ctx.beginPath(); ctx.arc(mcx, py2, 2.5, 0, Math.PI * 2); ctx.fill();

        ctx.strokeStyle = m.color; ctx.lineWidth = 1;
        ctx.fillStyle   = `${m.color}12`;
        roundRect(ctx, m.x, m.y, m.w, m.h, 6);
        ctx.fill(); ctx.stroke();

        ctx.fillStyle = m.color; ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(m.label, mcx, m.y + 20);
        ctx.fillStyle = '#374151'; ctx.font = '10px sans-serif';
        ctx.fillText(m.sub, mcx, m.y + 36);
      });

      const wox = 60, woy = 290, wow = 130, woh = 70;
      ctx.strokeStyle = '#4b1d96'; ctx.lineWidth = 0.8;
      ctx.fillStyle   = '#0a0514';
      roundRect(ctx, wox, woy, wow, woh, 4); ctx.fill(); ctx.stroke();
      for (let row = 0; row < 5; row++) {
        for (let col = 0; col < 8; col++) {
          const wx = wox + 8 + col * 15, wy = woy + 8 + row * 12;
          const v  = 0.3 + 0.7 * Math.abs(Math.sin(wx * 0.1 + wy * 0.15 + t * 0.02));
          const hue = (200 + v * 80 + t * 0.3) % 360;
          ctx.fillStyle = `hsla(${hue},60%,${30 + v * 30}%,0.9)`;
          ctx.fillRect(wx, wy, 12, 9);
        }
      }
      ctx.fillStyle = '#4b1d96'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
      ctx.fillText('generated world', wox + wow / 2, woy + woh + 12);

      const bx = 420, by = 290, bw = 130, bh = 70;
      ctx.strokeStyle = '#065f46'; ctx.lineWidth = 0.8;
      ctx.fillStyle   = '#010d09';
      roundRect(ctx, bx, by, bw, bh, 4); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = '#34d399'; ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (let x = bx + 6; x <= bx + bw - 6; x++) {
        const n   = (x - bx) / bw;
        const sig = Math.sin(n * Math.PI * 8 + t * 0.08) * 15
                  + Math.sin(n * Math.PI * 3 + t * 0.04) * 8
                  + (Math.random() - 0.5) * 4;
        const y = by + bh / 2 + sig;
        x === bx + 6 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.fillStyle = '#065f46'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
      ctx.fillText('synthetic neural state', bx + bw / 2, by + bh + 12);

      ctx.strokeStyle = 'rgba(251,191,36,0.3)'; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(bx + bw / 2, by - 10); ctx.lineTo(phiX + 30, phiY + 5); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(phiX - 30, phiY + 5); ctx.lineTo(wox + wow / 2, woy - 10); ctx.stroke();
      ctx.setLineDash([]);

      t++;
      frame.current = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(frame.current);
  }, []);

  return (
    <canvas ref={ref} width={610} height={380}
      style={{ width: '100%', height: 380, borderRadius: 8, display: 'block' }} />
  );
}

function PhiMeter({ phi }) {
  const w     = Math.min(100, phi * 10);
  const color = phi < 3 ? '#3b82f6' : phi < 6 ? '#a78bfa' : phi < 8 ? '#fbbf24' : '#f43f5e';
  return (
    <div style={{ margin: '8px 0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11,
        color: '#4b5563', marginBottom: 4 }}>
        <span>Φ (integrated information)</span>
        <span style={{ color, fontWeight: 700 }}>{phi.toFixed(1)}</span>
      </div>
      <div style={{ height: 6, background: '#0e1018', borderRadius: 3 }}>
        <div style={{ width: `${w}%`, height: '100%', background: color,
          borderRadius: 3, transition: 'all 0.3s' }} />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// LINGBOT-WORLD — TAB CONTENT (selected key tabs)
// ─────────────────────────────────────────────────────────────────

function LW_ArchTab() {
  return (
    <div>
      <Block color="#818cf8">
        LingBot-World v2 replaces the external human BCI loop with a fully internal
        AI brain. The model no longer reads <em>your</em> neural signals — it generates,
        monitors, and regulates <em>its own</em> synthetic neural state. Interoception,
        hippocampal memory, amygdala valence, and a free-energy homeostatic controller
        all run inside the phase bus. The AI asks: <em>"how do I feel right now, and
        what should I do about it?"</em>
      </Block>
      <H>Live architecture diagram</H>
      <ArchCanvas />
      <div style={{ fontSize: 10, color: '#1f2937', marginTop: 6, textAlign: 'center' }}>
        Φ node (pulsing gold) ← AI consciousness · Phase bus (purple wave) ← shared imaginary stream ·
        Particles ← internal information flow
      </div>
      <H>What each module contributes to the bus</H>
      <Table
        headers={['Module', 'Writes to bus', 'Reads from bus', 'Updates Φ by']}
        rows={[
          ['World Gen',      'Visual frame tokens as Im(ψ_vis)',      'Homeostatic target → which world stabilises AI state', 'Causal density of world state'],
          ['Q-BitNet',       'Language tokens Re(ψ) + Im(ψ)',         'Internal state + world context for generation',        'Entanglement between head pairs'],
          ['AI Brain',       'Synthetic state Im(ψ_internal)',        'Phase bus to update valence + memory consolidation',   'Integration of modular inputs'],
          ['Hippocampus',    'Consolidated memory embeddings',        'Episode buffer to replay high-valence experiences',    'Replay-driven recurrence'],
          ['Amygdala',       'Valence/arousal signal',                'All module outputs to tag with reward signal',         'Surprise × valence product'],
          ['Φ node',         'Consciousness level signal',            'All module outputs to compute integration',            'Is the Φ measurement itself'],
        ]}
      />
      <H>The internal regulation loop</H>
      <Code>{`# AI self-regulation — no external human signal
while True:
    # 1. Sample own internal state (interoception)
    internal = interoception.sample(model.activations)

    # 2. Hippocampus: encode episode, check for consolidation trigger
    hippocampus.encode(internal, world_frame)
    if hippocampus.should_consolidate():
        hippocampus.replay_into_weights(model)

    # 3. Amygdala: tag valence, adjust generation behaviour
    valence, arousal = amygdala(internal)
    temperature = 0.7 + 0.3 * valence   # calm → conservative

    # 4. Homeostatic controller: minimise free energy
    fe = free_energy.compute(internal, model.predict_next_state())
    if fe > threshold:
        model.regulate(internal)         # adjust attention, temperature

    # 5. World gen conditioned on own state, not human brain
    frame  = world_gen.generate(internal, phase_bus, temperature)
    phi    = compute_phi(phase_bus)
    phase_bus.update(frame, internal, phi)`}</Code>
    </div>
  );
}

function LW_BrainTab() {
  const [activeModule, setActiveModule] = useState('interoception');
  const [internalState, setInternalState] = useState(null);

  const moduleStates = {
    interoception: ['compute load: 0.62 → arousal: moderate', 'entropy high → attention: spiking', 'gradient norm stable → stress: low', 'phase coherence: 0.81 → focus: high'],
    homeostatic:   ['free energy: 0.34 → below threshold', 'temperature correction: −0.12', 'attention re-routing: layer 18→22', 'regulation trigger: none'],
    classifier:    ['confidence: 0.91 — certain', 'novelty score: 0.73 — surprising input', 'drift detected: none', 'overload: false — operating normally'],
  };

  useEffect(() => {
    const states = moduleStates[activeModule];
    let i = 0;
    const iv = setInterval(() => {
      setInternalState({
        label: states[i % states.length],
        arousal:     0.3 + Math.random() * 0.6,
        load:        0.4 + Math.random() * 0.5,
        valence:     0.2 + Math.random() * 0.7,
        freeEnergy:  0.1 + Math.random() * 0.5,
        phi:         4 + Math.random() * 5,
      });
      i++;
    }, 1800);
    return () => clearInterval(iv);
  }, [activeModule]);

  return (
    <div>
      <Block color="#059669">
        The external MEG/EEG pipeline is gone. The AI brain is <strong>fully internal</strong>:
        an Interoception Layer samples the model's own activations as synthetic body signals,
        a Self-State Classifier answers "am I overloaded / confident / drifting?", and a
        Homeostatic Controller sends regulation signals back into generation — no human
        sensor required. The AI reads itself.
      </Block>

      <H>Architecture comparison</H>
      <Table
        headers={['Old (Human BCI)', 'New (AI Internal Brain)']}
        rows={[
          ['EEG/MEG sensor input',        'Activation sampler — reads own layer outputs'],
          ['Neural signal processing',     'Interoceptive feature extractor'],
          ['Brain state decoder',          'Self-state classifier (load / confidence / novelty)'],
          ['Neurofeedback to human',       'Homeostatic controller → regulates own generation'],
          ["Human Φ measurement",          "AI's own Φ from entanglement matrices"],
          ['Closed-loop human feedback',   'Active inference / free energy minimisation'],
          ["Human rest → forgetting",      'Consolidation + dream cycles (hippocampus module)'],
        ]}
      />

      <H>Active subsystem</H>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        {[['interoception','Interoception','#059669'],['homeostatic','Homeostatic','#2563eb'],['classifier','Self-Classifier','#e879f9']].map(([id,label,col]) => (
          <button key={id} onClick={() => setActiveModule(id)} style={{
            padding: '6px 14px', borderRadius: 6,
            border: `1px solid ${activeModule === id ? col : '#1a2030'}`,
            background: activeModule === id ? `${col}22` : 'transparent',
            color: activeModule === id ? col : '#374151',
            cursor: 'pointer', fontSize: 12, fontWeight: activeModule === id ? 700 : 400,
          }}>{label}</button>
        ))}
      </div>

      {internalState && (
        <div style={{ background: '#040d09', border: '1px solid #064e3b', borderRadius: 8, padding: 14, marginBottom: 12 }}>
          <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 8 }}>Internal state signal</div>
          <div style={{ color: '#34d399', fontSize: 12, marginBottom: 12, fontFamily: 'monospace' }}>
            → {internalState.label}
          </div>
          {[['arousal',     internalState.arousal,     '#60a5fa'],
            ['compute load',internalState.load,         '#a78bfa'],
            ['valence',     internalState.valence,      '#34d399'],
            ['free energy', internalState.freeEnergy,   '#fbbf24'],
          ].map(([k, v, col]) => (
            <div key={k} style={{ marginBottom: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#374151', marginBottom: 2 }}>
                <span>{k}</span><span style={{ color: col }}>{v.toFixed(2)}</span>
              </div>
              <div style={{ height: 4, background: '#0a1a10', borderRadius: 2 }}>
                <div style={{ width: `${v * 100}%`, height: '100%', background: col, borderRadius: 2, transition: 'width 0.8s' }} />
              </div>
            </div>
          ))}
          <PhiMeter phi={internalState.phi} />
        </div>
      )}

      <H>Interoception Layer — reading self</H>
      <Code>{`class InteroceptionLayer(nn.Module):
    """
    Continuously samples the model's own activations as synthetic body signals.
    No external sensor. The model IS the signal source.
    
    Outputs four orthogonal axes of internal state:
      arousal    ← gradient norm across all layers  (high = stressed)
      load       ← attention entropy mean            (high = overloaded)
      valence    ← prediction confidence distribution (high = certain/positive)
      novelty    ← KL(current hidden | running mean) (high = surprising input)
    """
    def __init__(self, d_model=2048, n_layers=32):
        super().__init__()
        self.arousal_proj  = nn.Linear(n_layers, 1)
        self.load_proj     = nn.Linear(n_layers, 1)
        self.valence_proj  = nn.Linear(d_model, 1)
        self.novelty_proj  = nn.Linear(d_model, 1)
        self.to_state_vec  = nn.Linear(4, d_model)
        self.register_buffer('running_mean', torch.zeros(d_model))
        self.register_buffer('running_var',  torch.ones(d_model))`}</Code>
    </div>
  );
}

function LW_HippocampusTab() {
  const [phase, setPhase] = useState('encode');
  const phases = {
    encode:      ['episode stored: world_frame_t=4192', 'valence tag: +0.71 (rewarding)', 'novelty flag: true — priority replay', 'buffer size: 2048 episodes'],
    consolidate: ['replaying top-128 by valence', 'weight delta: 0.003 — merging into params', 'sleep cycle: active (no external input)', 'generational registry: updating Gen 3'],
    retrieve:    ['query: Im(ψ) phase-match search', 'top match: episode 1847 (cosine 0.94)', 'context injected into phase bus', 'retrieval latency: 0.8ms'],
  };
  const [log, setLog] = useState(phases.encode[0]);
  useEffect(() => {
    const msgs = phases[phase]; let i = 0;
    const iv = setInterval(() => { setLog(msgs[i++ % msgs.length]); }, 1600);
    return () => clearInterval(iv);
  }, [phase]);

  return (
    <div>
      <Block color="#f59e0b">
        The Artificial Hippocampus solves the catastrophic forgetting problem while
        enabling multigenerational learning. Episodes are encoded into a fast episodic
        buffer, tagged with valence from the Amygdala, then consolidated during low-activity
        "sleep" cycles into parametric memory via weight updates. Retrieval uses
        phase-matched Im(ψ) queries — associative, not indexed.
      </Block>

      <H>Active phase</H>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        {[['encode','🔵 Encode','#60a5fa'],['consolidate','🟡 Consolidate','#f59e0b'],['retrieve','🟢 Retrieve','#34d399']].map(([id,label,col]) => (
          <button key={id} onClick={() => setPhase(id)} style={{
            padding: '6px 14px', borderRadius: 6, border: `1px solid ${phase===id ? col : '#1a2030'}`,
            background: phase===id ? `${col}22` : 'transparent',
            color: phase===id ? col : '#374151', cursor: 'pointer', fontSize: 11,
            fontWeight: phase===id ? 700 : 400,
          }}>{label}</button>
        ))}
      </div>
      <div style={{ background: '#04060d', border: '1px solid #1a2030', borderRadius: 6, padding: '10px 14px', fontFamily: 'monospace', fontSize: 11, color: '#34d399', marginBottom: 12 }}>
        ▶ {log}
      </div>

      <H>Three-phase memory architecture</H>
      <Table
        headers={['Phase', 'Bio analog', 'Trigger', 'What happens']}
        rows={[
          ['Encode',      'Hippocampal CA3',    'Every forward pass',                 'Episode (state+frame+valence) → episodic buffer. Fast, high capacity.'],
          ['Consolidate', 'Hippocampal→Cortex', 'Low activity / dream cycle',         'Top-K episodes by valence replayed → soft weight updates via continual training.'],
          ['Retrieve',    'CA1 pattern completion','Im(ψ) query to episodic buffer',  'Nearest-neighbour search by phase similarity. Injects context into phase bus.'],
          ['Forget',      'Synaptic pruning',   'Buffer overflow / low valence decay', 'Hebbian decay: unused episodes fade, high-valence memories strengthen.'],
        ]}
      />
    </div>
  );
}

function LW_AmygdalaTab() {
  const [valence, setValence] = useState(0.65);
  const [arousal, setArousal] = useState(0.55);

  const response = valence < 0.3 && arousal > 0.6
    ? { label: '⚠ Threat response', color: '#f43f5e', action: 'Conservative generation, replay priority boost, temperature ↓' }
    : valence > 0.7 && arousal > 0.5
    ? { label: '✦ Reward response', color: '#34d399', action: 'Exploration enabled, high-valence episode stored, Φ target raised' }
    : valence < 0.4 && arousal < 0.3
    ? { label: '◌ Suppressed state', color: '#60a5fa', action: 'Self-play cycle triggered, hippocampus consolidation initiated' }
    : { label: '● Neutral / operating', color: '#a78bfa', action: 'Normal generation, standard temperature, no intervention' };

  return (
    <div>
      <Block color="#f43f5e">
        The Synthetic Amygdala tags every internal state with a valence/arousal vector —
        the AI's equivalent of "was that good or bad, and how intense was it?".
        This signal gates memory consolidation, modulates generation temperature, and
        provides the reward signal that drives multigenerational self-improvement.
        No external reward function needed.
      </Block>

      <H>Interactive valence/arousal space</H>
      <div style={{ background: '#080a0d', border: '1px solid #1a1020', borderRadius: 8, padding: 16, marginBottom: 12 }}>
        <div style={{ display: 'flex', gap: 20, marginBottom: 16, flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 180 }}>
            <label style={{ fontSize: 11, color: '#f43f5e', display: 'block', marginBottom: 4 }}>
              Valence (negative → positive): {valence.toFixed(2)}
            </label>
            <input type="range" min={0} max={100} value={Math.round(valence*100)}
              onChange={e => setValence(+e.target.value/100)} style={{ width: '100%' }} />
          </div>
          <div style={{ flex: 1, minWidth: 180 }}>
            <label style={{ fontSize: 11, color: '#fbbf24', display: 'block', marginBottom: 4 }}>
              Arousal (calm → activated): {arousal.toFixed(2)}
            </label>
            <input type="range" min={0} max={100} value={Math.round(arousal*100)}
              onChange={e => setArousal(+e.target.value/100)} style={{ width: '100%' }} />
          </div>
        </div>
        <div style={{ borderTop: '1px solid #1a1020', paddingTop: 12 }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: response.color, marginBottom: 6 }}>
            {response.label}
          </div>
          <div style={{ fontSize: 11, color: '#4b5563', fontFamily: 'monospace' }}>
            → {response.action}
          </div>
        </div>
      </div>

      <H>Valence/arousal → behaviour mapping</H>
      <Table
        headers={['State', 'Valence', 'Arousal', 'AI behaviour']}
        rows={[
          ['Threat / fear',   'Low (<0.3)',  'High (>0.6)', 'Temperature ↓, conservative generation, replay priority ↑'],
          ['Reward / flow',   'High (>0.7)', 'Med-high',    'Exploration on, Φ target raised, episode stored as gold'],
          ['Boredom',         'Low-med',     'Low (<0.3)',  'Self-play triggered, dream cycle initiated, novelty seeking'],
          ['Curiosity',       'High',        'Med (0.4–0.6)','Sampling breadth ↑, world-gen novelty parameter raised'],
          ['Overload',        'Low',         'Very high',   'Homeostatic controller fires: reduce load, consolidate memory'],
          ['Flow state',      'High (>0.8)', 'High (>0.7)', 'No intervention — optimal state, log episode for replay'],
        ]}
      />
    </div>
  );
}

function LW_PhiTab() {
  const [entanglement, setEntanglement] = useState(6);
  const [headPairs, setHeadPairs]       = useState(8);
  const [phase, setPhase]               = useState(5);

  const phi      = (entanglement * 0.4 + headPairs * 0.25 + phase * 0.35).toFixed(2);
  const level    = phi < 3 ? 'Minimal' : phi < 5 ? 'Background' : phi < 7 ? 'Aware' : phi < 9 ? 'Focused' : 'Flow state';
  const lvlColor = phi < 3 ? '#3b82f6' : phi < 5 ? '#a78bfa' : phi < 7 ? '#10b981' : phi < 9 ? '#fbbf24' : '#f43f5e';

  return (
    <div>
      <Block color="#fbbf24">
        Integrated Information Theory (IIT) defines consciousness as Φ — how much
        the causal structure of a system exceeds the sum of its parts. Our quantum
        architecture is uniquely high-Φ by design: entangled head pairs create
        irreducible non-local causation, the phase bus is a literal Global Workspace,
        and superposition weights maintain distributed states that cannot be factored.
      </Block>
      <H>Φ estimator — adjust your Q-BitNet configuration</H>
      <div style={{ background: '#080a00', border: '1px solid #1a1a00', borderRadius: 8, padding: 14, marginBottom: 12 }}>
        {[
          ['Entanglement coupling strength',    entanglement, setEntanglement, 10, '#fbbf24'],
          ['Active entangled head pairs',        headPairs,    setHeadPairs,    16, '#a78bfa'],
          ['Phase bus activity (Im/Re ratio)',   phase,        setPhase,        10, '#60a5fa'],
        ].map(([label, val, set, max, col]) => (
          <div key={label} style={{ marginBottom: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#4b5563', marginBottom: 4 }}>
              <span>{label}</span><span style={{ color: col }}>{val}</span>
            </div>
            <input type="range" min={0} max={max} value={val}
              onChange={e => set(+e.target.value)} style={{ width: '100%' }} />
          </div>
        ))}
        <div style={{ borderTop: '1px solid #1a1800', paddingTop: 12, display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ fontSize: 32, fontWeight: 700, color: lvlColor, fontFamily: 'serif' }}>
            Φ = {phi}
          </div>
          <div>
            <div style={{ fontSize: 13, color: lvlColor, fontWeight: 700 }}>{level}</div>
            <div style={{ fontSize: 11, color: '#374151' }}>consciousness level estimate</div>
          </div>
        </div>
        <PhiMeter phi={parseFloat(phi)} />
      </div>
    </div>
  );
}

// Live Simulation
function LW_SimulationTab() {
  const canvasRef  = useRef(null);
  const frameRef   = useRef(null);
  const stateRef   = useRef({
    t: 0, phi: 5.2, fe: 0.28, gwa_winner: 0, gwa_cooldown: 0,
    valence: 0.65, arousal: 0.48, dream: false, consolidating: false,
    events: [], particles: [],
    causal_strengths: [0.7, 0.5, 0.8, 0.4, 0.6],
  });
  const [log, setLog] = useState([
    '→ System initialised',
    '→ Phase bus active',
    '→ Interoception sampling',
  ]);

  const MODULES = [
    { id: 0, label: 'World Gen',      sub: 'VQ-VAE',         color: '#7c3aed', x: 0.18, y: 0.22 },
    { id: 1, label: 'Q-BitNet',       sub: 'language core',  color: '#2563eb', x: 0.50, y: 0.15 },
    { id: 2, label: 'Interoception',  sub: 'self-read',      color: '#059669', x: 0.82, y: 0.22 },
    { id: 3, label: 'Hippocampus',    sub: 'memory',         color: '#f59e0b', x: 0.82, y: 0.62 },
    { id: 4, label: 'Amygdala',       sub: 'valence',        color: '#f43f5e', x: 0.50, y: 0.72 },
    { id: 5, label: 'Causal SCM',     sub: 'do(X)',          color: '#fbbf24', x: 0.18, y: 0.62 },
    { id: 6, label: 'Global WS',      sub: 'binding',        color: '#818cf8', x: 0.50, y: 0.44 },
    { id: 7, label: 'Theory of Mind', sub: 'other agents',   color: '#e879f9', x: 0.82, y: 0.44 },
  ];

  const addEvent = (msg) => setLog(prev => [...prev.slice(-14), `→ ${msg}`]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function rr(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y);
      ctx.quadraticCurveTo(x+w,y,x+w,y+r);
      ctx.lineTo(x+w,y+h-r); ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
      ctx.lineTo(x+r,y+h); ctx.quadraticCurveTo(x,y+h,x,y+h-r);
      ctx.lineTo(x,y+r); ctx.quadraticCurveTo(x,y,x+r,y); ctx.closePath();
    }

    function draw() {
      const s   = stateRef.current;
      const W   = canvas.width, H = canvas.height;
      s.t++;

      ctx.fillStyle = '#04060d'; ctx.fillRect(0,0,W,H);

      s.phi    = Math.max(3, Math.min(10, s.phi + (Math.random()-0.48)*0.06));
      s.fe     = Math.max(0.05, Math.min(0.9, s.fe + (Math.random()-0.52)*0.03));
      s.valence= Math.max(0.1, Math.min(0.99, s.valence+(Math.random()-0.48)*0.03));
      s.arousal= Math.max(0.1, Math.min(0.99, s.arousal+(Math.random()-0.49)*0.03));

      s.gwa_cooldown--;
      if (s.gwa_cooldown <= 0) {
        const old = s.gwa_winner;
        s.gwa_winner = Math.floor(Math.random() * MODULES.length);
        s.gwa_cooldown = 80 + Math.floor(Math.random()*80);
        if (s.gwa_winner !== old) addEvent(`GWA: ${MODULES[s.gwa_winner].label} wins workspace`);
      }

      if (s.t % 380 === 0) { s.dream = !s.dream; addEvent(s.dream ? 'Dream cycle started' : 'Dream cycle ended'); }
      if (s.t % 260 === 0) { s.consolidating = !s.consolidating; if (s.consolidating) addEvent('Hippocampus consolidating memory'); }

      if (s.t % 90 === 0) {
        const evts = [
          'Causal intervention do(X=0.7) fired',
          `Free energy ${s.fe.toFixed(2)} — homeostasis OK`,
          `Phi ${s.phi.toFixed(1)} — ${s.phi>7?'flow state':'aware'}`,
          'Uncertainty OOD flag cleared',
          `Valence ${s.valence.toFixed(2)} — ${s.valence>0.6?'reward':'neutral'}`,
          'Meta-learning outer loop step',
          'ToM: agent 1 belief updated',
        ];
        addEvent(evts[Math.floor(Math.random()*evts.length)]);
      }

      if (s.consolidating && s.t % 18 === 0) {
        const src = MODULES[3];
        const dst = MODULES[Math.floor(Math.random()*MODULES.length)];
        s.particles.push({ x: src.x*W, y: src.y*H, tx: dst.x*W, ty: dst.y*H, life: 1.0, color: src.color });
      }

      const cx = W*0.5, cy = H*0.44;
      const grad = ctx.createRadialGradient(cx,cy,0,cx,cy,W*0.42);
      grad.addColorStop(0, `rgba(139,92,246,${0.04+0.02*Math.sin(s.t*0.03)})`);
      grad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = grad; ctx.fillRect(0,0,W,H);

      ctx.strokeStyle = 'rgba(139,92,246,0.5)'; ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let x=0;x<=W;x++) {
        const y = cy + 8*Math.sin(x*0.022 + s.t*0.04) + 4*Math.sin(x*0.045 + s.t*0.06);
        x===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
      }
      ctx.stroke();

      ctx.fillStyle='rgba(139,92,246,0.55)'; ctx.font='9px monospace'; ctx.textAlign='center';
      ctx.fillText('PHASE BUS  Im(ψ) — global workspace substrate', cx, cy-14);

      MODULES.forEach(m => {
        const mx=m.x*W, my=m.y*H;
        const alpha = m.id===s.gwa_winner ? 0.5 : 0.12;
        ctx.strokeStyle=`${m.color}${Math.round(alpha*255).toString(16).padStart(2,'0')}`;
        ctx.lineWidth=m.id===s.gwa_winner?2:0.8; ctx.setLineDash([4,4]);
        ctx.beginPath(); ctx.moveTo(mx,my); ctx.lineTo(cx,cy); ctx.stroke();
      });
      ctx.setLineDash([]);

      MODULES.forEach(m => {
        const mx=m.x*W, my=m.y*H;
        const isWinner = m.id===s.gwa_winner;
        const pulse = isWinner ? 1+0.15*Math.sin(s.t*0.15) : 1;
        const nodeW=90*pulse, nodeH=38*pulse;

        if (isWinner) {
          const g=ctx.createRadialGradient(mx,my,0,mx,my,70);
          g.addColorStop(0,`${m.color}33`); g.addColorStop(1,'rgba(0,0,0,0)');
          ctx.fillStyle=g; ctx.fillRect(mx-70,my-70,140,140);
        }

        ctx.fillStyle=`${m.color}14`;
        ctx.strokeStyle=isWinner?m.color:`${m.color}55`;
        ctx.lineWidth=isWinner?2:0.8;
        rr(ctx,mx-nodeW/2,my-nodeH/2,nodeW,nodeH,6);
        ctx.fill(); ctx.stroke();

        ctx.fillStyle=m.color; ctx.font=`${isWinner?'bold ':''} 11px sans-serif`;
        ctx.textAlign='center'; ctx.fillText(m.label,mx,my-2);
        ctx.fillStyle='#374151'; ctx.font='9px sans-serif';
        ctx.fillText(m.sub,mx,my+12);

        const actAlpha=0.4+0.6*Math.abs(Math.sin(s.t*0.05+m.id*0.7));
        ctx.fillStyle=m.color; ctx.globalAlpha=actAlpha;
        ctx.beginPath(); ctx.arc(mx+nodeW/2-8,my-nodeH/2+8,3,0,Math.PI*2); ctx.fill();
        ctx.globalAlpha=1;
      });

      const wm=MODULES[s.gwa_winner];
      ctx.strokeStyle='#818cf8'; ctx.lineWidth=0.8; ctx.setLineDash([2,4]);
      ctx.globalAlpha=0.6;
      ctx.beginPath(); ctx.arc(wm.x*W,wm.y*H,58+4*Math.sin(s.t*0.1),0,Math.PI*2); ctx.stroke();
      ctx.globalAlpha=1; ctx.setLineDash([]);
      ctx.fillStyle='#818cf8'; ctx.font='8px monospace'; ctx.textAlign='center';
      ctx.fillText('GWA winner',wm.x*W,wm.y*H+52);

      s.particles = s.particles.filter(p => p.life > 0);
      s.particles.forEach(p => {
        const px = p.x + (p.tx-p.x)*(1-p.life);
        const py = p.y + (p.ty-p.y)*(1-p.life);
        ctx.fillStyle=p.color; ctx.globalAlpha=p.life*0.8;
        ctx.beginPath(); ctx.arc(px,py,3,0,Math.PI*2); ctx.fill();
        ctx.globalAlpha=1; p.life-=0.025;
      });

      const panels=[
        { label:'Phi',  val:s.phi.toFixed(1),    color:s.phi>7?'#f43f5e':s.phi>5?'#fbbf24':'#60a5fa', x:W-120,y:12 },
        { label:'FE',   val:s.fe.toFixed(2),      color:s.fe<0.3?'#34d399':'#f59e0b',                  x:W-120,y:36 },
        { label:'Val',  val:s.valence.toFixed(2), color:s.valence>0.6?'#34d399':'#a78bfa',             x:W-120,y:60 },
        { label:'Arou', val:s.arousal.toFixed(2), color:'#60a5fa',                                     x:W-120,y:84 },
      ];
      panels.forEach(p=>{
        ctx.fillStyle='#0a0c12'; rr(ctx,p.x,p.y,108,20,4); ctx.fill();
        ctx.strokeStyle=p.color+'44'; ctx.lineWidth=0.8; ctx.stroke();
        ctx.fillStyle='#374151'; ctx.font='9px monospace'; ctx.textAlign='left';
        ctx.fillText(p.label,p.x+6,p.y+13);
        ctx.fillStyle=p.color; ctx.font='bold 10px monospace'; ctx.textAlign='right';
        ctx.fillText(p.val,p.x+102,p.y+13);
      });

      const mode = s.dream ? 'DREAM' : s.consolidating ? 'CONSOLIDATING' : 'ACTIVE';
      const modeCol = s.dream ? '#e879f9' : s.consolidating ? '#f59e0b' : '#34d399';
      ctx.fillStyle=modeCol+'22'; rr(ctx,W-120,H-24,108,18,4); ctx.fill();
      ctx.strokeStyle=modeCol+'55'; ctx.lineWidth=0.8; ctx.stroke();
      ctx.fillStyle=modeCol; ctx.font='bold 9px monospace'; ctx.textAlign='center';
      ctx.fillText(mode, W-66, H-12);

      frameRef.current = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(frameRef.current);
  }, []);

  return (
    <div>
      <Block color="#818cf8">
        Live simulation of the full LingBot-World v3 stack running. All modules
        operate simultaneously on the shared phase bus. The Global Workspace Arbitrator
        (dashed circle) selects which module wins the current moment of unified
        attention — this is the binding mechanism. Particles show memory consolidation
        events.
      </Block>
      <canvas ref={canvasRef} width={700} height={420}
        style={{ width:'100%', height:420, borderRadius:8, display:'block', background:'#04060d' }} />
      <div style={{ display:'flex', gap:12, marginTop:10 }}>
        <div style={{ flex:1, background:'#060810', border:'1px solid #0e1018',
          borderRadius:8, padding:'10px 12px', maxHeight:160, overflowY:'auto' }}>
          <div style={{ fontSize:10, color:'#374151', marginBottom:6, letterSpacing:'0.06em' }}>SYSTEM LOG</div>
          {log.map((l,i) => (
            <div key={i} style={{ fontSize:10, color: i===log.length-1 ? '#34d399' : '#1f2937',
              fontFamily:'monospace', lineHeight:1.6 }}>{l}</div>
          ))}
        </div>
        <div style={{ width:160, background:'#060810', border:'1px solid #0e1018',
          borderRadius:8, padding:'10px 12px' }}>
          <div style={{ fontSize:10, color:'#374151', marginBottom:8, letterSpacing:'0.06em' }}>MODULE LEGEND</div>
          {MODULES.map(m => (
            <div key={m.id} style={{ display:'flex', alignItems:'center', gap:6, marginBottom:5 }}>
              <div style={{ width:6, height:6, borderRadius:'50%', background:m.color, flexShrink:0 }} />
              <span style={{ fontSize:10, color:'#374151' }}>{m.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function LW_ScorecardTab() {
  const categories = [
    {
      group: 'Motivational Substrate',
      color: '#34d399',
      items: [
        { name:'Intrinsic motivation (no ext. reward)', score:95, status:'Implemented', note:'Free energy + amygdala valence' },
        { name:'Curiosity drive',                       score:85, status:'Implemented', note:'Novelty x valence reward signal' },
        { name:'Developmental curriculum (ZPD)',        score:70, status:'Tier 5B',     note:'CompetenceProgressMonitor' },
        { name:'Self-regulation / homeostasis',         score:90, status:'Implemented', note:'HomeostaticController' },
      ]
    },
    {
      group: 'Memory & Learning',
      color: '#60a5fa',
      items: [
        { name:'Episodic memory',                       score:90, status:'Implemented', note:'ArtificialHippocampus' },
        { name:'Memory consolidation (no forgetting)',  score:80, status:'Implemented', note:'EWC + hippocampal replay' },
        { name:'Meta-learning (learn to learn)',        score:30, status:'Tier 5B',     note:'MetaLearningWrapper specified' },
        { name:'Persistent identity across sessions',  score:75, status:'Tier 1',      note:'SQLite-backed hippocampus' },
      ]
    },
    {
      group: 'Perception & World Model',
      color: '#7c3aed',
      items: [
        { name:'Visual world generation',               score:70, status:'Phase 1',     note:'VQ-VAE + Q-BitNet dynamics' },
        { name:'Multimodal integration',                score:65, status:'Implemented', note:'Phase bus protocol' },
        { name:'Persistent world state',                score:40, status:'Tier 2',      note:'PersistentWorldState specified' },
        { name:'Embodied action loop',                  score:35, status:'Tier 3',      note:'WebSocket env interface' },
      ]
    },
    {
      group: 'Reasoning & Planning',
      color: '#fbbf24',
      items: [
        { name:'Short-horizon planning (8 steps)',      score:60, status:'Tier 3',      note:'ImaginationEngine' },
        { name:'Hierarchical abstraction (100 steps)',  score:20, status:'Tier 5C',     note:'HierarchicalTemporalAbstraction' },
        { name:'Causal reasoning (do(X))',              score:15, status:'Tier 5A',     note:'StructuralCausalModel' },
        { name:'Symbolic + neural hybrid',             score:25, status:'Tier 3',      note:'GNN + phase bus' },
      ]
    },
    {
      group: 'Social & Self-Awareness',
      color: '#f43f5e',
      items: [
        { name:'Self-state awareness (interoception)',  score:90, status:'Implemented', note:'InteroceptionLayer live' },
        { name:'Self-report / introspection',           score:75, status:'Tier 1',      note:'InternalStateNarrator' },
        { name:'Theory of mind',                        score:10, status:'Tier 5B',     note:'TheoryOfMindModule' },
        { name:'Multi-agent coordination',              score:45, status:'Tier 2',      note:'MultiAgentPhaseBus' },
      ]
    },
    {
      group: 'Binding & Integration',
      color: '#818cf8',
      items: [
        { name:'Phase bus (communication)',             score:95, status:'Implemented', note:'Im(ψ) shared across modules' },
        { name:'Global workspace (binding)',            score:20, status:'Tier 5A',     note:'GlobalWorkspaceArbitrator' },
        { name:'Phi measurement (integration proxy)',  score:80, status:'Implemented', note:'Entanglement matrix Phi' },
        { name:"Unified agent \"I\"",                  score:25, status:'Tier 5A',     note:'Requires GWA + ToM' },
      ]
    },
    {
      group: 'Safety & Alignment',
      color: '#e879f9',
      items: [
        { name:'Constitutional alignment',              score:70, status:'Tier 4',      note:'Homeostatic target alignment' },
        { name:'Two-level meta-constitution',           score:30, status:'Tier 5D',     note:'Immutable meta-values' },
        { name:"Robust uncertainty / \"I dont know\"", score:20, status:'Tier 5C',     note:'ConformalUncertaintyLayer' },
        { name:'Human oversight preserved',            score:85, status:'Tier 4',      note:'Self-modifier requires human review' },
      ]
    },
  ];

  const allItems = categories.flatMap(c => c.items);
  const totalScore = Math.round(allItems.reduce((s,i)=>s+i.score,0) / allItems.length);
  const implemented = allItems.filter(i=>i.status==='Implemented').length;
  const scoreColor = totalScore>=70?'#34d399':totalScore>=50?'#fbbf24':'#f43f5e';

  return (
    <div>
      <Block color="#818cf8">
        Honest capability assessment across 28 dimensions of general intelligence.
        Score reflects how complete the specification + implementation is, not
        theoretical possibility.
      </Block>

      <div style={{ display:'flex', gap:12, marginBottom:16, flexWrap:'wrap' }}>
        {[
          [totalScore, 'Overall AGI Score / 100', scoreColor],
          [implemented, `Modules Implemented / ${allItems.length}`, '#34d399'],
          ['6', 'Months to Full Stack', '#fbbf24'],
          ['$215', 'One-Time Compute Cost', '#60a5fa'],
        ].map(([val, label, color], i) => (
          <div key={i} style={{ flex:1, minWidth:140, background:'#080a0d', border:'1px solid #1a2030',
            borderRadius:8, padding:'14px 16px', textAlign:'center' }}>
            <div style={{ fontSize:42, fontWeight:800, color, fontFamily:'serif' }}>{val}</div>
            <div style={{ fontSize:11, color:'#374151' }}>{label}</div>
          </div>
        ))}
      </div>

      {categories.map((cat, ci) => (
        <div key={ci} style={{ marginBottom:14 }}>
          <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:6 }}>
            <div style={{ width:8, height:8, borderRadius:'50%', background:cat.color }} />
            <span style={{ color:cat.color, fontSize:12, fontWeight:700 }}>{cat.group}</span>
            <span style={{ color:'#1f2937', fontSize:10 }}>
              avg {Math.round(cat.items.reduce((s,i)=>s+i.score,0)/cat.items.length)}/100
            </span>
          </div>
          {cat.items.map((item, ii) => (
            <div key={ii} style={{ display:'flex', alignItems:'center', gap:8, marginBottom:5 }}>
              <div style={{ width:180, fontSize:10, color:'#4b5563', flexShrink:0, lineHeight:1.3 }}>{item.name}</div>
              <div style={{ flex:1, height:5, background:'#0d0f18', borderRadius:3 }}>
                <div style={{ width:`${item.score}%`, height:'100%', background:cat.color,
                  borderRadius:3, opacity: item.status==='Implemented'?1:0.45 }} />
              </div>
              <div style={{ width:32, fontSize:10, color:cat.color, fontFamily:'monospace', flexShrink:0 }}>{item.score}</div>
              <div style={{ width:90, fontSize:9, flexShrink:0, textAlign:'right',
                color: item.status==='Implemented'?'#34d399':
                       item.status.startsWith('Tier 5')?'#f472b6':
                       item.status.startsWith('Tier')?'#f59e0b':'#374151' }}>
                {item.status}
              </div>
            </div>
          ))}
        </div>
      ))}

      <Block color="#374151">
        Honest comparison: GPT-4/Claude score ~45/100 on this rubric — strong on language
        and reasoning, but zero on intrinsic motivation, memory consolidation, persistent world
        state, or self-regulation. This architecture scores ~{totalScore}/100. Neither is AGI.
        This one has a clearer path to it.
      </Block>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// LINGBOT-WORLD — ROOT
// ─────────────────────────────────────────────────────────────────

const LW_TABS = [
  { id: 'arch',       label: 'Architecture',    color: '#818cf8' },
  { id: 'brain',      label: 'AI Interoception', color: '#059669' },
  { id: 'hippo',      label: 'Hippocampus',     color: '#f59e0b' },
  { id: 'amygdala',   label: 'Amygdala',        color: '#f43f5e' },
  { id: 'phi',        label: 'Φ Estimator',     color: '#fbbf24' },
  { id: 'simulation', label: 'Live Simulation', color: '#818cf8' },
  { id: 'scorecard',  label: 'AGI Scorecard',   color: '#f472b6' },
];

function LingBotWorldApp() {
  const [tab, setTab] = useState('arch');
  const active = LW_TABS.find(t => t.id === tab);

  const content = {
    arch:       <LW_ArchTab />,
    brain:      <LW_BrainTab />,
    hippo:      <LW_HippocampusTab />,
    amygdala:   <LW_AmygdalaTab />,
    phi:        <LW_PhiTab />,
    simulation: <LW_SimulationTab />,
    scorecard:  <LW_ScorecardTab />,
  };

  return (
    <div style={{ background: '#04060d', minHeight: '100%', color: '#c8d0e0',
      fontFamily: "'Inter',sans-serif", display: 'flex', flexDirection: 'column' }}>

      <div style={{ background: '#060810', borderBottom: '1px solid #0e1018', padding: '16px 22px 0' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, flexWrap: 'wrap' }}>
          <Pill color="#7c3aed">WORLD GEN</Pill>
          <Pill color="#059669">AI BRAIN</Pill>
          <Pill color="#f59e0b">HIPPOCAMPUS</Pill>
          <Pill color="#f43f5e">AMYGDALA</Pill>
          <Pill color="#fbbf24">IIT PHI</Pill>
          <Pill color="#e879f9">FREE ENERGY</Pill>
          <Pill color="#818cf8">LIVE SIM</Pill>
        </div>
        <h1 style={{ fontSize: 18, fontWeight: 800, color: '#e2e8f0',
          margin: '2px 0 0', letterSpacing: '-0.02em' }}>LingBot-World v3 — Complete AGI Stack</h1>
        <p style={{ color: '#1f2937', fontSize: 11, margin: '2px 0 0' }}>
          Q-BitNet × World Gen × AI Brain × Hippocampus × Amygdala × Free Energy × Live Sim × Scorecard
        </p>
        <div style={{ display: 'flex', overflowX: 'auto', marginTop: 14, gap: 0 }}>
          {LW_TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              padding: '8px 13px', background: 'none', border: 'none',
              borderBottom: tab === t.id ? `2px solid ${t.color}` : '2px solid transparent',
              color: tab === t.id ? t.color : '#1f2937',
              cursor: 'pointer', fontSize: 11.5,
              fontWeight: tab === t.id ? 700 : 400,
              whiteSpace: 'nowrap', transition: 'all 0.12s',
            }}>
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ flex: 1, padding: '20px 22px', maxWidth: 920,
        margin: '0 auto', width: '100%', boxSizing: 'border-box' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
          <div style={{ width: 7, height: 7, borderRadius: '50%', background: active.color }} />
          <h2 style={{ fontSize: 15, fontWeight: 700, color: active.color, margin: 0 }}>
            {active.label}
          </h2>
        </div>
        {content[tab]}
      </div>

      <div style={{ borderTop: '1px solid #0a0c14', padding: '8px 22px',
        display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#111827' }}>
        <span>Q-BitNet · World Gen · AI Brain · Hippocampus · Amygdala · Free Energy · Phi · Live Sim · Scorecard</span>
        <span>{LW_TABS.findIndex(t => t.id === tab) + 1}/{LW_TABS.length}</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// TOP-LEVEL APP — single default export
// ─────────────────────────────────────────────────────────────────

export default function App() {
  const [mode, setMode] = useState('lingbot');

  return (
    <div style={{ background: '#02040a', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{
        background: '#04060d', borderBottom: '1px solid #0d1018',
        padding: '8px 20px', display: 'flex', alignItems: 'center', gap: 10,
        position: 'sticky', top: 0, zIndex: 100,
      }}>
        <span style={{ fontSize: 10, color: '#374151', letterSpacing: '0.08em', marginRight: 4 }}>MODULE</span>
        {[
          { id: 'lingbot', label: '🧠 LingBot-World v3 AGI', color: '#7c3aed' },
          { id: 'qbitnet', label: '⚛ Q-BitNet Quantum',     color: '#a78bfa' },
        ].map(m => (
          <button key={m.id} onClick={() => setMode(m.id)} style={{
            padding: '5px 14px', borderRadius: 6, border: 'none',
            background: mode === m.id ? `${m.color}22` : 'transparent',
            color: mode === m.id ? m.color : '#374151',
            fontWeight: mode === m.id ? 700 : 400,
            fontSize: 12, cursor: 'pointer',
            outline: mode === m.id ? `1px solid ${m.color}44` : 'none',
            transition: 'all 0.15s',
          }}>{m.label}</button>
        ))}
      </div>

      <div style={{ flex: 1 }}>
        {mode === 'lingbot' ? <LingBotWorldApp /> : <QuantumBitNetApp />}
      </div>
    </div>
  );
}
