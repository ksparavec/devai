// Dev AI Lab docs — interactive widgets
// 1. Float-format bit-level explorer
// 2. Attention heatmap

(function () {
  'use strict';

  // ============================================================
  // 1. FLOAT FORMAT EXPLORER
  // ============================================================

  // Format specs. Each: [sign_bits, exp_bits, mant_bits, bias, label]
  const FORMATS = {
    fp32:    { sign: 1, exp: 8, mant: 23, bias: 127,  label: 'FP32',     bytes: 4 },
    fp16:    { sign: 1, exp: 5, mant: 10, bias: 15,   label: 'FP16',     bytes: 2 },
    bf16:    { sign: 1, exp: 8, mant: 7,  bias: 127,  label: 'BF16',     bytes: 2 },
    fp8e4m3: { sign: 1, exp: 4, mant: 3,  bias: 7,    label: 'FP8 E4M3', bytes: 1 },
    fp8e5m2: { sign: 1, exp: 5, mant: 2,  bias: 15,   label: 'FP8 E5M2', bytes: 1 },
    fp4e2m1: { sign: 1, exp: 2, mant: 1,  bias: 1,    label: 'FP4 E2M1', bytes: 0.5 }
  };

  // Decode an integer value (sign-exp-mant packed) into a float, given a format
  function decode(bits, fmt) {
    const total = fmt.sign + fmt.exp + fmt.mant;
    const sign = (bits >> (fmt.exp + fmt.mant)) & 1;
    const expRaw = (bits >> fmt.mant) & ((1 << fmt.exp) - 1);
    const mant = bits & ((1 << fmt.mant) - 1);
    const expMax = (1 << fmt.exp) - 1;

    // Special: all-ones exponent
    if (expRaw === expMax && fmt.exp > 2) {
      if (mant === 0) return { value: sign ? -Infinity : Infinity, repr: sign ? '-∞' : '+∞', kind: 'inf' };
      return { value: NaN, repr: 'NaN', kind: 'nan' };
    }

    // Denormal (subnormal): exp == 0
    let value, mantFloat;
    if (expRaw === 0) {
      if (mant === 0) return { value: sign ? -0 : 0, repr: sign ? '-0' : '+0', kind: 'zero' };
      mantFloat = mant / (1 << fmt.mant);
      value = mantFloat * Math.pow(2, 1 - fmt.bias);
    } else {
      mantFloat = 1 + mant / (1 << fmt.mant);
      value = mantFloat * Math.pow(2, expRaw - fmt.bias);
    }
    if (sign) value = -value;

    return {
      value,
      repr: formatNum(value),
      mantFloat,
      expRaw,
      expBiased: expRaw - fmt.bias,
      sign,
      kind: 'normal'
    };
  }

  function formatNum(v) {
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 0.001 && a < 100000) return v.toPrecision(7).replace(/\.?0+$/, '');
    return v.toExponential(4);
  }

  function buildFormatWidget(container) {
    const fmtKey = container.dataset.format || 'fp16';
    container.innerHTML = '';

    const fmt = FORMATS[fmtKey];

    // Title
    const title = document.createElement('div');
    title.className = 'widget-title';
    title.textContent = 'Bit-level explorer';
    container.appendChild(title);

    // Format selector
    const ctrls = document.createElement('div');
    ctrls.className = 'widget-controls';
    const fmtLabel = document.createElement('label');
    fmtLabel.textContent = 'Format:';
    const fmtSel = document.createElement('div');
    fmtSel.className = 'btn-group';
    Object.keys(FORMATS).forEach(k => {
      const b = document.createElement('button');
      b.textContent = FORMATS[k].label;
      b.dataset.fmt = k;
      if (k === fmtKey) b.classList.add('active');
      fmtSel.appendChild(b);
    });
    fmtLabel.appendChild(fmtSel);
    ctrls.appendChild(fmtLabel);

    // Preset values
    const presetLabel = document.createElement('label');
    presetLabel.textContent = 'Try:';
    const presetSel = document.createElement('div');
    presetSel.className = 'btn-group';
    [
      ['1.0', 1.0],
      ['0.1', 0.1],
      ['π', Math.PI],
      ['-2.5', -2.5],
      ['max', null]
    ].forEach(([lbl, v]) => {
      const b = document.createElement('button');
      b.textContent = lbl;
      b.dataset.value = v === null ? 'max' : v;
      presetSel.appendChild(b);
    });
    presetLabel.appendChild(presetSel);
    ctrls.appendChild(presetLabel);

    container.appendChild(ctrls);

    // Bit-row
    const bitRow = document.createElement('div');
    bitRow.className = 'bit-row';
    container.appendChild(bitRow);

    // Readout
    const dl = document.createElement('dl');
    dl.className = 'format-readout';
    container.appendChild(dl);

    let currentBits = 0;
    let currentFmt = fmt;

    function render() {
      bitRow.innerHTML = '';
      const total = currentFmt.sign + currentFmt.exp + currentFmt.mant;

      // Label row (S / E / M)
      const lblRow = document.createElement('div');
      lblRow.className = 'bit-label-row';
      const styleFlex = (el, n) => { el.style.flex = `${n} 1 0`; };

      const lblS = document.createElement('span'); lblS.textContent = `sign (${currentFmt.sign})`;
      styleFlex(lblS, currentFmt.sign);
      const lblE = document.createElement('span'); lblE.textContent = `exponent (${currentFmt.exp})`;
      styleFlex(lblE, currentFmt.exp);
      const lblM = document.createElement('span'); lblM.textContent = `mantissa (${currentFmt.mant})`;
      styleFlex(lblM, currentFmt.mant);
      lblS.style.color = '#ff7b72';
      lblE.style.color = 'var(--warn)';
      lblM.style.color = 'var(--good)';
      lblRow.appendChild(lblS);
      lblRow.appendChild(lblE);
      lblRow.appendChild(lblM);
      bitRow.appendChild(lblRow);

      // Bit cells
      const cellsDiv = document.createElement('div');
      cellsDiv.className = 'bit-cells';
      for (let i = total - 1; i >= 0; i--) {
        const cell = document.createElement('div');
        cell.className = 'bit-cell';
        const bit = (currentBits >> i) & 1;
        if (bit) cell.classList.add('on');
        // Region
        if (i >= currentFmt.exp + currentFmt.mant) cell.classList.add('region-sign');
        else if (i >= currentFmt.mant) cell.classList.add('region-exp');
        else cell.classList.add('region-mant');
        cell.textContent = bit;
        cell.addEventListener('click', () => {
          currentBits ^= (1 << i);
          render();
        });
        cellsDiv.appendChild(cell);
      }
      bitRow.appendChild(cellsDiv);

      // Readout
      const dec = decode(currentBits, currentFmt);
      dl.innerHTML = '';
      const add = (k, v) => {
        const dt = document.createElement('dt'); dt.textContent = k;
        const dd = document.createElement('dd'); dd.innerHTML = v;
        dl.appendChild(dt); dl.appendChild(dd);
      };
      const total2 = currentFmt.sign + currentFmt.exp + currentFmt.mant;
      const binStr = currentBits.toString(2).padStart(total2, '0');
      const hexStr = '0x' + currentBits.toString(16).padStart(Math.ceil(total2 / 4), '0').toUpperCase();
      add('Decoded', `<span style="color:var(--accent);font-size:1.2em">${dec.repr}</span>`);
      add('Binary', binStr);
      add('Hex', hexStr);
      if (dec.kind === 'normal') {
        add('Sign', dec.sign ? '−' : '+');
        add('Exponent', `${dec.expRaw} − ${currentFmt.bias} = ${dec.expBiased}`);
        add('Mantissa', dec.mantFloat.toFixed(currentFmt.mant > 6 ? 6 : 4));
      }
    }

    fmtSel.addEventListener('click', e => {
      if (e.target.tagName !== 'BUTTON') return;
      const k = e.target.dataset.fmt;
      currentFmt = FORMATS[k];
      currentBits = currentBits & ((1 << (currentFmt.sign + currentFmt.exp + currentFmt.mant)) - 1);
      fmtSel.querySelectorAll('button').forEach(b => b.classList.toggle('active', b === e.target));
      render();
    });

    presetSel.addEventListener('click', e => {
      if (e.target.tagName !== 'BUTTON') return;
      const v = e.target.dataset.value;
      let target;
      if (v === 'max') {
        // Max finite: exponent all-ones-minus-1, mantissa all-ones
        const expMax = (1 << currentFmt.exp) - 1;
        const expVal = (currentFmt.exp > 2) ? expMax - 1 : expMax;
        const mantVal = (1 << currentFmt.mant) - 1;
        currentBits = (expVal << currentFmt.mant) | mantVal;
      } else {
        currentBits = encodeFloat(parseFloat(v), currentFmt);
      }
      render();
    });

    render();
  }

  function encodeFloat(value, fmt) {
    if (value === 0) return 0;
    const sign = value < 0 ? 1 : 0;
    value = Math.abs(value);
    let exp = Math.floor(Math.log2(value));
    let mant = value / Math.pow(2, exp) - 1; // in [0, 1)
    let expBiased = exp + fmt.bias;
    const expMax = (1 << fmt.exp) - 1;
    if (expBiased >= expMax) expBiased = expMax - 1;
    if (expBiased <= 0) {
      // Subnormal — approximate
      expBiased = 0;
      mant = value / Math.pow(2, 1 - fmt.bias);
    }
    const mantBits = Math.min(Math.round(mant * (1 << fmt.mant)), (1 << fmt.mant) - 1);
    return (sign << (fmt.exp + fmt.mant)) | (expBiased << fmt.mant) | mantBits;
  }

  // ============================================================
  // 2. ATTENTION HEATMAP WIDGET
  // ============================================================

  function buildAttentionWidget(container) {
    container.innerHTML = '';

    const title = document.createElement('div');
    title.className = 'widget-title';
    title.textContent = 'Scaled dot-product attention — interactive';
    container.appendChild(title);

    const tokens = (container.dataset.tokens || 'The cat sat on the mat').split(' ');
    const d_model = 4;

    // Random-ish but deterministic vectors per token
    function hash(s, idx) {
      let h = 0;
      for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
      return ((Math.sin(h * 9973 + idx * 17.7) * 10000) % 1 + 1) % 1;
    }
    let vectors = tokens.map(t => Array.from({length: d_model}, (_, i) => hash(t, i) * 2 - 1));

    // Controls
    const ctrls = document.createElement('div');
    ctrls.className = 'widget-controls';

    const focusLabel = document.createElement('label');
    focusLabel.textContent = 'Query token:';
    const focusSel = document.createElement('div');
    focusSel.className = 'btn-group';
    tokens.forEach((t, i) => {
      const b = document.createElement('button');
      b.textContent = t;
      b.dataset.idx = i;
      if (i === tokens.length - 1) b.classList.add('active');
      focusSel.appendChild(b);
    });
    focusLabel.appendChild(focusSel);
    ctrls.appendChild(focusLabel);

    const causalLabel = document.createElement('label');
    const causalCheck = document.createElement('input');
    causalCheck.type = 'checkbox';
    causalCheck.checked = true;
    causalLabel.appendChild(causalCheck);
    causalLabel.appendChild(document.createTextNode(' Causal mask'));
    ctrls.appendChild(causalLabel);

    container.appendChild(ctrls);

    // Visualization area
    const viz = document.createElement('div');
    viz.style.cssText = 'display:grid;gap:var(--space-4)';
    container.appendChild(viz);

    let queryIdx = tokens.length - 1;

    function render() {
      viz.innerHTML = '';
      const N = tokens.length;
      const d = d_model;

      // Compute scores
      const q = vectors[queryIdx];
      const scores = vectors.map((v, j) => {
        if (causalCheck.checked && j > queryIdx) return -Infinity;
        let s = 0;
        for (let i = 0; i < d; i++) s += q[i] * v[i];
        return s / Math.sqrt(d);
      });
      const finite = scores.filter(s => isFinite(s));
      const maxS = Math.max(...finite);
      const expS = scores.map(s => isFinite(s) ? Math.exp(s - maxS) : 0);
      const sumE = expS.reduce((a, b) => a + b, 0);
      const weights = expS.map(e => e / sumE);

      // SVG diagram
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      const W = 700, H = 200;
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      svg.style.width = '100%';
      svg.style.maxWidth = W + 'px';
      svg.style.display = 'block';
      svg.style.margin = '0 auto';

      const tokenY = 30;
      const queryY = 170;
      const xStep = W / (N + 1);

      // Tokens row (keys)
      tokens.forEach((t, j) => {
        const x = xStep * (j + 1);
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x - 35);
        rect.setAttribute('y', tokenY - 14);
        rect.setAttribute('width', 70);
        rect.setAttribute('height', 26);
        rect.setAttribute('rx', 4);
        rect.setAttribute('fill', j === queryIdx ? 'var(--accent)' : 'var(--bg-elev-2)');
        rect.setAttribute('stroke', 'var(--border-strong)');
        g.appendChild(rect);

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', x);
        text.setAttribute('y', tokenY + 4);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', j === queryIdx ? 'white' : 'var(--text)');
        text.setAttribute('font-family', 'JetBrains Mono, monospace');
        text.setAttribute('font-size', '13');
        text.textContent = t;
        g.appendChild(text);

        // Weight label below
        const wt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        wt.setAttribute('x', x);
        wt.setAttribute('y', tokenY + 32);
        wt.setAttribute('text-anchor', 'middle');
        wt.setAttribute('fill', 'var(--text-muted)');
        wt.setAttribute('font-family', 'JetBrains Mono, monospace');
        wt.setAttribute('font-size', '11');
        wt.textContent = weights[j] > 0 ? (weights[j] * 100).toFixed(0) + '%' : '—';
        g.appendChild(wt);

        svg.appendChild(g);
      });

      // Query row (Q) bottom
      const qx = xStep * (queryIdx + 1);
      const qrect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      qrect.setAttribute('x', qx - 60);
      qrect.setAttribute('y', queryY - 14);
      qrect.setAttribute('width', 120);
      qrect.setAttribute('height', 26);
      qrect.setAttribute('rx', 4);
      qrect.setAttribute('fill', 'var(--bg-code)');
      qrect.setAttribute('stroke', 'var(--accent)');
      qrect.setAttribute('stroke-width', '2');
      svg.appendChild(qrect);

      const qtext = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      qtext.setAttribute('x', qx);
      qtext.setAttribute('y', queryY + 4);
      qtext.setAttribute('text-anchor', 'middle');
      qtext.setAttribute('fill', 'var(--accent)');
      qtext.setAttribute('font-family', 'JetBrains Mono, monospace');
      qtext.setAttribute('font-size', '13');
      qtext.textContent = `Q("${tokens[queryIdx]}")`;
      svg.appendChild(qtext);

      // Attention lines
      weights.forEach((w, j) => {
        if (w <= 0) return;
        const x = xStep * (j + 1);
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        line.setAttribute('d', `M ${qx} ${queryY - 14} Q ${qx} ${(queryY + tokenY) / 2}, ${x} ${tokenY + 13}`);
        line.setAttribute('fill', 'none');
        line.setAttribute('stroke', 'var(--accent)');
        line.setAttribute('stroke-width', Math.max(0.5, w * 8));
        line.setAttribute('opacity', Math.max(0.15, w));
        svg.appendChild(line);
      });

      viz.appendChild(svg);

      // Scores table
      const tbl = document.createElement('div');
      tbl.className = 'table-wrap';
      tbl.style.fontSize = 'var(--text-sm)';
      tbl.innerHTML = `
        <table>
          <thead>
            <tr><th>j</th><th>token</th><th>q · k</th><th>÷ √d</th><th>softmax</th></tr>
          </thead>
          <tbody>
            ${tokens.map((t, j) => `
              <tr>
                <td>${j}</td>
                <td><code>${t}</code></td>
                <td>${isFinite(scores[j]) ? (scores[j] * Math.sqrt(d)).toFixed(3) : '<span style="color:var(--text-dim)">masked</span>'}</td>
                <td>${isFinite(scores[j]) ? scores[j].toFixed(3) : '—'}</td>
                <td><strong>${(weights[j] * 100).toFixed(1)}%</strong></td>
              </tr>`).join('')}
          </tbody>
        </table>`;
      viz.appendChild(tbl);
    }

    focusSel.addEventListener('click', e => {
      if (e.target.tagName !== 'BUTTON') return;
      queryIdx = parseInt(e.target.dataset.idx, 10);
      focusSel.querySelectorAll('button').forEach(b => b.classList.toggle('active', b === e.target));
      render();
    });
    causalCheck.addEventListener('change', render);

    render();
  }

  // ============================================================
  // 3. SAMPLING STRATEGIES WIDGET
  // ============================================================

  function buildSamplingWidget(container) {
    container.innerHTML = '';

    const title = document.createElement('div');
    title.className = 'widget-title';
    title.textContent = 'Sampling explorer — adjust knobs, watch the distribution';
    container.appendChild(title);

    // 5-token toy vocab
    const tokens = ['cat', 'dog', 'fish', 'bird', 'fox'];
    const logitsBase = [3.0, 2.5, 0.5, 0.3, -1.0];

    // Controls
    const ctrls = document.createElement('div');
    ctrls.className = 'widget-controls';
    ctrls.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:var(--space-3);';

    function makeSlider(labelText, min, max, step, init, suffix) {
      const wrap = document.createElement('label');
      wrap.style.cssText = 'display:flex;flex-direction:column;gap:6px;font-size:var(--text-sm);';
      const labelLine = document.createElement('span');
      labelLine.style.color = 'var(--text-muted)';
      const valSpan = document.createElement('strong');
      valSpan.style.color = 'var(--accent)';
      valSpan.textContent = init.toFixed(step < 1 ? 2 : 0) + (suffix || '');
      labelLine.innerHTML = `${labelText}: `;
      labelLine.appendChild(valSpan);
      const input = document.createElement('input');
      input.type = 'range';
      input.min = min; input.max = max; input.step = step; input.value = init;
      input.style.width = '100%';
      input.addEventListener('input', () => {
        valSpan.textContent = Number(input.value).toFixed(step < 1 ? 2 : 0) + (suffix || '');
        render();
      });
      wrap.appendChild(labelLine);
      wrap.appendChild(input);
      return { wrap, input };
    }

    const tempCtl = makeSlider('Temperature', 0, 2, 0.05, 1.0);
    const topKCtl = makeSlider('top-k (0 = off)', 0, 5, 1, 0);
    const topPCtl = makeSlider('top-p (1 = off)', 0.1, 1, 0.05, 1.0);
    const minPCtl = makeSlider('min-p (0 = off)', 0, 0.5, 0.01, 0.0);
    ctrls.appendChild(tempCtl.wrap);
    ctrls.appendChild(topKCtl.wrap);
    ctrls.appendChild(topPCtl.wrap);
    ctrls.appendChild(minPCtl.wrap);
    container.appendChild(ctrls);

    // Reset button row
    const resetRow = document.createElement('div');
    resetRow.style.cssText = 'display:flex;gap:var(--space-2);margin-top:var(--space-3);flex-wrap:wrap;';
    [
      ['Greedy (T=0)',        { t: 0,   k: 0,  p: 1,    m: 0    }],
      ['Code (T=0.2, p=0.95)', { t: 0.2, k: 0,  p: 0.95, m: 0    }],
      ['Chat (T=0.7, p=0.9)',  { t: 0.7, k: 0,  p: 0.9,  m: 0    }],
      ['Creative (T=1.2)',     { t: 1.2, k: 0,  p: 0.95, m: 0    }],
      ['top-k=2',              { t: 1.0, k: 2,  p: 1,    m: 0    }],
      ['min-p=0.1',            { t: 1.5, k: 0,  p: 1,    m: 0.1  }]
    ].forEach(([label, preset]) => {
      const b = document.createElement('button');
      b.className = 'btn';
      b.textContent = label;
      b.addEventListener('click', () => {
        tempCtl.input.value = preset.t;
        topKCtl.input.value = preset.k;
        topPCtl.input.value = preset.p;
        minPCtl.input.value = preset.m;
        // Fire input handlers
        [tempCtl, topKCtl, topPCtl, minPCtl].forEach(c => c.input.dispatchEvent(new Event('input')));
      });
      resetRow.appendChild(b);
    });
    container.appendChild(resetRow);

    // Output area
    const out = document.createElement('div');
    out.style.cssText = 'margin-top:var(--space-4);';
    container.appendChild(out);

    function softmax(logits, T) {
      if (T === 0) {
        // Greedy: 100% on max
        const m = Math.max(...logits);
        return logits.map(l => l === m ? 1 : 0);
      }
      const scaled = logits.map(l => l / T);
      const m = Math.max(...scaled);
      const exp = scaled.map(l => Math.exp(l - m));
      const s = exp.reduce((a, b) => a + b, 0);
      return exp.map(e => e / s);
    }

    function applyTopK(probs, k) {
      if (k <= 0 || k >= probs.length) return probs;
      const idx = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map(([, i]) => i);
      const keep = new Set(idx);
      const out = probs.map((p, i) => keep.has(i) ? p : 0);
      const s = out.reduce((a, b) => a + b, 0);
      return s > 0 ? out.map(p => p / s) : out;
    }

    function applyTopP(probs, p) {
      if (p >= 1) return probs;
      const sorted = probs.map((pr, i) => [pr, i]).sort((a, b) => b[0] - a[0]);
      let cum = 0;
      const keep = new Set();
      for (const [pr, i] of sorted) {
        keep.add(i);
        cum += pr;
        if (cum >= p) break;
      }
      const out = probs.map((pr, i) => keep.has(i) ? pr : 0);
      const s = out.reduce((a, b) => a + b, 0);
      return s > 0 ? out.map(pr => pr / s) : out;
    }

    function applyMinP(probs, minP) {
      if (minP <= 0) return probs;
      const max = Math.max(...probs);
      const thresh = minP * max;
      const out = probs.map(p => p >= thresh ? p : 0);
      const s = out.reduce((a, b) => a + b, 0);
      return s > 0 ? out.map(p => p / s) : out;
    }

    function render() {
      const T = parseFloat(tempCtl.input.value);
      const K = parseInt(topKCtl.input.value, 10);
      const P = parseFloat(topPCtl.input.value);
      const M = parseFloat(minPCtl.input.value);

      const afterTemp = softmax(logitsBase, T);
      const afterK    = applyTopK(afterTemp, K);
      const afterP    = applyTopP(afterK, P);
      const afterMinP = applyMinP(afterP, M);

      // Bars: show base distribution alongside final distribution
      out.innerHTML = '';

      const wrap = document.createElement('div');
      wrap.style.cssText = 'display:grid;grid-template-columns:1fr 1fr;gap:var(--space-4);';

      function bars(distrib, label, color, labelColor) {
        const col = document.createElement('div');
        const ttl = document.createElement('div');
        ttl.style.cssText = `font-size:var(--text-sm);color:${labelColor || 'var(--text-muted)'};margin-bottom:var(--space-2);text-transform:uppercase;letter-spacing:0.05em;font-weight:600;`;
        ttl.textContent = label;
        col.appendChild(ttl);
        const max = Math.max(...distrib, 0.01);
        tokens.forEach((tok, i) => {
          const row = document.createElement('div');
          row.style.cssText = 'display:grid;grid-template-columns:60px 1fr 60px;gap:8px;align-items:center;margin-bottom:6px;font-family:JetBrains Mono,monospace;font-size:0.85rem;';
          const name = document.createElement('span');
          name.textContent = tok;
          name.style.color = 'var(--text)';
          const barWrap = document.createElement('div');
          barWrap.style.cssText = 'height:18px;background:var(--bg-code);border-radius:3px;overflow:hidden;border:1px solid var(--border);';
          const bar = document.createElement('div');
          const pct = (distrib[i] / max) * 100;
          bar.style.cssText = `height:100%;width:${pct}%;background:${color};transition:width 200ms ease;`;
          barWrap.appendChild(bar);
          const val = document.createElement('span');
          val.style.color = distrib[i] > 0 ? 'var(--text)' : 'var(--text-dim)';
          val.style.textAlign = 'right';
          val.textContent = (distrib[i] * 100).toFixed(1) + '%';
          row.appendChild(name);
          row.appendChild(barWrap);
          row.appendChild(val);
          col.appendChild(row);
        });
        return col;
      }

      wrap.appendChild(bars(softmax(logitsBase, 1.0), 'Raw model probabilities', 'var(--text-muted)', 'var(--text-muted)'));
      wrap.appendChild(bars(afterMinP, 'After all filters → sample from this', 'var(--accent)', 'var(--accent)'));
      out.appendChild(wrap);

      // Pipeline trace
      const trace = document.createElement('div');
      trace.style.cssText = 'margin-top:var(--space-4);padding:var(--space-3);background:var(--bg-code);border:1px solid var(--border);border-radius:var(--radius-sm);font-family:JetBrains Mono,monospace;font-size:0.8rem;color:var(--text-muted);line-height:1.6;';
      const tokensKept = afterMinP.filter(p => p > 0).length;
      trace.innerHTML = `
        <div><span style="color:var(--text-dim)">// pipeline trace</span></div>
        <div>logits   = [${logitsBase.join(', ')}]</div>
        <div>temp     = ${T.toFixed(2)}   <span style="color:var(--text-dim)">${T === 0 ? '→ greedy: argmax wins' : T < 1 ? '→ sharper distribution' : T > 1 ? '→ flatter distribution' : '→ raw distribution'}</span></div>
        <div>top_k    = ${K}      <span style="color:var(--text-dim)">${K === 0 ? '→ disabled' : `→ keep top ${K} only`}</span></div>
        <div>top_p    = ${P.toFixed(2)}   <span style="color:var(--text-dim)">${P >= 1 ? '→ disabled' : `→ keep smallest nucleus whose cumulative ≥ ${P}`}</span></div>
        <div>min_p    = ${M.toFixed(2)}   <span style="color:var(--text-dim)">${M === 0 ? '→ disabled' : `→ drop tokens below ${M.toFixed(2)} × max_p`}</span></div>
        <div style="margin-top:6px;color:var(--accent)">→ ${tokensKept}/${tokens.length} tokens survive · sample one from the right column</div>`;
      out.appendChild(trace);
    }

    render();
  }

  // ============================================================
  // INITIALIZE
  // ============================================================

  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-widget="float-format"]').forEach(buildFormatWidget);
    document.querySelectorAll('[data-widget="attention"]').forEach(buildAttentionWidget);
    document.querySelectorAll('[data-widget="sampling"]').forEach(buildSamplingWidget);
  });
})();
