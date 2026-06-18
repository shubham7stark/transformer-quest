/* ============================================================================
 * LEVEL 05 — Positional Encoding   (order 5)
 * ----------------------------------------------------------------------------
 * Same pattern as the reference level 01:
 *   - one IIFE, ONE TQ.registerLevel call, NO other globals
 *   - render() = short explanation blocks + PRIMARY interactive visualizations,
 *     every number genuinely computed via TQ math (TQ.sinusoidalPE, TQ.attention)
 *   - all color via TQ.colorFor / TQ.heatmap / TQ.vectorView (one color language)
 *   - 4 quiz questions
 *
 * Math shown here, all real:
 *   PANEL A — TQ.sinusoidalPE(nPos,16) heatmap + a canvas plot of one column's
 *     sin/cos wave across positions; the per-dimension wavelength is computed
 *     directly (2π·10000^(2·floor(dim/2)/16)).
 *   PANEL B — embedding vs PE(pos) vs embedding+PE (TQ.addMat) for a chosen toy
 *     token + position, plus the live L2 distance ||PE_a - PE_b|| between two slots.
 *   PANEL C — attention computed from input = embeddings (PE off) or
 *     addMat(embeddings, PE) (PE on), via TQ.toyQKV + TQ.attention; a "swap
 *     cat⇄sat" rebuilds the input with rows 1,2 swapped BEFORE adding PE. The
 *     'The' weight row + output vector + per-component output delta show that
 *     attention is permutation-equivariant WITHOUT PE and order-aware WITH it.
 * ========================================================================== */

(function () {
  "use strict";

  TQ.registerLevel({
    id: "positional-encoding",
    order: 5,
    title: "Positional Encoding",
    icon: "🧭",
    tagline: "Attention is order-blind by default — give it a sense of position.",
    preread: "Illustrated Transformer / Basics",
    objectives: [
      "Understand self-attention is permutation-equivariant (no built-in order): shuffle the inputs and the outputs just shuffle the same way",
      "Read a sinusoidal PE matrix: positions × dimensions, low dims high-frequency, high dims low-frequency",
      "See that adding PE to an embedding changes the vector, and swapping two words changes their encodings"
    ],

    render: function (root) {
      var toy = TQ.toy;
      var tokens = toy.tokens;
      var emb = toy.embeddings;
      var d = toy.dModel;          // 16
      var n = tokens.length;       // 6

      /* ============================================================= intro */
      root.appendChild(TQ.block(
        TQ.h(2, "Attention is a set operation, not a sequence operation"),
        TQ.p(
          TQ.el("em", { text: "(A set has no order — {cat, sat} is the same as {sat, cat}; a sequence does.) " }),
          "Here's the unsettling fact your CNN intuition won't prepare you for: self-attention has ",
          TQ.el("strong", { text: "no idea what order the tokens are in" }), ". A convolution is hard-wired ",
          "to position — a 3×3 kernel literally reads \"the pixel above-left, above, above-right.\" Locality and ",
          "order are baked into the weights. Attention threw all of that away. Each output token is ",
          TQ.math("softmax(Q·Kᵀ)·V"),
          " — a weighted average over every token, where the weights come only from ",
          TQ.el("strong", { text: "content" }), " (dot products of vectors), never from index."
        ),
        TQ.p(
          "Formally, attention is ", TQ.el("strong", { text: "permutation-equivariant" }),
          " (shuffle the inputs and the outputs just shuffle the same way, nothing else changes)",
          ": permute the input rows and you get the exact same outputs, just permuted the same way. An ",
          "unmoved token's output vector is bit-for-bit identical whether the rest of the sentence is in order ",
          "or shuffled. \"The cat sat on the mat\" and \"mat the on sat cat The\" produce the same bag of ",
          "representations. That's a problem, because word order is most of grammar."
        )
      ));

      root.appendChild(TQ.block(
        TQ.h(2, "The fix: tag each position with a vector, and ADD it in"),
        TQ.p(
          "Before the first attention block, we give position its own embedding — a vector ", TQ.math("PE(pos)"),
          " that depends only on ", TQ.el("strong", { text: "where" }), " a token sits, not ",
          TQ.el("strong", { text: "what" }), " it is — and add it onto the token embedding: ",
          TQ.math("input = embedding + PE"), "."
        ),
        TQ.p(
          "Two things to internalize. First, it's ", TQ.el("strong", { text: "added, not concatenated" }),
          ". You don't grow ", TQ.math("d_model"), "; the position signal lives in the same 16 dimensions as ",
          "meaning, and the network learns to read it back out. Second, once you've added it, a token's input ",
          "vector now depends on its slot — the same word at position 1 vs position 2 becomes two different ",
          "vectors. Now when attention takes dot products, position leaks into the scores, and swapping two ",
          "words actually changes the result. Order is back on the menu."
        )
      ));

      root.appendChild(TQ.block(
        TQ.h(2, "Why sinusoids: a multi-scale clock"),
        TQ.p(
          "The original Transformer builds PE from sines and cosines: ",
          TQ.math("PE(pos,2i) = sin(pos / 10000^(2i/d))"), ", ",
          TQ.math("PE(pos,2i+1) = cos(pos / 10000^(2i/d))"),
          ". Read the matrix as positions (rows) × dimensions (columns), where columns pair up ",
          "(0,1),(2,3),… each pair sharing one frequency. Low columns spin fast — column 0 is ",
          TQ.math("sin(pos/1)"), ", a full cycle every ~6 positions — and high columns spin glacially, ",
          "barely moving across a short sentence. It's a binary-counter-as-waves: fast (low) dimensions are the ",
          "ones digit that flips every step, slow (high) dimensions are the high digits that encode coarse position."
        ),
        TQ.p(
          "Two payoffs. (1) It's deterministic and unbounded — position 5000 gets a clean code even if you ",
          "never trained past 512, so it ", TQ.el("strong", { text: "extrapolates" }), ". (2) The gap between two ",
          "positions always produces the same similarity — so the model can learn \"look k tokens back\" once and ",
          "reuse it everywhere. (Mathematically, the relative offset is a fixed linear/rotation function of the PEs.) ",
          "This lets attention learn \"attend 3 tokens back\" as a single pattern — ",
          TQ.el("strong", { text: "see Panel D" }), ", where you slide the absolute position and watch the ",
          "similarity refuse to move."
        ),
        TQ.callout(
          "(Preview — fully covered in L9, nothing to memorize now.) This is the fixed, hand-designed scheme. " +
          "Modern models often skip it for RoPE, which rotates Q and K " +
          "by position-dependent angles instead of adding anything — and RoPE comes back to bite us in the MLA " +
          "level, where DeepSeek has to carve out a \"decoupled\" RoPE key because rotation doesn't play nicely " +
          "with their latent KV compression."
        )
      ));

      /* ------- orienting diagram: WHERE PE enters the pipeline (static SVG) ----- */
      // Panels A-D all zoom INTO PE; none show the embedding -> (+PE) -> attention
      // dataflow the Takeaway ("input = meaning + position") depends on. Colors via
      // CSS vars / currentColor only — inherits the dark theme.
      root.appendChild(TQ.figure(
        '<svg viewBox="0 0 560 140" width="560" height="140" role="img" ' +
          'aria-label="Positional encoding is added onto token embeddings before the first attention block" ' +
          'font-family="var(--mono)" font-size="13">' +
          '<defs><marker id="tq-l5-arrow" viewBox="0 0 10 10" refX="9" refY="5" ' +
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
            '<path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>' +
          // stage 1: token embeddings
          '<text x="74" y="34" text-anchor="middle" fill="var(--ink-mute)" font-size="11">token embeddings</text>' +
          '<rect x="14" y="44" width="120" height="38" rx="7" fill="var(--panel-hi)" stroke="var(--line)"/>' +
          '<text x="74" y="68" text-anchor="middle" fill="var(--ink)">embed (n × d)</text>' +
          // the (+) node
          '<circle cx="280" cy="63" r="20" fill="var(--panel-hi)" stroke="var(--accent-2)"/>' +
          '<text x="280" y="69" text-anchor="middle" fill="var(--accent-2)" font-size="20">+</text>' +
          // stage 3: input to attention
          '<text x="486" y="34" text-anchor="middle" fill="var(--ink-mute)" font-size="11">input to attention</text>' +
          '<rect x="426" y="44" width="120" height="38" rx="7" fill="var(--panel-hi)" stroke="var(--line)"/>' +
          '<text x="486" y="68" text-anchor="middle" fill="var(--ink)">input (n × d)</text>' +
          // PE feeding the (+) from below
          '<rect x="200" y="104" width="160" height="30" rx="7" fill="var(--panel-hi)" stroke="var(--line)"/>' +
          '<text x="280" y="123" text-anchor="middle" fill="var(--accent)" font-size="12">PE(pos) — sinusoidal (n × d)</text>' +
          // arrows
          '<g stroke="currentColor" stroke-width="1.5" fill="none" color="var(--ink-faint)">' +
            '<line x1="138" y1="63" x2="256" y2="63" marker-end="url(#tq-l5-arrow)"/>' +
            '<line x1="304" y1="63" x2="422" y2="63" marker-end="url(#tq-l5-arrow)"/>' +
            '<line x1="280" y1="100" x2="280" y2="87" marker-end="url(#tq-l5-arrow)"/>' +
          '</g>' +
          '<text x="280" y="34" text-anchor="middle" fill="var(--ink-faint)" font-size="10">elementwise add, d_model unchanged</text>' +
        '</svg>',
        "Positional encoding enters once, before the first attention block: each position's sinusoidal vector is " +
        "added onto its token embedding. Same d_model in, same d_model out — it is added, not concatenated."
      ));

      /* ===================================================== PANEL A: PE map */
      var panelA = TQ.block(
        TQ.h(2, "Panel A · The sinusoidal PE matrix"),
        TQ.p(
          "Below is ", TQ.math("PE(pos, 16)"), " straight from ", TQ.math("TQ.sinusoidalPE"),
          ": rows are positions, columns are dimensions, colored on our shared scale (fixed to ",
          TQ.math("[-1, 1]"), " since sin/cos live there). Drag ", TQ.el("strong", { text: "dimension" }),
          " to highlight one column; the canvas plots that column's wave across positions. Watch the ",
          "frequency drop as the dimension climbs."
        )
      );
      var legendA = TQ.el("div", { class: "tq-legend" },
        TQ.el("span", { text: "−1" }),
        TQ.el("div", { class: "tq-legend-scale" }),
        TQ.el("span", { text: "+1" })
      );
      panelA.appendChild(legendA);

      var stateA = { nPos: 8, dim: 0 };

      var heatHolderA = TQ.el("div", { class: "tq-grow" });
      var wavePanel = TQ.canvasPanel(360, 200, function (ctx, w, h) {
        drawWave(ctx, w, h);
      });
      var waveInfo = TQ.el("div", { class: "tq-pe-waveinfo" });

      function peMatrix() { return TQ.sinusoidalPE(stateA.nPos, d); }

      function wavelengthFor(dim) {
        // pair (2i, 2i+1) share frequency 1/10000^(2i/d); period = 2π·10000^(2i/d)
        var twoI = 2 * Math.floor(dim / 2);
        return 2 * Math.PI * Math.pow(10000, twoI / d);
      }

      function buildHeatA() {
        var PE = peMatrix();
        return TQ.heatmap(PE, {
          rowLabels: TQ.range(stateA.nPos).map(function (p) { return "pos " + p; }),
          colLabels: TQ.range(d).map(function (i) { return String(i); }),
          cellSize: 30,
          min: -1, max: 1,
          selectedCol: stateA.dim,
          format: function (v) { return TQ.fmt(v, 2); },
          title: "PE(pos, dim) — selected dimension highlighted"
        });
      }

      function refreshHeatA() {
        heatHolderA.innerHTML = "";
        heatHolderA.appendChild(buildHeatA());
      }

      function drawWave(ctx, w, h) {
        var PE = peMatrix();
        var dim = stateA.dim;
        var isSin = (dim % 2 === 0);
        var pad = 28;
        var x0 = pad, x1 = w - 10;
        var y0 = 12, y1 = h - 22;
        var midY = (y0 + y1) / 2;
        // axes
        ctx.strokeStyle = TQ.cssVar("--ink", 0.16, "#e9edf8");
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x0, midY); ctx.lineTo(x1, midY); ctx.stroke(); // y=0
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.stroke();     // axis
        // labels for +1 / 0 / -1
        ctx.fillStyle = TQ.cssVar("--ink", 0.6, "#e8ecf6");
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText("+1", 4, y0 + 8);
        ctx.fillText("0", 10, midY + 3);
        ctx.fillText("-1", 6, y1);

        var nP = stateA.nPos;
        function px(p) { return nP > 1 ? x0 + (p / (nP - 1)) * (x1 - x0) : (x0 + x1) / 2; }
        function py(v) { return midY - v * (y1 - y0) / 2; }

        // line
        ctx.strokeStyle = TQ.colorFor(0.82);
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (var p = 0; p < nP; p++) {
          var v = PE[p][dim];
          if (p === 0) ctx.moveTo(px(p), py(v)); else ctx.lineTo(px(p), py(v));
        }
        ctx.stroke();
        // points colored by value on the shared scale
        for (var q = 0; q < nP; q++) {
          var val = PE[q][dim];
          ctx.fillStyle = TQ.colorFor((val + 1) / 2);
          ctx.beginPath();
          ctx.arc(px(q), py(val), 4, 0, Math.PI * 2);
          ctx.fill();
        }
        // caption
        ctx.fillStyle = TQ.cssVar("--ink", 0.7, "#e8ecf6");
        ctx.font = "11px ui-monospace, monospace";
        ctx.fillText("dim " + dim + " = " + (isSin ? "sin" : "cos") + "(pos / 10000^" +
          TQ.fmt(2 * Math.floor(dim / 2) / d, 3) + ")", x0, h - 6);
      }

      function refreshWaveInfo() {
        var dim = stateA.dim;
        var isSin = (dim % 2 === 0);
        var wl = wavelengthFor(dim);
        var pairLo = 2 * Math.floor(dim / 2), pairHi = pairLo + 1;
        waveInfo.innerHTML = "";
        waveInfo.appendChild(TQ.el("div", { class: "tq-pe-chips" },
          TQ.kv("dimension", String(dim)),
          TQ.kv("function", isSin ? "sin" : "cos"),
          TQ.kv("wavelength", wl >= 1000 ? Math.round(wl).toLocaleString() + " pos" : TQ.fmt(wl, 1) + " pos"),
          TQ.kv("pair", "(" + pairLo + ", " + pairHi + ")")
        ));
        waveInfo.appendChild(TQ.note(
          isSin
            ? "This column is the SINE of its frequency. Its partner (dim " + pairHi + ") is the COSINE at the " +
              "same frequency — 90° out of phase. A sin/cos pair is exactly what makes relative position a clean " +
              "rotation — the property Panel D measures: PE(p)·PE(p+k) depends only on the gap k, not on p."
            : "This column is the COSINE of its frequency. Its partner (dim " + pairLo + ") is the SINE at the " +
              "same frequency — 90° out of phase. A sin/cos pair is exactly what makes relative position a clean " +
              "rotation — the property Panel D measures: PE(p)·PE(p+k) depends only on the gap k, not on p."
        ));
      }

      var dimSlider = TQ.slider({
        min: 0, max: d - 1, step: 1, value: stateA.dim, label: "dimension (column)",
        format: function (v) { return "dim " + Math.round(v); },
        onInput: function (v) {
          stateA.dim = Math.round(v);
          refreshHeatA();
          wavePanel.redraw();
          refreshWaveInfo();
        }
      });
      var posSlider = TQ.slider({
        min: 4, max: 20, step: 1, value: stateA.nPos, label: "positions (rows)",
        format: function (v) { return Math.round(v) + " rows"; },
        onInput: function (v) {
          stateA.nPos = Math.round(v);
          refreshHeatA();
          wavePanel.redraw();
        }
      });

      panelA.appendChild(TQ.el("div", { class: "tq-controls-row" }, dimSlider.el, posSlider.el));
      panelA.appendChild(TQ.el("div", { class: "tq-flexrow tq-pe-arow" },
        heatHolderA,
        TQ.el("div", { class: "tq-panel tq-pe-wavecol" },
          TQ.el("div", { class: "tq-slider-label", text: "Selected dimension across positions" }),
          wavePanel.el,
          waveInfo
        )
      ));
      panelA.appendChild(TQ.callout(
        "Drag dimension from 0 upward and the wave stretches out. Dim 0 completes a full cycle in ~6 positions; " +
        "by dim 14 the wavelength is ~19,000 positions, so it's essentially flat across one sentence. Low dims = " +
        "fine position, high dims = coarse position — together they pin down an exact slot, like bits of a counter."
      ));
      // Minimal, correct PyTorch that reproduces exactly the matrix drawn above —
      // the same formula TQ.sinusoidalPE(n, d) implements, then added to embeddings.
      panelA.appendChild(TQ.code(
        "import torch, math\n" +
        "\n" +
        "def sinusoidal_pe(n_pos, d_model):\n" +
        "    # PE[pos, 2i]   = sin(pos / 10000**(2i/d))\n" +
        "    # PE[pos, 2i+1] = cos(pos / 10000**(2i/d))\n" +
        "    pos = torch.arange(n_pos).unsqueeze(1)          # (n_pos, 1)\n" +
        "    i   = torch.arange(0, d_model, 2)               # even indices 0,2,4,...\n" +
        "    div = torch.pow(10000, i / d_model)             # one frequency per pair\n" +
        "    pe = torch.zeros(n_pos, d_model)\n" +
        "    pe[:, 0::2] = torch.sin(pos / div)              # even dims -> sin\n" +
        "    pe[:, 1::2] = torch.cos(pos / div)              # odd dims  -> cos\n" +
        "    return pe\n" +
        "\n" +
        "n_pos, d_model = 6, 16\n" +
        "pe = sinusoidal_pe(n_pos, d_model)   # (6, 16) - exactly the heatmap in Panel A\n" +
        "\n" +
        "# add it onto the token embeddings (NOT concatenate): same d_model\n" +
        "input_embeddings = token_embeddings + pe   # input = meaning + position\n",
        { lang: "python", label: "sinusoidal positional encoding",
          caption: "This is TQ.sinusoidalPE(n, d) with n = 6, d = 16 — the same peRows this level adds to the toy " +
                   "embeddings. div = 10000**(2i/d) is the per-pair wavelength Panel A's slider sweeps. Note the last " +
                   "line uses '+ pe' (not concat), which is why d_model stays 16." }
      ));
      refreshHeatA();
      refreshWaveInfo();
      root.appendChild(panelA);

      /* ============================================ PANEL B: add PE to a token */
      var panelB = TQ.block(
        TQ.h(2, "Panel B · Adding PE changes the vector"),
        TQ.p(
          "Pick a toy token and a position. We show its embedding, the position code ", TQ.math("PE(pos)"),
          ", and their elementwise sum ", TQ.math("embedding + PE"), " — three colored vectors. Several cells ",
          "visibly change color: the encoding now depends on the slot. The distance readout makes \"position ",
          "changes the vector\" a number, not a vibe."
        )
      );

      var peRows = TQ.sinusoidalPE(n, d); // 6 x 16 — one row per slot in the sentence
      var stateB = { tok: 1, posA: 1, posB: 2 };

      var tokSel = TQ.segmented({
        options: tokens.map(function (t, i) { return { label: t, value: i }; }),
        value: stateB.tok,
        onChange: function (v) { stateB.tok = v; updateB(); }
      });
      var posSliderB = TQ.slider({
        min: 0, max: n - 1, step: 1, value: stateB.posA, label: "position (slot)",
        format: function (v) { return "pos " + Math.round(v); },
        onInput: function (v) { stateB.posA = Math.round(v); updateB(); }
      });
      var posSliderB2 = TQ.slider({
        min: 0, max: n - 1, step: 1, value: stateB.posB, label: "compare-against position",
        format: function (v) { return "pos " + Math.round(v); },
        onInput: function (v) { stateB.posB = Math.round(v); updateB(); }
      });

      var outB = TQ.el("div", { class: "tq-panel" });

      function updateB() {
        var ti = stateB.tok, pa = stateB.posA, pb = stateB.posB;
        var e = emb[ti];
        var plus = TQ.addMat([e], [peRows[pa]])[0]; // embedding + PE(pa)
        // distance between the SAME token at two positions; embedding cancels:
        // ||(e+PE_a) - (e+PE_b)|| = ||PE_a - PE_b||
        var diff = [];
        for (var k = 0; k < d; k++) diff.push(peRows[pa][k] - peRows[pb][k]);
        var dist = TQ.norm(diff);

        outB.innerHTML = "";
        outB.appendChild(TQ.vectorView(e, { label: tokens[ti] + " — embedding", cellSize: 20 }));
        outB.appendChild(TQ.vectorView(peRows[pa], { label: "PE(pos " + pa + ")", cellSize: 20, max: 1 }));
        outB.appendChild(TQ.vectorView(plus, { label: tokens[ti] + " + PE(pos " + pa + ")  →  input", cellSize: 20 }));

        outB.appendChild(TQ.el("div", { class: "tq-pe-distrow" },
          TQ.el("div", { class: "tq-sim-cos" },
            TQ.el("span", { class: "tq-stat-cap", text:
              "L2 distance: \"" + tokens[ti] + "\" at pos " + pa + " vs pos " + pb }),
            TQ.el("span", { class: "tq-stat-big", text: TQ.fmt(dist, 3) })
          ),
          (pa === pb
            ? TQ.badge("same slot → distance 0", "default")
            : TQ.badge("position moved the vector", "info"))
        ));
        outB.appendChild(TQ.note(
          "The token embedding cancels in the difference, so this distance is exactly ‖PE(" + pa + ") − PE(" +
          pb + ")‖ — it depends only on the two slots, not on which word you picked. Adjacent toy slots sit ≈1.0 apart."
        ));
      }

      panelB.appendChild(TQ.el("div", { class: "tq-flexcol", style: { gap: "12px" } },
        TQ.el("div", {}, TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "token"), tokSel.el),
        TQ.el("div", { class: "tq-controls-row" }, posSliderB.el, posSliderB2.el),
        outB
      ));
      updateB();
      root.appendChild(panelB);

      /* ====================================== PANEL C: permutation demo */
      var panelC = TQ.block(
        TQ.h(2, "Panel C · Swap two words — does the output move?"),
        TQ.p(
          "Now the payoff. We derive ", TQ.math("Q, K, V"), " from the input (the documented ",
          TQ.math("TQ.toyQKV"), " convention, ", TQ.math("dk = 8"), ") and run real ", TQ.math("TQ.attention"),
          ". Input is either the raw embeddings (PE ", TQ.el("strong", { text: "off" }), ") or ",
          TQ.math("embeddings + PE"), " (PE ", TQ.el("strong", { text: "on" }),
          "). \"Swap cat⇄sat\" rebuilds the input with rows 1 and 2 traded ",
          TQ.el("strong", { text: "before" }), " PE is added, then reruns. We watch the attention-weight row ",
          "for \"The\" and its output vector — original vs swapped."
        )
      );

      var dk = 8;
      var stateC = { pe: false, swapped: false };

      var peToggle = TQ.toggle({
        label: "add positional encoding",
        value: stateC.pe,
        onChange: function (v) { stateC.pe = v; updateC(); }
      });
      var swapBtn = TQ.el("button", { class: "tq-btn tq-btn-accent", type: "button" });
      swapBtn.addEventListener("click", function () { stateC.swapped = !stateC.swapped; updateC(); });

      var outC = TQ.el("div", { class: "tq-panel" });

      function buildInput(swapped, withPE) {
        // start from embeddings; swap cat (row1) & sat (row2) if requested
        var rows = [];
        for (var i = 0; i < n; i++) rows.push(emb[i].slice());
        if (swapped) { var tmp = rows[1]; rows[1] = rows[2]; rows[2] = tmp; }
        if (withPE) rows = TQ.addMat(rows, peRows);
        return rows;
      }

      function runAttn(input) {
        // deterministic projections via the documented convention (seeds 101/202/303)
        var Wq = TQ.randMatrix(d, dk, 101);
        var Wk = TQ.randMatrix(d, dk, 202);
        var Wv = TQ.randMatrix(d, dk, 303);
        var Q = TQ.matmul(input, Wq);
        var K = TQ.matmul(input, Wk);
        var V = TQ.matmul(input, Wv);
        return TQ.attention(Q, K, V);
      }

      function updateC() {
        swapBtn.textContent = stateC.swapped ? "↺ Unswap (back to original order)" : "⇄ Swap cat ⇄ sat";

        var orig = runAttn(buildInput(false, stateC.pe));
        var swap = runAttn(buildInput(true, stateC.pe));

        // 'The' is row 0 — unmoved by the cat/sat swap
        var wOrig = orig.weights[0];
        var wSwap = swap.weights[0];
        var oOrig = orig.output[0];
        var oSwap = swap.output[0];

        // per-component output delta (orig vs swap)
        var delta = [];
        var maxAbsDelta = 0;
        for (var k = 0; k < oOrig.length; k++) {
          var dv = oSwap[k] - oOrig[k];
          delta.push(dv);
          if (Math.abs(dv) > maxAbsDelta) maxAbsDelta = Math.abs(dv);
        }
        var changed = maxAbsDelta > 1e-9;

        // labels: under swap the sentence order is The, sat, cat, on, the, mat
        var origLabels = tokens.slice();
        var swapLabels = tokens.slice();
        var t2 = swapLabels[1]; swapLabels[1] = swapLabels[2]; swapLabels[2] = t2;

        outC.innerHTML = "";
        outC.appendChild(TQ.el("div", { class: "tq-pe-statebar" },
          TQ.badge(stateC.pe ? "PE: ON" : "PE: OFF", stateC.pe ? "good" : "default"),
          TQ.badge(stateC.swapped ? "order: cat⇄sat swapped" : "order: original", stateC.swapped ? "warn" : "info"),
          TQ.el("span", { class: "tq-pe-statenote", text:
            stateC.pe
              ? "Inputs carry position → swapping changes the scores."
              : "Inputs are pure content → swapping only permutes columns." })
        ));

        outC.appendChild(TQ.el("div", { class: "tq-flexrow tq-pe-crow" },
          TQ.el("div", { class: "tq-pe-col" },
            TQ.el("div", { class: "tq-slider-label", text: "Attention weights for \"The\" — ORIGINAL" }),
            TQ.barRow(wOrig, { labels: origLabels, max: 1, format: function (v) { return TQ.fmt(v, 3); } })
          ),
          TQ.el("div", { class: "tq-pe-col" },
            TQ.el("div", { class: "tq-slider-label", text: "Attention weights for \"The\" — SWAPPED" }),
            TQ.barRow(wSwap, { labels: swapLabels, max: 1, format: function (v) { return TQ.fmt(v, 3); } })
          )
        ));

        outC.appendChild(TQ.el("div", { class: "tq-flexrow tq-pe-crow" },
          TQ.el("div", { class: "tq-pe-col" },
            TQ.vectorView(oOrig, { label: "output[\"The\"] — ORIGINAL", cellSize: 22, showValues: false })
          ),
          TQ.el("div", { class: "tq-pe-col" },
            TQ.vectorView(oSwap, { label: "output[\"The\"] — SWAPPED", cellSize: 22, showValues: false })
          )
        ));

        outC.appendChild(TQ.el("div", { class: "tq-pe-deltawrap" },
          TQ.el("div", { class: "tq-slider-label" },
            "Per-component change in output[\"The\"] (swapped − original)  ",
            changed
              ? TQ.badge("Δ nonzero — output MOVED", "warn")
              : TQ.badge("✓ all zero — output IDENTICAL", "good")
          ),
          TQ.vectorView(delta, { label: "Δ output", cellSize: 22, max: Math.max(maxAbsDelta, 1e-6), showValues: true })
        ));

        outC.appendChild(changed
          ? TQ.callout(
              "With PE ON: \"cat\" and \"sat\" now carry different position tags, so swapping them changes the " +
              "dot products. \"The\"'s weight row is genuinely different and its output vector moves (the Δ row lights " +
              "up). THIS contrast is the entire motivation for positional encoding.")
          : TQ.callout(
              "With PE OFF: the weight row for \"The\" just trades its cat/sat columns — same numbers — and the output " +
              "vector is byte-identical (Δ is all zero). This is permutation-equivariance you can stare at: an unmoved " +
              "token sees the same multiset of keys/values, so its output cannot change. Flip PE on and swap again."));
      }

      panelC.appendChild(TQ.el("div", { class: "tq-controls-row" }, peToggle.el, swapBtn));
      panelC.appendChild(outC);
      updateC();
      root.appendChild(panelC);

      /* ============ PANEL D: relative position — the dot product only sees the gap */
      var panelD = TQ.block(
        TQ.h(2, "Panel D · Relative position: the dot product only sees the gap"),
        TQ.p(
          "Panel C proved we ", TQ.el("strong", { text: "need" }), " order. This panel shows ",
          TQ.el("strong", { text: "why sinusoids in particular" }),
          ". Take two positions a fixed offset ", TQ.math("k"), " apart and dot their codes: ",
          TQ.math("sim(p, k) = PE(p) · PE(p+k)"),
          ". The astonishing part — slide the reference position ", TQ.math("p"),
          " with ", TQ.math("k"), " held fixed and the dot product barely moves, even though the two raw ",
          "vectors underneath churn wildly. The similarity depends almost entirely on the ",
          TQ.el("strong", { text: "gap" }), ", not on where you start. That is exactly what lets attention ",
          "learn \"attend ", TQ.math("k"), " tokens back\" as one reusable pattern."
        )
      );

      // A tall, real PE table so p and k slide over a genuine range. d stays 16.
      var nPosD = 40;
      var Kmax = 12;
      var peD = TQ.sinusoidalPE(nPosD, d); // 40 x 16 sinusoids — all real
      var stateD = { p: 4, k: 3, kAlt: 5, overlay: false };

      function simD(a, b) { return TQ.dot(peD[a], peD[b]); } // PE[a]·PE[b]

      // s(k) = PE[ref]·PE[ref+kk] for kk in 0..Kmax — the offset curve from one ref.
      function offsetCurve(ref) {
        var s = [];
        for (var kk = 0; kk <= Kmax; kk++) s.push(simD(ref, ref + kk));
        return s;
      }

      // Sample the pinned similarity at the CURRENT k across several reference
      // positions p' = 0,4,8,... (all valid, i.e. p'+k < nPosD). The spread of
      // these is the live "drift" — translation-invariance turned into a number.
      function driftSamples(k) {
        var vals = [];
        for (var pp = 0; pp + k < nPosD; pp += 4) vals.push(simD(pp, pp + k));
        return vals;
      }

      var ctrlP = TQ.slider({
        min: 0, max: nPosD - 1 - Kmax, step: 1, value: stateD.p, label: "reference position p",
        format: function (v) { return "p = " + Math.round(v); },
        onInput: function (v) { stateD.p = Math.round(v); updateD(); }
      });
      var ctrlK = TQ.slider({
        min: 0, max: Kmax, step: 1, value: stateD.k, label: "offset k",
        format: function (v) { return "k = " + Math.round(v); },
        onInput: function (v) { stateD.k = Math.round(v); updateD(); }
      });
      var ctrlKalt = TQ.segmented({
        options: [{ label: "k′ = 1", value: 1 }, { label: "k′ = 5", value: 5 }, { label: "k′ = 9", value: 9 }],
        value: stateD.kAlt,
        onChange: function (v) { stateD.kAlt = v; updateD(); }
      });
      var ctrlOverlay = TQ.toggle({
        label: "overlay the curve at a second reference position (p + 7)",
        value: stateD.overlay,
        onChange: function (v) { stateD.overlay = v; updateD(); }
      });

      // The similarity-vs-offset canvas. Draws s(k) for the current p (and,
      // when overlay is on, for p2 = p+7) with the chosen k marked.
      var curveCanvas = TQ.canvasPanel(380, 220, function (ctx, w, h) {
        drawCurveD(ctx, w, h);
      });
      var pinOut = TQ.el("div", { class: "tq-panel tq-pe-pin" });
      var vecOut = TQ.el("div", { class: "tq-pe-relcol" });
      var curveCap = TQ.el("div", { class: "tq-pe-relcap" });

      function curveExtent() {
        // y-range across every curve we might draw, so axes are stable while sliding.
        var all = offsetCurve(stateD.p);
        var p2 = stateD.p + 7;
        if (p2 + Kmax < nPosD) all = all.concat(offsetCurve(p2));
        return { lo: Math.min(0, TQ.minOf(all)), hi: TQ.maxOf(all) };
      }

      function drawCurveD(ctx, w, h) {
        var ext = curveExtent();
        var lo = ext.lo, hi = ext.hi, span = (hi - lo) || 1;
        var pad = 30;
        var x0 = pad, x1 = w - 12, y0 = 14, y1 = h - 26;
        function px(kk) { return x0 + (kk / Kmax) * (x1 - x0); }
        function py(v) { return y1 - ((v - lo) / span) * (y1 - y0); }

        // axes
        ctx.strokeStyle = TQ.cssVar("--ink", 0.16, "#e9edf8");
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.stroke();   // y axis
        ctx.beginPath(); ctx.moveTo(x0, py(0)); ctx.lineTo(x1, py(0)); ctx.stroke(); // y=0
        ctx.fillStyle = TQ.cssVar("--ink", 0.6, "#e8ecf6");
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText("sim", 4, y0 + 8);
        ctx.fillText("0", 12, py(0) + 3);
        ctx.fillText("k=0", x0 - 4, y1 + 14);
        ctx.fillText("k=" + Kmax, x1 - 30, y1 + 14);

        function plot(ref, t, withPoints) {
          var s = offsetCurve(ref);
          ctx.strokeStyle = TQ.colorFor(t);
          ctx.lineWidth = 2;
          ctx.beginPath();
          for (var kk = 0; kk <= Kmax; kk++) {
            if (kk === 0) ctx.moveTo(px(kk), py(s[kk])); else ctx.lineTo(px(kk), py(s[kk]));
          }
          ctx.stroke();
          if (withPoints) {
            for (var q = 0; q <= Kmax; q++) {
              ctx.fillStyle = TQ.colorFor(t);
              ctx.globalAlpha = 0.7;
              ctx.beginPath(); ctx.arc(px(q), py(s[q]), 2.5, 0, Math.PI * 2); ctx.fill();
              ctx.globalAlpha = 1;
            }
          }
        }

        // overlay curve at p2 first (so it sits under the primary), then primary.
        var p2 = stateD.p + 7;
        if (stateD.overlay && p2 + Kmax < nPosD) plot(p2, 0.5, false);
        plot(stateD.p, 0.82, true);

        // mark the chosen k on the primary curve.
        var sk = simD(stateD.p, stateD.p + stateD.k);
        var mx = px(stateD.k), my = py(sk);
        ctx.strokeStyle = TQ.cssVar("--ink", 0.28, "#e8ecf6");
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(mx, py(0)); ctx.lineTo(mx, my); ctx.stroke();
        ctx.fillStyle = TQ.colorForSigned(sk, Math.max(Math.abs(hi), Math.abs(lo), 1));
        ctx.beginPath(); ctx.arc(mx, my, 5, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = TQ.cssVar("--accent", 1, "#f2a93b");
        ctx.lineWidth = 1.5; ctx.stroke();
        ctx.fillStyle = TQ.cssVar("--ink", 0.85, "#e8ecf6");
        ctx.font = "11px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillText("k=" + stateD.k + " · " + TQ.fmt(sk, 3), mx, my - 9);
        ctx.textAlign = "left";
      }

      function updateD() {
        var p = stateD.p, k = stateD.k, kAlt = stateD.kAlt;
        var simK = simD(p, p + k);                 // PE(p)·PE(p+k)
        var simK2 = simD(p, p + kAlt);             // control offset PE(p)·PE(p+k′)
        var selfSim = simD(p, p);                  // the k=0 peak

        // drift across reference positions at the CURRENT k.
        var ds = driftSamples(k);
        var drift = TQ.maxOf(ds) - TQ.minOf(ds);

        curveCanvas.redraw();

        // (1) pinned readout + drift badge + control chip
        pinOut.innerHTML = "";
        pinOut.appendChild(TQ.el("div", { class: "tq-pe-relrow" },
          TQ.el("div", { class: "tq-sim-cos" },
            TQ.el("span", { class: "tq-stat-cap", text: "PE(p) · PE(p+k)   [p = " + p + ", k = " + k + "]" }),
            TQ.el("span", { class: "tq-stat-big", text: TQ.fmt(simK, 3) })
          ),
          (drift < 1e-6
            ? TQ.badge("invariant to absolute position", "good")
            : TQ.badge("drifts at this k", "info"))
        ));
        pinOut.appendChild(TQ.el("div", { class: "tq-pe-chips" },
          TQ.kv("drift across p", TQ.fmt(drift, 4)),
          TQ.kv("sampled p′", "0, 4, … (" + ds.length + " refs)"),
          TQ.kv("self-similarity (k=0)", TQ.fmt(selfSim, 3))
        ));
        pinOut.appendChild(TQ.el("div", { class: "tq-pe-relrow" },
          TQ.el("div", { class: "tq-sim-cos" },
            TQ.el("span", { class: "tq-stat-cap", text: "control: PE(p) · PE(p+k′)   [k′ = " + kAlt + "]" }),
            TQ.el("span", { class: "tq-stat-big", text: TQ.fmt(simK2, 3) })
          ),
          TQ.badge("different offset → different similarity", "info")
        ));

        // (3) the two raw vectors that churn while the dot product holds
        vecOut.innerHTML = "";
        vecOut.appendChild(TQ.vectorView(peD[p], { label: "PE(p = " + p + ")", cellSize: 18, max: 1 }));
        vecOut.appendChild(TQ.vectorView(peD[p + k], { label: "PE(p+k = " + (p + k) + ")", cellSize: 18, max: 1 }));

        // (2)/overlay caption: how close are the two curves?
        curveCap.innerHTML = "";
        var p2 = p + 7;
        if (stateD.overlay && p2 + Kmax < nPosD) {
          var c1 = offsetCurve(p), c2 = offsetCurve(p2);
          var gaps = [];
          for (var kk = 0; kk <= Kmax; kk++) gaps.push(Math.abs(c1[kk] - c2[kk]));
          var maxGap = TQ.maxOf(gaps);
          curveCap.appendChild(TQ.el("span", { text:
            "Two curves: s(k) at p = " + p + " and at p2 = " + p2 + ". Largest vertical gap across all k is " }));
          curveCap.appendChild(TQ.el("strong", { text: TQ.fmt(maxGap, 4) }));
          curveCap.appendChild(TQ.el("span", { text: " — the curve barely depends on where you start." }));
        } else {
          curveCap.appendChild(TQ.el("span", { text:
            "s(k) = PE(" + p + ")·PE(" + p + "+k) over k = 0…" + Kmax +
            ". Peaks at k = 0 (self-similarity " + TQ.fmt(selfSim, 2) + ") and decays as the gap grows — " +
            "that decay is a usable notion of relative distance. Toggle the overlay to prove it barely moves with p." }));
        }
      }

      panelD.appendChild(TQ.el("div", { class: "tq-controls-row" }, ctrlP.el, ctrlK.el));
      panelD.appendChild(TQ.el("div", { class: "tq-controls-row" },
        TQ.el("div", {},
          TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "control offset k′"),
          ctrlKalt.el
        ),
        TQ.el("div", {}, ctrlOverlay.el)
      ));
      panelD.appendChild(TQ.el("div", { class: "tq-flexrow tq-pe-drow" },
        TQ.el("div", { class: "tq-pe-relcol tq-grow" }, pinOut, vecOut),
        TQ.el("div", { class: "tq-panel tq-pe-curvecol" },
          TQ.el("div", { class: "tq-slider-label", text: "similarity vs offset   s(k) = PE(p)·PE(p+k)" }),
          curveCanvas.el,
          curveCap
        )
      ));
      panelD.appendChild(TQ.note(
        "This is exact, not approximate — and it holds at ANY d_model, including the 16 used here. Each sin·sin + " +
        "cos·cos pair collapses to cos(k / 10000^(2i/d)), in which the absolute position p has cancelled, so " +
        "PE(p)·PE(p+k) = Σ_i cos(k / 10000^(2i/d)) is provably a function of the gap k alone. The drift you see " +
        "(~1e-15) is pure floating-point rounding, not a truncation artifact. Slide p and watch the readout hold " +
        "to the displayed precision while the two vectors above recolor completely."
      ));
      panelD.appendChild(TQ.callout(
        "This is the deep reason sinusoids beat arbitrary learned tags: a single relative offset k always produces " +
        "the same similarity, anywhere in the sequence. Attention can learn one \"look k tokens back\" pattern and " +
        "reuse it at every position — and RoPE (the L9 payoff) takes this idea further, baking the rotation directly " +
        "into Q and K instead of adding a vector at all."
      ));
      updateD();
      root.appendChild(panelD);

      /* ----------------------------------------------------- wrap-up */
      root.appendChild(TQ.block(
        TQ.h(2, "Takeaway"),
        TQ.p(
          "Raw self-attention is a function over a ", TQ.el("strong", { text: "set" }),
          "; positional encoding is the one ingredient that turns it back into a function over a ",
          TQ.el("strong", { text: "sequence" }), ". Sinusoidal PE is the original fixed recipe: a multi-scale ",
          "clock added straight onto the embeddings. Learned PE and RoPE are the alternatives — and RoPE returns ",
          "in the MLA level (L9), where rotating Q/K clashes with low-rank KV compression and forces a ",
          "\"decoupled\" RoPE key."
        ),
        TQ.callout("Mental model to carry forward: input = meaning + position, sharing the same dimensions. " +
                   "Everything attention does downstream now has order to work with.")
      ));

      /* ----------------------------------------------- go deeper (resources) */
      root.appendChild(TQ.resources("Go deeper — positional encoding", [
        {
          label: "Kazemnejad — Transformer positional encoding",
          url: "https://kazemnejad.com/blog/transformer_architecture_positional_encoding/",
          kind: "blog",
          note: "The clearest derivation of why sin/cos pairs make relative position fall out as a fixed rotation — Panel D, explained."
        },
        {
          label: "The Annotated Transformer (PE in code)",
          url: "https://nlp.seas.harvard.edu/annotated-transformer/",
          kind: "code",
          note: "Harvard's line-by-line PyTorch — its PositionalEncoding module is the real version of the Panel A snippet."
        },
        {
          label: "Attention Is All You Need (original sinusoidal PE, §3.5)",
          url: "https://arxiv.org/abs/1706.03762",
          kind: "paper",
          note: "The source. Section 3.5 introduces exactly the sin/cos formula this level plots."
        },
        {
          label: "Illustrated Transformer — where PE fits in the stack",
          url: "https://jalammar.github.io/illustrated-transformer/",
          kind: "blog",
          note: "Visual tour that shows the embedding + PE step sitting in front of the first attention block."
        },
        {
          label: "RoFormer / RoPE — going deeper than additive PE",
          url: "https://arxiv.org/abs/2104.09864",
          kind: "paper",
          note: "The rotary alternative previewed in the callouts; returns in the L9 (MLA) level."
        }
      ]));

      /* scoped styles (colors only via CSS vars / TQ.colorFor — no hardcoded hex) */
      injectOnce("tq-lvl05-css",
        ".tq-pe-arow{align-items:flex-start;gap:20px}" +
        ".tq-pe-wavecol{min-width:380px;display:flex;flex-direction:column;gap:10px}" +
        ".tq-pe-waveinfo{display:flex;flex-direction:column;gap:8px}" +
        ".tq-pe-chips{display:flex;gap:8px;flex-wrap:wrap}" +
        ".tq-pe-distrow{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:6px 0 4px}" +
        ".tq-pe-statebar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}" +
        ".tq-pe-statenote{color:var(--ink-mute);font-size:13px}" +
        ".tq-pe-crow{align-items:flex-start;gap:20px;margin-bottom:10px}" +
        ".tq-pe-col{flex:1 1 240px;min-width:220px;display:flex;flex-direction:column;gap:8px}" +
        ".tq-pe-deltawrap{display:flex;flex-direction:column;gap:8px;margin-top:8px}" +
        ".tq-pe-drow{align-items:flex-start;gap:20px}" +
        ".tq-pe-relcol{display:flex;flex-direction:column;gap:10px;min-width:300px}" +
        ".tq-pe-curvecol{min-width:392px;display:flex;flex-direction:column;gap:10px}" +
        ".tq-pe-pin{display:flex;flex-direction:column;gap:10px;padding:12px}" +
        ".tq-pe-relrow{display:flex;align-items:center;gap:14px;flex-wrap:wrap}" +
        ".tq-pe-relcap{font-size:13px;color:var(--ink-mute);line-height:1.5}");
    },

    quiz: [
      {
        q: "Without positional encoding, you swap two words in the input and recompute attention. What happens to the output vector of a third, unmoved token?",
        choices: [
          "It is unchanged (bit-for-bit identical)",
          "It changes, because attention is sensitive to order",
          "It becomes zero",
          "It changes only if the swapped words were adjacent"
        ],
        answer: 0,
        explain: "Self-attention is permutation-equivariant (shuffle the inputs, the outputs just shuffle the same " +
                 "way): it's a content-weighted average over a set. An " +
                 "unmoved token sees the same multiset of keys/values, so its output is identical. That " +
                 "order-blindness is exactly why PE is needed."
      },
      {
        q: "In the original Transformer, how is positional information combined with the token embedding?",
        choices: [
          "Concatenated, doubling d_model",
          "Added elementwise: input = embedding + PE, same dimensionality",
          "Multiplied elementwise (gating)",
          "Appended as a single extra integer dimension holding the index"
        ],
        answer: 1,
        explain: "PE is ADDED to the embedding, keeping d_model fixed (16 here). The position signal shares the " +
                 "same dimensions as meaning; the model has ample capacity to disentangle them."
      },
      {
        q: "In the sinusoidal PE matrix, why do low-index dimensions look like fast stripes and high-index dimensions look nearly constant across a short sentence?",
        choices: [
          "Low dims are noise; high dims carry the real signal",
          "Each dimension pair uses frequency 1/10000^(2i/d), so high dims have very long wavelengths and barely move over a few positions",
          "High dims are zeroed out during training",
          "It's a rendering artifact of the color scale"
        ],
        answer: 1,
        explain: "The wavelength grows geometrically with the dimension index. Dim 0 cycles every ~6 positions; " +
                 "dim 14 has a wavelength of ~19,000 positions, so it's essentially flat across one sentence. Fast " +
                 "and slow dims together pin down an exact position, like bits of a counter."
      },
      {
        q: "Sinusoidal PE is the fixed, hand-designed scheme. Which statement about alternatives is correct?",
        choices: [
          "RoPE also adds a vector to the embedding, just with learned values",
          "RoPE rotates the query and key vectors by position-dependent angles instead of adding anything, and it returns in the MLA level where it must be decoupled from latent KV compression",
          "Learned positional embeddings are mathematically identical to sinusoidal ones",
          "Modern LLMs concatenate the position index as a raw scalar"
        ],
        answer: 1,
        explain: "RoPE injects position by rotating Q and K (not by adding a PE vector), which makes relative " +
                 "position fall out of the dot product. In DeepSeek's MLA, RoPE clashes with low-rank KV " +
                 "compression, forcing a separate 'decoupled' RoPE key — the L9 payoff."
      }
    ]
  });

  // tiny helper local to this module (NOT a global): inject scoped CSS once.
  function injectOnce(id, css) {
    if (document.getElementById(id)) return;
    var s = document.createElement("style");
    s.id = id;
    s.textContent = css;
    document.head.appendChild(s);
  }
})();
