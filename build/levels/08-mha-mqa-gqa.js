/* ============================================================================
 * LEVEL 08 — MHA → MQA → GQA: Sharing the KV   (order 8)
 * ----------------------------------------------------------------------------
 * Pattern (copied from 01-tokens-embeddings.js):
 *   - one IIFE, ONE TQ.registerLevel call, NO other globals
 *   - render() = short explanation blocks + PRIMARY interactive visualizations,
 *     every number genuinely computed from the live controls (plain real JS)
 *   - all color via TQ.colorFor / TQ.barRow / TQ.canvasPanel (one color language)
 *   - 2-4 quiz questions
 *
 * The math here is the KV-cache memory formula (exact, no fakes):
 *   cacheBytes(n_kv) = 2 (K&V) × layers × seq_len × n_kv × head_dim × bytes_per_elem
 *   MHA  : n_kv = n_q_heads
 *   GQA  : n_kv = g   (1 < g < n_q)
 *   MQA  : n_kv = 1
 *   savings vs MHA = 1 − n_kv / n_q_heads
 *
 * TWO linked primaries on a shared mode switch (MHA | GQA | MQA):
 *   (A) a query-heads → KV-heads WIRING diagram on a TQ.canvasPanel
 *   (B) a LIVE KV-cache calculator: sliders feed the real formula, three modes
 *       rendered as a TQ.barRow in GB/MB, plus a savings-% readout + the literal
 *       substituted formula string on screen.
 *
 * Structured on the axes (layers, seq, n_kv, head_dim, bytes) so L9 can add MLA
 * as another n_kv-equivalent (latent dim) bar on the same chart.
 * ========================================================================== */

(function () {
  "use strict";

  TQ.registerLevel({
    id: "mha-mqa-gqa",
    order: 8,
    title: "MHA → MQA → GQA: Sharing the KV",
    icon: "🗜",
    tagline: "At long context the KV cache dominates memory — so let query heads share K/V.",
    preread: "Memory-Efficient Attention (core)",
    objectives: [
      "See the query-heads → KV-heads mapping for MHA (1:1), GQA (groups share), MQA (all share 1)",
      "Compute KV-cache memory live and watch it shrink as KV heads are shared",
      "Name the tradeoff: MQA cheapest but can degrade; GQA the sweet spot (Llama-2/3)"
    ],

    render: function (root) {
      /* ---------------------------------------------------- intro narrative */
      root.appendChild(TQ.block(
        TQ.h(2, "The KV cache is the new memory wall"),
        TQ.p(
          "At inference a transformer generates ", TQ.el("strong", { text: "one token at a time" }),
          ". For every new token, attention needs the Keys and Values of ",
          TQ.el("em", { text: "every previous token" }),
          " — and recomputing all of those past Keys and Values from scratch for every new token would cost ",
          "on the order of seq_len² work (you'd redo the whole history each step). So we cache them instead: the ",
          TQ.el("strong", { text: "KV cache" }),
          ". It grows linearly with context length, and is stored per layer, for both K and V."
        ),
        TQ.el("div", { class: "tq-formula-banner" }, TQ.math(
          "KV bytes = 2 (K&V) × layers × seq_len × n_kv_heads × head_dim × bytes_per_elem"
        )),
        TQ.callout(
          "The load-bearing subtlety: cache size depends on the number of KEY/VALUE heads, " +
          "NOT the number of QUERY heads. Queries are computed fresh each step and thrown away — " +
          "they're never cached. Keys and Values are what you pay rent on. That asymmetry is the whole trick."
        ),
        // Static orienting figure: WHERE the KV cache lives and WHY it grows,
        // before the interactive wiring canvas (which assumes you know this).
        // Theme colors only via CSS vars / currentColor — no hardcoded hex.
        TQ.figure(
          '<svg viewBox="0 0 560 130" width="560" height="130" role="img" ' +
            'aria-label="Decode loop: past tokens store K,V into the per-layer KV cache; a new token attends back to all of them and appends its own K,V" ' +
            'font-family="var(--mono)" font-size="12">' +
            '<defs><marker id="tq-l8-arrow" viewBox="0 0 10 10" refX="9" refY="5" ' +
              'markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
              '<path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>' +
            // the KV cache box (per layer)
            '<text x="200" y="20" text-anchor="middle" fill="var(--ink-mute)" font-size="11">KV cache (per layer)</text>' +
            '<rect x="24" y="30" width="352" height="44" rx="9" fill="var(--panel-hi)" ' +
              'stroke="var(--line)"/>' +
            // four stored [K,V] slots for past tokens t1..t4
            '<g>' +
              '<rect x="38" y="40" width="70" height="24" rx="5" fill="none" stroke="var(--cool-deep)"/>' +
              '<text x="73" y="56" text-anchor="middle" fill="var(--ink-soft)">t1 [K,V]</text>' +
              '<rect x="118" y="40" width="70" height="24" rx="5" fill="none" stroke="var(--cool-deep)"/>' +
              '<text x="153" y="56" text-anchor="middle" fill="var(--ink-soft)">t2 [K,V]</text>' +
              '<rect x="198" y="40" width="70" height="24" rx="5" fill="none" stroke="var(--cool-deep)"/>' +
              '<text x="233" y="56" text-anchor="middle" fill="var(--ink-soft)">t3 [K,V]</text>' +
              '<rect x="278" y="40" width="70" height="24" rx="5" fill="none" stroke="var(--cool-deep)"/>' +
              '<text x="313" y="56" text-anchor="middle" fill="var(--ink-soft)">t4 [K,V]</text>' +
            '</g>' +
            // the appended slot for the new token t5
            '<rect x="392" y="40" width="70" height="24" rx="5" fill="none" ' +
              'stroke="var(--accent-2)" stroke-dasharray="4 3"/>' +
            '<text x="427" y="56" text-anchor="middle" fill="var(--accent-2)">t5 [K,V]</text>' +
            '<text x="427" y="80" text-anchor="middle" fill="var(--ink-faint)" font-size="10">appended</text>' +
            // new token t5 at generation time
            '<rect x="468" y="92" width="74" height="26" rx="6" fill="var(--panel-hi)" ' +
              'stroke="var(--accent-2)"/>' +
            '<text x="505" y="109" text-anchor="middle" fill="var(--accent-2)">new tok t5</text>' +
            // attention arrows: t5 reads back from every cached K,V
            '<g stroke="currentColor" stroke-width="1.4" fill="none" color="var(--ink-faint)">' +
              '<path d="M470 100 C 300 110, 110 96, 73 66" marker-end="url(#tq-l8-arrow)"/>' +
              '<path d="M474 100 C 320 112, 180 96, 153 66" marker-end="url(#tq-l8-arrow)"/>' +
              '<path d="M478 100 C 360 112, 250 96, 233 66" marker-end="url(#tq-l8-arrow)"/>' +
              '<path d="M482 100 C 400 110, 320 96, 313 66" marker-end="url(#tq-l8-arrow)"/>' +
            '</g>' +
            '<text x="250" y="124" text-anchor="middle" fill="var(--ink-faint)" font-size="10">' +
              't5 attends back to every cached K,V</text>' +
          '</svg>',
          "Decode time: each past token left one [K,V] slot in the per-layer cache. The new token " +
          "attends back to all of them, then appends its own [K,V]. The cache grows by one slot per token " +
          "per layer — and n_kv (not n_q) sets each slot's width."
        )
      ));

      root.appendChild(TQ.block(
        TQ.h(2, "Three ways to wire query heads to KV heads"),
        TQ.p(
          "Quick vocabulary: a transformer splits attention into ", TQ.el("strong", { text: "n_q query heads" }),
          ", each working in a slice of the model called ", TQ.el("strong", { text: "head_dim" }),
          " (typically head_dim = d_model / n_heads). ",
          TQ.el("strong", { text: "n_kv" }), " is how many distinct Key/Value heads we keep — the lever this level is about. ",
          "The three variants below just change how query heads are wired to those KV heads."
        ),
        TQ.note(
          "Optional analogy for the CNN-fluent: this is the same structural move as grouped / depthwise " +
          "convolution (splitting channels into independent groups), just applied to attention heads instead of channels."
        ),
        TQ.el("ul", { class: "tq-mode-list" },
          TQ.el("li", {},
            TQ.el("strong", { text: "MHA (Multi-Head): " }),
            "n_kv = n_q. Every query head gets its own private K and V. Max expressiveness, max cache. 1:1."),
          TQ.el("li", {},
            TQ.el("strong", { text: "MQA (Multi-Query): " }),
            "n_kv = 1. ALL query heads share one K/V. Cache shrinks by a factor of n_q (e.g. 64×). " +
            "Cheapest — but forcing every query head to read from one shared K/V can measurably hurt quality " +
            "(Google's PaLM model noted small-but-real degradation and training instability)."),
          TQ.el("li", {},
            TQ.el("strong", { text: "GQA (Grouped-Query): " }),
            "n_kv = g, with 1 < g < n_q. Query heads are partitioned into g groups; each shares one K/V. " +
            "It interpolates exactly between MHA (g = n_q) and MQA (g = 1). Llama-2 70B and Llama-3 use " +
            "GQA with 8 KV heads — the sweet spot. Savings vs MHA = ", TQ.math("1 − n_kv/n_q"), ".")
        )
      ));

      /* ---- the same idea, grounded in real PyTorch (code precedes play) --- */
      root.appendChild(TQ.block(
        TQ.h(2, "The same idea in PyTorch"),
        TQ.p(
          "All three variants are ", TQ.el("strong", { text: "one" }), " head-projection with a single knob, ",
          TQ.math("n_kv_heads"), ". Notice K and V are projected to ", TQ.el("strong", { text: "fewer" }),
          " heads than Q — that smaller projection is exactly what shrinks the cache."
        ),
        TQ.code(
          "import torch\n" +
          "import torch.nn as nn\n" +
          "import torch.nn.functional as F\n" +
          "\n" +
          "d_model, n_q_heads, d_head = 4096, 32, 128\n" +
          "\n" +
          "# the one knob that picks the variant:\n" +
          "#   MHA -> n_q_heads (32) | MQA -> 1 | GQA -> groups (e.g. 8)\n" +
          "# for GQA: 1 < n_kv_heads < n_q_heads, and it must divide n_q_heads evenly\n" +
          "n_kv_heads = 8\n" +
          "\n" +
          "# Q is projected to all query heads; K and V to only n_kv_heads (smaller!)\n" +
          "q_proj = nn.Linear(d_model, n_q_heads  * d_head, bias=False)\n" +
          "k_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)\n" +
          "v_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)\n" +
          "\n" +
          "def attend(x):                       # x: (B, T, d_model)\n" +
          "    B, T, _ = x.shape\n" +
          "    q = q_proj(x).view(B, T, n_q_heads,  d_head).transpose(1, 2)\n" +
          "    k = k_proj(x).view(B, T, n_kv_heads, d_head).transpose(1, 2)\n" +
          "    v = v_proj(x).view(B, T, n_kv_heads, d_head).transpose(1, 2)\n" +
          "\n" +
          "    # each group of query heads reuses one shared KV head:\n" +
          "    reps = n_q_heads // n_kv_heads   # MHA->1, MQA->n_q, GQA->n_q/groups\n" +
          "    k = k.repeat_interleave(reps, dim=1)   # -> (B, n_q_heads, T, d_head)\n" +
          "    v = v.repeat_interleave(reps, dim=1)\n" +
          "    return F.scaled_dot_product_attention(q, k, v, is_causal=True)\n" +
          "\n" +
          "# only K,V get cached at decode time (queries are recomputed, never stored):\n" +
          "# kv_cache_bytes = 2 * n_layers * seq * n_kv_heads * d_head * bytes_per_elem  # 2 = K and V\n",
          { lang: "python", label: "MHA / MQA / GQA in one head",
            caption: "n_kv_heads is the whole story: it sizes the K,V projections (and the cache), while " +
              "repeat_interleave hands each query-head group its shared KV head — exactly the wiring drawn below." }
        )
      ));

      /* ============================================================ *
       *  SHARED MODE STATE + the two linked primaries
       * ============================================================ */

      // Discrete context-length stops (power-of-two tokens, clean GB numbers).
      var SEQ_STOPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144];
      function seqLabel(n) {
        return (n >= 1024 ? (n / 1024) + "k" : n) + " tok";
      }

      // ---- live control state (read fresh from the widgets every recompute)
      var state = {
        mode: "GQA",
        layers: 80,
        seqIdx: 5,          // -> 131072 (128k)
        nq: 64,
        headDim: 128,
        bytes: 2,           // fp16
        g: 8                // GQA groups (snapped to a divisor of nq)
      };

      // TRUE divisors of nq (g must divide n_q so heads partition evenly into g
      // groups — that is the defining constraint of GQA). Always includes 1 and nq,
      // so the groups slider interpolates exactly from MQA (g=1) to MHA (g=n_q).
      function divisorStops(nq) {
        var out = [];
        for (var v = 1; v <= nq; v++) if (nq % v === 0) out.push(v);
        return out;
      }
      function snapToStops(val, stops) {
        var best = stops[0], bd = Infinity;
        for (var i = 0; i < stops.length; i++) {
          var d = Math.abs(stops[i] - val);
          if (d < bd) { bd = d; best = stops[i]; }
        }
        return best;
      }

      // ---- the real formula (plain JS, every factor live) ----------------
      function cacheBytes(n_kv) {
        var seq = SEQ_STOPS[state.seqIdx];
        return 2 * state.layers * seq * n_kv * state.headDim * state.bytes;
      }
      var GB = 1024 * 1024 * 1024;
      var MB = 1024 * 1024;
      function fmtBytes(b) {
        if (b >= GB) return TQ.fmt(b / GB, 1) + " GB";
        if (b >= MB) return TQ.fmt(b / MB, 1) + " MB";
        return TQ.fmt(b / 1024, 1) + " KB";
      }
      // n_kv for a given mode given current state
      function nkvFor(mode) {
        if (mode === "MHA") return state.nq;
        if (mode === "MQA") return 1;
        return TQ.clamp(state.g, 1, state.nq); // GQA
      }

      /* ----------------------------- (A) WIRING DIAGRAM (canvas) ---------- */
      var DRAW_CAP = 16; // cap drawn query heads for legibility; calc uses true counts

      var wiring = TQ.canvasPanel(420, 320, function (ctx, w, h) {
        var nqDrawn = Math.min(state.nq, DRAW_CAP);
        var nkvTrue = nkvFor(state.mode);
        // how many KV heads to actually draw, scaled to the drawn query count
        var ratio = nqDrawn / state.nq;            // fraction of heads shown
        var nkvDrawn = Math.max(1, Math.round(nkvTrue * ratio));
        if (state.mode === "MHA") nkvDrawn = nqDrawn;
        if (state.mode === "MQA") nkvDrawn = 1;

        var padTop = 44, padBot = 24;
        var leftX = 92, rightX = w - 92;
        var qSpan = h - padTop - padBot;
        var kSpan = h - padTop - padBot;

        function qy(i) { return padTop + (nqDrawn === 1 ? qSpan / 2 : (i / (nqDrawn - 1)) * qSpan); }
        function ky(j) { return padTop + (nkvDrawn === 1 ? kSpan / 2 : (j / (nkvDrawn - 1)) * kSpan); }

        // group index of drawn query head i -> drawn kv head index
        function kvOf(i) {
          if (state.mode === "MHA") return i;
          if (state.mode === "MQA") return 0;
          // GQA: partition the drawn heads into nkvDrawn contiguous groups
          var per = nqDrawn / nkvDrawn;
          return Math.min(nkvDrawn - 1, Math.floor(i / per));
        }

        // column captions
        ctx.font = "600 13px " + "ui-sans-serif, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillStyle = TQ.cssVar("--ink-soft", 1, "#aab3cd");
        ctx.fillText("Query heads (n_q = " + state.nq + ")", leftX, 22);
        ctx.fillText("KV heads (n_kv = " + nkvTrue + ")", rightX, 22);

        // wires first (under the dots)
        ctx.lineWidth = 2;
        for (var i = 0; i < nqDrawn; i++) {
          var j = kvOf(i);
          var t = nkvDrawn <= 1 ? 0.5 : j / (nkvDrawn - 1);
          var col = TQ.colorFor(0.25 + 0.6 * t); // color by which KV group it joins
          ctx.strokeStyle = col;
          ctx.globalAlpha = 0.75;
          ctx.beginPath();
          ctx.moveTo(leftX + 9, qy(i));
          ctx.bezierCurveTo((leftX + rightX) / 2, qy(i), (leftX + rightX) / 2, ky(j), rightX - 9, ky(j));
          ctx.stroke();
        }
        ctx.globalAlpha = 1;

        // query dots (left)
        for (var qi = 0; qi < nqDrawn; qi++) {
          var jq = kvOf(qi);
          var tq = nkvDrawn <= 1 ? 0.5 : jq / (nkvDrawn - 1);
          ctx.fillStyle = TQ.colorFor(0.25 + 0.6 * tq);
          ctx.beginPath();
          ctx.arc(leftX, qy(qi), 7, 0, Math.PI * 2);
          ctx.fill();
        }
        // kv dots (right)
        for (var kj = 0; kj < nkvDrawn; kj++) {
          var tk = nkvDrawn <= 1 ? 0.5 : kj / (nkvDrawn - 1);
          ctx.fillStyle = TQ.colorFor(0.25 + 0.6 * tk);
          ctx.beginPath();
          ctx.arc(rightX, ky(kj), 9, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = TQ.cssVar("--ink", 0.35, "#e9edf8");
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // footnote when we capped the drawing
        if (state.nq > DRAW_CAP) {
          ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
          ctx.fillStyle = TQ.cssVar("--ink-mute", 1, "#6f7aa0");
          ctx.textAlign = "center";
          ctx.fillText("showing " + nqDrawn + " of " + state.nq + " heads — pattern is identical", w / 2, h - 6);
        }
      });

      /* ----------------------------- (B) CALCULATOR ---------------------- */
      var barsHolder = TQ.el("div", { class: "tq-grow" });
      var savingsOut = TQ.el("div", { class: "tq-kvcalc-savings" });
      var formulaOut = TQ.el("div", { class: "tq-kvcalc-formula" });

      function modeIndex(mode) { return mode === "MHA" ? 0 : (mode === "GQA" ? 1 : 2); }

      function recompute() {
        var nq = state.nq;
        var g = TQ.clamp(state.g, 1, nq);
        var mhaB = cacheBytes(nq);
        var gqaB = cacheBytes(g);
        var mqaB = cacheBytes(1);

        // ---- bars (real bytes -> GB, highlight selected mode) ----
        barsHolder.innerHTML = "";
        barsHolder.appendChild(TQ.barRow([mhaB, gqaB, mqaB], {
          labels: ["MHA", "GQA (g=" + g + ")", "MQA"],
          max: mhaB,
          highlight: modeIndex(state.mode),
          format: fmtBytes
        }));

        // ---- savings vs MHA for the selected mode ----
        var nkv = nkvFor(state.mode);
        var savings = (1 - nkv / nq) * 100;
        var selB = cacheBytes(nkv);
        savingsOut.innerHTML = "";
        savingsOut.appendChild(TQ.el("span", { class: "tq-stat-cap",
          text: state.mode + " cache (n_kv = " + nkv + ")" }));
        savingsOut.appendChild(TQ.el("span", { class: "tq-stat-big", text: fmtBytes(selB) }));
        savingsOut.appendChild(TQ.el("div", { class: "tq-kvcalc-savrow" },
          TQ.badge(TQ.fmt(savings, 1) + "% saved vs MHA",
            savings > 80 ? "good" : (savings > 1 ? "info" : "default")),
          TQ.el("span", { class: "tq-kvcalc-savnote",
            text: state.mode === "MHA" ? "baseline" : "= 1 − " + nkv + "/" + nq })
        ));

        // ---- literal substituted formula string ----
        var seq = SEQ_STOPS[state.seqIdx];
        formulaOut.innerHTML = "";
        formulaOut.appendChild(TQ.math(
          "2 × " + state.layers + " × " + seq + " × " + nkv + " × " +
          state.headDim + " × " + state.bytes + " = " + fmtBytes(selB)
        ));

        wiring.redraw();
      }

      // ---- mode switch (shared: redraws wiring AND highlights the bar) ----
      var modeSeg = TQ.segmented({
        options: [{ label: "MHA", value: "MHA" }, { label: "GQA", value: "GQA" }, { label: "MQA", value: "MQA" }],
        value: state.mode,
        onChange: function (v) {
          state.mode = v;
          groupsWrap.style.display = (v === "GQA") ? "" : "none";
          recompute();
        }
      });

      // ---- sliders -------------------------------------------------------
      var sLayers = TQ.slider({
        min: 4, max: 120, step: 4, value: state.layers, label: "Layers",
        format: function (v) { return String(Math.round(v)); },
        onInput: function (v) { state.layers = Math.round(v); recompute(); }
      });

      var sSeq = TQ.slider({
        min: 0, max: SEQ_STOPS.length - 1, step: 1, value: state.seqIdx, label: "Context length",
        format: function (v) { return seqLabel(SEQ_STOPS[Math.round(v)]); },
        onInput: function (v) { state.seqIdx = Math.round(v); recompute(); }
      });

      var sHeadDim = TQ.slider({
        min: 32, max: 256, step: 32, value: state.headDim, label: "head_dim",
        format: function (v) { return String(Math.round(v)); },
        onInput: function (v) { state.headDim = Math.round(v); recompute(); }
      });

      // groups slider (only meaningful in GQA) — snaps to true divisors of nq
      var sGroups = TQ.slider({
        min: 1, max: state.nq, step: 1, value: state.g, label: "KV groups (g)",
        format: function (v) { return String(snapToStops(Math.round(v), divisorStops(state.nq))); },
        onInput: function (v) {
          state.g = snapToStops(Math.round(v), divisorStops(state.nq));
          recompute();
        }
      });
      var groupsWrap = TQ.el("div", { class: "tq-grow" }, sGroups.el);
      groupsWrap.style.display = (state.mode === "GQA") ? "" : "none";

      // query-heads slider — re-clamps groups when nq changes
      var sNq = TQ.slider({
        min: 8, max: 128, step: 8, value: state.nq, label: "Query heads (n_q)",
        format: function (v) { return String(Math.round(v)); },
        onInput: function (v) {
          state.nq = Math.round(v);
          // re-snap g to a valid divisor of the new nq, keep it ≤ nq
          var stops = divisorStops(state.nq);
          state.g = snapToStops(TQ.clamp(state.g, 1, state.nq), stops);
          sGroups.set(state.g);
          // rebuild groups slider range to new nq (cheap: replace its max)
          var inp = sGroups.el.querySelector(".tq-slider-input");
          if (inp) { inp.max = String(state.nq); inp.value = String(state.g); }
          recompute();
        }
      });

      // ---- bytes/elem toggle (fp16 / fp8) --------------------------------
      var bytesSeg = TQ.segmented({
        options: [{ label: "fp16 (2 B)", value: 2 }, { label: "fp8 (1 B)", value: 1 }],
        value: state.bytes,
        onChange: function (v) { state.bytes = v; recompute(); }
      });

      /* ----------------------------- assemble the calculator block ------- */
      var legend = TQ.el("div", { class: "tq-legend" },
        TQ.el("span", { text: "less cache" }),
        TQ.el("div", { class: "tq-legend-scale" }),
        TQ.el("span", { text: "more cache" })
      );

      var calcBlock = TQ.block(
        TQ.h(2, "Live KV-cache calculator"),
        TQ.p(
          "Pick a mode, then drag the knobs. Every bar is the real formula computed in plain JS. ",
          "Bars are GB of cache for ", TQ.el("strong", { text: "one sequence" }),
          ", relative to MHA (the longest bar)."
        ),
        TQ.el("div", { class: "tq-kvcalc-modebar" },
          TQ.el("span", { class: "tq-slider-label", text: "Attention variant" }),
          modeSeg.el
        ),
        TQ.el("div", { class: "tq-flexrow tq-kvcalc-main" },
          // left: wiring diagram
          TQ.el("div", { class: "tq-panel tq-kvcalc-wire" },
            TQ.el("div", { class: "tq-heatmap-title", text: "query → KV wiring" }),
            wiring.el
          ),
          // right: bars + readouts
          TQ.el("div", { class: "tq-panel tq-grow tq-kvcalc-bars" },
            legend,
            barsHolder,
            savingsOut,
            formulaOut
          )
        ),
        TQ.el("div", { class: "tq-controls-row tq-kvcalc-controls" },
          sLayers.el, sSeq.el, sNq.el, sHeadDim.el, groupsWrap
        ),
        TQ.el("div", { class: "tq-kvcalc-bytes" },
          TQ.el("span", { class: "tq-slider-label", text: "bytes / elem" }),
          bytesSeg.el
        )
      );
      root.appendChild(calcBlock);

      // initial paint
      recompute();

      /* ----------------------------- what to observe --------------------- */
      root.appendChild(TQ.block(
        TQ.h(2, "What the knobs prove"),
        TQ.el("ul", { class: "tq-mode-list" },
          TQ.el("li", {}, "Switch ", TQ.el("strong", { text: "MHA → GQA → MQA" }),
            ": the wiring collapses from many private wires, to a few fat shared bundles, to a single hub — " +
            "and the matching bar shrinks in lockstep. Sharing K/V = fewer distinct KV heads = less cache."),
          TQ.el("li", {}, "Drag ", TQ.el("strong", { text: "Query heads" }),
            " up while in MQA mode: the MQA bar does NOT budge. Cache tracks KV heads, not query heads — " +
            "queries are recomputed and discarded each step, never cached. (MHA's bar, where n_kv = n_q, does grow.)"),
          TQ.el("li", {}, "In GQA, sweep ", TQ.el("strong", { text: "KV groups (g)" }),
            " from 1 to n_q: the GQA bar slides continuously from sitting on top of MQA to sitting on top of MHA. " +
            "GQA is literally a dial between the two; savings = 1 − g/n_q."),
          TQ.el("li", {}, "Flip ", TQ.el("strong", { text: "fp16 → fp8" }),
            ": all three bars halve at once. Precision is an orthogonal lever that multiplies with head-sharing."),
          TQ.el("li", {}, "At 64 query heads, GQA(g=8) shows ~87.5% savings (1 − 8/64) and MQA ~98.4% (1 − 1/64) — " +
            "quantifying the 'GQA recovers most of the win' story.")
        ),
        TQ.callout(
          "The model weights never changed across all of this. The only knob that moved memory was how many " +
          "KV heads you keep. That's why every modern long-context model uses GQA or stronger (MLA, next level) — " +
          "the KV cache, not the parameters, is what bounds context length and batch size in production."
        )
      ));

      /* ----------------------------------------------- go deeper (resources) */
      root.appendChild(TQ.resources("Go deeper — MHA / MQA / GQA", [
        {
          label: "Fast Transformer Decoding: One Write-Head is All You Need (MQA)",
          url: "https://arxiv.org/abs/1911.02150",
          kind: "paper",
          note: "The original MQA paper — share a single K/V head across all query heads to slash decode-time memory."
        },
        {
          label: "GQA: Training Generalized Multi-Query Transformer Models",
          url: "https://arxiv.org/abs/2305.13245",
          kind: "paper",
          note: "Introduces grouped-query attention as the interpolation between MHA and MQA you just dialed through."
        },
        {
          label: "Llama 2 — open foundation models that use GQA",
          url: "https://arxiv.org/abs/2307.09288",
          kind: "paper",
          note: "A flagship open model that adopts GQA (8 KV heads) in practice — the 'sweet spot' in production."
        },
        {
          label: "kipply — Transformer Inference Arithmetic",
          url: "https://kipply.github.io/transformer-inference-arithmetic/",
          kind: "blog",
          note: "Where the 2 × layers × seq × n_kv × head_dim × bytes formula comes from, with the full memory math."
        }
      ]));

      /* small scoped styles — colors come from CSS variables only, no hex */
      injectOnce("tq-lvl08-css",
        ".tq-formula-banner{margin:10px 0;padding:10px 14px;background:var(--panel-hi);" +
          "border:1px solid var(--line);border-radius:10px;text-align:center}" +
        ".tq-mode-list{margin:8px 0 0;padding-left:20px;display:flex;flex-direction:column;gap:8px;line-height:1.5}" +
        ".tq-mode-list li{padding-left:2px}" +
        ".tq-kvcalc-modebar{display:flex;align-items:center;gap:12px;margin:6px 0 14px;flex-wrap:wrap}" +
        ".tq-kvcalc-main{align-items:stretch;gap:18px;flex-wrap:wrap}" +
        ".tq-kvcalc-wire{display:flex;flex-direction:column;gap:8px;min-width:300px}" +
        ".tq-kvcalc-bars{display:flex;flex-direction:column;gap:14px;min-width:300px;justify-content:flex-start}" +
        ".tq-kvcalc-savings{display:flex;flex-direction:column;gap:4px}" +
        ".tq-kvcalc-savrow{display:flex;align-items:center;gap:10px;flex-wrap:wrap}" +
        ".tq-kvcalc-savnote{font-family:var(--mono);font-size:12px;color:var(--ink-mute)}" +
        ".tq-kvcalc-formula{margin-top:4px}" +
        ".tq-kvcalc-controls{margin-top:18px;gap:18px;flex-wrap:wrap}" +
        ".tq-kvcalc-controls>*{min-width:170px;flex:1 1 170px}" +
        ".tq-kvcalc-bytes{display:flex;align-items:center;gap:12px;margin-top:14px;flex-wrap:wrap}");
    },

    quiz: [
      {
        q: "In the KV-cache formula 2 × layers × seq_len × n_kv_heads × head_dim × bytes_per_elem, what does the leading 2 account for?",
        choices: [
          "The two bytes of an fp16 element",
          "Storing both the Key and the Value tensors",
          "Two transformer blocks per layer",
          "Doubling for the forward and backward pass"
        ],
        answer: 1,
        explain: "The 2 is because attention caches BOTH K and V for every past token. bytes_per_elem " +
                 "(fp16=2, fp8=1) is a separate factor; there's no backward pass at inference."
      },
      {
        q: "You drag the 'query heads' slider from 16 to 64 while staying in MQA mode. What happens to the KV-cache size?",
        choices: [
          "It quadruples",
          "It grows linearly with query heads",
          "It stays exactly the same",
          "It shrinks"
        ],
        answer: 2,
        explain: "MQA has n_kv_heads = 1 regardless of query-head count. The cache depends on KV heads, not " +
                 "query heads — queries are recomputed each step and never cached. That's the whole point of MQA/GQA."
      },
      {
        q: "A model has 64 query heads. You configure GQA with 8 KV groups. What is the KV-cache savings versus MHA?",
        choices: [
          "50%",
          "Exactly 8×, i.e. ~87.5% savings",
          "98.4% savings",
          "No savings — GQA caches the same as MHA"
        ],
        answer: 1,
        explain: "Savings vs MHA = 1 − n_kv/n_q = 1 − 8/64 = 0.875 → an 8× reduction. MQA (n_kv=1) would give " +
                 "1 − 1/64 ≈ 98.4%, but GQA trades a little of that for better quality."
      },
      {
        q: "Why is GQA (e.g. Llama-2 70B, Llama-3) usually preferred over MQA in production?",
        choices: [
          "GQA uses less memory than MQA",
          "MQA cannot support long context at all",
          "GQA recovers nearly all of MHA's quality while still cutting the cache massively; MQA's single shared K/V can measurably degrade quality",
          "MQA is incompatible with the KV cache"
        ],
        answer: 2,
        explain: "MQA is the cheapest but collapsing all heads onto one K/V can hurt quality and destabilize " +
                 "training. GQA keeps a handful of KV heads, recovering most of MHA's quality while still cutting the cache ~8×."
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
