/* ============================================================================
 * LEVEL 10 — Mixture of Experts (MoE)   (order 10)
 * ----------------------------------------------------------------------------
 * MoE replaces the dense FFN sub-layer (L6) with N independent expert FFNs +
 * a tiny ROUTER. The headline: decouple PARAMETER COUNT (knowledge/capacity)
 * from PER-TOKEN COMPUTE (FLOPs). Only k of N experts run per token, but total
 * params scale with N.
 *
 * Everything is genuinely computed from TQ math on the shared toy sentence
 * ("The cat sat on the mat", 6 tokens x dModel=16) — NO hardcoded routing/gate
 * values:
 *
 *   ROUTER:  W_g = TQ.randMatrix(16, N, 5100[+1 when "collapsed"])
 *            logits = TQ.matmul(embeddings, W_g)        -> [6 x N]
 *            gates  = TQ.softmaxRows(logits)            -> [6 x N], each row Σ=1
 *   TOP-k:   per row, pick the k largest gate weights (argmax sort on live row);
 *            selected logits re-softmaxed via TQ.softmax -> renorm gates Σ=1.000
 *   EXPERTS: E_i(x) = relu(x·W1_i)·W2_i, W1_i=randMatrix(16,8,5200+i),
 *            W2_i=randMatrix(8,16,5300+i); blended y = Σ_top-k renorm_i·E_i(x)
 *            (+ shared experts always-on) computed live for the clicked token.
 *   PARAMS:  expertParams = 2·d_model·d_ff ; total = (N+S)·expertParams + router,
 *            active = (k+S)·expertParams + router ; fraction ≈ (k+S)/(N+S).
 *   LOAD:    per-expert = count of the 6 tokens that selected each expert.
 *
 * Pattern (copied from 06/08/09): one IIFE, ONE TQ.registerLevel, NO other
 * globals; all color via TQ.colorFor / heatmap / barRow / vectorView; the L6
 * is-masked idiom dims non-selected experts; the L8/L9 active-vs-total +
 * substituted-formula calculator returns; row-Σ=1.000 renorm reflex from L6/L8.
 * ========================================================================== */

(function () {
  "use strict";

  TQ.registerLevel({
    id: "moe",
    order: 10,
    title: "Mixture of Experts (MoE)",
    icon: "🧩",
    tagline: "Scale knowledge without scaling compute: N expert FFNs, but only k run per token.",
    preread: "Mixtral / DeepSeek-V3",
    objectives: [
      "See MoE as a drop-in replacement for the dense FFN sub-layer (L6): one router + N expert FFNs",
      "Watch real top-k routing on the toy sentence — per token, independent, renormalized to Σ=1.000",
      "Separate ACTIVE params (k+shared, the per-token FLOPs) from TOTAL params (N+shared, the capacity)",
      "Name why load balancing matters: a router can collapse onto a few experts, wasting the rest"
    ],

    render: function (root) {
      var toy = TQ.toy;
      var tokens = toy.tokens;     // ["The","cat","sat","on","the","mat"]
      var emb = toy.embeddings;    // 6 x 16
      var dModel = toy.dModel;     // 16
      var n = tokens.length;       // 6

      /* ============================================================== *
       *  INTRO — the FFN is where knowledge lives
       * ============================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "The FFN is where knowledge lives — so what if we had many of them?"),
        TQ.p(
          "Back in Level 6 you learned the load-bearing fact about the feed-forward sub-layer: it expands ",
          "then contracts (~4× d_model), and because of that it holds roughly ",
          TQ.el("strong", { text: "two-thirds of a transformer's parameters" }),
          ". Attention ", TQ.el("em", { text: "routes" }), " information between tokens; the FFN is where the model ",
          TQ.el("strong", { text: "stores and transforms what it knows" }), ", one token at a time."
        ),
        TQ.p(
          "Here is the obvious-in-hindsight question: if the FFN is the knowledge, why have ",
          TQ.el("strong", { text: "one" }), "? ", TQ.el("strong", { text: "Mixture of Experts" }),
          " replaces that single dense FFN with ", TQ.math("N"),
          " independent expert FFNs plus a tiny ", TQ.el("strong", { text: "router" }),
          " (a gating network). The router scores the experts for each token and sends the token to only the ",
          TQ.math("top-k"), " of them. The rest stay asleep."
        ),
        TQ.p(
          "That one move ", TQ.el("strong", { text: "decouples two things that used to grow together" }),
          ": ", TQ.el("strong", { text: "parameter count" }), " (capacity / how much the model can know) and ",
          TQ.el("strong", { text: "per-token compute" }), " (FLOPs). Add more experts and total params climb with ",
          TQ.math("N"), "; per-token compute only tracks ", TQ.math("k"), ". You buy knowledge cheaply."
        ),
        TQ.callout([
          "This IS the Level-6 feed-forward sub-layer — nothing else about the block changes. Attention still ",
          "routes between tokens; embeddings, residuals, and LayerNorm are untouched. MoE only changes ",
          TQ.el("strong", { text: "what happens inside the per-token MLP" }), "."
        ]),
        // Orienting figure: dense FFN -> router + N experts (k lit), shared dashed.
        // Theme colors via CSS vars / currentColor only; arrowhead marker like L6.
        TQ.figure(
          '<svg viewBox="0 0 560 260" width="560" height="260" role="img" ' +
            'aria-label="A dense FFN block morphs into an MoE block: a router fans one token to k of N expert FFNs, ' +
            'summed back onto the residual backbone, with an optional always-on shared expert" ' +
            'font-family="var(--mono)" font-size="11">' +
            '<defs><marker id="tq-l10-arrow" viewBox="0 0 10 10" refX="9" refY="5" ' +
              'markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
              '<path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>' +
            // ---- LEFT: dense block ----
            '<text x="92" y="18" text-anchor="middle" fill="var(--ink-mute)">Dense block</text>' +
            '<g stroke="currentColor" stroke-width="1.5" fill="none" color="var(--line)">' +
              '<line x1="92" y1="232" x2="92" y2="196" marker-end="url(#tq-l10-arrow)"/>' +
              '<line x1="92" y1="160" x2="92" y2="120" marker-end="url(#tq-l10-arrow)"/>' +
            '</g>' +
            '<rect x="44" y="160" width="96" height="36" rx="7" fill="var(--panel-hi)" stroke="var(--line)"/>' +
            '<text x="92" y="182" text-anchor="middle" fill="var(--ink)">FFN</text>' +
            '<text x="92" y="248" text-anchor="middle" fill="var(--ink-soft)">token x</text>' +
            '<text x="92" y="112" text-anchor="middle" fill="var(--ink-soft)">y</text>' +
            // arrow morph
            '<text x="188" y="130" text-anchor="middle" fill="var(--accent)" font-size="22">→</text>' +
            // ---- RIGHT: MoE block ----
            '<text x="392" y="18" text-anchor="middle" fill="var(--ink-mute)">MoE block (k of N)</text>' +
            // router box
            '<rect x="240" y="200" width="84" height="30" rx="7" fill="var(--panel-hi)" stroke="var(--accent)"/>' +
            '<text x="282" y="220" text-anchor="middle" fill="var(--accent)">router</text>' +
            '<text x="282" y="250" text-anchor="middle" fill="var(--ink-soft)">token x</text>' +
            '<line x1="282" y1="246" x2="282" y2="232" stroke="currentColor" stroke-width="1.5" color="var(--line)"/>' +
            // four expert boxes
            '<g font-size="10">' +
              '<rect x="344" y="40"  width="58" height="26" rx="6" fill="var(--panel-hi)" stroke="var(--accent)"/>' +
              '<text x="373" y="57"  text-anchor="middle" fill="var(--accent)">E1</text>' +
              '<rect x="344" y="84"  width="58" height="26" rx="6" fill="var(--panel-hi)" stroke="var(--line)"/>' +
              '<text x="373" y="101" text-anchor="middle" fill="var(--ink-mute)">E2</text>' +
              '<rect x="344" y="128" width="58" height="26" rx="6" fill="var(--panel-hi)" stroke="var(--accent)"/>' +
              '<text x="373" y="145" text-anchor="middle" fill="var(--accent)">E3</text>' +
              '<rect x="344" y="172" width="58" height="26" rx="6" fill="var(--panel-hi)" stroke="var(--line)"/>' +
              '<text x="373" y="189" text-anchor="middle" fill="var(--ink-mute)">E4</text>' +
            '</g>' +
            // wires router -> experts: E1,E3 lit (accent), E2,E4 dim (line)
            '<g fill="none" stroke-width="1.6">' +
              '<path d="M324 206 C 336 120, 332 60, 344 53"  stroke="var(--accent)" marker-end="url(#tq-l10-arrow)"/>' +
              '<path d="M324 209 C 336 150, 334 104, 344 97"  stroke="var(--line)"/>' +
              '<path d="M324 212 C 338 175, 336 145, 344 141" stroke="var(--accent)" marker-end="url(#tq-l10-arrow)"/>' +
              '<path d="M324 215 C 338 198, 336 188, 344 185" stroke="var(--line)"/>' +
            '</g>' +
            // sum node + back to backbone
            '<circle cx="468" cy="119" r="13" fill="var(--panel-hi)" stroke="var(--accent)"/>' +
            '<text x="468" y="124" text-anchor="middle" fill="var(--accent)" font-size="14">Σ</text>' +
            '<g fill="none" stroke-width="1.6">' +
              '<path d="M402 53  C 440 60, 452 100, 458 108" stroke="var(--accent)"/>' +
              '<path d="M402 141 C 440 134, 452 130, 458 128" stroke="var(--accent)"/>' +
            '</g>' +
            '<line x1="481" y1="119" x2="520" y2="119" stroke="currentColor" stroke-width="1.5" ' +
              'color="var(--line)" marker-end="url(#tq-l10-arrow)"/>' +
            '<text x="534" y="123" text-anchor="middle" fill="var(--ink-soft)">y</text>' +
            // optional shared expert (always-on, dashed info)
            '<rect x="344" y="214" width="76" height="22" rx="5" fill="none" stroke="var(--info)" stroke-dasharray="4 3"/>' +
            '<text x="382" y="229" text-anchor="middle" fill="var(--info)" font-size="9">shared (always on)</text>' +
            '<path d="M420 225 C 452 220, 460 150, 462 132" fill="none" stroke="var(--info)" ' +
              'stroke-dasharray="4 3"/>' +
          '</svg>',
          "MoE swaps the single FFN for a router + N experts; each token lights only k of them (here E1, E3) " +
          "plus any always-on shared expert, summed back onto the backbone. Total params grow with N; " +
          "per-token compute grows with k."
        )
      ));

      /* ============================================================== *
       *  THE ROUTER — narrative
       * ============================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "The router: a linear layer that scores experts, per token"),
        TQ.p(
          "The router is the smallest network in the building: one linear layer. For a token vector ",
          TQ.math("x"), " it computes ", TQ.math("g(x) = x · W_g"), ", giving one ",
          TQ.el("strong", { text: "logit per expert" }), ". A ", TQ.math("softmax"),
          " turns those into ", TQ.el("strong", { text: "gate weights" }), " (a probability over the N experts). ",
          "We keep the ", TQ.math("top-k"), " highest-weighted experts, ",
          TQ.el("strong", { text: "renormalize" }), " their gates to sum to 1, and output a weighted blend:"
        ),
        TQ.el("div", { class: "tq-l10-banner" },
          TQ.math("y = Σ over the selected i of  gate_i · Expert_i(x)")
        ),
        TQ.p(
          "Two details are the whole point. First, routing is ", TQ.el("strong", { text: "per token" }),
          " and ", TQ.el("strong", { text: "independent" }), " — not per sequence. The word ",
          TQ.el("em", { text: '"cat"' }), " and the word ", TQ.el("em", { text: '"sat"' }),
          " can land on completely different experts, decided fresh from each token's own vector. Second, only ",
          TQ.math("k"), " of ", TQ.math("N"), " experts ever execute, so ",
          TQ.el("strong", { text: "per-token FLOPs scale with k while total params scale with N" }), "."
        ),
        TQ.el("ul", { class: "tq-l10-vocab" },
          TQ.el("li", {}, TQ.el("strong", { text: "N experts: " }),
            "how many independent FFNs the layer holds (the capacity dial)."),
          TQ.el("li", {}, TQ.el("strong", { text: "top-k: " }),
            "how many experts each token actually uses (the compute dial). Switch=1, Mixtral=2, DeepSeek-V3=8."),
          TQ.el("li", {}, TQ.el("strong", { text: "active vs total params: " }),
            "active ≈ (k + shared) experts run per token; total = all N (+ shared) sit in memory."),
          TQ.el("li", {}, TQ.el("strong", { text: "shared expert: " }),
            "an FFN applied to every token, never gated off (DeepSeek) — common knowledge, always on.")
        ),
        TQ.note(
          "Conceptual rhyme with Level 4: multi-head attention splits one big computation into several " +
          "specialized sub-computations, then combines. Experts are the same instinct — except heads ALL run " +
          "every token, while experts are sparsely SELECTED. MoE sparsifies the FFN the way GQA (L8) sparsified the KV."
        )
      ));

      /* ============================================================== *
       *  SHARED ROUTING STATE + the math (genuinely computed)
       * ============================================================== */

      // ---- live control state ----
      var state = {
        N: 8,            // experts
        k: 2,            // top-k
        dff: 4 * dModel, // expert hidden width (default 4x)
        shared: 0,       // always-on shared experts (0..2)
        collapsed: false,// swap router seed for a collapse-prone one
        selRow: 1        // selected token row ("cat")
      };

      var ROUTER_SEED = 5100;          // balanced router gate matrix seed
      var COLLAPSE_SEED = 5101;        // a deliberately collapse-prone seed
      // Expert FFN weight seeds: W1_i = randMatrix(16,dff,5200+i), W2_i = randMatrix(dff,16,5300+i)
      var W1_BASE = 5200, W2_BASE = 5300;

      // Router gate matrix [16 x N] for the current seed/N (recomputed live).
      function gateMatrix() {
        var seed = state.collapsed ? COLLAPSE_SEED : ROUTER_SEED;
        // A collapse-prone router: skew columns so experts 0,1 carry more signal
        // and the rest are damped. We still compute it from TQ.randMatrix (no
        // hardcoded gates); the actual pile-up is driven by the additive router
        // bias added in routerOutputs() below.
        var Wg = TQ.randMatrix(dModel, state.N, seed);
        if (state.collapsed) {
          for (var r = 0; r < dModel; r++) {
            for (var c = 0; c < state.N; c++) {
              // experts 0 and 1 get amplified, the rest damped -> routing collapse
              Wg[r][c] *= (c < 2 ? 2.6 : 0.25);
            }
          }
        }
        return Wg;
      }

      // logits [6 x N] = embeddings · W_g ; gates = softmaxRows(logits).
      // In the collapsed case we add a positive router BIAS toward experts 0 and 1
      // (zero-mean weights alone can't reliably starve experts) — this models a
      // router that has learned to favor a couple of experts. Still fully computed:
      // logits come from the real matmul, only the per-expert bias is added.
      function routerOutputs() {
        var Wg = gateMatrix();
        var logits = TQ.matmul(emb, Wg);   // 6 x N
        if (state.collapsed) {
          for (var p = 0; p < n; p++) {
            logits[p][0] += 3;             // favored experts get a learned bias
            logits[p][1] += 3;
          }
        }
        var gates = TQ.softmaxRows(logits); // 6 x N, each row Σ=1.000
        return { logits: logits, gates: gates };
      }

      // top-k expert indices for a single gate row (argmax-style sort on live row).
      function topkIndices(row, k) {
        var idx = TQ.range(row.length);
        idx.sort(function (a, b) { return row[b] - row[a]; }); // descending
        return idx.slice(0, k);
      }

      // renormalized gates over the selected experts (re-softmax the selected LOGITS,
      // exactly the L6/L8 "row Σ stays 1.000" reflex).
      function renormSelected(logitRow, sel) {
        var selLogits = sel.map(function (i) { return logitRow[i]; });
        return TQ.softmax(selLogits); // sums to 1.000
      }

      // One toy expert FFN: E_i(x) = relu(x·W1_i)·W2_i  (16 -> dff -> 16).
      function expertFFN(x, i) {
        var W1 = TQ.randMatrix(dModel, state.dff, W1_BASE + i);
        var W2 = TQ.randMatrix(state.dff, dModel, W2_BASE + i);
        var h = TQ.reluVec(TQ.matmul([x], W1)[0]);  // 1 x dff -> relu
        return TQ.matmul([h], W2)[0];               // 1 x 16
      }

      // Blended output for token p: shared (always) + Σ_top-k renorm_i · E_i(x).
      function blendedOutput(p, sel, renorm) {
        var x = emb[p];
        var y = TQ.zeros(dModel);
        var s;
        // shared experts: always on (use ids offset so seeds differ from routed)
        for (s = 0; s < state.shared; s++) {
          var es = expertFFN(x, 900 + s);
          for (var d0 = 0; d0 < dModel; d0++) y[d0] += es[d0];
        }
        // routed top-k, weighted by renormalized gate
        for (s = 0; s < sel.length; s++) {
          var ei = expertFFN(x, sel[s]);
          var w = renorm[s];
          for (var d1 = 0; d1 < dModel; d1++) y[d1] += w * ei[d1];
        }
        return y;
      }

      // per-expert load: how many of the 6 tokens selected each expert (top-k).
      function expertLoad(gates) {
        var load = TQ.zeros(state.N);
        for (var p = 0; p < n; p++) {
          var sel = topkIndices(gates[p], state.k);
          for (var s = 0; s < sel.length; s++) load[sel[s]] += 1;
        }
        return load;
      }

      /* ============================================================== *
       *  PRIMARY INTERACTIVE A — router heatmap + top-k mask + load
       * ============================================================== */
      var heatHolder = TQ.el("div", { class: "tq-grow" });
      var rowCaption = TQ.el("div", { class: "tq-l10-caption" });
      var rowChips = TQ.el("div", { class: "tq-l10-chips" });
      var blendHolder = TQ.el("div", { class: "tq-l10-blend" });
      var loadHolder = TQ.el("div", { class: "tq-grow" });

      // Build the heatmap (rows=tokens, cols=experts), then dim non-selected
      // top-k cells using the L6 is-masked idiom.
      function buildRouterHeat(gates) {
        var colLabels = [];
        for (var c = 0; c < state.N; c++) colLabels.push("E" + c);
        var heat = TQ.heatmap(gates, {
          rowLabels: tokens, colLabels: colLabels,
          cellSize: state.N > 12 ? 34 : 40,
          min: 0, max: 1, selectedRow: state.selRow,
          format: function (v) { return TQ.fmt(v, 2); },
          onCell: function (r) { selectRow(r); }
        });
        // dim every cell that is NOT in its row's top-k (reuse L6 .is-masked)
        var cells = heat.querySelectorAll(".tq-hm-cell");
        var idx = 0;
        for (var rr = 0; rr < n; rr++) {
          var selSet = {};
          var selr = topkIndices(gates[rr], state.k);
          for (var z = 0; z < selr.length; z++) selSet[selr[z]] = true;
          for (var cc = 0; cc < state.N; cc++) {
            if (!selSet[cc]) cells[idx].classList.add("is-masked");
            else cells[idx].classList.add("is-chosen");
            idx++;
          }
        }
        return heat;
      }

      function selectRow(r) {
        state.selRow = TQ.clamp(r, 0, n - 1);
        refreshRouter();
      }

      function refreshRouter() {
        var out = routerOutputs();
        var gates = out.gates;
        var logits = out.logits;

        // heatmap
        var fresh = buildRouterHeat(gates);
        if (heatHolder.firstChild) heatHolder.replaceChild(fresh, heatHolder.firstChild);
        else heatHolder.appendChild(fresh);

        // selected token detail
        var p = state.selRow;
        var sel = topkIndices(gates[p], state.k);
        var renorm = renormSelected(logits[p], sel);

        rowCaption.innerHTML = "";
        var expertNames = sel.map(function (i) { return "E" + i; }).join(", ");
        rowCaption.appendChild(TQ.el("span", {},
          "Token ", TQ.el("strong", { text: '"' + tokens[p] + '"' }), " (#", String(p),
          ") routes to its top-", String(state.k), ": ",
          TQ.el("strong", { text: expertNames }),
          ". Their gate weights renormalize to a clean probability over just those ", String(state.k),
          " experts."
        ));

        // renormalized gate chips + row Σ = 1.000 reflex
        rowChips.innerHTML = "";
        for (var s = 0; s < sel.length; s++) {
          rowChips.appendChild(TQ.kv("E" + sel[s] + " gate", TQ.fmt(renorm[s], 3)));
        }
        rowChips.appendChild(TQ.badge("row Σ (selected) = " + TQ.fmt(TQ.sum(renorm), 3), "good"));

        // live blended output vector
        blendHolder.innerHTML = "";
        var y = blendedOutput(p, sel, renorm);
        blendHolder.appendChild(TQ.el("div", { class: "tq-l10-blend-lbl" },
          "Blended output ",
          TQ.math("y = " + (state.shared ? state.shared + "·shared + " : "") +
            "Σ gate_i · E_i(x)"),
          " for ", TQ.el("strong", { text: '"' + tokens[p] + '"' }),
          " (computed live, 16-dim — same shape as x):"
        ));
        blendHolder.appendChild(TQ.vectorView(y, { cellSize: 20, label: "" }));

        // load bar across all N experts
        var load = expertLoad(gates);
        var labels = [];
        for (var c = 0; c < state.N; c++) labels.push("E" + c);
        loadHolder.innerHTML = "";
        loadHolder.appendChild(TQ.barRow(load, {
          labels: labels, max: n,
          format: function (v) { return Math.round(v) + " / " + n; }
        }));
        var maxLoad = TQ.maxOf(load);
        var usedCount = load.filter(function (v) { return v > 0; }).length;
        loadHolder.appendChild(TQ.el("div", { class: "tq-l10-loadnote" },
          state.collapsed
            ? TQ.el("span", {}, TQ.badge("collapsed", "warn"),
                " only ", TQ.el("strong", { text: String(usedCount) + " of " + state.N }),
                " experts get any tokens — the router piled up to " + Math.round(maxLoad) +
                " tokens on its favorites. The rest are dead weight.")
            : TQ.el("span", {}, TQ.badge("balanced", "good"),
                " tokens spread across ", TQ.el("strong", { text: String(usedCount) + " of " + state.N }),
                " experts. This is what the load-balance loss is fighting to preserve.")
        ));

        // keep the linked param calculator in sync (same N/k/shared/dff state)
        if (typeof recomputeParams === "function") recomputeParams();
      }

      // ---- controls for interactive A ----
      var sN = TQ.slider({
        min: 4, max: 16, step: 1, value: state.N, label: "N experts",
        format: function (v) { return String(Math.round(v)); },
        onInput: function (v) {
          state.N = Math.round(v);
          // re-clamp k to [1, N] exactly like L8's groups re-clamp on nq change
          state.k = TQ.clamp(state.k, 1, state.N);
          var kin = sK.el.querySelector(".tq-slider-input");
          if (kin) { kin.max = String(state.N); kin.value = String(state.k); }
          sK.set(state.k);
          refreshRouter();
        }
      });

      var sK = TQ.slider({
        min: 1, max: state.N, step: 1, value: state.k, label: "top-k (experts per token)",
        format: function (v) { return String(Math.round(v)); },
        onInput: function (v) {
          state.k = TQ.clamp(Math.round(v), 1, state.N);
          refreshRouter();
        }
      });

      var collapseToggle = TQ.toggle({
        label: "collapsed router (watch the load pile up)",
        value: state.collapsed,
        onChange: function (on) { state.collapsed = on; refreshRouter(); }
      });

      var routerBlock = TQ.block(
        TQ.h(2, "Watch real top-k routing on the toy sentence"),
        TQ.p(
          "Every cell below is a ", TQ.el("strong", { text: "genuinely computed" }), " softmax gate weight: ",
          TQ.math("gates = softmax(embeddings · W_g)"), ", with ", TQ.math("W_g"),
          " a real ", TQ.math("16 × N"), " matrix. Rows are the 6 tokens, columns the ", TQ.math("N"),
          " experts; each row sums to 1.000. The ", TQ.el("strong", { text: "top-k cells are highlighted" }),
          "; the rest are dimmed (the same mask idiom you used for causal attention in L6). ",
          "Click any row to see which experts that token chose, the renormalized gates, and the live blended output."
        ),
        TQ.el("div", { class: "tq-legend" },
          TQ.el("span", { text: "gate 0.0" }),
          TQ.el("div", { class: "tq-legend-scale" }),
          TQ.el("span", { text: "gate 1.0" })
        ),
        TQ.el("div", { class: "tq-l10-controls" }, sN.el, sK.el,
          TQ.el("div", { class: "tq-l10-toggle-cell" }, collapseToggle.el)),
        heatHolder,
        TQ.el("div", { class: "tq-l10-detail" },
          rowCaption, rowChips, blendHolder
        ),
        TQ.el("div", { class: "tq-l10-loadwrap" },
          TQ.el("div", { class: "tq-l10-sub", text: "Per-expert load (how many of the 6 tokens chose each expert)" }),
          loadHolder
        ),
        TQ.callout([
          "Because the rows genuinely differ, ", TQ.el("strong", { text: "different tokens route to different experts" }),
          " — that's not a design choice, it falls straight out of the numbers. Flip ",
          TQ.el("strong", { text: "collapsed router" }), " on and the load bar piles onto one or two experts: ",
          "that's the failure mode load balancing exists to prevent."
        ])
      );
      root.appendChild(routerBlock);

      /* ============================================================== *
       *  PRIMARY INTERACTIVE B — active vs total params calculator
       *  (linked to the SAME N/k/shared state)
       * ============================================================== */
      var barsHolder = TQ.el("div", { class: "tq-grow" });
      var fracOut = TQ.el("div", { class: "tq-l10-frac" });
      var formulaOut = TQ.el("div", { class: "tq-l10-formula" });
      var presetNote = TQ.el("div", { class: "tq-l10-preset-note" });

      var MILLION = 1e6;
      function fmtParams(p) {
        if (p >= 1e9) return TQ.fmt(p / 1e9, 2) + " B";
        if (p >= MILLION) return TQ.fmt(p / MILLION, 1) + " M";
        return TQ.fmt(p / 1e3, 1) + " K";
      }

      // The real arithmetic (every factor live). NOTE: to make the bars land on
      // recognizable production scales we report params for a realistic d_model;
      // here d_model is fixed at a real-ish 4096 for the bar math (the toy's 16
      // would be invisible). The (k+S)/(N+S) fraction is identical regardless.
      var D_MODEL_REAL = 4096;
      function expertParams() { return 2 * D_MODEL_REAL * state.dff_real(); }
      // dff_real scales the toy slider (which moves in units of d_model multiples)
      // up to the real d_model so the bars are production-sized.
      state.dff_real = function () {
        // state.dff is in toy units (multiples of toy dModel=16). Convert to a
        // multiple-of-d_model and apply to the real d_model.
        var mult = state.dff / dModel; // e.g. 64/16 = 4x
        return Math.round(mult * D_MODEL_REAL);
      };

      function routerParams() { return D_MODEL_REAL * state.N; }

      function recomputeParams() {
        var eP = expertParams();
        var totalRouted = state.N * eP;
        var activeRouted = state.k * eP;
        var sharedP = state.shared * eP;
        var total = totalRouted + sharedP + routerParams();
        var active = activeRouted + sharedP + routerParams();

        // two-bar: active per token vs total in layer
        barsHolder.innerHTML = "";
        barsHolder.appendChild(TQ.barRow([active, total], {
          labels: ["active / token", "total in layer"],
          max: total,
          highlight: 0,
          format: fmtParams
        }));

        // live (k+S)/(N+S) fraction readout
        var fracNum = state.k + state.shared;
        var fracDen = state.N + state.shared;
        var pct = (fracNum / fracDen) * 100;
        fracOut.innerHTML = "";
        fracOut.appendChild(TQ.el("span", { class: "tq-l10-frac-cap",
          text: "params run per token" }));
        fracOut.appendChild(TQ.el("span", { class: "tq-l10-frac-big",
          text: TQ.fmt(pct, 1) + "%" }));
        fracOut.appendChild(TQ.el("div", { class: "tq-l10-frac-row" },
          TQ.badge("(k+S)/(N+S) = (" + state.k + "+" + state.shared + ")/(" +
            state.N + "+" + state.shared + ")", pct < 50 ? "good" : "info"),
          TQ.el("span", { class: "tq-l10-frac-note",
            text: "= " + fracNum + "/" + fracDen })
        ));

        // substituted formula string (the L8/L9 idiom)
        formulaOut.innerHTML = "";
        formulaOut.appendChild(TQ.math(
          "active = (k+S)·2·d·d_ff + router = (" + state.k + "+" + state.shared + ")·" +
          fmtParams(eP) + " + " + fmtParams(routerParams()) + " = " + fmtParams(active)
        ));
        formulaOut.appendChild(TQ.math(
          "total = (N+S)·2·d·d_ff + router = (" + state.N + "+" + state.shared + ")·" +
          fmtParams(eP) + " + " + fmtParams(routerParams()) + " = " + fmtParams(total)
        ));
      }

      // ---- d_ff slider (in toy units; default 4x d_model = 64) ----
      var sDff = TQ.slider({
        min: 2 * dModel, max: 8 * dModel, step: dModel, value: state.dff,
        label: "d_ff (expert hidden width)",
        format: function (v) { return Math.round(v / dModel) + "× d_model"; },
        onInput: function (v) { state.dff = Math.round(v); recomputeParams(); }
      });

      // ---- shared-expert count (0..2) ----
      var sShared = TQ.slider({
        min: 0, max: 2, step: 1, value: state.shared, label: "shared experts (always on)",
        format: function (v) { return String(Math.round(v)); },
        onInput: function (v) { state.shared = Math.round(v); refreshRouter(); /* shared affects blended y too */ }
      });

      // ---- real-model preset selector ----
      var presetSeg = TQ.segmented({
        options: [
          { label: "Dense", value: "dense" },
          { label: "Switch (top-1)", value: "switch" },
          { label: "Mixtral 8×7B", value: "mixtral" },
          { label: "DeepSeek-V3", value: "deepseek" }
        ],
        value: "mixtral",
        onChange: applyPreset
      });

      function applyPreset(which) {
        if (which === "dense") {
          // Dense FFN as a degenerate MoE: k = N, so every "expert" runs (no sparsity).
          state.N = 4; state.k = 4; state.shared = 0;
          presetNote.textContent =
            "Dense: one FFN per token — here modeled as k = N (no sparsity), so active = total. " +
            "This is the L6 block before MoE.";
        } else if (which === "switch") {
          state.N = 8; state.k = 1; state.shared = 0;
          presetNote.textContent =
            "Switch Transformer: top-1 routing. Each token uses exactly ONE expert — the maximally sparse MoE.";
        } else if (which === "mixtral") {
          state.N = 8; state.k = 2; state.shared = 0;
          presetNote.textContent =
            "Mixtral 8×7B: 8 experts, top-2. ~47B params total but only ~13B active per token — the headline MoE.";
        } else if (which === "deepseek") {
          // DeepSeek-V3: 256 routed top-8 + 1 shared. N capped at 16 here for the
          // toy heatmap, but k/shared show the ratio shape; note explains the real config.
          state.N = 16; state.k = 8; state.shared = 1;
          presetNote.textContent =
            "DeepSeek-V3: 256 routed experts, top-8, PLUS a shared expert (aux-loss-free balancing). " +
            "Shown here scaled to N=16 for the heatmap — the active/total ratio shape is the same: " +
            "(8+1) of (256+1) ≈ 3.5% of routed params run per token.";
        }
        state.N = TQ.clamp(state.N, 4, 16);
        state.k = TQ.clamp(state.k, 1, state.N);
        // push to widgets
        sN.set(state.N);
        var kin = sK.el.querySelector(".tq-slider-input");
        if (kin) { kin.max = String(state.N); kin.value = String(state.k); }
        sK.set(state.k);
        sShared.set(state.shared);
        refreshRouter();
      }

      var paramsBlock = TQ.block(
        TQ.h(2, "Active vs total params: scale knowledge, not compute"),
        TQ.p(
          "Same ", TQ.math("N"), " and ", TQ.math("k"), " as the router above — now turned into the number that ",
          "matters for cost. One expert FFN has ", TQ.math("2 · d_model · d_ff"), " params. ",
          TQ.el("strong", { text: "Total" }), " params scale with all ", TQ.math("N"),
          " experts (your capacity); ", TQ.el("strong", { text: "active" }),
          " params — the per-token FLOPs — scale with only the ", TQ.math("k"),
          " selected ones, plus any always-on shared expert, plus the tiny router."
        ),
        TQ.el("div", { class: "tq-l10-presetbar" },
          TQ.el("span", { class: "tq-slider-label", text: "Real-model preset" }),
          presetSeg.el
        ),
        presetNote,
        TQ.el("div", { class: "tq-l10-paramsmain" },
          TQ.el("div", { class: "tq-panel tq-grow tq-l10-bars" },
            barsHolder, fracOut, formulaOut
          )
        ),
        TQ.el("div", { class: "tq-l10-controls" }, sDff.el, sShared.el),
        TQ.callout([
          "Snap to ", TQ.el("strong", { text: "Mixtral 8×7B" }), " (8 experts, top-2): the active bar is a sliver of the total bar. ",
          "That model is ~47B params of knowledge but only ~13B run for any given token. ",
          "Add a ", TQ.el("strong", { text: "shared expert" }),
          " and it joins BOTH bars — common knowledge you always pay for. ", "active << total is the whole game."
        ])
      );
      root.appendChild(paramsBlock);

      // initial paint of BOTH linked primaries
      presetNote.textContent =
        "Mixtral 8×7B: 8 experts, top-2. ~47B params total but only ~13B active per token — the headline MoE.";
      refreshRouter();   // also triggers recomputeParams via the linked hook

      /* ============================================================== *
       *  CODE — a sparse MoE FFN in PyTorch
       * ============================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "The same idea in PyTorch"),
        TQ.p(
          "A sparse MoE FFN sub-layer is short: a router (one ", TQ.math("nn.Linear"),
          "), a list of expert FFNs, a ", TQ.math("topk"), " over the router logits, a ",
          TQ.math("softmax"), " to renormalize the chosen gates, and a loop that runs ",
          TQ.el("strong", { text: "only the k selected experts" }), " per token. Optionally a shared expert is added unconditionally."
        ),
        TQ.code(
          "import torch\n" +
          "import torch.nn as nn\n" +
          "import torch.nn.functional as F\n" +
          "\n" +
          "class Expert(nn.Module):                 # one expert = an ordinary FFN\n" +
          "    def __init__(self, d_model, d_ff):\n" +
          "        super().__init__()\n" +
          "        self.net = nn.Sequential(\n" +
          "            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)\n" +
          "        )\n" +
          "    def forward(self, x): return self.net(x)\n" +
          "\n" +
          "class MoE(nn.Module):\n" +
          "    def __init__(self, d_model, d_ff, n_experts, top_k, n_shared=0):\n" +
          "        super().__init__()\n" +
          "        self.router  = nn.Linear(d_model, n_experts, bias=False)   # the gate g(x)=x.W_g\n" +
          "        self.experts = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_experts))\n" +
          "        self.shared  = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_shared))\n" +
          "        self.top_k   = top_k\n" +
          "\n" +
          "    def forward(self, x):                 # x: (tokens, d_model) -- routing is PER TOKEN\n" +
          "        logits = self.router(x)                         # (tokens, n_experts)\n" +
          "        topv, topi = logits.topk(self.top_k, dim=-1)    # pick k experts per token\n" +
          "        gate = F.softmax(topv, dim=-1)                  # renormalize over the k chosen\n" +
          "\n" +
          "        y = torch.zeros_like(x)                         # accumulate the blended output\n" +
          "        for s in self.shared: y = y + s(x)              # shared experts: ALWAYS on\n" +
          "        for slot in range(self.top_k):                  # only k routed experts run / token\n" +
          "            idx, w = topi[:, slot], gate[:, slot:slot+1]\n" +
          "            for e in range(len(self.experts)):\n" +
          "                m = idx == e\n" +
          "                if m.any(): y[m] += w[m] * self.experts[e](x[m])\n" +
          "        return y\n" +
          "\n" +
          "# total params ~ N experts, but only k (+ shared) run per token:\n" +
          "#   active/token ~ (k + n_shared) / (N + n_shared) of the expert params.\n" +
          "# Mixtral 8x7B: N=8, k=2 -> ~47B total but ~13B active per token.\n",
          { lang: "python", label: "a sparse MoE FFN (top-k routing)",
            caption: "router -> topk -> softmax(topk_logits) is exactly the heatmap above (logits, pick top-k, " +
              "renormalize to Σ=1.000). The per-expert loop runs only k experts per token — the active bar. " +
              "Shared experts add to BOTH active and total; load balancing (aux loss, or DeepSeek-V3's " +
              "aux-loss-free per-expert bias) keeps the router from collapsing onto a few experts." }
        )
      ));

      /* ============================================================== *
       *  SHARED EXPERTS & VARIANTS
       * ============================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "Switch, Mixtral, DeepSeek — and why load balancing matters"),
        TQ.p(
          "The same machinery scales across three landmark configurations (try the presets above):"
        ),
        TQ.el("ul", { class: "tq-l10-vocab" },
          TQ.el("li", {}, TQ.el("strong", { text: "Switch Transformer — top-1. " }),
            "The maximally sparse choice: each token uses exactly one expert. Cheapest routing, simplest math."),
          TQ.el("li", {}, TQ.el("strong", { text: "Mixtral 8×7B — top-2 of 8. " }),
            "~47B total params, ~13B active per token. The model that made MoE mainstream."),
          TQ.el("li", {}, TQ.el("strong", { text: "DeepSeek-V3 — 256 routed top-8 + shared. " }),
            "Fine-grained routed experts that SPECIALIZE, plus always-on ",
            TQ.el("strong", { text: "shared expert(s)" }),
            " that capture common knowledge. This is the L9 DeepSeek thread continuing: MLA shrinks the KV cache, " +
            "MoE shrinks the active FFN — the same model pairs both.")
        ),
        TQ.p(
          TQ.el("strong", { text: "Load balancing. " }),
          "Left alone, the router tends to ", TQ.el("strong", { text: "collapse" }),
          " — a few experts win every token, the rest never train and become dead capacity (flip the collapsed " +
          "toggle above to watch it). Two fixes are used in practice: an ",
          TQ.el("strong", { text: "auxiliary load-balance loss" }),
          " (Switch / GShard) that penalizes uneven usage, or DeepSeek-V3's ",
          TQ.el("strong", { text: "aux-loss-free per-expert bias" }),
          " that nudges routing toward even usage without an extra loss term. Either way, the goal is the same: ",
          "keep every expert fed so the capacity you paid for is actually used."
        ),
        TQ.note(
          "Shared expert vs routed expert is the key DeepSeek distinction: a shared expert is NEVER gated off " +
          "(every token, always), so it adds to both active and total params — active/token ≈ (k + n_shared)/(N + n_shared). " +
          "Routed experts are sparsely top-k selected and specialize."
        )
      ));

      /* ============================================================== *
       *  GO DEEPER — approved MoE links
       * ============================================================== */
      root.appendChild(TQ.resources("Go deeper — Mixture of Experts", [
        { label: "Hugging Face — Mixture of Experts Explained", kind: "blog",
          url: "https://huggingface.co/blog/moe",
          note: "the friendliest visual intro to routing, sparsity, and load balancing" },
        { label: "Switch Transformers (top-1 routing)", kind: "paper",
          url: "https://arxiv.org/abs/2101.03961",
          note: "the maximally sparse MoE and the original auxiliary load-balance loss" },
        { label: "Mixtral of Experts (8 experts, top-2)", kind: "paper",
          url: "https://arxiv.org/abs/2401.04088",
          note: "~47B total / ~13B active — the model that made MoE mainstream" },
        { label: "DeepSeekMoE (fine-grained + shared experts)", kind: "paper",
          url: "https://arxiv.org/abs/2401.06066",
          note: "where the shared-expert idea (common knowledge, always on) comes from" },
        { label: "DeepSeek-V3 (256 routed + shared, aux-loss-free balancing)", kind: "paper",
          url: "https://arxiv.org/abs/2412.19437",
          note: "256 routed top-8 + shared, paired with MLA from Level 9; bias-based balancing" }
      ]));

      /* ============================================================== *
       *  WRAP-UP
       * ============================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "Scale knowledge without scaling compute"),
        TQ.p(
          "That's the whole idea, stated as a decoupling: ", TQ.el("strong", { text: "total params" }),
          " (what the model knows) and ", TQ.el("strong", { text: "active params" }),
          " (what it computes per token) no longer have to grow together. Add experts and capacity climbs; ",
          "top-k holds the per-token cost flat. Mixtral 8×7B is ~47B total but ~13B active; DeepSeek-V3 keeps ",
          "hundreds of experts alive while running a small fraction each step."
        ),
        TQ.callout([
          "Carry forward to the boss synthesis: MoE slots into the full forward pass exactly where the L6 dense " +
          "FFN sat. Combine it with the KV-cache tricks (GQA/MLA) and you have the modern frontier recipe — ",
          TQ.el("strong", { text: "sparse attention memory + sparse FFN compute" }),
          ", both decoupling a cost axis from quality."
        ])
      ));

      /* ---- scoped styles — colors via CSS variables only, no hardcoded hex --- */
      injectOnce("tq-lvl10-css",
        ".tq-l10-banner{margin:10px 0;padding:10px 14px;background:var(--panel-hi);" +
          "border:1px solid var(--line);border-radius:10px;text-align:center}" +
        ".tq-l10-vocab{margin:8px 0 0;padding-left:20px;display:flex;flex-direction:column;gap:7px;line-height:1.5}" +
        ".tq-l10-controls{display:flex;gap:18px;flex-wrap:wrap;margin:14px 0;align-items:flex-end}" +
        ".tq-l10-controls>*{min-width:180px;flex:1 1 180px}" +
        ".tq-l10-toggle-cell{display:flex;align-items:flex-end;min-height:38px}" +
        ".tq-l10-detail{margin-top:14px;display:flex;flex-direction:column;gap:10px;" +
          "background:var(--panel-hi);border:1px solid var(--line);border-radius:12px;padding:14px}" +
        ".tq-l10-caption{font-size:13.5px;color:var(--ink-soft);line-height:1.5}" +
        ".tq-l10-chips{display:flex;gap:8px;flex-wrap:wrap;align-items:center}" +
        ".tq-l10-blend{display:flex;flex-direction:column;gap:6px;margin-top:4px}" +
        ".tq-l10-blend-lbl{font-size:12.5px;color:var(--ink-mute);line-height:1.5}" +
        ".tq-l10-loadwrap{margin-top:16px}" +
        ".tq-l10-sub{font-size:12px;font-weight:700;color:var(--ink-mute);" +
          "text-transform:uppercase;letter-spacing:.04em;margin-bottom:8px}" +
        ".tq-l10-loadnote{margin-top:8px;font-size:13px;color:var(--ink-soft);line-height:1.5}" +
        ".tq-l10-presetbar{display:flex;align-items:center;gap:12px;margin:6px 0 12px;flex-wrap:wrap}" +
        ".tq-l10-preset-note{font-size:13px;color:var(--ink-soft);background:var(--panel-hi);" +
          "border:1px solid var(--line);border-radius:10px;padding:9px 12px;margin-bottom:14px;line-height:1.5}" +
        ".tq-l10-paramsmain{display:flex;gap:18px;flex-wrap:wrap}" +
        ".tq-l10-bars{display:flex;flex-direction:column;gap:14px;min-width:300px}" +
        ".tq-l10-frac{display:flex;flex-direction:column;gap:4px}" +
        ".tq-l10-frac-cap{font-size:12px;font-weight:700;color:var(--ink-mute);" +
          "text-transform:uppercase;letter-spacing:.04em}" +
        ".tq-l10-frac-big{font-size:30px;font-weight:800;color:var(--ink);" +
          "font-variant-numeric:tabular-nums;line-height:1}" +
        ".tq-l10-frac-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}" +
        ".tq-l10-frac-note{font-family:var(--mono);font-size:12px;color:var(--ink-mute)}" +
        ".tq-l10-formula{display:flex;flex-direction:column;gap:6px;margin-top:4px}" +
        /* L6 mask idiom: dim non-selected experts; brighten the chosen top-k */
        ".tq-hm-cell.is-masked{opacity:.3;filter:saturate(.4)}" +
        ".tq-hm-cell.is-masked .tq-hm-val{color:var(--ink-mute)}" +
        ".tq-hm-cell.is-masked.is-rowsel{opacity:.42}" +
        ".tq-hm-cell.is-chosen{box-shadow:0 0 0 2px var(--accent) inset}");
    },

    quiz: [
      {
        q: "In a top-2-of-8 MoE layer (like Mixtral 8×7B), which statement about parameters and compute is correct?",
        choices: [
          "Both total parameters and per-token compute scale with all 8 experts",
          "Total parameters scale with all 8 experts (capacity), but per-token compute scales with only the 2 selected experts plus the router — MoE decouples parameter count from per-token FLOPs",
          "Per-token compute scales with 8 experts but only 2 are ever stored in memory",
          "Total and active parameters are always equal in an MoE layer"
        ],
        answer: 1,
        explain: "MoE's whole point is the decoupling: TOTAL params grow with N (the 8 experts = capacity/knowledge), " +
          "but only the k=2 selected experts (+ router, + any shared) actually run per token. Mixtral 8×7B is " +
          "~47B total but only ~13B active per token."
      },
      {
        q: "How does an MoE router decide which experts a token uses, and at what granularity?",
        choices: [
          "It picks experts once per sequence, so every token in a sentence uses the same experts",
          "A linear gate g(x)=x·W_g gives a logit per expert; softmax → gate weights; the top-k are selected and renormalized to sum to 1 — and routing is PER TOKEN and independent, so different tokens can use different experts",
          "It always sends every token to every expert and averages the results",
          "The router is trained separately and frozen, so routing never depends on the token"
        ],
        answer: 1,
        explain: "The router is one linear layer: logits = x·W_g, softmax to gate weights, keep the top-k and " +
          "renormalize them to Σ=1. Crucially routing is PER TOKEN and independent — 'cat' and 'sat' can land on " +
          "completely different experts."
      },
      {
        q: "What is a shared expert (as in DeepSeek-V3), and how does it differ from a routed expert?",
        choices: [
          "A shared expert is one that multiple GPUs hold a copy of, for redundancy",
          "A shared expert is applied to EVERY token (never gated off) to capture common knowledge, whereas routed experts are sparsely top-k selected and specialize — so it adds to both active and total params: active/token ≈ (k + n_shared)/(N + n_shared)",
          "A shared expert replaces the router entirely",
          "A shared expert is only used during training and removed at inference"
        ],
        answer: 1,
        explain: "A shared expert always runs on every token (common knowledge), so it adds to both the active and " +
          "the total parameter count. Routed experts are the ones the router top-k selects, and they specialize. " +
          "active/token ≈ (k + n_shared)/(N + n_shared)."
      },
      {
        q: "Why do MoE models need load balancing, and how is it usually provided?",
        choices: [
          "To make the experts all produce the same output, for consistency",
          "Without it the router can collapse onto a few experts — starving the rest and wasting capacity — so it's fixed with an auxiliary load-balance loss (Switch/GShard) or DeepSeek-V3's aux-loss-free per-expert bias",
          "To ensure every token always uses all N experts equally",
          "Load balancing is purely a hardware concern and has no effect on training"
        ],
        answer: 1,
        explain: "A free-running router tends to collapse onto its favorite experts, leaving the rest untrained and " +
          "useless (you saw the load bar pile up with the collapsed toggle). The fix is an auxiliary load-balance " +
          "loss, or DeepSeek-V3's aux-loss-free per-expert bias that nudges usage toward even."
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
