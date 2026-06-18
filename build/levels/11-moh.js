/* ============================================================================
 * LEVEL 11 — Mixture of Heads (MoH)   (order 11)
 * ----------------------------------------------------------------------------
 * Follows the gold-standard pattern (see 04-multi-head.js / 08-mha-mqa-gqa.js):
 *   - one IIFE, ONE TQ.registerLevel call, NO other globals
 *   - render() = friendly intro -> PRIMARY interactive -> TQ.code -> TQ.resources
 *   - every number genuinely computed via TQ math (NO hardcoded routing/gate nums)
 *   - all color via TQ.colorFor / TQ.heatmap / TQ.vectorView / TQ.barRow
 *
 * Math shown, all real & deterministic (arXiv 2410.11842):
 *   - shared toy sentence -> TQ.toy.embeddings (6 x 16)
 *   - H = 8 heads, each with its OWN seeded Wq/Wk/Wv via TQ.toyQKV(dk=2,
 *     base + h*10 + {1,2,3}) — EXACTLY as L4 builds heads. dk = 16/8 = 2.
 *     per-head attention via TQ.attention -> .weights (6x6) and .output (6x2).
 *   - HEAD ROUTER: Wr = TQ.randMatrix(dModel, H, 700)  (== nn.Linear(d_model,H))
 *     gate logits = TQ.matmul(X, Wr) -> 6x8 logit matrix.
 *   - per token: local top-k helper sorts the logit row desc, takes k indices;
 *     router weights g = TQ.softmax over ONLY the selected logits (sum ~1.00).
 *   - MoH output / token = sum over selected h of g_h * head_h.output[token]
 *     (a real weighted sum of 2-dim head vectors), then x Wo
 *     (TQ.randMatrix(H*dk, dModel, 999)) -> 6x16.
 *   - MHA baseline = the SAME head outputs concat'd (equal weight 1) x Wo -> 6x16.
 *   - shared heads: first s heads ALWAYS active (DeepSeek-style); remaining k-s
 *     chosen by router top-k among the routed heads; weights re-softmaxed over
 *     the active set. All recomputed live.
 *   - active-vs-total compute: k/H fraction + TQ.barRow comparing active-head
 *     FLOPs (∝ k) of MoH vs MHA (∝ H) — same style as L8's calculator.
 * ========================================================================== */

(function () {
  "use strict";

  TQ.registerLevel({
    id: "moh",
    order: 11,
    title: "Mixture of Heads (MoH)",
    icon: "🎭",
    tagline: "MoE applied to attention heads: route per token, keep top-k, weight them.",
    preread: "MoH paper (2410.11842)",
    objectives: [
      "See that standard multi-head attention secretly SUMS all heads with equal weight 1",
      "Add a router that scores heads per token, keeps the top-k, and WEIGHTS them",
      "Read active-vs-total compute: grow total heads H without growing active compute (k)"
    ],

    render: function (root) {
      var toy = TQ.toy;
      var tokens = toy.tokens;            // ["The","cat","sat","on","the","mat"]
      var X = toy.embeddings;            // 6 x 16
      var dModel = toy.dModel;          // 16
      var H = 8;                        // attention heads (the "experts")
      var dk = dModel / H;             // 2
      var n = tokens.length;          // 6
      var BASE = 1100;               // seed base; per-head = BASE + h*10 + {1,2,3}

      /* ----------------------------------------- build every head (real, L4) */
      // heads[h] = { att } where att = {scores, scaled, weights, output, dk}.
      // EXACTLY L4's construction: each head its own seeded Wq/Wk/Wv.
      var heads = [];
      for (var h = 0; h < H; h++) {
        var qkv = TQ.toyQKV(dk, BASE + h * 10 + 1, BASE + h * 10 + 2, BASE + h * 10 + 3);
        heads.push({ att: TQ.attention(qkv.Q, qkv.K, qkv.V) }); // weights 6x6, output 6x2
      }

      /* --------------------------------------------- the HEAD ROUTER (real) */
      // Wr is literally nn.Linear(d_model, H): one logit per HEAD per token.
      var Wr = TQ.randMatrix(dModel, H, 700);     // 16 x 8
      var gateLogits = TQ.matmul(X, Wr);          // 6 x 8 — genuinely computed
      // Wo: (H*dk) x dModel = 16 x 16, the shared output projection (as L4).
      var Wo = TQ.randMatrix(H * dk, dModel, 999); // 16 x 16

      /* ---- selection + softmax-over-selected, with DeepSeek-style shared heads.
       * Returns, per token, the ACTIVE head set and the router weight on each
       * active head (softmax over ONLY the active logits -> sums to ~1.00).
       * Shared heads (first s) are ALWAYS active; the remaining (k - s) slots
       * are filled by the router's top picks among the NON-shared heads. The
       * softmax is taken over the union (shared + routed) logits — so shared
       * heads still get a learned weight, they just can't be gated off. */
      function routeToken(tokIdx, k, s) {
        var logits = gateLogits[tokIdx];
        s = TQ.clamp(s, 0, k);
        var shared = [];
        for (var i = 0; i < s; i++) shared.push(i);           // heads 0..s-1 forced on
        // candidate routed heads = all heads NOT in the shared set
        var cand = [];
        for (var j = s; j < H; j++) cand.push(j);
        // sort candidates by logit desc (stable enough for a toy), take top (k - s)
        cand.sort(function (a, b) { return logits[b] - logits[a]; });
        var routed = cand.slice(0, Math.max(0, k - s));
        var active = shared.concat(routed);
        active.sort(function (a, b) { return a - b; });        // keep head order tidy
        // softmax over ONLY the active logits -> learned weights that sum to ~1
        var selLogits = active.map(function (hi) { return logits[hi]; });
        var w = TQ.softmax(selLogits);                          // genuinely computed
        var gByHead = {};
        for (var p = 0; p < active.length; p++) gByHead[active[p]] = w[p];
        return { active: active, shared: shared, routed: routed, gByHead: gByHead };
      }

      // MoH final vector for a token given (k, s): weighted-sparse sum then x Wo.
      function mohFinal(tokIdx, k, s) {
        var rt = routeToken(tokIdx, k, s);
        // weighted, SPARSE concat: place g_h * head_h.output in each head's slot,
        // zero in the dropped slots. Width stays H*dk = d_model so Wo is unchanged.
        var concat = TQ.zeros(H * dk);
        for (var hi = 0; hi < H; hi++) {
          var g = rt.gByHead[hi];
          if (g === undefined) continue;                        // dropped head -> 0s
          var ov = heads[hi].att.output[tokIdx];                // length dk
          for (var d = 0; d < dk; d++) concat[hi * dk + d] = g * ov[d];
        }
        var fin = TQ.matmul([concat], Wo)[0];                   // 1 x 16 -> 16
        return { rt: rt, concat: concat, final: fin };
      }

      // Standard MHA final vector for a token: plain concat (equal weight 1) x Wo.
      function mhaFinal(tokIdx) {
        var concat = [];
        for (var hi = 0; hi < H; hi++) {
          var ov = heads[hi].att.output[tokIdx];
          for (var d = 0; d < dk; d++) concat.push(ov[d]);      // implicit weight 1
        }
        return { concat: concat, final: TQ.matmul([concat], Wo)[0] };
      }

      /* ===================================================================== *
       *  SECTION 1 — intro narrative: MHA secretly SUMS all heads (weight 1)
       * ===================================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "Multi-head attention secretly sums all heads — with equal weight 1"),
        TQ.p(
          "Back in ", TQ.el("strong", { text: "Level 4" }), " you built ",
          TQ.math("MultiHead(X) = Concat(head_1..head_H)·Wo"),
          ". Look closely at what ", TQ.math("Concat·Wo"), " actually does. Stacking the ",
          TQ.el("strong", { text: "H" }), " head outputs side by side and multiplying by ", TQ.math("Wo"),
          " is mathematically the same as ", TQ.el("strong", { text: "summing each head's contribution" }),
          " — each head writes into its own slice of the concatenated vector, and ", TQ.math("Wo"),
          " adds those slices up. Every head shows up in that sum with an ",
          TQ.el("strong", { text: "implicit weight of exactly 1" }), "."
        ),
        TQ.p(
          "That's the quiet limitation. Standard MHA gives every head the same say for ",
          TQ.el("em", { text: "every" }), " token. The model never gets to announce ",
          "\"head 5 matters a lot for THIS word, head 2 barely matters here.\" ",
          "It also runs all ", TQ.math("H"), " heads for every token, whether they help or not. ",
          "That is exactly the door ", TQ.el("strong", { text: "Mixture of Heads (MoH)" }), " walks through."
        ),
        TQ.callout(
          "Reframe: Concat·Wo IS a sum, and the implicit per-head weight is 1. MoH keeps the same head " +
          "outputs but replaces that equal, dense sum with a learned, SPARSE weighted sum."
        )
      ));

      /* ===================================================================== *
       *  SECTION 2 — "MoE, but the experts are attention heads" + orienting fig
       * ===================================================================== */
      var moeBlock = TQ.block(
        TQ.h(2, "MoE, but the experts are attention heads"),
        TQ.p(
          "Mixture-of-Experts (MoE) — from ", TQ.el("strong", { text: "Level 10" }),
          " — routes each token to a sparse ", TQ.math("top-k"),
          " subset of FFN experts and combines them with ", TQ.el("strong", { text: "learned gate weights" }),
          ". MoH does the identical move one level down: treat each ", TQ.el("strong", { text: "attention head" }),
          " as an expert, add a ", TQ.el("strong", { text: "router" }), " that scores the heads per token, ",
          "keep only the ", TQ.math("top-k"), ", and combine them with learned weights. The mechanism — router, ",
          TQ.math("top-k"), ", softmax gate — is the same; the experts are heads instead of FFNs."
        ),
        TQ.p(
          "So MoH stacks ", TQ.el("strong", { text: "two upgrades" }), " on top of L4's multi-head attention: ",
          "(1) heads are ", TQ.el("strong", { text: "weighted, not equal" }),
          " — the model decides which heads matter for THIS token; and (2) it is ",
          TQ.el("strong", { text: "sparse" }), " — only ", TQ.math("k"), " of ", TQ.math("H"),
          " heads run per token, so you can grow total heads ", TQ.math("H"),
          " without growing active compute."
        )
      );
      // Orienting figure BEFORE the interactive: the MHA->MoH structural diff.
      // Theme colors via CSS vars / currentColor only — no hardcoded hex.
      moeBlock.appendChild(TQ.figure(
        '<svg viewBox="0 0 720 250" width="720" height="250" role="img" ' +
          'aria-label="MHA sums all H heads with equal weight one; MoH routes per token, keeps top-k heads, and weights them" ' +
          'font-family="var(--mono)" font-size="11">' +
          '<defs><marker id="tq-l11-arrow" viewBox="0 0 10 10" refX="9" refY="5" ' +
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
            '<path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>' +
          // ---------- LEFT PANEL: MHA (equal dense sum) ----------
          '<text x="170" y="18" text-anchor="middle" fill="var(--ink-soft)" font-size="13" font-weight="600">MHA — equal dense sum</text>' +
          '<g stroke="currentColor" stroke-width="1.3" fill="none" color="var(--ink-faint)">' +
            '<line x1="118" y1="40"  x2="196" y2="116" marker-end="url(#tq-l11-arrow)"/>' +
            '<line x1="118" y1="74"  x2="196" y2="120" marker-end="url(#tq-l11-arrow)"/>' +
            '<line x1="118" y1="108" x2="196" y2="124" marker-end="url(#tq-l11-arrow)"/>' +
            '<line x1="118" y1="142" x2="196" y2="128" marker-end="url(#tq-l11-arrow)"/>' +
            '<line x1="118" y1="176" x2="196" y2="132" marker-end="url(#tq-l11-arrow)"/>' +
          '</g>' +
          '<g>' +
            '<rect x="40" y="30"  width="78" height="20" rx="4" fill="var(--panel-hi)" stroke="var(--line)"/><text x="79" y="44"  text-anchor="middle" fill="var(--ink)">head 1</text>' +
            '<rect x="40" y="64"  width="78" height="20" rx="4" fill="var(--panel-hi)" stroke="var(--line)"/><text x="79" y="78"  text-anchor="middle" fill="var(--ink)">head 2</text>' +
            '<rect x="40" y="98"  width="78" height="20" rx="4" fill="var(--panel-hi)" stroke="var(--line)"/><text x="79" y="112" text-anchor="middle" fill="var(--ink)">head 3</text>' +
            '<rect x="40" y="132" width="78" height="20" rx="4" fill="var(--panel-hi)" stroke="var(--line)"/><text x="79" y="146" text-anchor="middle" fill="var(--ink)">…</text>' +
            '<rect x="40" y="166" width="78" height="20" rx="4" fill="var(--panel-hi)" stroke="var(--line)"/><text x="79" y="180" text-anchor="middle" fill="var(--ink)">head H</text>' +
          '</g>' +
          // ×1 weight tags on every arrow
          '<g fill="var(--ink-mute)" font-size="9" text-anchor="middle">' +
            '<text x="152" y="72">×1</text><text x="152" y="92">×1</text><text x="152" y="112">×1</text>' +
            '<text x="152" y="132">×1</text><text x="152" y="152">×1</text>' +
          '</g>' +
          '<circle cx="210" cy="124" r="16" fill="var(--panel-hi)" stroke="var(--line)"/>' +
          '<text x="210" y="129" text-anchor="middle" fill="var(--ink)" font-size="15">Σ</text>' +
          '<line x1="226" y1="124" x2="252" y2="124" stroke="currentColor" color="var(--ink-faint)" marker-end="url(#tq-l11-arrow)"/>' +
          '<rect x="252" y="112" width="58" height="24" rx="6" fill="var(--panel-hi)" stroke="var(--line)"/><text x="281" y="128" text-anchor="middle" fill="var(--ink)">·Wo</text>' +
          // divider
          '<line x1="360" y1="20" x2="360" y2="232" stroke="var(--line-soft)" stroke-width="1"/>' +
          // ---------- RIGHT PANEL: MoH (sparse weighted sum) ----------
          '<text x="540" y="18" text-anchor="middle" fill="var(--accent)" font-size="13" font-weight="600">MoH — sparse weighted sum</text>' +
          '<rect x="430" y="28" width="96" height="22" rx="6" fill="var(--panel-hi)" stroke="var(--accent-2)"/><text x="478" y="43" text-anchor="middle" fill="var(--accent-2)">router</text>' +
          // router fans a faint score line to each head
          '<g stroke="var(--accent-2)" stroke-width="0.9" fill="none" opacity="0.55">' +
            '<line x1="430" y1="42" x2="528" y2="74"/><line x1="430" y1="42" x2="528" y2="108"/>' +
            '<line x1="430" y1="42" x2="528" y2="142"/><line x1="430" y1="42" x2="528" y2="176"/>' +
          '</g>' +
          '<g>' +
            '<rect x="528" y="64"  width="78" height="20" rx="4" fill="var(--panel-hi)" stroke="var(--accent)"/><text x="567" y="78"  text-anchor="middle" fill="var(--ink)">head 1 ✓</text>' +
            '<rect x="528" y="98"  width="78" height="20" rx="4" fill="var(--panel)" stroke="var(--line-soft)" stroke-dasharray="4 3"/><text x="567" y="112" text-anchor="middle" fill="var(--ink-faint)">head 2 ✕</text>' +
            '<rect x="528" y="132" width="78" height="20" rx="4" fill="var(--panel-hi)" stroke="var(--accent)"/><text x="567" y="146" text-anchor="middle" fill="var(--ink)">head 3 ✓</text>' +
            '<rect x="528" y="166" width="78" height="20" rx="4" fill="var(--panel)" stroke="var(--line-soft)" stroke-dasharray="4 3"/><text x="567" y="180" text-anchor="middle" fill="var(--ink-faint)">head H ✕</text>' +
          '</g>' +
          // selected arrows (solid, tagged g) + dropped arrows (dashed/grey)
          '<g stroke="var(--accent)" stroke-width="1.5" fill="none">' +
            '<line x1="606" y1="74"  x2="648" y2="120" marker-end="url(#tq-l11-arrow)"/>' +
            '<line x1="606" y1="142" x2="648" y2="128" marker-end="url(#tq-l11-arrow)"/>' +
          '</g>' +
          '<g stroke="var(--line-soft)" stroke-width="1" fill="none" stroke-dasharray="4 3" opacity="0.6">' +
            '<line x1="606" y1="108" x2="648" y2="122"/>' +
            '<line x1="606" y1="176" x2="648" y2="126"/>' +
          '</g>' +
          '<g fill="var(--accent)" font-size="9" text-anchor="middle">' +
            '<text x="626" y="92">×g₁</text><text x="626" y="150">×g₃</text>' +
          '</g>' +
          '<circle cx="662" cy="124" r="16" fill="var(--panel-hi)" stroke="var(--accent)"/>' +
          '<text x="662" y="129" text-anchor="middle" fill="var(--ink)" font-size="15">Σ</text>' +
          '<line x1="678" y1="124" x2="700" y2="124" stroke="var(--accent)" marker-end="url(#tq-l11-arrow)"/>' +
          '<text x="540" y="232" text-anchor="middle" fill="var(--ink-mute)" font-size="10">only top-k heads kept, each with a learned weight gₕ</text>' +
        '</svg>',
        "The whole level is one structural diff against L4. LEFT: MHA feeds all H heads into a Σ with an " +
        "implicit ×1 on every arrow — equal, dense. RIGHT: a router scores the heads per token, keeps only " +
        "the top-k (solid, weighted by gₕ), drops the rest (dashed), then Σ·Wo. Equal dense sum → learned sparse weighted sum."
      ));
      root.appendChild(moeBlock);

      /* ===================================================================== *
       *  PRIMARY INTERACTIVE — the head router (heatmap + combine + contrast)
       * ===================================================================== */
      // ---- shared live state (read fresh on every recompute) --------------
      var state = {
        k: 4,            // top-k heads kept per token (1..H)
        s: 0,            // shared heads (always active), 0..k
        tok: 1,          // traced token (default 'cat')
        showMHA: false   // combine panel mode: false=MoH, true=standard MHA
      };

      // forward decls so cross-updates work
      var heatHolder = TQ.el("div", { class: "tq-grow" });
      var combinePanel = TQ.el("div", { class: "tq-panel tq-grow tq-moh-combine" });
      var contrastPanel = TQ.el("div", { class: "tq-panel tq-grow tq-moh-contrast" });
      var computePanel = TQ.el("div", { class: "tq-panel tq-grow tq-moh-compute" });

      var headLabels = [];
      for (var hl = 0; hl < H; hl++) headLabels.push("h" + (hl + 1));

      /* ---- the router heatmap: rows=tokens, cols=heads, colored by the
       * router gate weight g_h (per token, softmax over the active set; dropped
       * heads -> 0). Selected cells get an outline, shared an extra badge, the
       * traced row a row-highlight. Clicking a row sets the traced token. */
      function buildHeatmap() {
        // gateWeights[t][h] = g_h if active else 0 (genuinely computed per row)
        var gateWeights = [];
        var rtByTok = [];
        for (var t = 0; t < n; t++) {
          var rt = routeToken(t, state.k, state.s);
          rtByTok.push(rt);
          var row = [];
          for (var hh = 0; hh < H; hh++) {
            row.push(rt.gByHead[hh] === undefined ? 0 : rt.gByHead[hh]);
          }
          gateWeights.push(row);
        }
        var heat = TQ.heatmap(gateWeights, {
          rowLabels: tokens, colLabels: headLabels, cellSize: 46,
          min: 0, max: 1, selectedRow: state.tok,
          format: function (v) { return v < 1e-9 ? "·" : TQ.fmt(v, 2); }
        });
        // decorate cells: outline selected, badge shared, dim dropped; wire clicks
        var cells = heat.querySelectorAll(".tq-hm-cell");
        var idx = 0;
        for (var r = 0; r < n; r++) {
          var rt2 = rtByTok[r];
          for (var c = 0; c < H; c++) {
            (function (rr, cc, cell) {
              var isShared = rt2.shared.indexOf(cc) !== -1;
              var isRouted = rt2.routed.indexOf(cc) !== -1;
              var isActive = isShared || isRouted;
              if (isActive) cell.classList.add("tq-moh-on");
              else cell.classList.add("tq-moh-off");
              if (isShared) cell.classList.add("tq-moh-shared");
              // clickable row -> set traced token
              cell.classList.add("is-clickable");
              cell.setAttribute("role", "button");
              cell.setAttribute("tabindex", "0");
              var g = gateWeights[rr][cc];
              cell.title = tokens[rr] + " · " + headLabels[cc] + " · " +
                (isShared ? "SHARED (always on) g=" + TQ.fmt(g, 3)
                  : isRouted ? "routed (top-k) g=" + TQ.fmt(g, 3)
                    : "dropped (not selected)");
              cell.addEventListener("click", function () { setTok(rr); });
              cell.addEventListener("keydown", function (e) {
                if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setTok(rr); }
              });
            })(r, c, cells[idx]);
            idx++;
          }
        }
        return heat;
      }

      function refreshHeatmap() {
        heatHolder.innerHTML = "";
        heatHolder.appendChild(buildHeatmap());
      }

      function setTok(t) {
        state.tok = t;
        tokSel.set(t);
        refreshHeatmap();
        renderCombine();
        renderContrast();
      }

      /* ---- combine panel: for the traced token, each ACTIVE head's 2-dim
       * output × its router weight g_h, the weighted sum, then ·Wo -> 16-dim.
       * In MHA mode: all H heads, equal weight 1, plain sum. */
      function renderCombine() {
        combinePanel.innerHTML = "";
        var t = state.tok;
        var mode = state.showMHA ? "MHA" : "MoH";

        combinePanel.appendChild(TQ.el("div", { class: "tq-moh-combine-head" },
          TQ.el("span", { class: "tq-stat-cap", text: "combining heads for token" }),
          TQ.el("span", { class: "tq-moh-qtoken", text: "\"" + tokens[t] + "\" (#" + t + ")" }),
          TQ.badge(mode + " mode", state.showMHA ? "default" : "good")
        ));

        var rt = routeToken(t, state.k, state.s);
        var headRows = TQ.el("div", { class: "tq-moh-headvecs" });

        // weighted concat we sum into (for the live sum vector readout)
        var summed = TQ.zeros(dk);

        for (var hi = 0; hi < H; hi++) {
          var active = state.showMHA ? true : (rt.gByHead[hi] !== undefined);
          var g = state.showMHA ? 1 : rt.gByHead[hi];           // MHA: weight 1
          var isShared = rt.shared.indexOf(hi) !== -1;
          if (active) {
            var ov = heads[hi].att.output[t];                   // length dk
            var weighted = [];
            for (var d = 0; d < dk; d++) {
              var wv = g * ov[d];
              weighted.push(wv);
              summed[d] += wv;
            }
            var label = TQ.el("div", { class: "tq-moh-headvec-label" },
              TQ.el("span", { text: "head " + (hi + 1) }),
              TQ.badge("g=" + TQ.fmt(g, 2), state.showMHA ? "default" : "info"),
              (!state.showMHA && isShared) ? TQ.badge("shared", "good") : null
            );
            headRows.appendChild(TQ.el("div", { class: "tq-moh-headvec" },
              label,
              TQ.vectorView(weighted, { cellSize: 30, showValues: true,
                max: TQ.maxOf(ov.map(function (x) { return Math.abs(x); })) || 1 })
            ));
          } else {
            headRows.appendChild(TQ.el("div", { class: "tq-moh-headvec is-dropped" },
              TQ.el("div", { class: "tq-moh-headvec-label" },
                TQ.el("span", { text: "head " + (hi + 1) }),
                TQ.badge("dropped", "default")),
              TQ.el("div", { class: "tq-moh-dropbox", text: "not selected · contributes 0" })
            ));
          }
        }

        // step 1: per-head weighted vectors
        var nActive = state.showMHA ? H : rt.active.length;
        combinePanel.appendChild(TQ.el("div", { class: "tq-moh-step" },
          TQ.el("div", { class: "tq-moh-step-label",
            text: "1 · " + nActive + " active head" + (nActive === 1 ? "" : "s") +
              " × weight (" + (state.showMHA ? "all 1" : "learned gₕ") + ")" }),
          headRows
        ));

        // step 2: weighted sum -> a single dk-vector (per-head slot view)
        combinePanel.appendChild(TQ.el("div", { class: "tq-moh-step" },
          TQ.el("div", { class: "tq-moh-step-label",
            text: "2 · weighted sum over active heads → " + dk + "-dim per-head blend" }),
          TQ.vectorView(summed, { cellSize: 26, showValues: true })
        ));

        // step 3: the real path is the sparse H*dk concat × Wo -> 16-dim
        var out = state.showMHA ? mhaFinal(t) : mohFinal(t, state.k, state.s);
        combinePanel.appendChild(TQ.el("div", { class: "tq-moh-step" },
          TQ.el("div", { class: "tq-moh-step-label",
            text: "3 · place in head slots (" + (H * dk) + "-wide) × Wo → final " + dModel + "-dim vector" }),
          TQ.vectorView(out.final, { cellSize: 22, showValues: true })
        ));

        // weights-sum sanity readout (proves softmax-over-selected sums to ~1)
        if (!state.showMHA) {
          var wsum = 0;
          for (var ai = 0; ai < rt.active.length; ai++) wsum += rt.gByHead[rt.active[ai]];
          combinePanel.appendChild(TQ.note(
            "Router weights on the " + rt.active.length + " active head" +
            (rt.active.length === 1 ? "" : "s") + " are a softmax over ONLY those logits, so they sit in [0,1] " +
            "and sum to " + TQ.fmt(wsum, 2) + " for \"" + tokens[t] + "\". Dropped heads contribute exactly 0. " +
            (state.s > 0
              ? ("The first " + state.s + " head" + (state.s === 1 ? " is" : "s are") +
                 " SHARED — always active regardless of their score.")
              : "Flip \"Show standard MHA\" to watch this collapse to all " + H + " heads at equal weight 1.")
          ));
        } else {
          combinePanel.appendChild(TQ.note(
            "Standard MHA: all " + H + " heads contribute with implicit weight 1 (a plain, equal, dense sum). " +
            "No head can be turned up or down for this token. Flip back to MoH to route + weight them."
          ));
        }
      }

      /* ---- side-by-side contrast: MHA strip vs MoH strip for the SAME token,
       * plus which heads were kept/dropped and the cosine between the two finals. */
      function renderContrast() {
        contrastPanel.innerHTML = "";
        var t = state.tok;
        var rt = routeToken(t, state.k, state.s);
        var mha = mhaFinal(t);
        var moh = mohFinal(t, state.k, state.s);

        contrastPanel.appendChild(TQ.el("div", { class: "tq-moh-combine-head" },
          TQ.el("span", { class: "tq-stat-cap", text: "same token, two ways to combine" }),
          TQ.el("span", { class: "tq-moh-qtoken", text: "\"" + tokens[t] + "\" (#" + t + ")" })
        ));

        // MHA strip — all heads, equal weight 1
        var mhaChips = TQ.el("div", { class: "tq-moh-chiprow" });
        for (var a = 0; a < H; a++) {
          mhaChips.appendChild(TQ.el("span", { class: "tq-moh-chip is-on" },
            "h" + (a + 1), TQ.el("span", { class: "tq-moh-chip-w", text: "1" })));
        }
        contrastPanel.appendChild(TQ.el("div", { class: "tq-moh-strip" },
          TQ.el("div", { class: "tq-moh-strip-head" },
            TQ.badge("MHA", "default"),
            TQ.el("span", { text: "all " + H + " heads · equal weight 1 · plain sum" })),
          mhaChips,
          TQ.vectorView(mha.final, { cellSize: 20, showValues: false })
        ));

        // MoH strip — only active heads, learned weights
        var mohChips = TQ.el("div", { class: "tq-moh-chiprow" });
        for (var b = 0; b < H; b++) {
          var g = rt.gByHead[b];
          var on = g !== undefined;
          var isShared = rt.shared.indexOf(b) !== -1;
          mohChips.appendChild(TQ.el("span", {
            class: "tq-moh-chip" + (on ? " is-on" : " is-off") + (isShared ? " is-shared" : "")
          }, "h" + (b + 1), on ? TQ.el("span", { class: "tq-moh-chip-w", text: TQ.fmt(g, 2) }) : null));
        }
        contrastPanel.appendChild(TQ.el("div", { class: "tq-moh-strip" },
          TQ.el("div", { class: "tq-moh-strip-head" },
            TQ.badge("MoH", "good"),
            TQ.el("span", { text: state.k + " of " + H + " heads · learned weights gₕ · weighted sparse sum" })),
          mohChips,
          TQ.vectorView(moh.final, { cellSize: 20, showValues: false })
        ));

        // kept / dropped readout + cosine between the two final vectors
        var kept = rt.active.map(function (hi) { return "h" + (hi + 1); });
        var dropped = [];
        for (var dh = 0; dh < H; dh++) if (rt.gByHead[dh] === undefined) dropped.push("h" + (dh + 1));
        var cos = TQ.cosine(mha.final, moh.final);
        contrastPanel.appendChild(TQ.el("div", { class: "tq-moh-readout" },
          TQ.badge("kept: " + kept.join(", "), "good"),
          TQ.badge("dropped: " + (dropped.length ? dropped.join(", ") : "none"), dropped.length ? "warn" : "default"),
          TQ.badge("cos(MHA, MoH) = " + TQ.fmt(cos, 2), "info")
        ));
        contrastPanel.appendChild(TQ.note(
          "Same head outputs feed both. MHA sums all " + H + " with weight 1; MoH keeps only " +
          kept.length + " and weights them by the router. The two final " + dModel +
          "-dim vectors differ (cosine " + TQ.fmt(cos, 2) + ") — MoH has genuinely reshaped this token's mix, " +
          "for free at decode because dropped heads never run."
        ));
      }

      // ---- controls -------------------------------------------------------
      var kSlider = TQ.slider({
        min: 1, max: H, step: 1, value: state.k, label: "Top-k heads kept",
        format: function (v) { return Math.round(v) + " / " + H; },
        onInput: function (v) {
          state.k = Math.round(v);
          if (state.s > state.k) { state.s = state.k; sShared.set(state.s); }
          // keep shared slider's max ≤ k
          var inp = sShared.el.querySelector(".tq-slider-input");
          if (inp) { inp.max = String(state.k); }
          refreshHeatmap(); renderCombine(); renderContrast(); renderCompute();
        }
      });

      var sShared = TQ.slider({
        min: 0, max: state.k, step: 1, value: state.s, label: "Shared heads (always on)",
        format: function (v) { return String(Math.round(v)); },
        onInput: function (v) {
          state.s = TQ.clamp(Math.round(v), 0, state.k);
          refreshHeatmap(); renderCombine(); renderContrast();
        }
      });

      var tokSel = TQ.segmented({
        options: tokens.map(function (tk, i) { return { label: tk, value: i }; }),
        value: state.tok,
        onChange: function (v) { state.tok = v; refreshHeatmap(); renderCombine(); renderContrast(); }
      });

      var mhaSeg = TQ.segmented({
        options: [{ label: "MoH (weighted sparse)", value: false }, { label: "Standard MHA (equal sum)", value: true }],
        value: state.showMHA,
        onChange: function (v) { state.showMHA = v; renderCombine(); }
      });

      // initial paint of the three live panels
      refreshHeatmap();
      renderCombine();
      renderContrast();

      var routerBlock = TQ.block(
        TQ.h(2, "The head router — score, select top-k, weight"),
        TQ.p(
          "The heatmap below is the live ", TQ.el("strong", { text: "router gate" }),
          ": rows are the 6 tokens, columns are the ", TQ.math("H = " + H),
          " heads, and each cell is colored by the router weight ", TQ.math("g_h"),
          " that token gives that head. The router is one ", TQ.math("nn.Linear(d_model, H)"),
          " (here ", TQ.math("Wr"), ", a real ", dModel + "×" + H,
          " matrix); the gate logits are ", TQ.math("X·Wr"),
          ". Outlined cells are the ", TQ.el("strong", { text: "selected" }), " top-k heads for that token; ",
          "dimmed cells were dropped. Drag ", TQ.el("strong", { text: "top-k" }),
          " and watch the sparsity change; every move re-runs the per-token softmax over the new selected set."
        ),
        TQ.el("div", { class: "tq-controls-row tq-moh-controls" },
          TQ.el("div", { class: "tq-grow" }, kSlider.el),
          TQ.el("div", { class: "tq-grow" }, sShared.el)
        ),
        TQ.el("div", { class: "tq-moh-tokbar" },
          TQ.el("span", { class: "tq-slider-label", text: "Traced token" }),
          tokSel.el
        ),
        TQ.el("div", { class: "tq-legend" },
          TQ.el("span", { text: "0 (dropped / low)" }),
          TQ.el("div", { class: "tq-legend-scale" }),
          TQ.el("span", { text: "1 (full weight)" })
        ),
        TQ.el("div", { class: "tq-flexrow tq-moh-routerrow" },
          TQ.el("div", { class: "tq-panel tq-grow tq-moh-heatwrap" },
            TQ.el("div", { class: "tq-heatmap-title", text: "router gate weights (token × head)" }),
            heatHolder
          ),
          combinePanel
        ),
        TQ.el("div", { class: "tq-moh-modebar" },
          TQ.el("span", { class: "tq-slider-label", text: "Combine panel mode" }),
          mhaSeg.el
        )
      );
      root.appendChild(routerBlock);

      /* ===================================================================== *
       *  SIDE-BY-SIDE — standard MHA vs MoH for the SAME token
       * ===================================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "Standard MHA vs MoH, same token"),
        TQ.p(
          "Both strips below use the ", TQ.el("strong", { text: "exact same head outputs" }),
          " and the ", TQ.el("strong", { text: "same token selector" }), " as the router above. ",
          "The only difference is the combine rule: MHA keeps all ", TQ.math("H"),
          " heads at weight 1 (a plain sum), MoH keeps the top-k and weights them. ",
          "The kept/dropped chips and the cosine between the two final vectors show how much the routing reshaped the token."
        ),
        contrastPanel
      ));

      /* ===================================================================== *
       *  SHARED + ROUTED heads (DeepSeek-style)
       * ===================================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "Shared + routed heads (DeepSeek-style)"),
        TQ.p(
          "Use the ", TQ.el("strong", { text: "shared heads" }), " slider in the router panel above. ",
          "The first ", TQ.math("s"), " heads become ", TQ.el("strong", { text: "shared" }),
          ": always active for every token (badged in the heatmap), capturing common patterns every token needs. ",
          "The remaining ", TQ.math("k − s"), " slots are filled by the router's top picks among the rest, and the ",
          "softmax weights are taken over the whole active set — so shared heads still get a learned weight, ",
          "they just can't be gated off. This is exactly DeepSeek's shared-expert split, applied to heads."
        ),
        TQ.callout(
          "Because MoH reuses the SAME head outputs as standard MHA, you don't have to train from scratch: " +
          "MoH can be obtained by CONTINUED-TRAINING an existing MHA model (e.g. LLaMA) into a MoH model — " +
          "bolt on the router, keep training, let it learn which heads matter per token."
        )
      ));

      /* ===================================================================== *
       *  ACTIVE vs TOTAL compute — k/H fraction + FLOPs bar (L8-style)
       * ===================================================================== */
      function renderCompute() {
        computePanel.innerHTML = "";
        var k = state.k;
        var frac = k / H;
        // active-head FLOPs ∝ k (MoH) vs ∝ H (MHA): one shared proportional unit.
        var bars = TQ.barRow([H, k], {
          labels: ["MHA active (H = " + H + ")", "MoH active (k = " + k + ")"],
          max: H,
          highlight: 1,
          format: function (v) { return Math.round(v) + " heads"; }
        });
        computePanel.appendChild(TQ.el("div", { class: "tq-moh-compstat" },
          TQ.el("span", { class: "tq-stat-cap", text: "active heads per token" }),
          TQ.el("span", { class: "tq-stat-big", text: k + " / " + H }),
          TQ.badge(TQ.fmt(frac * 100, 0) + "% of heads active", frac <= 0.5 ? "good" : "info")
        ));
        computePanel.appendChild(bars);
        computePanel.appendChild(TQ.math(
          "active fraction = k / H = " + k + " / " + H + " = " + TQ.fmt(frac, 2)
        ));
        computePanel.appendChild(TQ.note(
          "Per-token attention compute scales with the number of heads that actually RUN. MoH runs " + k +
          " of " + H + " (" + TQ.fmt(frac * 100, 0) + "%). The dropped heads cost nothing this token, so you can " +
          "grow total heads H — more specialists to route among — while active compute stays pinned near top-k. " +
          "Same active-vs-total framing as the KV-cache calculator in Level 8."
        ));
      }
      renderCompute();

      root.appendChild(TQ.block(
        TQ.h(2, "Active vs total compute"),
        TQ.p(
          "This is the payoff of sparsity. The bar compares ", TQ.el("strong", { text: "active" }),
          " heads: MHA always runs all ", TQ.math("H"), " heads; MoH runs only ", TQ.math("k"),
          ". The router lets you scale ", TQ.el("strong", { text: "total" }),
          " heads without scaling the heads that actually fire per token — more capacity, flat active cost. ",
          "Drag the top-k slider above and watch the MoH bar move."
        ),
        computePanel
      ));

      /* ===================================================================== *
       *  CODE — PyTorch MoH attention vs the equal-sum MHA baseline
       * ===================================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "MoH attention in PyTorch"),
        TQ.p(
          "Maps 1:1 onto this level's toy (", TQ.math("d_model = 16"), ", ", TQ.math("H = 8"), ", ",
          TQ.math("dk = 2"), ", ", TQ.math("top_k = 4"),
          "). The forward pass is identical to standard MHA up to the H head outputs; the only change is the last ",
          "few lines — a router scores the heads, top-k keeps a sparse subset, softmax gives the learned weights, ",
          "and the equal sum becomes a weighted sparse sum."
        ),
        TQ.code(
          "import torch\n" +
          "import torch.nn as nn\n" +
          "import torch.nn.functional as F\n" +
          "\n" +
          "class MoHAttention(nn.Module):\n" +
          "    \"\"\"Multi-head attention as a mixture of heads (arXiv 2410.11842).\n" +
          "    Same H head outputs as standard MHA, but a router picks top-k heads\n" +
          "    per token and WEIGHTS them, instead of summing all H with weight 1.\"\"\"\n" +
          "    def __init__(self, d_model=16, n_heads=8, top_k=4, n_shared=0):\n" +
          "        super().__init__()\n" +
          "        self.n_heads, self.d_k = n_heads, d_model // n_heads   # 8, 2\n" +
          "        self.top_k, self.n_shared = top_k, n_shared\n" +
          "        self.w_q = nn.Linear(d_model, d_model)\n" +
          "        self.w_k = nn.Linear(d_model, d_model)\n" +
          "        self.w_v = nn.Linear(d_model, d_model)\n" +
          "        self.w_o = nn.Linear(d_model, d_model)\n" +
          "        self.router = nn.Linear(d_model, n_heads)   # one score per HEAD\n" +
          "\n" +
          "    def forward(self, x):                            # x: (B, n, d_model)\n" +
          "        B, n, _ = x.shape\n" +
          "        # --- identical to standard MHA up to here: H head outputs ---\n" +
          "        def split(t): return t.view(B, n, self.n_heads, self.d_k).transpose(1, 2)\n" +
          "        q, k, v = split(self.w_q(x)), split(self.w_k(x)), split(self.w_v(x))\n" +
          "        scores = q @ k.transpose(-2, -1) / (self.d_k ** 0.5)   # (B,H,n,n)\n" +
          "        attn = F.softmax(scores, dim=-1)\n" +
          "        head_out = attn @ v                          # (B, H, n, d_k)\n" +
          "\n" +
          "        #   STANDARD MHA would now do: out = concat(head_out) -> equal sum, w=1\n" +
          "        #   concat == implicit sum of per-head slices, every head weight 1.\n" +
          "\n" +
          "        # --- MoH: route over heads, keep top-k, learn the weights ---\n" +
          "        gate = self.router(x)                        # (B, n, H) one logit / head\n" +
          "        sel = gate                                   # logits used only for SELECTION\n" +
          "        if self.n_shared:                            # DeepSeek-style shared heads\n" +
          "            sel = gate.clone()                       # force shared heads to always\n" +
          "            sel[..., :self.n_shared] = float('inf')  # rank in the top-k (selection)\n" +
          "        topi = sel.topk(self.top_k, dim=-1).indices  # pick k head indices per token\n" +
          "        topw = gate.gather(-1, topi)                 # their REAL logits (not inf)\n" +
          "        w = F.softmax(topw, dim=-1)                  # weights over SELECTED heads\n" +
          "        g = torch.zeros_like(gate).scatter(-1, topi, w)   # (B, n, H), sparse\n" +
          "        # weighted, SPARSE sum over heads (vs MHA's equal sum):\n" +
          "        ho = head_out.permute(0, 2, 1, 3)            # (B, n, H, d_k)\n" +
          "        out = (ho * g.unsqueeze(-1)).reshape(B, n, self.n_heads * self.d_k)\n" +
          "        return self.w_o(out)                         # (B, n, d_model)\n" +
          "\n" +
          "# MHA:  out = sum_h head_h                 (all H, weight 1)   -- dense, equal\n" +
          "# MoH:  out = sum_{h in TopK} g_h * head_h                     -- sparse, weighted\n",
          { lang: "python", label: "MoH attention",
            caption: "The only change from standard MHA is the last few lines: the router scores the H heads " +
              "per token, top-k keeps a sparse subset, softmax gives learned weights gₕ, and the equal sum " +
              "becomes a weighted sparse sum. Grow H freely; active compute stays ~top_k heads." }
        )
      ));

      /* ===================================================================== *
       *  WRAP-UP + GO DEEPER
       * ===================================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "Where this goes"),
        TQ.p(
          "MoH is one instance of a bigger pattern: ", TQ.el("strong", { text: "make the model sparse and routed" }),
          " wherever it was dense and uniform. MoE did it to the FFN, MoH does it to attention heads, and the " +
          "same router + top-k + learned-gate machinery shows up across modern large models."
        ),
        TQ.callout(
          "Carry forward: L4's equal sum of heads → MoH's weighted sparse sum. Same head outputs, " +
          "plus a router that says which heads matter for THIS token — and only those run."
        )
      ));

      root.appendChild(TQ.resources("Go deeper — MoH and its MoE roots", [
        { label: "MoH: Multi-Head Attention as Mixture-of-Head Attention (paper)", kind: "paper",
          url: "https://arxiv.org/abs/2410.11842",
          note: "the source — heads as experts, router + top-k + learned weights, shared heads, continued-training from MHA" },
        { label: "Hugging Face — Mixture of Experts Explained (blog)", kind: "blog",
          url: "https://huggingface.co/blog/moe",
          note: "the MoE mechanism MoH borrows: route a token to a sparse top-k of experts with learned gates" },
        { label: "DeepSeekMoE (fine-grained + shared experts, paper)", kind: "paper",
          url: "https://arxiv.org/abs/2401.06066",
          note: "the shared-expert split MoH reuses as 'shared heads' (always-on common work + routed specialists)" },
        { label: "The Illustrated Transformer (blog)", kind: "blog",
          url: "https://jalammar.github.io/illustrated-transformer/",
          note: "refresher on the Concat·Wo multi-head formula MoH upgrades" },
        { label: "Attention Is All You Need (paper)", kind: "paper",
          url: "https://arxiv.org/abs/1706.03762",
          note: "the original multi-head attention — the equal, dense sum MoH starts from" }
      ]));

      /* ----------------------------- scoped styles (CSS vars only, no hex) -- */
      injectOnce("tq-lvl11-css",
        ".tq-moh-controls{gap:18px;flex-wrap:wrap}" +
        ".tq-moh-controls>*{min-width:200px}" +
        ".tq-moh-tokbar{display:flex;align-items:center;gap:12px;margin:10px 0;flex-wrap:wrap}" +
        ".tq-moh-modebar{display:flex;align-items:center;gap:12px;margin-top:14px;flex-wrap:wrap}" +
        ".tq-moh-routerrow{align-items:stretch;gap:18px;flex-wrap:wrap;margin-top:8px}" +
        ".tq-moh-heatwrap{display:flex;flex-direction:column;gap:8px;min-width:320px}" +
        // selected / dropped / shared decorations on heatmap cells
        ".tq-moh-heatwrap .tq-hm-cell.tq-moh-on{outline:2px solid var(--accent);outline-offset:-2px;" +
          "z-index:1;border-radius:4px}" +
        ".tq-moh-heatwrap .tq-hm-cell.tq-moh-off{opacity:.34}" +
        ".tq-moh-heatwrap .tq-hm-cell.tq-moh-shared{outline:2px solid var(--good);" +
          "box-shadow:inset 0 0 0 2px var(--panel)}" +
        ".tq-moh-combine{display:flex;flex-direction:column;gap:14px;min-width:320px}" +
        ".tq-moh-combine-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap}" +
        ".tq-moh-qtoken{font-size:17px;font-weight:700}" +
        ".tq-moh-step{display:flex;flex-direction:column;gap:6px}" +
        ".tq-moh-step-label{font-size:12px;color:var(--ink-mute);font-family:var(--mono);letter-spacing:.02em}" +
        ".tq-moh-headvecs{display:flex;gap:14px;flex-wrap:wrap}" +
        ".tq-moh-headvec{display:flex;flex-direction:column;gap:4px;transition:opacity .2s}" +
        ".tq-moh-headvec.is-dropped{opacity:.4}" +
        ".tq-moh-headvec-label{display:flex;align-items:center;gap:6px;font-size:12px;" +
          "color:var(--ink-mute);font-family:var(--mono);flex-wrap:wrap}" +
        ".tq-moh-dropbox{font-family:var(--mono);font-size:11px;color:var(--ink-faint);" +
          "border:1px dashed var(--line-soft);border-radius:6px;padding:6px 8px}" +
        // contrast strips
        ".tq-moh-contrast{display:flex;flex-direction:column;gap:14px}" +
        ".tq-moh-strip{display:flex;flex-direction:column;gap:8px;padding:10px 12px;" +
          "background:var(--panel-hi);border:1px solid var(--line);border-radius:10px}" +
        ".tq-moh-strip-head{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--ink-soft);flex-wrap:wrap}" +
        ".tq-moh-chiprow{display:flex;gap:6px;flex-wrap:wrap}" +
        ".tq-moh-chip{display:inline-flex;align-items:center;gap:5px;font-family:var(--mono);font-size:11px;" +
          "padding:3px 7px;border-radius:6px;border:1px solid var(--line-soft);color:var(--ink-mute)}" +
        ".tq-moh-chip.is-on{border-color:var(--accent);color:var(--ink);background:var(--panel)}" +
        ".tq-moh-chip.is-off{opacity:.42;text-decoration:line-through}" +
        ".tq-moh-chip.is-shared{border-color:var(--good)}" +
        ".tq-moh-chip-w{font-size:10px;color:var(--accent-2);font-weight:600}" +
        ".tq-moh-readout{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:2px}" +
        // compute panel
        ".tq-moh-compute{display:flex;flex-direction:column;gap:12px}" +
        ".tq-moh-compstat{display:flex;align-items:center;gap:12px;flex-wrap:wrap}");
    },

    quiz: [
      {
        q: "Standard multi-head attention computes MultiHead(X) = Concat(head_1..head_H)·Wo. In what sense does MoH call this a special case, and what does MoH change?",
        choices: [
          "MoH uses more heads of the same size; the combine rule is unchanged",
          "Concat·Wo is mathematically a SUM of per-head contributions with every head at implicit weight 1; MoH replaces that equal, dense sum with a learned, sparse weighted sum over the router's top-k heads",
          "MoH changes the softmax inside each head's attention",
          "MoH removes Wo entirely and averages the heads"
        ],
        answer: 1,
        explain: "Concat then Wo adds up each head's slice — a dense sum where every head's implicit weight is 1. " +
                 "MoH keeps the same head outputs but combines them as out = sum over the top-k heads of g_h·head_h, " +
                 "then ·Wo: a learned, sparse, weighted sum instead of an equal one."
      },
      {
        q: "Why is MoH described as 'Mixture-of-Experts applied to attention heads'?",
        choices: [
          "Because both happen to use 8 of something",
          "Because MoH replaces the FFN with an attention layer",
          "MoE routes each token to a sparse top-k of FFN experts and combines them with learned gates; MoH does the identical mechanism with the ATTENTION HEADS as the experts",
          "Because heads and experts are unrelated and the name is just marketing"
        ],
        answer: 2,
        explain: "Same machinery — a router scores candidates per token, top-k keeps a sparse subset, a softmax gives " +
                 "learned combine weights — applied one level down: the 'experts' are attention heads instead of FFNs."
      },
      {
        q: "With H = 8 heads and top-k = 4, what is true about per-token compute and about the router weights on the selected heads?",
        choices: [
          "All 8 heads still run; MoH only reweights them afterward",
          "Only 4 of 8 heads run (k/H = 50% active), so you can grow H without growing active compute; the router weights on the 4 selected heads are a softmax over only those logits, so they are in [0,1] and sum to ~1.00 per token",
          "The weights on the selected heads sum to k (=4), not 1",
          "Per-token compute scales with H, not k"
        ],
        answer: 1,
        explain: "MoH runs only the top-k heads (4/8 = 50% here), so total heads H can grow while active compute stays " +
                 "near k. The gate is a softmax over ONLY the selected logits, so those weights live in [0,1] and sum to " +
                 "about 1.00 — and genuinely differ token to token."
      },
      {
        q: "What do SHARED heads add (DeepSeek-style), and can MoH be obtained without training from scratch?",
        choices: [
          "Shared heads are the ones the router drops; and MoH must always be trained from scratch",
          "Shared heads disable the router; and MoH can only be distilled from a larger model",
          "Shared heads are always active (never gated off) to capture common patterns every token needs, while the remaining top-(k−s) routed heads specialize per token; and yes — MoH can be obtained by continued-training an existing MHA model (e.g. LLaMA) into MoH",
          "Shared heads double the active compute; and MoH requires a brand-new architecture"
        ],
        answer: 2,
        explain: "Shared heads are forced active for every token (common, always-needed work), while the routed top-(k−s) " +
                 "heads specialize per token. Because MoH reuses the same head outputs as MHA, you can continued-train an " +
                 "existing MHA model (e.g. LLaMA) into MoH rather than training from scratch."
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
