/* ============================================================================
 * LEVEL 04 — Multi-Head Attention   (order 4)
 * ----------------------------------------------------------------------------
 * Follows the gold-standard pattern (see 01-tokens-embeddings.js):
 *   - one IIFE, ONE TQ.registerLevel call, NO other globals
 *   - render() = short explanation blocks + PRIMARY interactive visualization
 *   - every number genuinely computed via TQ math (no hardcoded attention)
 *   - all color via TQ.colorFor / TQ.heatmap / TQ.vectorView (one color language)
 *
 * Math shown, all real & deterministic:
 *   - the shared toy sentence -> TQ.toy.embeddings (6 x 16)
 *   - H = 4 heads, each with its OWN seeded Wq/Wk/Wv (base + head*10 + {1,2,3})
 *     via TQ.toyQKV(dk, ...). dk = dModel / H = 16 / 4 = 4.
 *   - per-head attention via TQ.attention -> .weights is the 6x6 heatmap shown
 *     in each tab; genuinely differs across heads because seeds differ.
 *   - Concat + Wo: per token concat the H length-4 outputs -> length 16, then
 *     ConcatAll (6 x 16) · Wo (16 x 16) -> Final (6 x 16). All via TQ.matmul.
 *   - "sharpest head" = argmin over heads of row entropy of att.weights[query].
 * ========================================================================== */

(function () {
  "use strict";

  TQ.registerLevel({
    id: "multi-head",
    order: 4,
    title: "Multi-Head Attention",
    icon: "🧠",
    tagline: "Run attention several times in parallel subspaces, then concatenate.",
    preread: "Illustrated Transformer",
    objectives: [
      "See that each head has its own learned Q/K/V projections (its own subspace)",
      "Observe different heads attend to different relations (different patterns on the same sentence)",
      "Follow concat-of-heads → output projection back to d_model"
    ],

    render: function (root) {
      var toy = TQ.toy;
      var tokens = toy.tokens;
      var X = toy.embeddings;       // 6 x 16
      var dModel = toy.dModel;      // 16
      var H = 4;                    // heads
      var dk = dModel / H;          // 4
      var n = tokens.length;        // 6
      var BASE = 400;               // seed base; per-head = BASE + h*10 + {1,2,3}

      /* ---------------------------------------------- build every head (real) */
      // heads[h] = { qkv, att } where att = {scores, scaled, weights, output, dk}
      var heads = [];
      for (var h = 0; h < H; h++) {
        var qkv = TQ.toyQKV(dk, BASE + h * 10 + 1, BASE + h * 10 + 2, BASE + h * 10 + 3);
        var att = TQ.attention(qkv.Q, qkv.K, qkv.V); // weights: 6x6, output: 6x4
        heads.push({ qkv: qkv, att: att });
      }

      // ConcatAll (6 x 16): horizontally stack the four 6x4 head outputs.
      var concatAll = [];
      for (var t = 0; t < n; t++) {
        var rowC = [];
        for (var hh = 0; hh < H; hh++) {
          var ov = heads[hh].att.output[t];
          for (var k = 0; k < ov.length; k++) rowC.push(ov[k]);
        }
        concatAll.push(rowC); // length H*dk = 16
      }
      // Wo: (H*dk) x dModel = 16 x 16, learned output projection (deterministic).
      var Wo = TQ.randMatrix(H * dk, dModel, 999);
      var Final = TQ.matmul(concatAll, Wo); // 6 x 16

      // Shannon entropy of a probability row (lower = more peaked / confident).
      function entropy(p) {
        var e = 0;
        for (var i = 0; i < p.length; i++) {
          var w = p[i];
          if (w > 1e-12) e += -w * Math.log(w);
        }
        return e;
      }
      // Sharpest head for a given query token index = argmin head-row entropy.
      function sharpestHead(q) {
        var best = 0, bestE = Infinity;
        for (var hi = 0; hi < H; hi++) {
          var e = entropy(heads[hi].att.weights[q]);
          if (e < bestE) { bestE = e; best = hi; }
        }
        return best;
      }
      // argmax index of an array (first max wins) — used for each head's peak key.
      function argmaxIdx(arr) {
        var bi = 0;
        for (var ai = 1; ai < arr.length; ai++) if (arr[ai] > arr[bi]) bi = ai;
        return bi;
      }

      /* ----------------------------------------------------- intro narrative */
      root.appendChild(TQ.block(
        TQ.h(2, "One attention head is a single point of view"),
        TQ.p(
          "You just built scaled dot-product attention: project tokens into ",
          TQ.math("Q, K, V"), ", score every query against every key, softmax, blend the values. ",
          "But one set of projection matrices (", TQ.math("Wq, Wk, Wv"),
          ") forces every relationship through ", TQ.el("strong", { text: "one" }),
          " learned lens. If that lens is tuned to track \"which noun does this verb act on,\" it has no ",
          "spare capacity to also track \"which article binds this noun\" or \"what's three tokens back.\" ",
          "A single head must compromise across all the relations in a sentence at once."
        )
      ));

      root.appendChild(TQ.block(
        TQ.h(2, "Run attention H times in parallel subspaces"),
        TQ.p(
          "Multi-head attention is the obvious fix: instead of one big ", TQ.math("d_model"),
          "-wide attention, run ", TQ.el("strong", { text: "H" }), " smaller ones in parallel, each with its ",
          TQ.el("strong", { text: "own" }), " ", TQ.math("Wq/Wk/Wv"), ". Each head projects the SAME input ",
          "embeddings into its own ", TQ.math("dk"), "-dimensional subspace (", TQ.math("dk = d_model / H"),
          ") and does the full scores → scale by ", TQ.math("1/√dk"),
          " → softmax → weighted-V pipeline independently."
        ),
        // on-screen arithmetic strip — make the shape bookkeeping explicit
        TQ.el("div", { class: "tq-mh-arith" },
          TQ.el("span", { class: "tq-mh-arith-item" }, TQ.math("dk = d_model / H = " + dModel + " / " + H + " = " + dk)),
          TQ.el("span", { class: "tq-mh-arith-item" }, TQ.math("concat width = H × dk = " + H + " × " + dk + " = " + (H * dk) + " = d_model"))
        ),
        TQ.p(
          "(\"Subspace\" just means a smaller slice of the representation space — each head looks at only ",
          TQ.math("dk"), " of the ", TQ.math("d_model"), " coordinates, reached through its own learned projection, ",
          "rather than the full width.) ",
          "Here ", TQ.math("d_model = " + dModel), ", ", TQ.math("H = " + H), ", so ", TQ.math("dk = " + dk),
          " — four ", dk + "-dimensional attentions running side by side. ",
          TQ.el("strong", { text: "If you've seen CNNs" }), " (optional analogy — skip if not): ",
          "this is almost exactly that intuition — a conv layer doesn't apply one filter, it applies a ",
          TQ.el("strong", { text: "bank" }),
          " of filters, each tuned to a different pattern (edges, texture, color blobs). A head is an attention ",
          "\"filter\"; the bank gives the layer multiple simultaneous ways to relate tokens."
        ),
        // Orienting diagram: the whole split → parallel heads → concat → Wo
        // pipeline at a glance, BEFORE the interactive parts dissect each stage.
        // Static SVG, theme colors via CSS vars / currentColor (no hardcoded hex).
        TQ.figure(
          '<svg viewBox="0 0 720 230" width="720" height="230" role="img" ' +
            'aria-label="Multi-head attention pipeline: split into heads, attend in parallel, concatenate, project with Wo" ' +
            'font-family="var(--mono)" font-size="12">' +
            '<defs><marker id="tq-l4-arrow" viewBox="0 0 10 10" refX="9" refY="5" ' +
              'markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
              '<path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>' +
            '<g stroke="currentColor" stroke-width="1.4" fill="none" color="var(--ink-faint)">' +
              // X -> split fan-out
              '<line x1="92" y1="115" x2="150" y2="115" marker-end="url(#tq-l4-arrow)"/>' +
              '<line x1="156" y1="115" x2="300" y2="35"  marker-end="url(#tq-l4-arrow)"/>' +
              '<line x1="156" y1="115" x2="300" y2="88"  marker-end="url(#tq-l4-arrow)"/>' +
              '<line x1="156" y1="115" x2="300" y2="141" marker-end="url(#tq-l4-arrow)"/>' +
              '<line x1="156" y1="115" x2="300" y2="194" marker-end="url(#tq-l4-arrow)"/>' +
              // heads -> concat converge
              '<line x1="430" y1="35"  x2="540" y2="108" marker-end="url(#tq-l4-arrow)"/>' +
              '<line x1="430" y1="88"  x2="540" y2="112" marker-end="url(#tq-l4-arrow)"/>' +
              '<line x1="430" y1="141" x2="540" y2="118" marker-end="url(#tq-l4-arrow)"/>' +
              '<line x1="430" y1="194" x2="540" y2="122" marker-end="url(#tq-l4-arrow)"/>' +
              // concat -> Wo -> out
              '<line x1="600" y1="115" x2="630" y2="115" marker-end="url(#tq-l4-arrow)"/>' +
            '</g>' +
            // X input
            '<text x="48" y="92" text-anchor="middle" fill="var(--ink-mute)" font-size="11">input</text>' +
            '<rect x="14" y="100" width="76" height="32" rx="7" fill="var(--panel-hi)" stroke="var(--line)"/>' +
            '<text x="52" y="120" text-anchor="middle" fill="var(--ink)">X (6×16)</text>' +
            // split node label
            '<text x="150" y="150" text-anchor="middle" fill="var(--ink-faint)" font-size="10">split H=4</text>' +
            // four head boxes
            '<g>' +
              '<rect x="300" y="20"  width="130" height="30" rx="6" fill="var(--panel-hi)" stroke="var(--accent)"/>' +
              '<text x="365" y="39"  text-anchor="middle" fill="var(--ink)">head 1 · dk=4</text>' +
              '<rect x="300" y="73"  width="130" height="30" rx="6" fill="var(--panel-hi)" stroke="var(--accent)"/>' +
              '<text x="365" y="92"  text-anchor="middle" fill="var(--ink)">head 2 · dk=4</text>' +
              '<rect x="300" y="126" width="130" height="30" rx="6" fill="var(--panel-hi)" stroke="var(--accent)"/>' +
              '<text x="365" y="145" text-anchor="middle" fill="var(--ink)">head 3 · dk=4</text>' +
              '<rect x="300" y="179" width="130" height="30" rx="6" fill="var(--panel-hi)" stroke="var(--accent)"/>' +
              '<text x="365" y="198" text-anchor="middle" fill="var(--ink)">head 4 · dk=4</text>' +
            '</g>' +
            '<text x="365" y="13" text-anchor="middle" fill="var(--ink-mute)" font-size="10">attend in parallel → each 6×4</text>' +
            // concat node
            '<rect x="540" y="98" width="60" height="34" rx="7" fill="var(--panel-hi)" stroke="var(--accent-2)"/>' +
            '<text x="570" y="119" text-anchor="middle" fill="var(--ink)">concat</text>' +
            '<text x="570" y="150" text-anchor="middle" fill="var(--ink-faint)" font-size="10">6×16</text>' +
            // Wo box
            '<rect x="630" y="99" width="74" height="32" rx="7" fill="var(--panel-hi)" stroke="var(--line)"/>' +
            '<text x="667" y="119" text-anchor="middle" fill="var(--ink)">Wo (16×16)</text>' +
            '<text x="667" y="150" text-anchor="middle" fill="var(--ink-faint)" font-size="10">out 6×16</text>' +
            // arrow shape labels
            '<text x="120" y="108" text-anchor="middle" fill="var(--ink-faint)" font-size="10">6×16</text>' +
            '<text x="250" y="78"  text-anchor="middle" fill="var(--ink-faint)" font-size="10">6×4 ×4</text>' +
          '</svg>',
          "The whole pipeline at a glance: split d_model into H heads, attend in parallel subspaces, " +
          "concatenate back to d_model, then mix with Wo. The tabs and tracer below let you play with each stage."
        )
      ));

      /* ---------------------------------- PRIMARY interactive: heads as tabs */
      // Shared query selector drives BOTH the row highlight inside each head's
      // heatmap AND the Concat+Wo panel below. Clicking a heatmap row syncs it.
      var selQuery = 1; // default 'cat'
      var highlightSharp = false;

      // Per-head enable flags for the side-by-side comparison block (all ON).
      var enabledHeads = [];
      for (var eh = 0; eh < H; eh++) enabledHeads.push(true);

      // Forward decls so cross-updates work.
      var concatPanel = TQ.el("div", { class: "tq-panel tq-grow tq-mh-concat" });
      var querySel = TQ.segmented({
        options: tokens.map(function (tk, i) { return { label: tk, value: i }; }),
        value: selQuery,
        onChange: function (v) { selQuery = v; renderConcat(); refreshActiveHeatmap(); renderCompare(); }
      });
      var sharpToggle = TQ.toggle({
        label: "Highlight sharpest head",
        value: false,
        onChange: function (v) { highlightSharp = v; renderConcat(); }
      });

      // Track the currently-mounted heatmap so a query change can re-highlight
      // its selected row without flipping tabs.
      var activeHeadIdx = 0;
      var activeHeatHolder = null;

      function buildHeadHeatmap(headIdx) {
        return TQ.heatmap(heads[headIdx].att.weights, {
          rowLabels: tokens, colLabels: tokens, cellSize: 44,
          min: 0, max: 1, // attention weights are probabilities in [0,1]
          selectedRow: selQuery,
          format: function (v) { return TQ.fmt(v, 2); }
        });
      }

      function wireHeatRows(node) {
        // Make whole rows clickable to set the query (clicking any cell in a
        // row selects that query). Kernel already makes cells keyboard-able.
        var cells = node.querySelectorAll(".tq-hm-cell");
        var idx = 0;
        for (var r = 0; r < n; r++) {
          for (var c = 0; c < n; c++) {
            (function (rr, cell) {
              cell.classList.add("is-clickable");
              cell.setAttribute("role", "button");
              cell.setAttribute("tabindex", "0");
              cell.addEventListener("click", function () { setQuery(rr); });
              cell.addEventListener("keydown", function (e) {
                if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setQuery(rr); }
              });
            })(r, cells[idx]);
            idx++;
          }
        }
      }

      function setQuery(q) {
        selQuery = q;
        querySel.set(q);
        refreshActiveHeatmap();
        renderConcat();
        renderCompare();
      }

      function refreshActiveHeatmap() {
        if (!activeHeatHolder) return;
        var fresh = buildHeadHeatmap(activeHeadIdx);
        wireHeatRows(fresh);
        activeHeatHolder.replaceChild(fresh, activeHeatHolder.firstChild);
      }

      var tabItems = [];
      for (var ht = 0; ht < H; ht++) {
        (function (headIdx) {
          tabItems.push({
            label: "Head " + (headIdx + 1),
            render: function (container) {
              activeHeadIdx = headIdx;
              var heat = buildHeadHeatmap(headIdx);
              wireHeatRows(heat);
              activeHeatHolder = TQ.el("div", { class: "tq-grow" }, heat);

              // Per-head quick stats so the "different lens" point is concrete.
              var rowEntropies = [];
              for (var q = 0; q < n; q++) rowEntropies.push(entropy(heads[headIdx].att.weights[q]));
              var meanE = TQ.mean(rowEntropies);
              var maxPossible = Math.log(n); // uniform-row entropy ceiling

              var info = TQ.el("div", { class: "tq-panel tq-mh-headinfo" },
                TQ.el("div", { class: "tq-mh-headinfo-h" },
                  TQ.el("strong", { text: "Head " + (headIdx + 1) }),
                  TQ.badge("own seeds " + (BASE + headIdx * 10 + 1) + "/" +
                    (BASE + headIdx * 10 + 2) + "/" + (BASE + headIdx * 10 + 3), "info")
                ),
                TQ.el("div", { class: "tq-mh-stats" },
                  TQ.kv("dk", String(dk)),
                  TQ.kv("avg row entropy", TQ.fmt(meanE, 2)),
                  TQ.kv("uniform max", TQ.fmt(maxPossible, 2))
                ),
                TQ.note("Each cell = how much the query (row) reads from the key (col). " +
                  "Every row is a softmax, so it sums to ≈1.00 and lives in [0,1]. " +
                  "(\"Row entropy\" above = how spread-out vs. peaked a row is: low = the head " +
                  "focuses hard on one key, high = it spreads attention evenly across keys.) " +
                  "Click any row to trace that query through Concat + Wo below.")
              );

              container.appendChild(TQ.el("div", { class: "tq-flexrow tq-mh-tabrow" },
                activeHeatHolder, info));
            }
          });
        })(ht);
      }

      var headsBlock = TQ.block(
        TQ.h(2, "Four heads, same sentence — genuinely different patterns"),
        TQ.p(
          "Each tab below is a real attention-weights heatmap (rows = query token, cols = key token) ",
          "for one head, computed from that head's ", TQ.el("strong", { text: "own seeded" }), " ",
          TQ.math("Wq/Wk/Wv"), ". The heads differ because each gets a different random seed for its ",
          "projections (", TQ.math("base + head·10"), "), so each carves out a genuinely different subspace ",
          "— same input, four different lenses. Nothing is faked; flip between tabs and watch the pattern change."
        )
      );
      var legend = TQ.el("div", { class: "tq-legend" },
        TQ.el("span", { text: "0 (ignore)" }),
        TQ.el("div", { class: "tq-legend-scale" }),
        TQ.el("span", { text: "1 (focus)" })
      );
      headsBlock.appendChild(legend);
      headsBlock.appendChild(TQ.tabs(tabItems));
      headsBlock.appendChild(TQ.callout(
        "Don't over-romanticize it: in a trained model heads CAN specialize (one ends up syntax-ish, " +
        "another positional, etc.), but they're not cleanly labeled modules, and pruning studies show many " +
        "heads are redundant. Say \"they can specialize,\" not \"head 7 IS the syntax head.\""
      ));
      root.appendChild(headsBlock);

      /* ============================================================= *
       *  PRIMARY interactive — SAME query, four lenses, + union coverage.
       *  The comparison the tabs make impossible: all four heads' attention
       *  ROWS for ONE query, stacked on the same six key columns, plus a
       *  computed UNION row (elementwise max across the ENABLED heads).
       *  Drives off the existing module-scope selQuery (no second selector).
       * ============================================================= */
      var comparePanel = TQ.el("div", { class: "tq-panel tq-grow tq-mh-cmp" });

      // one head-row strip: heads[h].att.weights[q] as a 1×6 heatmap, [0,1] scale
      function headRowStrip(h, q, dim) {
        var strip = TQ.heatmap([heads[h].att.weights[q]], {
          colLabels: tokens,
          rowLabels: ["head " + (h + 1)],
          cellSize: 44, min: 0, max: 1,
          selectedCol: argmaxIdx(heads[h].att.weights[q]),
          format: function (v) { return TQ.fmt(v, 2); }
        });
        return TQ.el("div", { class: "tq-mh-cmp-row" + (dim ? " is-off" : "") }, strip);
      }

      function renderCompare() {
        comparePanel.innerHTML = "";
        var q = selQuery;

        // header — which query we're comparing across heads
        comparePanel.appendChild(TQ.el("div", { class: "tq-mh-concat-head" },
          TQ.el("span", { class: "tq-stat-cap", text: "same query, four lenses" }),
          TQ.el("span", { class: "tq-mh-qtoken", text: "\"" + tokens[q] + "\" (#" + q + ")" })
        ));

        // shared column-header alignment is handled by each heatmap's colLabels.
        var grid = TQ.el("div", { class: "tq-mh-cmp-grid" });

        // four head rows (dim + excluded from union when toggled off)
        for (var hi = 0; hi < H; hi++) {
          grid.appendChild(headRowStrip(hi, q, !enabledHeads[hi]));
        }

        // UNION row = elementwise max across the ENABLED heads (genuinely live).
        var enabledIdx = [];
        for (var ei = 0; ei < H; ei++) if (enabledHeads[ei]) enabledIdx.push(ei);
        var union = [];
        for (var k = 0; k < n; k++) {
          if (enabledIdx.length === 0) { union.push(0); continue; }
          union.push(TQ.maxOf(enabledIdx.map(function (h) { return heads[h].att.weights[q][k]; })));
        }
        var unionStrip = TQ.heatmap([union], {
          colLabels: tokens,
          rowLabels: ["union (max)"],
          cellSize: 44, min: 0, max: 1,
          format: function (v) { return TQ.fmt(v, 2); }
        });
        grid.appendChild(TQ.el("div", { class: "tq-mh-cmp-row tq-mh-cmp-union" }, unionStrip));
        comparePanel.appendChild(grid);

        // readout: per-head argmax chips (enabled only) + distinct-keys-covered badge
        var distinct = {};
        var readout = TQ.el("div", { class: "tq-mh-cmp-readout" });
        readout.appendChild(TQ.el("span", { class: "tq-stat-cap", text: "peaks" }));
        for (var ci = 0; ci < H; ci++) {
          var am = argmaxIdx(heads[ci].att.weights[q]);
          var on = enabledHeads[ci];
          if (on) distinct[am] = true;
          readout.appendChild(TQ.badge(
            "head" + (ci + 1) + "→" + tokens[am] + (on ? "" : " (off)"),
            on ? "info" : "default"
          ));
        }
        var nDistinct = Object.keys(distinct).length;
        readout.appendChild(TQ.badge(
          nDistinct + " distinct key" + (nDistinct === 1 ? "" : "s") + " covered",
          nDistinct >= 3 ? "good" : "warn"
        ));
        comparePanel.appendChild(readout);

        // honest "they differ in WHERE, not in sharpness" annotation via entropy
        var ents = [];
        for (var pe = 0; pe < H; pe++) ents.push(entropy(heads[pe].att.weights[q]));
        comparePanel.appendChild(TQ.note(
          "Each strip is one head's softmax row for \"" + tokens[q] + "\" over the six keys (same [0,1] color " +
          "scale as the tabs). Each head's argmax key (its column label) is highlighted. The heads differ mostly " +
          "in WHERE they peak, not how " +
          "sharp they are — row entropies here are " +
          ents.map(function (e) { return TQ.fmt(e, 2); }).join(" / ") +
          " (uniform ceiling ln 6 = " + TQ.fmt(Math.log(n), 2) + " — the entropy a row would have if attention " +
          "were spread perfectly evenly across all six keys). The bottom UNION row is the elementwise " +
          "max across the ENABLED heads: drop to one head and it collapses to that head's narrow row; enable all " +
          "four and it lights up across keys no single head reached."
        ));
      }

      // four head on/off toggles (all default ON) — recompute union live.
      var cmpToggles = TQ.el("div", { class: "tq-mh-cmp-toggles" });
      for (var tg = 0; tg < H; tg++) {
        (function (hIdx) {
          var tgl = TQ.toggle({
            label: "Head " + (hIdx + 1),
            value: true,
            onChange: function (on) { enabledHeads[hIdx] = on; renderCompare(); }
          });
          cmpToggles.appendChild(tgl.el);
        })(tg);
      }

      var compareBlock = TQ.block(
        TQ.h(2, "Same query, four lenses — and what they cover together"),
        TQ.p(
          "The tabs above show one head at a time, so you have to hold one 6×6 grid in your head while you flip ",
          "to the next. Here is the comparison the tabs can't make: pick a query (the ",
          TQ.el("strong", { text: "same selector that drives Concat below" }),
          ") and see all four heads' attention ", TQ.el("strong", { text: "rows" }),
          " for it, stacked on the same six key columns."
        ),
        TQ.p(
          "The bottom ", TQ.el("strong", { text: "union" }), " row is computed live as the elementwise ",
          TQ.math("max"), " across the enabled heads. Toggle heads off and watch coverage shrink to a single ",
          "narrow row; turn them back on and the union blooms across keys no single head reaches alone — the ",
          "concrete meaning of \"multiple simultaneous ways to relate tokens.\""
        ),
        TQ.el("div", { class: "tq-controls-row" },
          TQ.el("div", {},
            TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "Heads in the union"),
            cmpToggles)
        ),
        comparePanel,
        TQ.callout(
          "No single head covers cat→{The, mat, on} at once — the bank does. That breadth is exactly what " +
          "Concat + Wo merges into one vector next."
        )
      );
      renderCompare();
      root.appendChild(compareBlock);

      /* --------------------------- Concat + Wo panel (depends on selQuery) */
      function renderConcat() {
        concatPanel.innerHTML = "";
        var q = selQuery;
        var sharp = sharpestHead(q);

        concatPanel.appendChild(TQ.el("div", { class: "tq-mh-concat-head" },
          TQ.el("span", { class: "tq-stat-cap", text: "tracing query token" }),
          TQ.el("span", { class: "tq-mh-qtoken", text: "\"" + tokens[q] + "\" (#" + q + ")" })
        ));

        // 1) each head's dk-vector output for this token
        var headRows = TQ.el("div", { class: "tq-mh-headvecs" });
        for (var hi = 0; hi < H; hi++) {
          var isSharp = (hi === sharp);
          var dim = highlightSharp && !isSharp;
          var wrap = TQ.el("div", { class: "tq-mh-headvec" + (dim ? " is-dim" : "") },
            TQ.el("div", { class: "tq-mh-headvec-label" },
              TQ.el("span", { text: "head " + (hi + 1) + " · " + dk + "d" }),
              (highlightSharp && isSharp)
                ? TQ.badge("sharpest ✦", "good")
                : null
            ),
            TQ.vectorView(heads[hi].att.output[q], { cellSize: 26, showValues: true })
          );
          headRows.appendChild(wrap);
        }
        concatPanel.appendChild(TQ.el("div", { class: "tq-mh-step" },
          TQ.el("div", { class: "tq-mh-step-label", text: "1 · each head outputs a " + dk + "-dim vector" }),
          headRows
        ));

        // 2) concatenate -> length 16 = d_model
        concatPanel.appendChild(TQ.el("div", { class: "tq-mh-step" },
          TQ.el("div", { class: "tq-mh-step-label", text: "2 · concatenate → " + (H * dk) + "-dim (" + H + " × " + dk + " = " + (H * dk) + ")" }),
          TQ.vectorView(concatAll[q], { cellSize: 22, showValues: true })
        ));

        // 3) project by Wo -> final d_model vector
        concatPanel.appendChild(TQ.el("div", { class: "tq-mh-step" },
          TQ.el("div", { class: "tq-mh-step-label", text: "3 · × Wo (" + (H * dk) + "×" + dModel + ") → final " + dModel + "-dim vector" }),
          TQ.vectorView(Final[q], { cellSize: 22, showValues: true })
        ));

        concatPanel.appendChild(TQ.note(
          "Concat returns you to model width; " +
          "Wo is the learned mixer that lets the heads talk — without it you'd just have four " +
          "disconnected subspace results. " + (highlightSharp
            ? ("Sharpest head for \"" + tokens[q] + "\" is head " + (sharp + 1) +
               " (lowest row entropy = most confident).")
            : "Toggle \"sharpest head\" to spotlight the most confident head for this query.")
        ));
      }

      var concatBlock = TQ.block(
        TQ.h(2, "Concat, then project back: the part everyone skips"),
        TQ.p(
          "Each head outputs an ", TQ.math("n × dk"), " matrix (one ", TQ.math("dk"),
          "-vector per token). You can't just hand four separate ", TQ.math("dk"),
          "-vectors to the next layer — it expects width ", TQ.math("d_model"),
          ". So per token you ", TQ.el("strong", { text: "concatenate" }), " the H head-outputs back to length ",
          TQ.math("H·dk = " + (H * dk)), ", then multiply by ONE more learned matrix ",
          TQ.math("Wo"), " of shape ", TQ.math((H * dk) + "×" + dModel), ". ",
          TQ.math("MultiHead(X) = Concat(head_1..head_H)·Wo"), "."
        ),
        TQ.p(
          "Pick which token to trace; watch its four short ", TQ.math("dk=" + dk),
          " vectors snap together into one ", TQ.math((H * dk) + "-cell"),
          " vector, which then becomes a DIFFERENT ", TQ.math(dModel + "-cell"),
          " vector after ", TQ.math("Wo"), "."
        ),
        TQ.el("div", { class: "tq-controls-row" },
          TQ.el("div", {},
            TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "Query token"),
            querySel.el),
          TQ.el("div", {},
            TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "Options"),
            sharpToggle.el)
        ),
        concatPanel
      );
      renderConcat();
      // Minimal, correct PyTorch forward pass that maps 1:1 onto this toy:
      // d_model=16, H=4, dk=4. The (B, n, H, dk) reshape is the batched view
      // that runs all heads in parallel — exactly what the tracer above showed.
      concatBlock.appendChild(TQ.code(
        "import torch\n" +
        "import torch.nn as nn\n" +
        "import torch.nn.functional as F\n" +
        "\n" +
        "class MultiHeadAttention(nn.Module):\n" +
        "    def __init__(self, d_model=16, n_heads=4):\n" +
        "        super().__init__()\n" +
        "        self.n_heads = n_heads\n" +
        "        self.d_k = d_model // n_heads        # 16 // 4 = 4 (this level's dk)\n" +
        "        self.w_q = nn.Linear(d_model, d_model)   # one (16->16) projection each\n" +
        "        self.w_k = nn.Linear(d_model, d_model)\n" +
        "        self.w_v = nn.Linear(d_model, d_model)\n" +
        "        self.w_o = nn.Linear(d_model, d_model)   # the learned output mixer\n" +
        "\n" +
        "    def forward(self, x):                    # x: (B, n, d_model)\n" +
        "        B, n, _ = x.shape\n" +
        "        # project, then split d_model into (n_heads, d_k) and move heads up front\n" +
        "        # (B, n, d_model) -> (B, n, H, d_k) -> (B, H, n, d_k): the batched view\n" +
        "        q = self.w_q(x).view(B, n, self.n_heads, self.d_k).transpose(1, 2)\n" +
        "        k = self.w_k(x).view(B, n, self.n_heads, self.d_k).transpose(1, 2)\n" +
        "        v = self.w_v(x).view(B, n, self.n_heads, self.d_k).transpose(1, 2)\n" +
        "        # scaled dot-product attention, applied across ALL heads at once\n" +
        "        scores = q @ k.transpose(-2, -1) / (self.d_k ** 0.5)   # (B, H, n, n)\n" +
        "        attn = F.softmax(scores, dim=-1)     # one softmax runs all H heads\n" +
        "        out = attn @ v                       # (B, H, n, d_k)\n" +
        "        # transpose + reshape back: this IS the concat, width H*d_k = 16 = d_model\n" +
        "        out = out.transpose(1, 2).reshape(B, n, self.n_heads * self.d_k)\n" +
        "        return self.w_o(out)                 # (B, n, d_model)\n",
        { lang: "python", label: "multi-head attention",
          caption: "The (B, n, H, dk) reshape is the whole trick: one batched softmax runs all H heads in " +
                   "parallel; transpose+reshape is the concat; w_o is the mixer." }
      ));
      root.appendChild(concatBlock);

      /* --------------------------------------------------- wrap-up takeaway */
      root.appendChild(TQ.block(
        TQ.h(2, "Why subspaces — and where this goes next"),
        TQ.p(
          "Splitting ", TQ.math("d_model"), " into H slices of size ", TQ.math("dk"),
          " isn't free capacity — total parameters are roughly the same as one full-width head. The win is ",
          TQ.el("strong", { text: "structural" }), ": H independent softmaxes can each commit fully to one ",
          "relation instead of one softmax averaging them all."
        ),
        TQ.callout(
          "Carry forward: MultiHead(X) = Concat(head_1..head_H)·Wo. The final " + dModel +
          "-dim vector you just built is exactly what flows into the residual + LayerNorm + FFN of a " +
          "transformer block — the next level."
        )
      ));

      /* ----------------------------------------------------- go deeper links */
      root.appendChild(TQ.resources("Go deeper", [
        { label: "The Illustrated Transformer", kind: "blog",
          url: "https://jalammar.github.io/illustrated-transformer/",
          note: "the visual walkthrough this level's preread points at — multi-head section" },
        { label: "3Blue1Brown — Attention, visually explained", kind: "video",
          url: "https://www.youtube.com/watch?v=eMlx5fFNoYc",
          note: "geometric intuition for what each head's Q/K/V does" },
        { label: "Andrej Karpathy — Let's build GPT from scratch", kind: "video",
          url: "https://www.youtube.com/watch?v=kCc8FmEb1nY",
          note: "codes the exact (B, n, H, dk) reshape above, from scratch" },
        { label: "The Annotated Transformer", kind: "code",
          url: "https://nlp.seas.harvard.edu/annotated-transformer/",
          note: "line-by-line PyTorch alongside the original paper" },
        { label: "Attention Is All You Need (§3.2.2, Multi-Head Attention)", kind: "paper",
          url: "https://arxiv.org/abs/1706.03762",
          note: "the source — the Concat·Wo formula in its original form" }
      ]));

      /* small scoped styles (colors via CSS vars only — no hardcoded hex) */
      injectOnce("tq-lvl04-css",
        ".tq-mh-arith{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}" +
        ".tq-mh-arith-item{display:inline-flex}" +
        ".tq-mh-tabrow{align-items:flex-start;gap:20px}" +
        ".tq-mh-headinfo{min-width:240px;display:flex;flex-direction:column;gap:10px}" +
        ".tq-mh-headinfo-h{display:flex;align-items:center;justify-content:space-between;gap:8px}" +
        ".tq-mh-stats{display:flex;gap:8px;flex-wrap:wrap}" +
        ".tq-mh-concat{display:flex;flex-direction:column;gap:14px}" +
        ".tq-mh-concat-head{display:flex;align-items:center;gap:10px}" +
        ".tq-mh-qtoken{font-size:17px;font-weight:700}" +
        ".tq-mh-step{display:flex;flex-direction:column;gap:6px}" +
        ".tq-mh-step-label{font-size:12px;color:var(--ink-mute);font-family:var(--mono);letter-spacing:.02em}" +
        ".tq-mh-headvecs{display:flex;gap:16px;flex-wrap:wrap}" +
        ".tq-mh-headvec{display:flex;flex-direction:column;gap:4px;transition:opacity .2s}" +
        ".tq-mh-headvec.is-dim{opacity:.32}" +
        ".tq-mh-headvec-label{display:flex;align-items:center;gap:8px;font-size:12px;" +
          "color:var(--ink-mute);font-family:var(--mono)}" +
        ".tq-mh-cmp{display:flex;flex-direction:column;gap:14px}" +
        ".tq-mh-cmp-grid{display:flex;flex-direction:column;gap:6px}" +
        ".tq-mh-cmp-row{transition:opacity .2s}" +
        ".tq-mh-cmp-row.is-off{opacity:.3}" +
        ".tq-mh-cmp-union{margin-top:4px}" +
        ".tq-mh-cmp-union .tq-heatmap{outline:1px solid var(--accent);outline-offset:3px;border-radius:6px}" +
        ".tq-mh-cmp-toggles{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0}" +
        ".tq-mh-cmp-readout{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:4px 0}");
    },

    quiz: [
      {
        q: "In multi-head attention with d_model = 16 and H = 4 heads, what is the dimension dk that each head works in, and why that value?",
        choices: [
          "dk = 16, because every head sees the full model width",
          "dk = 64, because dk is always 64 in transformers",
          "dk = 4, because dk = d_model / H so the H head outputs concatenate back to d_model",
          "dk = 8, because heads always use half of d_model"
        ],
        answer: 2,
        explain: "dk = d_model / H = 16/4 = 4. Choosing dk this way means concatenating the H head outputs " +
                 "(H·dk = 4·4 = 16) lands exactly back at d_model, ready for the Wo projection."
      },
      {
        q: "In this level the four heads produce visibly different attention heatmaps on the SAME sentence. What actually causes the difference?",
        choices: [
          "Each head is given a different random seed for its Wq/Wk/Wv projections, so each operates in a different subspace",
          "The sentence is secretly changed for each head",
          "Later heads see the softmax output of earlier heads",
          "Only the colors differ; the underlying weights are identical"
        ],
        answer: 0,
        explain: "Each head has its OWN learned (here, seeded via base + head·10) Q/K/V projections, carving " +
                 "out a different subspace. Same input, different lens, genuinely different real attention " +
                 "weights — nothing faked."
      },
      {
        q: "After the H heads each produce their dk-dimensional outputs, how does multi-head attention return to a single d_model vector per token?",
        choices: [
          "It averages the H head outputs elementwise",
          "It sums the H attention-weight matrices, then applies V",
          "It concatenates the H head outputs (width H·dk = d_model) and multiplies by a learned output matrix Wo",
          "It takes only the output of the sharpest head and discards the rest"
        ],
        answer: 2,
        explain: "MultiHead(X) = Concat(head_1..head_H)·Wo. Concatenation restores width H·dk = d_model, and " +
                 "the learned Wo (here 16×16) mixes the per-subspace results into one vector — it's how the " +
                 "heads combine."
      },
      {
        q: "A colleague claims 'each attention head in a trained LLM is a clean, individually interpretable module — head 7 IS the syntax head.' What's the accurate refinement?",
        choices: [
          "Correct — every head maps to exactly one human-named linguistic function",
          "Heads CAN specialize and some show interpretable patterns, but they're not cleanly labeled modules and many are redundant",
          "Heads are identical to each other after training",
          "Interpretability is impossible because attention weights aren't real numbers"
        ],
        answer: 1,
        explain: "Heads can and do specialize (positional, syntactic-ish, etc.) and some are strikingly " +
                 "interpretable, but they aren't guaranteed clean single-function modules — pruning work shows " +
                 "substantial redundancy. Say 'they can specialize,' not 'each head = one function.'"
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
