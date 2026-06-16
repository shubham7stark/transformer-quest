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
          "Here ", TQ.math("d_model = " + dModel), ", ", TQ.math("H = " + H), ", so ", TQ.math("dk = " + dk),
          " — four ", dk + "-dimensional attentions running side by side. This is almost exactly your CNN intuition: ",
          "a conv layer doesn't apply one filter, it applies a ", TQ.el("strong", { text: "bank" }),
          " of filters, each tuned to a different pattern (edges, texture, color blobs). A head is an attention ",
          "\"filter\"; the bank gives the layer multiple simultaneous ways to relate tokens."
        )
      ));

      /* ---------------------------------- PRIMARY interactive: heads as tabs */
      // Shared query selector drives BOTH the row highlight inside each head's
      // heatmap AND the Concat+Wo panel below. Clicking a heatmap row syncs it.
      var selQuery = 1; // default 'cat'
      var highlightSharp = false;

      // Forward decls so cross-updates work.
      var concatPanel = TQ.el("div", { class: "tq-panel tq-grow tq-mh-concat" });
      var querySel = TQ.segmented({
        options: tokens.map(function (tk, i) { return { label: tk, value: i }; }),
        value: selQuery,
        onChange: function (v) { selQuery = v; renderConcat(); refreshActiveHeatmap(); }
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
          "color:var(--ink-mute);font-family:var(--mono)}");
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
