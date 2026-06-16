/* ============================================================================
 * LEVEL 01 — Tokens & Embeddings   (order 1)  [REFERENCE / GOLD-STANDARD]
 * ----------------------------------------------------------------------------
 * The pattern every other level copies:
 *   - one IIFE, ONE TQ.registerLevel call, NO other globals
 *   - render() = short explanation blocks + at least one PRIMARY interactive
 *     visualization, every number genuinely computed via TQ math
 *   - all color via TQ.colorFor / TQ.heatmap / TQ.vectorView (one color language)
 *   - 2-4 quiz questions
 *
 * Math shown here, all real:
 *   - the shared toy sentence -> tokens -> TQ.toy.embeddings (16-dim vectors)
 *   - each token rendered as a TQ.vectorView (colored cells = vector components)
 *   - a full cosine-similarity matrix (TQ.cosine) shown as a clickable heatmap
 *   - a "compare two tokens" panel where the user picks A & B and watches the
 *     real dot product / norms / cosine update, building the intuition that
 *     "similar words = closer vectors".
 * ========================================================================== */

(function () {
  "use strict";

  TQ.registerLevel({
    id: "tokens-embeddings",
    order: 1,
    title: "Tokens & Embeddings",
    icon: "🔠",
    tagline: "Text becomes numbers a network can actually do math on.",
    preread: "Illustrated Transformer (input side)",
    objectives: [
      "See how a sentence is split into tokens and each token becomes a learned vector",
      "Read an embedding as a row of numbers (here, 16 of them) — like a CNN feature vector, but for a word",
      "Use cosine similarity to feel that related words land in nearby directions"
    ],

    render: function (root) {
      var toy = TQ.toy;
      var tokens = toy.tokens;
      var emb = toy.embeddings;
      var d = toy.dModel;
      var n = tokens.length;

      /* -------------------------------------------------- intro narrative */
      root.appendChild(TQ.block(
        TQ.h(2, "From characters to vectors"),
        TQ.p(
          "A transformer never sees letters. The first thing it does is chop text into ",
          TQ.el("strong", { text: "tokens" }),
          " (roughly: words or word-pieces) and hand each token its own vector — an ",
          TQ.el("strong", { text: "embedding" }),
          ". You already know this shape of object: it's a learned feature vector, exactly like the ",
          "512-d descriptor popping out of a CNN before the classifier. The twist is that here the vector ",
          "stands for a piece of language, and it's looked up from a big trainable table — one row per vocabulary entry."
        ),
        TQ.p(
          "Our toy sentence for the whole game is fixed so you can build intuition without moving targets:"
        ),
        TQ.el("div", { class: "tq-token-strip" },
          tokens.map(function (tk, i) {
            return TQ.el("span", { class: "tq-token-chip" },
              TQ.el("span", { class: "tq-token-idx", text: "#" + i }),
              TQ.el("span", { class: "tq-token-word", text: tk })
            );
          })
        ),
        TQ.note("Tokenization here is one-token-per-word for clarity. Real tokenizers (BPE) " +
                "would split rarer words into sub-pieces — e.g. \"tokenization\" → \"token\" + \"ization\" — " +
                "but the idea is identical: every piece gets a vector.")
      ));

      /* -------------------------------------- each token as a colored vector */
      var vectorBlock = TQ.block(
        TQ.h(2, "An embedding is just a vector of numbers"),
        TQ.p(
          "Below, each token's ", TQ.math("d_model = " + d),
          "-dimensional embedding is drawn as a row of cells. Each cell is one component; its color is the ",
          TQ.el("strong", { text: "value" }),
          " on our shared scale (cool/dark = low, bright/warm = high). Same color language is used everywhere in this game."
        )
      );
      var legend = TQ.el("div", { class: "tq-legend" },
        TQ.el("span", { text: "low" }),
        TQ.el("div", { class: "tq-legend-scale" }),
        TQ.el("span", { text: "high" })
      );
      vectorBlock.appendChild(legend);

      var vecList = TQ.el("div", { class: "tq-vec-list" });
      for (var vi = 0; vi < n; vi++) {
        vecList.appendChild(TQ.vectorView(emb[vi], { label: tokens[vi], cellSize: 24 }));
      }
      vectorBlock.appendChild(vecList);
      vectorBlock.appendChild(TQ.callout(
        "Notice \"The\" (#0) and \"the\" (#4) look almost identical — same word, near-identical vectors. " +
        "\"cat\" and \"mat\" share a family resemblance too. That visual similarity is the whole point: " +
        "meaning lives in the geometry."
      ));
      root.appendChild(vectorBlock);

      /* ------------------------------------- PRIMARY interactive: similarity */
      // Precompute the full cosine-similarity matrix (real TQ.cosine).
      var simMatrix = [];
      for (var r = 0; r < n; r++) {
        var rowSim = [];
        for (var c = 0; c < n; c++) rowSim.push(TQ.cosine(emb[r], emb[c]));
        simMatrix.push(rowSim);
      }

      var simBlock = TQ.block(
        TQ.h(2, "Do these vectors actually mean anything?"),
        TQ.p(
          "If embeddings are meaningful, related tokens should point in similar directions. We measure ",
          "direction-agreement with ", TQ.el("strong", { text: "cosine similarity" }), ": ",
          TQ.math("cos(a,b) = (a·b) / (‖a‖ ‖b‖)"),
          ". It ignores length and asks only \"how aligned?\" — ", TQ.el("strong", { text: "+1" }),
          " = same direction, ", TQ.el("strong", { text: "0" }), " = unrelated/orthogonal, ",
          TQ.el("strong", { text: "−1" }), " = opposite."
        ),
        TQ.p(
          TQ.el("strong", { text: "Click any cell" }),
          " in the matrix to break that pair down on the right. The diagonal is each token with itself, so it's always 1.00."
        )
      );

      // layout: heatmap (left) + breakdown panel (right)
      var detail = TQ.el("div", { class: "tq-panel tq-grow tq-sim-detail" });

      function renderDetail(i, j) {
        detail.innerHTML = "";
        var a = emb[i], b = emb[j];
        var dotv = TQ.dot(a, b);
        var na = TQ.norm(a), nb = TQ.norm(b);
        var cos = TQ.cosine(a, b);
        var verdict = cos > 0.85 ? ["Twins", "good"]
                    : cos > 0.5 ? ["Related", "info"]
                    : cos > 0.15 ? ["Loosely related", "default"]
                    : cos > -0.15 ? ["Basically unrelated", "default"]
                    : ["Opposed", "warn"];

        detail.appendChild(TQ.el("div", { class: "tq-sim-head" },
          TQ.el("span", { class: "tq-sim-pair" },
            TQ.el("strong", { text: tokens[i] }),
            TQ.el("span", { class: "tq-sim-vs", text: "·" }),
            TQ.el("strong", { text: tokens[j] })
          ),
          TQ.badge(verdict[0], verdict[1])
        ));

        detail.appendChild(TQ.vectorView(a, { label: tokens[i], cellSize: 18 }));
        detail.appendChild(TQ.vectorView(b, { label: tokens[j], cellSize: 18 }));

        detail.appendChild(TQ.el("div", { class: "tq-sim-stats" },
          TQ.kv("a · b", TQ.fmt(dotv, 2)),
          TQ.kv("‖a‖", TQ.fmt(na, 2)),
          TQ.kv("‖b‖", TQ.fmt(nb, 2))
        ));

        detail.appendChild(TQ.el("div", { class: "tq-sim-cos" },
          TQ.el("span", { class: "tq-stat-cap", text: "cosine similarity" }),
          TQ.el("span", { class: "tq-stat-big", text: TQ.fmt(cos, 3) })
        ));
        // a tiny bar so the magnitude is felt, mapped -1..1 -> 0..1
        detail.appendChild(TQ.barRow([cos], {
          labels: ["cos"], max: 1,
          format: function (v) { return TQ.fmt(v, 3); }
        }));

        if (i === j) {
          detail.appendChild(TQ.note("A vector compared with itself is perfectly aligned → cosine = 1.00. " +
            "That's why the diagonal glows brightest."));
        }
      }

      // We rebuild the (cheap 6x6) heatmap on each click so the selection
      // outline moves. The kernel's own onCell would survive a rebuild, but we
      // want full control of the selected-row/col props, so we wire clicks
      // ourselves via rewire() and leave onCell off the def.
      function buildHeat(selI, selJ) {
        return TQ.heatmap(simMatrix, {
          rowLabels: tokens, colLabels: tokens, cellSize: 46,
          min: -1, max: 1, // fix scale so colors mean the same thing as the formula range
          selectedRow: selI, selectedCol: selJ,
          format: function (v) { return TQ.fmt(v, 2); }
        });
      }
      var heat = buildHeat(1, 5);

      function rewire(node) {
        var cells = node.querySelectorAll(".tq-hm-cell");
        var k = 0;
        for (var rr = 0; rr < n; rr++) {
          for (var cc = 0; cc < n; cc++) {
            (function (i2, j2, cell) {
              cell.classList.add("is-clickable");
              cell.setAttribute("role", "button");
              cell.setAttribute("tabindex", "0");
              cell.addEventListener("click", function () { select(i2, j2); });
              cell.addEventListener("keydown", function (e) {
                if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(i2, j2); }
              });
            })(rr, cc, cells[k]);
            k++;
          }
        }
      }

      var heatHolder = TQ.el("div", { class: "tq-grow" }, heat);

      function select(i, j) {
        var fresh = buildHeat(i, j);
        rewire(fresh);
        heatHolder.replaceChild(fresh, heatHolder.firstChild);
        renderDetail(i, j);
      }

      // initial wiring + first detail
      rewire(heat);
      renderDetail(1, 5);

      simBlock.appendChild(TQ.el("div", { class: "tq-flexrow tq-sim-row" }, heatHolder, detail));
      simBlock.appendChild(TQ.note(
        "These embeddings are deterministically generated for the lesson (seeded, with a little hand-built " +
        "semantic structure) so the geometry is legible. In a real model these numbers are learned by " +
        "gradient descent over billions of tokens — but they're read and used exactly like this."
      ));
      root.appendChild(simBlock);

      /* ----------------------------- secondary interactive: pick & compare */
      var pickBlock = TQ.block(
        TQ.h(2, "Play: pick two tokens"),
        TQ.p("Same machinery, driven by you. Choose token A and token B; the dot product, norms, and " +
             "cosine recompute live from the real vectors.")
      );

      var aSel = TQ.segmented({
        options: tokens.map(function (t, i) { return { label: t, value: i }; }),
        value: 1,
        onChange: function () { updatePick(); }
      });
      var bSel = TQ.segmented({
        options: tokens.map(function (t, i) { return { label: t, value: i }; }),
        value: 4,
        onChange: function () { updatePick(); }
      });
      var pickOut = TQ.el("div", { class: "tq-panel" });

      function updatePick() {
        var i = aSel.get(), j = bSel.get();
        pickOut.innerHTML = "";
        var cos = TQ.cosine(emb[i], emb[j]);
        pickOut.appendChild(TQ.el("div", { class: "tq-sim-cos" },
          TQ.el("span", { class: "tq-stat-cap", text: "cos( " + tokens[i] + " , " + tokens[j] + " )" }),
          TQ.el("span", { class: "tq-stat-big", text: TQ.fmt(cos, 3) })
        ));
        pickOut.appendChild(TQ.barRow([cos], { labels: ["cos"], max: 1, format: function (v) { return TQ.fmt(v, 3); } }));
        pickOut.appendChild(TQ.vectorView(emb[i], { label: tokens[i], cellSize: 18 }));
        pickOut.appendChild(TQ.vectorView(emb[j], { label: tokens[j], cellSize: 18 }));
      }

      pickBlock.appendChild(TQ.el("div", { class: "tq-flexcol", style: { gap: "12px" } },
        TQ.el("div", { class: "tq-controls-row" },
          TQ.el("div", {}, TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "Token A"), aSel.el),
          TQ.el("div", {}, TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "Token B"), bSel.el)
        ),
        pickOut
      ));
      updatePick();
      root.appendChild(pickBlock);

      /* --------------------------------------------------- wrap-up takeaway */
      root.appendChild(TQ.block(
        TQ.h(2, "Why this matters for what's coming"),
        TQ.p(
          "Everything downstream — attention, multi-head, the KV cache, MLA — operates on vectors like these. ",
          "Attention's core move is comparing vectors with dot products (you just did that by hand). ",
          "If two tokens are aligned, their dot product is large, and one will ", TQ.el("strong", { text: "attend" }),
          " to the other. That's the next level."
        ),
        TQ.callout("Mental model to carry forward: a token = a point/direction in a high-dimensional space, " +
                   "and \"meaning\" = where it sits relative to everything else.")
      ));

      /* small scoped styles this level needs (kept minimal; colors come from
         CSS variables only — no hardcoded hex here) */
      injectOnce("tq-lvl01-css",
        ".tq-token-strip{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 14px}" +
        ".tq-token-chip{display:inline-flex;flex-direction:column;align-items:center;gap:2px;" +
          "background:var(--panel-hi);border:1px solid var(--line);border-radius:10px;padding:8px 14px}" +
        ".tq-token-idx{font-size:10px;color:var(--ink-mute);font-family:var(--mono)}" +
        ".tq-token-word{font-size:16px;font-weight:700}" +
        ".tq-vec-list{display:flex;flex-direction:column;gap:8px;margin-top:8px}" +
        ".tq-sim-row{align-items:flex-start;gap:20px}" +
        ".tq-sim-detail{min-width:260px;display:flex;flex-direction:column;gap:12px}" +
        ".tq-sim-head{display:flex;align-items:center;justify-content:space-between;gap:10px}" +
        ".tq-sim-pair{font-size:17px}.tq-sim-vs{color:var(--ink-mute);margin:0 7px}" +
        ".tq-sim-stats{display:flex;gap:8px;flex-wrap:wrap}" +
        ".tq-sim-cos{display:flex;flex-direction:column;gap:2px}");
    },

    quiz: [
      {
        q: "What is a token embedding, in one sentence?",
        choices: [
          "The raw UTF-8 bytes of the word",
          "A learned vector that represents a token, looked up from a trainable table",
          "The position of the word in the sentence",
          "The softmax probability of the next word"
        ],
        answer: 1,
        explain: "An embedding is a trainable vector (one row per vocab entry). It's the numeric stand-in " +
                 "for a token — analogous to a learned feature vector in a CNN, but for language."
      },
      {
        q: "Two tokens have cosine similarity ≈ 0.97. What does that tell you?",
        choices: [
          "Their vectors are nearly opposite in direction",
          "They are unrelated / orthogonal",
          "Their vectors point in almost the same direction (very related)",
          "One vector is exactly 0.97× the length of the other"
        ],
        answer: 2,
        explain: "Cosine measures direction-agreement, ignoring length. ~0.97 means the vectors are almost " +
                 "perfectly aligned — like \"The\" vs \"the\" in our toy sentence."
      },
      {
        q: "Cosine similarity divides the dot product by the product of the norms. Why drop the magnitude?",
        choices: [
          "To make the number always positive",
          "So the comparison reflects direction (meaning), not how long the vectors happen to be",
          "Because dot products are too slow to compute",
          "To convert the result into a probability"
        ],
        answer: 1,
        explain: "Normalizing by ‖a‖‖b‖ isolates orientation. Two aligned vectors score ~1 even if one is " +
                 "much longer — useful because we care about what a token means, not its incidental scale."
      },
      {
        q: "Why does it matter that the SAME toy sentence is used across the whole game?",
        choices: [
          "It makes the math run faster",
          "It's required by the transformer architecture",
          "Continuity: you can watch the identical tokens flow through attention, heads, and caching without re-learning the setup",
          "Sentences must always have six words"
        ],
        answer: 2,
        explain: "Holding the sentence constant means every later level reuses these exact vectors, so you " +
                 "track one concrete example end-to-end instead of juggling a new example each time."
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
