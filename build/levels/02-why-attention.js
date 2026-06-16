/* ============================================================================
 * LEVEL 02 — Why Attention? (Q/K/V)   (order 2)
 * ----------------------------------------------------------------------------
 * Same pattern as level 01:
 *   - one IIFE, ONE TQ.registerLevel call, NO other globals
 *   - render() = explanation blocks + a PRIMARY interactive "soft database
 *     lookup" bench, every number genuinely computed via TQ math
 *   - all color via TQ.colorFor / TQ.barRow / TQ.vectorView (one color language)
 *   - 4 quiz questions
 *
 * Math shown here, all real (from the shared toy sentence):
 *   - Q/K/V via TQ.toyQKV(8, 101, 202, 303)  (dk = 8)
 *   - for a chosen query i: scores_i[j] = TQ.dot(Q[i], K[j])
 *   - weights_i = TQ.softmax(scores_i)   <-- UNSCALED on purpose; the /sqrt(dk)
 *     scaling is deferred to level 3 (we mention it but do NOT apply it here, so
 *     the displayed weights match the "raw -> softmax" story). NOTE: we do NOT
 *     use TQ.attention().weights because that one divides by sqrt(dk).
 *   - output_i = sum_j weights_i[j] * V[j]
 * ========================================================================== */

(function () {
  "use strict";

  TQ.registerLevel({
    id: "why-attention",
    order: 2,
    title: "Why Attention? (Q/K/V)",
    icon: "🔍",
    tagline: "A soft, differentiable database lookup: queries ask, keys advertise, values deliver.",
    preread: "Illustrated Transformer",
    objectives: [
      "Grasp attention as a soft key-value lookup BEFORE seeing the formula",
      "See that a Query·Key dot product is a 'how relevant?' score, and softmax turns scores into a blend",
      "Understand that the output is a weighted average of Values, not a hard pick"
    ],

    render: function (root) {
      var toy = TQ.toy;
      var tokens = toy.tokens;
      var n = tokens.length;
      var dk = 8;

      /* ----- derive Q/K/V deterministically (the documented convention) ----- */
      var qkv = TQ.toyQKV(dk, 101, 202, 303);
      var Q = qkv.Q, K = qkv.K, V = qkv.V; // each n x dk

      // For a query index i: raw scores against every key, then UNSCALED softmax.
      function scoresFor(i) {
        var s = [];
        for (var j = 0; j < n; j++) s.push(TQ.dot(Q[i], K[j]));
        return s;
      }
      function weightsFor(i) {
        return TQ.softmax(scoresFor(i)); // softmax on RAW scores (no /sqrt(dk))
      }
      // output_i = sum_j w[j] * V[j]   (length dk)
      function outputFor(i) {
        var w = weightsFor(i);
        var out = TQ.zeros(dk);
        for (var j = 0; j < n; j++) {
          for (var c = 0; c < dk; c++) out[c] += w[j] * V[j][c];
        }
        return out;
      }
      function argmaxIdx(arr) {
        var bi = 0;
        for (var i = 1; i < arr.length; i++) if (arr[i] > arr[bi]) bi = i;
        return bi;
      }
      // Signed color scale for RAW scores so negatives read as cool/dark and
      // positives as warm/bright (TQ.barRow's bar LENGTH is |value|, so without
      // this a large negative score would look identical to a large positive one).
      function signedColorFn(scores) {
        var m = 0;
        for (var k = 0; k < scores.length; k++) m = Math.max(m, Math.abs(scores[k]));
        if (!m) m = 1;
        return function (v) { return TQ.colorForSigned(v, m); };
      }

      /* ----------------------------------------------- intro: the problem */
      root.appendChild(TQ.block(
        TQ.h(2, "The problem attention solves"),
        TQ.p(
          "A CNN mixes information ", TQ.el("strong", { text: "locally" }),
          " — a 3×3 kernel only sees its neighbors, and you stack layers to slowly grow the receptive field. ",
          "Language doesn't respect locality: in ",
          TQ.el("em", { text: "\"the cat that the dog chased sat on the mat,\"" }),
          " the word that binds ", TQ.el("strong", { text: "sat" }), " to its subject is ",
          TQ.el("strong", { text: "cat" }), " — seven tokens back."
        ),
        TQ.p(
          TQ.el("strong", { text: "Attention" }),
          " is the transformer's answer: a layer where every token can pull information directly from every ",
          "other token in ", TQ.el("strong", { text: "one hop" }),
          " — and, crucially, it ", TQ.el("em", { text: "learns which ones matter" }),
          " instead of using a fixed stencil. No fixed receptive field, no recurrence: just learned, ",
          "content-dependent routing of information."
        )
      ));

      /* ------------------------------------- mental model: database lookup */
      root.appendChild(TQ.block(
        TQ.h(2, "The mental model: a soft, differentiable database lookup"),
        TQ.p(
          "Think of a key-value store (a Python dict / hash map). You issue a ",
          TQ.el("strong", { text: "Query" }), "; you compare it against every ",
          TQ.el("strong", { text: "Key" }), "; the Key that matches hands back its ",
          TQ.el("strong", { text: "Value" }), ". Attention is the ",
          TQ.el("em", { text: "soft" }), " version of exactly this."
        ),
        TQ.p(
          "Each token emits three vectors, all linear projections of its embedding: a ",
          TQ.el("strong", { text: "Query" }), " (\"what am I looking for?\"), a ",
          TQ.el("strong", { text: "Key" }), " (\"what do I advertise / what am I about?\"), and a ",
          TQ.el("strong", { text: "Value" }), " (\"what I'll actually contribute if you pick me\"). ",
          "The lookup is fuzzy: instead of one Key winning, ", TQ.el("em", { text: "every" }),
          " Key gets a relevance score and the answer is a blend of all the Values weighted by those scores."
        ),
        TQ.callout(
          "Why \"differentiable\"? A hard hash-map lookup (argmax) has zero gradient almost everywhere, so it " +
          "can't be trained by gradient descent. A softmax blend has gradient everywhere — that's the whole " +
          "reason attention is learnable end-to-end."
        ),
        TQ.note(
          "Q/K/V here are real: each is the toy embedding matrix times a distinct projection " +
          "(Wq, Wk, Wv = TQ.randMatrix with seeds 101 / 202 / 303), giving dk = " + dk + "-dim vectors."
        )
      ));

      /* ----------------------- scoring + softmax narrative ----------------------- */
      root.appendChild(TQ.block(
        TQ.h(2, "Scoring with a dot product, blending with a softmax"),
        TQ.p(
          "How relevant is Key ", TQ.math("j"), " to Query ", TQ.math("i"), "? Take their dot product: ",
          TQ.math("score_ij = Q_i · K_j"), ". From Level 1 you know a dot product is large when two vectors ",
          "are aligned — so this is just \"how much does what ", TQ.math("i"),
          " wants match what ", TQ.math("j"), " advertises?\""
        ),
        TQ.p(
          "That gives a row of raw scores (any real numbers, positive or negative). To turn them into a usable ",
          "blend, run softmax: ", TQ.math("weights_i = softmax(score_i)"),
          ". Softmax exponentiates and normalizes, so the weights are all positive and sum to exactly ",
          TQ.el("strong", { text: "1" }), " — a probability distribution over where token ", TQ.math("i"),
          " pays attention. The output is then ", TQ.math("output_i = Σ_j weights_ij · V_j"), "."
        ),
        TQ.p(
          "Note what this is ", TQ.el("strong", { text: "NOT" }), ": it's not argmax, not a hard pick of the ",
          "single best key. A token can put 50% of its attention on one neighbor and spread the rest."
        ),
        TQ.note(
          "One detail deferred to Level 3: real attention divides the scores by √dk before softmax, to keep " +
          "numbers from blowing up as vectors get longer. The story is identical — only the scale changes — so " +
          "here we keep the math UNSCALED and the displayed weights match the raw → softmax narrative exactly."
        )
      ));

      /* ============================================================= *
       *  PRIMARY INTERACTIVE — the soft database lookup bench
       * ============================================================= */
      var benchBlock = TQ.block(
        TQ.h(2, "Bench: pick a query token and watch the lookup"),
        TQ.p(
          "Pick the ", TQ.el("strong", { text: "query token" }),
          " below. Row 1 is its raw ", TQ.math("Q·K"),
          " score against every key token; row 2 is the softmax weights derived from those scores; row 3 ",
          "assembles the ", TQ.el("strong", { text: "output" }),
          " as a weighted blend of the six Value vectors. Every number is computed live from the real toy embeddings."
        )
      );

      var legend = TQ.el("div", { class: "tq-legend" },
        TQ.el("span", { text: "low" }),
        TQ.el("div", { class: "tq-legend-scale" }),
        TQ.el("span", { text: "high" })
      );
      benchBlock.appendChild(legend);

      // --- controls: query selector + raw/softmax toggle ---
      var querySel = TQ.segmented({
        options: tokens.map(function (t, i) { return { label: t, value: i }; }),
        value: 1, // default 'cat'
        onChange: function () { updateBench(); }
      });
      var showWeights = TQ.toggle({
        label: "softmax weights (off = compare raw scores)",
        value: true,
        onChange: function () { updateBench(); }
      });

      var controls = TQ.el("div", { class: "tq-controls-row tq-wa-controls" },
        TQ.el("div", {},
          TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "Query token  i"),
          querySel.el
        ),
        TQ.el("div", {},
          TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "Middle row"),
          showWeights.el
        )
      );
      benchBlock.appendChild(controls);

      // readout strip + the three output regions
      var readout = TQ.el("div", { class: "tq-wa-readout" });
      var midCaption = TQ.el("div", { class: "tq-wa-caption" });
      var scoreRegion = TQ.el("div", { class: "tq-panel tq-wa-region" });
      var midRegion = TQ.el("div", { class: "tq-panel tq-wa-region" });
      var blendRegion = TQ.el("div", { class: "tq-panel tq-wa-region" });

      benchBlock.appendChild(readout);
      benchBlock.appendChild(TQ.el("div", { class: "tq-wa-grid" },
        TQ.el("div", {},
          TQ.el("div", { class: "tq-wa-rowhead", text: "1 · raw Q·K scores" }),
          scoreRegion
        ),
        TQ.el("div", {},
          TQ.el("div", { class: "tq-wa-rowhead tq-wa-midhead" }),
          midCaption,
          midRegion
        ),
        TQ.el("div", {},
          TQ.el("div", { class: "tq-wa-rowhead", text: "3 · blend of Values → output" }),
          blendRegion
        )
      ));

      function updateBench() {
        var i = querySel.get();
        var scores = scoresFor(i);
        var weights = weightsFor(i);
        var output = outputFor(i);
        var topKey = argmaxIdx(weights);
        var sumW = TQ.sum(weights);
        var weightsOn = showWeights.get();

        // --- readout strip: top key, its weight, running sum ---
        readout.innerHTML = "";
        readout.appendChild(TQ.kv("query", tokens[i]));
        readout.appendChild(TQ.kv("top key", tokens[topKey]));
        readout.appendChild(TQ.kv("its weight", TQ.fmt(weights[topKey], 3)));
        readout.appendChild(TQ.kv("Σ weights", TQ.fmt(sumW, 3)));
        readout.appendChild(weights[topKey] < 0.99
          ? TQ.badge("soft blend (no weight = 1.0)", "info")
          : TQ.badge("near-hard pick", "warn"));

        // --- row 1: raw scores (signed bars; negatives read correctly) ---
        scoreRegion.innerHTML = "";
        scoreRegion.appendChild(TQ.barRow(scores, {
          labels: tokens,
          highlight: argmaxIdx(scores),
          colorFn: signedColorFn(scores),
          format: function (v) { return TQ.fmt(v, 2); }
        }));

        // --- middle head + caption depends on toggle ---
        var midHead = midRegion.parentNode.querySelector(".tq-wa-midhead");
        midRegion.innerHTML = "";
        if (weightsOn) {
          midHead.textContent = "2 · softmax weights";
          midCaption.innerHTML = "";
          midCaption.appendChild(TQ.el("span", { text: "weights: positive, sum = " }));
          midCaption.appendChild(TQ.el("strong", { text: TQ.fmt(sumW, 2) }));
          midRegion.appendChild(TQ.barRow(weights, {
            labels: tokens, max: 1, highlight: topKey,
            format: function (v) { return TQ.fmt(v, 3); }
          }));
        } else {
          midHead.textContent = "2 · raw scores again (compare lens — toggle on to normalize)";
          var hasNeg = TQ.minOf(scores) < 0;
          midCaption.innerHTML = "";
          midCaption.appendChild(TQ.el("span", { text: "same as row 1 on purpose: raw scores are any real number" +
            (hasNeg ? " (includes negatives)" : "") + " — they don't sum to anything tidy; toggle on to see softmax normalize them" }));
          midRegion.appendChild(TQ.barRow(scores, {
            labels: tokens, highlight: argmaxIdx(scores),
            colorFn: signedColorFn(scores),
            format: function (v) { return TQ.fmt(v, 2); }
          }));
        }

        // --- row 3: each Value dimmed to its weight, then the assembled output ---
        blendRegion.innerHTML = "";
        var stack = TQ.el("div", { class: "tq-wa-vstack" });
        for (var j = 0; j < n; j++) {
          var w = weights[j];
          // opacity tracks weight so high-weight Values "fade in"
          var op = TQ.clamp(0.18 + 0.82 * w, 0.18, 1);
          var vrow = TQ.el("div", {
            class: "tq-wa-vrow" + (j === topKey ? " is-top" : ""),
            style: { opacity: String(op) }
          },
            TQ.vectorView(V[j], { label: tokens[j], cellSize: 18 }),
            TQ.el("span", { class: "tq-wa-vweight", text: "× " + TQ.fmt(w, 3) })
          );
          stack.appendChild(vrow);
        }
        blendRegion.appendChild(stack);
        blendRegion.appendChild(TQ.el("div", { class: "tq-wa-outsep", text: "Σ  weighted average  =" }));
        blendRegion.appendChild(TQ.el("div", { class: "tq-wa-output" },
          TQ.vectorView(output, { label: "output for " + tokens[i], cellSize: 22, showValues: false })
        ));
      }

      benchBlock.appendChild(TQ.callout(
        "Try query \"The\" (or \"the\"): it attends hardest to \"cat\" (~0.50) — the article reaching for its " +
        "noun — and \"the\" (#4) gives nearly the same pattern, echoing Level 1's twin geometry. Query \"sat\" " +
        "leans on \"on\" (~0.49); \"on\" attends back to \"sat\" — but the two weights differ, because Q and K " +
        "are different projections. Attention is content-dependent AND asymmetric."
      ));
      root.appendChild(benchBlock);
      updateBench();

      /* ============================================================= *
       *  STEPPER — the full pipeline, one step at a time
       * ============================================================= */
      var stepBlock = TQ.block(
        TQ.h(2, "Walkthrough: Project → Score → Normalize → Blend"),
        TQ.p(
          "Step through the pipeline for the currently selected query (",
          TQ.el("strong", { text: tokens[querySel.get()] }),
          " — change it above and the walkthrough follows). Use Prev / Next / Play."
        )
      );

      function curQuery() { return querySel.get(); }

      var stepper = TQ.stepper([
        {
          label: "1 · Project",
          run: function (c) {
            var i = curQuery();
            c.appendChild(TQ.p(
              "Each token's embedding is multiplied by a learned projection. ",
              TQ.math("Q_i = emb_i · Wq"), " for the query; every token gets a ",
              TQ.math("K = emb · Wk"), " (Values are made the same way with ", TQ.math("Wv"), ")."
            ));
            c.appendChild(TQ.el("div", { class: "tq-wa-region tq-panel" },
              TQ.vectorView(Q[i], { label: "Q  (query \"" + tokens[i] + "\")", cellSize: 20 })
            ));
            var keyStack = TQ.el("div", { class: "tq-wa-vstack tq-panel tq-wa-region" });
            for (var j = 0; j < n; j++) {
              keyStack.appendChild(TQ.vectorView(K[j], { label: "K · " + tokens[j], cellSize: 18 }));
            }
            c.appendChild(TQ.el("div", { class: "tq-wa-rowhead", text: "Keys for all six tokens" }));
            c.appendChild(keyStack);
          }
        },
        {
          label: "2 · Score",
          run: function (c) {
            var i = curQuery();
            var scores = scoresFor(i);
            var am = argmaxIdx(scores);
            c.appendChild(TQ.p(
              TQ.math("score_ij = Q_i · K_j"), " across the six keys. Highest (the argmax key) is ",
              TQ.el("strong", { text: tokens[am] }), " at ", TQ.math(TQ.fmt(scores[am], 2)),
              ". Scores can be negative — that's fine, softmax handles it next."
            ));
            c.appendChild(TQ.el("div", { class: "tq-wa-region tq-panel" },
              TQ.barRow(scores, { labels: tokens, highlight: am, colorFn: signedColorFn(scores),
                format: function (v) { return TQ.fmt(v, 2); } })
            ));
          }
        },
        {
          label: "3 · Normalize",
          run: function (c) {
            var i = curQuery();
            var scores = scoresFor(i);
            var weights = weightsFor(i);
            c.appendChild(TQ.p(
              "Softmax exponentiates and normalizes: negatives vanish, everything becomes positive, and the ",
              "bars rescale to sum to exactly ", TQ.el("strong", { text: TQ.fmt(TQ.sum(weights), 2) }), "."
            ));
            c.appendChild(TQ.el("div", { class: "tq-wa-rowhead", text: "before · raw scores" }));
            c.appendChild(TQ.el("div", { class: "tq-wa-region tq-panel" },
              TQ.barRow(scores, { labels: tokens, colorFn: signedColorFn(scores),
                format: function (v) { return TQ.fmt(v, 2); } })
            ));
            c.appendChild(TQ.el("div", { class: "tq-wa-rowhead", text: "after · softmax weights (sum = 1)" }));
            c.appendChild(TQ.el("div", { class: "tq-wa-region tq-panel" },
              TQ.barRow(weights, { labels: tokens, max: 1, highlight: argmaxIdx(weights),
                format: function (v) { return TQ.fmt(v, 3); } })
            ));
          }
        },
        {
          label: "4 · Blend",
          run: function (c) {
            var i = curQuery();
            var weights = weightsFor(i);
            var output = outputFor(i);
            var topKey = argmaxIdx(weights);
            c.appendChild(TQ.p(
              "Stack the six Value vectors, each dimmed to its weight, then sum: ",
              TQ.math("output_i = Σ_j weights_ij · V_j"),
              ". The result leans toward the top key (", TQ.el("strong", { text: tokens[topKey] }),
              ") but is never identical to any single V — a weighted average, not a copy."
            ));
            var stack = TQ.el("div", { class: "tq-wa-vstack tq-panel tq-wa-region" });
            for (var j = 0; j < n; j++) {
              var w = weights[j];
              var op = TQ.clamp(0.18 + 0.82 * w, 0.18, 1);
              stack.appendChild(TQ.el("div", {
                class: "tq-wa-vrow" + (j === topKey ? " is-top" : ""),
                style: { opacity: String(op) }
              },
                TQ.vectorView(V[j], { label: tokens[j], cellSize: 18 }),
                TQ.el("span", { class: "tq-wa-vweight", text: "× " + TQ.fmt(w, 3) })
              ));
            }
            c.appendChild(stack);
            c.appendChild(TQ.el("div", { class: "tq-wa-outsep", text: "Σ  =" }));
            c.appendChild(TQ.el("div", { class: "tq-wa-output tq-panel tq-wa-region" },
              TQ.vectorView(output, { label: "output for " + tokens[i], cellSize: 22 })
            ));
          }
        }
      ]);
      stepBlock.appendChild(stepper);
      root.appendChild(stepBlock);

      /* --------------------------------------------------- wrap-up takeaway */
      root.appendChild(TQ.block(
        TQ.h(2, "What to carry forward"),
        TQ.p(
          "Attention = a soft database lookup. ", TQ.el("strong", { text: "Query" }),
          " asks, ", TQ.el("strong", { text: "Key" }), " advertises (matched by dot product), ",
          TQ.el("strong", { text: "Value" }), " delivers — and softmax turns arbitrary scores into a blend that ",
          "sums to 1. No single token wins outright; the output is a weighted average."
        ),
        TQ.callout(
          "Next level: we add the missing /√dk scaling and stack several of these lookups in parallel " +
          "(multi-head). Same lookup, run many times with different projections."
        )
      ));

      /* scoped styles for this level — colors only via CSS variables, never hex */
      injectOnce("tq-lvl02-css",
        ".tq-wa-controls{align-items:flex-start;gap:28px;margin:6px 0 14px;flex-wrap:wrap}" +
        ".tq-wa-readout{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:6px 0 14px}" +
        ".tq-wa-grid{display:flex;flex-direction:column;gap:16px}" +
        ".tq-wa-rowhead{font-size:12px;font-weight:700;letter-spacing:.02em;color:var(--ink-mute);" +
          "text-transform:uppercase;margin-bottom:6px}" +
        ".tq-wa-caption{font-size:13px;color:var(--ink-mute);margin-bottom:8px}" +
        ".tq-wa-region{padding:12px}" +
        ".tq-wa-vstack{display:flex;flex-direction:column;gap:6px}" +
        ".tq-wa-vrow{display:flex;align-items:center;gap:12px;transition:opacity .25s ease}" +
        ".tq-wa-vrow.is-top{outline:1px solid var(--line);outline-offset:3px;border-radius:8px}" +
        ".tq-wa-vweight{font-family:var(--mono);font-size:12px;color:var(--ink-mute);min-width:64px}" +
        ".tq-wa-outsep{font-family:var(--mono);font-size:12px;color:var(--ink-mute);" +
          "margin:10px 0 6px;letter-spacing:.04em}" +
        ".tq-wa-output{padding-top:2px}");
    },

    quiz: [
      {
        q: "In the Q/K/V lookup metaphor, what role does the Value vector play?",
        choices: [
          "It scores how relevant each token is to the query",
          "It is the content a token contributes to the output if it gets attended to",
          "It is the position of the token in the sentence",
          "It is the softmax-normalized attention weight"
        ],
        answer: 1,
        explain: "Query asks, Key advertises (and is matched against the query via dot product), and Value is " +
                 "the actual content delivered — the output is a weighted average of Values."
      },
      {
        q: "The raw attention scores for a query are Q·K dot products. Why pass them through a softmax before using them?",
        choices: [
          "To make the largest score win and discard the rest (hard argmax)",
          "To turn arbitrary real-valued scores into positive weights that sum to 1, giving a soft blend",
          "To divide the scores by the square root of the key dimension",
          "To convert the Value vectors into Keys"
        ],
        answer: 1,
        explain: "Raw scores can be any real number (including negatives). Softmax exponentiates and normalizes " +
                 "them into a probability distribution — all positive, summing to 1 — so the output is a soft " +
                 "weighted average, not a hard pick."
      },
      {
        q: "For query 'sat' in the toy sentence, the attention weights come out roughly [0.06, 0.22, 0.05, 0.49, 0.07, 0.11] over [The, cat, sat, on, the, mat]. What is the resulting output vector?",
        choices: [
          "Exactly the Value vector of 'on', since it has the highest weight",
          "A blend of all six Value vectors, weighted by those numbers — leaning toward 'on' but not equal to it",
          "The Value vector of 'sat' itself, since it's the query",
          "The average of the Query and Key vectors of 'on'"
        ],
        answer: 1,
        explain: "Output = Σ weight·Value across ALL tokens. 'on' dominates at 0.49 so the result leans its way, " +
                 "but the other five Values still contribute — it's a weighted average, never a hard copy of the top one."
      },
      {
        q: "Token i attends to token j with some weight. Is the weight token j gives to token i necessarily the same?",
        choices: [
          "Yes, attention weights are always symmetric",
          "No — Q and K are different learned projections, so Q_i·K_j generally differs from Q_j·K_i",
          "Yes, because the dot product is commutative",
          "Only if i and j are the same word"
        ],
        answer: 1,
        explain: "Although the dot product a·b equals b·a, the attention score uses DIFFERENT vectors for the two " +
                 "roles: Q_i·K_j vs Q_j·K_i. Since Q and K are separate projections of the embeddings, attention " +
                 "is directional — 'on'→'sat' need not match 'sat'→'on'."
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
