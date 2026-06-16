/* ============================================================================
 * LEVEL 03 — Scaled Dot-Product Self-Attention   (order 3)
 * ----------------------------------------------------------------------------
 * Follows the gold-standard pattern from 01-tokens-embeddings.js:
 *   - one IIFE, ONE TQ.registerLevel call, NO other globals
 *   - render() = short explanation blocks + a PRIMARY interactive lab, every
 *     number genuinely computed via TQ math (no hardcoded weights/curves)
 *   - all color via TQ.colorFor / TQ.heatmap / TQ.barRow (one color language)
 *   - 2-4 quiz questions
 *
 * Math shown here, all real:
 *   - {Q,K,V} = TQ.toyQKV(dk, 101,202,303) on the shared 6-token sentence
 *   - base scores S = Q·Kᵀ are FIXED (independent of the slider)
 *   - on every slider input: scaled = S/s, weights = TQ.softmaxRows(scaled),
 *     redrawing the 6×6 heatmap + a 3-stage breakdown + a live peak-weight stat
 *   - a dk segmented control rebuilds Q,K,V so raw-score spread grows with dk
 *   - "snap to √dk" returns the divisor to the healthy regime
 * ========================================================================== */

(function () {
  "use strict";

  TQ.registerLevel({
    id: "self-attention",
    order: 3,
    title: "Scaled Dot-Product Self-Attention",
    icon: "🧮",
    tagline: "The real mechanism: softmax(QKᵀ / √dk) V, every number live.",
    preread: "Illustrated Transformer / Basics",
    objectives: [
      "Compute self-attention end-to-end on the shared sentence with real numbers",
      "Trace one query row: raw dot product → ÷√dk → softmax weight",
      "Explain WHY we divide by √dk (large dot products saturate softmax → vanishing gradients)"
    ],

    render: function (root) {
      var tokens = TQ.toy.tokens;
      var n = tokens.length;

      /* -------------------------------------------------- intro narrative */
      root.appendChild(TQ.block(
        TQ.h(2, "Self-attention: every token re-asks the room about itself"),
        TQ.p(
          "In your CNN days a pixel's new value came from a fixed convolution kernel — the same learned ",
          "weights slid over every location. Attention rips that fixedness out. Each token emits three ",
          "projections of its own embedding: a ", TQ.el("strong", { text: "Query" }), " (\"what am I looking for?\"), ",
          "a ", TQ.el("strong", { text: "Key" }), " (\"what do I offer?\"), and a ",
          TQ.el("strong", { text: "Value" }), " (\"what I'll actually hand over\"). We build them with three ",
          "learned matrices: ", TQ.math("Q = E·Wq"), ", ", TQ.math("K = E·Wk"), ", ", TQ.math("V = E·Wv"),
          ", where ", TQ.math("E"), " is the 6×16 toy embedding matrix and each ", TQ.math("W"), " is 16×dk."
        ),
        TQ.p(
          TQ.el("strong", { text: "\"Self\"" }), "-attention just means Q, K and V all come from the ",
          TQ.el("strong", { text: "same sequence" }), " (these same six tokens) — a token compares itself ",
          "against its own neighbors, not some other input. The output for each token is a content-weighted ",
          "blend of every token's Value, where the weights are recomputed on the fly from how well that token's ",
          "Query aligns with every Key. No fixed kernel; the mixing weights come from the data every forward pass. ",
          "That data-dependent routing is why transformers eat long-range dependencies for breakfast where a CNN ",
          "needed many stacked layers to see across an image."
        )
      ));

      root.appendChild(TQ.block(
        TQ.h(2, "The formula is one line — and every piece is a dot product you already trust"),
        TQ.p(
          TQ.math("Attention(Q,K,V) = softmax(QKᵀ / √dk) · V"),
          ". Read it right-to-left. ", TQ.math("QKᵀ"), " is the score matrix: entry ", TQ.math("[i,j]"),
          " is the raw dot product of token i's Query with token j's Key — the same un-normalized alignment ",
          "you computed by hand in Level 1, but now between learned projections. Big dot product = strong match ",
          "= token i wants j's Value. Then divide every score by ", TQ.math("√dk"),
          " (the temperature fix, next block). Then ", TQ.el("strong", { text: "softmax" }),
          " runs across each ROW, turning scores into a probability distribution that sums to 1.0 — the ",
          "attention weights. Finally multiply that weight matrix by ", TQ.math("V"),
          ", so each token's output is a convex combination of all Values, dominated by the tokens it scored highest against."
        ),
        TQ.callout(
          "Nothing magic — three matmuls, one scale, one softmax. The lab below runs exactly this pipeline on the " +
          "shared sentence, with the numbers recomputed live as you turn the knobs."
        )
      ));

      root.appendChild(TQ.block(
        TQ.h(2, "Why ÷√dk: the dot product's variance grows with dk, and softmax has no chill"),
        TQ.p(
          "A score is a dot product of two ", TQ.math("dk"), "-dimensional vectors: ",
          TQ.math("q·k = Σ qᵢkᵢ"), " over dk terms. If the components are roughly independent with unit variance, ",
          "that sum has variance ~dk, so its standard deviation grows like ", TQ.math("√dk"),
          ". Crank dk from 8 to 64 and the typical magnitude of your raw scores roughly triples ",
          "(√64/√8 ≈ 2.8×) — for free, just from adding dimensions, with no change in actual semantic alignment."
        ),
        TQ.p(
          "Now feed those into softmax. Softmax is exponential, so it's brutally sensitive to the ",
          TQ.el("strong", { text: "spread" }), " of its inputs: when scores are large and spread out, ",
          TQ.math("exp()"), " makes the biggest one swamp the rest and the distribution collapses toward one-hot. ",
          "You know exactly why that's poison — softmax in its saturated regime has gradients near zero (the same ",
          "vanishing-gradient flatness you fought with saturated sigmoids), so the model can barely learn which ",
          "token to attend to. Dividing by ", TQ.math("√dk"), " cancels the dimension-induced inflation, holding ",
          "the score variance ~constant regardless of head size, keeping softmax in its smooth, high-gradient regime. ",
          "It's a temperature knob set to exactly the value that neutralizes dk."
        ),
        TQ.note(
          "The interactive below lets you yank that knob and watch the peak weight climb toward 1.0 as you shrink " +
          "the divisor — saturation, live, on real recomputed numbers."
        )
      ));

      root.appendChild(TQ.block(
        TQ.h(2, "Causal masking: how a decoder forbids peeking at the future"),
        TQ.p(
          "Everything so far let every token attend to every other token — including ones to its RIGHT. That's fine ",
          "for an encoder, but a ", TQ.el("strong", { text: "decoder" }), " generates left-to-right: when it predicts ",
          "token i it cannot be allowed to look at tokens it hasn't produced yet. The fix is a ",
          TQ.el("strong", { text: "causal mask" }), " — before softmax, set every \"future\" score ",
          TQ.math("S[i,j] with j>i"), " to ", TQ.math("−∞"), ". Since ", TQ.math("exp(−∞)=0"),
          ", those cells get exactly zero weight, and — this is the load-bearing part — softmax ",
          TQ.el("strong", { text: "renormalizes the surviving cells in that same row so they still sum to 1.0" }),
          ". The masked mass doesn't vanish; it gets redistributed to the past."
        ),
        TQ.note(
          "Flip the new \"causal mask (decoder)\" toggle in the lab below and watch the upper triangle of the heatmap " +
          "go dark while the surviving cells in each row brighten to absorb the freed weight — the row-Σ chip stays " +
          "pinned at 1.000 the whole time. That redistribution is the entire idea, and it's invisible in a static grid."
        )
      ));

      /* ================================================================== *
       *  PRIMARY INTERACTIVE — the scaled-dot-product attention lab
       * ================================================================== */

      var lab = TQ.block(
        TQ.h(2, "The lab: softmax(QKᵀ / scale) · V, recomputed live"),
        TQ.p(
          "The heatmap is the full 6×6 attention-weight matrix on the shared sentence. ",
          TQ.el("strong", { text: "Click any query ROW" }), " to expand a three-stage breakdown for that ",
          "token — raw ", TQ.math("Q·K"), ", scaled (÷scale), and the softmax weight. Drag the ",
          TQ.el("strong", { text: "scaling-factor slider" }), ": as the divisor shrinks toward 0 you'll watch the ",
          "row sharpen and the ", TQ.el("strong", { text: "peak weight" }), " climb toward 1.0 (softmax saturating ",
          "into a near-one-hot spike). At the proper ", TQ.math("√dk"), " it sits in a smooth, healthy regime. ",
          "Flip the ", TQ.el("strong", { text: "causal mask (decoder)" }), " toggle to block future keys ",
          "(", TQ.math("j>i"), "): the upper triangle drops to 0.000 and each row renormalizes over the past — ",
          "the row-Σ chip stays at 1.000 throughout."
        )
      );

      // ---- live state -------------------------------------------------------
      // dk, base scores S (fixed per dk), the current divisor, and selected row.
      var SEED_Q = 101, SEED_K = 202, SEED_V = 303;
      var dk = 8;
      var qkv, baseScores, V, sqrtDk;
      var divisor = Math.sqrt(dk);
      var selectedRow = 0;
      var causal = false; // decoder causal mask: block future keys (j > i)

      function rebuildForDk(newDk) {
        dk = newDk;
        qkv = TQ.toyQKV(dk, SEED_Q, SEED_K, SEED_V);
        // base scores are FIXED, independent of the slider
        baseScores = TQ.matmul(qkv.Q, TQ.transpose(qkv.K)); // 6×6, == TQ.attention(...).scores
        V = qkv.V;
        sqrtDk = Math.sqrt(dk);
      }
      rebuildForDk(dk);

      // Apply the causal mask to a copy of the base scores: future keys (j > i)
      // get −∞ so exp() drives them to exactly 0. When causal is off, returns
      // baseScores untouched. (Math.exp(-Infinity) === 0, so masked cells get
      // EXACTLY zero weight and the surviving cells renormalize to sum to 1.0.)
      function maskedScores() {
        if (!causal) return baseScores;
        var M = [];
        for (var i = 0; i < baseScores.length; i++) {
          var row = [];
          for (var j = 0; j < baseScores[i].length; j++) {
            row.push(j > i ? -Infinity : baseScores[i][j]);
          }
          M.push(row);
        }
        return M;
      }

      // weights from the CURRENT divisor (and mask) — genuine recompute through
      // softmax. The mask is applied to the UNscaled scores, then the existing
      // /scale + softmax pipeline runs unchanged.
      function computeWeights() {
        var scaled = TQ.scaleMat(maskedScores(), 1 / divisor); // Sᵢⱼ / s  (−∞ stays −∞)
        return TQ.softmaxRows(scaled);                         // per-row softmax
      }

      // peak weight of the selected row + mean of every row's max (saturation metric)
      function meanPeak(weights) {
        var s = 0;
        for (var i = 0; i < weights.length; i++) s += TQ.maxOf(weights[i]);
        return s / weights.length;
      }

      // ---- DOM holders ------------------------------------------------------
      var heatHolder = TQ.el("div", { class: "tq-grow" });
      var statRow = TQ.el("div", { class: "tq-sa-stats" });
      var breakdown = TQ.el("div", { class: "tq-panel tq-sa-breakdown" });
      // renormalization-proof callout under the heatmap, shown only when masked
      var maskCallout = TQ.el("div", { class: "tq-callout tq-sa-maskcallout", style: { display: "none" } });

      // legend (shared color language)
      var legend = TQ.el("div", { class: "tq-legend" },
        TQ.el("span", { text: "0.0" }),
        TQ.el("div", { class: "tq-legend-scale" }),
        TQ.el("span", { text: "1.0" })
      );

      function buildHeat(weights) {
        return TQ.heatmap(weights, {
          rowLabels: tokens, colLabels: tokens, cellSize: 46,
          min: 0, max: 1, // fixed scale so colors mean the same probability everywhere
          selectedRow: selectedRow,
          // masked (future) cells are genuine 0 under the mask — render '∅' so
          // they read as "forbidden", not merely "tiny".
          format: function (v) { return (causal && v === 0) ? "∅" : TQ.fmt(v, 2); }
        });
      }

      // After a heatmap is built, tag the strict-upper-triangle cells as masked
      // (only when causal is on) so scoped CSS can dim them. Purely cosmetic —
      // the numbers themselves are already 0 from the −∞ softmax.
      function markMasked(node) {
        var cells = node.querySelectorAll(".tq-hm-cell");
        var k = 0;
        for (var r = 0; r < n; r++) {
          for (var c = 0; c < n; c++) {
            if (causal && c > r) cells[k].classList.add("is-masked");
            k++;
          }
        }
      }

      // rewire heatmap cells: clicking ANY cell selects that token's query row.
      function rewire(node) {
        var cells = node.querySelectorAll(".tq-hm-cell");
        var k = 0;
        for (var r = 0; r < n; r++) {
          for (var c = 0; c < n; c++) {
            (function (rowIdx, cell) {
              cell.classList.add("is-clickable");
              cell.setAttribute("role", "button");
              cell.setAttribute("tabindex", "0");
              cell.addEventListener("click", function () { selectRow(rowIdx); });
              cell.addEventListener("keydown", function (e) {
                if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectRow(rowIdx); }
              });
            })(r, cells[k]);
            k++;
          }
        }
      }

      function renderStats(weights) {
        statRow.innerHTML = "";
        var peak = TQ.maxOf(weights[selectedRow]);
        var mp = meanPeak(weights);
        var rowSum = TQ.sum(weights[selectedRow]);

        var regime = peak > 0.85 ? ["near one-hot — saturated", "warn"]
                   : peak > 0.45 ? ["sharpening", "info"]
                   : ["smooth blend — healthy", "good"];

        var peakStat = TQ.el("div", { class: "tq-sa-peak" },
          TQ.el("span", { class: "tq-stat-cap", text: "peak weight (row " + tokens[selectedRow] + ")" }),
          TQ.el("span", { class: "tq-stat-big", text: TQ.fmt(peak, 3) }),
          TQ.badge(regime[0], regime[1]),
          causal ? TQ.badge("causal · lower-triangular", "info") : null
        );
        statRow.appendChild(peakStat);
        statRow.appendChild(TQ.el("div", { class: "tq-sa-chips" },
          TQ.kv("divisor", TQ.fmt(divisor, 2)),
          TQ.kv("√dk", TQ.fmt(sqrtDk, 2)),
          TQ.kv("mean peak (all rows)", TQ.fmt(mp, 3)),
          TQ.kv("row Σ", TQ.fmt(rowSum, 3))
        ));
      }

      function renderBreakdown(weights) {
        breakdown.innerHTML = "";
        var r = selectedRow;
        // For the bar DISPLAY of stages 1 & 2, show masked future entries as 0 so
        // TQ.barRow never tries to draw −∞. The actual softmax (stage 3 / weights)
        // still used −∞ internally, so its zeros are genuinely computed.
        var raw = baseScores[r].map(function (x, j) { return (causal && j > r) ? 0 : x; });
        var scaled = raw.map(function (x, j) { return (causal && j > r) ? 0 : x / divisor; });
        var w = weights[r];
        var rawMax = TQ.maxOf([].concat.apply([], baseScores).map(function (x) { return Math.abs(x); }));

        breakdown.appendChild(TQ.el("div", { class: "tq-sa-bd-head" },
          TQ.el("span", { class: "tq-sa-bd-title" },
            "Query row ",
            TQ.el("strong", { text: "\"" + tokens[r] + "\"" }),
            " (#", String(r), ") vs every Key"
          )
        ));

        breakdown.appendChild(TQ.el("div", { class: "tq-sa-stage" },
          TQ.el("div", { class: "tq-sa-stage-label" },
            TQ.el("span", { class: "tq-sa-stage-n", text: "1" }),
            TQ.el("span", { text: "raw  " }), TQ.math("Q·Kᵀ")
          ),
          TQ.barRow(raw, {
            labels: tokens, max: rawMax || 1,
            format: function (v) { return TQ.fmt(v, 2); }
          })
        ));

        breakdown.appendChild(TQ.el("div", { class: "tq-sa-stage" },
          TQ.el("div", { class: "tq-sa-stage-label" },
            TQ.el("span", { class: "tq-sa-stage-n", text: "2" }),
            TQ.el("span", { text: "scaled  ÷ " + TQ.fmt(divisor, 2) })
          ),
          // Stage 2 gets its OWN bar scale (rawMax/divisor) so the bars never
          // clamp when the divisor < 1 (scaled magnitudes exceed the raw range).
          // Keeping each stage on its own scale is what makes the "uniform
          // shrink/grow" read correctly across the whole divisor range.
          TQ.barRow(scaled, {
            labels: tokens, max: (rawMax / divisor) || 1,
            format: function (v) { return TQ.fmt(v, 2); }
          })
        ));

        breakdown.appendChild(TQ.el("div", { class: "tq-sa-stage" },
          TQ.el("div", { class: "tq-sa-stage-label" },
            TQ.el("span", { class: "tq-sa-stage-n", text: "3" }),
            TQ.el("span", { text: "softmax weight  " }),
            TQ.badge("Σ = " + TQ.fmt(TQ.sum(w), 3), "info")
          ),
          TQ.barRow(w, {
            labels: tokens, max: 1,
            format: function (v) { return TQ.fmt(v, 3); }
          })
        ));

        breakdown.appendChild(TQ.note(
          "Stage 2 is just stage 1 uniformly shrunk by the same factor everywhere — yet softmax turns that " +
          "uniform shrink into a dramatically flatter distribution. Softmax cares about the SPREAD of its inputs, " +
          "and the divisor controls that spread."
        ));
      }

      // master refresh: recompute weights, redraw heatmap + stats + breakdown.
      function refresh() {
        var weights = computeWeights();
        var fresh = buildHeat(weights);
        rewire(fresh);
        markMasked(fresh);
        if (heatHolder.firstChild) heatHolder.replaceChild(fresh, heatHolder.firstChild);
        else heatHolder.appendChild(fresh);
        renderStats(weights);
        renderBreakdown(weights);
        // the renormalization-proof callout under the heatmap appears only when masked
        maskCallout.style.display = causal ? "" : "none";
        if (causal) {
          var rs = TQ.sum(weights[selectedRow]);
          maskCallout.innerHTML = "";
          maskCallout.appendChild(TQ.el("span", { class: "tq-callout-mark", text: "✦", "aria-hidden": "true" }));
          maskCallout.appendChild(TQ.el("div", { class: "tq-callout-body" },
            "Row ", TQ.el("strong", { text: "\"" + tokens[selectedRow] + "\"" }),
            " now blocks the " + (n - 1 - selectedRow) + " future key" + ((n - 1 - selectedRow) === 1 ? "" : "s") +
            " to its right (shown ∅). Softmax redistributed that mass across the " + (selectedRow + 1) +
            " allowed cell" + (selectedRow === 0 ? "" : "s") + " — and the row still sums to ",
            TQ.el("strong", { text: TQ.fmt(rs, 3) }),
            ". Masking didn't delete probability; it relocated it to the past."
          ));
        }
      }

      function selectRow(r) {
        selectedRow = TQ.clamp(r, 0, n - 1);
        refresh();
      }

      // ---- controls ---------------------------------------------------------
      var slider = TQ.slider({
        min: 0.1, max: 12, step: 0.05, value: divisor,
        label: "scaling divisor",
        format: function (v) { return TQ.fmt(v, 2); },
        onInput: function (v) { divisor = v; refresh(); }
      });

      var snapBtn = TQ.el("button", { class: "tq-btn tq-btn-accent", type: "button" },
        "snap to √dk (" + TQ.fmt(sqrtDk, 2) + ")"
      );
      snapBtn.addEventListener("click", function () {
        divisor = sqrtDk;
        slider.set(divisor);
        refresh();
      });

      var causalToggle = TQ.toggle({
        label: "causal mask (decoder) — block future keys (j>i)",
        value: false,
        onChange: function (on) { causal = on; refresh(); }
      });

      var dkSeg = TQ.segmented({
        options: [
          { label: "dk 4", value: 4 },
          { label: "dk 8", value: 8 },
          { label: "dk 16", value: 16 },
          { label: "dk 32", value: 32 }
        ],
        value: 8,
        onChange: function (newDk) {
          rebuildForDk(newDk);
          divisor = sqrtDk;            // snap divisor to the new √dk
          slider.set(divisor);
          snapBtn.textContent = "snap to √dk (" + TQ.fmt(sqrtDk, 2) + ")";
          refresh();
        }
      });

      var controls = TQ.el("div", { class: "tq-sa-controls" },
        TQ.el("div", { class: "tq-sa-ctrl-cell tq-grow" }, slider.el),
        TQ.el("div", { class: "tq-sa-ctrl-cell" },
          TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "head size dk"),
          dkSeg.el
        ),
        TQ.el("div", { class: "tq-sa-ctrl-cell" }, snapBtn),
        TQ.el("div", { class: "tq-sa-ctrl-cell tq-sa-maskcell" },
          TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "masking"),
          causalToggle.el
        )
      );

      lab.appendChild(controls);
      lab.appendChild(legend);
      lab.appendChild(TQ.el("div", { class: "tq-flexrow tq-sa-row" },
        TQ.el("div", { class: "tq-flexcol tq-grow", style: { gap: "10px" } }, heatHolder, statRow, maskCallout),
        breakdown
      ));
      lab.appendChild(TQ.callout(
        "Drag the divisor DOWN toward 0.1 and watch the selected row collapse toward one-hot — the peak weight " +
        "climbs from ≈0.29 (at √dk) up past 0.55, 0.74, toward 1.00. The heatmap row goes from a soft gradient " +
        "to a single blazing cell. THAT is softmax saturation, on real numbers."
      ));
      lab.appendChild(TQ.note(
        "Try this too: set dk to 32 and the slider to 1.0 (no scaling). The raw scores spread wider — std climbs " +
        "roughly as √dk — and the unscaled distribution sharpens for free. That inflation is the entire motivation " +
        "for the √dk denominator. Because softmax weights always sum to exactly 1.000, the output is always a " +
        "convex combination of Values: a token can amplify or ignore neighbors, but never invents signal outside the Value set."
      ));

      // first paint
      refresh();
      root.appendChild(lab);

      /* --------------------------------------------------- wrap-up takeaway */
      root.appendChild(TQ.block(
        TQ.h(2, "What you just built"),
        TQ.p(
          "One head of self-attention, end to end, on the shared sentence: project to Q/K/V, score with ",
          TQ.math("QKᵀ"), ", scale by ", TQ.math("1/√dk"), ", softmax each row into attention weights, blend the ",
          "Values. The next level stacks several of these heads side by side — each with its own Wq/Wk/Wv — so the ",
          "model can attend along multiple relationships at once."
        ),
        TQ.callout("Mental model to carry forward: attention = data-dependent routing. The weights are a fresh " +
                   "softmax over learned dot products every forward pass — there is no fixed kernel."),
        TQ.p(
          "You also saw the ", TQ.el("strong", { text: "causal mask" }), " mechanic up close: −∞ on the future, then ",
          "softmax renormalizes so every row still sums to 1.0. Hold onto that. In Levels 6 and 7 you'll meet it again ",
          "from the other side — there the decoder's KV-cache makes masking essentially ", TQ.el("em", { text: "free" }),
          ", because future positions simply don't exist in the cache yet. Same rule (\"only look left\"), seen here as ",
          "the mechanism and there as why it costs nothing to enforce."
        )
      ));

      /* small scoped styles (colors via CSS variables only — no hardcoded hex) */
      injectOnce("tq-lvl03-css",
        ".tq-sa-controls{display:flex;gap:18px;align-items:flex-end;flex-wrap:wrap;margin:6px 0 14px}" +
        ".tq-sa-ctrl-cell{display:flex;flex-direction:column;justify-content:flex-end}" +
        ".tq-sa-row{align-items:flex-start;gap:20px;margin-top:6px}" +
        ".tq-sa-breakdown{min-width:300px;display:flex;flex-direction:column;gap:12px}" +
        ".tq-sa-stats{display:flex;flex-direction:column;gap:8px}" +
        ".tq-sa-peak{display:flex;align-items:center;gap:12px;flex-wrap:wrap}" +
        ".tq-sa-chips{display:flex;gap:8px;flex-wrap:wrap}" +
        ".tq-sa-bd-head{font-size:15px;border-bottom:1px solid var(--line);padding-bottom:8px}" +
        ".tq-sa-stage{display:flex;flex-direction:column;gap:6px}" +
        ".tq-sa-stage-label{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--ink-soft)}" +
        ".tq-sa-stage-n{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;" +
          "border-radius:50%;background:var(--panel-hi);border:1px solid var(--line);font-size:11px;font-weight:700}" +
        ".tq-sa-bd-title{font-size:15px}" +
        // causal-mask treatment: dim the strict-upper-triangle cells, mute their
        // '∅' glyph, and drop a subtle diagonal hatch so "forbidden" reads at a glance.
        ".tq-hm-cell.is-masked{opacity:.32;filter:saturate(.4)}" +
        ".tq-hm-cell.is-masked .tq-hm-val{color:var(--ink-mute)}" +
        ".tq-hm-cell.is-masked.is-rowsel{opacity:.45}" +
        ".tq-sa-maskcell{min-width:200px}" +
        ".tq-sa-maskcallout{margin-top:2px}");
    },

    quiz: [
      {
        q: "In self-attention, where do Q, K, and V come from?",
        choices: [
          "Q from the current sequence, but K and V are loaded from a separate external memory bank",
          "All three are three different learned linear projections (E·Wq, E·Wk, E·Wv) of the SAME input sequence's embeddings",
          "Q and K are projections of the input; V is the raw embedding passed through unchanged",
          "They are three independent learned tables looked up by token id, like the embedding table"
        ],
        answer: 1,
        explain: "'Self' means Q, K, V are all linear projections of the same tokens' embeddings via distinct " +
                 "learned matrices Wq/Wk/Wv. (When Q comes from a different sequence than K/V, it's cross-attention.)"
      },
      {
        q: "Why divide the QKᵀ scores by √dk before the softmax?",
        choices: [
          "To make the scores sum to 1 so they form a probability distribution",
          "Because the dot product of two dk-dim vectors has variance ~dk (std ~√dk); without the fix, larger heads inflate scores, pushing softmax into a saturated near-one-hot regime with vanishing gradients",
          "√dk is the number of attention heads, so it averages their contributions",
          "To convert the dot product into a cosine similarity by normalizing the vector lengths"
        ],
        answer: 1,
        explain: "A dot product over dk terms has variance ~dk, so magnitudes grow like √dk purely from " +
                 "dimensionality. Dividing by √dk keeps score variance roughly constant, holding softmax in its " +
                 "smooth, high-gradient regime. Softmax itself (not the division) does the summing-to-1."
      },
      {
        q: "As you shrink the scaling divisor toward 0 in the lab, the selected row's peak attention weight climbs toward 1.0. What is happening?",
        choices: [
          "A bug — softmax weights should never change when you only rescale the inputs uniformly",
          "Dividing by a smaller number multiplies the score spread up; softmax's exponential then makes the top score dominate, collapsing the distribution toward one-hot (saturation)",
          "The Value vectors are being amplified, so the output grows",
          "The model is becoming more accurate because it's now confident about one token"
        ],
        answer: 1,
        explain: "Smaller divisor → larger effective spread → exp() amplifies the gap → distribution collapses " +
                 "toward one-hot. That saturated regime has near-zero gradients, which is exactly the failure " +
                 "mode √dk is designed to prevent."
      },
      {
        q: "A single output vector of self-attention is best described as:",
        choices: [
          "The Value of whichever token had the single highest score (a hard argmax pick)",
          "A convex combination (weighted average, weights summing to 1) of ALL tokens' Value vectors, weighted by the softmaxed scores",
          "The element-wise product of the Query and Key vectors",
          "The raw QKᵀ score row, before any normalization"
        ],
        answer: 1,
        explain: "weights·V is a weighted average of every token's Value, with non-negative weights that sum to " +
                 "exactly 1.0 (a convex combination). It can emphasize or ignore tokens but never produces signal " +
                 "outside the span of the Values."
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
