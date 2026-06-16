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

      // Interactive state for the "See it" lookup (does NOT touch bench/stepper):
      //   curT    = softmax temperature for the CANVAS only (1.0 = canonical)
      //   pinnedKey = which Key/Value row is dissected (-1 = none)
      //   hoverKey  = transient hover row (-1 = none)
      var curT = 1.0;
      var pinnedKey = -1;
      var hoverKey = -1;

      // For a query index i: raw scores against every key, then UNSCALED softmax.
      function scoresFor(i) {
        var s = [];
        for (var j = 0; j < n; j++) s.push(TQ.dot(Q[i], K[j]));
        return s;
      }
      function weightsFor(i) {
        return TQ.softmax(scoresFor(i)); // softmax on RAW scores (no /sqrt(dk))
      }
      // Tempered weights for the canvas: divide raw scores by T, THEN softmax.
      // T = 1 reproduces weightsFor(i) exactly, so the bench/stepper stay canonical.
      function tempWeightsFor(i, T) {
        var s = scoresFor(i);
        var ts = [];
        for (var j = 0; j < n; j++) ts.push(s[j] / T);
        return TQ.softmax(ts);
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

      /* ============================================================= *
       *  PRIMARY VISUAL — see Q/K/V: three role-vectors + the lookup
       *  they drive. This query selector also drives the bench + stepper.
       * ============================================================= */
      var querySel = TQ.segmented({
        options: tokens.map(function (t, i) { return { label: t, value: i }; }),
        value: 1, // default 'cat'
        onChange: function () { updateConcept(); updateBench(); }
      });

      var conceptBlock = TQ.block(
        TQ.h(2, "See it: three vectors, one lookup"),
        TQ.p(
          "Before any formula, here is the whole idea as a picture. Pick a ",
          TQ.el("strong", { text: "query token" }),
          " and watch two things: the three role-vectors it projects from its embedding, and the soft ",
          "lookup those vectors drive."
        )
      );
      conceptBlock.appendChild(TQ.el("div", { class: "tq-controls-row" },
        TQ.el("div", {},
          TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "Query token  i"),
          querySel.el
        )
      ));

      // legend: what each of the three vectors MEANS
      conceptBlock.appendChild(TQ.el("div", { class: "tq-qkv-legend" },
        TQ.el("span", { class: "tq-qkv-chip is-q" }, TQ.el("b", { text: "🔍 Query" }), " — what am I looking for?"),
        TQ.el("span", { class: "tq-qkv-chip is-k" }, TQ.el("b", { text: "🏷️ Key" }), " — what do I advertise?"),
        TQ.el("span", { class: "tq-qkv-chip is-v" }, TQ.el("b", { text: "📦 Value" }), " — what do I deliver if picked?")
      ));

      // Part A — one embedding projected into Q, K, V (all real vectors)
      conceptBlock.appendChild(TQ.el("div", { class: "tq-wa-rowhead",
        text: "1 · one token → three role-vectors   (Q, K, V = embedding × Wq, Wk, Wv)" }));
      var projRegion = TQ.el("div", { class: "tq-panel tq-wa-region tq-qkv-proj" });
      conceptBlock.appendChild(projRegion);

      // Part B — the soft lookup, drawn (Query scores Keys -> softmax weights pull Values -> output)
      conceptBlock.appendChild(TQ.el("div", { class: "tq-wa-rowhead",
        text: "2 · the soft lookup: the Query scores every Key, softmax weights pull the Values into one blended output" }));
      var lookup = TQ.canvasPanel(680, 320, drawLookup);
      conceptBlock.appendChild(TQ.el("div", { class: "tq-panel tq-wa-region tq-qkv-canvaswrap" }, lookup.el));
      conceptBlock.appendChild(TQ.note(
        "Edge thickness = magnitude. Left edges are raw Q·K scores (cool = negative, warm = positive); right " +
        "edges are softmax weights (brighter = more of that Value flows into the output). The strongest match is " +
        "outlined in amber. Change the query and the whole picture re-routes — that is content-dependent attention. " +
        "Click any Key/Value row to dissect WHY it got its score."
      ));

      /* --- (B) softmax TEMPERATURE knob — canvas-only; bench stays canonical --- */
      var tempSlider = TQ.slider({
        min: 0.25, max: 4, step: 0.05, value: 1,
        label: "softmax temperature  T  (rescales scores before softmax)",
        format: function (v) { return "T = " + TQ.fmt(v, 2); },
        onInput: function (v) {
          curT = v;
          lookup.redraw();
          updateDissect();    // keep the panel's "vs softmax weight" readout live
          updateTempStatus(); // keep the near-hard / soft / flattening pill in sync with T
        }
      });
      var tempStatus = TQ.el("span", { class: "tq-qkv-tempstatus" });
      var tempRow = TQ.el("div", { class: "tq-controls-row tq-qkv-temprow" },
        TQ.el("div", { style: { flex: "1 1 320px" } }, tempSlider.el),
        tempStatus
      );
      conceptBlock.appendChild(tempRow);
      conceptBlock.appendChild(TQ.note(
        "Low T sharpens the blend toward a near-hard argmax; high T flattens it toward uniform 1/n = " +
        TQ.fmt(1 / n, 3) + ". Temperature touches only the RIGHT-side (weight) edges — the left Q·K scores " +
        "and the dissection below are unchanged, because T rescales relevance into a blend, it doesn't change " +
        "the relevance. This is exactly the knob Level 3 turns with /√dk."
      ));

      /* --- (A) the dissection panel: score = Q_i · K_j, proven on screen --- */
      var dissectPanel = TQ.el("div", { class: "tq-panel tq-wa-region tq-qkv-dissect" });
      conceptBlock.appendChild(dissectPanel);

      root.appendChild(conceptBlock);

      // Elementwise-product strip + summed dot product for query i vs key j.
      // Renders nothing-but-a-placeholder until the learner pins a key.
      function updateDissect() {
        var i = querySel.get();
        var j = pinnedKey;
        dissectPanel.innerHTML = "";
        if (j < 0) {
          dissectPanel.classList.add("is-empty");
          dissectPanel.appendChild(TQ.el("div", { class: "tq-qkv-dissect-empty" },
            TQ.el("span", { class: "tq-qkv-dissect-empty-mark", text: "↑", "aria-hidden": "true" }),
            TQ.el("span", { text: "Click a Key / Value row in the diagram to see WHY it got that score — " +
              "its Q·K dot product, assembled term by term." })
          ));
          return;
        }
        dissectPanel.classList.remove("is-empty");

        var qi = Q[i], kj = K[j];
        // elementwise products q_d · k_d — the score is literally their sum.
        var prod = [];
        for (var d = 0; d < dk; d++) prod.push(qi[d] * kj[d]);
        var score = TQ.dot(qi, kj);          // = TQ.sum(prod), by definition
        var sumProd = TQ.sum(prod);
        // shared signed magnitude so positive cells read warm, negative cool.
        var pm = TQ.maxOf(prod.map(function (x) { return Math.abs(x); })) || 1;

        // tempered weight for THIS key, at the canvas's current temperature.
        var tw = tempWeightsFor(i, curT);
        var topIdx = argmaxIdx(tw);

        dissectPanel.appendChild(TQ.el("div", { class: "tq-qkv-dissect-head" },
          TQ.el("span", { class: "tq-qkv-dissect-title" },
            "Why this score?  ",
            TQ.el("b", { class: "tq-qkv-dissect-q", text: "Q · " + tokens[i] }),
            "  •  ",
            TQ.el("b", { class: "tq-qkv-dissect-k", text: "K · " + tokens[j] })
          ),
          TQ.badge(j === topIdx ? "top key at T = " + TQ.fmt(curT, 2) : "key #" + j,
                   j === topIdx ? "good" : "default")
        ));

        // row 1 / row 2: the two role-vectors, side by side conceptually (stacked).
        dissectPanel.appendChild(TQ.vectorView(qi, {
          label: "Q · " + tokens[i] + "   (what \"" + tokens[i] + "\" looks for)", cellSize: 18
        }));
        dissectPanel.appendChild(TQ.vectorView(kj, {
          label: "K · " + tokens[j] + "   (what \"" + tokens[j] + "\" advertises)", cellSize: 18
        }));
        // row 3: the signed elementwise product strip — warm cells add, cool cells subtract.
        dissectPanel.appendChild(TQ.el("div", { class: "tq-qkv-prodwrap" },
          TQ.vectorView(prod, {
            label: "q_d × k_d  (per dim — these are summed)",
            cellSize: 18, showValues: true, max: pm
          })
        ));

        // the equation line: Σ of the strip IS the score the canvas edge encodes.
        dissectPanel.appendChild(TQ.el("div", { class: "tq-qkv-eqline" },
          TQ.math("Σ q_d·k_d  =  Q·K  =  " + TQ.fmt(score, 3)),
          TQ.el("span", { class: "tq-qkv-eqcheck",
            text: "(sum of the strip = " + TQ.fmt(sumProd, 3) + " ✓)" })
        ));
        dissectPanel.appendChild(TQ.el("div", { class: "tq-qkv-eqkvs" },
          TQ.kv("dot product Q·K", TQ.fmt(score, 3)),
          TQ.kv("vs softmax weight", TQ.fmt(tw[j], 3)),
          TQ.kv("at temperature", "T = " + TQ.fmt(curT, 2))
        ));
        dissectPanel.appendChild(TQ.note(
          "The warm cells (positive q_d·k_d) push the score up; the cool cells (negative) pull it down. " +
          "Their sum is the single number on the canvas edge — that is all a \"relevance score\" ever is."
        ));
      }

      // keep the temperature status pill (near-hard vs flattening) in sync.
      function updateTempStatus() {
        var i = querySel.get();
        var tw = tempWeightsFor(i, curT);
        var peak = TQ.maxOf(tw);
        var uniform = 1 / n;
        tempStatus.innerHTML = "";
        if (peak >= 0.6) {
          tempStatus.appendChild(TQ.badge("near-hard pick · peak " + TQ.fmt(peak, 2), "warn"));
        } else if (peak <= uniform * 1.6) {
          tempStatus.appendChild(TQ.badge("flattening → uniform " + TQ.fmt(uniform, 3), "info"));
        } else {
          tempStatus.appendChild(TQ.badge("soft blend · peak " + TQ.fmt(peak, 2), "default"));
        }
      }

      // builder for a single role card in Part A
      function roleCard(tag, name, ask, vec, cls) {
        return TQ.el("div", { class: "tq-qkv-card " + cls },
          TQ.el("div", { class: "tq-qkv-cardhead" },
            TQ.el("span", { class: "tq-qkv-tag", text: tag }),
            TQ.el("span", { class: "tq-qkv-name", text: name })
          ),
          TQ.el("div", { class: "tq-qkv-ask", text: ask }),
          TQ.vectorView(vec, { cellSize: 16 })
        );
      }

      function updateConcept() {
        var i = querySel.get();
        // Part A: embedding -> Q, K, V
        projRegion.innerHTML = "";
        projRegion.appendChild(TQ.el("div", { class: "tq-qkv-card is-emb" },
          TQ.el("div", { class: "tq-qkv-cardhead" },
            TQ.el("span", { class: "tq-qkv-tag", text: "emb" }),
            TQ.el("span", { class: "tq-qkv-name", text: "“" + tokens[i] + "”" })
          ),
          TQ.el("div", { class: "tq-qkv-ask", text: "the token's " + TQ.toy.dModel + "-dim embedding (Level 1)" }),
          TQ.vectorView(TQ.toy.embeddings[i], { cellSize: 13 })
        ));
        projRegion.appendChild(TQ.el("div", { class: "tq-qkv-arrow",
          html: "× W<sub>q</sub><br>× W<sub>k</sub><br>× W<sub>v</sub><br>➜" }));
        projRegion.appendChild(TQ.el("div", { class: "tq-qkv-roles" },
          roleCard("Q", "🔍 Query", "what “" + tokens[i] + "” is looking for", Q[i], "is-q"),
          roleCard("K", "🏷️ Key", "what “" + tokens[i] + "” advertises", K[i], "is-k"),
          roleCard("V", "📦 Value", "what “" + tokens[i] + "” delivers if picked", V[i], "is-v")
        ));
        // Part B: redraw the lookup diagram + refresh the dissection / temp status
        // for the (possibly newly-selected) query, re-dissecting any pinned key.
        lookup.redraw();
        updateDissect();
        updateTempStatus();
      }

      // Geometry shared by drawLookup AND the canvas hit-test, so a click maps to
      // the same row rectangle the canvas drew. Returns {rowY,tnx,tW,rowH,top0,bot}.
      function lookupGeom(w, h) {
        var top0 = 56, bot = h - 16;
        var rowH = (bot - top0) / n;
        var tnx = Math.round(w * 0.47);
        var tW = 122;
        return { top0: top0, bot: bot, rowH: rowH, tnx: tnx, tW: tW };
      }
      // Which Key/Value row (if any) does a canvas-local (x,y) fall on? -1 = none.
      // We give the clickable band a generous height (full rowH) so the whole
      // lane — node + its score/weight numbers — is a target.
      function hitTestRow(x, y, w, h) {
        var g = lookupGeom(w, h);
        if (x < g.tnx - g.tW / 2 - 56 || x > g.tnx + g.tW / 2 + 64) return -1;
        for (var j = 0; j < n; j++) {
          var cy = g.top0 + g.rowH * j + g.rowH / 2;
          if (y >= cy - g.rowH / 2 && y <= cy + g.rowH / 2) return j;
        }
        return -1;
      }

      // canvas: the soft lookup for the currently selected query
      function drawLookup(ctx, w, h) {
        var i = querySel.get();
        var scores = scoresFor(i);
        // RIGHT-side edges use the TEMPERED weights (canvas-only). curT = 1 gives
        // the canonical weightsFor(i) exactly, so the default picture is unchanged.
        var weights = tempWeightsFor(i, curT);
        var maxAbs = 1, maxW = 1e-6, j;
        for (j = 0; j < n; j++) { maxAbs = Math.max(maxAbs, Math.abs(scores[j])); maxW = Math.max(maxW, weights[j]); }
        var top = argmaxIdx(weights);

        var ink     = TQ.cssVar("--ink", 1, "#e9edf8");
        var mute    = TQ.cssVar("--ink-mute", 1, "#6f7aa0");
        var soft    = TQ.cssVar("--ink-soft", 1, "#aab3cd");
        var line    = TQ.cssVar("--line", 1, "#25304f");
        var panel   = TQ.cssVar("--panel", 1, "#131a2c");
        var panelHi = TQ.cssVar("--panel-hi", 1, "#1d2747");
        var accent  = TQ.cssVar("--accent", 1, "#f2a93b");
        var good    = TQ.cssVar("--good", 1, "#46d39a");
        var mono    = TQ.cssVar("--mono", 1, "ui-monospace, monospace");
        var font    = TQ.cssVar("--font", 1, "system-ui, sans-serif");

        var top0 = 56, bot = h - 16;
        var rowH = (bot - top0) / n;
        function rowY(k) { return top0 + rowH * k + rowH / 2; }
        var qx = 86, tnx = Math.round(w * 0.47), ox = w - 92;
        var qW = 108, qH = 46, tW = 122, tH = Math.min(30, rowH - 10), oW = 104, oH = 48;
        var cy = (rowY(0) + rowY(n - 1)) / 2;

        function rr(x, y, ww, hh, r) {
          ctx.beginPath();
          ctx.moveTo(x + r, y);
          ctx.arcTo(x + ww, y, x + ww, y + hh, r);
          ctx.arcTo(x + ww, y + hh, x, y + hh, r);
          ctx.arcTo(x, y + hh, x, y, r);
          ctx.arcTo(x, y, x + ww, y, r);
          ctx.closePath();
        }

        // column mini-headers
        ctx.textBaseline = "middle";
        ctx.textAlign = "center";
        ctx.font = "10px " + mono;
        ctx.fillStyle = mute;
        ctx.fillText("Q·K  score", (qx + tnx) / 2, 20);
        ctx.fillText("KEYS · VALUES", tnx, 20);
        ctx.fillText("softmax weight", (tnx + ox) / 2, 20);

        // edges (drawn under the nodes)
        for (j = 0; j < n; j++) {
          var y = rowY(j);
          var sNorm = Math.abs(scores[j]) / maxAbs;
          var wNorm = weights[j] / maxW;
          // left: query -> key, width by |score|, signed color
          ctx.strokeStyle = TQ.colorForSigned(scores[j], maxAbs);
          ctx.globalAlpha = 0.28 + 0.55 * sNorm;
          ctx.lineWidth = 1 + 8 * sNorm;
          var x0 = qx + qW / 2, x1 = tnx - tW / 2, mx = (x0 + x1) / 2;
          ctx.beginPath();
          ctx.moveTo(x0, cy);
          ctx.bezierCurveTo(mx, cy, mx, y, x1, y);
          ctx.stroke();
          // right: key -> output, width by weight, brightness by weight
          ctx.strokeStyle = TQ.colorFor(0.18 + 0.82 * wNorm);
          ctx.globalAlpha = 0.30 + 0.6 * wNorm;
          ctx.lineWidth = 1 + 11 * wNorm;
          var x2 = tnx + tW / 2, x3 = ox - oW / 2, mx2 = (x2 + x3) / 2;
          ctx.beginPath();
          ctx.moveTo(x2, y);
          ctx.bezierCurveTo(mx2, y, mx2, cy, x3, cy);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }

        // token (Key/Value) nodes + per-row score & weight numbers
        for (j = 0; j < n; j++) {
          var yy = rowY(j);
          var isTop = j === top;
          var isPinned = j === pinnedKey;
          var isHover = j === hoverKey;
          // subtle full-lane highlight for the pinned/hovered row (drawn first).
          if (isPinned || isHover) {
            rr(tnx - tW / 2 - 52, yy - rowH / 2 + 2, tW + 116, rowH - 4, 8);
            ctx.fillStyle = TQ.cssVar("--accent", isPinned ? 0.1 : 0.05, "#f2a93b");
            ctx.fill();
          }
          rr(tnx - tW / 2, yy - tH / 2, tW, tH, 8);
          ctx.fillStyle = panel; ctx.fill();
          ctx.lineWidth = (isTop || isPinned) ? 2 : 1;
          ctx.strokeStyle = (isTop || isPinned) ? accent : line; ctx.stroke();
          // amber selection ring so canvas + dissection panel stay visually linked.
          if (isPinned) {
            rr(tnx - tW / 2 - 4, yy - tH / 2 - 4, tW + 8, tH + 8, 10);
            ctx.lineWidth = 2;
            ctx.strokeStyle = TQ.cssVar("--accent", 0.9, "#f2a93b");
            ctx.setLineDash([4, 3]);
            ctx.stroke();
            ctx.setLineDash([]);
          }
          ctx.fillStyle = ink;
          ctx.font = (isTop ? "bold " : "") + "13px " + font;
          ctx.textAlign = "center";
          ctx.fillText(tokens[j], tnx, yy);
          ctx.font = "11px " + mono;
          ctx.textAlign = "right"; ctx.fillStyle = soft;
          ctx.fillText(scores[j].toFixed(2), tnx - tW / 2 - 8, yy);
          ctx.textAlign = "left"; ctx.fillStyle = isTop ? accent : soft;
          ctx.fillText(weights[j].toFixed(3), tnx + tW / 2 + 8, yy);
        }

        // query node (left)
        rr(qx - qW / 2, cy - qH / 2, qW, qH, 10);
        ctx.fillStyle = panelHi; ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = accent; ctx.stroke();
        ctx.textAlign = "center";
        ctx.fillStyle = mute; ctx.font = "9px " + mono;
        ctx.fillText("QUERY", qx, cy - 10);
        ctx.fillStyle = ink; ctx.font = "bold 14px " + font;
        ctx.fillText(tokens[i], qx, cy + 9);

        // output node (right)
        rr(ox - oW / 2, cy - oH / 2, oW, oH, 10);
        ctx.fillStyle = panelHi; ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = good; ctx.stroke();
        ctx.fillStyle = mute; ctx.font = "9px " + mono;
        ctx.fillText("OUTPUT", ox, cy - 10);
        ctx.fillStyle = ink; ctx.font = "12px " + mono;
        ctx.fillText("Σ w·V", ox, cy + 9);
      }

      // --- canvas interactivity: click pins a Key/Value row, hover previews it ---
      // Map a pointer event to canvas-local logical (x,y) (the drawFn coord space).
      function canvasXY(ev) {
        var rect = lookup.canvas.getBoundingClientRect();
        // logical drawing width/height are 680 x 320 (see canvasPanel call).
        var sx = 680 / (rect.width || 680);
        var sy = 320 / (rect.height || 320);
        return { x: (ev.clientX - rect.left) * sx, y: (ev.clientY - rect.top) * sy };
      }
      lookup.canvas.style.cursor = "pointer";
      lookup.canvas.addEventListener("click", function (ev) {
        var p = canvasXY(ev);
        var j = hitTestRow(p.x, p.y, 680, 320);
        if (j < 0) return;
        pinnedKey = (pinnedKey === j) ? -1 : j; // click the pinned row again to clear
        lookup.redraw();
        updateDissect();
      });
      lookup.canvas.addEventListener("mousemove", function (ev) {
        var p = canvasXY(ev);
        var j = hitTestRow(p.x, p.y, 680, 320);
        if (j !== hoverKey) { hoverKey = j; lookup.redraw(); }
      });
      lookup.canvas.addEventListener("mouseleave", function () {
        if (hoverKey !== -1) { hoverKey = -1; lookup.redraw(); }
      });

      updateConcept();
      updateDissect();
      updateTempStatus();

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
          "The same lookup as the diagram above, now as exact bars. Using the ",
          TQ.el("strong", { text: "query token you picked above" }),
          ", row 1 is its raw ", TQ.math("Q·K"),
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

      // --- control: raw/softmax toggle (the QUERY is chosen in the visual above) ---
      var showWeights = TQ.toggle({
        label: "softmax weights (off = compare raw scores)",
        value: true,
        onChange: function () { updateBench(); }
      });

      var controls = TQ.el("div", { class: "tq-controls-row tq-wa-controls" },
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
        ".tq-wa-output{padding-top:2px}" +
        ".tq-qkv-legend{display:flex;gap:10px;flex-wrap:wrap;margin:2px 0 16px}" +
        ".tq-qkv-chip{font-size:12.5px;color:var(--ink-soft);background:var(--panel);" +
          "border:1px solid var(--line-soft);border-radius:999px;padding:5px 13px}" +
        ".tq-qkv-chip b{font-weight:700}" +
        ".tq-qkv-chip.is-q b{color:var(--accent)}" +
        ".tq-qkv-chip.is-k b{color:var(--cool)}" +
        ".tq-qkv-chip.is-v b{color:var(--good)}" +
        ".tq-qkv-proj{display:flex;align-items:center;gap:16px;flex-wrap:wrap}" +
        ".tq-qkv-arrow{font-family:var(--mono);font-size:12px;color:var(--ink-mute);" +
          "line-height:1.55;text-align:center;white-space:nowrap}" +
        // (B) temperature row
        ".tq-qkv-temprow{align-items:center;gap:16px;margin:14px 0 4px;flex-wrap:wrap}" +
        ".tq-qkv-tempstatus{display:flex;align-items:center}" +
        // (A) dissection panel
        ".tq-qkv-dissect{margin-top:12px;display:flex;flex-direction:column;gap:10px}" +
        ".tq-qkv-dissect.is-empty{align-items:center;justify-content:center;" +
          "min-height:96px;border-style:dashed}" +
        ".tq-qkv-dissect-empty{display:flex;align-items:center;gap:10px;" +
          "color:var(--ink-mute);font-size:13px;max-width:520px;text-align:left}" +
        ".tq-qkv-dissect-empty-mark{font-size:20px;color:var(--accent);line-height:1}" +
        ".tq-qkv-dissect-head{display:flex;align-items:center;justify-content:space-between;" +
          "gap:12px;flex-wrap:wrap;margin-bottom:2px}" +
        ".tq-qkv-dissect-title{font-size:13.5px;color:var(--ink)}" +
        ".tq-qkv-dissect-q{color:var(--accent)}" +
        ".tq-qkv-dissect-k{color:var(--cool)}" +
        ".tq-qkv-prodwrap{padding:8px 0 2px;border-top:1px solid var(--line-soft);" +
          "border-bottom:1px solid var(--line-soft)}" +
        ".tq-qkv-eqline{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:2px}" +
        ".tq-qkv-eqcheck{font-family:var(--mono);font-size:12px;color:var(--ink-mute)}" +
        ".tq-qkv-eqkvs{display:flex;gap:8px;flex-wrap:wrap;align-items:center}" +
        ".tq-qkv-roles{display:flex;gap:12px;flex-wrap:wrap}" +
        ".tq-qkv-card{background:var(--panel);border:1px solid var(--line-soft);" +
          "border-top:3px solid var(--line);border-radius:10px;padding:10px 12px}" +
        ".tq-qkv-card.is-emb{border-top-color:var(--ink-faint)}" +
        ".tq-qkv-card.is-q{border-top-color:var(--accent)}" +
        ".tq-qkv-card.is-k{border-top-color:var(--cool)}" +
        ".tq-qkv-card.is-v{border-top-color:var(--good)}" +
        ".tq-qkv-cardhead{display:flex;align-items:baseline;gap:8px;margin-bottom:3px}" +
        ".tq-qkv-tag{font-family:var(--mono);font-size:11px;font-weight:700;color:var(--ink-mute)}" +
        ".tq-qkv-name{font-size:13px;font-weight:700;color:var(--ink)}" +
        ".tq-qkv-ask{font-size:12px;color:var(--ink-soft);margin-bottom:9px;max-width:210px}" +
        ".tq-qkv-canvaswrap{display:flex;justify-content:center;overflow-x:auto}");
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
