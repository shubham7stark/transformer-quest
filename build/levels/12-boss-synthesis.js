/* ============================================================================
 * LEVEL 12 — Boss Level: The Whole Picture   (order 12)
 * ----------------------------------------------------------------------------
 * Synthesis level. No new math — it threads the SAME toy sentence
 * (TQ.toy) through every stage you've already built, then makes the
 * memory/quality trade-off numeric.
 *
 * Two PRIMARY interactions, all numbers genuinely computed via TQ math:
 *   1) A clickable 7-stage pipeline strip (Embed → Position → Q/K/V →
 *      Scaled Attention → Multi-Head → Block → KV Cache). Clicking a stage
 *      reveals a "what & why" card with a live number pulled from the toy:
 *        - Embed:    emb[1] ('cat') as a vectorView, d_model=16
 *        - Position: sinusoidalPE(6,16) row pos=1 as a vectorView
 *        - Q/K/V:    toyQKV(4,...) -> Q row for 'cat'
 *        - Scaled:   attention(Q,K,V) -> weights[1] as a barRow (sums to 1),
 *                    dk=4, 1/sqrt(4)=0.50, plus unscaled-vs-scaled peakiness
 *        - Multi:    2 heads, concat (6x8), ·Wo back to 16; concat[1]
 *        - Block:    project head output to 16, residual + LayerNorm
 *        - KV Cache: per-token-per-layer element formula
 *   2) A "spot the bottleneck" calculator: pick scheme (MHA/MQA/GQA/MLA),
 *      seq_len (log-ish steps), fp16/fp8 -> real KV-cache GB + savings.
 *
 * All color via TQ.colorFor / heatmap / vectorView / barRow (one language).
 * ========================================================================== */

(function () {
  "use strict";

  TQ.registerLevel({
    id: "boss-synthesis",
    order: 12,
    title: "Boss Level: The Whole Picture",
    icon: "🏁",
    tagline: "Connect the whole chain — and prove you can reason across it.",
    preread: "All four",
    objectives: [
      "Recap the full journey: tokens → attention → heads → position → block → KV cache → MHA/MQA/GQA/MLA → MoE → MoH",
      "Connect ideas across levels (why scaling, why the cache grows, why MLA wins, how MoE/MoH decouple capacity from compute)",
      "Pass a tougher mixed quiz and hit a celebratory completion state"
    ],

    render: function (root) {
      var toy = TQ.toy;
      var tokens = toy.tokens;
      var emb = toy.embeddings;
      var d = toy.dModel;          // 16
      var n = tokens.length;       // 6
      var CAT = 1;                 // index of "cat" — the token we trace

      /* ---------------------------------------------------- intro narrative */
      root.appendChild(TQ.block(
        TQ.h(2, "One sentence forward through the whole stack"),
        TQ.p(
          "You've built every piece; here's the chain in one breath. A token becomes a ",
          TQ.el("strong", { text: "vector" }), " (embedding). ",
          "Position gets stamped in (sinusoidal / RoPE) so order isn't lost. Each token projects to a ",
          TQ.el("strong", { text: "query, key, value" }), ". Attention scores every query against every key with a ",
          "dot product, divides by ", TQ.math("√dk"), " so the softmax doesn't saturate (collapse all its " +
          "weight onto one token), normalizes to weights, ",
          "and mixes the values — that's one head. Many heads run in parallel, each reading a different ",
          "relationship; their outputs concat and pass through ", TQ.math("Wo"), ". Wrap that in ",
          TQ.el("strong", { text: "residual + LayerNorm" }), ", add an FFN, stack the block N times, and predict ",
          "the next token."
        ),
        TQ.p(
          "The ", TQ.el("strong", { text: "KV cache" }), " is the one thing that survives between tokens at ",
          "inference — and it's the thing that bankrupts you at long context. ",
          TQ.el("strong", { text: "MHA / MQA / GQA / MLA" }), " are all answers to one question: ",
          TQ.el("em", { text: "how do we shrink that cache without lobotomizing the model?" }),
          " Click through the pipeline below — the same six-word toy sentence you've followed since Level 1 ",
          "runs through every stage, so the recap is concrete, not abstract."
        ),
        TQ.p(
          "And there's one more axis the last two levels opened up: ", TQ.el("strong", { text: "sparsity" }),
          ". ", TQ.el("strong", { text: "Mixture of Experts (MoE)" }), " replaces the dense FFN sub-layer with a ",
          "router + ", TQ.math("N"), " expert FFNs, firing only the ", TQ.math("top-k"),
          " per token — so ", TQ.el("em", { text: "total" }), " params (capacity) grow with ", TQ.math("N"),
          " while ", TQ.el("em", { text: "active" }), " params (per-token FLOPs) track ", TQ.math("k"), ". ",
          TQ.el("strong", { text: "Mixture of Heads (MoH)" }), " is that exact same trick applied to attention: a ",
          "router scores the heads per token, keeps the ", TQ.math("top-k"),
          ", and weights them — turning multi-head attention's silent equal-weight sum into a learned, sparse one. ",
          "Both decouple ", TQ.el("strong", { text: "what the model knows from what it spends per token" }), "."
        ),
        TQ.el("div", { class: "tq-token-strip" },
          tokens.map(function (tk, i) {
            return TQ.el("span", { class: "tq-token-chip" + (i === CAT ? " is-trace" : "") },
              TQ.el("span", { class: "tq-token-idx", text: "#" + i }),
              TQ.el("span", { class: "tq-token-word", text: tk })
            );
          })
        ),
        TQ.note("We trace token #1, \"cat\", through every stage so you watch one concrete vector move."),
        // Whole-map diagram: the full decoder-only skeleton at a glance — the
        // stacked xN loop, final norm and lm_head that the click-through strip
        // (which reveals stages one at a time and stops at KV cache) never shows
        // in one view. Static SVG; theme colors only via CSS vars / currentColor.
        TQ.figure(
          '<svg viewBox="0 0 720 130" width="720" height="130" role="img" ' +
            'aria-label="Whole decoder-only skeleton: tokens to embed plus position to N transformer blocks to final LayerNorm to lm_head to next-token probabilities" ' +
            'font-family="var(--mono)" font-size="11">' +
            '<defs><marker id="tq-l10-arrow" viewBox="0 0 10 10" refX="9" refY="5" ' +
              'markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
              '<path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>' +
            // stage boxes
            '<rect x="8" y="48" width="78" height="34" rx="7" fill="var(--panel-hi)" stroke="var(--line)"/>' +
            '<text x="47" y="69" text-anchor="middle" fill="var(--ink)">tokens</text>' +
            '<rect x="120" y="48" width="104" height="34" rx="7" fill="var(--panel-hi)" stroke="var(--line)"/>' +
            '<text x="172" y="64" text-anchor="middle" fill="var(--ink)">embed</text>' +
            '<text x="172" y="77" text-anchor="middle" fill="var(--ink-mute)" font-size="9">+ position</text>' +
            // stacked block (xN) — drawn as offset rects to suggest depth
            '<rect x="276" y="38" width="150" height="54" rx="9" fill="none" stroke="var(--line-soft)"/>' +
            '<rect x="270" y="32" width="150" height="54" rx="9" fill="none" stroke="var(--line-soft)"/>' +
            '<rect x="264" y="44" width="150" height="54" rx="9" fill="var(--panel-hi)" stroke="var(--accent)"/>' +
            '<text x="339" y="62" text-anchor="middle" fill="var(--ink)">transformer block</text>' +
            '<text x="339" y="75" text-anchor="middle" fill="var(--ink-mute)" font-size="9">attn + FFN, each residual+LN</text>' +
            '<text x="339" y="86" text-anchor="middle" fill="var(--info)" font-size="8">attn→MoH · FFN→MoE (sparse)</text>' +
            '<text x="339" y="26" text-anchor="middle" fill="var(--accent)" font-size="11">× N</text>' +
            '<rect x="450" y="48" width="92" height="34" rx="7" fill="var(--panel-hi)" stroke="var(--line)"/>' +
            '<text x="496" y="69" text-anchor="middle" fill="var(--ink)">final LN</text>' +
            '<rect x="568" y="48" width="78" height="34" rx="7" fill="var(--panel-hi)" stroke="var(--line)"/>' +
            '<text x="607" y="69" text-anchor="middle" fill="var(--ink)">lm_head</text>' +
            // final probabilities glyph
            '<g>' +
              '<rect x="676" y="46" width="10" height="14" rx="2" fill="var(--accent)"/>' +
              '<rect x="676" y="62" width="10" height="9" rx="2" fill="var(--cool)"/>' +
              '<rect x="676" y="73" width="10" height="6" rx="2" fill="var(--info)"/>' +
              '<text x="681" y="93" text-anchor="middle" fill="var(--ink-mute)" font-size="9">P(next)</text>' +
            '</g>' +
            // arrows
            '<g stroke="currentColor" stroke-width="1.5" fill="none" color="var(--ink-faint)">' +
              '<line x1="90" y1="65" x2="116" y2="65" marker-end="url(#tq-l10-arrow)"/>' +
              '<line x1="228" y1="65" x2="260" y2="65" marker-end="url(#tq-l10-arrow)"/>' +
              '<line x1="418" y1="65" x2="446" y2="65" marker-end="url(#tq-l10-arrow)"/>' +
              '<line x1="546" y1="65" x2="564" y2="65" marker-end="url(#tq-l10-arrow)"/>' +
              '<line x1="650" y1="65" x2="672" y2="65" marker-end="url(#tq-l10-arrow)"/>' +
            '</g>' +
          '</svg>',
          "The whole decoder-only skeleton in one view: tokens → embed (+ positional encoding) → a " +
          "transformer block (attention + FFN, each wrapped in residual + LayerNorm) stacked × N → a " +
          "final LayerNorm → lm_head, producing next-token probabilities. The clickable strip below walks " +
          "the stages one at a time; this map shows the stacked shape they live inside. Sparsity slots into " +
          "the block's two sub-layers: the FFN can become a Mixture of Experts (MoE), and attention a " +
          "Mixture of Heads (MoH) — both fire only top-k per token, so total params grow without growing per-token compute."
        )
      ));

      /* ============================================================ *
       *  PRIMARY 1 — the clickable pipeline strip
       * ============================================================ */

      // --- precompute everything once (all real TQ math) -----------------
      var dk = 4;

      // Stage 2: positional encoding
      var PE = TQ.sinusoidalPE(n, d);

      // Stage 3: Q/K/V (single head, documented seeds)
      var qkv = TQ.toyQKV(dk, 101, 202, 303);

      // Stage 4: scaled attention
      var att = TQ.attention(qkv.Q, qkv.K, qkv.V);
      var scaleFactor = 1 / Math.sqrt(att.dk);          // 0.50 for dk=4
      var weightsCat = att.weights[CAT];                 // valid prob dist (sums to 1)
      var unscaledWeightsCat = TQ.softmax(att.scores[CAT]); // softmax of UNscaled scores

      // Stage 5: multi-head — two heads, concat (6x8), ·Wo -> 6x16
      var head0 = TQ.toyQKV(dk, 101, 202, 303);
      var head1 = TQ.toyQKV(dk, 111, 212, 313);
      var o0 = TQ.attention(head0.Q, head0.K, head0.V).output; // 6x4
      var o1 = TQ.attention(head1.Q, head1.K, head1.V).output; // 6x4
      var concat = [];
      for (var rc = 0; rc < n; rc++) concat.push(o0[rc].concat(o1[rc])); // 6x8
      var Wo = TQ.randMatrix(2 * dk, d, 404);            // 8x16
      var multiOut = TQ.matmul(concat, Wo);              // 6x16

      // Stage 6: block — residual + post-LayerNorm on the trace token.
      // The head output dim (8 after concat·Wo it's 16) — use multiOut[1] which
      // is already d_model=16, so the residual add is dimension-clean.
      var blockX = emb[CAT];                             // 16-d embedding
      var blockSub = multiOut[CAT];                      // 16-d attention sublayer output
      var blockRes = [];
      for (var bi = 0; bi < d; bi++) blockRes.push(blockX[bi] + blockSub[bi]); // x + sublayer
      var blockNorm = TQ.layerNorm(blockRes);            // post-LN

      // helper for inline labelled stat chips
      function statChip(label, value) {
        return TQ.kv(label, value);
      }

      // each stage: {key, icon, title, oneLine, render(container)}
      var stages = [
        {
          key: "embed", icon: "🔠", title: "Embed",
          why: "Token → a learned d_model-vector. Meaning lives in this geometry.",
          run: function (c) {
            c.appendChild(TQ.p(
              TQ.el("strong", { text: "What & why: " }),
              "the token \"cat\" is looked up as a ", TQ.math("d_model = " + d),
              "-dimensional vector. Everything downstream operates on rows like this one."
            ));
            c.appendChild(TQ.vectorView(emb[CAT], { label: "emb[\"cat\"]", cellSize: 22 }));
            c.appendChild(TQ.el("div", { class: "tq-boss-stats" },
              statChip("d_model", String(d)),
              statChip("‖emb‖", TQ.fmt(TQ.norm(emb[CAT]), 2))
            ));
          }
        },
        {
          key: "pos", icon: "📍", title: "Position",
          why: "Order is stamped in so the model knows token 1 from token 5.",
          run: function (c) {
            c.appendChild(TQ.p(
              TQ.el("strong", { text: "What & why: " }),
              "attention itself is order-blind, so we add a positional signal. Sinusoidal PE for ",
              TQ.math("pos = " + CAT), " uses ", TQ.math("PE(pos,2i)=sin(pos/10000^(2i/d))"),
              " on even dims, ", TQ.math("cos(…)"), " on odd."
            ));
            c.appendChild(TQ.vectorView(PE[CAT], { label: "PE[pos=1]", cellSize: 22, max: 1 }));
            c.appendChild(TQ.note("The real input to the next stage is emb + PE; here we show the PE row " +
              "alone so the positional fingerprint is visible."));
          }
        },
        {
          key: "qkv", icon: "🎯", title: "Q / K / V",
          why: "Each token projects to a query, key, and value via Wq, Wk, Wv.",
          run: function (c) {
            c.appendChild(TQ.p(
              TQ.el("strong", { text: "What & why: " }),
              "three learned projections turn each embedding into a ", TQ.el("strong", { text: "query" }),
              " (what am I looking for?), a ", TQ.el("strong", { text: "key" }),
              " (what do I offer?), and a ", TQ.el("strong", { text: "value" }),
              " (what I'll pass on). Here ", TQ.math("dk = " + dk), "."
            ));
            c.appendChild(TQ.vectorView(qkv.Q[CAT], { label: "Q[\"cat\"]", cellSize: 26 }));
            c.appendChild(TQ.vectorView(qkv.K[CAT], { label: "K[\"cat\"]", cellSize: 26 }));
            c.appendChild(TQ.vectorView(qkv.V[CAT], { label: "V[\"cat\"]", cellSize: 26 }));
            c.appendChild(TQ.el("div", { class: "tq-boss-stats" },
              statChip("Q = emb·Wq", "16→" + dk),
              statChip("seeds", "101 / 202 / 303")
            ));
          }
        },
        {
          key: "scaled", icon: "⚖️", title: "Scaled Attention",
          why: "softmax(QKᵀ/√dk)·V — the 1/√dk keeps the softmax from saturating.",
          run: function (c) {
            c.appendChild(TQ.p(
              TQ.el("strong", { text: "What & why: " }),
              "score \"cat\"'s query against every key, divide by ", TQ.math("√dk"),
              ", softmax to weights, mix the values. The weights below are a valid probability ",
              "distribution — they sum to ", TQ.el("strong", { text: TQ.fmt(TQ.sum(weightsCat), 2) }), "."
            ));
            c.appendChild(TQ.barRow(weightsCat, {
              labels: tokens, max: 1,
              format: function (v) { return TQ.fmt(v, 3); }
            }));
            c.appendChild(TQ.el("div", { class: "tq-boss-stats" },
              statChip("dk", String(att.dk)),
              statChip("1/√dk", TQ.fmt(scaleFactor, 2)),
              statChip("Σ weights", TQ.fmt(TQ.sum(weightsCat), 2))
            ));
            c.appendChild(TQ.callout(
              "Why √dk exists: dot products of dk components have std ∝ √dk, so bigger dk means bigger raw " +
              "scores (logits). Without the divide, those scores get huge and the softmax spikes to near " +
              "one-hot (all the weight on a single token, the rest ~0) — so gradients vanish and learning " +
              "stalls. Compare the two distributions below: UNscaled (left) is peakier than scaled (right)."
            ));
            var cmp = TQ.el("div", { class: "tq-boss-cmp" },
              TQ.el("div", {},
                TQ.el("div", { class: "tq-slider-label", text: "UNscaled  softmax(QKᵀ)" }),
                TQ.barRow(unscaledWeightsCat, { labels: tokens, max: 1, format: function (v) { return TQ.fmt(v, 3); } })
              ),
              TQ.el("div", {},
                TQ.el("div", { class: "tq-slider-label", text: "Scaled  softmax(QKᵀ/√dk)" }),
                TQ.barRow(weightsCat, { labels: tokens, max: 1, format: function (v) { return TQ.fmt(v, 3); } })
              )
            );
            c.appendChild(cmp);
          }
        },
        {
          key: "multi", icon: "🧠", title: "Multi-Head",
          why: "Several heads read different relationships in parallel, then concat·Wo.",
          run: function (c) {
            c.appendChild(TQ.p(
              TQ.el("strong", { text: "What & why: " }),
              "run two independent heads (distinct seeds), giving each token a ", TQ.math("2·dk = " + (2 * dk)),
              "-dim concatenated output, then project back to ", TQ.math("d_model = " + d),
              " with ", TQ.math("Wo"), ". More query heads = more relationships modeled."
            ));
            c.appendChild(TQ.vectorView(concat[CAT], { label: "concat[\"cat\"] (6×8 row)", cellSize: 24 }));
            c.appendChild(TQ.vectorView(multiOut[CAT], { label: "·Wo → d_model=16", cellSize: 22 }));
            c.appendChild(TQ.el("div", { class: "tq-boss-stats" },
              statChip("heads", "2"),
              statChip("concat", n + "×" + (2 * dk)),
              statChip("after Wo", n + "×" + d)
            ));
          }
        },
        {
          key: "block", icon: "🧱", title: "Block",
          why: "residual (x + sublayer) then LayerNorm — repeated N times.",
          run: function (c) {
            c.appendChild(TQ.p(
              TQ.el("strong", { text: "What & why: " }),
              "the sublayer output is added back to the input (the ", TQ.el("strong", { text: "residual" }),
              ", ", TQ.math("x + sublayer"), " — keeping the original ", TQ.math("x"),
              " gives gradients a clean shortcut so deep stacks still train), then normalized. Order is ",
              TQ.el("strong", { text: "residual-THEN-LayerNorm" }),
              " (\"post-LN\" means the norm comes after the residual add, as in the Illustrated Transformer). ",
              "We use the multi-head output for \"cat\" (already ", TQ.math("d_model=16"),
              ") so the residual add is dimension-clean."
            ));
            c.appendChild(TQ.vectorView(blockX, { label: "x (embedding)", cellSize: 20 }));
            c.appendChild(TQ.vectorView(blockSub, { label: "sublayer out", cellSize: 20 }));
            c.appendChild(TQ.vectorView(blockRes, { label: "x + sublayer (residual)", cellSize: 20 }));
            c.appendChild(TQ.vectorView(blockNorm, { label: "LayerNorm(residual)", cellSize: 20 }));
            c.appendChild(TQ.el("div", { class: "tq-boss-stats" },
              statChip("mean(LN)", TQ.fmt(TQ.mean(blockNorm), 2)),
              statChip("‖LN‖", TQ.fmt(TQ.norm(blockNorm), 2))
            ));
            c.appendChild(TQ.note("LayerNorm pulls the vector to zero mean / unit variance — note mean(LN) ≈ 0.00. " +
              "An FFN follows (the per-token compute), then the whole block stacks N times."));
          }
        },
        {
          key: "cache", icon: "💾", title: "KV Cache",
          why: "Keys & values for every past token are stored — this is what grows.",
          run: function (c) {
            var perTok = 2 * 1 * dk; // for ONE head in the toy: 2·n_kv·head_dim
            c.appendChild(TQ.p(
              TQ.el("strong", { text: "What & why: " }),
              "at inference each new token reuses the K and V of every prior token, so we cache them. ",
              "Per token per layer the cache holds ", TQ.math("2 · n_kv_heads · head_dim"),
              " numbers. For our single toy head (", TQ.math("head_dim = " + dk),
              "): ", TQ.el("strong", { text: "2 · 1 · " + dk + " = " + perTok }), " elements per token."
            ));
            c.appendChild(TQ.vectorView(qkv.K[CAT], { label: "cached K[\"cat\"]", cellSize: 24 }));
            c.appendChild(TQ.vectorView(qkv.V[CAT], { label: "cached V[\"cat\"]", cellSize: 24 }));
            c.appendChild(TQ.callout(
              "This cache grows linearly with sequence length — every past token must be kept. That linear " +
              "wall is exactly why long context is a memory problem. Scroll down to size it for a real model."
            ));
          }
        }
      ];

      // build the strip + the detail panel
      var pipeBlock = TQ.block(
        TQ.h(2, "The pipeline, click-by-click"),
        TQ.p("Each stage below is a button. Click one to see its ", TQ.el("strong", { text: "what & why" }),
          " plus a live number computed from the toy sentence. The arrows show the flow of \"cat\" forward.")
      );

      var detail = TQ.el("div", { class: "tq-panel tq-grow tq-boss-detail" });
      var stripBtns = [];

      function selectStage(idx) {
        for (var i = 0; i < stripBtns.length; i++) {
          stripBtns[i].classList.toggle("is-active", i === idx);
          stripBtns[i].setAttribute("aria-selected", i === idx ? "true" : "false");
        }
        detail.innerHTML = "";
        detail.appendChild(TQ.el("div", { class: "tq-boss-detail-head" },
          TQ.el("span", { class: "tq-boss-detail-icon", text: stages[idx].icon }),
          TQ.el("span", { class: "tq-boss-detail-title", text: (idx + 1) + ". " + stages[idx].title }),
          TQ.badge("stage " + (idx + 1) + " / " + stages.length, "info")
        ));
        stages[idx].run(detail);
      }

      var strip = TQ.el("div", { class: "tq-boss-strip", role: "tablist" });
      stages.forEach(function (st, idx) {
        var btn = TQ.el("button", {
          class: "tq-boss-stage", type: "button", role: "tab",
          "aria-selected": "false", title: st.why
        },
          TQ.el("span", { class: "tq-boss-stage-icon", text: st.icon }),
          TQ.el("span", { class: "tq-boss-stage-label", text: st.title })
        );
        btn.addEventListener("click", function () { selectStage(idx); });
        btn.addEventListener("keydown", function (e) {
          if (e.key === "ArrowRight") { e.preventDefault(); var nx = (idx + 1) % stages.length; selectStage(nx); stripBtns[nx].focus(); }
          if (e.key === "ArrowLeft") { e.preventDefault(); var pv = (idx - 1 + stages.length) % stages.length; selectStage(pv); stripBtns[pv].focus(); }
        });
        stripBtns.push(btn);
        strip.appendChild(btn);
        if (idx < stages.length - 1) {
          strip.appendChild(TQ.el("span", { class: "tq-boss-arrow", text: "→", "aria-hidden": "true" }));
        }
      });

      pipeBlock.appendChild(strip);
      pipeBlock.appendChild(detail);
      selectStage(0);

      // The same stages as runnable-style PyTorch pseudocode, one-to-one with the
      // strip above. Each comment names the level it came from so the skeleton
      // literally ties the whole course together. Kept a skeleton (not training
      // code) — the def blocks are what every stage you clicked adds up to.
      pipeBlock.appendChild(TQ.code(
        "import torch\n" +
        "import torch.nn as nn\n" +
        "import torch.nn.functional as F\n" +
        "\n" +
        "def block_forward(x, attn, ffn, ln1, ln2, kv_cache=None):\n" +
        "    # one transformer block = residual + LayerNorm around (a) attention, (b) FFN\n" +
        "    # attn does softmax(Q @ K.T / sqrt(dk)) @ V  (L3 scaling, L4 multi-head).\n" +
        "    # MHA/MQA/GQA/MLA only change HOW K and V are stored/shared (L8/L9);\n" +
        "    # kv_cache holds every past token's K and V so we don't recompute them (L7).\n" +
        "    x = x + attn(ln1(x), kv_cache=kv_cache)   # causal self-attention sublayer\n" +
        "    x = x + ffn(ln2(x))                       # position-wise feed-forward sublayer\n" +
        "    return x\n" +
        "\n" +
        "def model_forward(token_ids, embed, pos_enc, blocks, ln_f, lm_head, kv=None):\n" +
        "    x = embed(token_ids)                      # token -> vector lookup (L1)\n" +
        "    x = pos_enc(x)                            # stamp in order: sinusoidal / RoPE (L5)\n" +
        "    for i, block in enumerate(blocks):        # stack N transformer blocks\n" +
        "        x = block_forward(x, *block, kv_cache=(kv[i] if kv else None))\n" +
        "    x = ln_f(x)                              # final LayerNorm (L6)\n" +
        "    logits = lm_head(x)                      # project to vocab: (seq, vocab)\n" +
        "    return logits                            # softmax(logits[-1]) = next-token probs\n",
        { lang: "python", label: "decoder-only forward",
          caption: "The whole course as one forward pass: embed + position -> N blocks (attention + FFN, " +
            "each residual + LayerNorm) -> final LayerNorm -> lm_head. Each comment names the level it came from. " +
            "Note this skeleton norms BEFORE each sublayer (x + sublayer(ln(x))) — that's \"pre-LN\", the modern " +
            "GPT/nanoGPT default, which is why a final ln_f is needed; the Block stage above shows the classic " +
            "\"post-LN\" (norm after the residual add) from the original paper. Both are valid; production decoders use pre-LN." }
      ));

      root.appendChild(pipeBlock);

      /* ========================================================== *
       *  Forces framing
       * ========================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "The three forces that decide everything downstream"),
        TQ.p(
          "Almost every design choice is a fight between three quantities. ",
          TQ.el("strong", { text: "(1) Quality" }), " — how many independent relationships attention can model, ",
          "which scales with the number of query heads. ",
          TQ.el("strong", { text: "(2) Memory at inference" }), " — dominated by the KV cache, which grows as ",
          TQ.math("2 · n_layers · seq_len · n_kv_heads · head_dim · bytes"), ". ",
          TQ.el("strong", { text: "(3) The √dk + RoPE guardrails" }), " — which keep the math numerically and ",
          "positionally sane regardless of size."
        ),
        TQ.p(
          TQ.el("strong", { text: "MQA / GQA" }), " attack memory by cutting ", TQ.math("n_kv_heads"),
          " (saving ", TQ.math("1 − n_kv/n_q"), ") while keeping all query heads for quality. ",
          TQ.el("strong", { text: "MLA" }), " goes further: it stops caching K and V directly and instead caches a ",
          "small latent ", TQ.math("c"), " — a compressed summary the keys and values are rebuilt from — ",
          "paying for it with a separate, ",
          "decoupled RoPE key (a tiny extra key that carries position), because you can't bake rotary position ",
          "into a compressed latent and still rebuild clean per-head keys."
        ),
        TQ.p(
          TQ.el("strong", { text: "MoE and MoH add a fourth axis: sparsity." }),
          " The first three forces are about a ", TQ.el("em", { text: "dense" }),
          " model where every parameter fires for every token. Sparsity breaks that tie. ",
          TQ.el("strong", { text: "MoE" }), " swaps the dense FFN for ", TQ.math("N"),
          " experts and a router, running only ", TQ.math("k"), " per token: ",
          TQ.el("strong", { text: "total params" }), " (capacity) scale with ", TQ.math("N"), ", but ",
          TQ.el("strong", { text: "active params" }), " (the per-token FLOPs) only with ", TQ.math("k"),
          " — so a model can ", TQ.el("em", { text: "know" }),
          " far more than it ", TQ.el("em", { text: "computes" }), " (Mixtral fires 2 of 8; DeepSeek-V3, 8 of 256). ",
          TQ.el("strong", { text: "MoH" }), " is literally MoE applied to attention heads: ",
          "standard multi-head attention secretly sums all ", TQ.math("H"),
          " heads with equal weight 1; MoH routes per token, keeps the ", TQ.math("top-k"),
          " heads, and ", TQ.el("strong", { text: "weights" }), " them — so total heads ", TQ.math("H"),
          " can grow while active compute stays at ", TQ.math("k"), ". Same decoupling, applied to the other sub-layer."
        )
      ));

      /* ========================================================== *
       *  PRIMARY 2 — "spot the bottleneck" calculator
       * ========================================================== */

      var calcBlock = TQ.block(
        TQ.h(2, "Spot the bottleneck: size the KV cache"),
        TQ.p("Pick an attention scheme, a context length, and a precision. The real KV-cache size and the ",
          "savings-vs-MHA recompute live. Config below: a 32-layer, 32-query-head model (head_dim 128), with ",
          "MLA's latent sizes (d_c 512, d_rope 64) from the DeepSeek-V2 ballpark used in the pre-reads.")
      );

      // fixed model config
      var N_LAYERS = 32, HEAD_DIM = 128, N_Q = 32;
      var D_C = 512, D_ROPE = 64;          // MLA latent + decoupled RoPE key (DeepSeek-V2)
      var N_KV = { MHA: 32, GQA: 8, MQA: 1 };

      var scheme = "MHA";
      var seqLen = 8192;
      var bytes = 2; // fp16

      function cacheBytes(sch, seq, by) {
        if (sch === "MLA") {
          // ONE latent replaces BOTH K and V -> no factor of 2
          return N_LAYERS * seq * (D_C + D_ROPE) * by;
        }
        return 2 * N_LAYERS * seq * N_KV[sch] * HEAD_DIM * by;
      }
      function fmtGB(b) { return TQ.fmt(b / 1e9, 2) + " GB"; }

      var schemeSeg = TQ.segmented({
        options: [
          { label: "MHA", value: "MHA" },
          { label: "GQA (8)", value: "GQA" },
          { label: "MQA", value: "MQA" },
          { label: "MLA", value: "MLA" }
        ],
        value: "MHA",
        onChange: function (v) { scheme = v; updateCalc(); }
      });

      // log-ish discrete seq lengths via a slider over indices
      var SEQS = [512, 2048, 8192, 32768, 131072];
      var seqSlider = TQ.slider({
        min: 0, max: SEQS.length - 1, step: 1, value: 2,
        label: "Context length (seq_len)",
        format: function (i) { return SEQS[i].toLocaleString() + " tok"; },
        onInput: function (i) { seqLen = SEQS[Math.round(i)]; updateCalc(); }
      });

      var precToggle = TQ.toggle({
        label: "fp8 (bytes = 1) instead of fp16 (bytes = 2)",
        value: false,
        onChange: function (on) { bytes = on ? 1 : 2; updateCalc(); }
      });

      var calcOut = TQ.el("div", { class: "tq-panel" });

      function updateCalc() {
        calcOut.innerHTML = "";

        var mha = cacheBytes("MHA", seqLen, bytes);
        var cur = cacheBytes(scheme, seqLen, bytes);
        var savings = mha > 0 ? (1 - cur / mha) : 0;

        // headline stats
        var stats = TQ.el("div", { class: "tq-boss-calc-stats" },
          TQ.el("div", { class: "tq-boss-calc-stat is-primary" },
            TQ.el("span", { class: "tq-stat-cap", text: scheme + " KV cache" }),
            TQ.el("span", { class: "tq-stat-big", text: fmtGB(cur) }),
            TQ.el("span", { class: "tq-boss-calc-sub", text: "at " + seqLen.toLocaleString() + " tok, fp" + (bytes === 2 ? "16" : "8") })
          ),
          TQ.el("div", { class: "tq-boss-calc-stat" },
            TQ.el("span", { class: "tq-stat-cap", text: "vs MHA baseline" }),
            TQ.el("span", { class: "tq-stat-big", text: fmtGB(mha) })
          ),
          TQ.el("div", { class: "tq-boss-calc-stat is-good" },
            TQ.el("span", { class: "tq-stat-cap", text: "memory saved" }),
            TQ.el("span", { class: "tq-stat-big", text: (savings > 0 ? TQ.fmt(savings * 100, 1) + "%" : "0%") })
          )
        );
        calcOut.appendChild(stats);

        // comparison bars across all schemes at the current seq/bytes
        var order = ["MHA", "GQA", "MQA", "MLA"];
        var vals = order.map(function (s) { return cacheBytes(s, seqLen, bytes) / 1e9; });
        var maxV = TQ.maxOf(vals);
        calcOut.appendChild(TQ.el("div", { class: "tq-slider-label", style: { margin: "10px 0 4px" }, text: "KV cache by scheme (GB)" }));
        calcOut.appendChild(TQ.barRow(vals, {
          labels: order, max: maxV,
          format: function (v) { return TQ.fmt(v, 2) + " GB"; },
          colorFn: function (v, i) { return order[i] === scheme ? TQ.colorFor(0.85) : TQ.colorFor(0.45); },
          highlight: order.indexOf(scheme)
        }));

        // per-token-per-layer element accounting + scheme-specific note
        var perTok, formula;
        if (scheme === "MLA") {
          perTok = D_C + D_ROPE;
          var mhaPerTok = 2 * N_Q * HEAD_DIM; // 2·32·128 = 8192 in this config
          formula = TQ.callout(
            "MLA caches a single latent: " + TQ.fmt(D_C, 0) + " + " + TQ.fmt(D_ROPE, 0) +
            " = " + perTok + " numbers/token/layer (the +" + D_ROPE + " is the decoupled RoPE key that carries " +
            "position), replacing MHA's 2·" + N_Q + "·" + HEAD_DIM + " = " + mhaPerTok.toLocaleString() +
            " — a ~" + TQ.fmt(mhaPerTok / perTok, 0) + "× shrink, while KEEPING all " + N_Q + " query heads."
          );
        } else {
          perTok = 2 * N_KV[scheme] * HEAD_DIM;
          var save = 1 - N_KV[scheme] / N_Q;
          formula = TQ.note(
            scheme + ": n_kv_heads = " + N_KV[scheme] + " of " + N_Q + " query heads → " +
            "2·" + N_KV[scheme] + "·" + HEAD_DIM + " = " + perTok.toLocaleString() +
            " elements/token/layer. Head-sharing saving vs MHA = 1 − " + N_KV[scheme] + "/" + N_Q +
            " = " + TQ.fmt(save * 100, 1) + "%."
          );
        }
        calcOut.appendChild(TQ.el("div", { class: "tq-boss-stats", style: { marginTop: "10px" } },
          TQ.kv("n_layers", String(N_LAYERS)),
          TQ.kv("head_dim", String(HEAD_DIM)),
          TQ.kv("n_q_heads", String(N_Q)),
          TQ.kv("bytes", String(bytes)),
          TQ.kv("per tok/layer", perTok.toLocaleString())
        ));
        calcOut.appendChild(formula);
      }

      calcBlock.appendChild(TQ.el("div", { class: "tq-boss-calc-controls" },
        TQ.el("div", { class: "tq-boss-calc-ctl" },
          TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" }, text: "Attention scheme" }),
          schemeSeg.el
        ),
        TQ.el("div", { class: "tq-boss-calc-ctl tq-grow" }, seqSlider.el),
        TQ.el("div", { class: "tq-boss-calc-ctl" }, precToggle.el)
      ));
      calcBlock.appendChild(calcOut);
      calcBlock.appendChild(TQ.callout(
        "Three things to feel here: (1) the cache grows STRICTLY linearly with seq_len — doubling context " +
        "doubles memory for every scheme, no exceptions; that linear wall is unavoidable. (2) GQA(8) is " +
        "exactly 75% smaller than MHA; MQA ~97%. (3) MLA sits far below MHA and below GQA(8), and — unlike " +
        "MQA — it KEEPS all query heads. At this config MLA (0.30 GB) is a bit larger than MQA (0.13 GB), " +
        "but MQA pays for that with a real quality hit; MLA's win is matching MHA-level quality at a " +
        "fraction of MHA's cache — that's the 'why MLA wins' moment, made numeric."
      ));
      updateCalc();
      root.appendChild(calcBlock);

      /* ----------------------------------------------------- final takeaway */
      root.appendChild(TQ.block(
        TQ.h(2, "You've connected the whole chain"),
        TQ.p(
          "Tokens became vectors. Position got stamped in. Q/K/V let tokens query each other; ", TQ.math("√dk"),
          " kept the softmax sane; many heads read many relationships; residual + LayerNorm + FFN made a block; ",
          "stacking blocks made a model. The KV cache — the one thing that survives between tokens — is the ",
          "wall that MHA, MQA, GQA, and MLA all fight in different ways. And once the dense block was built, ",
          TQ.el("strong", { text: "sparsity" }), " let it grow: ", TQ.el("strong", { text: "MoE" }),
          " makes the FFN a router + many experts (top-k fire), and ", TQ.el("strong", { text: "MoH" }),
          " does the same to attention heads — both letting total params (capacity) outrun active params (per-token compute)."
        ),
        TQ.callout("The boss takeaway: seq_len is the linear term nobody can delete. Every clever scheme only " +
          "shrinks the PER-TOKEN constant. MLA shrinks it the most while paying the least quality — that's why " +
          "it's the clever one. Now go ship something memory-efficient.")
      ));

      /* ----------------------------------------------- go deeper (resources) */
      root.appendChild(TQ.resources("Go deeper — build the whole thing yourself", [
        {
          label: "Andrej Karpathy — nanoGPT",
          url: "https://github.com/karpathy/nanoGPT",
          kind: "code",
          note: "A tiny, readable decoder-only GPT in PyTorch — the real version of the model_forward skeleton above."
        },
        {
          label: "Andrej Karpathy — Let's build GPT from scratch",
          url: "https://www.youtube.com/watch?v=kCc8FmEb1nY",
          kind: "video",
          note: "Builds the entire forward pass live, from embedding to lm_head — every stage in this level, in order."
        },
        {
          label: "The Illustrated Transformer",
          url: "https://jalammar.github.io/illustrated-transformer/",
          kind: "blog",
          note: "Jay Alammar's visual walkthrough of the whole stack — the picture this recap puts numbers on."
        },
        {
          label: "Attention Is All You Need",
          url: "https://arxiv.org/abs/1706.03762",
          kind: "paper",
          note: "The original paper that introduced scaled dot-product attention and the transformer block."
        }
      ]));

      /* scoped styles — colors only via CSS variables / colorFor, no hardcoded hex */
      injectOnce("tq-lvl10-css",
        ".tq-token-strip{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 14px}" +
        ".tq-token-chip{display:inline-flex;flex-direction:column;align-items:center;gap:2px;" +
          "background:var(--panel-hi);border:1px solid var(--line);border-radius:10px;padding:8px 14px}" +
        ".tq-token-chip.is-trace{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}" +
        ".tq-token-idx{font-size:10px;color:var(--ink-mute);font-family:var(--mono)}" +
        ".tq-token-word{font-size:16px;font-weight:700}" +
        ".tq-boss-strip{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:6px 0 16px}" +
        ".tq-boss-stage{display:inline-flex;flex-direction:column;align-items:center;gap:4px;" +
          "background:var(--panel-hi);border:1px solid var(--line);border-radius:12px;padding:10px 12px;" +
          "cursor:pointer;color:var(--ink);min-width:84px;transition:transform .12s,border-color .12s}" +
        ".tq-boss-stage:hover{transform:translateY(-2px);border-color:var(--accent)}" +
        ".tq-boss-stage.is-active{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent);background:var(--panel)}" +
        ".tq-boss-stage-icon{font-size:20px;line-height:1}" +
        ".tq-boss-stage-label{font-size:11px;font-weight:600;text-align:center}" +
        ".tq-boss-arrow{color:var(--ink-mute);font-size:16px;flex:0 0 auto}" +
        ".tq-boss-detail{display:flex;flex-direction:column;gap:12px;min-height:120px}" +
        ".tq-boss-detail-head{display:flex;align-items:center;gap:10px}" +
        ".tq-boss-detail-icon{font-size:22px}" +
        ".tq-boss-detail-title{font-size:17px;font-weight:700}" +
        ".tq-boss-stats{display:flex;gap:8px;flex-wrap:wrap}" +
        ".tq-boss-cmp{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;margin-top:6px}" +
        ".tq-boss-calc-controls{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-end;margin-bottom:14px}" +
        ".tq-boss-calc-ctl{display:flex;flex-direction:column}" +
        ".tq-boss-calc-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}" +
        ".tq-boss-calc-stat{display:flex;flex-direction:column;gap:2px;background:var(--panel-hi);" +
          "border:1px solid var(--line);border-radius:10px;padding:10px 14px}" +
        ".tq-boss-calc-stat.is-primary{border-color:var(--accent)}" +
        ".tq-boss-calc-stat.is-good{border-color:var(--good,var(--accent))}" +
        ".tq-boss-calc-sub{font-size:11px;color:var(--ink-mute)}");
    },

    quiz: [
      {
        q: "You're serving a model at 128k context and KV-cache memory is the wall. Your eng wants to switch MHA→MQA; your researcher warns about quality loss. Which statement is the most accurate trade-off?",
        choices: [
          "MQA and MHA use identical memory; the only difference is speed",
          "MQA collapses all key/value heads to one (≈97% cache cut with n_q=32) but every query head now reads the same K/V, reducing attention's representational resolution — GQA or MLA are the middle/better grounds",
          "MQA increases the KV cache because it adds a latent vector",
          "Switching to MQA changes the FFN, not the attention cache"
        ],
        answer: 1,
        explain: "KV cache = 2·n_layers·seq·n_kv_heads·head_dim·bytes, so n_kv_heads=1 saves 1−1/32≈97%. But all " +
                 "query heads share one K/V, so you lose head diversity — GQA keeps a few KV groups, MLA keeps " +
                 "full query heads via a cached latent."
      },
      {
        q: "What specifically breaks if you drop the 1/√dk scaling in softmax(QKᵀ/√dk)V as dk grows?",
        choices: [
          "Nothing — it's a cosmetic constant",
          "The dot-product scores' magnitude grows with dk, pushing softmax toward a near one-hot distribution, so gradients through it vanish and training/attention destabilizes",
          "The output dimension changes, breaking the residual add",
          "Position information is lost"
        ],
        answer: 1,
        explain: "Scores are sums of ~dk products; their std scales like √dk. Without dividing by √dk, large dk " +
                 "makes logits huge, softmax saturates to nearly one-hot, gradients flatten, and learning stalls. " +
                 "The √dk keeps logit variance ~constant."
      },
      {
        q: "MLA caches a small latent c per token and reconstructs K and V from it. Why does it ALSO need a separate decoupled RoPE key?",
        choices: [
          "To store the value vector, which the latent can't represent",
          "Because RoPE applies a position-dependent rotation to keys that can't be folded cleanly into the low-rank up-projection, so position is carried by a small dedicated key alongside the compressed content key",
          "To double the cache size for redundancy",
          "Because softmax requires two key vectors per token"
        ],
        answer: 1,
        explain: "RoPE rotates keys by absolute position; that rotation can't be absorbed into the static low-rank " +
                 "reconstruction of K without entangling the up-projection. DeepSeek decouples it: the content key " +
                 "comes from the latent, a small extra RoPE key carries position."
      },
      {
        q: "Across the stack, which quantity in the KV-cache formula is the one you CANNOT reduce by architectural tricks — the reason long context is fundamentally a memory problem?",
        choices: [
          "n_kv_heads — fixed by the vocab",
          "head_dim — fixed by the tokenizer",
          "seq_len — it scales linearly with context length and the cache must hold every past token's K/V; MHA/MQA/GQA/MLA only shrink the per-token cost, not the linear growth",
          "bytes — quantization can't change it"
        ],
        answer: 2,
        explain: "Cache grows linearly in seq_len because every prior token's K/V must be retained for " +
                 "autoregressive decoding. Head-sharing (MQA/GQA) and latent compression (MLA) shrink the " +
                 "per-token constant; none remove the linear-in-context growth."
      },
      {
        q: "A Mixture-of-Experts layer has N = 8 expert FFNs and routes top-k = 2 per token (no shared expert). Compared to a single dense FFN of the same per-expert size, what is true about its parameters and per-token compute?",
        choices: [
          "Both total params and per-token compute grow ~8× — MoE is strictly more expensive everywhere",
          "Total params grow ~8× (all N experts live in memory = capacity), but per-token compute grows only ~2× (only the top-k = 2 experts run) — capacity is decoupled from FLOPs",
          "Per-token compute grows ~8× but total params stay the same as one dense FFN",
          "MoE shrinks total params because experts share weights; compute is unchanged"
        ],
        answer: 1,
        explain: "MoE's whole point is decoupling TOTAL params (all N experts sit in memory ≈ N× the capacity) " +
                 "from ACTIVE params / per-token FLOPs (only the routed top-k experts run ≈ k×). With N=8, k=2 you " +
                 "get ~8× the knowledge for ~2× the compute. That's why Mixtral (2 of 8) and DeepSeek-V3 (8 of 256) " +
                 "are huge in params but cheap per token."
      },
      {
        q: "Mixture of Heads (MoH) is described as 'MoE applied to attention heads.' Which statement captures the analogy most precisely?",
        choices: [
          "MoH compresses the KV cache into a latent, exactly like MLA, but for the FFN",
          "Standard multi-head attention already SUMS all H heads with equal weight 1; MoH adds a router that scores heads per token, keeps the top-k, and replaces the equal weights with learned router weights — the FFN-expert routing of MoE, moved onto attention heads so total heads H can grow without growing active compute k",
          "MoH runs every head every time but stores the outputs in fp8 to save memory",
          "MoH and MoE are unrelated: MoE routes tokens to layers, MoH routes layers to tokens"
        ],
        answer: 1,
        explain: "MoE routes each token to top-k of N expert FFNs and weights them; MoH does the identical thing to " +
                 "attention heads. The key reframing: plain MHA = Concat(head_1..head_H)·Wo is a sum of heads with " +
                 "weight 1 each — a dense, equal-weight special case. MoH makes that sum sparse (top-k of H) and " +
                 "weighted (router softmax), so you can scale total heads H while active compute tracks k. Shared " +
                 "heads (always-on, DeepSeek-style) parallel MoE's shared experts."
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
