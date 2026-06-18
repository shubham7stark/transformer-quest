/* ============================================================================
 * LEVEL 07 — Generation & the KV Cache   (order 7)
 * ----------------------------------------------------------------------------
 * Follows the gold-standard pattern of 01-tokens-embeddings.js:
 *   - one IIFE, ONE TQ.registerLevel call, NO other globals
 *   - render() = short explanation blocks + a PRIMARY interactive sandbox,
 *     every number genuinely computed via TQ math (no faked attention/memory)
 *   - all color via TQ.colorFor / TQ.heatmap / TQ.vectorView / TQ.barRow
 *   - 4 quiz questions
 *
 * The math, all real and continuous with earlier levels (same toy sentence):
 *   - dk = head_dim slider. Wq=randMatrix(16,dk,101), Wk=(...,202), Wv=(...,303)
 *     via TQ.toyQKV, exactly the documented Q/K/V convention.
 *   - K_all = emb·Wk, V_all = emb·Wv, Q_all = emb·Wq (6 x dk each).
 *   - At step t (1..6): the K/V caches are the first t rows of K_all/V_all,
 *     displayed TRANSPOSED (dk x t) so each click appends a real COLUMN.
 *   - The new query q_t = Q_all[t-1] attends over the cache:
 *       scores[j] = dot(q_t, K_cache[j]); scaled = scores/sqrt(dk);
 *       weights = softmax(scaled)  -> automatically causal (cache holds j<=t-1).
 *   - Work counters: with-cache total = n (one new k,v per step);
 *     recompute-all total = n(n+1)/2. Savings = recompute/cache, all computed.
 *   - Causal-mask grid: 6x6 lower-triangular, rendered with colorFor.
 *   - Memory panel: cache_bytes = 2·n_layers·seq_len·n_kv_heads·head_dim·bytes,
 *     computed live from sliders, formatted in GB.
 * ========================================================================== */

(function () {
  "use strict";

  TQ.registerLevel({
    id: "kv-cache",
    order: 7,
    title: "Generation & the KV Cache",
    icon: "📼",
    tagline: "Decoders generate one token at a time — so cache the past instead of recomputing it.",
    preread: "Memory-Efficient Attention (setup)",
    objectives: [
      "Understand autoregressive decoding: one token per step, each attends to all prior tokens",
      "See why recomputing every past K/V each step is wasteful, and how a cache fixes it",
      "Feel the compute-with-cache vs without-cache contrast that motivates the next levels"
    ],

    render: function (root) {
      var toy = TQ.toy;
      var tokens = toy.tokens;
      var emb = toy.embeddings;
      var N = tokens.length;

      /* ----------------------------------------------------- intro narrative */
      root.appendChild(TQ.block(
        TQ.h(2, "Decoders generate left-to-right, one token at a time"),
        TQ.p(
          "A CNN sees the whole image at once: one forward pass, done. An ",
          TQ.el("strong", { text: "autoregressive decoder" }),
          " is different — it produces text one token per step, and each new token is fed back in as input for the next step. ",
          "(", TQ.el("em", { text: "Autoregressive" }), " just means each output token becomes part of the input for the next step.) ",
          "At step ", TQ.math("t"), " the model has the sequence so far (", TQ.math("t"),
          " tokens) and computes a single new token, ", TQ.math("t+1"), "."
        ),
        TQ.p(
          "Crucially, when computing the representation for position ", TQ.math("t"),
          ", attention lets that position look at every ", TQ.el("strong", { text: "earlier" }),
          " position — but never a later one (there is none yet). This \"only look left\" rule is ",
          TQ.el("strong", { text: "causal masking" }),
          ": future positions are masked out (set to −∞ before softmax — because ", TQ.math("e^(−∞) = 0"),
          ", that position gets exactly zero weight). ",
          "It's the architectural reason a language model can be trained to predict the next word without cheating by peeking ahead."
        )
      ));

      // Orienting diagram: the step-and-feedback LOOP at a glance — the one thing
      // the single-step sandbox below does not make obvious. Static inline SVG,
      // colors via CSS vars / currentColor only (no hardcoded hex).
      root.appendChild(TQ.figure(
        '<svg viewBox="0 0 560 132" width="560" height="132" role="img" ' +
          'aria-label="Autoregressive generation loop: token t is projected to q,k,v; k,v are appended to the KV cache; the query attends over the cache to produce an output; the next token is sampled and fed back" ' +
          'font-family="var(--mono)" font-size="11">' +
          '<defs><marker id="tq-l7-arrow" viewBox="0 0 10 10" refX="9" refY="5" ' +
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
            '<path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>' +
          // stage 1: input token t
          '<rect x="8" y="40" width="86" height="34" rx="7" fill="var(--panel-hi)" stroke="var(--line)"/>' +
          '<text x="51" y="61" text-anchor="middle" fill="var(--ink)">token t</text>' +
          // stage 2: project q,k,v
          '<rect x="120" y="40" width="96" height="34" rx="7" fill="var(--panel-hi)" stroke="var(--line)"/>' +
          '<text x="168" y="56" text-anchor="middle" fill="var(--ink)">project</text>' +
          '<text x="168" y="69" text-anchor="middle" fill="var(--ink-mute)" font-size="10">q_t, k_t, v_t</text>' +
          // stage 3: KV cache (stack of past columns + 1 new highlighted column)
          '<text x="294" y="30" text-anchor="middle" fill="var(--ink-mute)" font-size="10">append k_t,v_t → KV-cache</text>' +
          '<rect x="246" y="40" width="14" height="34" rx="2" fill="var(--panel-hi)" stroke="var(--line)"/>' +
          '<rect x="262" y="40" width="14" height="34" rx="2" fill="var(--panel-hi)" stroke="var(--line)"/>' +
          '<rect x="278" y="40" width="14" height="34" rx="2" fill="var(--panel-hi)" stroke="var(--line)"/>' +
          '<rect x="294" y="40" width="14" height="34" rx="2" fill="var(--accent)" stroke="var(--accent)"/>' +
          '<text x="301" y="88" text-anchor="middle" fill="var(--accent)" font-size="9">new col</text>' +
          // stage 4: attend -> out
          '<rect x="338" y="40" width="96" height="34" rx="7" fill="var(--panel-hi)" stroke="var(--line)"/>' +
          '<text x="386" y="56" text-anchor="middle" fill="var(--ink)">attend</text>' +
          '<text x="386" y="69" text-anchor="middle" fill="var(--ink-mute)" font-size="10">q_t over cache → out</text>' +
          // stage 5: sample next token
          '<rect x="458" y="40" width="94" height="34" rx="7" fill="var(--panel-hi)" stroke="var(--line)"/>' +
          '<text x="505" y="56" text-anchor="middle" fill="var(--ink)">sample</text>' +
          '<text x="505" y="69" text-anchor="middle" fill="var(--ink-mute)" font-size="10">token t+1</text>' +
          // forward arrows
          '<g stroke="currentColor" stroke-width="1.5" fill="none" color="var(--ink-faint)">' +
            '<line x1="96" y1="57" x2="116" y2="57" marker-end="url(#tq-l7-arrow)"/>' +
            '<line x1="218" y1="57" x2="242" y2="57" marker-end="url(#tq-l7-arrow)"/>' +
            '<line x1="310" y1="57" x2="334" y2="57" marker-end="url(#tq-l7-arrow)"/>' +
            '<line x1="436" y1="57" x2="456" y2="57" marker-end="url(#tq-l7-arrow)"/>' +
          '</g>' +
          // curved feedback arrow: token t+1 loops back to the input box
          '<g stroke="var(--accent)" stroke-width="1.5" fill="none" color="var(--accent)">' +
            '<path d="M505 76 C505 116, 51 116, 51 78" marker-end="url(#tq-l7-arrow)"/>' +
          '</g>' +
          '<text x="280" y="128" text-anchor="middle" fill="var(--accent)" font-size="10">feed back / next step</text>' +
          // note: only k,v are stored
          '<text x="294" y="18" text-anchor="middle" fill="var(--ink-faint)" font-size="10">only k,v are stored — q_t is discarded after this step</text>' +
        '</svg>',
        "Autoregressive generation is a loop: project the new token to q,k,v; append only k,v to the KV-cache; " +
        "let the query attend over the whole cache; sample the next token and feed it back. The sandbox below shows ONE turn of this loop."
      ));

      root.appendChild(TQ.block(
        TQ.h(2, "The same K and V get recomputed every step — that's the waste"),
        TQ.p(
          "Recall self-attention: each token projects its embedding into a query ", TQ.math("q"),
          ", a key ", TQ.math("k"), ", and a value ", TQ.math("v"),
          " (", TQ.math("Q=XWq, K=XWk, V=XWv"),
          " — i.e. multiply each embedding by a learned matrix, recap from L3). To produce the next token at step ", TQ.math("t"),
          ", you only need the NEW token's query ", TQ.math("q_t"),
          ". That ", TQ.math("q_t"), " dot-products against the keys of ALL tokens, softmaxes, and reads out a weighted sum of their values."
        ),
        TQ.p(
          "Here's the catch: the keys and values of tokens ", TQ.math("0..t−1"),
          " are identical to what they were last step — the past doesn't change. The naive loop recomputes K and V for the ",
          TQ.el("strong", { text: "entire prefix every single step" }),
          ". Over a generation of ", TQ.math("T"), " tokens that's roughly ",
          TQ.math("1+2+...+T ≈ T²/2"), " key/value computations. Quadratic, for information you already had."
        )
      ));

      root.appendChild(TQ.block(
        TQ.h(2, "Cache the past K and V; only the new token does work"),
        TQ.p(
          "The fix is embarrassingly simple: keep a ", TQ.el("strong", { text: "cache" }),
          " of all past keys and values. At step ", TQ.math("t"), " you compute only ", TQ.math("k_t"),
          " and ", TQ.math("v_t"), " (one token's worth of projection), append them as a new column to the K-cache and V-cache, then attend: ",
          TQ.math("weights = softmax(q_t · K_cacheᵀ / √dk)"), ", ", TQ.math("out = weights · V_cache"), "."
        ),
        TQ.p(
          "Per-step K/V ", TQ.el("em", { text: "projection" }), " work drops from ~", TQ.math("t"), " to ~", TQ.math("1"),
          ", so the running total of that recompute over ", TQ.math("T"), " tokens falls from ~", TQ.math("T²/2"),
          " to ~", TQ.math("T"), " — linear. (The query still dot-products against all ", TQ.math("t"),
          " cached keys, so attention ", TQ.el("em", { text: "scoring" }), " is still ", TQ.math("O(t)"),
          " per step — the cache kills redundant projection, not the intrinsic quadratic of attention itself.)"
        ),
        TQ.callout(
          "Note what is NOT cached: queries. A past token's query was only ever needed to compute that token's OWN output — " +
          "which already happened and won't change. Future tokens need past KEYS and VALUES (to attend to them), never past queries. " +
          "So you cache K and V, and throw the old q's away."
        )
      ));

      /* =====================================================================
       * PRIMARY INTERACTIVE — autoregressive-decoding sandbox
       * ===================================================================*/
      var sandbox = TQ.block(
        TQ.h(2, "Sandbox: generate the sentence, watch the cache grow"),
        TQ.p(
          "Below, the toy sentence is decoded one token at a time. Each ",
          TQ.el("strong", { text: "Generate next token" }),
          " click computes only the new token's ", TQ.math("k_t, v_t"),
          ", appends them as a new COLUMN to the growing K-cache and V-cache, and shows the new query ",
          TQ.math("q_t"), " attending across the whole cache. Two counters tally the cumulative key/value work."
        )
      );

      // --- live state -----------------------------------------------------
      var dk = 4;                 // head_dim slider value
      var t = 1;                  // tokens revealed so far (1..N)
      var mode = "cache";         // "cache" | "recompute"
      var cacheWork = 0;          // cumulative with-cache token-projections
      var recomputeWork = 0;      // cumulative recompute-all token-projections

      // projection matrices + full K/V/Q for current dk (rebuilt when dk changes)
      var qkv, K_all, V_all, Q_all;
      function rebuildProjections() {
        qkv = TQ.toyQKV(dk, 101, 202, 303);
        K_all = qkv.K; V_all = qkv.V; Q_all = qkv.Q; // each N x dk
      }
      rebuildProjections();

      // recompute the cumulative work counters from scratch for the current t,
      // so they're always consistent (no drift) and genuinely computed.
      function recountWork() {
        cacheWork = t;                  // 1 new (k,v) per step => total t
        recomputeWork = t * (t + 1) / 2; // sum_{s=1..t} s
      }
      recountWork();

      // controls + panels (filled by redraw)
      var legend = TQ.el("div", { class: "tq-legend" },
        TQ.el("span", { text: "low" }),
        TQ.el("div", { class: "tq-legend-scale" }),
        TQ.el("span", { text: "high" })
      );

      var genBtn = TQ.el("button", { class: "tq-btn tq-btn-accent", type: "button" });
      var resetBtn = TQ.el("button", { class: "tq-btn tq-btn-ghost", type: "button", text: "↺ Reset" });

      var modeSeg = TQ.segmented({
        options: [{ label: "With cache", value: "cache" }, { label: "Recompute all", value: "recompute" }],
        value: "cache",
        onChange: function (v) { mode = v; redraw(); }
      });

      var dkSlider = TQ.slider({
        min: 2, max: 16, step: 1, value: dk, label: "head_dim (dk)",
        format: function (v) { return String(Math.round(v)); },
        onInput: function (v) {
          dk = Math.round(v);
          rebuildProjections();
          redraw();
        }
      });

      // token strip showing revealed vs pending tokens
      var stripWrap = TQ.el("div", { class: "tq-kv-strip" });
      // caches (transposed dk x t heatmaps)
      var cacheWrap = TQ.el("div", { class: "tq-kv-caches" });
      // attention row for newest query
      var attnWrap = TQ.el("div", { class: "tq-panel tq-grow" });
      // work counters
      var workWrap = TQ.el("div", { class: "tq-kv-work" });

      function renderStrip() {
        stripWrap.innerHTML = "";
        for (var i = 0; i < N; i++) {
          var revealed = i < t;
          var isNew = i === t - 1;
          var cls = "tq-kv-tok" + (revealed ? " is-on" : " is-off") + (isNew ? " is-new" : "");
          stripWrap.appendChild(TQ.el("div", { class: cls, title: revealed ? "position " + i : "not generated yet" },
            TQ.el("span", { class: "tq-kv-tok-idx", text: "#" + i }),
            TQ.el("span", { class: "tq-kv-tok-word", text: revealed ? tokens[i] : "·" })
          ));
        }
      }

      // Build a dk x t heatmap from the first t rows of M (each row = one token's
      // vector); transpose so a click adds a column. Fixed color scale over the
      // whole matrix so past columns NEVER change color as new ones appear.
      function cacheHeat(M, title, scaleMax) {
        var rowsT = t;
        var cache = [];
        for (var r = 0; r < rowsT; r++) cache.push(M[r]); // t x dk
        var disp = TQ.transpose(cache);                   // dk x t
        var colLabels = [];
        for (var c = 0; c < rowsT; c++) colLabels.push(tokens[c]);
        var rowLabels = [];
        for (var d = 0; d < dk; d++) rowLabels.push("d" + d);
        return TQ.heatmap(disp, {
          title: title, rowLabels: rowLabels, colLabels: colLabels,
          cellSize: 30, min: -scaleMax, max: scaleMax,
          selectedCol: rowsT - 1,
          format: function (v) { return TQ.fmt(v, 1); }
        });
      }

      function renderCaches() {
        cacheWrap.innerHTML = "";
        // fixed symmetric scale from the FULL matrices so colors are stable
        var flatK = [].concat.apply([], K_all);
        var flatV = [].concat.apply([], V_all);
        var kMax = Math.max(Math.abs(TQ.minOf(flatK)), Math.abs(TQ.maxOf(flatK))) || 1;
        var vMax = Math.max(Math.abs(TQ.minOf(flatV)), Math.abs(TQ.maxOf(flatV))) || 1;

        var kPanel = TQ.el("div", { class: "tq-panel tq-grow" },
          cacheHeat(K_all, "K-cache  (dk=" + dk + " × " + t + " tokens)", kMax),
          TQ.note("Each click appends ONE column — the newest token's key. Past columns never change color: the past is frozen.")
        );
        var vPanel = TQ.el("div", { class: "tq-panel tq-grow" },
          cacheHeat(V_all, "V-cache  (dk=" + dk + " × " + t + " tokens)", vMax),
          TQ.note(mode === "recompute"
            ? "Recompute-all mode: every step re-derives K and V for ALL " + t + " tokens (the whole cache, not just the new column)."
            : "With-cache mode: only the new column is computed; the rest is read straight from memory.")
        );
        cacheWrap.appendChild(kPanel);
        cacheWrap.appendChild(vPanel);
      }

      function renderAttention() {
        attnWrap.innerHTML = "";
        var q = Q_all[t - 1];                  // newest query
        var scores = [], j;
        for (j = 0; j < t; j++) scores.push(TQ.dot(q, K_all[j]));
        var inv = 1 / Math.sqrt(dk);
        var scaled = scores.map(function (s) { return s * inv; });
        var weights = TQ.softmax(scaled);      // length t — automatically causal
        var labels = [];
        for (j = 0; j < t; j++) labels.push(tokens[j]);

        attnWrap.appendChild(TQ.el("div", { class: "tq-heatmap-title",
          text: "Attention of the newest query  q(" + tokens[t - 1] + ")  over the cache" }));
        attnWrap.appendChild(TQ.barRow(weights, {
          labels: labels, max: 1, format: function (v) { return TQ.fmt(v, 3); },
          highlight: t - 1
        }));
        attnWrap.appendChild(TQ.note(
          "Weights are non-zero only on positions ≤ " + (t - 1) + " (causal): there is no row for future tokens, " +
          "so they can't be attended to. Earlier tokens' queries are gone — we never recompute them."
        ));
      }

      function renderWork() {
        workWrap.innerHTML = "";
        var ratio = cacheWork > 0 ? (recomputeWork / cacheWork) : 1;
        var perStepRecompute = t; // this step's recompute-all cost
        var savedThisStep = perStepRecompute - 1;

        var cacheStat = TQ.el("div", { class: "tq-kv-stat" + (mode === "cache" ? " is-active" : "") },
          TQ.el("span", { class: "tq-stat-cap", text: "with cache — K/V projections" + (mode === "cache" ? " ◀ active" : "") }),
          TQ.el("span", { class: "tq-stat-big", text: String(cacheWork) }),
          TQ.el("span", { class: "tq-kv-stat-sub", text: "+1 this step (only the new token)" })
        );
        var recomputeStat = TQ.el("div", { class: "tq-kv-stat" + (mode === "recompute" ? " is-active" : "") },
          TQ.el("span", { class: "tq-stat-cap", text: "recompute all — K/V projections" + (mode === "recompute" ? " ◀ active" : "") }),
          TQ.el("span", { class: "tq-stat-big", text: String(recomputeWork) }),
          TQ.el("span", { class: "tq-kv-stat-sub", text: "+" + perStepRecompute + " this step (re-did all " + t + ")" })
        );
        var ratioStat = TQ.el("div", { class: "tq-kv-stat is-ratio" },
          TQ.el("span", { class: "tq-stat-cap", text: "wasted work avoided" }),
          TQ.el("span", { class: "tq-stat-big", text: TQ.fmt(ratio, 2) + "×" }),
          TQ.el("span", { class: "tq-kv-stat-sub", text: "saved " + savedThisStep + " projection" + (savedThisStep === 1 ? "" : "s") + " this step" })
        );

        workWrap.appendChild(TQ.el("div", { class: "tq-kv-stat-grid" }, cacheStat, recomputeStat, ratioStat));

        // honest extrapolation to a long context
        var T = 1000;
        var bigRatio = (T + 1) / 2; // recompute T(T+1)/2 vs cache T  -> (T+1)/2
        workWrap.appendChild(TQ.note(
          "Extrapolating honestly: for T tokens recompute-all does T(T+1)/2 projections vs T with the cache → ratio (T+1)/2. " +
          "At T=" + T + " that's ~" + TQ.fmt(bigRatio, 0) + "× the K/V work avoided. The two curves are linear vs quadratic — they diverge fast."
        ));
      }

      function syncButtons() {
        genBtn.textContent = t >= N ? "Sentence complete ✓" : "Generate next token →";
        genBtn.disabled = t >= N;
        resetBtn.disabled = t <= 1;
      }

      function redraw() {
        recountWork();
        renderStrip();
        renderCaches();
        renderAttention();
        renderWork();
        syncButtons();
      }

      genBtn.addEventListener("click", function () {
        if (t >= N) return;
        t += 1;
        redraw();
      });
      resetBtn.addEventListener("click", function () {
        t = 1;
        redraw();
      });

      var controls = TQ.el("div", { class: "tq-controls-row tq-kv-controls" },
        genBtn, resetBtn,
        TQ.el("div", { class: "tq-kv-mode" },
          TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "mode"),
          modeSeg.el
        ),
        dkSlider.el
      );

      sandbox.appendChild(controls);
      sandbox.appendChild(legend);
      sandbox.appendChild(stripWrap);
      sandbox.appendChild(workWrap);
      sandbox.appendChild(cacheWrap);
      sandbox.appendChild(attnWrap);
      redraw();
      root.appendChild(sandbox);

      /* ---------------------------------- the loop above, as real PyTorch code */
      root.appendChild(TQ.block(
        TQ.h(2, "The same loop, in code"),
        TQ.p(
          "Here is the sandbox as raw PyTorch — framework-light matmuls so the cache mechanics aren't hidden. ",
          "Each ", TQ.el("strong", { text: "Generate next token" }), " click runs one iteration of ", TQ.math("generate_with_cache"),
          ": the new column appended to ", TQ.math("K_cache"), "/", TQ.math("V_cache"),
          " is exactly the new heatmap column you watched grow above. The ", TQ.math("naive_no_cache"),
          " contrast reprojects K and V for the WHOLE prefix every step — the wasted ", TQ.math("~T²/2"), " vs ", TQ.math("~T"), " work, made visible."
        ),
        TQ.code(
          "import torch\n" +
          "import torch.nn.functional as F\n" +
          "\n" +
          "# learned projections (recap from L3); dk = head_dim\n" +
          "d_model, dk = 16, 4\n" +
          "Wq = torch.randn(d_model, dk)\n" +
          "Wk = torch.randn(d_model, dk)\n" +
          "Wv = torch.randn(d_model, dk)\n" +
          "\n" +
          "def step(x_t, K_cache, V_cache):\n" +
          "    # 1. project ONLY the new token  x_t: (d_model,)\n" +
          "    q_t = x_t @ Wq                       # (dk,)\n" +
          "    k_t = x_t @ Wk                       # (dk,)\n" +
          "    v_t = x_t @ Wv                       # (dk,)\n" +
          "\n" +
          "    # 2. append k_t, v_t as a new row/column of the cache\n" +
          "    K_cache = torch.cat([K_cache, k_t[None, :]], dim=0)  # (t, dk)\n" +
          "    V_cache = torch.cat([V_cache, v_t[None, :]], dim=0)  # (t, dk)\n" +
          "\n" +
          "    # 3. the single new query attends over the WHOLE cache\n" +
          "    scores = q_t @ K_cache.T / (dk ** 0.5)              # (t,)\n" +
          "    weights = F.softmax(scores, dim=-1)                 # (t,) causal for free\n" +
          "    out = weights @ V_cache                             # (dk,)\n" +
          "    return out, K_cache, V_cache\n" +
          "\n" +
          "def generate_with_cache(embeds, n_new):\n" +
          "    # embeds: (prompt_len, d_model). Cache grows one column per step.\n" +
          "    K_cache = embeds @ Wk                 # prefill: (prompt_len, dk)\n" +
          "    V_cache = embeds @ Wv                 # the whole prompt is now cached\n" +
          "    # the last prompt position already gave us its q; sample the FIRST new\n" +
          "    # token from it (its k,v are already in the cache -- don't re-append).\n" +
          "    q_last = embeds[-1] @ Wq                            # (dk,)\n" +
          "    scores = q_last @ K_cache.T / (dk ** 0.5)          # (prompt_len,)\n" +
          "    out = F.softmax(scores, dim=-1) @ V_cache          # (dk,)\n" +
          "    x_t = next_embed(out)                 # first generated token\n" +
          "    for _ in range(n_new):\n" +
          "        # x_t is a genuinely NEW token: project it once, append, attend.\n" +
          "        out, K_cache, V_cache = step(x_t, K_cache, V_cache)\n" +
          "        x_t = next_embed(out)             # sample/argmax -> feed back\n" +
          "    return K_cache                        # 1 (k,v) projection per step -> ~T total\n" +
          "\n" +
          "def naive_no_cache(embeds, n_new):\n" +
          "    # WITHOUT a cache: every step recomputes K, V for the ENTIRE prefix.\n" +
          "    X = embeds\n" +
          "    for _ in range(n_new):\n" +
          "        K = X @ Wk                        # (t, dk) re-derived from scratch\n" +
          "        V = X @ Wv                        # (t, dk) re-derived from scratch\n" +
          "        q_t = (X[-1] @ Wq)                # only the last query is used\n" +
          "        weights = F.softmax(q_t @ K.T / (dk ** 0.5), dim=-1)\n" +
          "        out = weights @ V\n" +
          "        X = torch.cat([X, next_embed(out)[None, :]], dim=0)\n" +
          "    # K,V recomputed for 1+2+...+T tokens -> ~T**2/2 projections of pure waste\n",
          { lang: "python", label: "KV-cache generation loop",
            caption: "Each Generate-next-token click = one call to step(): the appended row of K_cache/V_cache is the new heatmap column above. " +
                     "naive_no_cache reprojects the whole prefix every step — the quadratic waste the cache removes." }
        )
      ));

      /* ----------------------------------------- causal mask grid (the rule) */
      var maskBlock = TQ.block(
        TQ.h(2, "Why \"only look left\" comes for free"),
        TQ.p(
          "The cache only ever holds positions ≤ ", TQ.math("t"),
          ", so softmaxing over it is automatically causal. During training (whole sequence at once) you instead apply a ",
          TQ.el("strong", { text: "causal mask" }), ": for query position ", TQ.math("i"),
          ", any future key ", TQ.math("j > i"),
          " is set to −∞ before softmax, so its weight is exactly 0. Below, warm cells are allowed (", TQ.math("j ≤ i"),
          "); dark cells are masked future positions."
        )
      );
      var mask = [];
      for (var mi = 0; mi < N; mi++) {
        var mrow = [];
        for (var mj = 0; mj < N; mj++) mrow.push(mj <= mi ? 1 : 0);
        mask.push(mrow);
      }
      maskBlock.appendChild(TQ.heatmap(mask, {
        title: "Causal mask  M[i][j] = 1 if j ≤ i else 0  (allowed vs −∞)",
        rowLabels: tokens, colLabels: tokens, cellSize: 40, min: 0, max: 1,
        format: function (v) { return v ? "✓" : "∅"; }
      }));
      maskBlock.appendChild(TQ.note(
        "Row = the querying token; column = the token it may read. Lower-triangular: each token sees itself and the past, " +
        "never the future. Dark (∅) cells become −∞ pre-softmax → weight 0."
      ));
      root.appendChild(maskBlock);

      /* -------------------------------- memory formula panel (bridge to L8/9) */
      root.appendChild(TQ.block(
        TQ.h(2, "The cache buys speed but costs memory — and that bill is the next two levels"),
        TQ.p(
          "Caching trades recompute for storage, and the storage grows with everything: ",
          TQ.math("cache_bytes = 2 · n_layers · seq_len · n_kv_heads · head_dim · bytes"),
          " (the 2 is K and V; don't worry about ", TQ.math("n_kv_heads"),
          " yet — that term is exactly what the next level unpacks). It scales linearly with sequence length, so a 100k-token context is a genuinely large, " +
          "ever-growing tensor in GPU memory — often dwarfing the model weights for long contexts. ",
          "That's the real bottleneck of LLM serving: not FLOPs, but KV-cache memory."
        )
      ));

      var memBlock = TQ.block(
        TQ.h(3, "Feel the cache grow"),
        TQ.p("Move the sliders — the byte total is computed live from the standard formula. Watch it move linearly with every term.")
      );

      var mem = { layers: 32, seq: 8192, kvHeads: 32, headDim: 128, bytes: 2 };
      var memOut = TQ.el("div", { class: "tq-panel" });

      function fmtBytes(b) {
        var gb = b / (1024 * 1024 * 1024);
        if (gb >= 1) return TQ.fmt(gb, 2) + " GB";
        var mb = b / (1024 * 1024);
        return TQ.fmt(mb, 1) + " MB";
      }

      function updateMem() {
        memOut.innerHTML = "";
        var bytes = 2 * mem.layers * mem.seq * mem.kvHeads * mem.headDim * mem.bytes;
        memOut.appendChild(TQ.el("div", { class: "tq-kv-stat" },
          TQ.el("span", { class: "tq-stat-cap", text: "cache_bytes" }),
          TQ.el("span", { class: "tq-stat-big", text: fmtBytes(bytes) }),
          TQ.el("span", { class: "tq-kv-stat-sub",
            text: "2 × " + mem.layers + " × " + mem.seq + " × " + mem.kvHeads + " × " + mem.headDim + " × " + mem.bytes + " bytes" })
        ));
        memOut.appendChild(TQ.callout(
          "Levels 8–9 attack exactly the n_kv_heads and head_dim terms: MQA/GQA shrink n_kv_heads by sharing keys/values " +
          "across query heads, and MLA compresses K and V into a small shared latent. You can't appreciate those tricks until you've felt the cache grow."
        ));
      }

      var sLayers = TQ.slider({ min: 1, max: 96, step: 1, value: mem.layers, label: "n_layers",
        format: function (v) { return String(Math.round(v)); },
        onInput: function (v) { mem.layers = Math.round(v); updateMem(); } });
      var sSeq = TQ.slider({ min: 128, max: 131072, step: 128, value: mem.seq, label: "seq_len (tokens)",
        format: function (v) { return String(Math.round(v)); },
        onInput: function (v) { mem.seq = Math.round(v); updateMem(); } });
      var sHeads = TQ.slider({ min: 1, max: 64, step: 1, value: mem.kvHeads, label: "n_kv_heads",
        format: function (v) { return String(Math.round(v)); },
        onInput: function (v) { mem.kvHeads = Math.round(v); updateMem(); } });
      var sHeadDim = TQ.slider({ min: 16, max: 256, step: 16, value: mem.headDim, label: "head_dim",
        format: function (v) { return String(Math.round(v)); },
        onInput: function (v) { mem.headDim = Math.round(v); updateMem(); } });
      var sBytes = TQ.slider({ min: 1, max: 4, step: 1, value: mem.bytes, label: "bytes/elem",
        format: function (v) { return String(Math.round(v)) + " (" + ({1: "fp8", 2: "fp16", 4: "fp32"}[Math.round(v)] || "?") + ")"; },
        onInput: function (v) { mem.bytes = Math.round(v); updateMem(); } });

      memBlock.appendChild(TQ.el("div", { class: "tq-kv-mem-controls" },
        sLayers.el, sSeq.el, sHeads.el, sHeadDim.el, sBytes.el));
      memBlock.appendChild(memOut);
      updateMem();
      root.appendChild(memBlock);

      /* ----------------------------------------------- go deeper (resources) */
      root.appendChild(TQ.resources("Go deeper — generation & the KV cache", [
        {
          label: "kipply — Transformer Inference Arithmetic",
          url: "https://kipply.github.io/transformer-inference-arithmetic/",
          kind: "blog",
          note: "Works out the actual KV-cache byte budget and why memory, not FLOPs, bounds long-context serving."
        },
        {
          label: "Andrej Karpathy — Let's build GPT from scratch",
          url: "https://www.youtube.com/watch?v=kCc8FmEb1nY",
          kind: "video",
          note: "Builds a decoder end to end, including the causal mask and autoregressive generation loop."
        },
        {
          label: "Andrej Karpathy — nanoGPT",
          url: "https://github.com/karpathy/nanoGPT",
          kind: "code",
          note: "A clean, readable GPT with a real generate() loop — see the cache and sampling in production-style code."
        },
        {
          label: "The Annotated Transformer",
          url: "https://nlp.seas.harvard.edu/annotated-transformer/",
          kind: "code",
          note: "Line-by-line transformer with the attention + masking math you just clicked through."
        },
        {
          label: "The Illustrated Transformer",
          url: "https://jalammar.github.io/illustrated-transformer/",
          kind: "blog",
          note: "Jay Alammar's visual walkthrough of Q/K/V attention — the projections this cache stores."
        }
      ]));

      /* ---------------------------------------------------- wrap-up takeaway */
      root.appendChild(TQ.block(
        TQ.h(2, "Carry this forward"),
        TQ.callout(
          "The KV cache turns quadratic K/V recompute into linear K/V work (attention scoring over the cache is still " +
          "O(t) per step — the cache attacks redundant projection and memory, not attention's intrinsic quadratic) — but it " +
          "creates a memory tensor that grows with context " +
          "length, layers, kv-heads, and head_dim. Shrinking that tensor is the whole job of MQA, GQA (Level 8) and MLA (Level 9)."
        )
      ));

      /* scoped styles — colors only via CSS variables, never hardcoded hex */
      injectOnce("tq-lvl07-css",
        ".tq-kv-controls{align-items:flex-end;gap:16px;flex-wrap:wrap;margin-bottom:12px}" +
        ".tq-kv-mode{display:flex;flex-direction:column}" +
        ".tq-kv-strip{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 14px}" +
        ".tq-kv-tok{display:inline-flex;flex-direction:column;align-items:center;gap:2px;" +
          "min-width:46px;border:1px solid var(--line);border-radius:10px;padding:7px 10px;" +
          "background:var(--panel-hi);transition:opacity .2s,border-color .2s}" +
        ".tq-kv-tok.is-off{opacity:.32}" +
        ".tq-kv-tok.is-new{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}" +
        ".tq-kv-tok-idx{font-size:10px;color:var(--ink-mute);font-family:var(--mono)}" +
        ".tq-kv-tok-word{font-size:15px;font-weight:700}" +
        ".tq-kv-caches{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;margin-top:6px}" +
        ".tq-kv-work{margin:6px 0 10px}" +
        ".tq-kv-stat-grid{display:flex;gap:14px;flex-wrap:wrap}" +
        ".tq-kv-stat{display:flex;flex-direction:column;gap:2px;background:var(--panel-hi);" +
          "border:1px solid var(--line);border-radius:12px;padding:12px 16px;min-width:180px;flex:1}" +
        ".tq-kv-stat.is-ratio{border-color:var(--accent)}" +
        ".tq-kv-stat.is-active{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}" +
        ".tq-kv-stat-sub{font-size:11px;color:var(--ink-mute)}" +
        ".tq-kv-mem-controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));" +
          "gap:12px 18px;margin:8px 0 14px}");
    },

    quiz: [
      {
        q: "During autoregressive generation with a KV cache, what does the model actually compute from scratch at each new step?",
        choices: [
          "The keys and values for all tokens seen so far",
          "Only the new token's query, key, and value; it reuses the cached keys/values for all past tokens",
          "Nothing — the entire output is read directly from the cache",
          "The full attention matrix for every pair of tokens in the sequence"
        ],
        answer: 1,
        explain: "Past tokens' keys/values don't change, so they're cached. Each step only projects the new token's q, k, v and attends q_t against the cached K and V."
      },
      {
        q: "Why are queries NOT stored in the KV cache, while keys and values are?",
        choices: [
          "Queries are too large to store",
          "A past token's query was only needed to produce that token's own output (already done); future tokens attend to past keys/values, never past queries",
          "Queries are recomputed faster than keys, so caching them is pointless",
          "Queries are identical across all tokens, so one copy suffices"
        ],
        answer: 1,
        explain: "Attention uses the NEW query against ALL past keys/values. Old queries have no future use once their token's output is computed, so only K and V are cached."
      },
      {
        q: "Generating T tokens, how does total key/value projection work compare with vs. without the cache?",
        choices: [
          "Both are about T — the cache saves memory, not compute",
          "Without cache ~T²/2 (recompute the whole prefix each step); with cache ~T (one new token per step) — a linear vs quadratic difference",
          "With cache is slower because of the lookup overhead",
          "Without cache ~T; with cache ~T²/2"
        ],
        answer: 1,
        explain: "Recomputing K/V for the length-t prefix each step sums to 1+2+...+T ≈ T²/2. Caching computes one new token's K/V per step → ~T total. The savings grow with context."
      },
      {
        q: "The KV cache fixes wasted recompute. What new problem does it create that levels 8–9 (MQA/GQA/MLA) exist to solve?",
        choices: [
          "It makes attention less accurate",
          "It requires recomputing queries instead",
          "Its memory grows linearly with context (2·n_layers·seq_len·n_kv_heads·head_dim·bytes), becoming the dominant cost for long contexts",
          "It breaks causal masking, letting tokens see the future"
        ],
        answer: 2,
        explain: "The cache trades compute for memory. Cache size scales with sequence length × layers × kv-heads × head_dim, so long contexts make it huge — MQA/GQA shrink n_kv_heads and MLA compresses K/V into a latent."
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
