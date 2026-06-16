/* ============================================================================
 * LEVEL 09 — Multi-Head Latent Attention (DeepSeek)   (order 9)
 * ----------------------------------------------------------------------------
 * The payoff of the KV-cache arc. MHA/GQA/MQA shrink the cache by sharing K/V
 * across heads. MLA asks a sharper question: why cache per-head K/V at all?
 * Down-project every token to ONE small latent vector c (that's all you cache,
 * plus a tiny decoupled RoPE key), and UP-project back to rich per-head K/V at
 * use time. It's a low-rank factorization of the KV projection.
 *
 * Everything shown is genuinely computed via TQ math (no faked numbers):
 *   - per-head K/V from TQ.toyQKV(head_dim, seedQ+i, seedK+i, seedV+i)
 *   - latent c = embeddings · W_DKV (W_DKV = TQ.randMatrix(dModel, d_c, seed))
 *   - rope key  r = embeddings · W_rope
 *   - reconstructed K_i = c · W_UK_i, V_i = c · W_UV_i  (low-rank up-projection)
 *   - the four cache formulas (MHA/GQA/MQA/MLA) on ONE shared TQ.barRow axis
 *
 * Pattern copied from level 01: one IIFE, ONE TQ.registerLevel, no other
 * globals, all color via TQ.colorFor / vectorView / barRow.
 * ========================================================================== */

(function () {
  "use strict";

  TQ.registerLevel({
    id: "mla",
    order: 9,
    title: "Multi-Head Latent Attention (DeepSeek)",
    icon: "💎",
    tagline: "Don't just share KV heads — compress them into a tiny shared latent you cache.",
    preread: "DeepSeek MLA",
    objectives: [
      "Understand MLA: down-project K/V into one small latent vector c (cached), up-project to per-head K/V at use time",
      "See the decoupled RoPE key carried separately alongside the latent",
      "Compare MLA cache vs MHA/GQA/MQA on the same axes — MLA can match MHA quality while caching far less than even GQA"
    ],

    render: function (root) {
      var toy = TQ.toy;
      var tokens = toy.tokens;
      var emb = toy.embeddings;
      var dModel = toy.dModel;
      var n = tokens.length;

      /* ============================================================== *
       *  NARRATIVE BLOCK 1 — the cache is the bottleneck
       * ============================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "The cache is the bottleneck — so stop caching K and V"),
        TQ.p(
          "You've met the KV cache: per token, per layer, you stash every head's ",
          TQ.el("strong", { text: "K" }), " and ", TQ.el("strong", { text: "V" }),
          " so attention over history isn't recomputed each step. Plain ",
          TQ.el("strong", { text: "MHA" }), " caches ", TQ.math("2 · n_heads · head_dim"),
          " numbers per token per layer. ", TQ.el("strong", { text: "GQA/MQA" }),
          " shrank that by letting heads ", TQ.el("em", { text: "share" }),
          " K/V (fewer kv-heads)."
        ),
        TQ.p(
          TQ.el("strong", { text: "MLA asks a sharper question:" }),
          " why cache per-head K/V at all? Cache one small shared latent vector ",
          TQ.math("c"), " and reconstruct every head's K and V from it on the fly. ",
          "You already know this move from your CV days — it's a ",
          TQ.el("strong", { text: "low-rank factorization" }),
          ". The full per-head K/V projection is a fat matrix; MLA factors it as ",
          "“squeeze down to ", TQ.math("d_c"), ", then expand back up.” ",
          "You store only the squeezed thing."
        ),
        TQ.callout([
          "Bridge from SVD: replacing a fat projection with down-then-up (rank ",
          TQ.math("d_c"), ") is exactly rank-k truncation. The latent ", TQ.math("c"),
          " is the compressed code; the up-projections are the basis that re-expands it."
        ])
      ));

      /* ============================================================== *
       *  NARRATIVE BLOCK 2 — down/up projection
       * ============================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "Down-project to cache, up-project to use"),
        TQ.p(
          "Each token's embedding ", TQ.math("h"), " is multiplied by a down-projection ",
          TQ.math("W_DKV"), " to produce a latent ", TQ.math("c"), " of dimension ",
          TQ.math("d_c"), " — small (think 512 when the full K/V across all heads would be ~16k). ",
          TQ.math("c"), " is the ", TQ.el("strong", { text: "only" }),
          " thing written to the cache (plus the rope key below)."
        ),
        TQ.p(
          "At attention time you up-project: ", TQ.math("K_i = c · W_UK_i"),
          " and ", TQ.math("V_i = c · W_UV_i"), " give head ", TQ.math("i"),
          "'s keys and values. Because ", TQ.math("W_UK"), " and ", TQ.math("W_UV"),
          " are fixed model weights, they don't live in the cache — they're paid once. ",
          "So a cache that scaled with ", TQ.math("n_heads · head_dim"),
          " now scales with just ", TQ.math("d_c"), ". That's the whole trick, and it's ",
          "why MLA can match MHA quality (the up-projection lets each head recover a rich, ",
          "distinct K/V) while caching less than even GQA."
        )
      ));

      /* ============================================================== *
       *  NARRATIVE BLOCK 3 — decoupled RoPE
       * ============================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "Why RoPE has to ride alongside (the decoupled key)"),
        TQ.p(
          "Here's the subtle bug MLA designs around. ", TQ.el("strong", { text: "RoPE" }),
          " (the rotary position trick from L5) rotates each key by an angle that depends on ",
          "the token's absolute position ", TQ.el("em", { text: "before" }),
          " the dot product — so ", TQ.math("q·k"), " ends up depending only on relative distance. ",
          "If you bake RoPE into the cached latent, the rotation is frozen at cache-time and you can ",
          "no longer absorb the up-projection ", TQ.math("W_UK"),
          " into the query side; the position math and the low-rank math fight."
        ),
        TQ.p(
          "Make that 'absorb' concrete. The content score is ", TQ.math("q · K_iᵀ = (x·W_Q) · (c·W_UK)ᵀ"),
          ". Matrix multiply is associative, so you can pre-fold ", TQ.math("W_UK"),
          " into the query projection once: ", TQ.math("W_Q' = W_Q · W_UKᵀ"),
          ", and then the score is just ", TQ.math("(x·W_Q') · cᵀ"),
          " — the per-head key ", TQ.math("K_i"),
          " is never actually materialized from the latent at decode time; you dot the query straight against the tiny cached ",
          TQ.math("c"), ". That folding is exactly what makes MLA cheap. But RoPE inserts a ",
          TQ.el("em", { text: "position-dependent rotation" }), " R(pos) between ", TQ.math("c"),
          " and the query (", TQ.math("q·R(pos)·(c·W_UK)ᵀ"),
          "), and R(pos) changes every token — so it can't be pre-folded into a single fixed ",
          TQ.math("W_Q'"), ". That's the collision, and why the rope key is split out."
        ),
        TQ.p(
          "DeepSeek's fix: split the key in two. A ", TQ.el("strong", { text: "content" }),
          " part comes from the latent ", TQ.math("c"),
          " (no position, fully compressible). A small ", TQ.el("strong", { text: "separate" }),
          " rope key ", TQ.math("k_rope"), " of dimension ", TQ.math("d_rope"),
          " carries the rotary position info and is cached as-is beside ", TQ.math("c"),
          ". The query gets a matching rope part. Final score = (content from latent) + (rope part). ",
          "You cache ", TQ.math("d_c + d_rope"),
          " per token per layer — still tiny — and position survives compression."
        )
      ));

      /* ============================================================== *
       *  PRIMARY VISUAL 1 — collapse / cache / expand stepper
       * ============================================================== */
      var animBlock = TQ.block(
        TQ.h(2, "Watch the collapse → cache → expand"),
        TQ.p(
          "Real numbers from our toy sentence. We follow token ",
          TQ.el("strong", { text: "“cat”" }),
          " (#1). Step through: many tall per-head K/V columns ", TQ.el("strong", { text: "collapse" }),
          " into ONE short latent ", TQ.math("c"), " (+ a small rope key ", TQ.math("r"),
          "); that latent is all that's cached; then it ", TQ.el("strong", { text: "expands" }),
          " back out to full per-head K/V at use time — no head merged away."
        )
      );
      var legend = TQ.el("div", { class: "tq-legend" },
        TQ.el("span", { text: "low" }),
        TQ.el("div", { class: "tq-legend-scale" }),
        TQ.el("span", { text: "high" })
      );
      animBlock.appendChild(legend);

      // ---- fixed dims for the animation (kept legible / small) ----
      var ANIM_HEADS = 4;     // show 4 heads so the collapse is dramatic but readable
      var ANIM_HEAD_DIM = 8;  // per-head K/V dim for the toy
      var ANIM_DC = 6;        // toy latent dim (short column)
      var ANIM_DROPE = 3;     // toy rope key dim (tiny)
      var TOK = 1;            // "cat"

      // seeds (distinct so heads/projections differ deterministically)
      var SEED_BASE_Q = 1100, SEED_BASE_K = 2200, SEED_BASE_V = 3300;
      var SEED_DKV = 9001, SEED_ROPE = 9101, SEED_BASE_UK = 9200, SEED_BASE_UV = 9400;

      // per-head K and V (real), for the followed token
      var headK = [], headV = [];
      for (var hh = 0; hh < ANIM_HEADS; hh++) {
        var qkv = TQ.toyQKV(ANIM_HEAD_DIM, SEED_BASE_Q + hh, SEED_BASE_K + hh, SEED_BASE_V + hh);
        headK.push(qkv.K[TOK]);
        headV.push(qkv.V[TOK]);
      }

      // latent c = emb · W_DKV  (real); take the followed token's row
      var W_DKV = TQ.randMatrix(dModel, ANIM_DC, SEED_DKV);
      var Cmat = TQ.matmul(emb, W_DKV);
      var cVec = Cmat[TOK];

      // rope key r = emb · W_rope (real)
      var W_ROPE = TQ.randMatrix(dModel, ANIM_DROPE, SEED_ROPE);
      var Rmat = TQ.matmul(emb, W_ROPE);
      var rVec = Rmat[TOK];

      // reconstructed per-head K/V from the latent: K_i = c · W_UK_i (real)
      // (c is 1xd_c here; wrap as a 1-row matrix to matmul with d_c x head_dim)
      var cRow = [cVec];
      var reK = [], reV = [];
      for (var uh = 0; uh < ANIM_HEADS; uh++) {
        var W_UK = TQ.randMatrix(ANIM_DC, ANIM_HEAD_DIM, SEED_BASE_UK + uh);
        var W_UV = TQ.randMatrix(ANIM_DC, ANIM_HEAD_DIM, SEED_BASE_UV + uh);
        reK.push(TQ.matmul(cRow, W_UK)[0]);
        reV.push(TQ.matmul(cRow, W_UV)[0]);
      }

      // helper: a labelled column of K and V vectorViews for one head
      function headColumn(label, kvec, vvec, dim) {
        return TQ.el("div", { class: "tq-mla-headcol" },
          TQ.el("div", { class: "tq-mla-headcap", text: label }),
          TQ.vectorView(kvec, { label: "K", cellSize: 16 }),
          TQ.vectorView(vvec, { label: "V", cellSize: 16 }),
          TQ.el("div", { class: "tq-mla-dim", text: dim + "-d" })
        );
      }

      function stepPerHead(container, dim, ks, vs, capPrefix) {
        var cols = TQ.el("div", { class: "tq-mla-cols" });
        for (var i = 0; i < ANIM_HEADS; i++) {
          cols.appendChild(headColumn((capPrefix || "head ") + i, ks[i], vs[i], dim));
        }
        container.appendChild(cols);
      }

      function latentColumn() {
        return TQ.el("div", { class: "tq-mla-latentcol" },
          TQ.el("div", { class: "tq-mla-headcap", text: "latent c" }),
          TQ.vectorView(cVec, { label: "c", cellSize: 18 }),
          TQ.el("div", { class: "tq-mla-dim", text: "d_c = " + ANIM_DC })
        );
      }
      function ropeColumn(greyed) {
        return TQ.el("div", { class: "tq-mla-ropecol" + (greyed ? "" : " is-live") },
          TQ.el("div", { class: "tq-mla-headcap", text: "rope key r" }),
          TQ.vectorView(rVec, { label: "r", cellSize: 18 }),
          TQ.el("div", { class: "tq-mla-dim", text: "d_rope = " + ANIM_DROPE })
        );
      }

      var stepper = TQ.stepper([
        {
          label: "1 · Per-head K/V (what MHA caches)",
          run: function (c) {
            c.appendChild(TQ.p(
              ANIM_HEADS + " heads, each with its own ", TQ.math(ANIM_HEAD_DIM + "-d"),
              " K and V for token “cat”. MHA stores ", TQ.el("strong", { text: "all of these" }),
              " — that's ", TQ.math("2 · n_heads · head_dim"), " numbers per token per layer."
            ));
            stepPerHead(c, ANIM_HEAD_DIM, headK, headV);
          }
        },
        {
          label: "2 · Down-project → latent c (+ rope key r)",
          run: function (c) {
            c.appendChild(TQ.p(
              "All those columns collapse into ONE short latent ", TQ.math("c = h · W_DKV"),
              " of dim ", TQ.math("d_c = " + ANIM_DC),
              ", with a tiny decoupled rope key ", TQ.math("r"), " beside it."
            ));
            c.appendChild(TQ.el("div", { class: "tq-mla-merge" },
              latentColumn(), ropeColumn(false)
            ));
            c.appendChild(TQ.note(
              "The fat per-head K/V is gone — squeezed to a single low-rank code plus a position tag."
            ));
          }
        },
        {
          label: "3 · Cache only c (+ r) — this is ALL that's stored",
          run: function (c) {
            c.appendChild(TQ.p(
              "Everything except ", TQ.math("c"), " and ", TQ.math("r"),
              " is greyed out: the per-head K/V is NOT in the cache. ",
              "Only ", TQ.math("d_c + d_rope = " + (ANIM_DC + ANIM_DROPE)),
              " numbers per token per layer are written."
            ));
            var ghost = TQ.el("div", { class: "tq-mla-cols is-ghost" });
            for (var i = 0; i < ANIM_HEADS; i++) {
              ghost.appendChild(headColumn("head " + i, headK[i], headV[i], ANIM_HEAD_DIM));
            }
            c.appendChild(TQ.el("div", { class: "tq-mla-merge" },
              latentColumn(), ropeColumn(false),
              TQ.el("span", { class: "tq-mla-arrow", text: "✗ not cached →" }),
              ghost
            ));
            c.appendChild(TQ.callout(
              "The rope key never disappears — position info is cached separately and survives " +
              "compression (this is the L5 RoPE thread)."
            ));
          }
        },
        {
          label: "4 · Up-project at use time → rich per-head K/V again",
          run: function (c) {
            c.appendChild(TQ.p(
              "At attention time the tiny ", TQ.math("c"), " fans back OUT: ",
              TQ.math("K_i = c · W_UK_i"), ", ", TQ.math("V_i = c · W_UV_i"),
              ". Each head gets its own full ", TQ.math(ANIM_HEAD_DIM + "-d"),
              " K/V again — distinct per head, reconstructed not stored."
            ));
            c.appendChild(TQ.el("div", { class: "tq-mla-merge" },
              latentColumn(),
              TQ.el("span", { class: "tq-mla-arrow", text: "→ expand →" })
            ));
            stepPerHead(c, ANIM_HEAD_DIM, reK, reV, "K_/V_ head ");
            c.appendChild(TQ.note(
              "These are reconstructed via real " + TQ.fmt(ANIM_DC, 0) + "→" + ANIM_HEAD_DIM +
              " up-projections — no head was merged away, unlike GQA/MQA."
            ));
          }
        }
      ]);
      animBlock.appendChild(stepper);
      root.appendChild(animBlock);

      /* ============================================================== *
       *  PRIMARY VISUAL 2 — cache calculator (MHA/GQA/MQA/MLA)
       * ============================================================== */
      var calcBlock = TQ.block(
        TQ.h(2, "The cache calculator — four modes, one axis"),
        TQ.p(
          "Now the punchline on bytes. Same model config across all four modes so the bars are ",
          "directly comparable. Cache per token per layer:"
        ),
        TQ.el("div", { class: "tq-mla-formulas" },
          TQ.kv("MHA", "2 · n_heads · head_dim · bytes"),
          TQ.kv("GQA", "2 · n_kv_heads · head_dim · bytes"),
          TQ.kv("MQA", "2 · 1 · head_dim · bytes"),
          TQ.kv("MLA", "(d_c + d_rope) · bytes")
        ),
        TQ.note(
          "Note the MLA formula has NO factor of 2 and NO n_heads: the single latent c reconstructs " +
          "BOTH K and V for ALL heads, and the rope key is added once."
        )
      );

      // ---- fixed model config (matches L8 realistic defaults) ----
      var CFG = { nHeads: 32, headDim: 128, bytes: 2, nLayers: 60, seqLen: 8192 };

      // ---- live MLA + GQA knobs ----
      var dcVal = 512;
      var dropeVal = 64;
      var groupsVal = 4; // GQA groups -> n_kv_heads = nHeads / groups
      var axisMode = "perTok"; // or "total"

      function nKvHeads() {
        var k = Math.round(CFG.nHeads / groupsVal);
        return TQ.clamp(k, 1, CFG.nHeads);
      }

      // cache bytes per token per layer for each mode (REAL arithmetic)
      function bytesPerTok() {
        var mha = 2 * CFG.nHeads * CFG.headDim * CFG.bytes;
        var gqa = 2 * nKvHeads() * CFG.headDim * CFG.bytes;
        var mqa = 2 * 1 * CFG.headDim * CFG.bytes;
        var mla = (dcVal + dropeVal) * CFG.bytes;
        return { mha: mha, gqa: gqa, mqa: mqa, mla: mla };
      }

      function scaleFactor() {
        return axisMode === "total" ? (CFG.nLayers * CFG.seqLen) : 1;
      }

      // human-readable byte formatter
      function fmtBytes(v) {
        if (!isFinite(v)) return "—";
        if (v >= 1e9) return TQ.fmt(v / 1e9, 2) + " GB";
        if (v >= 1e6) return TQ.fmt(v / 1e6, 2) + " MB";
        if (v >= 1e3) return TQ.fmt(v / 1e3, 2) + " KB";
        return TQ.fmt(v, 0) + " B";
      }

      var barHolder = TQ.el("div", { class: "tq-grow" });
      var savingsHolder = TQ.el("div", {});

      function rebuild() {
        var per = bytesPerTok();
        var sf = scaleFactor();
        var vals = [per.mha * sf, per.gqa * sf, per.mqa * sf, per.mla * sf];

        barHolder.innerHTML = "";
        barHolder.appendChild(TQ.barRow(vals, {
          labels: ["MHA", "GQA-" + nKvHeads() + "kv", "MQA", "MLA"],
          max: vals[0], // MHA is the reference top of the axis
          format: fmtBytes,
          highlight: 3
        }));

        // savings callout (real ratios)
        var savVsMha = (1 - per.mla / per.mha) * 100;
        var savVsGqa = (1 - per.mla / per.gqa) * 100;
        var ratioMha = per.mha / per.mla;
        savingsHolder.innerHTML = "";
        savingsHolder.appendChild(TQ.callout(
          TQ.el("div", { class: "tq-flexcol", style: { gap: "4px" } },
            TQ.el("div", {},
              TQ.el("strong", { text: TQ.fmt(savVsMha, 1) + "% less than MHA" }),
              TQ.el("span", { class: "tq-mla-mute", text:
                "  (" + fmtBytes(per.mla * sf) + " vs " + fmtBytes(per.mha * sf) +
                " · " + TQ.fmt(ratioMha, 1) + "× smaller)" })
            ),
            TQ.el("div", {},
              (savVsGqa >= 0
                ? TQ.el("span", { text: TQ.fmt(savVsGqa, 1) + "% less than GQA-" + nKvHeads() + "kv too" })
                : TQ.el("span", { text: TQ.fmt(-savVsGqa, 1) + "% MORE than GQA-" + nKvHeads() + "kv (raise groups or lower d_c)" }))
            )
          )
        ));
      }

      // ---- controls ----
      var dcSlider = TQ.slider({
        min: 32, max: 1024, step: 32, value: dcVal, label: "d_c (latent dim)",
        format: function (v) { return TQ.fmt(v, 0); },
        onInput: function (v) { dcVal = v; rebuild(); }
      });
      var dropeSlider = TQ.slider({
        min: 0, max: 128, step: 8, value: dropeVal, label: "d_rope (decoupled rope key)",
        format: function (v) { return TQ.fmt(v, 0); },
        onInput: function (v) { dropeVal = v; rebuild(); }
      });
      var groupsSlider = TQ.slider({
        min: 1, max: 32, step: 1, value: groupsVal, label: "GQA groups (→ n_kv_heads = n_heads / groups)",
        format: function (v) { return TQ.fmt(v, 0) + " (" + Math.max(1, Math.round(CFG.nHeads / v)) + " kv)"; },
        onInput: function (v) { groupsVal = v; rebuild(); }
      });
      var axisSeg = TQ.segmented({
        options: [
          { label: "per token / layer", value: "perTok" },
          { label: "total cache (× layers × seq)", value: "total" }
        ],
        value: axisMode,
        onChange: function (v) { axisMode = v; rebuild(); }
      });

      var cfgChips = TQ.el("div", { class: "tq-mla-cfg" },
        TQ.kv("n_heads", String(CFG.nHeads)),
        TQ.kv("head_dim", String(CFG.headDim)),
        TQ.kv("bytes/elem", CFG.bytes + " (fp16)"),
        TQ.kv("n_layers", String(CFG.nLayers)),
        TQ.kv("seq_len", String(CFG.seqLen))
      );

      calcBlock.appendChild(cfgChips);
      calcBlock.appendChild(TQ.el("div", { class: "tq-controls-row tq-mla-controls" },
        dcSlider.el, dropeSlider.el, groupsSlider.el
      ));
      calcBlock.appendChild(TQ.el("div", { class: "tq-flexrow", style: { margin: "10px 0", alignItems: "center", gap: "10px" } },
        TQ.el("span", { class: "tq-slider-label", text: "axis" }), axisSeg.el
      ));
      calcBlock.appendChild(TQ.el("div", { class: "tq-panel" }, barHolder));
      calcBlock.appendChild(savingsHolder);
      calcBlock.appendChild(TQ.note(
        "Slide d_c down and MLA shrinks below MQA; slide it up and MLA rises toward GQA — that's the " +
        "rank-k fidelity knob you know from SVD. d_rope adds a small fixed floor that doesn't depend on n_heads."
      ));
      rebuild();
      root.appendChild(calcBlock);

      /* ============================================================== *
       *  WRAP-UP
       * ============================================================== */
      root.appendChild(TQ.block(
        TQ.h(2, "The emotional payoff"),
        TQ.p(
          "MQA < GQA < MHA was the L8 story. MLA drops ", TQ.el("strong", { text: "below GQA" }),
          " (and below MQA at small ", TQ.math("d_c"), ") while — per the design intent — keeping ",
          TQ.el("strong", { text: "MHA-level expressivity" }),
          ". GQA/MQA force heads to literally share K/V (a real quality hit); MLA reconstructs each head's ",
          "K/V from a shared latent through ", TQ.el("em", { text: "distinct" }),
          " up-projections, so heads stay individualized while only the small latent is cached."
        ),
        TQ.callout(
          "Near-lowest cache (far below MHA and GQA; comparable to MQA) AND full quality at once — that " +
          "juxtaposition is why DeepSeek-V2/V3 ship MLA. MQA gets slightly lower cache but sacrifices quality; " +
          "MLA keeps every head. It's not magic: it's a low-rank factorization of the KV projection with " +
          "position carried separately."
        )
      ));

      /* scoped styles — colors come only from CSS variables (no hardcoded hex) */
      injectOnce("tq-lvl09-css",
        ".tq-mla-cols{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;margin:6px 0}" +
        ".tq-mla-cols.is-ghost{opacity:.28;filter:grayscale(1)}" +
        ".tq-mla-headcol,.tq-mla-latentcol,.tq-mla-ropecol{display:flex;flex-direction:column;gap:6px;" +
          "align-items:flex-start;background:var(--panel-hi);border:1px solid var(--line);" +
          "border-radius:10px;padding:10px 12px}" +
        ".tq-mla-latentcol{border-color:var(--accent);box-shadow:var(--glow)}" +
        ".tq-mla-ropecol{border-style:dashed}" +
        ".tq-mla-ropecol.is-live{border-color:var(--info)}" +
        ".tq-mla-headcap{font-size:12px;font-weight:700;color:var(--ink-soft);font-family:var(--mono)}" +
        ".tq-mla-dim{font-size:10px;color:var(--ink-mute);font-family:var(--mono)}" +
        ".tq-mla-merge{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:6px 0}" +
        ".tq-mla-arrow{font-family:var(--mono);font-size:12px;color:var(--ink-mute);white-space:nowrap}" +
        ".tq-mla-formulas{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}" +
        ".tq-mla-cfg{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 12px}" +
        ".tq-mla-controls{align-items:flex-start;gap:18px;flex-wrap:wrap}" +
        ".tq-mla-mute{color:var(--ink-mute);font-size:12px}");
    },

    quiz: [
      {
        q: "In MLA, what is actually written to the KV cache for each token (per layer)?",
        choices: [
          "The full per-head K and V for every head, like MHA",
          "One shared latent vector c (plus a small decoupled rope key) — per-head K/V are reconstructed from it at use time",
          "Only the query vectors Q for every head",
          "The averaged K/V across all heads, like MQA"
        ],
        answer: 1,
        explain: "MLA caches the small latent c and a small rope key. Each head's K and V are up-projected " +
                 "from c (K_i = c·W_UK_i, V_i = c·W_UV_i) at attention time — they are never stored."
      },
      {
        q: "Why does MLA keep a separate 'decoupled' RoPE key instead of compressing position into the latent c?",
        choices: [
          "RoPE keys are cheaper to compute than content keys",
          "Rotary position is applied before the dot product and depends on absolute position; baking it into the cached latent would break the low-rank factorization, so a small position-carrying key rides alongside c",
          "DeepSeek wanted a larger cache for accuracy",
          "Because RoPE only works on the value vectors, not the keys"
        ],
        answer: 1,
        explain: "RoPE's position-dependent rotation can't be absorbed into the up-projection if it's frozen " +
                 "inside the cached latent. Splitting off a small rope key keeps c position-free (fully " +
                 "compressible) while preserving relative-position info."
      },
      {
        q: "Using fp16 (2 bytes), n_heads=32, head_dim=128, MHA caches 2×32×128×2 = 16384 B per token per layer. With d_c=512 and d_rope=64, what does MLA cache, and roughly what saving?",
        choices: [
          "2×512×2 = 2048 B, ~88% saving",
          "(512+64)×2 = 1152 B, ~93% saving",
          "(512+64)×2×32 = 36864 B, it's actually larger",
          "512×64×2 = 65536 B, ~no saving"
        ],
        answer: 1,
        explain: "MLA caches (d_c + d_rope) × bytes = (512+64)×2 = 1152 B — no factor of 2 and no n_heads " +
                 "factor, since the single latent reconstructs both K and V for all heads. That's ~93% less " +
                 "than MHA's 16384 B."
      },
      {
        q: "The big claim is 'MLA matches MHA quality while caching less than GQA.' Which mechanism makes the quality part plausible?",
        choices: [
          "It uses more attention heads than MHA",
          "Each head's K/V is reconstructed from c via its own learned up-projection, so heads stay distinct and expressive — unlike GQA/MQA which force heads to literally share the same K/V",
          "It skips the softmax to save compute",
          "It caches Q as well as the latent"
        ],
        answer: 1,
        explain: "GQA/MQA reduce cache by making heads share identical K/V (a real expressivity hit). MLA " +
                 "instead reconstructs per-head K/V from a shared latent through distinct up-projections, so " +
                 "heads remain individualized while only the small latent is cached."
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
