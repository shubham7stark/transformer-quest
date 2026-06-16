/* ============================================================================
 * LEVEL 06 — The Transformer Block   (order 6)
 * ----------------------------------------------------------------------------
 * Assemble a full post-LN block and animate ONE toy token's 16-dim vector
 * through every sub-layer, every number computed live by TQ math:
 *
 *   z   = LayerNorm(x + MHA(x))
 *   out = LayerNorm(z + FFN(z))      FFN(z) = max(0, zW1+b1)W2 + b2
 *
 * Pattern (copied from 01-tokens-embeddings.js):
 *   - one IIFE, ONE TQ.registerLevel call, NO other globals
 *   - short explanation blocks + a PRIMARY interactive (TQ.stepper) walkthrough
 *   - all color through TQ.colorFor / TQ.vectorView / TQ.barRow (one language)
 *   - 4 quiz questions
 *
 * All quantities are genuinely computed:
 *   qkv = TQ.toyQKV(16, 101, 202, 303); att = TQ.attention(qkv.Q,qkv.K,qkv.V)
 *   x = embeddings[p]; a = att.output[p] (blend of all 6 V rows via att.weights[p])
 *   r1 = x + a; z = LayerNorm(r1) -> mean=0, var=1 (computed, not asserted)
 *   FFN: W1=randMatrix(16,32,711), W2=randMatrix(32,16,713); h = relu(zW1+b1)
 *   r2 = z + ff; out = LayerNorm(r2) -> mean=0, var=1; out is 16-dim like x.
 * ========================================================================== */

(function () {
  "use strict";

  TQ.registerLevel({
    id: "transformer-block",
    order: 6,
    title: "The Transformer Block",
    icon: "🧱",
    tagline: "Assemble it: Attention + Add&Norm + Feed-Forward + Add&Norm, stacked.",
    preread: "Illustrated Transformer (capstone)",
    objectives: [
      "Assemble a full block: Multi-Head Attention → residual+LayerNorm → FFN → residual+LayerNorm",
      "Map residual connections onto your CNN skip-connection (ResNet) intuition",
      "Understand why we LayerNorm (stable activation scale → trainable deep stacks)"
    ],

    render: function (root) {
      var toy = TQ.toy;
      var tokens = toy.tokens;
      var emb = toy.embeddings;
      var d = toy.dModel;        // 16
      var n = tokens.length;     // 6
      var dHidden = 32;          // 2x expansion here for legibility (real models use ~4x; same shape lesson)

      /* ---- fixed, deterministic block weights (seeds frozen) ------------- */
      var qkv = TQ.toyQKV(d, 101, 202, 303);
      var att = TQ.attention(qkv.Q, qkv.K, qkv.V); // scaled dot-product, /sqrt(16)=/4
      // We project the attention output (dk=16) straight back as the MHA result
      // (single conceptual head here; W_O = identity-shaped). att.output is n x 16.

      var W1 = TQ.randMatrix(d, dHidden, 711);
      var W2 = TQ.randMatrix(dHidden, d, 713);
      var b1row = TQ.randMatrix(1, dHidden, 715, 0.3)[0];
      var b2row = TQ.randMatrix(1, d, 717, 0.3)[0];

      function addVec(u, v) {
        var out = [];
        for (var i = 0; i < u.length; i++) out.push(u[i] + v[i]);
        return out;
      }
      function variance(vec) {
        var mu = TQ.mean(vec);
        var s = 0;
        for (var i = 0; i < vec.length; i++) s += (vec[i] - mu) * (vec[i] - mu);
        return s / vec.length;
      }

      // Run the WHOLE block for a single position p; return every intermediate.
      function runBlock(p) {
        var x = emb[p].slice();
        var a = att.output[p].slice();           // MHA output for this token
        var r1 = addVec(x, a);                    // residual #1 (x + MHA)
        var z = TQ.layerNorm(r1);                 // LayerNorm #1

        var hRaw = addVec(TQ.matmul([z], W1)[0], b1row); // zW1 + b1
        var h = TQ.reluVec(hRaw);                 // ReLU
        var ff = addVec(TQ.matmul([h], W2)[0], b2row);   // hW2 + b2
        var r2 = addVec(z, ff);                   // residual #2 (z + FFN)
        var out = TQ.layerNorm(r2);               // LayerNorm #2

        return {
          x: x, a: a, r1: r1, z: z,
          hRaw: hRaw, h: h, ff: ff, r2: r2, out: out
        };
      }

      /* ----------------------------------------------------- intro narrative */
      root.appendChild(TQ.block(
        TQ.h(2, "The block is a loop body — and you've already debugged its twin"),
        TQ.p(
          "A transformer is a stack of identical ", TQ.el("strong", { text: "blocks" }),
          " (GPT-2 had 12–48; modern LLMs run 30–120). One block does exactly two things, in order: ",
          "(1) ", TQ.el("strong", { text: "mix tokens together" }), " with attention, then ",
          "(2) ", TQ.el("strong", { text: "think about each token individually" }), " with a small MLP. ",
          "Wrapped around each is the move you already trust from ResNet: a ",
          TQ.el("strong", { text: "skip connection" }), "."
        ),
        TQ.p(
          "The post-LN block computes ",
          TQ.math("z = LayerNorm(x + MHA(x))"), ", then ",
          TQ.math("out = LayerNorm(z + FFN(z))"), ". Read those ",
          TQ.math("x +"), " and ", TQ.math("z +"),
          " literally — they are residual connections. The sub-layer doesn't have to reproduce its " +
          "input; it only learns a ", TQ.el("strong", { text: "delta" }), " to add. Identity is free, so " +
          "gradients have a clean highway down a 100-layer stack and the thing actually trains. Without " +
          "residuals, deep transformers degrade and die exactly like deep plain CNNs did."
        ),
        TQ.callout(
          "CNN bridge: residual = skip connection (ResNet, 2015). The block learns x + (a small correction), " +
          "not x replaced. That's why gradients reach the bottom of a very deep stack."
        )
      ));

      root.appendChild(TQ.block(
        TQ.h(2, "LayerNorm: the same fix as BatchNorm, but per-token so it survives inference"),
        TQ.p(
          "You normalized activations in CNNs with ", TQ.el("strong", { text: "BatchNorm" }),
          " — subtract the mean, divide by the std, restore a learnable scale/shift — to keep activation " +
          "scale stable so deep stacks stay trainable. ", TQ.el("strong", { text: "LayerNorm" }),
          " is the same idea with one critical change of axis."
        ),
        TQ.p(
          "BatchNorm normalizes each feature ", TQ.el("strong", { text: "across the batch" }),
          " (it leans on other examples). LayerNorm normalizes each token's vector ",
          TQ.el("strong", { text: "across its own features" }),
          " — no cross-token, no cross-batch dependency. That matters because an LLM generates one token " +
          "at a time with a batch of one and variable sequence length; a stat that depends on neighbors " +
          "would be poison at inference."
        ),
        TQ.p(
          "So for a single ", TQ.math("d_model = " + d), " vector, LayerNorm forces those ", String(d),
          " numbers to have ", TQ.el("strong", { text: "mean 0 and variance 1" }),
          " (then applies learned gain γ and bias β, here 1 and 0). The residual sum ",
          TQ.math("x + sublayer(x)"), " can blow up the scale; LayerNorm immediately tames it back. ",
          "In the walkthrough below you'll watch the mean and variance go from messy to exactly 0 and 1 — ",
          TQ.el("strong", { text: "computed, not asserted" }), "."
        )
      ));

      root.appendChild(TQ.block(
        TQ.h(2, "The FFN is a per-token MLP — and it's where most of the parameters live"),
        TQ.p(
          "After attention lets tokens look at each other, the feed-forward network processes each token " +
          "independently: ", TQ.math("FFN(z) = max(0, zW1 + b1)W2 + b2"),
          ". It's a two-layer MLP — Linear, ReLU, Linear — and you've built a thousand of these."
        ),
        TQ.p(
          "Two things make it transformer-flavored. First, it's ", TQ.el("strong", { text: "position-wise" }),
          ": the SAME W1, W2 are applied at every position, like a 1×1 convolution sliding over the sequence " +
          "(shared weights, applied independently per location). Attention is the only place tokens talk; the " +
          "FFN is purely local per-token computation. Second, it ",
          TQ.el("strong", { text: "expands then contracts" }),
          " — the hidden layer is typically ~4× d_model (GPT-2/GPT-3 are exactly 4×; Llama-2-7B is " +
          "4096→11008, ~2.7×, because its SwiGLU trims the hidden size to pay for an extra gate matrix), " +
          "so despite being 'just an MLP' it holds roughly two-thirds of a transformer's parameters. Attention ",
          TQ.el("strong", { text: "routes" }), " information; the FFN is where the model ",
          TQ.el("strong", { text: "stores and transforms" }), " what it knows."
        ),
        TQ.note(
          "Modern models often swap ReLU for GELU/SwiGLU and put LayerNorm BEFORE each sub-layer " +
          "('pre-LN') for steadier gradients. We use the original post-LN here so the ordering is concrete."
        )
      ));

      /* ====================================================================
       *  PRIMARY INTERACTIVE: stepper walkthrough of one token's vector
       * ================================================================== */
      var primary = TQ.block(
        TQ.h(2, "Watch one token flow through the whole block"),
        TQ.p(
          "Pick a token, then step through all 7 stages. Every cell color is its value on the shared scale; " +
          "flip ", TQ.el("strong", { text: "show raw numbers" }), " to read the components. ",
          "Only ", TQ.el("strong", { text: "one" }),
          " stage uses the other tokens (attention) — everything else is per-token."
        )
      );

      // legend (matches level 01)
      primary.appendChild(TQ.el("div", { class: "tq-legend" },
        TQ.el("span", { text: "low" }),
        TQ.el("div", { class: "tq-legend-scale" }),
        TQ.el("span", { text: "high" })
      ));

      // ---- shared control state -----------------------------------------
      var pos = 1;          // default 'cat' (#1)
      var showRaw = false;
      var depth = 1;        // stack-depth caption control
      var data = runBlock(pos);

      var posSel = TQ.segmented({
        options: tokens.map(function (t, i) { return { label: t, value: i }; }),
        value: pos,
        onChange: function (v) { pos = v; data = runBlock(pos); rebuildStepper(); }
      });

      var rawToggle = TQ.toggle({
        label: "show raw numbers",
        value: showRaw,
        onChange: function (v) { showRaw = v; rebuildStepper(); }
      });

      var depthSel = TQ.segmented({
        options: [{ label: "1 block", value: 1 }, { label: "6 blocks", value: 6 }, { label: "48 blocks", value: 48 }],
        value: depth,
        onChange: function (v) { depth = v; updateDepthCaption(); }
      });

      var depthCaption = TQ.note("");
      function updateDepthCaption() {
        depthCaption.innerHTML = "";
        var msg;
        if (depth === 1) {
          msg = "1 block. You're watching this single block in full. Real models stack many of these.";
        } else {
          msg = depth + " blocks stacked. The output below (16-dim, same shape as the input) becomes the " +
            "input to the next block, " + depth + " times over. Crucially the weights are NOT shared — each " +
            "block has its own attention and FFN parameters. Same shape, different transformation, repeated.";
        }
        depthCaption.appendChild(TQ.el("span", { text: msg }));
      }
      updateDepthCaption();

      var controls = TQ.el("div", { class: "tq-controls-row tq-block-controls" },
        TQ.el("div", {},
          TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "Token"),
          posSel.el),
        TQ.el("div", {},
          TQ.el("div", { class: "tq-slider-label", style: { marginBottom: "6px" } }, "Stack depth"),
          depthSel.el),
        TQ.el("div", { style: { alignSelf: "flex-end" } }, rawToggle.el)
      );
      primary.appendChild(controls);

      // ---- small rendering helpers for stages ---------------------------
      function vv(values, label) {
        return TQ.vectorView(values, { label: label, cellSize: 22, showValues: showRaw });
      }

      function statPanel(title, vec) {
        var mu = TQ.mean(vec);
        var vr = variance(vec);
        return TQ.el("div", { class: "tq-tb-stats" },
          TQ.el("span", { class: "tq-tb-stats-title", text: title }),
          TQ.kv("mean", TQ.fmt(mu, 2)),
          TQ.kv("var", TQ.fmt(vr, 2)),
          TQ.kv("min", TQ.fmt(TQ.minOf(vec), 2)),
          TQ.kv("max", TQ.fmt(TQ.maxOf(vec), 2))
        );
      }

      function residualView(label1, v1, label2, v2, sumLabel, sum) {
        return TQ.el("div", { class: "tq-tb-residual" },
          vv(v1, label1),
          TQ.el("div", { class: "tq-tb-plus", text: "+" }),
          vv(v2, label2),
          TQ.el("div", { class: "tq-tb-eq", text: "=" }),
          vv(sum, sumLabel)
        );
      }

      // ---- the 7 stages -------------------------------------------------
      function buildSteps() {
        return [
          {
            label: "1 · Input x",
            run: function (c) {
              c.appendChild(TQ.p(
                "The block's input for token ", TQ.el("strong", { text: tokens[pos] }),
                " — its ", TQ.math("d_model = " + d), " vector. (In a real stack this is the previous " +
                "block's output; here it's the embedding.)"
              ));
              c.appendChild(vv(data.x, "x  (" + tokens[pos] + ")"));
              c.appendChild(statPanel("input stats", data.x));
            }
          },
          {
            label: "2 · Multi-head attention output",
            run: function (c) {
              c.appendChild(TQ.p(
                "The ", TQ.el("strong", { text: "only" }),
                " cross-token step. Token ", TQ.el("strong", { text: tokens[pos] }),
                " blends the value vectors of all ", String(n),
                " tokens, weighted by attention. Those weights are real (scaled dot-product, ÷√" + d + "):"
              ));
              c.appendChild(TQ.barRow(att.weights[pos], {
                labels: tokens, max: 1,
                format: function (v) { return TQ.fmt(v, 3); }
              }));
              c.appendChild(vv(data.a, "MHA(x)"));
              c.appendChild(TQ.note(
                "For legibility this block runs a single full-width head (dk = d_model, Wo = identity); the " +
                "Concat-then-Wo machinery from Level 4 (dk = d_model / H, then a learned output projection) would " +
                "slot in here unchanged and produce the same d_model-wide output."
              ));
              c.appendChild(TQ.note(
                "Every later stage is per-token: it never looks at the other five tokens again."
              ));
            }
          },
          {
            label: "3 · Residual #1  (x + MHA)",
            run: function (c) {
              c.appendChild(TQ.p(
                "The skip connection. We ", TQ.el("strong", { text: "add" }),
                " the attention output back onto the input — the sum is ", TQ.el("strong", { text: "x nudged" }),
                ", not x replaced. The sub-layer only had to learn a delta (the ResNet move)."
              ));
              c.appendChild(residualView("x", data.x, "MHA(x)", data.a, "x + MHA(x)", data.r1));
              c.appendChild(statPanel("after residual (scale can drift)", data.r1));
            }
          },
          {
            label: "4 · LayerNorm #1",
            run: function (c) {
              c.appendChild(TQ.p(
                "Normalize across the token's own ", String(d),
                " features: subtract the mean, divide by the std, apply γ=1, β=0. Watch the stats ",
                TQ.el("strong", { text: "snap" }), " to mean 0, var 1 — computed from the numbers, not asserted."
              ));
              c.appendChild(TQ.el("div", { class: "tq-tb-residual" },
                vv(data.r1, "x + MHA(x)"),
                TQ.el("div", { class: "tq-tb-eq", text: "→" }),
                vv(data.z, "z = LN(...)")
              ));
              c.appendChild(TQ.el("div", { class: "tq-tb-statrow" },
                statPanel("before LN", data.r1),
                statPanel("after LN", data.z)
              ));
            }
          },
          {
            label: "5 · FFN internals  (Linear → ReLU → Linear)",
            run: function (c) {
              c.appendChild(TQ.p(
                "Per-token MLP: expand ", TQ.math(d + " → " + dHidden),
                ", ReLU (clamps negatives to exactly 0), then contract ",
                TQ.math(dHidden + " → " + d),
                ". The SAME W1, W2 act at every position — like a 1×1 conv."
              ));
              c.appendChild(vv(data.hRaw, "zW1 + b1   (hidden, " + dHidden + "-dim, pre-ReLU)"));
              c.appendChild(vv(data.h, "max(0, ·)   (after ReLU — negatives now 0)"));
              var zeroed = 0;
              for (var i = 0; i < data.hRaw.length; i++) if (data.hRaw[i] <= 0) zeroed++;
              c.appendChild(TQ.el("div", { class: "tq-tb-stats" },
                TQ.el("span", { class: "tq-tb-stats-title", text: "ReLU" }),
                TQ.kv("hidden dim", String(dHidden)),
                TQ.kv("clamped to 0", zeroed + " / " + dHidden)
              ));
              c.appendChild(vv(data.ff, "FFN(z) = hW2 + b2   (back to " + d + "-dim)"));
            }
          },
          {
            label: "6 · Residual #2  (z + FFN)",
            run: function (c) {
              c.appendChild(TQ.p(
                "Second skip connection, same pattern. ", TQ.math("z"),
                " plus the FFN's delta — the model adds what the per-token MLP computed without throwing z away."
              ));
              c.appendChild(residualView("z", data.z, "FFN(z)", data.ff, "z + FFN(z)", data.r2));
              c.appendChild(statPanel("after residual (scale can drift again)", data.r2));
            }
          },
          {
            label: "7 · LayerNorm #2  →  block output",
            run: function (c) {
              c.appendChild(TQ.p(
                "Final normalize. The output is ", TQ.el("strong", { text: d + "-dimensional" }),
                " — the ", TQ.el("strong", { text: "same shape as the input x" }),
                ". That's exactly why blocks stack: out of one block is in to the next."
              ));
              c.appendChild(TQ.el("div", { class: "tq-tb-residual" },
                vv(data.r2, "z + FFN(z)"),
                TQ.el("div", { class: "tq-tb-eq", text: "→" }),
                vv(data.out, "out = LN(...)")
              ));
              c.appendChild(TQ.el("div", { class: "tq-tb-statrow" },
                statPanel("before LN", data.r2),
                statPanel("after LN", data.out)
              ));
              c.appendChild(TQ.el("div", { class: "tq-tb-shapecheck" },
                TQ.badge("in: " + d + "-dim", "info"),
                TQ.el("span", { class: "tq-tb-arrow", text: "→ block →" }),
                TQ.badge("out: " + data.out.length + "-dim", "good")
              ));
              c.appendChild(depthCaption);
            }
          }
        ];
      }

      var stepperHolder = TQ.el("div", { class: "tq-tb-stepper-holder" });
      function rebuildStepper() {
        stepperHolder.innerHTML = "";
        stepperHolder.appendChild(TQ.stepper(buildSteps()));
      }
      rebuildStepper();
      primary.appendChild(stepperHolder);

      primary.appendChild(TQ.callout(
        "The load-bearing split: attention is the ONLY place tokens exchange information; LayerNorm and the " +
        "FFN both operate per-token. Mix, then think locally — repeated, with fresh weights, dozens of times."
      ));
      root.appendChild(primary);

      /* --------------------------------------------------- wrap-up takeaway */
      root.appendChild(TQ.block(
        TQ.h(2, "Why this is the whole architecture"),
        TQ.p(
          "Stack this block N times, add embeddings at the bottom and an output head on top, and you have a " +
          "transformer. Residuals keep the gradient highway open; LayerNorm keeps the scale sane; attention " +
          "routes; the FFN remembers. Everything you learned earlier — embeddings, dot-product attention — " +
          "plugs in right here."
        ),
        TQ.callout(
          "Carry forward: the block preserves shape (16-dim in, 16-dim out). Shape-preservation is what makes " +
          "a deep stack possible — and the KV-cache tricks coming next live inside that attention sub-layer."
        )
      ));

      /* ----------------------------------- encoder vs decoder vs decoder-only */
      root.appendChild(TQ.block(
        TQ.h(2, "Two flavors of block: encoder vs decoder (and why we go decoder-only)"),
        TQ.p(
          "The block you just assembled comes in two flavors. An ", TQ.el("strong", { text: "encoder block" }),
          " uses ", TQ.el("strong", { text: "bidirectional" }), " self-attention — no mask, every token may look both " +
          "left and right — so it builds a rich representation of an entire input (great for understanding/classification, " +
          "e.g. BERT). A ", TQ.el("strong", { text: "decoder block" }), " uses ",
          TQ.el("strong", { text: "masked" }), " self-attention — each token may attend only to itself and earlier " +
          "tokens — so it can generate left-to-right without peeking at the future."
        ),
        TQ.p(
          "The original Transformer was an ", TQ.el("strong", { text: "encoder–decoder" }),
          " (built for translation): the encoder reads the source sentence, and each decoder block adds a third " +
          "sub-layer — ", TQ.el("strong", { text: "cross-attention" }), " — where the ",
          TQ.el("strong", { text: "queries come from the decoder" }), " but the ",
          TQ.el("strong", { text: "keys and values come from the encoder's output" }),
          ". That's the only structural difference from the self-attention you already know: Q from one sequence, K/V from another."
        ),
        TQ.el("div", { class: "tq-encdec-diagram", "aria-hidden": "true" },
          TQ.el("div", { class: "tq-encdec-col" },
            TQ.el("div", { class: "tq-encdec-cap", text: "Encoder block" }),
            TQ.el("div", { class: "tq-encdec-sub", text: "self-attn (bidirectional, no mask)" }),
            TQ.el("div", { class: "tq-encdec-sub", text: "Add & Norm" }),
            TQ.el("div", { class: "tq-encdec-sub", text: "FFN" }),
            TQ.el("div", { class: "tq-encdec-sub", text: "Add & Norm" })
          ),
          TQ.el("div", { class: "tq-encdec-arrow", text: "K,V →" }),
          TQ.el("div", { class: "tq-encdec-col is-dec" },
            TQ.el("div", { class: "tq-encdec-cap", text: "Decoder block" }),
            TQ.el("div", { class: "tq-encdec-sub", text: "self-attn (MASKED — past only)" }),
            TQ.el("div", { class: "tq-encdec-sub", text: "Add & Norm" }),
            TQ.el("div", { class: "tq-encdec-sub is-cross", text: "cross-attn (Q from here, K/V from encoder)" }),
            TQ.el("div", { class: "tq-encdec-sub", text: "Add & Norm" }),
            TQ.el("div", { class: "tq-encdec-sub", text: "FFN" }),
            TQ.el("div", { class: "tq-encdec-sub", text: "Add & Norm" })
          )
        ),
        TQ.callout(
          "Modern LLMs (GPT, Llama, DeepSeek) are ", TQ.el("strong", { text: "decoder-only" }),
          ": they drop the encoder and its cross-attention entirely and just stack masked-self-attention decoder " +
          "blocks. That's why the rest of this game lives in the decoder world — when Level 7 says a 'decoder generates " +
          "left-to-right,' this masked block is exactly what it means."
        )
      ));

      /* scoped styles — colors via CSS variables only, no hardcoded hex */
      injectOnce("tq-lvl06-css",
        ".tq-encdec-diagram{display:flex;align-items:flex-start;gap:14px;flex-wrap:wrap;margin:14px 0 4px}" +
        ".tq-encdec-col{flex:1 1 220px;min-width:200px;border:1px solid var(--line);border-radius:12px;" +
          "padding:12px;background:var(--panel-hi)}" +
        ".tq-encdec-col.is-dec{border-color:var(--accent-2)}" +
        ".tq-encdec-cap{font-size:13px;font-weight:800;color:var(--ink);margin-bottom:8px;" +
          "text-transform:uppercase;letter-spacing:.04em}" +
        ".tq-encdec-sub{font-size:12.5px;color:var(--ink-mute);font-family:var(--mono);" +
          "border:1px solid var(--line);border-radius:8px;padding:6px 9px;margin:6px 0;background:var(--panel)}" +
        ".tq-encdec-sub.is-cross{color:var(--accent-2);border-color:var(--accent-2)}" +
        ".tq-encdec-arrow{align-self:center;font-family:var(--mono);font-size:13px;color:var(--accent-2);" +
          "font-weight:700;white-space:nowrap}" +
        ".tq-block-controls{align-items:flex-start;gap:22px;flex-wrap:wrap;margin:6px 0 14px}" +
        ".tq-tb-stepper-holder{margin-top:6px}" +
        ".tq-tb-residual{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:10px 0}" +
        ".tq-tb-plus,.tq-tb-eq{font-size:22px;font-weight:700;color:var(--ink-mute);padding:0 2px}" +
        ".tq-tb-statrow{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px}" +
        ".tq-tb-stats{display:flex;align-items:center;gap:8px;flex-wrap:wrap;" +
          "background:var(--panel-hi);border:1px solid var(--line);border-radius:10px;" +
          "padding:8px 12px;margin-top:10px}" +
        ".tq-tb-stats-title{font-size:12px;font-weight:700;color:var(--ink-mute);" +
          "text-transform:uppercase;letter-spacing:.04em;margin-right:4px}" +
        ".tq-tb-shapecheck{display:flex;align-items:center;gap:10px;margin-top:12px}" +
        ".tq-tb-arrow{font-family:var(--mono);color:var(--ink-mute);font-size:13px}");
    },

    quiz: [
      {
        q: "In the post-LN block out = LayerNorm(z + FFN(z)), what is the role of the `z +` term?",
        choices: [
          "It is a residual/skip connection — the sub-layer learns a delta added to its input, keeping gradients flowing in a deep stack (just like ResNet)",
          "It concatenates z onto the FFN output, doubling the vector's dimension",
          "It is the positional encoding being re-added at every layer",
          "It applies the softmax that turns FFN outputs into attention weights"
        ],
        answer: 0,
        explain: "`z +` is the residual/skip connection from ResNet. The FFN only learns a delta on top of the " +
                 "identity, so deep stacks stay trainable and gradients reach the bottom layers."
      },
      {
        q: "LayerNorm and BatchNorm both stabilize activation scale. What is the key difference that makes LayerNorm the right choice for an autoregressive LLM?",
        choices: [
          "LayerNorm has no learnable parameters, so it is cheaper",
          "LayerNorm normalizes across each token's own features, with no dependence on other tokens or the batch — so it behaves identically at training and single-token inference",
          "LayerNorm normalizes across the batch dimension, averaging over many sentences",
          "LayerNorm replaces the need for an activation function inside the FFN"
        ],
        answer: 1,
        explain: "BatchNorm's statistics depend on the batch/neighbors, which breaks down at one-token-at-a-time " +
                 "inference. LayerNorm normalizes within each token's feature vector, so it is independent of " +
                 "sequence length and batch size."
      },
      {
        q: "Within one transformer block, where do different token positions exchange information?",
        choices: [
          "In the feed-forward network, which mixes all positions together",
          "In LayerNorm, which averages across the sequence",
          "Only in the multi-head attention sub-layer; the FFN and LayerNorm operate per-token independently",
          "Nowhere — tokens never interact inside a block"
        ],
        answer: 2,
        explain: "Attention is the only cross-token operation. The FFN applies the same weights to each position " +
                 "independently (like a 1×1 conv), and LayerNorm normalizes each token's own vector — neither " +
                 "looks at other positions."
      },
      {
        q: "The FFN is FFN(z) = max(0, zW1 + b1)W2 + b2 with a hidden layer typically ~4× d_model. Why does this 'just an MLP' matter so much?",
        choices: [
          "It is where tokens attend to each other",
          "It performs the positional encoding",
          "Because of the 4× expansion it holds the majority of the model's parameters — it's where the model stores and transforms what it knows, per token",
          "It guarantees the output has mean 0 and variance 1"
        ],
        answer: 2,
        explain: "The 4× expand-then-contract MLP, applied at every position with shared weights, accounts for " +
                 "roughly two-thirds of a transformer's parameters. Attention routes information; the FFN is the " +
                 "per-token compute/storage."
      },
      {
        q: "What distinguishes a decoder block from an encoder block, and what extra sub-layer does an encoder–decoder's decoder add?",
        choices: [
          "The decoder uses bidirectional attention; it adds a second FFN to read the encoder output",
          "The decoder uses MASKED self-attention (each token sees only itself and earlier tokens); in an encoder–decoder it also adds a cross-attention sub-layer where Q comes from the decoder but K/V come from the encoder's output",
          "The decoder removes LayerNorm; it adds positional encoding that the encoder lacks",
          "There is no difference — encoder and decoder blocks are identical, the names are just conventions"
        ],
        answer: 1,
        explain: "An encoder block uses bidirectional (unmasked) self-attention; a decoder block masks attention so " +
                 "it can generate left-to-right. The original encoder–decoder also gives each decoder block a " +
                 "cross-attention sub-layer (Q from the decoder, K/V from the encoder). Modern LLMs are decoder-only: " +
                 "masked self-attention blocks with no encoder and no cross-attention."
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
