/* ============================================================================
 * TRANSFORMER QUEST — KERNEL
 * ----------------------------------------------------------------------------
 * window.TQ : math/visual utilities + the game engine.
 * Single file, plain vanilla JS (ES5/ES6). No frameworks, no build, no network.
 * Runs by double-clicking index.html from file://.
 *
 * Builders: every level is a separate IIFE that calls TQ.registerLevel(def)
 * EXACTLY ONCE and leaks NO other globals. Rely ONLY on the helpers documented
 * below — they are the stable contract.
 *
 * ============================================================================
 * MATH API  (deterministic — no Math.random anywhere; everything seed-based)
 * ----------------------------------------------------------------------------
 *  TQ.rng(seed)                      -> function()->float in [0,1)  (mulberry32)
 *  TQ.randMatrix(rows,cols,seed,scale=1) -> rows x cols 2D array in [-scale,scale]
 *  TQ.matmul(A,B)                    -> A·B   (A: m x n, B: n x p -> m x p)
 *  TQ.transpose(M)                   -> Mᵀ
 *  TQ.addMat(A,B)                    -> elementwise A+B (same shape)
 *  TQ.scaleMat(M,s)                  -> elementwise M*s
 *  TQ.dot(a,b)                       -> scalar dot product of two vectors
 *  TQ.softmax(arr)                   -> numerically-stable probability vector
 *  TQ.softmaxRows(M)                 -> softmax applied to each row of M
 *  TQ.attention(Q,K,V)               -> { scores, scaled, weights, output }
 *                                       scores  = Q·Kᵀ
 *                                       scaled  = scores / sqrt(dk)
 *                                       weights = softmaxRows(scaled)
 *                                       output  = weights·V
 *  TQ.cosine(a,b)                    -> cosine similarity in [-1,1]
 *  TQ.clamp(x,lo,hi)                 -> x clamped to [lo,hi]
 *  TQ.lerp(a,b,t)                    -> linear interpolation
 *  TQ.fmt(x,dp=2)                    -> number formatted to dp decimals (string)
 *  --- extras (documented, builders may use) ---
 *  TQ.norm(a)                        -> L2 norm (length) of a vector
 *  TQ.zeros(rows,cols)               -> rows x cols matrix of 0  (cols omitted -> vector)
 *  TQ.layerNorm(vec, gain=1, bias=0) -> mean/var normalized vector
 *  TQ.relu(x) / TQ.reluVec(v)        -> ReLU on scalar / vector
 *  TQ.sum(arr) / TQ.mean(arr) / TQ.maxOf(arr) / TQ.minOf(arr)
 *  TQ.range(n)                       -> [0,1,...,n-1]
 *  TQ.sinusoidalPE(nPos, dModel)     -> nPos x dModel sinusoidal positional matrix
 *
 * ============================================================================
 * SHARED TOY EXAMPLE  (continuity — the SAME sentence everywhere)
 * ----------------------------------------------------------------------------
 *  TQ.toy = {
 *    tokens:     ["The","cat","sat","on","the","mat"],
 *    dModel:     16,
 *    embeddings: tokens.length x dModel deterministic matrix (in ~[-1,1])
 *  }
 *  CONVENTION for deriving Q/K/V deterministically inside a level:
 *    var Wq = TQ.randMatrix(TQ.toy.dModel, dk, 101);   // query projection
 *    var Wk = TQ.randMatrix(TQ.toy.dModel, dk, 202);   // key   projection
 *    var Wv = TQ.randMatrix(TQ.toy.dModel, dk, 303);   // value projection
 *    var Q  = TQ.matmul(TQ.toy.embeddings, Wq); // etc.
 *  Pick distinct seeds per head (e.g. base + headIndex) for multi-head.
 *  Helper that bakes this convention: TQ.toyQKV(dk, seedQ, seedK, seedV).
 *  The exact seeds don't matter pedagogically; the SENTENCE staying constant does.
 *
 * ============================================================================
 * VISUAL HELPERS  (return real DOM nodes; styled by styles.css; dark theme)
 * ----------------------------------------------------------------------------
 *  TQ.el(tag, attrs, ...children)
 *      hyperscript. attrs may include {class,text,html,onclick,style,title,...}.
 *      style may be a string or an object. children may be nodes, strings,
 *      arrays, or null/false (ignored). Returns the element.
 *  TQ.colorFor(t)            t in [0,1] -> CSS color on the ONE shared scale
 *                            (low = cool/dark, high = bright/warm). Used by
 *                            heatmaps AND vectors so all visuals share a language.
 *  TQ.colorForSigned(x,max)  convenience: map x in [-max,max] -> colorFor via
 *                            normalization to [0,1].
 *  TQ.heatmap(matrix, opts)  opts: {rowLabels,colLabels,onCell,selectedRow,
 *                            selectedCol,cellSize,min,max,format,title}.
 *                            Cells colored via colorFor (auto-normalized to the
 *                            data range unless min/max given). onCell(r,c,value).
 *  TQ.barRow(values, opts)   opts:{labels,max,format,colorFn,highlight}.
 *                            Horizontal bars (one shared scale color by value).
 *  TQ.vectorView(values,opts) opts:{label,max,format,cellSize,showValues}.
 *                            A row of colored cells for ONE vector.
 *  TQ.matrixGrid(matrix,opts) opts:{rowLabels,colLabels,format,title}. A plain
 *                            numeric table (not heat-colored) for showing raw values.
 *  TQ.slider(opts)           opts:{min,max,step,value,label,format,onInput}.
 *                            -> {el, get(), set(v)}
 *  TQ.toggle(opts)           opts:{label,value,onChange} -> {el,get,set} (checkbox)
 *  TQ.segmented(opts)        opts:{options:[{label,value}],value,onChange}
 *                            -> {el,get,set}  (segmented button group)
 *  TQ.tabs(items)            items=[{label,render(container)}] -> element
 *  TQ.stepper(steps)         steps=[{label,run(container)}] -> element with
 *                            Prev/Next/Play controls (animated walkthroughs).
 *  TQ.canvasPanel(w,h,drawFn) -> {el, ctx, redraw()}  (drawFn(ctx,w,h))
 *  TQ.note(text|node)        styled aside (muted)
 *  TQ.callout(text|node)     styled aside (accented / "aha" box)
 *  TQ.math(text)             inline formula styling (monospace pill)
 *  TQ.block(...children)     a lesson section wrapper (.tq-block)
 *  TQ.h(level,text)          a heading node (level 2..4)
 *  TQ.p(...children)         a paragraph node (supports inline nodes)
 *  TQ.row(...children)/TQ.col(...children) fl: layout helpers
 *  TQ.badge(text,kind)       small pill label; kind in {default,good,warn,info}
 *  TQ.kv(label,value)        a small label:value stat chip
 *
 * ============================================================================
 * GAME ENGINE
 * ----------------------------------------------------------------------------
 *  TQ.registerLevel(def)     def: {id,order,title,icon,tagline,preread,
 *                                  objectives:[...], render(root), quiz:[...]}
 *                            quiz item: {q,choices:[...],answer:<idx>,explain}
 *  Engine boots on DOMContentLoaded: MAP screen (locked cards) -> LEVEL view
 *  (objectives -> render -> Check Understanding quiz) -> quiz gating awards XP,
 *  marks complete, unlocks next. Progress persists in localStorage(tq_progress)
 *  with graceful fallback when storage is blocked (file://). Reset control in UI.
 * ========================================================================== */

(function () {
  "use strict";

  var TQ = {};

  /* ------------------------------------------------------------------ *
   *  MATH
   * ------------------------------------------------------------------ */

  // mulberry32 PRNG — deterministic, seedable, good enough for toy data.
  TQ.rng = function (seed) {
    var a = (seed >>> 0) || 1;
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  TQ.randMatrix = function (rows, cols, seed, scale) {
    scale = (scale === undefined) ? 1 : scale;
    var r = TQ.rng(seed);
    var M = [];
    for (var i = 0; i < rows; i++) {
      var row = [];
      for (var j = 0; j < cols; j++) {
        row.push((r() * 2 - 1) * scale); // [-scale, scale)
      }
      M.push(row);
    }
    return M;
  };

  TQ.zeros = function (rows, cols) {
    if (cols === undefined) {
      var v = [];
      for (var i = 0; i < rows; i++) v.push(0);
      return v;
    }
    var M = [];
    for (var r = 0; r < rows; r++) {
      var row = [];
      for (var c = 0; c < cols; c++) row.push(0);
      M.push(row);
    }
    return M;
  };

  TQ.matmul = function (A, B) {
    var m = A.length, n = A[0].length, p = B[0].length;
    var out = [];
    for (var i = 0; i < m; i++) {
      var row = [];
      for (var j = 0; j < p; j++) {
        var s = 0;
        for (var k = 0; k < n; k++) s += A[i][k] * B[k][j];
        row.push(s);
      }
      out.push(row);
    }
    return out;
  };

  TQ.transpose = function (M) {
    var rows = M.length, cols = M[0].length;
    var out = [];
    for (var j = 0; j < cols; j++) {
      var row = [];
      for (var i = 0; i < rows; i++) row.push(M[i][j]);
      out.push(row);
    }
    return out;
  };

  TQ.addMat = function (A, B) {
    var out = [];
    for (var i = 0; i < A.length; i++) {
      var row = [];
      for (var j = 0; j < A[i].length; j++) row.push(A[i][j] + B[i][j]);
      out.push(row);
    }
    return out;
  };

  TQ.scaleMat = function (M, s) {
    var out = [];
    for (var i = 0; i < M.length; i++) {
      var row = [];
      for (var j = 0; j < M[i].length; j++) row.push(M[i][j] * s);
      out.push(row);
    }
    return out;
  };

  TQ.dot = function (a, b) {
    var s = 0;
    for (var i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  };

  TQ.norm = function (a) {
    return Math.sqrt(TQ.dot(a, a));
  };

  TQ.sum = function (arr) {
    var s = 0;
    for (var i = 0; i < arr.length; i++) s += arr[i];
    return s;
  };

  TQ.mean = function (arr) {
    return arr.length ? TQ.sum(arr) / arr.length : 0;
  };

  TQ.maxOf = function (arr) {
    var m = -Infinity;
    for (var i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
    return m;
  };

  TQ.minOf = function (arr) {
    var m = Infinity;
    for (var i = 0; i < arr.length; i++) if (arr[i] < m) m = arr[i];
    return m;
  };

  TQ.range = function (n) {
    var out = [];
    for (var i = 0; i < n; i++) out.push(i);
    return out;
  };

  // Numerically-stable softmax (subtract max before exp).
  TQ.softmax = function (arr) {
    var m = TQ.maxOf(arr);
    var exps = [];
    var s = 0;
    for (var i = 0; i < arr.length; i++) {
      var e = Math.exp(arr[i] - m);
      exps.push(e);
      s += e;
    }
    if (s === 0) s = 1;
    for (var j = 0; j < exps.length; j++) exps[j] /= s;
    return exps;
  };

  TQ.softmaxRows = function (M) {
    var out = [];
    for (var i = 0; i < M.length; i++) out.push(TQ.softmax(M[i]));
    return out;
  };

  // Scaled dot-product attention. Returns each intermediate so levels can show
  // the full pipeline: scores -> scaled -> weights -> output. dk inferred from Q.
  TQ.attention = function (Q, K, V) {
    var dk = Q[0].length;
    var scores = TQ.matmul(Q, TQ.transpose(K)); // (nq x nk)
    var inv = 1 / Math.sqrt(dk);
    var scaled = TQ.scaleMat(scores, inv);
    var weights = TQ.softmaxRows(scaled);
    var output = TQ.matmul(weights, V);
    return { scores: scores, scaled: scaled, weights: weights, output: output, dk: dk };
  };

  TQ.cosine = function (a, b) {
    var na = TQ.norm(a), nb = TQ.norm(b);
    if (na === 0 || nb === 0) return 0;
    return TQ.dot(a, b) / (na * nb);
  };

  TQ.clamp = function (x, lo, hi) {
    return x < lo ? lo : (x > hi ? hi : x);
  };

  TQ.lerp = function (a, b, t) {
    return a + (b - a) * t;
  };

  TQ.fmt = function (x, dp) {
    dp = (dp === undefined) ? 2 : dp;
    if (x === undefined || x === null || (typeof x === "number" && !isFinite(x))) return "—";
    var n = Number(x);
    // avoid "-0.00"
    if (Math.abs(n) < Math.pow(10, -dp) / 2) n = 0;
    return n.toFixed(dp);
  };

  TQ.relu = function (x) { return x > 0 ? x : 0; };
  TQ.reluVec = function (v) {
    var out = [];
    for (var i = 0; i < v.length; i++) out.push(TQ.relu(v[i]));
    return out;
  };

  // LayerNorm over a single vector (the transformer-block level uses this).
  TQ.layerNorm = function (vec, gain, bias) {
    gain = (gain === undefined) ? 1 : gain;
    bias = (bias === undefined) ? 0 : bias;
    var mu = TQ.mean(vec);
    var varc = 0;
    for (var i = 0; i < vec.length; i++) varc += (vec[i] - mu) * (vec[i] - mu);
    varc /= vec.length;
    var inv = 1 / Math.sqrt(varc + 1e-5);
    var out = [];
    for (var j = 0; j < vec.length; j++) out.push((vec[j] - mu) * inv * gain + bias);
    return out;
  };

  // Standard sinusoidal positional encoding (Vaswani et al.).
  TQ.sinusoidalPE = function (nPos, dModel) {
    var M = [];
    for (var pos = 0; pos < nPos; pos++) {
      var row = [];
      for (var i = 0; i < dModel; i++) {
        var twoI = 2 * Math.floor(i / 2);
        var denom = Math.pow(10000, twoI / dModel);
        var angle = pos / denom;
        row.push((i % 2 === 0) ? Math.sin(angle) : Math.cos(angle));
      }
      M.push(row);
    }
    return M;
  };

  /* ------------------------------------------------------------------ *
   *  SHARED TOY EXAMPLE
   * ------------------------------------------------------------------ */

  // Gram-Schmidt: turn a set of row-vectors into an orthonormal set. Used to
  // give the toy "concept axes" independent directions so cosine geometry is
  // clean (article ⟂ noun ⟂ action ⟂ place). Internal to the toy builder.
  function orthonormal(rows) {
    var out = [];
    for (var i = 0; i < rows.length; i++) {
      var v = rows[i].slice();
      for (var k = 0; k < out.length; k++) {
        var proj = TQ.dot(v, out[k]);
        for (var d = 0; d < v.length; d++) v[d] -= proj * out[k][d];
      }
      var nrm = TQ.norm(v) || 1;
      for (var d2 = 0; d2 < v.length; d2++) v[d2] /= nrm;
      out.push(v);
    }
    return out;
  }

  (function buildToy() {
    var tokens = ["The", "cat", "sat", "on", "the", "mat"];
    var dModel = 16;
    // Deterministic embeddings. We hand-craft a little semantic structure so
    // cosine similarity is *meaningful* in level 1: "The"/"the" are near-twins,
    // "cat"/"mat" share a noun-ish "thing in the scene" direction, and action
    // words sit elsewhere. Each token = a small base random vector (idiosyncratic
    // wiggle) blended toward a set of shared "concept" axes that dominate.
    // Low base scale keeps the concept geometry legible; the claims in the
    // level prose ("The"/"the" ≈ twins; cat/mat related; sat/the unrelated) hold.
    var base = TQ.randMatrix(tokens.length, dModel, 1337, 0.22);
    // "the" (#4) is the same surface word as "The" (#0): let it inherit most of
    // #0's idiosyncratic wiggle so the two come out as genuine near-twins
    // (high cosine), the way a real tokenizer would map them to related ids.
    for (var bd = 0; bd < dModel; bd++) base[4][bd] = base[0][bd] * 0.85 + base[4][bd] * 0.15;
    // 4 orthogonalized concept axes: article, noun/thing, action, place/scene.
    var concepts = orthonormal(TQ.randMatrix(4, dModel, 7, 1));
    // weights[token] = how much of each concept axis to mix in.
    var mix = [
      [1.00, 0.00, 0.00, 0.00], // The   -> pure article
      [0.05, 0.85, 0.00, 0.40], // cat   -> thing/noun (+ part of the scene)
      [0.00, 0.00, 1.00, 0.00], // sat   -> pure action (sits far from the nouns)
      [0.05, 0.00, 0.10, 0.85], // on    -> place/scene (preposition anchors location)
      [1.00, 0.00, 0.00, 0.00], // the   -> pure article (≈ "The")
      [0.05, 0.65, 0.00, 0.60]  // mat   -> thing/noun + place (a cat-adjacent object)
    ];
    var emb = [];
    for (var t = 0; t < tokens.length; t++) {
      var row = [];
      for (var d = 0; d < dModel; d++) {
        var v = base[t][d];
        for (var c = 0; c < concepts.length; c++) v += mix[t][c] * concepts[c][d];
        row.push(v);
      }
      emb.push(row);
    }
    TQ.toy = { tokens: tokens, dModel: dModel, embeddings: emb };
  })();

  // Convenience that bakes the documented Q/K/V convention.
  TQ.toyQKV = function (dk, seedQ, seedK, seedV) {
    var d = TQ.toy.dModel;
    var Wq = TQ.randMatrix(d, dk, seedQ);
    var Wk = TQ.randMatrix(d, dk, seedK);
    var Wv = TQ.randMatrix(d, dk, seedV);
    return {
      Q: TQ.matmul(TQ.toy.embeddings, Wq),
      K: TQ.matmul(TQ.toy.embeddings, Wk),
      V: TQ.matmul(TQ.toy.embeddings, Wv),
      Wq: Wq, Wk: Wk, Wv: Wv
    };
  };

  /* ------------------------------------------------------------------ *
   *  VISUAL HELPERS
   * ------------------------------------------------------------------ */

  function applyStyle(node, style) {
    if (!style) return;
    if (typeof style === "string") { node.setAttribute("style", style); return; }
    for (var k in style) {
      if (style.hasOwnProperty(k)) node.style[k] = style[k];
    }
  }

  function appendChild(node, child) {
    if (child === null || child === undefined || child === false || child === true) return;
    if (Array.isArray(child)) {
      for (var i = 0; i < child.length; i++) appendChild(node, child[i]);
      return;
    }
    if (child.nodeType) { node.appendChild(child); return; }
    node.appendChild(document.createTextNode(String(child)));
  }

  TQ.el = function (tag, attrs) {
    var node = document.createElement(tag);
    attrs = attrs || {};
    for (var key in attrs) {
      if (!attrs.hasOwnProperty(key)) continue;
      var val = attrs[key];
      if (val === null || val === undefined || val === false) continue;
      if (key === "class" || key === "className") node.className = val;
      else if (key === "text") node.textContent = val;
      else if (key === "html") node.innerHTML = val;
      else if (key === "style") applyStyle(node, val);
      else if (key === "onclick") node.addEventListener("click", val);
      else if (key === "oninput") node.addEventListener("input", val);
      else if (key === "onchange") node.addEventListener("change", val);
      else if (key === "dataset" && typeof val === "object") {
        for (var dk in val) if (val.hasOwnProperty(dk)) node.dataset[dk] = val[dk];
      } else if (key.indexOf("aria-") === 0 || key === "role" || key === "title" ||
                 key === "type" || key === "value" || key === "min" || key === "max" ||
                 key === "step" || key === "href" || key === "tabindex" || key === "id") {
        node.setAttribute(key, val);
      } else {
        node.setAttribute(key, val);
      }
    }
    for (var i = 2; i < arguments.length; i++) appendChild(node, arguments[i]);
    return node;
  };

  // --- ONE shared perceptual color scale ---------------------------------
  // t in [0,1]. Low = cool & dark (deep slate-blue), mid = teal/cyan, high =
  // bright & warm (amber). Implemented as a small multi-stop gradient in HSL
  // with brightness rising with t so it reads on a dark theme. EVERY visual
  // (heatmaps, vectors, bars) flows through here for a single color language.
  TQ.colorFor = function (t) {
    t = TQ.clamp(isFinite(t) ? t : 0, 0, 1);
    // stops: [t, h, s, l]
    var stops = [
      [0.00, 232, 45, 16],   // deep cool slate
      [0.30, 200, 65, 34],   // blue
      [0.55, 178, 70, 46],   // teal/cyan
      [0.78, 44, 92, 56],    // amber
      [1.00, 33, 100, 64]    // bright warm
    ];
    var a = stops[0], b = stops[stops.length - 1];
    for (var i = 0; i < stops.length - 1; i++) {
      if (t >= stops[i][0] && t <= stops[i + 1][0]) { a = stops[i]; b = stops[i + 1]; break; }
    }
    var span = (b[0] - a[0]) || 1;
    var f = (t - a[0]) / span;
    var h = TQ.lerp(a[1], b[1], f);
    var s = TQ.lerp(a[2], b[2], f);
    var l = TQ.lerp(a[3], b[3], f);
    return "hsl(" + h.toFixed(1) + "," + s.toFixed(1) + "%," + l.toFixed(1) + "%)";
  };

  // Map a signed value in [-max,max] onto the shared scale (0 -> mid).
  TQ.colorForSigned = function (x, max) {
    max = max || 1;
    var t = (x / max + 1) / 2;
    return TQ.colorFor(t);
  };

  // Read a CSS custom property (e.g. "--ink-mute") off :root so <canvas>
  // strokes/fills stay on the shared color language instead of hardcoded hex.
  // `alpha` (0..1) optionally tints the resolved color; falls back to `fallback`.
  TQ.cssVar = function (name, alpha, fallback) {
    var v = "";
    try {
      v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    } catch (e) { v = ""; }
    if (!v) v = fallback || "#6f7aa0";
    if (alpha === undefined || alpha === null || alpha >= 1) return v;
    // Resolve to rgba for hex (#rgb / #rrggbb) so we can apply alpha on canvas.
    var hex = v.charAt(0) === "#" ? v.slice(1) : "";
    if (hex.length === 3) hex = hex.replace(/(.)/g, "$1$1");
    if (hex.length === 6) {
      var r = parseInt(hex.slice(0, 2), 16),
          g = parseInt(hex.slice(2, 4), 16),
          bch = parseInt(hex.slice(4, 6), 16);
      return "rgba(" + r + "," + g + "," + bch + "," + TQ.clamp(alpha, 0, 1) + ")";
    }
    return v; // non-hex value: return as-is (alpha not applied)
  };

  // Pick readable text color (black/white) for a given [0,1] scale position.
  function inkFor(t) {
    return t > 0.5 ? "#0b1020" : "#e8ecf6";
  }

  TQ.note = function (content) {
    return TQ.el("div", { class: "tq-note" }, content);
  };

  TQ.callout = function (content) {
    return TQ.el("div", { class: "tq-callout" },
      TQ.el("span", { class: "tq-callout-mark", text: "✦", "aria-hidden": "true" }),
      TQ.el("div", { class: "tq-callout-body" }, content)
    );
  };

  TQ.math = function (text) {
    return TQ.el("code", { class: "tq-math", text: text });
  };

  TQ.block = function () {
    var node = TQ.el("section", { class: "tq-block" });
    for (var i = 0; i < arguments.length; i++) appendChild(node, arguments[i]);
    return node;
  };

  TQ.h = function (level, text) {
    return TQ.el("h" + (level || 2), { class: "tq-h", text: text });
  };

  TQ.p = function () {
    var node = TQ.el("p", { class: "tq-p" });
    for (var i = 0; i < arguments.length; i++) appendChild(node, arguments[i]);
    return node;
  };

  TQ.row = function () {
    var node = TQ.el("div", { class: "tq-flexrow" });
    for (var i = 0; i < arguments.length; i++) appendChild(node, arguments[i]);
    return node;
  };

  TQ.col = function () {
    var node = TQ.el("div", { class: "tq-flexcol" });
    for (var i = 0; i < arguments.length; i++) appendChild(node, arguments[i]);
    return node;
  };

  TQ.badge = function (text, kind) {
    return TQ.el("span", { class: "tq-badge tq-badge-" + (kind || "default"), text: text });
  };

  TQ.kv = function (label, value) {
    return TQ.el("span", { class: "tq-kv" },
      TQ.el("span", { class: "tq-kv-label", text: label }),
      TQ.el("span", { class: "tq-kv-val", text: value })
    );
  };

  // Heatmap. Auto-normalizes to data range unless {min,max} provided.
  TQ.heatmap = function (matrix, opts) {
    opts = opts || {};
    var rows = matrix.length, cols = matrix[0].length;
    var lo = (opts.min !== undefined) ? opts.min : TQ.minOf([].concat.apply([], matrix));
    var hi = (opts.max !== undefined) ? opts.max : TQ.maxOf([].concat.apply([], matrix));
    var span = (hi - lo) || 1;
    var cs = opts.cellSize || 40;
    var fmt = opts.format || function (v) { return TQ.fmt(v, 2); };

    var wrap = TQ.el("div", { class: "tq-heatmap-wrap" });
    if (opts.title) wrap.appendChild(TQ.el("div", { class: "tq-heatmap-title", text: opts.title }));

    var grid = TQ.el("div", { class: "tq-heatmap" });
    // template: optional corner + col labels
    var totalCols = (opts.rowLabels ? 1 : 0) + cols;
    grid.style.gridTemplateColumns =
      (opts.rowLabels ? "auto " : "") + "repeat(" + cols + ", " + cs + "px)";

    if (opts.colLabels) {
      if (opts.rowLabels) grid.appendChild(TQ.el("div", { class: "tq-hm-corner" }));
      for (var c = 0; c < cols; c++) {
        grid.appendChild(TQ.el("div", {
          class: "tq-hm-collabel" + (opts.selectedCol === c ? " is-sel" : ""),
          text: opts.colLabels[c]
        }));
      }
    }

    for (var r = 0; r < rows; r++) {
      if (opts.rowLabels) {
        grid.appendChild(TQ.el("div", {
          class: "tq-hm-rowlabel" + (opts.selectedRow === r ? " is-sel" : ""),
          text: opts.rowLabels[r]
        }));
      }
      for (var cc = 0; cc < cols; cc++) {
        var v = matrix[r][cc];
        var t = (v - lo) / span;
        var isSel = (opts.selectedRow === r);
        var cellAttrs = {
          class: "tq-hm-cell" + (isSel ? " is-rowsel" : "") +
                 (opts.selectedCol === cc && isSel ? " is-cellsel" : ""),
          style: {
            width: cs + "px", height: cs + "px",
            background: TQ.colorFor(t), color: inkFor(t)
          },
          title: (opts.rowLabels ? opts.rowLabels[r] + " · " : "") +
                 (opts.colLabels ? opts.colLabels[cc] + " · " : "") + fmt(v)
        };
        var cell = TQ.el("div", cellAttrs, TQ.el("span", { class: "tq-hm-val", text: fmt(v) }));
        if (opts.onCell) {
          (function (rr, ccc, val) {
            cell.classList.add("is-clickable");
            cell.setAttribute("role", "button");
            cell.setAttribute("tabindex", "0");
            cell.addEventListener("click", function () { opts.onCell(rr, ccc, val); });
            cell.addEventListener("keydown", function (e) {
              if (e.key === "Enter" || e.key === " ") { e.preventDefault(); opts.onCell(rr, ccc, val); }
            });
          })(r, cc, v);
        }
        grid.appendChild(cell);
      }
    }
    wrap.appendChild(grid);
    return wrap;
  };

  // Plain numeric table (NOT heat colored) — for raw matrices like Wq, etc.
  TQ.matrixGrid = function (matrix, opts) {
    opts = opts || {};
    var rows = matrix.length, cols = matrix[0].length;
    var fmt = opts.format || function (v) { return TQ.fmt(v, 2); };
    var wrap = TQ.el("div", { class: "tq-mgrid-wrap" });
    if (opts.title) wrap.appendChild(TQ.el("div", { class: "tq-heatmap-title", text: opts.title }));
    var tbl = TQ.el("table", { class: "tq-mgrid" });
    if (opts.colLabels) {
      var thead = TQ.el("tr", {});
      if (opts.rowLabels) thead.appendChild(TQ.el("th", { class: "tq-mgrid-corner" }));
      for (var c = 0; c < cols; c++) thead.appendChild(TQ.el("th", { text: opts.colLabels[c] }));
      tbl.appendChild(thead);
    }
    for (var r = 0; r < rows; r++) {
      var tr = TQ.el("tr", {});
      if (opts.rowLabels) tr.appendChild(TQ.el("th", { class: "tq-mgrid-rowlabel", text: opts.rowLabels[r] }));
      for (var cc = 0; cc < cols; cc++) tr.appendChild(TQ.el("td", { text: fmt(matrix[r][cc]) }));
      tbl.appendChild(tr);
    }
    wrap.appendChild(tbl);
    return wrap;
  };

  // Horizontal bars. Default color-by-value uses the shared scale.
  TQ.barRow = function (values, opts) {
    opts = opts || {};
    var max = (opts.max !== undefined) ? opts.max : TQ.maxOf(values.map(function (v) { return Math.abs(v); }));
    if (!max) max = 1;
    var fmt = opts.format || function (v) { return TQ.fmt(v, 3); };
    var wrap = TQ.el("div", { class: "tq-bars" });
    for (var i = 0; i < values.length; i++) {
      var v = values[i];
      var frac = TQ.clamp(Math.abs(v) / max, 0, 1);
      var color = opts.colorFn ? opts.colorFn(v, i) : TQ.colorFor(frac);
      var labelText = opts.labels ? opts.labels[i] : String(i);
      var rowEl = TQ.el("div", { class: "tq-bar-row" + (opts.highlight === i ? " is-hl" : "") },
        TQ.el("div", { class: "tq-bar-label", text: labelText }),
        TQ.el("div", { class: "tq-bar-track" },
          TQ.el("div", { class: "tq-bar-fill", style: { width: (frac * 100) + "%", background: color } })
        ),
        TQ.el("div", { class: "tq-bar-num", text: fmt(v) })
      );
      wrap.appendChild(rowEl);
    }
    return wrap;
  };

  // One vector as a row of colored cells. Normalizes by max(|v|) unless given.
  TQ.vectorView = function (values, opts) {
    opts = opts || {};
    var max = (opts.max !== undefined) ? opts.max : TQ.maxOf(values.map(function (v) { return Math.abs(v); }));
    if (!max) max = 1;
    var cs = opts.cellSize || 22;
    var fmt = opts.format || function (v) { return TQ.fmt(v, 2); };
    var showValues = !!opts.showValues;
    var wrap = TQ.el("div", { class: "tq-vector" });
    if (opts.label !== undefined) wrap.appendChild(TQ.el("div", { class: "tq-vector-label", text: opts.label }));
    var cells = TQ.el("div", { class: "tq-vector-cells" });
    for (var i = 0; i < values.length; i++) {
      var v = values[i];
      var t = (v / max + 1) / 2; // signed -> [0,1]
      var cell = TQ.el("div", {
        class: "tq-vector-cell",
        style: { width: cs + "px", height: cs + "px", background: TQ.colorFor(t), color: inkFor(t) },
        title: "[" + i + "] " + fmt(v)
      }, showValues ? TQ.el("span", { class: "tq-vector-cellval", text: fmt(v) }) : null);
      cells.appendChild(cell);
    }
    wrap.appendChild(cells);
    return wrap;
  };

  TQ.slider = function (opts) {
    opts = opts || {};
    var min = (opts.min !== undefined) ? opts.min : 0;
    var max = (opts.max !== undefined) ? opts.max : 1;
    var step = (opts.step !== undefined) ? opts.step : 0.01;
    var value = (opts.value !== undefined) ? opts.value : min;
    var fmt = opts.format || function (v) { return TQ.fmt(v, 2); };

    var valOut = TQ.el("span", { class: "tq-slider-val", text: fmt(value) });
    var input = TQ.el("input", {
      type: "range", class: "tq-slider-input",
      min: String(min), max: String(max), step: String(step), value: String(value)
    });
    var labelRow = TQ.el("div", { class: "tq-slider-head" },
      opts.label ? TQ.el("span", { class: "tq-slider-label", text: opts.label }) : null,
      valOut
    );
    var wrap = TQ.el("div", { class: "tq-slider" }, labelRow, input);

    function emit() {
      var v = parseFloat(input.value);
      valOut.textContent = fmt(v);
      if (opts.onInput) opts.onInput(v);
    }
    input.addEventListener("input", emit);

    return {
      el: wrap,
      get: function () { return parseFloat(input.value); },
      set: function (v) { input.value = String(v); valOut.textContent = fmt(parseFloat(input.value)); }
    };
  };

  TQ.toggle = function (opts) {
    opts = opts || {};
    var checked = !!opts.value;
    var input = TQ.el("input", { type: "checkbox", class: "tq-toggle-input" });
    input.checked = checked;
    var knob = TQ.el("span", { class: "tq-toggle-track" }, TQ.el("span", { class: "tq-toggle-knob" }));
    var wrap = TQ.el("label", { class: "tq-toggle" }, input, knob,
      opts.label ? TQ.el("span", { class: "tq-toggle-label", text: opts.label }) : null);
    input.addEventListener("change", function () {
      if (opts.onChange) opts.onChange(input.checked);
    });
    return {
      el: wrap,
      get: function () { return input.checked; },
      set: function (v) { input.checked = !!v; }
    };
  };

  TQ.segmented = function (opts) {
    opts = opts || {};
    var options = opts.options || [];
    var value = opts.value !== undefined ? opts.value : (options[0] && options[0].value);
    var wrap = TQ.el("div", { class: "tq-segmented", role: "tablist" });
    var btns = [];
    function sync() {
      for (var i = 0; i < btns.length; i++) {
        var on = options[i].value === value;
        btns[i].classList.toggle("is-active", on);
        btns[i].setAttribute("aria-selected", on ? "true" : "false");
      }
    }
    options.forEach(function (o, idx) {
      var b = TQ.el("button", { class: "tq-seg-btn", type: "button", role: "tab", text: o.label });
      b.addEventListener("click", function () {
        value = o.value; sync();
        if (opts.onChange) opts.onChange(value);
      });
      btns.push(b); wrap.appendChild(b);
    });
    sync();
    return {
      el: wrap,
      get: function () { return value; },
      set: function (v) { value = v; sync(); }
    };
  };

  TQ.tabs = function (items) {
    var wrap = TQ.el("div", { class: "tq-tabs" });
    var bar = TQ.el("div", { class: "tq-tabbar", role: "tablist" });
    var body = TQ.el("div", { class: "tq-tabbody" });
    var tabButtons = [];

    function select(idx) {
      for (var i = 0; i < tabButtons.length; i++) {
        var on = i === idx;
        tabButtons[i].classList.toggle("is-active", on);
        tabButtons[i].setAttribute("aria-selected", on ? "true" : "false");
        tabButtons[i].setAttribute("tabindex", on ? "0" : "-1");
      }
      body.innerHTML = "";
      items[idx].render(body);
    }

    items.forEach(function (item, idx) {
      var b = TQ.el("button", { class: "tq-tab", type: "button", role: "tab", text: item.label });
      b.addEventListener("click", function () { select(idx); });
      b.addEventListener("keydown", function (e) {
        if (e.key === "ArrowRight") { e.preventDefault(); select((idx + 1) % items.length); tabButtons[(idx + 1) % items.length].focus(); }
        if (e.key === "ArrowLeft") { e.preventDefault(); var p = (idx - 1 + items.length) % items.length; select(p); tabButtons[p].focus(); }
      });
      tabButtons.push(b);
      bar.appendChild(b);
    });

    wrap.appendChild(bar);
    wrap.appendChild(body);
    if (items.length) select(0);
    return wrap;
  };

  // Stepper: Prev / Next / Play through a sequence of steps. Each step.run gets
  // a fresh container. Play auto-advances; stops at the end.
  TQ.stepper = function (steps) {
    var wrap = TQ.el("div", { class: "tq-stepper" });
    var stageBody = TQ.el("div", { class: "tq-step-body" });
    var idx = 0;
    var playing = false;
    var timer = null;

    var dots = TQ.el("div", { class: "tq-step-dots" });
    var dotEls = steps.map(function (_, i) {
      var d = TQ.el("button", { class: "tq-step-dot", type: "button", "aria-label": "Step " + (i + 1) });
      d.addEventListener("click", function () { stop(); go(i); });
      dots.appendChild(d);
      return d;
    });

    var label = TQ.el("div", { class: "tq-step-label" });
    var counter = TQ.el("span", { class: "tq-step-counter" });
    var prevBtn = TQ.el("button", { class: "tq-btn tq-btn-ghost", type: "button", text: "‹ Prev" });
    var nextBtn = TQ.el("button", { class: "tq-btn tq-btn-ghost", type: "button", text: "Next ›" });
    var playBtn = TQ.el("button", { class: "tq-btn tq-btn-accent", type: "button", text: "▶ Play" });

    function render() {
      stageBody.innerHTML = "";
      label.textContent = steps[idx].label || ("Step " + (idx + 1));
      counter.textContent = (idx + 1) + " / " + steps.length;
      steps[idx].run(stageBody);
      for (var i = 0; i < dotEls.length; i++) dotEls[i].classList.toggle("is-active", i === idx);
      prevBtn.disabled = idx === 0;
      nextBtn.disabled = idx === steps.length - 1;
    }
    function go(i) { idx = TQ.clamp(i, 0, steps.length - 1); render(); }
    function next() { if (idx < steps.length - 1) { go(idx + 1); } else { stop(); } }
    function prev() { go(idx - 1); }
    function stop() { playing = false; playBtn.textContent = "▶ Play"; if (timer) { clearInterval(timer); timer = null; } }
    function play() {
      if (playing) { stop(); return; }
      if (idx === steps.length - 1) go(0);
      playing = true; playBtn.textContent = "⏸ Pause";
      timer = setInterval(function () {
        if (idx >= steps.length - 1) { stop(); return; }
        next();
      }, 1600);
    }

    prevBtn.addEventListener("click", function () { stop(); prev(); });
    nextBtn.addEventListener("click", function () { stop(); next(); });
    playBtn.addEventListener("click", play);

    var controls = TQ.el("div", { class: "tq-step-controls" },
      prevBtn, playBtn, nextBtn, counter
    );
    wrap.appendChild(TQ.el("div", { class: "tq-step-head" }, label, dots));
    wrap.appendChild(stageBody);
    wrap.appendChild(controls);
    render();
    return wrap;
  };

  TQ.canvasPanel = function (width, height, drawFn) {
    var ratio = window.devicePixelRatio || 1;
    var canvas = TQ.el("canvas", { class: "tq-canvas" });
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    var ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    var wrap = TQ.el("div", { class: "tq-canvas-wrap" }, canvas);
    function redraw() {
      ctx.clearRect(0, 0, width, height);
      if (drawFn) drawFn(ctx, width, height);
    }
    redraw();
    return { el: wrap, ctx: ctx, canvas: canvas, redraw: redraw };
  };

  /* ------------------------------------------------------------------ *
   *  GAME ENGINE
   * ------------------------------------------------------------------ */

  var LEVELS = [];
  var LEVELS_BY_ID = {};

  TQ.registerLevel = function (def) {
    if (!def || !def.id) { console.error("registerLevel: missing def/id", def); return; }
    if (LEVELS_BY_ID[def.id]) { console.warn("registerLevel: duplicate id", def.id); return; }
    LEVELS_BY_ID[def.id] = def;
    LEVELS.push(def);
  };

  // -- persistence (graceful fallback when localStorage blocked under file://)
  var STORAGE_KEY = "tq_progress";
  var memStore = null; // fallback in-memory progress
  function storageAvailable() {
    try {
      var k = "__tq_test__";
      window.localStorage.setItem(k, "1");
      window.localStorage.removeItem(k);
      return true;
    } catch (e) { return false; }
  }
  var HAS_STORAGE = storageAvailable();

  function defaultProgress() { return { completed: {}, xp: 0 }; }

  function loadProgress() {
    if (!HAS_STORAGE) return memStore || (memStore = defaultProgress());
    try {
      var raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return defaultProgress();
      var parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") return defaultProgress();
      if (!parsed.completed) parsed.completed = {};
      if (typeof parsed.xp !== "number") parsed.xp = 0;
      return parsed;
    } catch (e) { return defaultProgress(); }
  }

  function saveProgress(p) {
    if (!HAS_STORAGE) { memStore = p; return; }
    try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(p)); }
    catch (e) { memStore = p; }
  }

  var XP_PER_LEVEL = 100;

  function sortedLevels() {
    return LEVELS.slice().sort(function (a, b) { return (a.order || 0) - (b.order || 0); });
  }

  function isUnlocked(levels, idx, progress) {
    if (idx === 0) return true;
    var prev = levels[idx - 1];
    return !!progress.completed[prev.id];
  }

  // ---- App state / routing -------------------------------------------------
  var appRoot = null;
  var state = { view: "map", levelId: null };

  function mount(node) {
    appRoot.innerHTML = "";
    appRoot.appendChild(node);
    window.scrollTo(0, 0);
  }

  function progressPct(levels, progress) {
    if (!levels.length) return 0;
    var done = 0;
    for (var i = 0; i < levels.length; i++) if (progress.completed[levels[i].id]) done++;
    return Math.round((done / levels.length) * 100);
  }

  // ---- Top bar (persistent header) ----------------------------------------
  function buildTopBar(levels, progress, opts) {
    opts = opts || {};
    var pct = progressPct(levels, progress);

    var bar = TQ.el("div", { class: "tq-progress-track" },
      TQ.el("div", { class: "tq-progress-fill", style: { width: pct + "%" } })
    );

    var brand = TQ.el("button", { class: "tq-brand", type: "button", title: "Back to map" },
      TQ.el("span", { class: "tq-brand-mark", text: "⬡", "aria-hidden": "true" }),
      TQ.el("span", { class: "tq-brand-name", text: "Transformer Quest" })
    );
    brand.addEventListener("click", function () { goMap(); });

    var xpChip = TQ.el("div", { class: "tq-xp", title: "Experience" },
      TQ.el("span", { class: "tq-xp-star", text: "✦", "aria-hidden": "true" }),
      TQ.el("span", { class: "tq-xp-num", text: progress.xp + " XP" })
    );
    var pctChip = TQ.el("div", { class: "tq-pct", text: pct + "%" });

    var resetBtn = TQ.el("button", { class: "tq-reset", type: "button", title: "Reset all progress", text: "reset" });
    resetBtn.addEventListener("click", function () {
      if (window.confirm("Reset all progress and XP? This cannot be undone.")) {
        var fresh = defaultProgress();
        saveProgress(fresh);
        goMap();
      }
    });

    var right = TQ.el("div", { class: "tq-top-right" }, pctChip, xpChip, resetBtn);
    var top = TQ.el("header", { class: "tq-topbar" },
      TQ.el("div", { class: "tq-top-inner" }, brand, right),
      bar
    );
    return top;
  }

  // ---- MAP screen ----------------------------------------------------------
  function renderMap() {
    state.view = "map";
    var levels = sortedLevels();
    var progress = loadProgress();

    var wrap = TQ.el("div", { class: "tq-app" });
    wrap.appendChild(buildTopBar(levels, progress));

    var allDone = levels.length && progressPct(levels, progress) === 100;

    var hero = TQ.el("div", { class: "tq-hero" },
      TQ.el("h1", { class: "tq-hero-title", text: "Transformer Quest" }),
      TQ.el("p", { class: "tq-hero-sub", html:
        "From <strong>tokens</strong> to <strong>attention</strong> to the <strong>KV-cache</strong> tricks " +
        "(MHA · MQA · GQA · MLA) that make long-context LLMs affordable. " +
        "Manipulate the knobs, watch the real numbers move." }),
      allDone ? TQ.el("div", { class: "tq-hero-done" }, TQ.badge("All levels complete — boss slain 🏁", "good")) : null
    );

    var grid = TQ.el("div", { class: "tq-map-grid" });
    levels.forEach(function (lvl, idx) {
      var unlocked = isUnlocked(levels, idx, progress);
      var done = !!progress.completed[lvl.id];

      var statusIcon = done ? "✓" : (unlocked ? "▶" : "🔒");
      var card = TQ.el("button", {
        class: "tq-card" + (done ? " is-done" : "") + (unlocked ? "" : " is-locked"),
        type: "button",
        "aria-label": lvl.title + (unlocked ? "" : " (locked)")
      },
        TQ.el("div", { class: "tq-card-top" },
          TQ.el("span", { class: "tq-card-icon", text: lvl.icon || "•" }),
          TQ.el("span", { class: "tq-card-order", text: "Lv " + lvl.order }),
          TQ.el("span", { class: "tq-card-status", text: statusIcon })
        ),
        TQ.el("div", { class: "tq-card-title", text: lvl.title }),
        TQ.el("div", { class: "tq-card-tagline", text: lvl.tagline || "" }),
        TQ.el("div", { class: "tq-card-foot" },
          TQ.el("span", { class: "tq-card-preread", text: lvl.preread || "" }),
          done ? TQ.badge("done", "good") : (unlocked ? TQ.badge("open", "info") : TQ.badge("locked", "default"))
        )
      );
      if (unlocked) {
        card.addEventListener("click", function () { goLevel(lvl.id); });
      } else {
        card.disabled = true;
      }
      grid.appendChild(card);
    });

    var main = TQ.el("main", { class: "tq-main" }, hero, grid,
      TQ.el("div", { class: "tq-map-foot" },
        HAS_STORAGE ? null : TQ.note("Storage is blocked (you're on file://) — the game runs fine, but progress won't be saved between reloads.")
      )
    );
    wrap.appendChild(main);
    mount(wrap);
  }

  // ---- QUIZ ----------------------------------------------------------------
  function renderQuiz(container, level, onPass) {
    var quiz = level.quiz || [];
    var answers = {};            // qIndex -> chosen index
    var locked = {};             // qIndex -> bool (answered/locked)

    var card = TQ.el("div", { class: "tq-quiz" });
    card.appendChild(TQ.el("h3", { class: "tq-quiz-h" },
      TQ.el("span", { text: "Check Understanding" }),
      TQ.el("span", { class: "tq-quiz-count", text: quiz.length + (quiz.length === 1 ? " question" : " questions") })
    ));

    var status = TQ.el("div", { class: "tq-quiz-status", text: "Answer all questions to pass and unlock the next level." });

    function refreshStatus() {
      var answered = 0, correct = 0;
      for (var i = 0; i < quiz.length; i++) {
        if (locked[i]) { answered++; if (answers[i] === quiz[i].answer) correct++; }
      }
      if (answered < quiz.length) {
        status.className = "tq-quiz-status";
        status.textContent = answered + " / " + quiz.length + " answered — " + correct + " correct so far.";
        passBtn.disabled = true;
      } else if (correct === quiz.length) {
        status.className = "tq-quiz-status is-pass";
        status.textContent = "Perfect — " + correct + " / " + quiz.length + ". You may advance.";
        passBtn.disabled = false;
      } else {
        status.className = "tq-quiz-status is-fail";
        status.textContent = correct + " / " + quiz.length + " correct. Review the explanations, then retry the misses.";
        passBtn.disabled = true;
      }
    }

    quiz.forEach(function (item, qi) {
      var qEl = TQ.el("div", { class: "tq-q" });
      qEl.appendChild(TQ.el("div", { class: "tq-q-text" },
        TQ.el("span", { class: "tq-q-num", text: "Q" + (qi + 1) }),
        TQ.el("span", { text: item.q })
      ));
      var choicesEl = TQ.el("div", { class: "tq-q-choices" });
      var explainEl = TQ.el("div", { class: "tq-q-explain" });

      item.choices.forEach(function (choice, ci) {
        var btn = TQ.el("button", { class: "tq-choice", type: "button" },
          TQ.el("span", { class: "tq-choice-key", text: String.fromCharCode(65 + ci) }),
          TQ.el("span", { class: "tq-choice-text", text: choice })
        );
        btn.addEventListener("click", function () {
          if (locked[qi]) return;
          locked[qi] = true;
          answers[qi] = ci;
          var isRight = ci === item.answer;
          // mark this choice + reveal the correct one; lock all
          var allBtns = choicesEl.querySelectorAll(".tq-choice");
          for (var b = 0; b < allBtns.length; b++) {
            allBtns[b].disabled = true;
            if (b === item.answer) allBtns[b].classList.add("is-correct");
          }
          if (!isRight) btn.classList.add("is-wrong");
          // icon + text — never color alone
          btn.appendChild(TQ.el("span", { class: "tq-choice-mark", text: isRight ? "✓" : "✗" }));
          explainEl.className = "tq-q-explain is-shown " + (isRight ? "is-right" : "is-wrong");
          explainEl.appendChild(TQ.el("span", { class: "tq-explain-tag", text: isRight ? "Correct" : "Not quite" }));
          explainEl.appendChild(TQ.el("span", { text: item.explain }));
          refreshStatus();
        });
        choicesEl.appendChild(btn);
      });

      qEl.appendChild(choicesEl);
      qEl.appendChild(explainEl);
      card.appendChild(qEl);
    });

    var passBtn = TQ.el("button", { class: "tq-btn tq-btn-accent tq-quiz-pass", type: "button", text: "Complete level →" });
    passBtn.disabled = true;
    passBtn.addEventListener("click", function () { onPass(); });

    card.appendChild(status);
    card.appendChild(TQ.el("div", { class: "tq-quiz-actions" }, passBtn));
    container.appendChild(card);
    refreshStatus();
  }

  // ---- LEVEL view ----------------------------------------------------------
  function renderLevel(levelId) {
    var levels = sortedLevels();
    var level = LEVELS_BY_ID[levelId];
    if (!level) { renderMap(); return; }
    var idx = -1;
    for (var i = 0; i < levels.length; i++) if (levels[i].id === levelId) { idx = i; break; }
    var progress = loadProgress();

    if (!isUnlocked(levels, idx, progress)) { renderMap(); return; }

    state.view = "level";
    state.levelId = levelId;

    var wrap = TQ.el("div", { class: "tq-app" });
    wrap.appendChild(buildTopBar(levels, progress));

    var backBtn = TQ.el("button", { class: "tq-btn tq-btn-ghost tq-back", type: "button", text: "‹ Map" });
    backBtn.addEventListener("click", function () { goMap(); });

    var head = TQ.el("div", { class: "tq-level-head" },
      TQ.el("div", { class: "tq-level-head-top" },
        backBtn,
        TQ.el("span", { class: "tq-level-icon", text: level.icon || "•" }),
        TQ.el("div", { class: "tq-level-titles" },
          TQ.el("h1", { class: "tq-level-title", text: "Lv " + level.order + " · " + level.title }),
          TQ.el("div", { class: "tq-level-tagline", text: level.tagline || "" })
        ),
        TQ.badge(level.preread || "", "info")
      )
    );

    if (level.objectives && level.objectives.length) {
      var objList = TQ.el("ul", { class: "tq-objectives" });
      level.objectives.forEach(function (o) {
        objList.appendChild(TQ.el("li", { class: "tq-objective" },
          TQ.el("span", { class: "tq-objective-mark", text: "◆", "aria-hidden": "true" }),
          TQ.el("span", { text: o })
        ));
      });
      head.appendChild(TQ.el("div", { class: "tq-objectives-wrap" },
        TQ.el("div", { class: "tq-objectives-label", text: "In this level you'll" }), objList));
    }

    var lessonEl = TQ.el("div", { class: "tq-lesson" });
    try {
      level.render(lessonEl);
    } catch (err) {
      console.error("Level render failed:", levelId, err);
      lessonEl.appendChild(TQ.callout("This level hit a snag while rendering. (" + (err && err.message) + ")"));
    }

    var quizWrap = TQ.el("div", { class: "tq-quiz-wrap" });
    renderQuiz(quizWrap, level, function () {
      var p = loadProgress();
      var firstTime = !p.completed[level.id];
      p.completed[level.id] = true;
      if (firstTime) p.xp += XP_PER_LEVEL;
      saveProgress(p);
      showCompletion(level, levels, idx);
    });

    var main = TQ.el("main", { class: "tq-main tq-level-main" }, head, lessonEl, quizWrap);
    wrap.appendChild(main);
    mount(wrap);
  }

  // Sync the persistent top bar (progress fill, % chip, XP) to current
  // progress without a full re-render — so the reward shows the moment the
  // completion modal opens, not only after returning to the map.
  function refreshTopBar() {
    if (!appRoot) return;
    var p = loadProgress();
    var pct = progressPct(sortedLevels(), p);
    var fill = appRoot.querySelector(".tq-progress-fill");
    if (fill) fill.style.width = pct + "%";
    var pctChip = appRoot.querySelector(".tq-pct");
    if (pctChip) pctChip.textContent = pct + "%";
    var xpNum = appRoot.querySelector(".tq-xp-num");
    if (xpNum) xpNum.textContent = p.xp + " XP";
  }

  // ---- Completion interstitial --------------------------------------------
  function showCompletion(level, levels, idx) {
    var progress = loadProgress();
    var isLast = idx === levels.length - 1;
    var nextLevel = isLast ? null : levels[idx + 1];

    // reflect the freshly-awarded XP / progress in the header immediately
    refreshTopBar();

    var overlay = TQ.el("div", { class: "tq-overlay" });
    var confetti = TQ.el("div", { class: "tq-confetti", "aria-hidden": "true" });
    for (var c = 0; c < 28; c++) {
      var piece = TQ.el("span", { class: "tq-confetti-bit" });
      piece.style.left = (Math.round((c / 28) * 100)) + "%";
      piece.style.animationDelay = (((c % 7) * 90)) + "ms";
      piece.style.background = TQ.colorFor((c % 7) / 6);
      confetti.appendChild(piece);
    }

    var actions = TQ.el("div", { class: "tq-complete-actions" });
    if (nextLevel) {
      var nextBtn = TQ.el("button", { class: "tq-btn tq-btn-accent", type: "button",
        text: "Next: " + nextLevel.title + " →" });
      nextBtn.addEventListener("click", function () { goLevel(nextLevel.id); });
      actions.appendChild(nextBtn);
    }
    var mapBtn = TQ.el("button", { class: "tq-btn tq-btn-ghost", type: "button", text: "Back to map" });
    mapBtn.addEventListener("click", function () { goMap(); });
    actions.appendChild(mapBtn);

    var modal = TQ.el("div", { class: "tq-complete" },
      confetti,
      TQ.el("div", { class: "tq-complete-icon", text: isLast ? "🏁" : (level.icon || "✓") }),
      TQ.el("h2", { class: "tq-complete-title", text: isLast ? "Quest complete!" : "Level complete!" }),
      TQ.el("p", { class: "tq-complete-sub", text: isLast
        ? "You've connected the whole chain — from embeddings to MLA. Go ship something memory-efficient."
        : "Nice. " + level.title + " cleared." }),
      TQ.el("div", { class: "tq-complete-xp" },
        TQ.badge("+" + XP_PER_LEVEL + " XP", "good"),
        TQ.el("span", { class: "tq-complete-total", text: "Total: " + progress.xp + " XP" })
      ),
      actions
    );
    overlay.appendChild(modal);
    overlay.addEventListener("click", function (e) { if (e.target === overlay) goMap(); });
    appRoot.appendChild(overlay);
  }

  // ---- navigation ----------------------------------------------------------
  function goMap() { renderMap(); }
  function goLevel(id) { renderLevel(id); }

  TQ.goMap = goMap;
  TQ.goLevel = goLevel;

  // expose a tiny bit of read-only introspection for debugging
  TQ._levels = function () { return sortedLevels(); };
  TQ.hasStorage = function () { return HAS_STORAGE; };

  function boot() {
    appRoot = document.getElementById("app");
    if (!appRoot) {
      appRoot = TQ.el("div", { id: "app" });
      document.body.appendChild(appRoot);
    }
    if (!LEVELS.length) {
      appRoot.appendChild(TQ.callout("No levels registered. Add level modules that call TQ.registerLevel(...)."));
      return;
    }
    renderMap();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

  window.TQ = TQ;
})();
