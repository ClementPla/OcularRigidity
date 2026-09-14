/* Regression statistics recomputed in the browser, from the *visible* points.
 *
 * Streamlit only reruns Python on a widget change, and a Plotly legend click is
 * not one: hiding a group redraws the chart client-side and the server never
 * hears about it. So a stats row rendered with st.metric goes stale the moment
 * anyone clicks the legend -- it keeps reporting the whole cohort while the plot
 * shows a subset. Computing them here instead keeps the number and the picture
 * describing the same rows.
 *
 * Everything below matches scipy: the Pearson and Spearman p-values are the
 * two-sided Student-t test on n-2 degrees of freedom (what scipy.stats.pearsonr
 * and .spearmanr report), via a regularized incomplete beta.
 */

function _mean(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

/* Ranks with ties averaged -- the tie rule Spearman needs. */
function _rankAverage(a) {
  const idx = a.map((v, i) => i).sort((p, q) => a[p] - a[q]);
  const r = new Array(a.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && a[idx[j + 1]] === a[idx[i]]) j++;
    const shared = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[idx[k]] = shared;
    i = j + 1;
  }
  return r;
}

function _pearson(x, y) {
  const n = x.length;
  const mx = _mean(x);
  const my = _mean(y);
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  if (sxx === 0 || syy === 0) return NaN;
  return sxy / Math.sqrt(sxx * syy);
}

function _linfit(x, y) {
  const n = x.length;
  const mx = _mean(x);
  const my = _mean(y);
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) * (x[i] - mx);
  }
  if (sxx === 0) return { slope: NaN, intercept: NaN };
  const slope = sxy / sxx;
  return { slope: slope, intercept: my - slope * mx };
}

/* Continued fraction for the incomplete beta (Numerical Recipes 6.4). */
function _betacf(a, b, x) {
  const TINY = 1e-30;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < TINY) d = TINY;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 3e-16) break;
  }
  return h;
}

function _gammaln(z) {
  const g = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
  ];
  let x = z;
  let tmp = z + 5.5;
  tmp -= (z + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += g[j] / ++x;
  return -tmp + Math.log((2.5066282746310005 * ser) / z);
}

/* Regularized incomplete beta I_x(a, b). */
function _betai(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(
    _gammaln(a + b) - _gammaln(a) - _gammaln(b) + a * Math.log(x) + b * Math.log(1 - x)
  );
  if (x < (a + 1) / (a + b + 2)) return (front * _betacf(a, b, x)) / a;
  return 1 - (front * _betacf(b, a, 1 - x)) / b;
}

/* Two-sided p for a correlation of r on n points (Student-t, df = n - 2). */
function _pFromR(r, n) {
  const df = n - 2;
  if (!(df > 0) || !isFinite(r)) return NaN;
  if (Math.abs(r) >= 1) return 0;
  const t2 = (r * r * df) / (1 - r * r);
  return _betai(0.5 * df, 0.5, df / (df + t2));
}

/* n, Pearson, Spearman and the OLS line over the finite pairs of x and y. */
function regressionStats(x, y) {
  const xs = [];
  const ys = [];
  for (let i = 0; i < x.length; i++) {
    const a = x[i];
    const b = y[i];
    if (a === null || b === null) continue;
    const fa = Number(a);
    const fb = Number(b);
    if (Number.isFinite(fa) && Number.isFinite(fb)) {
      xs.push(fa);
      ys.push(fb);
    }
  }
  const n = xs.length;
  if (n < 3) return { n: n };
  const r = _pearson(xs, ys);
  const rho = _pearson(_rankAverage(xs), _rankAverage(ys));
  const fit = _linfit(xs, ys);
  return {
    n: n,
    r: r,
    p_r: _pFromR(r, n),
    rho: rho,
    p_rho: _pFromR(rho, n),
    slope: fit.slope,
    intercept: fit.intercept,
    xmin: Math.min.apply(null, xs),
    xmax: Math.max.apply(null, xs),
  };
}

/* ---- Plotly wiring ---------------------------------------------------- */

const _FIT_ROLE = "fit";
const _STATS_NAME = "live-stats";
const _NBSP = "\u00a0";

function _pad(s, width) {
  s = String(s);
  return s + _NBSP.repeat(Math.max(0, width - s.length));
}

function _fmtP(p) {
  if (!isFinite(p)) return "n/a";
  if (p < 1e-4) return p.toExponential(1);
  return p.toFixed(4);
}

function _fmtG(v, digits) {
  if (!isFinite(v)) return "n/a";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return Number(v.toPrecision(digits)).toString();
}

function _statsText(s, hiddenCount) {
  if (!s || s.n < 3) {
    return "<b>N" + _NBSP.repeat(8) + (s ? s.n : 0) + "</b><br>too few points to fit";
  }
  const lines = [
    "<b>" + _pad("N", 11) + s.n + "</b>",
    _pad("Pearson r", 11) + s.r.toFixed(3) + _NBSP.repeat(3) + "p " + _fmtP(s.p_r),
    _pad("Spearman \u03c1", 11) + s.rho.toFixed(3) + _NBSP.repeat(3) + "p " + _fmtP(s.p_rho),
    _pad("Slope", 11) + _fmtG(s.slope, 4),
  ];
  if (hiddenCount > 0) {
    lines.push(
      "<i>" + hiddenCount + " group" + (hiddenCount > 1 ? "s" : "") + " hidden</i>"
    );
  }
  return lines.join("<br>");
}

/* plotly.py >= 6 ships numeric columns as {dtype, bdata} base64 rather than as
 * JSON arrays, and leaves them that way on gd.data (it decodes into _fullData,
 * which is private). Anything reading the points back has to undo it. */
const _DTYPES = {
  f8: Float64Array,
  f4: Float32Array,
  i4: Int32Array,
  i2: Int16Array,
  i1: Int8Array,
  u4: Uint32Array,
  u2: Uint16Array,
  u1: Uint8Array,
};

function _asArray(v) {
  if (!v) return [];
  if (Array.isArray(v) || ArrayBuffer.isView(v)) return v;
  if (v.bdata !== undefined && v.dtype) {
    const Ctor = _DTYPES[v.dtype];
    if (!Ctor) return [];
    const bin = atob(v.bdata);
    const buf = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    return new Ctor(buf.buffer);
  }
  return [];
}

/* x/y of every trace the legend currently shows, minus the fit line itself. */
function _visibleXY(gd) {
  const x = [];
  const y = [];
  let hidden = 0;
  gd.data.forEach(function (t) {
    if (t.meta && t.meta.role === _FIT_ROLE) return;
    if (t.visible === "legendonly" || t.visible === false) {
      hidden++;
      return;
    }
    const tx = _asArray(t.x);
    const ty = _asArray(t.y);
    for (let i = 0; i < Math.min(tx.length, ty.length); i++) {
      x.push(tx[i]);
      y.push(ty[i]);
    }
  });
  return { x: x, y: y, hidden: hidden };
}

function _fitLine(s, logx) {
  if (!s || s.n < 3 || !isFinite(s.slope)) return { x: [], y: [] };
  const N = 64;
  const x = [];
  const y = [];
  // Sampled rather than drawn end-to-end: on a log x-axis the OLS line is
  // straight in the data, not on the screen.
  const lo = logx ? Math.log10(Math.max(s.xmin, Number.MIN_VALUE)) : s.xmin;
  const hi = logx ? Math.log10(Math.max(s.xmax, Number.MIN_VALUE)) : s.xmax;
  for (let i = 0; i < N; i++) {
    const t = lo + ((hi - lo) * i) / (N - 1);
    const xv = logx ? Math.pow(10, t) : t;
    x.push(xv);
    y.push(s.intercept + s.slope * xv);
  }
  return { x: x, y: y };
}

/* Draw the figure and keep the stats box and the fit line in step with the
 * legend.
 *
 * Triggered by `plotly_restyle` / `plotly_afterplot`, which plotly emits *after*
 * it has applied a change. The obvious-looking trigger, `plotly_legendclick`,
 * is wrong twice over: it fires BEFORE the new visibility is applied, so
 * reading gd.data from it reports the state before the click and the box sits
 * permanently one click behind; and no fixed deferral repairs that, because how
 * long the redraw takes is plotly's business, not ours.
 *
 * Listening after the fact means our own restyle of the fit line re-enters the
 * handler, which is what hung the tab in the first version. Two things stop it:
 * `busy` swallows anything arriving mid-update (queueing it, so a fast click is
 * never dropped), and an unchanged signature makes the trailing run a no-op.
 * Together they terminate: at worst one wasted pass that changes nothing. */
function initLiveRegression(divId, figure, config, opts) {
  opts = opts || {};
  const gd = document.getElementById(divId);
  let busy = false;
  let pending = false;
  let last = null;

  function refresh() {
    // A click arriving mid-update is queued, not dropped: dropping it would
    // leave exactly the stale box this whole file exists to prevent. The
    // signature check below makes the trailing run a no-op when it is one.
    if (busy) {
      pending = true;
      return Promise.resolve();
    }
    const vis = _visibleXY(gd);
    const s = regressionStats(vis.x, vis.y);
    const text = _statsText(s, vis.hidden);
    const line = _fitLine(s, !!opts.logx);
    // Nothing moved (a legend click on an already-hidden trace, a redraw):
    // skip, so we never emit an event we would have to react to.
    const sig = text + "|" + line.x.length + "|" + line.y[0] + "|" + line.y[line.y.length - 1];
    if (sig === last) return Promise.resolve();
    last = sig;

    busy = true;
    const jobs = [];
    const fitIdx = gd.data.findIndex(function (t) {
      return t.meta && t.meta.role === _FIT_ROLE;
    });
    if (fitIdx >= 0) {
      jobs.push(Plotly.restyle(gd, { x: [line.x], y: [line.y] }, [fitIdx]));
    }
    if (opts.showStats) {
      const anns = gd.layout.annotations || [];
      const k = anns.findIndex(function (a) {
        return a.name === _STATS_NAME;
      });
      if (k >= 0) {
        const upd = {};
        upd["annotations[" + k + "].text"] = text;
        jobs.push(Plotly.relayout(gd, upd));
      }
    }
    return Promise.all(jobs).then(
      function () {
        // Released on the next macrotask, after any event plotly emitted from
        // inside restyle/relayout has already been swallowed by the guard.
        return new Promise(function (done) {
          setTimeout(function () {
            busy = false;
            if (pending) {
              pending = false;
              refresh();
            }
            done();
          }, 0);
        });
      },
      function (err) {
        busy = false;
        pending = false;
        throw err;
      }
    );
  }

  Plotly.newPlot(gd, figure.data, figure.layout, config).then(function () {
    refresh();
    // Both, because which one lands depends on how the change was made: a
    // legend toggle is a restyle, an autorange or a resize only redraws.
    gd.on("plotly_restyle", refresh);
    gd.on("plotly_afterplot", refresh);
    window.addEventListener("resize", function () {
      Plotly.Plots.resize(gd);
    });
  });

  return { refresh: refresh };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { regressionStats: regressionStats, _rankAverage: _rankAverage, _statsText: _statsText, _visibleXY: _visibleXY, _fitLine: _fitLine, _asArray: _asArray };
}
