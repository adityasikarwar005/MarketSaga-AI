/* Run services/dashboard/chart.html's own script against real generated payloads.
 *
 * The page has had two defects that were invisible to everything else in the
 * suite: a layout feedback loop that hung the renderer, and a chain that was
 * eleven strikes wearing the word "every". Both were found by opening it. The
 * Python side is well tested and tests the *shape* of what it writes; nothing
 * tested whether the page could read it, and the failure mode there is an empty
 * pane beside a ladder that looks complete - no error, nothing in the console.
 *
 * So this loads the real files, stubs just enough browser to let the page's own
 * `init()` run, and reports what it drew. It is not a rendering test - the
 * canvas is a no-op - it is a "the page and the packer still agree" test, which
 * is the part that spans two languages and cannot be checked from either side.
 *
 * Usage: node render_check.mjs <dashboard-dir>   ->  JSON on stdout
 */
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const dir = process.argv[2];
const problems = [];

/* ---- the smallest DOM the page will accept ---------------------------- */

// Elements the page reads back after writing: it sets innerHTML and then
// queries the result. Parsing the attributes it needs is enough; parsing HTML
// properly is not this file's job.
function parseChildren(html) {
  const out = [];
  const re = /<button\b([^>]*)>/g;
  let m;
  while ((m = re.exec(html))) {
    const attrs = {};
    for (const a of m[1].matchAll(/([\w-]+)="([^"]*)"/g)) attrs[a[1]] = a[2];
    const dataset = {};
    for (const [k, v] of Object.entries(attrs)) {
      if (k.startsWith("data-")) dataset[k.slice(5)] = v;
    }
    out.push({
      tag: "button",
      attrs,
      dataset,
      className: attrs.class || "",
      onclick: null,
      setAttribute(k, v) { this.attrs[k] = v; },
      getAttribute(k) { return this.attrs[k]; },
      scrollIntoView() {},
    });
  }
  return out;
}

const noopCtx = new Proxy(
  { canvas: null },
  {
    get(target, prop) {
      if (prop in target) return target[prop];
      // Every 2d-context method is a no-op and every property is writable.
      return typeof prop === "string" && prop === "measureText"
        ? () => ({ width: 10 })
        : () => {};
    },
    set() { return true; },
  },
);

function makeEl(id) {
  return {
    id,
    // Children are parsed once when innerHTML is written and kept. They have to
    // be the *same objects* on every query: the page sets innerHTML, queries the
    // result, assigns onclick to what comes back, and later queries again to
    // update aria-current. Re-parsing per call hands each of those a fresh
    // throwaway, so every handler the page installs lands on an object nothing
    // else will ever see - and the page looks like it rendered and does nothing.
    _kids: [],
    get innerHTML() { return this._html || ""; },
    set innerHTML(v) { this._html = v; this._kids = parseChildren(v); },
    textContent: "",
    hidden: false,
    onclick: null,
    onmousemove: null,
    onmouseleave: null,
    width: 0,
    height: 0,
    clientWidth: 900,
    clientHeight: 300,
    style: {},
    dataset: {},
    getContext: () => noopCtx,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 900, height: 300 }),
    setAttribute(k, v) { this[k] = v; },
    getAttribute(k) { return this[k]; },
    scrollIntoView() {},
    showModal() { this.open = true; },
    close() { this.open = false; },
    querySelectorAll(sel) {
      const kids = this._kids;
      if (sel.startsWith("[")) {
        const [, k, v] = sel.match(/\[([\w-]+)="([^"]*)"\]/) || [];
        return kids.filter((c) => c.attrs[k] === v);
      }
      const cls = sel.replace(/^\./, "");
      return kids.filter((c) => c.className.split(/\s+/).includes(cls) || sel === "button");
    },
    querySelector(sel) { return this.querySelectorAll(sel)[0] || null; },
  };
}

const els = new Map();
const el = (id) => {
  if (!els.has(id)) els.set(id, makeEl(id));
  return els.get(id);
};

// Payloads are already on `window`; make the page's own loader resolve against
// that rather than the network, so the loader itself is still exercised.
const pending = [];
globalThis.document = {
  head: { appendChild: (node) => pending.push(node) },
  getElementById: el,
  createElement: () => ({ set src(v) { this._src = v; }, get src() { return this._src; } }),
  documentElement: {},
};
globalThis.window = globalThis;
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "#000" });
globalThis.devicePixelRatio = 1;
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.requestAnimationFrame = (fn) => fn();

/* ---- the real files --------------------------------------------------- */

const payloadNames = readdirSync(dir).filter((f) => /^chart_[A-Z][A-Z0-9]*_\d{8}\.js$/.test(f));
for (const name of ["chart_index.js", ...payloadNames]) {
  (0, eval)(readFileSync(join(dir, name), "utf8"));
}
if (!globalThis.CHART_INDEX) throw new Error("chart_index.js defined no CHART_INDEX");

const html = readFileSync(join(dir, "chart.html"), "utf8");
const body = html.slice(html.lastIndexOf("<script>") + 8, html.lastIndexOf("</script>"));

// The loader appends a <script> and waits for onload. Nothing fetches here, so
// fire onload once the global it names is present - which is exactly the
// condition the real page is waiting for.
const drained = setInterval(() => {
  while (pending.length) {
    const node = pending.shift();
    if (node.onload) node.onload();
  }
}, 0);

let threw = null;
try {
  (0, eval)(body);
} catch (e) {
  threw = `${e.name}: ${e.message}`;
}

await new Promise((r) => setTimeout(r, 50));

/* ---- what it drew ----------------------------------------------------- */

// Snapshot the first expiry before opening the second. Everything below reads
// live elements, and the click repaints all of them - the header included, which
// is how "expiry" came back as June 2029 on a report about September.
const rail = el("rail").querySelectorAll("button");
const rows = el("rows").querySelectorAll(".row");
const first = {
  railButtons: rail.length,
  railSelected: rail.filter((b) => b.attrs["aria-selected"] === "true").length,
  ladderRows: rows.length,
  ladderCurrent: rows.filter((b) => b.attrs["aria-current"] === "true").length,
  callTitle: el("ceName").textContent,
  putTitle: el("peName").textContent,
  callIv: el("ceIv").textContent,
  forward: el("fwd").textContent,
  expiry: el("expiry").textContent,
  asOf: el("asOf").textContent,
  counts: `${el("count").textContent} ${el("listed").textContent}`,
  verdict: el("verdict").innerHTML,
};

// Open the second expiry, if there is one. That is the whole point of the rail
// and it is also the only path that fetches a second payload, so it is where a
// file/global mismatch or a payload the page cannot read would show up.
let secondVerdict = null;
let secondLadder = null;
if (rail.length > 1 && rail[1].onclick) {
  try {
    // The drain interval is still running, which is what lets this await
    // resolve: openExpiry awaits load(), load() appends a script and waits for
    // onload, and only the interval fires it.
    await rail[1].onclick();
    await new Promise((r) => setTimeout(r, 50));
    secondVerdict = el("verdict").innerHTML;
    secondLadder = el("rows").querySelectorAll(".row").length;
  } catch (e) {
    problems.push(`opening the second expiry threw ${e.name}: ${e.message}`);
  }
}

const report = {
  threw,
  underlyings: CHART_INDEX.underlyings.map((u) => u.name),
  ...first,
  carriedVerdict: secondVerdict,
  carriedLadderRows: secondLadder,
  problems,
};
clearInterval(drained);
console.log(JSON.stringify(report, null, 2));
