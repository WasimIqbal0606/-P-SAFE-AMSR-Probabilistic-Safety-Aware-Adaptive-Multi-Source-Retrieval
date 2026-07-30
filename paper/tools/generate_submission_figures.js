/* Generate publication figures as SVG, PNG, and vector PDF.
 *
 * Run paper/tools/build_evidence_tables.py first. This script uses only
 * results/validated and the generated baseline CSV.
 */

const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");
const sharp = require("sharp");

const ROOT = path.resolve(__dirname, "..", "..");
const RESULTS = path.join(ROOT, "results", "validated");
const OUT = path.join(ROOT, "paper", "figures");
const SUPP = path.join(ROOT, "paper", "supplementary");
fs.mkdirSync(OUT, { recursive: true });
fs.mkdirSync(SUPP, { recursive: true });

const datasets = ["scifact", "fiqa", "nfcorpus", "arguana"];
const datasetLabels = {
  scifact: "SciFact",
  fiqa: "FiQA",
  nfcorpus: "NFCorpus",
  arguana: "ArguAna",
};
const seeds = [42, 123, 2026];
const blue = "#276FBF";
const blueLight = "#9DC3E6";
const orange = "#D17A22";
const olive = "#647A3D";
const pink = "#B85C7A";
const ink = "#1D2733";
const muted = "#5F6B76";
const grid = "#D7DDE3";
const pale = "#F6F8FA";
const white = "#FFFFFF";

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function runDir(dataset, seed, mode) {
  return path.join(RESULTS, dataset, `seed_${seed}`, mode);
}

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function svgDoc(width, height, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="${width}" height="${height}" fill="${white}"/>
  <style>
    text { font-family: Arial, Helvetica, sans-serif; fill: ${ink}; }
    .title { font-size: 34px; font-weight: 700; }
    .subtitle { font-size: 19px; fill: ${muted}; }
    .axis { font-size: 17px; fill: ${muted}; }
    .label { font-size: 18px; }
    .small { font-size: 15px; fill: ${muted}; }
    .value { font-size: 16px; font-weight: 600; }
  </style>
  ${body}
</svg>`;
}

function line(x1, y1, x2, y2, color = grid, width = 2, dash = "") {
  const dashAttr = dash ? ` stroke-dasharray="${dash}"` : "";
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"${dashAttr}/>`;
}

function text(x, y, value, cls = "label", anchor = "start", extra = "") {
  return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}" ${extra}>${esc(value)}</text>`;
}

function rect(x, y, width, height, fill, stroke = "none", radius = 10, sw = 2) {
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
}

function circle(x, y, radius, fill, stroke = white, sw = 3) {
  return `<circle cx="${x}" cy="${y}" r="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
}

function arrow(x1, y1, x2, y2, color = muted, width = 3) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = 11;
  const a1 = angle + Math.PI * 0.82;
  const a2 = angle - Math.PI * 0.82;
  const p1 = `${x2 + head * Math.cos(a1)},${y2 + head * Math.sin(a1)}`;
  const p2 = `${x2 + head * Math.cos(a2)},${y2 + head * Math.sin(a2)}`;
  return `${line(x1, y1, x2, y2, color, width)}<polygon points="${x2},${y2} ${p1} ${p2}" fill="${color}"/>`;
}

function boxWithLines(x, y, width, height, title, lines, accent) {
  let body = rect(x, y, width, height, white, accent, 12, 3);
  body += text(x + width / 2, y + 33, title, "label", "middle", 'font-weight="700"');
  lines.forEach((entry, idx) => {
    body += text(x + width / 2, y + 62 + idx * 24, entry, "small", "middle");
  });
  return body;
}

function architectureFigure() {
  const width = 1800;
  const height = 860;
  let body = text(70, 64, "B-P-SAFE-AMSR Binary Retrieval Routing", "title");
  body += text(
    70,
    98,
    "A calibrated routing layer over existing Dense and Deep Hybrid endpoints",
    "subtitle"
  );

  body += boxWithLines(70, 315, 170, 105, "Query", ["held-out input"], blue);
  body += boxWithLines(
    330,
    205,
    270,
    105,
    "Dense retrieval",
    ["FAISS IndexFlatIP", "top-50 scores"],
    blue
  );
  body += boxWithLines(
    330,
    350,
    270,
    105,
    "Lexical retrieval",
    ["BM25 top-100", "score distribution"],
    orange
  );
  body += boxWithLines(
    330,
    495,
    270,
    105,
    "Semantic neighbours",
    ["cached top-k lists", "degree features"],
    olive
  );
  body += boxWithLines(
    710,
    315,
    270,
    125,
    "Feature vector",
    ["25 query/retrieval signals", "train-only standardization"],
    blue
  );
  body += boxWithLines(
    1085,
    205,
    300,
    180,
    "Predictive models",
    [
      "P(gain), P(regression)",
      "expected ΔnDCG@10",
      "expected latency",
      "sigmoid calibration objective",
    ],
    pink
  );
  body += boxWithLines(
    1085,
    465,
    300,
    145,
    "Utility and mode gate",
    ["U(A6 | x) versus Dense=0", "thresholds + logged override", "validation-selected settings"],
    orange
  );
  body += boxWithLines(
    1505,
    225,
    230,
    105,
    "A0: Dense",
    ["return Dense ranking"],
    blue
  );
  body += boxWithLines(
    1505,
    500,
    230,
    135,
    "A6: Deep Hybrid",
    ["Dense + BM25 + graph", "fusion + Cross-Encoder"],
    orange
  );

  body += arrow(240, 367, 330, 257);
  body += arrow(240, 367, 330, 402);
  body += arrow(240, 367, 330, 547);
  body += arrow(600, 257, 710, 352);
  body += arrow(600, 402, 710, 377);
  body += arrow(600, 547, 710, 405);
  body += arrow(980, 360, 1085, 295);
  body += arrow(1235, 385, 1235, 465);
  body += arrow(1385, 525, 1505, 277, blue);
  body += arrow(1385, 550, 1505, 567, orange);
  body += text(1435, 365, "fallback", "small", "middle");
  body += text(1440, 545, "escalate", "small", "middle");

  body += rect(70, 708, 1665, 86, pale, grid, 10, 1);
  body += text(
    100,
    742,
    "Scope:",
    "value",
    "start"
  );
  body += text(
    180,
    742,
    "binary Dense-versus-Deep-Hybrid routing; retrieval-quality regression risk, not content-safety risk.",
    "small"
  );
  body += text(
    100,
    773,
    "Graph construction:",
    "value"
  );
  body += text(
    275,
    773,
    "FAISS flat inner-product top-k neighbour lists cached as an expansion graph; no HNSW index.",
    "small"
  );
  return { name: "fig1_architecture", width, height, svg: svgDoc(width, height, body) };
}

function qualityLatencyFigure() {
  const width = 1500;
  const height = 920;
  const left = 145;
  const right = 90;
  const top = 150;
  const bottom = 120;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const rows = datasets.map((dataset) => {
    const values = seeds.map((seed) => readJson(path.join(runDir(dataset, seed, "high_recall"), "extended_metrics.json")));
    const xs = values.map((item) => 100 * item.latency_saving_vs_best_hybrid);
    const ys = values.map((item) => item.psafe_ndcg - item.dense_ndcg);
    const avg = (array) => array.reduce((sum, value) => sum + value, 0) / array.length;
    const sd = (array) => {
      const m = avg(array);
      return Math.sqrt(array.reduce((sum, value) => sum + (value - m) ** 2, 0) / (array.length - 1));
    };
    return { dataset, x: avg(xs), xsd: sd(xs), y: avg(ys), ysd: sd(ys) };
  });
  const xMin = 0;
  const xMax = 70;
  const yMin = 0;
  const yMax = 0.065;
  const sx = (value) => left + ((value - xMin) / (xMax - xMin)) * plotW;
  const sy = (value) => top + plotH - ((value - yMin) / (yMax - yMin)) * plotH;
  const colors = [blue, olive, orange, pink];
  const offsets = [
    [18, -18],
    [18, 28],
    [18, -18],
    [18, 28],
  ];

  let body = text(70, 60, "Quality Gain and End-to-End Latency Saving", "title");
  body += text(
    70,
    96,
    "High-recall mode; mean ± sample SD across data-split seeds 42, 123, and 2026",
    "subtitle"
  );
  for (let tick = 0; tick <= 70; tick += 10) {
    body += line(sx(tick), top, sx(tick), top + plotH, grid, 1);
    body += text(sx(tick), top + plotH + 32, `${tick}%`, "axis", "middle");
  }
  for (let tick = 0; tick <= 0.06 + 1e-9; tick += 0.01) {
    body += line(left, sy(tick), left + plotW, sy(tick), grid, 1);
    body += text(left - 18, sy(tick) + 6, tick.toFixed(2), "axis", "end");
  }
  body += line(left, top, left, top + plotH, ink, 2);
  body += line(left, top + plotH, left + plotW, top + plotH, ink, 2);
  body += text(left + plotW / 2, height - 45, "Latency saving vs always-on Deep Hybrid", "label", "middle");
  body += `<text x="40" y="${top + plotH / 2}" class="label" text-anchor="middle" transform="rotate(-90 40 ${top + plotH / 2})">Δ nDCG@10 vs Dense</text>`;
  rows.forEach((row, idx) => {
    const x = sx(row.x);
    const y = sy(row.y);
    const x1 = sx(Math.max(xMin, row.x - row.xsd));
    const x2 = sx(Math.min(xMax, row.x + row.xsd));
    const y1 = sy(Math.min(yMax, row.y + row.ysd));
    const y2 = sy(Math.max(yMin, row.y - row.ysd));
    body += line(x1, y, x2, y, colors[idx], 4);
    body += line(x1, y - 9, x1, y + 9, colors[idx], 3);
    body += line(x2, y - 9, x2, y + 9, colors[idx], 3);
    body += line(x, y1, x, y2, colors[idx], 4);
    body += line(x - 9, y1, x + 9, y1, colors[idx], 3);
    body += line(x - 9, y2, x + 9, y2, colors[idx], 3);
    body += circle(x, y, 12, colors[idx]);
    body += text(
      x + offsets[idx][0],
      y + offsets[idx][1],
      `${datasetLabels[row.dataset]}  ${row.y >= 0 ? "+" : ""}${row.y.toFixed(3)} / ${row.x.toFixed(1)}%`,
      "value"
    );
  });
  body += text(
    left + 10,
    top + 26,
    "Upper-right indicates larger mean quality gain with more measured latency saved.",
    "small"
  );
  return { name: "fig2_quality_latency", width, height, svg: svgDoc(width, height, body) };
}

function activationFigure() {
  const width = 1500;
  const height = 900;
  const left = 130;
  const right = 70;
  const top = 150;
  const bottom = 120;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const modes = ["balanced", "high_recall"];
  const colors = [blue, orange];
  const avg = (array) => array.reduce((sum, value) => sum + value, 0) / array.length;
  const sd = (array) => {
    const m = avg(array);
    return Math.sqrt(array.reduce((sum, value) => sum + (value - m) ** 2, 0) / (array.length - 1));
  };
  const rows = datasets.map((dataset) => ({
    dataset,
    values: modes.map((mode) => {
      const vals = seeds.map(
        (seed) => 100 * readJson(path.join(runDir(dataset, seed, mode), "extended_metrics.json")).hybrid_activation_rate
      );
      return { mean: avg(vals), sd: sd(vals) };
    }),
  }));
  const sy = (value) => top + plotH - (value / 100) * plotH;
  const groupW = plotW / datasets.length;
  const barW = 92;

  let body = text(70, 60, "Hybrid Activation Rate by Dataset and Mode", "title");
  body += text(70, 96, "Mean ± sample SD across three data-split seeds", "subtitle");
  for (let tick = 0; tick <= 100; tick += 20) {
    body += line(left, sy(tick), left + plotW, sy(tick), grid, 1);
    body += text(left - 16, sy(tick) + 6, `${tick}%`, "axis", "end");
  }
  body += line(left, top, left, top + plotH, ink, 2);
  body += line(left, top + plotH, left + plotW, top + plotH, ink, 2);
  body += `<text x="38" y="${top + plotH / 2}" class="label" text-anchor="middle" transform="rotate(-90 38 ${top + plotH / 2})">Hybrid activation rate</text>`;
  rows.forEach((row, datasetIdx) => {
    const center = left + groupW * (datasetIdx + 0.5);
    row.values.forEach((entry, modeIdx) => {
      const x = center + (modeIdx - 0.5) * (barW + 12) - barW / 2;
      const y = sy(entry.mean);
      const h = top + plotH - y;
      body += rect(x, y, barW, h, colors[modeIdx], colors[modeIdx], 2, 1);
      const errorTop = sy(Math.min(100, entry.mean + entry.sd));
      const errorBottom = sy(Math.max(0, entry.mean - entry.sd));
      const mid = x + barW / 2;
      body += line(mid, errorTop, mid, errorBottom, ink, 3);
      body += line(mid - 10, errorTop, mid + 10, errorTop, ink, 3);
      body += line(mid - 10, errorBottom, mid + 10, errorBottom, ink, 3);
      body += text(mid, y - 14, `${entry.mean.toFixed(1)}%`, "value", "middle");
    });
    body += text(center, top + plotH + 38, datasetLabels[row.dataset], "label", "middle");
  });
  body += rect(width - 405, 55, 20, 20, blue, blue, 1, 1);
  body += text(width - 375, 72, "Balanced", "small");
  body += rect(width - 240, 55, 20, 20, orange, orange, 1, 1);
  body += text(width - 210, 72, "High recall", "small");
  return { name: "fig3_activation", width, height, svg: svgDoc(width, height, body) };
}

function parseCsv(file) {
  const lines = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const header = lines[0].split(",");
  return lines.slice(1).map((lineValue) => {
    const values = lineValue.split(",");
    return Object.fromEntries(header.map((key, idx) => [key, values[idx]]));
  });
}

function baselineFigure() {
  const width = 1600;
  const height = 1080;
  const rows = parseCsv(path.join(SUPP, "router_baseline_comparison_seed42.csv"));
  const selected = new Set([
    "Dense-margin",
    "Dense-entropy",
    "Regression-only",
    "Classification-only",
    "B-P-SAFE",
  ]);
  const short = {
    "Dense-margin": "Margin",
    "Dense-entropy": "Entropy",
    "Regression-only": "Regression",
    "Classification-only": "Classification",
    "B-P-SAFE": "B-P-SAFE",
  };
  const methodOrder = ["B-P-SAFE", "Dense-margin", "Dense-entropy", "Regression-only", "Classification-only"];

  let body = text(60, 58, "Selected Router Baselines: Quality–Latency Trade-offs", "title");
  body += text(
    60,
    94,
    "Seed 42; dots show latency saving and labels show ΔnDCG@10 versus Dense; B-P-SAFE uses high-recall mode",
    "subtitle"
  );
  datasets.forEach((dataset, panelIdx) => {
    const col = panelIdx % 2;
    const rowIdx = Math.floor(panelIdx / 2);
    const x0 = 90 + col * 770;
    const y0 = 155 + rowIdx * 445;
    const panelW = 675;
    const panelH = 335;
    const panelRows = rows
      .filter(
      (item) =>
        item.dataset === datasetLabels[dataset] &&
        selected.has(item.method) &&
        (item.method !== "B-P-SAFE" || item.mode === "High recall")
      )
      .sort((a, b) => methodOrder.indexOf(a.method) - methodOrder.indexOf(b.method));
    const axisLeft = x0 + 150;
    const axisW = panelW - 175;
    const sx = (value) => axisLeft + (value / 85) * axisW;
    body += rect(x0 - 18, y0 - 42, panelW + 36, panelH + 85, white, grid, 8, 1);
    body += text(x0, y0 - 13, datasetLabels[dataset], "value");
    for (let tick = 0; tick <= 80; tick += 20) {
      body += line(sx(tick), y0, sx(tick), y0 + panelH - 12, grid, 1);
      body += text(sx(tick), y0 + panelH + 27, `${tick}%`, "axis", "middle");
    }
    panelRows.forEach((item, methodIdx) => {
      const name = short[item.method];
      const y = y0 + 48 + methodIdx * 58;
      const x = sx(100 * Number(item.latency_saving));
      const focal = item.method === "B-P-SAFE";
      const fill = focal ? blue : item.method.includes("Classification") ? olive : "#7B8794";
      body += text(axisLeft - 18, y + 6, name, focal ? "value" : "small", "end");
      body += line(axisLeft, y, axisLeft + axisW, y, "#EDF0F3", 1);
      body += circle(x, y, focal ? 10 : 8, fill);
      const delta = Number(item.delta_dense);
      const labelX = x > axisLeft + axisW - 110 ? x - 12 : x + 12;
      const anchor = x > axisLeft + axisW - 110 ? "end" : "start";
      body += text(
        labelX,
        y - 12,
        `${delta >= 0 ? "+" : ""}${delta.toFixed(3)} nDCG`,
        focal ? "value" : "small",
        anchor
      );
    });
  });
  body += text(width / 2, height - 25, "Latency saving vs always-on Deep Hybrid", "label", "middle");
  return { name: "fig4_baselines", width, height, svg: svgDoc(width, height, body) };
}

function latencyFigure() {
  const width = 1500;
  const height = 930;
  const directory = runDir("scifact", 42, "high_recall");
  const latency = readJson(path.join(directory, "latency_breakdown.json"));
  const csvLines = fs.readFileSync(path.join(directory, "latency_per_query.csv"), "utf8").trim().split(/\r?\n/);
  const headers = csvLines[0].split(",");
  const featureIdx = headers.indexOf("feature_extraction_ms");
  const featureMean =
    csvLines
      .slice(1)
      .map((lineValue) => Number(lineValue.split(",")[featureIdx]))
      .reduce((sum, value) => sum + value, 0) /
    (csvLines.length - 1);
  const data = [
    ["Dense search", latency.dense_search.mean],
    ["BM25 search", latency.bm25_search.mean],
    ["Feature extraction", featureMean],
    ["Router decision", latency.router_decision.mean],
    ["Graph expansion", latency.graph_expansion.mean],
    ["Fusion", latency.fusion.mean],
    ["Cross-Encoder contribution", latency.cross_encoder.mean],
  ];
  const left = 315;
  const right = 120;
  const top = 185;
  const bottom = 100;
  const plotW = width - left - right;
  const rowH = 74;
  const minLog = -2.2;
  const maxLog = 3.0;
  const sx = (value) => left + ((Math.log10(value) - minLog) / (maxLog - minLog)) * plotW;

  let body = text(60, 58, "Per-Query Latency Components", "title");
  body += text(
    60,
    94,
    "SciFact, seed 42, high-recall mode; Cross-Encoder mean includes zero for non-invoked queries",
    "subtitle"
  );
  [-2, -1, 0, 1, 2, 3].forEach((power) => {
    const x = left + ((power - minLog) / (maxLog - minLog)) * plotW;
    body += line(x, top - 20, x, top + data.length * rowH, grid, 1);
    body += text(x, top + data.length * rowH + 35, `10^${power}`, "axis", "middle");
  });
  data.forEach(([label, value], idx) => {
    const y = top + idx * rowH;
    body += text(left - 20, y + 24, label, "label", "end");
    const x0 = sx(10 ** minLog);
    const x1 = sx(Math.max(value, 10 ** minLog));
    const fill = label.startsWith("Cross") ? orange : blue;
    body += rect(x0, y + 3, Math.max(3, x1 - x0), 31, fill, fill, 2, 1);
    const digits = value < 0.1 ? 3 : value < 10 ? 2 : 1;
    body += text(Math.min(x1 + 12, width - right + 10), y + 27, `${value.toFixed(digits)} ms`, "value");
  });
  body += line(left, top + data.length * rowH + 2, left + plotW, top + data.length * rowH + 2, ink, 2);
  body += text(left + plotW / 2, height - 42, "Mean latency (ms, logarithmic scale)", "label", "middle");
  body += rect(70, 735, 505, 118, pale, grid, 8, 1);
  body += text(95, 772, `Measured total: ${latency.total.mean.toFixed(1)} ms/query`, "value");
  body += text(
    95,
    805,
    `Hybrid activation: ${(100 * latency.cross_encoder_calls_per_query).toFixed(1)}%`,
    "small"
  );
  body += text(
    95,
    833,
    `Conditional Cross-Encoder mean ≈ ${(latency.cross_encoder.mean / latency.cross_encoder_calls_per_query).toFixed(1)} ms/call`,
    "small"
  );
  return { name: "fig5_latency", width, height, svg: svgDoc(width, height, body) };
}

function multiseedFigure() {
  const width = 1700;
  const height = 900;
  const top = 160;
  const bottom = 120;
  const panelW = 700;
  const panelH = 575;
  const xStart = [125, 965];
  const colors = [blue, olive, orange];
  const jitter = [-26, 0, 26];

  let body = text(60, 58, "High-Recall Sensitivity Across Data-Split Seeds", "title");
  body += text(
    60,
    94,
    "Per-seed nDCG@10 and hybrid activation; lines connect only repeated split seeds within each dataset",
    "subtitle"
  );

  function panel(panelIdx, key, yMin, yMax, titleValue, formatValue) {
    const x0 = xStart[panelIdx];
    const y0 = top;
    const sx = (datasetIdx, seedIdx) =>
      x0 + (datasetIdx + 0.5) * (panelW / datasets.length) + jitter[seedIdx];
    const sy = (value) => y0 + panelH - ((value - yMin) / (yMax - yMin)) * panelH;
    body += rect(x0 - 35, y0 - 45, panelW + 70, panelH + 95, white, grid, 8, 1);
    body += text(x0, y0 - 14, titleValue, "value");
    const ticks = panelIdx === 0 ? [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] : [0, 20, 40, 60, 80, 100];
    ticks.forEach((tick) => {
      body += line(x0, sy(tick), x0 + panelW, sy(tick), grid, 1);
      body += text(x0 - 12, sy(tick) + 6, formatValue(tick), "axis", "end");
    });
    datasets.forEach((dataset, datasetIdx) => {
      const vals = seeds.map((seed) => {
        const item = readJson(path.join(runDir(dataset, seed, "high_recall"), "extended_metrics.json"));
        return key === "ndcg" ? item.psafe_ndcg : 100 * item.hybrid_activation_rate;
      });
      for (let seedIdx = 0; seedIdx < seeds.length - 1; seedIdx++) {
        body += line(
          sx(datasetIdx, seedIdx),
          sy(vals[seedIdx]),
          sx(datasetIdx, seedIdx + 1),
          sy(vals[seedIdx + 1]),
          "#AAB3BC",
          2
        );
      }
      vals.forEach((value, seedIdx) => {
        body += circle(sx(datasetIdx, seedIdx), sy(value), 8, colors[seedIdx], white, 2);
      });
      body += text(
        x0 + (datasetIdx + 0.5) * (panelW / datasets.length),
        y0 + panelH + 35,
        datasetLabels[dataset],
        "axis",
        "middle"
      );
    });
  }
  panel(0, "ndcg", 0.3, 0.8, "B-P-SAFE nDCG@10", (value) => value.toFixed(1));
  panel(1, "activation", 0, 100, "Hybrid activation rate", (value) => `${value}%`);

  seeds.forEach((seed, idx) => {
    const x = 565 + idx * 170;
    body += circle(x, 828, 8, colors[idx], white, 2);
    body += text(x + 16, 834, `Seed ${seed}`, "small");
  });
  return { name: "fig6_multiseed", width, height, svg: svgDoc(width, height, body) };
}

async function writeFigure(browser, figure) {
  const svgPath = path.join(OUT, `${figure.name}.svg`);
  const pngPath = path.join(OUT, `${figure.name}.png`);
  const pdfPath = path.join(OUT, `${figure.name}.pdf`);
  fs.writeFileSync(svgPath, figure.svg, "utf8");
  await sharp(Buffer.from(figure.svg)).png().toFile(pngPath);

  const page = await browser.newPage({ viewport: { width: figure.width, height: figure.height } });
  await page.setContent(
    `<html><head><style>@page{size:${figure.width}px ${figure.height}px;margin:0}html,body{margin:0;padding:0;width:${figure.width}px;height:${figure.height}px}</style></head><body>${figure.svg}</body></html>`
  );
  await page.pdf({
    path: pdfPath,
    width: `${figure.width}px`,
    height: `${figure.height}px`,
    margin: { top: "0px", right: "0px", bottom: "0px", left: "0px" },
    printBackground: true,
    preferCSSPageSize: true,
  });
  await page.close();
  return { svgPath, pngPath, pdfPath };
}

async function main() {
  const figures = [
    architectureFigure(),
    qualityLatencyFigure(),
    activationFigure(),
    baselineFigure(),
    latencyFigure(),
    multiseedFigure(),
  ];
  const browser = await chromium.launch({ headless: true });
  const outputs = [];
  try {
    for (const figure of figures) {
      outputs.push(await writeFigure(browser, figure));
      process.stdout.write(`Wrote ${figure.name}.{svg,png,pdf}\n`);
    }
  } finally {
    await browser.close();
  }
  const audit = [
    "# Figure Audit",
    "",
    "| Figure | Original issue | Fix | Status |",
    "|---|---|---|---|",
    "| 1 | Dark, oversized architecture with disconnected arrows and ambiguous pre/post-routing flow | Rebuilt as a light vector pipeline showing signals, models, utility gate, binary endpoints, and graph truth | PASS |",
    "| 2 | Dark raster Pareto plot mixed five datasets and raw nDCG scales; labels were difficult at column width | Rebuilt as mean quality gain versus measured latency saving with error bars for the four manuscript datasets | PASS |",
    "| 3 | Dark raster activation chart included an out-of-scope fifth dataset and a legend over the plot | Rebuilt as a light grouped chart with direct mean/SD annotations | PASS |",
    "| 4 | Prior action-distribution figure implied autonomous detection without exposing baseline competitiveness | Replaced with selected-router quality–latency small multiples; full values remain in tables | PASS |",
    "| 5 | Previous hard-query annotations were crowded and depended on stale values | Replaced with source-backed log-scale latency components and explicit activation/conditional timing | PASS |",
    "| 6 | Dark multi-seed bars were large and visually heavy | Rebuilt as compact per-seed dot-and-line panels for nDCG and activation | PASS |",
    "",
    "All six figures are exported as SVG, vector PDF, and PNG. Captions in the manuscript identify the data scope and metric definition.",
    "",
  ];
  fs.writeFileSync(path.join(SUPP, "figure_audit.md"), audit.join("\n"), "utf8");
  return outputs;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
