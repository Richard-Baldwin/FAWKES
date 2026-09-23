"""The arena: an animated movement simulator, side by side.

Baseline vs champion on the same world, same targets, same seed - the picture
that answers "did the training actually change how it moves?". Emits one
self-contained HTML file (no server, no dependencies) with play/pause, speed,
a scrubber and a practice/DR world toggle.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fawkes.envs.cascade import BASELINE_KNOBS, knobs_to_vec
from fawkes.envs.env import MovementEnv
from fawkes.envs.surface import DRFamily, Z_NAMES
# CHUNK-2


def _seg_dist(p, a, b) -> float:
    """Distance from point p to the segment a-b (cross-track)."""
    ab = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return float(np.linalg.norm(np.asarray(p) - np.asarray(a)))
    t = max(0.0, min(1.0, float(np.dot(np.asarray(p) - np.asarray(a), ab) / L2)))
    proj = np.asarray(a, dtype=float) + t * ab
    return float(np.linalg.norm(np.asarray(p) - proj))


def record_run(env: MovementEnv, knobs, seed: int, rows: int = 2, max_ticks: int = 900) -> dict:
    """One batch episode, recorded per tick: pose, heading, speed, cross-track."""
    env.reset(rows, seed)
    env.cascade.set_knobs_per_row(np.tile(knobs, (rows, 1)))
    series: list[list[dict]] = [[] for _ in range(rows)]

    def sample(i: int) -> dict:
        p = env.pos[i]
        return {
            "x": round(float(p[0]), 4),
            "y": round(float(p[1]), 4),
            "h": round(float(env.th[i]), 4),
            "v": round(float(np.hypot(env.vel_body[i, 0], env.vel_body[i, 1])), 3),
            "c": round(_seg_dist(p, env.leg_start[i], env.target[i, 0:2]), 4),
        }

    for _ in range(max_ticks):
        for i in range(rows):
            series[i].append(sample(i))
        _, _, done, _ = env.step(None)
        if np.all(done):
            for i in range(rows):
                series[i].append(sample(i))
            break
    targets = [
        [[round(float(t[0]), 4), round(float(t[1]), 4)] for t in env.task.targets[i]]
        for i in range(rows)
    ]
    endpoints = [[round(v * 1000, 1) for v in env.metrics[i]["endpoint"]] for i in range(rows)]
    return {
        "series": series,
        "targets": targets,
        "endpoints": endpoints,
        "z": [round(float(v), 3) for v in env.z()[0]],
    }
# CHUNK-3


def _champion_knobs(evidence_dir: Path):
    champ_path = Path(evidence_dir) / "cem_champion.json"
    if champ_path.exists():
        champ = json.loads(champ_path.read_text(encoding="utf-8"))
        return knobs_to_vec({k: float(v) for k, v in champ["knobs"].items()})
    return knobs_to_vec(BASELINE_KNOBS)


def build_arena(tag: str, out_html: str | None = None,
                practice_seed: int = 9101, dr_seed: int = 9003, rows: int = 2) -> Path:
    """Record baseline-vs-champion on a practice world and a DR world, emit HTML."""
    from fawkes import paths as P

    out_dir = P.EVIDENCE / tag
    cem = _champion_knobs(out_dir)
    base = knobs_to_vec(BASELINE_KNOBS)
    data = {}
    for name, family, seed in (
        ("practice", DRFamily.practice(), practice_seed),
        ("dr", DRFamily(), dr_seed),
    ):
        env = MovementEnv(family=family, legs=2)
        data[name] = {
            "family": family.family_id(),
            "seed": seed,
            "baseline": record_run(env, base, seed, rows=rows),
            "champion": record_run(env, cem, seed, rows=rows),
        }
    data["z_names"] = list(Z_NAMES)

    html = _ARENA_TEMPLATE.replace("__DATA__", json.dumps(data))
    html = html.replace("__TAG__", tag)
    out = Path(out_html) if out_html else out_dir / "arena.html"
    out.write_text(html, encoding="utf-8")
    return out
# CHUNK-4

_ARENA_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FAWKES arena - __TAG__</title>
<style>
  body { background:#101418; color:#d8dee6; font-family:Consolas,monospace; margin:0; padding:14px; }
  h1 { font-size:17px; margin:0 0 6px; font-weight:600; }
  h1 span { color:#4da3ff; }
  #bar { display:flex; gap:8px; align-items:center; margin:10px 0; flex-wrap:wrap; }
  button, select { background:#1b222b; color:#d8dee6; border:1px solid #2c3948;
    border-radius:6px; padding:5px 12px; font-family:inherit; font-size:13px; cursor:pointer; }
  button:hover, select:hover { background:#242e3a; }
  button.on { background:#2d4a6b; border-color:#4da3ff; }
  #scrub { flex:1; min-width:180px; }
  #wrap { display:flex; gap:14px; flex-wrap:wrap; }
  .panel { background:#141920; border:1px solid #232d39; border-radius:10px; padding:10px; }
  .panel h2 { font-size:13px; margin:0 0 6px; font-weight:600; }
  .panel.base h2 { color:#9aa7b5; } .panel.champ h2 { color:#4da3ff; }
  .hud { font-size:12px; color:#8b98a8; margin-top:6px; min-height:16px; }
  #foot { margin-top:12px; font-size:12px; color:#8b98a8; line-height:1.6; }
  canvas { display:block; background:#0b0e12; border-radius:6px; }
</style>
</head>
<body>
<h1><span>FAWKES arena</span> - __TAG__ - field-proven baseline vs trained champion, same world, same targets</h1>
<div id="bar">
  <button id="play">Pause</button>
  <button id="replay">Replay</button>
  <select id="speed">
    <option value="0.25">0.25x</option>
    <option value="0.5">0.5x</option>
    <option value="1" selected>1x</option>
    <option value="2">2x</option>
    <option value="4">4x</option>
  </select>
  <button id="wpractice" class="on">Practice world</button>
  <button id="wdr">DR world</button>
  <input id="scrub" type="range" min="0" max="100" value="0">
</div>
<div id="wrap">
  <div class="panel base"><h2>field-proven baseline</h2><canvas id="cbase" width="470" height="350"></canvas><div class="hud" id="hbase"></div></div>
  <div class="panel champ"><h2>FAWKES CEM champion</h2><canvas id="cchamp" width="470" height="350"></canvas><div class="hud" id="hchamp"></div></div>
</div>
<div id="foot"></div>
<script>
"use strict";
const DATA = __DATA__;
const COLORS = ["#4da3ff", "#ffd24d"];
const XMIN = -2.45, XMAX = 2.45, YMIN = -1.7, YMAX = 1.7;
const W = 470, H = 350;
let world = "practice", playing = true, tick = 0, speed = 1.0, last = 0;
const cbase = document.getElementById("cbase").getContext("2d");
const cchamp = document.getElementById("cchamp").getContext("2d");

function maxTick() {
  const r = DATA[world];
  let m = 0;
  for (const run of [r.baseline, r.champion])
    for (const s of run.series) m = Math.max(m, s.length - 1);
  return m;
}
function px(x) { return (x - XMIN) / (XMAX - XMIN) * W; }
function py(y) { return H - (y - YMIN) / (YMAX - YMIN) * H; }

function field(g) {
  g.clearRect(0, 0, W, H);
  g.strokeStyle = "#2c3948"; g.lineWidth = 1;
  g.strokeRect(px(XMIN), py(YMAX), W, H);
  g.strokeStyle = "#3a4a5c"; g.lineWidth = 1.5;
  g.strokeRect(px(-2.25), py(1.5), px(2.25) - px(-2.25), py(-1.5) - py(1.5));
  g.beginPath(); g.moveTo(px(0), py(1.5)); g.lineTo(px(0), py(-1.5)); g.stroke();
  g.beginPath(); g.arc(px(0), py(0), 26, 0, 7); g.stroke();
  g.strokeRect(px(-2.25), py(0.5), px(-1.75) - px(-2.25), py(-0.5) - py(0.5));
  g.strokeRect(px(1.75), py(0.5), px(2.25) - px(1.75), py(-0.5) - py(0.5));
}

function robot(g, s, col) {
  const x = px(s.x), y = py(s.y);
  g.save(); g.translate(x, y); g.rotate(s.h);
  g.fillStyle = col;
  g.beginPath(); g.moveTo(9, 0); g.lineTo(-6, 6); g.lineTo(-6, -6); g.closePath(); g.fill();
  g.restore();
}

function drawPanel(g, run, t) {
  field(g);
  for (let i = 0; i < run.series.length; i++) {
    const s = run.series[i], col = COLORS[i % COLORS.length];
    g.strokeStyle = col; g.lineWidth = 1.2; g.globalAlpha = 0.55;
    g.beginPath();
    const step = Math.max(1, Math.floor(s.length / 300));
    for (let k = 0; k <= Math.min(t, s.length - 1); k += step) {
      if (k === 0) g.moveTo(px(s[k].x), py(s[k].y));
      else g.lineTo(px(s[k].x), py(s[k].y));
    }
    g.stroke(); g.globalAlpha = 1.0;
    for (const tg of run.targets[i]) {
      g.strokeStyle = col; g.lineWidth = 1.4;
      g.beginPath(); g.moveTo(px(tg[0]) - 6, py(tg[1]) - 6); g.lineTo(px(tg[0]) + 6, py(tg[1]) + 6);
      g.moveTo(px(tg[0]) + 6, py(tg[1]) - 6); g.lineTo(px(tg[0]) - 6, py(tg[1]) + 6); g.stroke();
    }
    const at = s[Math.min(t, s.length - 1)];
    robot(g, at, col);
  }
}

function hud(run, t) {
  const parts = [];
  for (let i = 0; i < run.series.length; i++) {
    const s = run.series[i][Math.min(t, run.series[i].length - 1)];
    parts.push(`r${i}: ${(s.v).toFixed(2)} m/s, cross ${(s.c * 1000).toFixed(0)} mm`);
  }
  return parts.join(" &nbsp;|&nbsp; ");
}
function foot() {
  const r = DATA[world];
  let html = `<b>world:</b> ${r.family} (seed ${r.seed}) &nbsp; <b>z:</b> ` +
    DATA.z_names.map((n, i) => `${n}=${r.champion.z[i]}`).join(", ") + "<br>";
  for (const name of ["baseline", "champion"]) {
    const run = r[name];
    html += `<b>${name}</b> endpoint per leg: ` +
      run.series.map((_, i) => `r${i} [${run.endpoints[i].join(", ")}] mm`).join(" &nbsp; ") + "<br>";
  }
  html += "tick = 20 ms of simulated time; two robots per panel, same colours = same robot.";
  document.getElementById("foot").innerHTML = html;
}

const playBtn = document.getElementById("play");
const scrubEl = document.getElementById("scrub");

function draw() {
  const r = DATA[world], m = maxTick();
  drawPanel(cbase, r.baseline, Math.round(tick));
  drawPanel(cchamp, r.champion, Math.round(tick));
  document.getElementById("hbase").innerHTML = "t=" + (tick * 0.02).toFixed(2) + "s &nbsp; " + hud(r.baseline, Math.round(tick));
  document.getElementById("hchamp").innerHTML = "t=" + (tick * 0.02).toFixed(2) + "s &nbsp; " + hud(r.champion, Math.round(tick));
  scrubEl.max = m; scrubEl.value = Math.round(tick);
}

function loop(ts) {
  const dt = last ? ts - last : 16; last = ts;
  if (playing) {
    tick += speed * dt / 20.0;
    const m = maxTick();
    if (tick >= m) { tick = m; playing = false; playBtn.textContent = "Play"; }
  }
  draw();
  requestAnimationFrame(loop);
}

playBtn.onclick = () => {
  if (!playing && tick >= maxTick()) tick = 0;
  playing = !playing; playBtn.textContent = playing ? "Pause" : "Play";
};
document.getElementById("replay").onclick = () => { tick = 0; playing = true; playBtn.textContent = "Pause"; };
document.getElementById("speed").onchange = (e) => { speed = parseFloat(e.target.value); };
scrubEl.oninput = (e) => { tick = parseInt(e.target.value, 10); playing = false; playBtn.textContent = "Play"; };
for (const [id, w] of [["wpractice", "practice"], ["wdr", "dr"]]) {
  document.getElementById(id).onclick = () => {
    world = w; tick = 0; playing = true; playBtn.textContent = "Pause";
    document.getElementById("wpractice").classList.toggle("on", w === "practice");
    document.getElementById("wdr").classList.toggle("on", w === "dr");
    foot();
  };
}
foot();
requestAnimationFrame(loop);
</script>
</body>
</html>
""" 
