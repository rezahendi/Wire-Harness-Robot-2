/* Mission Control for the wire-harness cell: start builds, watch them live in 3D, replay them. */
(function () {
'use strict';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const reduceMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.status + ' ' + r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  return r.json();
}

// ------------------------------------------------------------------ words
const PHASE_SHORT = {
  pick_approach: 'move over the wire', pick_descend: 'touch down', pick_close: 'close on the wire',
  route_lift: 'lift the wire', route_transit: 'carry it over the fork', route_descend: 'lower past the fork',
  route_seat: 'snap it into the clip', route_release: 'let go and back off', route_realign: 'lift clear and line up',
  connector_tip: 'tip the connector over', connector_relocate: 'move the connector', connector_approach: 'move over the connector',
  connector_grasp: 'grasp the connector', connector_transit: 'carry it to the holder', connector_align: 'line up over the pocket',
  connector_descend: 'lower until it touches', connector_search: 'spiral search', connector_press: 'press until it latches',
  connector_release: 'let go and check', retreat: 'park the arm', learned_policy: 'camera images in, motions out',
  disturbance: 'disturbance', start: 'idle', init: 'idle', done: 'idle'
};
const IDLE = new Set(['start', 'init', 'done']);
const ASSIST_PHASES = new Set(['route_descend', 'route_seat', 'route_release', 'route_realign']);
const ACTOR = {
  groot: { text: 'GR00T N1.7 is driving', short: 'GR00T', color: '--groot' },
  assist: { text: 'Force-controlled seating', short: 'Seating', color: '--assist' },
  expert: { text: 'Force-guided expert skill', short: 'Expert', color: '--expert' },
  planner: { text: 'Nemotron is deciding', short: 'Nemotron', color: '--planner' },
  warn: { text: 'Disturbance', short: 'Disturbance', color: '--warn' },
  idle: { text: 'Idle', short: 'Idle', color: '--muted' }
};
const OUTCOME_TEXT = {
  routed: 'routed', seated: 'seated', graspable: 'graspable', home: 'parked', timeout: 'ran out of time',
  grasp_slipped: 'connector slipped', wire_not_retained: 'wire not retained', grasp_failed: 'grasp failed',
  lying_flat: 'lying flat'
};
// the session's refusals: nothing moved, the reason is in the call's messages
const REFUSALS = new Set(['previous_fork_not_seated', 'forks_not_routed', 'connector_standing', 'connector_blocked',
                          'infeasible_spec', 'unknown_fork']);
const SCENARIO_TEXT = { popped_wire: 'wire pulled out', slip_on_insert: 'connector slips', both: 'wire pulled out, connector slips' };
const EMPTY_DECISIONS = '<li class="empty-note">The planner\'s decisions appear here.</li>';

function callTitle(c) {
  const a = c.arguments || {};
  switch (c.name) {
    case 'get_status': return 'Look at the cell';
    case 'route_fork': return `Route the wire into ${a.fork_id}` + (a.attempt ? ' (retry)' : '');
    case 'insert_connector': return 'Insert the connector';
    case 'relocate_connector': return 'Move the connector';
    case 'tip_connector': return 'Tip the connector over';
    case 'retreat': return 'Park the arm';
    case 'inspect': return 'Camera check of every fixture';
    case 'finish': return 'Finish: hand in the report';
    default: return String(c.name || '').replace(/_/g, ' ');
  }
}
function segLabel(c) {          // a call's name on the timeline (calls that move the robot)
  const a = c.arguments || {};
  switch (c.name) {
    case 'route_fork': return (a.fork_id || 'fork') + (a.attempt ? ' again' : '');
    case 'insert_connector': return 'insert';
    case 'relocate_connector': return 'move';
    case 'tip_connector': return 'tip';
    case 'retreat': return 'park';
    case 'inspect': return 'look';
    default: return null;
  }
}
function outcomeOf(c) {
  const r = c.result;
  if (!r) return { text: 'running', cls: 'run' };
  if (c.name === 'finish') return { text: r.ok === false ? 'not accepted' : 'accepted', cls: r.ok === false ? 'bad' : 'ok' };
  if (c.name === 'get_status') return { text: 'state read', cls: 'ok' };
  if (c.name === 'inspect') return { text: 'checked', cls: 'ok' };
  if (r.error || REFUSALS.has(r.outcome)) return { text: 'refused', cls: 'bad' };
  const o = r.outcome || 'done';
  return { text: OUTCOME_TEXT[o] || String(o).replace(/_/g, ' '), cls: r.ok === false ? 'bad' : 'ok' };
}
const ROBOT_SKILLS = ['route_fork', 'insert_connector', 'relocate_connector', 'tip_connector', 'retreat'];
function executorOf(c, k) {     // who moved the robot in this call: what its result says, or what its frames showed
  const r = c.result || {};
  if (r.error || REFUSALS.has(r.outcome)) return null;
  const by = String(r.executed_by || '');
  const seen = V.seen.get(k);
  if (/seating assist/i.test(by) || (seen && seen.has('assist'))) return 'assist';
  if (/^GR00T/.test(by) || (seen && seen.has('groot'))) return 'groot';
  if ((seen && seen.has('expert')) || (c.result && ROBOT_SKILLS.includes(c.name))) return 'expert';
  return null;
}
function fmtTime(s) {
  if (s == null || !isFinite(s)) return '–';
  return s >= 60 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${(+s).toFixed(1)} s`;
}
function wilson(k, n, z = 1.96) {
  if (!n) return [0, 0];
  const p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d;
  const h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}
function modelName(m) {
  const s = String(m || '').split('/').pop();
  if (!s) return 'Nemotron';
  let mm = s.match(/nemotron-3[-_](super|ultra|nano)(?:[-_](\d+)b)?/i);
  if (mm) return `Nemotron 3 ${mm[1][0].toUpperCase()}${mm[1].slice(1).toLowerCase()}${mm[2] ? ' ' + mm[2] + 'B' : ''}`;
  mm = s.match(/llama-3[._]3-nemotron-super-49b/i);
  if (mm) return 'Llama 3.3 Nemotron Super 49B';
  return s;
}
function specName(file) {
  const base = String(file || '').split('/').pop();
  const s = S.specs.find((x) => x.file === base);
  return s ? s.name : base.replace(/\.ya?ml$/, '');
}

// ------------------------------------------------------------------ app state
const S = { status: {}, specs: [], spec: null, opts: { planner: 'nemotron', groot: '1', vision: '0', pace: '1' } };
let V = null;              // the open build: everything its events said so far

function newView(info, live) {
  return {
    id: info.id, info, live, scene: null, frames: [], logs: [], calls: [], dists: [], checks: [],
    pendingReason: '', grootCalls: new Set(), seen: new Map(), dirty: false, done: null, failed: null, infeasible: null,
    planner: '', model: '', t: 0, playing: false, speed: 1, ws: null, thinking: false,
    cur: -2, pinned: new Map()     // the call shown now; decisions the viewer opened or closed by hand
  };
}

// ------------------------------------------------------------------ events of a build
function handleEvent(ev) {
  if (!V) return;
  switch (ev.type) {
    case 'scene':
      V.scene = ev;
      initScene(ev);
      $('empty').hidden = true;
      break;
    case 'frame': addFrame(ev); break;
    case 'planner':
      V.planner = ev.planner; V.model = ev.model || '';
      $('planner-name').textContent = ev.planner === 'nemotron' ? modelName(V.model) : 'scripted policy, no LLM';
      $('planner-name').title = ev.planner === 'nemotron' ? (V.model || '') + ' on Nebius Token Factory' : '';
      V.thinking = V.live && ev.planner === 'nemotron';
      renderDecisions();
      break;
    case 'plan':
      V.pendingReason = ((ev.reasoning || '') + (ev.content ? '\n' + ev.content : '')).trim();
      break;
    case 'call_start':
      V.calls[ev.index] = { name: ev.name, arguments: ev.arguments || {}, t_start: ev.sim_time, t_end: null, result: null,
                            reasoning: V.pendingReason, checks: [] };
      V.pendingReason = '';
      V.thinking = false;
      renderDecisions();
      break;
    case 'call_end': {
      const c = V.calls[ev.index] || (V.calls[ev.index] = { name: ev.name, arguments: {}, t_start: ev.sim_time, reasoning: '', checks: [] });
      c.result = ev.result || {};
      c.t_end = ev.sim_time;
      V.thinking = V.live && V.planner === 'nemotron' && ev.name !== 'finish';
      renderDecisions();
      break;
    }
    case 'disturbance': V.dists.push({ label: ev.label, t: ev.sim_time }); break;
    case 'visual_check': {
      const last = V.calls[V.calls.length - 1];
      if (last) last.checks.push(ev);
      V.checks.push(ev);
      if (V.live) renderDecisions();
      break;
    }
    case 'infeasible': V.infeasible = ev.issues || []; renderResult(); break;
    case 'done':
    case 'failed':
      V[ev.type] = ev; V.thinking = false;
      if (V.info) V.info.status = ev.type;
      renderResult(); renderDecisions();
      if (V.live) { finishLive(); loadPast(); }
      break;
    default: break;
  }
}

function addFrame(ev) {
  const raw = atob(ev.pose);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  let actor;
  if (ev.mode === 1) actor = 'warn';
  else if (ev.phase === 'learned_policy') { actor = 'groot'; V.grootCalls.add(ev.call); }
  else if (V.grootCalls.has(ev.call) && ASSIST_PHASES.has(ev.phase)) actor = 'assist';
  else if (IDLE.has(ev.phase)) actor = 'idle';
  else actor = 'expert';
  let seen = V.seen.get(ev.call);
  if (!seen) V.seen.set(ev.call, (seen = new Set()));
  if (!seen.has(actor)) { seen.add(actor); V.dirty = true; }     // a call's executor pill can change
  V.frames.push({ t: ev.t, pose: new Int16Array(bytes.buffer), tcp: ev.tcp, wrench: ev.wrench, grip: ev.grip,
                  phase: ev.phase, call: ev.call, truth: ev.truth, mode: ev.mode, pokes: ev.pokes, actor });
  for (const m of ev.log || []) V.logs.push(m);
  if (V.live) V.t = ev.t;
}

// ------------------------------------------------------------------ 3D
let R3 = null;             // renderer, created once
let T3 = null;             // this build's scene
function getRenderer() {
  if (R3 !== null) return R3;
  if (!window.THREE || !THREE.OrbitControls) { R3 = false; return R3; }
  try {
    R3 = new THREE.WebGLRenderer({ canvas: $('canvas'), antialias: true });
    R3.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    R3.shadowMap.enabled = true;
    R3.shadowMap.type = THREE.PCFSoftShadowMap;
  } catch (err) { R3 = false; }
  return R3;
}
function disposeScene() {
  if (!T3) return;
  T3.controls.dispose();
  T3.scene.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) [].concat(o.material).forEach((m) => m.dispose()); });
  T3 = null;
  $('tags').innerHTML = '';
}
// the home view: the work area (forks, holder, clamp) seen from the front right, as large as the stage allows
function fitHome(sc, aspect) {
  const xy = sc.forks.map((f) => f.pos).concat([sc.holder]);
  const ci = sc.bodies.indexOf('clamp');
  if (ci >= 0 && sc.static[ci]) xy.push(sc.static[ci]);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const p of xy) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
  const pad = 0.07;
  x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;
  const pts = [];
  for (const x of [x0, x1]) for (const y of [y0, y1]) for (const z of [sc.board_z, sc.board_z + 0.09]) pts.push(new THREE.Vector3(x, y, z));
  const target = new THREE.Vector3((x0 + x1) / 2 - 0.02, (y0 + y1) / 2, sc.board_z + 0.04);
  const dir = new THREE.Vector3(0.56, -0.4, 0.47).normalize();
  const cam = new THREE.PerspectiveCamera(38, aspect, 0.01, 30);
  cam.up.set(0, 0, 1);
  const v = new THREE.Vector3();
  let lo = 0.15, hi = 5;
  for (let it = 0; it < 24; it++) {
    const d = (lo + hi) / 2;
    cam.position.copy(target).addScaledVector(dir, d);
    cam.lookAt(target);
    cam.updateMatrixWorld(true);
    let fits = true;
    for (const p of pts) {
      v.copy(p).project(cam);
      if (Math.abs(v.x) > 0.9 || v.y > 0.72 || v.y < -0.88) { fits = false; break; }
    }
    if (fits) hi = d; else lo = d;
  }
  return { pos: target.clone().addScaledVector(dir, hi * 1.12), target };   // a little room for the arm
}
function applyHome() {
  if (!T3 || !V || !V.scene) return;
  T3.home = fitHome(V.scene, T3.camera.aspect);
  if (T3.follow) return;
  T3.camera.position.copy(T3.home.pos);
  T3.controls.target.copy(T3.home.target);
  T3.controls.update();
}
function initScene(sc) {
  disposeScene();
  const renderer = getRenderer();
  if (!renderer) {
    $('empty').hidden = false;
    $('empty').innerHTML = '<div><h3>No 3D here</h3><p>This browser has no WebGL, so the cell cannot be drawn. The panels still follow the build.</p></div>';
    return;
  }
  THREE.Object3D.DefaultUp.set(0, 0, 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1.6, 0.01, 30);
  camera.up.set(0, 0, 1);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = !reduceMotion;
  controls.dampingFactor = 0.12;
  controls.minDistance = 0.08;
  controls.maxDistance = 4;
  controls.maxPolarAngle = Math.PI * 0.495;

  const look = new THREE.Vector3(sc.holder[0], sc.holder[1], sc.board_z);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6f757c, 0.62));
  const sun = new THREE.DirectionalLight(0xffffff, 0.66);
  sun.position.set(look.x + 0.7, look.y - 0.9, 2.2);
  sun.target.position.set(0.45, 0, sc.board_z);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -0.85, right: 0.85, top: 0.85, bottom: -0.85, near: 0.3, far: 5 });
  sun.shadow.bias = -0.0004;
  scene.add(sun, sun.target);
  const fill = new THREE.DirectionalLight(0xffffff, 0.2);
  fill.position.set(-1, 1, 1.2);
  scene.add(fill);

  const bodies = sc.bodies.map(() => { const g = new THREE.Group(); scene.add(g); return g; });
  for (const [b, p] of Object.entries(sc.static)) {
    bodies[b].position.set(p[0], p[1], p[2]);
    bodies[b].quaternion.set(p[4], p[5], p[6], p[3]);
  }
  const OVERRIDE = { board: 0xc9ae84, table: 0x8b9299 };
  for (const g of sc.geoms) {
    if (g.type === 0) continue;                         // the floor plane: the scene colour stands in for it
    const [a, b, c] = g.size;
    let geo;
    if (g.type === 6) geo = new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
    else if (g.type === 5) { geo = new THREE.CylinderGeometry(a, a, 2 * b, 28); geo.rotateX(Math.PI / 2); }
    else if (g.type === 3) { geo = new THREE.CapsuleGeometry(a, 2 * b, 5, 12); geo.rotateX(Math.PI / 2); }
    else if (g.type === 2) geo = new THREE.SphereGeometry(a, 16, 12);
    else if (g.type === 4) { geo = new THREE.SphereGeometry(1, 16, 12); geo.scale(a, b, c); }
    else continue;
    const color = new THREE.Color();
    if (OVERRIDE[g.name] !== undefined) color.setHex(OVERRIDE[g.name]); else color.setRGB(g.rgba[0], g.rgba[1], g.rgba[2]);
    const alpha = g.rgba[3];
    const isRoute = g.name.startsWith('route_');
    const mat = new THREE.MeshStandardMaterial({ color, roughness: g.name === 'board' ? 0.85 : 0.55, metalness: 0.05,
                                                 transparent: alpha < 1, opacity: alpha, depthWrite: alpha >= 1 });
    if (isRoute) { mat.color.setHex(0xffffff); mat.opacity = 0.7; }
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(g.pos[0], g.pos[1], g.pos[2]);
    mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
    const flat = g.name === 'board' || g.name === 'table' || isRoute;
    mesh.castShadow = !flat;
    mesh.receiveShadow = g.name === 'board' || g.name === 'table';
    bodies[g.body].add(mesh);
  }
  const moving = sc.moving.map((b) => bodies[b]);

  const mats = {};
  const basic = (key) => (mats[key] = mats[key] || new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95 }));
  const ring = (r, tube, key) => new THREE.Mesh(new THREE.TorusGeometry(r, tube, 10, 48), basic(key));
  function arrow(key) {
    const grp = new THREE.Group();
    const shaftGeo = new THREE.CylinderGeometry(1, 1, 1, 10); shaftGeo.rotateX(Math.PI / 2); shaftGeo.translate(0, 0, 0.5);
    const headGeo = new THREE.ConeGeometry(1, 1, 14); headGeo.rotateX(Math.PI / 2); headGeo.translate(0, 0, 0.5);
    const shaft = new THREE.Mesh(shaftGeo, basic(key)), head = new THREE.Mesh(headGeo, basic(key));
    grp.add(shaft, head);
    grp.userData = { shaft, head };
    grp.visible = false;
    scene.add(grp);
    return grp;
  }
  const actorRing = ring(0.0095, 0.0014, 'actor'); scene.add(actorRing);
  const tool = new THREE.Mesh(new THREE.SphereGeometry(0.0026, 14, 10), basic('force')); scene.add(tool);
  const force = arrow('force');
  const fixtures = sc.forks.map((f) => ({ id: f.id, pos: new THREE.Vector3(f.pos[0], f.pos[1], sc.board_z + 0.0015) }));
  fixtures.push({ id: 'holder', pos: new THREE.Vector3(sc.holder[0], sc.holder[1], sc.board_z + 0.0015) });
  const halos = fixtures.map(() => { const h = ring(0.03, 0.0021, 'planner'); h.visible = false; scene.add(h); return h; });
  const pokes = [arrow('warn'), arrow('warn'), arrow('warn')];
  const tagBox = $('tags');
  const mkTag = (cls, text) => { const el = document.createElement('div'); el.className = 'tag ' + cls; el.textContent = text; tagBox.appendChild(el); return el; };
  const fixTags = fixtures.map((f) => mkTag('fix', f.id === 'holder' ? (sc.connector || 'X1') + ' holder' : f.id));
  const actorTag = mkTag('who', '');           // not "actor": that class is the badge in the now strip
  const planTag = mkTag('who plan', '');
  T3 = { renderer, scene, camera, controls, home: null, moving, bodies, actorRing, tool, force, fixtures, halos, pokes, mats,
         fixTags, actorTag, planTag, follow: false, userMoved: false, lastTcp: null, w: 1, h: 1,
         poseScale: sc.pose_scale, quatScale: sc.quat_scale };
  controls.addEventListener('start', () => { if (T3) T3.userMoved = true; });
  applyColors();
  resize();
  setFollow(false);
}
function applyColors() {
  if (!T3) return;
  T3.scene.background = new THREE.Color(css('--scene') || '#dfe4e8');
  if (T3.mats.force) T3.mats.force.color.set(css('--bad'));
  if (T3.mats.planner) T3.mats.planner.color.set(css('--planner'));
  if (T3.mats.warn) T3.mats.warn.color.set(css('--warn'));
}
function setArrow(grp, origin, dir, len, radius) {
  if (len < 1e-4) { grp.visible = false; return; }
  grp.visible = true;
  grp.position.copy(origin);
  grp.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir.clone().normalize());
  const hl = Math.min(len * 0.45, radius * 5.5);
  grp.userData.shaft.scale.set(radius, radius, Math.max(len - hl, 1e-4));
  grp.userData.head.scale.set(radius * 2.6, radius * 2.6, hl);
  grp.userData.head.position.set(0, 0, len - hl);
}
const _v = window.THREE ? new THREE.Vector3() : null;
function place(el, pos, below) {      // a tag above (or below) a point of the scene
  _v.copy(pos).project(T3.camera);
  if (_v.z > 1 || _v.z < -1) { el.hidden = true; return; }
  const x = (_v.x + 1) / 2 * T3.w, y = (1 - _v.y) / 2 * T3.h;
  el.hidden = x < -40 || y < -20 || x > T3.w + 40 || y > T3.h + 20;
  el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) ` + (below ? 'translate(-50%, 6px)' : 'translate(-50%, -100%)');
}
function setFollow(on) {
  if (!T3) return;
  T3.follow = on;
  $('cam-follow').setAttribute('aria-pressed', on ? 'true' : 'false');
  $('cam-cell').setAttribute('aria-pressed', on ? 'false' : 'true');
  const { camera, controls } = T3;
  if (on && V && V.frames.length) {
    const f = V.frames[frameIndex(V.t)];
    const tcp = new THREE.Vector3(f.tcp[0] / 1e4, f.tcp[1] / 1e4, f.tcp[2] / 1e4);
    const dir = camera.position.clone().sub(controls.target).normalize();
    controls.target.copy(tcp);
    camera.position.copy(tcp.clone().add(dir.multiplyScalar(0.32)));
    T3.lastTcp = tcp.clone();
    controls.update();
  } else {
    T3.userMoved = false;
    applyHome();
  }
}
function resize() {
  if (!T3) return;
  const r = $('stage').getBoundingClientRect();
  if (r.width < 2 || r.height < 2) return;
  T3.w = r.width; T3.h = r.height;
  T3.renderer.setSize(r.width, r.height, false);
  T3.camera.aspect = r.width / r.height;
  T3.camera.updateProjectionMatrix();
  if (!T3.userMoved && !T3.follow) applyHome();
}
new ResizeObserver(resize).observe($('stage'));

const q0 = new Float32Array(4), qa = new Float32Array(4), qb = new Float32Array(4);
function draw3D(i, a) {
  const T = T3, F = V.frames, f0 = F[i], f1 = F[Math.min(i + 1, F.length - 1)];
  const ps = T.poseScale, qs = T.quatScale;
  for (let j = 0; j < T.moving.length; j++) {
    const o = j * 7, p = f0.pose, n = f1.pose;
    T.moving[j].position.set(
      (p[o] + (n[o] - p[o]) * a) / ps, (p[o + 1] + (n[o + 1] - p[o + 1]) * a) / ps, (p[o + 2] + (n[o + 2] - p[o + 2]) * a) / ps);
    qa[0] = p[o + 4] / qs; qa[1] = p[o + 5] / qs; qa[2] = p[o + 6] / qs; qa[3] = p[o + 3] / qs;
    qb[0] = n[o + 4] / qs; qb[1] = n[o + 5] / qs; qb[2] = n[o + 6] / qs; qb[3] = n[o + 3] / qs;
    THREE.Quaternion.slerpFlat(q0, 0, qa, 0, qb, 0, a);
    T.moving[j].quaternion.set(q0[0], q0[1], q0[2], q0[3]).normalize();
  }
  const tcp = new THREE.Vector3(
    (f0.tcp[0] + (f1.tcp[0] - f0.tcp[0]) * a) / 1e4, (f0.tcp[1] + (f1.tcp[1] - f0.tcp[1]) * a) / 1e4,
    (f0.tcp[2] + (f1.tcp[2] - f0.tcp[2]) * a) / 1e4);
  const actor = f0.actor;
  const moving = actor !== 'idle';
  T.actorRing.visible = moving;
  if (moving) {
    T.mats.actor.color.set(css(ACTOR[actor].color));
    T.actorRing.position.copy(tcp).setZ(tcp.z - 0.002);
  }
  T.tool.position.copy(tcp);
  const w = f0.wrench, Fm = Math.hypot(w[0], w[1], w[2]);
  if (Fm > 0.5) setArrow(T.force, tcp, new THREE.Vector3(w[0], w[1], w[2]), Math.min(Fm * 0.0045, 0.14), 0.0011);
  else T.force.visible = false;
  // the fixture the current step is about (none while the planner is choosing the next one)
  const c = V.live && V.thinking && !V.done ? null : V.calls[f0.call];
  const args = (c && c.arguments) || {};
  const pulse = reduceMotion ? 1 : 1 + 0.07 * Math.sin(performance.now() / 260);
  let anchor = null, anchorIdx = -1;
  T.halos.forEach((h, n) => {
    const fx = T.fixtures[n];
    const on = !!c && ((c.name === 'route_fork' && fx.id === args.fork_id) || (c.name === 'insert_connector' && fx.id === 'holder')
                       || c.name === 'inspect');
    h.visible = on;
    h.position.copy(fx.pos);
    h.scale.setScalar(pulse);
    if (on && c.name !== 'inspect') { anchor = fx.pos; anchorIdx = n; }
  });
  (f0.pokes || []).forEach((e, n) => {
    if (T.pokes[n]) setArrow(T.pokes[n], new THREE.Vector3(e[0], e[1], e[2]), new THREE.Vector3(e[3], e[4], e[5]),
                             Math.min(Math.hypot(e[3], e[4], e[5]) * 0.26, 0.07), 0.0016);
  });
  for (let n = (f0.pokes || []).length; n < T.pokes.length; n++) T.pokes[n].visible = false;
  if (T.follow) {
    if (!T.lastTcp) T.lastTcp = tcp.clone();
    const delta = tcp.clone().sub(T.lastTcp).multiplyScalar(0.12);
    T.controls.target.add(delta); T.camera.position.add(delta); T.lastTcp.add(delta);
  }
  T.controls.update();
  T.renderer.render(T.scene, T.camera);
  // labels
  T.fixtures.forEach((fx, n) => {            // the plan's tag names the fixture it is about
    if (n === anchorIdx) T.fixTags[n].hidden = true; else place(T.fixTags[n], fx.pos.clone().setZ(fx.pos.z + 0.045));
  });
  if (moving && actor !== 'warn') {
    const label = ACTOR[actor].short;
    if (T.actorTag.textContent !== label) T.actorTag.textContent = label;
    T.actorTag.style.background = css(ACTOR[actor].color);
    place(T.actorTag, tcp.clone().setZ(tcp.z + 0.03));
  } else T.actorTag.hidden = true;
  if (anchor && c) {           // under the halo, on the side facing the camera: clear of the arm and its tag
    const label = (V.planner === 'nemotron' ? 'Nemotron: ' : 'Plan: ') + callTitle(c);
    if (T.planTag.textContent !== label) T.planTag.textContent = label;
    T.planTag.style.background = css('--planner');
    const toCam = new THREE.Vector3(T.camera.position.x - anchor.x, T.camera.position.y - anchor.y, 0);
    if (toCam.lengthSq() > 1e-8) toCam.normalize();
    place(T.planTag, anchor.clone().addScaledVector(toCam, 0.036), true);
  } else T.planTag.hidden = true;
}

// ------------------------------------------------------------------ time
function frameIndex(t) {
  const F = V.frames;
  if (!F.length) return 0;
  let lo = 0, hi = F.length - 1;
  if (t <= F[0].t) return 0;
  if (t >= F[hi].t) return hi;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (F[mid].t <= t) lo = mid; else hi = mid; }
  return lo;
}
function frameAt(t) {
  const F = V.frames, i = frameIndex(t);
  if (i >= F.length - 1) return [i, 0];
  const dt = F[i + 1].t - F[i].t;
  return [i, dt > 0 ? Math.min(1, Math.max(0, (t - F[i].t) / dt)) : 0];
}
const tStart = () => (V.frames.length ? V.frames[0].t : 0);
const tEnd = () => (V.frames.length ? V.frames[V.frames.length - 1].t : 0);

// ------------------------------------------------------------------ decisions
function isOpen(k) { return V.pinned.has(k) ? V.pinned.get(k) : k === V.cur; }
function decisionHtml(c, k) {
  const o = outcomeOf(c), who = executorOf(c, k), r = c.result || {};
  const bits = [];
  if (who) bits.push(`<span class="pill ${who}">${who === 'assist' ? 'GR00T + seating' : ACTOR[who].short}</span>`);
  if (c.t_end != null && c.t_end - c.t_start >= 0.05) bits.push(`<span>${fmtTime(c.t_end - c.t_start)}</span>`);
  if (r.max_contact_force_N != null && who) bits.push(`<span>max ${(+r.max_contact_force_N).toFixed(1)} N</span>`);
  const why = c.reasoning
    ? `<div class="why"><span class="lbl">${V.planner === 'nemotron' ? 'Nemotron\'s reasoning' : 'Why'}</span>${esc(c.reasoning)}</div>` : '';
  const msgs = (r.messages && r.messages.length) ? `<div class="msgs">${r.messages.map((m) => `<span>${esc(m)}</span>`).join('')}</div>` : '';
  const checks = c.checks.length ? `<div class="msgs">${c.checks.map((v) => {
    const vd = v.verdict || {};
    return `<span class="cam">camera check, ${esc(v.target)}: ${vd.error ? 'no answer' : (vd.seated ? 'seated' : 'not seated')}${vd.confidence != null ? ' (' + Math.round(vd.confidence * 100) + '%)' : ''}</span>`;
  }).join('')}</div>` : '';
  return `<li class="decision${k === V.cur ? ' cur' : ''}" data-k="${k}"><details data-k="${k}"${isOpen(k) ? ' open' : ''}><summary>
    <span class="n">${k + 1}</span><span class="what">${esc(callTitle(c))}</span><span class="pill ${o.cls}">${esc(o.text)}</span>
    ${bits.length ? `<span class="who">${bits.join('')}</span>` : ''}</summary>${why}${msgs}${checks}</details></li>`;
}
function renderDecisions() {
  const box = $('decisions');
  if (!V) { box.innerHTML = EMPTY_DECISIONS; return; }
  const items = V.calls.map((c, k) => (c ? decisionHtml(c, k) : '')).join('');
  const think = V.live && V.thinking && !V.done && !V.failed
    ? `<li class="decision thinking"><details open><summary><span class="n">${V.calls.length + 1}</span>
       <span class="what"><span class="live-dot"></span>Nemotron is reading the result and choosing the next step…</span></summary></details></li>` : '';
  const following = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  box.innerHTML = (items + think) || EMPTY_DECISIONS;
  if (V.live && !V.done && following) box.scrollTop = box.scrollHeight;     // follow the newest unless scrolled up
}
function highlightCall(k) {
  if (k === V.cur) return;
  V.cur = k;
  const box = $('decisions');
  box.querySelectorAll('.decision[data-k]').forEach((el) => {
    const kk = +el.dataset.k;
    el.classList.toggle('cur', kk === k);
    const d = el.querySelector('details');
    if (d && !V.pinned.has(kk)) d.open = kk === k;
  });
  $('scrub-calls').querySelectorAll('span').forEach((el) => el.classList.toggle('cur', +el.dataset.k === k));
  const cur = box.querySelector(`.decision[data-k="${k}"]`);
  if (cur) {
    const br = box.getBoundingClientRect(), cr = cur.getBoundingClientRect();
    if (cr.top < br.top || cr.bottom > br.bottom) box.scrollTop += cr.top - br.top - 6;
  }
}
$('decisions').addEventListener('click', (e) => {
  const s = e.target.closest('summary');
  if (!s || !V) return;
  const d = s.parentElement, k = +d.dataset.k;
  if (!Number.isNaN(k)) V.pinned.set(k, !d.open);      // the click opens or closes it after this handler
});

// ------------------------------------------------------------------ the now strip and the robot card
const shown = {};
function setHTML(id, html) { if (shown[id] !== html) { shown[id] = html; $(id).innerHTML = html; } }
let lastStat = 0, lastStatI = -1;
function renderNow(i) {
  const f = i >= 0 ? V.frames[i] : null;
  const c = f ? V.calls[f.call] : null;
  const atLiveEdge = V.live && !V.done && !V.failed;
  let actor = f ? f.actor : 'idle';
  if (atLiveEdge && V.thinking) actor = 'planner';
  const actorEl = $('actor');
  const cls = 'actor ' + (actor === 'idle' ? '' : actor);
  if (actorEl.className !== cls) actorEl.className = cls;
  setHTML('actor', `<i></i>${esc(ACTOR[actor].text)}`);
  let text;
  if (actor === 'planner') text = V.calls.length ? 'Reading the last result and choosing the next step…' : 'Reading the spec and the cell, planning the first step…';
  else if (actor === 'warn') {
    const d = V.dists.find((x) => Math.abs(x.t - f.t) < 3);
    text = `<b>${esc(d ? d.label : 'Disturbance')}</b>: the planner is not told; it has to notice.`;
  } else if (c) {
    text = `<b>Step ${f.call + 1}: ${esc(callTitle(c))}</b>` + (IDLE.has(f.phase) ? '' : ` · ${esc(PHASE_SHORT[f.phase] || f.phase.replace(/_/g, ' '))}`);
  } else if (f && V.calls.length && f.call >= V.calls.length) {
    text = V.done ? (V.done.success ? 'Finished: every clip and the connector are in place.' : 'Finished, but not everything is in place.') : 'Finishing.';
  } else text = f ? 'Settling and looking at the cell.' : (V.live ? 'Starting the simulator…' : 'No frames.');
  setHTML('nowtext', text);
  setHTML('clock', (atLiveEdge ? '<span class="live-dot"></span>live · ' : '') + `robot time <b>${f ? (f.t - tStart()).toFixed(1) : '0.0'}</b> s`);
  if (f) highlightCall(f.call);
  // the robot card, a few times a second
  const now = performance.now();
  const busy = V.playing || atLiveEdge;
  if (!f || (busy && now - lastStat < 90) || (!busy && i === lastStatI && now - lastStat < 400)) return;
  lastStat = now; lastStatI = i;
  const sc = V.scene || {};
  const pt = (sc.phase_text || {})[f.phase] || (IDLE.has(f.phase) ? 'Holding still.' : (PHASE_SHORT[f.phase] || f.phase));
  $('phase-text').textContent = f.mode === 1 ? 'The robot holds still while the disturbance happens.'
    : (f.actor === 'assist' ? 'Force-controlled seating: ' + pt : pt);
  const w = f.wrench, Fm = Math.hypot(w[0], w[1], w[2]);
  setHTML('st-force', Fm.toFixed(1) + '<small>N</small>');
  setHTML('st-grip', Math.round(f.grip) + '<small>mm</small>');
  const prev = V.frames[Math.max(0, i - 1)];
  const dt = Math.max(1e-3, f.t - prev.t);
  const sp = i > 0 ? Math.hypot(f.tcp[0] - prev.tcp[0], f.tcp[1] - prev.tcp[1], f.tcp[2] - prev.tcp[2]) / 10 / dt : 0;
  setHTML('st-speed', Math.round(sp) + '<small>mm/s</small>');
  drawSpark(i);
  // what the simulator says is done
  const route = sc.route || [];
  setHTML('truth', route.map((fid, n) => `<span class="${f.truth[n] ? 'yes' : ''}">${esc(fid)} ${f.truth[n] ? 'wire in clip' : 'empty'}</span>`).join('') +
    `<span class="${f.truth[route.length] ? 'yes' : ''}">${esc(sc.connector || 'X1')} ${f.truth[route.length] ? 'in holder' : 'not in holder'}</span>`);
  $('spec-route').querySelectorAll('.node').forEach((el) => {
    const n = route.indexOf(el.dataset.id);
    el.classList.toggle('done', n >= 0 ? !!f.truth[n] : (el.dataset.id === sc.connector && !!f.truth[route.length]));
  });
  // the skill's log up to now
  let hi = V.logs.length;
  while (hi > 0 && V.logs[hi - 1][0] > f.t + 1e-6) hi--;
  const lines = V.logs.slice(Math.max(0, hi - 8), hi);
  const html = lines.length ? lines.map(([t, m]) => `<div><span class="t">${(t - tStart()).toFixed(1)} s</span>${esc(m)}</div>`).join('') : '–';
  if (shown.log !== html) { setHTML('log', html); $('log').scrollTop = 1e6; }
}

function drawSpark(iNow) {
  const spark = $('spark');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = spark.clientWidth, h = spark.clientHeight;
  if (!w || !h) return;
  if (spark.width !== Math.round(w * dpr) || spark.height !== Math.round(h * dpr)) { spark.width = Math.round(w * dpr); spark.height = Math.round(h * dpr); }
  const g = spark.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const padT = 4, padB = 3;
  g.strokeStyle = css('--line'); g.lineWidth = 1;
  for (const fr of [0, 0.5, 1]) { const y = Math.round(padT + (h - padT - padB) * fr) + 0.5; g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
  if (!V || iNow < 0 || !V.frames[iNow]) return;
  const F = V.frames, tNow = F[iNow].t, span = 12;
  const X = (t) => w * (t - (tNow - span)) / span;
  const Y = (v) => padT + (h - padT - padB) * (1 - Math.min(v / 30, 1));
  let i0 = iNow;
  while (i0 > 0 && F[i0 - 1].t >= tNow - span) i0--;
  const col = css('--bad');
  g.beginPath(); g.moveTo(X(F[i0].t), Y(0));
  for (let i = i0; i <= iNow; i++) { const f = F[i].wrench; g.lineTo(X(F[i].t), Y(Math.hypot(f[0], f[1], f[2]))); }
  g.lineTo(X(tNow), Y(0)); g.closePath();
  g.globalAlpha = 0.16; g.fillStyle = col; g.fill(); g.globalAlpha = 1;
  g.beginPath();
  for (let i = i0; i <= iNow; i++) {
    const f = F[i].wrench, y = Y(Math.hypot(f[0], f[1], f[2]));
    if (i === i0) g.moveTo(X(F[i].t), y); else g.lineTo(X(F[i].t), y);
  }
  g.strokeStyle = col; g.lineWidth = 1.6; g.stroke();
}

function renderBanner(i) {
  const b = $('banner');
  const f = V.frames[i];
  let html = '', cls = '';
  if (f) {
    const d = V.dists.find((x) => f.t >= x.t - 2.5 && f.t <= x.t + 3.0);
    if (d) { cls = 'warn'; html = `Disturbance: ${esc(d.label)} <small>the planner is not told</small>`; }
    else if (V.done && f.t >= tEnd() - 0.05) {
      const ok = !!V.done.success;
      cls = ok ? 'ok' : 'bad';
      html = ok ? 'Build complete <small>the simulator confirms every clip and the connector</small>'
                : 'Build incomplete <small>see the result below</small>';
    }
  }
  if (b.hidden !== !html) b.hidden = !html;
  if (html && shown.banner !== html) { shown.banner = html; b.innerHTML = html; b.className = 'banner ' + cls; }
}

// ------------------------------------------------------------------ result of the build
function renderResult() {
  const card = $('result-card');
  if (!V || (!V.done && !V.failed && !V.infeasible)) { card.hidden = true; return; }
  card.hidden = false;
  if (V.failed) {
    card.className = 'result bad';
    card.innerHTML = `<div class="result-head"><span class="result-big bad">The build stopped</span><span class="sub">something went wrong in the server</span></div>
      <details><summary>The error</summary><div class="error-box">${esc(V.failed.error || '')}</div></details>`;
    return;
  }
  if (V.infeasible && !V.done) {
    card.className = 'result bad';
    card.innerHTML = '<div class="result-head"><span class="result-big bad">The spec cannot be built</span></div>' + issuesHtml(V.infeasible);
    return;
  }
  const d = V.done, ok = !!d.success, truth = d.truth || {};
  card.className = 'result' + (ok ? '' : ' bad');
  if (V.infeasible && V.infeasible.some((i) => i.severity === 'error')) {      // refused before anything moved
    const verdict = d.claimed_success == null ? '' : `<li>${d.planner === 'nemotron' ? 'Nemotron' : 'the planner'} reported <b>${d.claimed_success ? 'success' : 'not done'}</b>${d.claimed_success ? ', wrongly' : ', correctly'}</li>`;
    card.innerHTML = `<div class="result-head"><span class="result-big bad">✗ Not built</span><span class="sub">the spec fails its checks, so the cell refused every step</span></div>
      <ul class="result-facts"><li><b>${d.tool_calls}</b> decision${d.tool_calls === 1 ? '' : 's'}</li>${verdict}</ul>
      <details><summary>Why it cannot be built</summary>${issuesHtml(V.infeasible)}${d.report ? `<div class="report">${esc(d.report)}</div>` : ''}</details>`;
    return;
  }
  const forks = truth.forks_routed || {};
  const nForks = Object.keys(forks).length, inForks = Object.values(forks).filter(Boolean).length;
  const sub = ok ? 'the simulator confirms every clip and the connector'
    : `the simulator: ${inForks} of ${nForks} clips hold the wire, the connector is ${truth.connector_seated ? 'in its holder' : 'not in its holder'}`;
  const facts = [`<li><b>${d.tool_calls}</b> decision${d.tool_calls === 1 ? '' : 's'}</li>`, `<li><b>${fmtTime(d.robot_time_s)}</b> of robot time</li>`,
                 `<li><b>${fmtTime(d.wall_time_s)}</b> wall time</li>`];
  if (truth.max_contact_force != null) facts.push(`<li>highest contact force <b>${(+truth.max_contact_force).toFixed(1)} N</b></li>`);
  const rt = d.routing;
  if (rt && rt.groot_routes) {
    const assisted = V.calls.filter((c, k) => c && c.name === 'route_fork' && executorOf(c, k) === 'assist' && c.result && c.result.ok).length;
    let s = assisted ? `GR00T + force-controlled seating routed <b>${rt.groot_ok} of ${rt.groot_routes}</b> forks (seating finished ${assisted})`
                     : `GR00T routed <b>${rt.groot_ok} of ${rt.groot_routes}</b> forks`;
    if (rt.expert_routes) s += `; the expert's retry <b>${rt.expert_ok} of ${rt.expert_routes}</b>`;
    facts.push(`<li>${s}</li>`);
  }
  if (d.claimed_success != null) {
    facts.push(`<li>${d.planner === 'nemotron' ? 'Nemotron' : 'the planner'} reported <b>${d.claimed_success ? 'success' : 'not done'}</b>` +
               `${d.claimed_success === ok ? ', the simulator agrees' : ', the simulator disagrees'}</li>`);
  }
  if (d.tokens) facts.push(`<li><b>${d.tokens.toLocaleString()}</b> tokens</li>`);
  if (d.scenario && d.scenario !== 'nominal') facts.push(`<li>disturbance: <b>${esc(SCENARIO_TEXT[d.scenario] || d.scenario)}</b></li>`);
  const files = (d.files || []).filter((f) => /\.(md|json)$/.test(f) && f !== 'events.json' && f !== 'build.json');
  const more = (d.report || files.length || V.infeasible) ? `<details><summary>${d.report ? 'The planner\'s report' : 'Files'}</summary>
      ${V.infeasible ? issuesHtml(V.infeasible) : ''}${d.report ? `<div class="report">${esc(d.report)}</div>` : ''}
      ${files.length ? `<div class="files">${files.map((f) => `<a href="/api/builds/${encodeURIComponent(V.id)}/files/${encodeURIComponent(f)}" target="_blank" rel="noopener">${esc(f)}</a>`).join('')}</div>` : ''}</details>` : '';
  card.innerHTML = `<div class="result-head"><span class="result-big ${ok ? 'ok' : 'bad'}">${ok ? '✓ Build complete' : '✗ Build incomplete'}</span>
    <span class="sub">${esc(sub)}</span></div><ul class="result-facts">${facts.join('')}</ul>${more}`;
}
function issuesHtml(issues) {
  if (!issues || !issues.length) return '<div class="ok">✓ Every check passed: it can be built.</div>';
  return issues.map((i) => `<div class="issue ${esc(i.severity)}"><b>${esc(i.severity)}</b><span>${esc(i.message)}</span></div>`).join('');
}

// ------------------------------------------------------------------ transport
function buildSpans() {
  const F = V.frames;
  if (!F.length) return;
  const t0 = tStart(), span = Math.max(tEnd() - t0, 1e-6);
  const pct = (t) => (100 * (t - t0) / span).toFixed(3);
  const out = [];
  let cur = null;
  for (const f of F) {
    const a = f.actor === 'idle' ? null : f.actor;
    if (!cur || cur.a !== a) { if (cur && cur.a) out.push(cur); cur = { a, s: f.t, e: f.t }; } else cur.e = f.t;
  }
  if (cur && cur.a) out.push(cur);
  $('scrub-track').innerHTML =
    out.map((s) => `<span class="${s.a}" style="left:${pct(s.s)}%;width:${Math.max(0.15, 100 * (s.e - s.s + 0.05) / span).toFixed(3)}%"></span>`).join('') +
    V.dists.map((d) => `<span class="warn" style="left:${pct(d.t)}%;width:0.35%" title="${esc(d.label)}"></span>`).join('');
  const segs = [];
  V.calls.forEach((c, k) => {
    const lab = c ? segLabel(c) : null;
    if (lab && c.t_end != null && c.t_end > c.t_start) segs.push({ k, lab, s: c.t_start, e: c.t_end, bad: outcomeOf(c).cls === 'bad' });
  });
  $('scrub-calls').innerHTML = segs.map((g) =>
    `<span data-k="${g.k}" class="${g.bad ? 'bad' : ''}${g.k === V.cur ? ' cur' : ''}" style="left:${pct(g.s)}%;width:${(100 * (g.e - g.s) / span).toFixed(3)}%" title="${esc(callTitle(V.calls[g.k]))}">${esc(g.lab)}</span>`).join('');
}
function setPlaying(p) {
  if (!V) return;
  if (p && V.t >= tEnd() - 1e-6) V.t = tStart();
  V.playing = p;
  $('play').textContent = p ? 'Pause' : 'Play';
}
function showTransport(on) {
  $('transport').hidden = !on;
  if (on) buildSpans();
}
function seek(t) {
  if (!V || !V.frames.length || V.live) return;
  V.t = Math.min(tEnd(), Math.max(tStart(), t));
  setPlaying(false);
}
$('play').addEventListener('click', () => setPlaying(!V || !V.playing));
$('speeds').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
  if (V) V.speed = +b.dataset.v;
  $('speeds').querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
}));
$('speeds').querySelector('button').setAttribute('aria-pressed', 'true');
$('scrub-range').addEventListener('input', (e) => {
  if (!V || !V.frames.length) return;
  seek(tStart() + (tEnd() - tStart()) * (+e.target.value / 1000));
});
$('cam-cell').addEventListener('click', () => setFollow(false));
$('cam-follow').addEventListener('click', () => setFollow(true));
document.addEventListener('keydown', (e) => {
  if (e.altKey || e.ctrlKey || e.metaKey || !V || V.live || !V.frames.length || $('view-build').hidden) return;
  if (e.target.closest && e.target.closest('button, input, select, textarea, summary, a')) return;
  if (e.key === ' ') { e.preventDefault(); setPlaying(!V.playing); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); seek(V.t + (e.shiftKey ? 5 : 1)); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); seek(V.t - (e.shiftKey ? 5 : 1)); }
  else if (e.key === 'Home') { e.preventDefault(); seek(tStart()); }
  else if (e.key === 'End') { e.preventDefault(); seek(tEnd()); }
});

// ------------------------------------------------------------------ main loop
let lastWall = null;
function loop(now) {
  if (V && V.frames.length) {
    if (V.playing && !V.live && lastWall !== null) {
      const dt = Math.min((now - lastWall) / 1000, 0.1);
      V.t = Math.min(tEnd(), V.t + dt * V.speed);
      if (V.t >= tEnd() - 1e-6) setPlaying(false);
    }
    if (V.live && !V.done) V.t = tEnd();
    if (V.dirty && V.live) { V.dirty = false; renderDecisions(); }
    const [i, a] = frameAt(V.t);
    if (T3) draw3D(i, a);
    renderNow(i);
    renderBanner(i);
    if (!$('transport').hidden) {
      const frac = (V.t - tStart()) / Math.max(tEnd() - tStart(), 1e-6);
      $('scrub-head').style.left = (100 * frac).toFixed(3) + '%';
      if (document.activeElement !== $('scrub-range')) $('scrub-range').value = Math.round(1000 * frac);
    }
  } else if (V && V.live) {
    renderNow(-1);
  }
  lastWall = now;
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// ------------------------------------------------------------------ opening builds
function resetView() {
  if (V && V.ws) { try { V.ws.close(); } catch (e) { /* gone */ } }
  disposeScene();
  for (const k of Object.keys(shown)) delete shown[k];
  lastStatI = -1;
  $('empty').hidden = false;
  $('banner').hidden = true;
  $('result-card').hidden = true;
  $('transport').hidden = true;
  $('decisions').innerHTML = EMPTY_DECISIONS;
  $('planner-name').textContent = '';
  $('log').textContent = '–';
  $('phase-text').textContent = '–';
  $('truth').innerHTML = '';
  $('actor').className = 'actor';
  $('actor').innerHTML = '<i></i>Idle';
  $('nowtext').textContent = 'No build open.';
  $('clock').textContent = '';
  ['st-force', 'st-grip', 'st-speed'].forEach((id, n) => { $(id).innerHTML = ['0.0<small>N</small>', '0<small>mm</small>', '0<small>mm/s</small>'][n]; });
  $('spec-route').querySelectorAll('.node').forEach((el) => el.classList.remove('done'));
  drawSpark(-1);
}
function openLive(info) {
  resetView();
  V = newView(info, true);
  $('empty').innerHTML = '<div><h3>Starting the build…</h3><p>The simulator is loading the harness and settling the wire.</p></div>';
  markCurrent();
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/builds/' + encodeURIComponent(info.id));
  V.ws = ws;
  const mine = V;
  ws.onmessage = (e) => { if (V === mine) handleEvent(JSON.parse(e.data)); };
  ws.onclose = () => { if (V === mine && !V.done && !V.failed) { refreshStatus(); loadPast(); } };
}
function finishLive() {
  if (!V) return;
  V.live = false;
  V.t = tEnd();
  setPlaying(false);
  if (V.frames.length) showTransport(true); else noFramesNote();
  refreshStatus();
}
function noFramesNote() {
  $('empty').hidden = false;
  $('empty').innerHTML = V.infeasible ? '<div><h3>Nothing was built</h3><p>The spec fails its checks, so the cell was not set up; see why below.</p></div>'
    : '<div><h3>Nothing to show in 3D</h3><p>This build stopped before the robot moved.</p></div>';
}
async function openPast(info) {
  if (info.status === 'running' || info.status === 'queued') { openLive(info); return; }
  resetView();
  V = newView(info, false);
  const mine = V;
  markCurrent();
  $('empty').innerHTML = '<div><h3>Loading the build…</h3></div>';
  try {
    const events = await api('/api/builds/' + encodeURIComponent(info.id) + '/events');
    for (const ev of events) { if (V !== mine) return; handleEvent(ev); }
    if (V !== mine) return;
    renderDecisions();
    renderResult();
    if (!V.frames.length) { noFramesNote(); return; }
    showTransport(true);
    V.t = tStart();
    setPlaying(true);
  } catch (err) {
    if (V !== mine) return;
    $('empty').hidden = false;
    $('empty').innerHTML = `<div><h3>Could not open this build</h3><p>${esc(err.message)}</p></div>`;
  }
}

// ------------------------------------------------------------------ past builds
let pastList = [];
async function loadPast() {
  try { pastList = await api('/api/builds'); } catch (e) { pastList = []; }
  const box = $('past');
  if (!pastList.length) { box.innerHTML = '<li class="empty-note">No builds yet.</li>'; return; }
  box.innerHTML = pastList.slice(0, 40).map((b) => {
    const o = b.options || {}, s = b.summary || {};
    const res = b.status === 'running' || b.status === 'queued' ? ['run', 'running']
      : b.status === 'failed' ? ['bad', 'stopped'] : (s.success ? ['ok', 'complete'] : ['bad', 'incomplete']);
    const when = new Date((b.created || 0) * 1000);
    const bits = [when.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
                  o.planner === 'nemotron' ? 'Nemotron' : 'scripted', o.groot != null ? 'GR00T' : 'expert',
                  o.scenario && o.scenario !== 'nominal' ? (SCENARIO_TEXT[o.scenario] || o.scenario) : '', 'board ' + o.seed].filter(Boolean);
    return `<li><button type="button" data-id="${esc(b.id)}" title="${esc(b.id)}"><span class="p-title">${esc(s.spec || specName(o.spec))}</span><span class="p-res ${res[0]}">${res[1]}</span>
      <span class="p-sub">${esc(bits.join(' · '))}</span></button></li>`;
  }).join('');
  box.querySelectorAll('button').forEach((btn) => btn.addEventListener('click', () => {
    const info = pastList.find((b) => b.id === btn.dataset.id);
    if (info) openPast(info);
  }));
  markCurrent();
}
function markCurrent() {
  $('past').querySelectorAll('button').forEach((b) => b.setAttribute('aria-current', V && b.dataset.id === V.id ? 'true' : 'false'));
}

// ------------------------------------------------------------------ specs and options
function segSetup(id, key, onChange) {
  const seg = $(id);
  seg.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    if (b.disabled) return;
    S.opts[key] = b.dataset.v;
    paintSeg(id, key);
    if (onChange) onChange();
  }));
  paintSeg(id, key);
}
function paintSeg(id, key) {
  $(id).querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.v === S.opts[key] ? 'true' : 'false'));
}
async function loadSpecs(select) {
  S.specs = await api('/api/specs');
  const sel = $('spec-select');
  sel.innerHTML = S.specs.map((s) => `<option value="${esc(s.file)}">${esc(s.name)}${s.route && s.route.length ? ' · ' + s.route.length + ' forks' : ''}${s.feasible ? '' : ' (cannot be built)'}${s.uploaded ? ' · ' + esc(s.file) : ''}</option>`).join('');
  const pick = select || (S.specs.find((s) => s.file === 'demo_3fork.yaml') || S.specs[0] || {}).file;
  if (pick) { sel.value = pick; selectSpec(pick); }
}
function selectSpec(file) {
  const s = S.specs.find((x) => x.file === file);
  S.spec = s || null;
  if (!s) return;
  const url = '/api/specs/' + encodeURIComponent(file) + '/drawing.png';
  const img = $('spec-drawing');
  img.classList.remove('ok');
  img.onload = () => img.classList.add('ok');
  img.src = url;
  img.alt = 'Assembly drawing of ' + s.name;
  $('drawing-link').href = url;
  const nodes = (s.route || []).map((f) => `<span class="node" data-id="${esc(f)}">${esc(f)}</span>`);
  if (s.connector) nodes.push(`<span class="node" data-id="${esc(s.connector)}">${esc(s.connector)}</span>`);
  $('spec-route').innerHTML = nodes.join('<span class="arrow">→</span>');
  $('spec-checks').innerHTML = s.error ? `<div class="issue error"><b>error</b><span>${esc(s.error)}</span></div>` : issuesHtml(s.issues);
  updateHints();
}
function updateHints() {
  const st = S.status, gs = st.groot_setup || {};
  const gb = $('opt-groot').querySelector('[data-v="1"]');
  const grootOk = st.groot != null && st.groot_reachable !== false;
  gb.disabled = !grootOk;
  if (!grootOk && S.opts.groot === '1') { S.opts.groot = '0'; paintSeg('opt-groot', 'groot'); }
  let gh;
  if (st.groot == null) gh = 'GR00T is off: this server was started without <code>--groot</code>.';
  else if (st.groot_reachable === false) gh = 'The GR00T policy server does not answer.';
  else if (S.opts.groot === '1') {
    gh = 'GR00T N1.7 routes each fork' + (gs.seat_assist != null ? `; force-controlled seating finishes a snap-in it is stuck on for ${gs.seat_assist} s` : '') +
         '; the planner can retry a failed fork with the expert.';
    if (S.spec && gs.trained_on && S.spec.file !== gs.trained_on) gh += ' GR00T was trained on the Door module (3 forks): on this harness, expect the expert to do more.';
  } else gh = 'The force-guided expert routes every fork.';
  $('groot-hint').innerHTML = gh;
  const nemo = $('opt-planner').querySelector('[data-v="nemotron"]');
  nemo.disabled = !st.nebius_key;
  $('opt-vision').querySelector('[data-v="1"]').disabled = !st.nebius_key;
  if (!st.nebius_key) {
    if (S.opts.planner === 'nemotron') { S.opts.planner = 'scripted'; paintSeg('opt-planner', 'planner'); }
    if (S.opts.vision === '1') { S.opts.vision = '0'; paintSeg('opt-vision', 'vision'); }
  }
  const start = $('start');
  const busy = !!st.busy;
  start.disabled = busy || !S.spec;
  let sh = '';
  if (busy) sh = 'A build is running; one at a time.';
  else if (S.spec && !S.spec.feasible) sh = 'This spec fails its checks: the planner will be told it cannot be built.';
  else if (!st.nebius_key) sh = 'No Nebius key on this server, so Nemotron and the vision check are off (set <code>NEBIUS_API_KEY</code>).';
  $('start-hint').innerHTML = sh;
}
async function refreshStatus() {
  try { S.status = await api('/api/status'); } catch (e) { S.status = { offline: true }; }
  const st = S.status;
  const chip = (state, text) => `<span class="chip ${state}"><i></i>${esc(text)}</span>`;
  $('status').innerHTML = st.offline ? chip('off', 'Server not answering') :
    chip(st.nebius_key ? 'on' : 'off', st.nebius_key ? 'Nemotron ready' : 'No Nebius key') +
    chip(st.groot == null ? '' : (st.groot_reachable !== false ? 'on' : 'off'), st.groot == null ? 'GR00T off' : (st.groot_reachable === false ? 'GR00T server down' : 'GR00T N1.7 ready')) +
    (st.busy ? chip('busy', 'build running') : '');
  updateHints();
}

$('spec-select').addEventListener('change', (e) => selectSpec(e.target.value));
$('spec-file').addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const info = await api('/api/specs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ file: file.name, yaml: text }) });
    await loadSpecs(info.file);
  } catch (err) {
    $('spec-checks').innerHTML = `<div class="issue error"><b>error</b><span>${esc(err.message)}</span></div>`;
  }
  e.target.value = '';
});
$('seed-dice').addEventListener('click', () => { $('opt-seed').value = Math.floor(100 + Math.random() * 9900); });
$('start').addEventListener('click', async () => {
  if (!S.spec) return;
  $('start').disabled = true;
  try {
    const body = {
      spec: S.spec.file, planner: S.opts.planner, use_groot: S.opts.groot === '1', scenario: $('opt-scenario').value,
      vision: S.opts.vision === '1', seed: Math.max(0, parseInt($('opt-seed').value || '0', 10) || 0),
      randomize: $('opt-random').checked, pace: +S.opts.pace
    };
    const info = await api('/api/builds', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    S.status.busy = true;
    updateHints();
    openLive(info);
    loadPast();
    refreshStatus();
  } catch (err) {
    $('start-hint').textContent = 'Could not start: ' + err.message;
    $('start').disabled = false;
  }
});

// ------------------------------------------------------------------ tabs and theme
function showTab(name) {
  const build = name === 'build';
  document.body.dataset.view = build ? 'build' : 'results';
  $('tab-build').setAttribute('aria-selected', build ? 'true' : 'false');
  $('tab-results').setAttribute('aria-selected', build ? 'false' : 'true');
  $('view-build').hidden = !build;
  $('view-results').hidden = build;
  if (!build) loadResults();
  try { history.replaceState(null, '', build ? '#build' : '#results'); } catch (e) { /* file:// */ }
  if (build) setTimeout(resize, 0);
}
$('tab-build').addEventListener('click', () => showTab('build'));
$('tab-results').addEventListener('click', () => showTab('results'));
function savedTheme() { try { return localStorage.getItem('mc-theme'); } catch (e) { return null; } }
function setTheme(th) {
  if (th) document.documentElement.setAttribute('data-theme', th); else document.documentElement.removeAttribute('data-theme');
  try { if (th) localStorage.setItem('mc-theme', th); else localStorage.removeItem('mc-theme'); } catch (e) { /* private mode */ }
  themeChanged();
}
function themeChanged() {
  applyColors();
  for (const k of ['truth', 'banner']) delete shown[k];
  lastStatI = -1;
  if (!V || !V.frames.length) drawSpark(-1);
  if (!$('view-results').hidden) loadResults(true);
}
$('theme').addEventListener('click', () => {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark' ||
    (!document.documentElement.getAttribute('data-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
  setTheme(dark ? 'light' : 'dark');
});
if (savedTheme()) setTheme(savedTheme());
if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', themeChanged);

// ------------------------------------------------------------------ results view
let results = null;
async function loadResults(redraw) {
  if (!results) { try { results = await api('/api/results'); } catch (e) { results = {}; } }
  if (results && (redraw || !$('results-body').dataset.done)) renderResults(results);
}
function renderResults(R) {
  const body = $('results-body');
  if (!R || !R.headline) { body.innerHTML = '<p class="empty-note">No results file on this server.</p>'; return; }
  body.dataset.done = '1';
  const pct = (k, n) => Math.round(100 * k / n);
  const tiles = R.headline.map((h) => `<article class="tile ${esc(h.key)}"><div class="big">${pct(h.k, h.n)}%</div>
    <div class="lbl">${esc(h.label)}</div><div class="frac">${h.k} of ${h.n}${h.key !== 'assist' ? ' wires routed' : ''}</div><p>${esc(h.detail)}</p></article>`).join('');
  body.innerHTML = `
    <header><div class="eyebrow">Results · updated ${esc(R.updated)}</div>
      <h2>Nemotron plans, GR00T N1.7 routes the wire, force control seats it</h2>
      <p>${esc(R.setup)}</p></header>
    <section class="tiles">${tiles}</section>
    <section class="stack3">
      <article class="card lvl planner"><div class="eyebrow">Level 3 · one decision per step</div><h3>NVIDIA Nemotron plans</h3>
        <p>Reads the harness spec and the cell's state, picks the next step (route F2, insert the connector, look again), checks every result and retries what failed.</p></article>
      <article class="card lvl groot"><div class="eyebrow">Level 2 · 20 motions a second</div><h3>GR00T N1.7 routes the wire</h3>
        <p>A vision-language-action model: from two camera views, the robot's state and "route the wire into fork F2" it moves the arm, grasps the wire and carries it over the clip. A force-controlled routine finishes a snap-in it gets stuck on.</p></article>
      <article class="card lvl ctrl"><div class="eyebrow">Level 1 · 500 times a second</div><h3>Admittance control keeps it soft</h3>
        <p>Every motion goes through a force-limited controller with a wrist force sensor, so the wire and the clips are never forced.</p></article>
    </section>
    <section class="two">
      <article class="card chart"><div class="card-head"><h2>Training rounds</h2><span class="sub">wires routed of ${R.rounds[0].n}, with 95% intervals</span></div>${roundsChart(R.rounds)}</article>
      <article class="card chart"><div class="card-head"><h2>Where the wires got to</h2><span class="sub">${esc(R.funnel.label)}</span></div>${funnelChart(R.funnel)}</article>
    </section>
    <section class="card"><h2>How GR00T was trained</h2><dl class="facts">${R.training.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl></section>
    <p class="next"><b>Next.</b> ${esc(R.next)}</p>`;
}
function roundsChart(rounds) {
  const W = 640, H = 300, L = 40, Rm = 10, Tm = 22, B = 56;
  const bw = (W - L - Rm) / rounds.length;
  const y = (v) => Tm + (H - Tm - B) * (1 - v);
  const col = (n) => (n === rounds.length - 1 ? css('--ok') : css('--groot'));
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Wires routed per training round">`;
  for (const g of [0, 0.25, 0.5, 0.75, 1]) s += `<line class="axis" x1="${L}" x2="${W - Rm}" y1="${y(g)}" y2="${y(g)}"/><text x="${L - 6}" y="${y(g) + 4}" text-anchor="end">${g * 100}%</text>`;
  rounds.forEach((r, n) => {
    const p = r.k / r.n, [lo, hi] = wilson(r.k, r.n), x = L + n * bw + bw * 0.2, w = bw * 0.6, cx = x + w / 2;
    s += `<rect x="${x}" y="${y(p)}" width="${w}" height="${y(0) - y(p)}" rx="4" fill="${col(n)}" opacity="0.88"><title>${r.k} of ${r.n}</title></rect>`;
    s += `<line x1="${cx}" x2="${cx}" y1="${y(hi)}" y2="${y(lo)}" stroke="${css('--ink')}" stroke-width="1.4"/>`;
    s += `<line x1="${cx - 7}" x2="${cx + 7}" y1="${y(hi)}" y2="${y(hi)}" stroke="${css('--ink')}" stroke-width="1.4"/>`;
    s += `<line x1="${cx - 7}" x2="${cx + 7}" y1="${y(lo)}" y2="${y(lo)}" stroke="${css('--ink')}" stroke-width="1.4"/>`;
    s += `<text class="val" x="${cx}" y="${y(hi) - 7}" text-anchor="middle">${Math.round(100 * p)}%</text>`;
    s += `<text x="${cx}" y="${H - B + 18}" text-anchor="middle">${esc(r.label)}</text>`;
    s += `<text class="sub" x="${cx}" y="${H - B + 34}" text-anchor="middle">${esc(r.sub)}</text>`;
  });
  return s + '</svg>';
}
function funnelChart(fu) {
  const W = 520, rowH = 40, H = fu.steps.length * rowH + 8, L = 150, Rm = 48, bar = W - L - Rm;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(fu.label)}">`;
  fu.steps.forEach(([label, k], n) => {
    const y = 4 + n * rowH, w = bar * k / fu.n;
    s += `<text x="${L - 10}" y="${y + 21}" text-anchor="end">${esc(label)}</text>`;
    s += `<rect x="${L}" y="${y + 6}" width="${bar}" height="22" rx="4" fill="${css('--panel-2')}"/>`;
    s += `<rect x="${L}" y="${y + 6}" width="${w}" height="22" rx="4" fill="${css('--groot')}" opacity="0.85"/>`;
    s += `<text class="val" x="${W - Rm + 8}" y="${y + 22}">${k}</text>`;
  });
  return s + '</svg>';
}

// ------------------------------------------------------------------ start
segSetup('opt-planner', 'planner', updateHints);
segSetup('opt-groot', 'groot', updateHints);
segSetup('opt-vision', 'vision');
segSetup('opt-pace', 'pace');
drawSpark(-1);
(async function boot() {
  await refreshStatus();
  try { await loadSpecs(); } catch (e) { $('spec-checks').innerHTML = `<div class="issue error"><b>error</b><span>${esc(e.message)}</span></div>`; }
  await loadPast();
  if (location.hash === '#results') showTab('results');
  const running = pastList.find((b) => b.status === 'running' || b.status === 'queued');
  if (running) openPast(running);
  setInterval(refreshStatus, 15000);
})();
})();
