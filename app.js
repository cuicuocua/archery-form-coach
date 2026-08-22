import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ===== CALIBRATE WITH COACH — placeholder target ranges, edit freely =====
const BOW_ARM_ANGLE_MIN = 165; // degrees at the bow elbow; 180 = perfectly straight
const BOW_ARM_ANGLE_MAX = 180;

const DRAW_ELBOW_HEIGHT_MIN = -5; // degrees; 0 = level with shoulder line, + = above
const DRAW_ELBOW_HEIGHT_MAX = 15;

const MIN_VISIBILITY = 0.6; // MediaPipe's 0–1 confidence per joint; below this we show "uncertain"

// These five are tuned for COMPOUND shooting (mechanical release aid, let-off held at full
// draw) — if the owner starts shooting recurve with this, expect to revisit all five, since a
// recurve archer anchors differently (fingers under the chin, not a release hand near the jaw)
// and can't hold nearly as steady (no let-off, fighting full poundage the whole time).
const FULL_DRAW_ANCHOR_MAX = 0.45; // draw-hand wrist must be this close to the mouth/nose, as a fraction of torso length, to count as "at anchor" — looser than you might expect because a release aid sits the hand further back near the jaw than fingers under the chin would; hand separation (below) does the real discriminating, this just filters out the grossly wrong
const FULL_DRAW_BOW_ARM_MIN = 150; // degrees; bow arm must be at least this straight to count as "drawn" (looser than the good-form target above — a freeze should still fire on so-so form)
const FULL_DRAW_HAND_SEP_MIN = 0.75; // the two wrists must be at least this far apart, as a fraction of torso length, to count as "drawn" — during the raise both hands are close together near the head, only at full draw are they a draw-length apart. THE key signal: a compound's draw length is fixed by a mechanical stop, so this is near-binary (mid-raise vs. hard against the wall) and can be set with confidence
const FULL_DRAW_STILL_MAX = 0.35; // the draw wrist may drift at most this much (as a fraction of torso length) per second and still count as "holding still" — kept tight (not loosened) because a compound archer at let-off is genuinely steady, unlike the fast continuous motion of the raise
const FULL_DRAW_HOLD_MS = 900; // must stay at full draw this long before we freeze. Compound let-off means the archer can comfortably hold for several seconds, so there's room to be generous here without risking a missed shot — 900ms firmly rules out passing through on the way up
const AUTO_FREEZE_HOLD_MS = 4000; // how long an automatic freeze holds the frame before releasing itself
// ===========================================================================

const DEBUG = location.search.includes("debug"); // ?debug in the URL shows the live trigger-condition overlay

// MediaPipe pose landmark indices (33-point model)
const L_SHOULDER = 11, R_SHOULDER = 12;
const L_ELBOW = 13, R_ELBOW = 14;
const L_WRIST = 15, R_WRIST = 16;
const L_HIP = 23, R_HIP = 24;
const NOSE = 0, MOUTH_L = 9, MOUTH_R = 10;

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const btnCamera = document.getElementById("btn-camera");
const btnHand = document.getElementById("btn-hand");
const btnFreeze = document.getElementById("btn-freeze");
const readoutBowArm = document.getElementById("readout-bowarm");
const valueBowArm = document.getElementById("value-bowarm");
const readoutElbow = document.getElementById("readout-elbow");
const valueElbow = document.getElementById("value-elbow");
const debugEl = document.getElementById("debug");
if (DEBUG) debugEl.classList.remove("hidden");

let poseLandmarker = null;
let stream = null;
let facingMode = "environment"; // rear camera first
let rightHanded = true;
let drawingUtils = null;

// Auto-freeze state machine. kind is one of:
//   'armed'    — watching for full draw
//   'holding'  — full draw seen, waiting out FULL_DRAW_HOLD_MS before triggering
//   'frozen'   — auto-froze the frame, waiting out AUTO_FREEZE_HOLD_MS before releasing
//   'cooldown' — just released; waits for the archer to leave full draw before re-arming
//   'manual'   — the owner tapped Freeze; only the button can end this, never the logic below
let freezeState = { kind: "armed" };

// Previous frame's draw-wrist position + timestamp, for the stillness check in isAtFullDraw
// below. Deliberately just one remembered frame, not a history buffer — cheap and enough.
let lastDrawWrist = null;

// Last frame's full-draw condition values, for the ?debug overlay only (see syncDebugOverlay).
// Stays null whenever isAtFullDraw bails out before it can compute them.
let debugInfo = null;

function isFrozen(state) {
  return state.kind === "frozen" || state.kind === "manual";
}

async function initPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

async function startCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  const mirrored = facingMode === "user";
  video.classList.toggle("mirrored", mirrored);
  canvas.classList.toggle("mirrored", mirrored);

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function angleAt(a, b, c) {
  // angle at point b, between rays b->a and b->c, in degrees
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.hypot(v1.x, v1.y);
  const mag2 = Math.hypot(v2.x, v2.y);
  if (mag1 === 0 || mag2 === 0) return null;
  const cos = Math.min(1, Math.max(-1, dot / (mag1 * mag2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

function visible(landmarks, idx) {
  const lm = landmarks[idx];
  return lm && (lm.visibility ?? 0) >= MIN_VISIBILITY;
}

function setReadout(readoutEl, valueEl, text, state) {
  valueEl.textContent = text;
  readoutEl.classList.remove("ok", "warn", "uncertain");
  readoutEl.classList.add(state);
}

function updateBowArmReadout(landmarks) {
  const bowShoulder = rightHanded ? L_SHOULDER : R_SHOULDER;
  const bowElbow = rightHanded ? L_ELBOW : R_ELBOW;
  const bowWrist = rightHanded ? L_WRIST : R_WRIST;

  if (![bowShoulder, bowElbow, bowWrist].every((i) => visible(landmarks, i))) {
    setReadout(readoutBowArm, valueBowArm, "— uncertain", "uncertain");
    return;
  }
  const angle = angleAt(landmarks[bowShoulder], landmarks[bowElbow], landmarks[bowWrist]);
  if (angle === null) {
    setReadout(readoutBowArm, valueBowArm, "— uncertain", "uncertain");
    return;
  }
  const ok = angle >= BOW_ARM_ANGLE_MIN && angle <= BOW_ARM_ANGLE_MAX;
  setReadout(readoutBowArm, valueBowArm, `${Math.round(angle)}°`, ok ? "ok" : "warn");
}

function torsoLength(landmarks, shoulderIdx, hipIdx) {
  if (!visible(landmarks, shoulderIdx) || !visible(landmarks, hipIdx)) return null;
  const s = landmarks[shoulderIdx];
  const h = landmarks[hipIdx];
  return Math.hypot(s.x - h.x, s.y - h.y);
}

function updateDrawElbowReadout(landmarks) {
  const drawShoulder = rightHanded ? R_SHOULDER : L_SHOULDER;
  const drawElbow = rightHanded ? R_ELBOW : L_ELBOW;
  const drawHip = rightHanded ? R_HIP : L_HIP;
  const bowShoulder = rightHanded ? L_SHOULDER : R_SHOULDER;
  const bowHip = rightHanded ? L_HIP : R_HIP;

  if (!visible(landmarks, drawShoulder) || !visible(landmarks, drawElbow)) {
    setReadout(readoutElbow, valueElbow, "— uncertain", "uncertain");
    return;
  }

  // Scale reference: torso length (shoulder-to-hip), draw side preferred, bow side as fallback.
  // Side-on framing makes the two shoulders nearly overlap in the image, so we use torso
  // length rather than the shoulder line itself as the "how big is this person" yardstick.
  const scale =
    torsoLength(landmarks, drawShoulder, drawHip) ?? torsoLength(landmarks, bowShoulder, bowHip);
  if (!scale) {
    setReadout(readoutElbow, valueElbow, "— uncertain", "uncertain");
    return;
  }

  const shoulder = landmarks[drawShoulder];
  const elbow = landmarks[drawElbow];
  // Image y grows downward, so a smaller elbow.y (higher on screen) means the elbow is higher.
  const verticalOffset = shoulder.y - elbow.y;
  const degrees = (Math.atan2(verticalOffset, scale) * 180) / Math.PI;

  const ok = degrees >= DRAW_ELBOW_HEIGHT_MIN && degrees <= DRAW_ELBOW_HEIGHT_MAX;
  const sign = degrees >= 0 ? "+" : "";
  setReadout(readoutElbow, valueElbow, `${sign}${Math.round(degrees)}°`, ok ? "ok" : "warn");
}

// The full-draw signal. Four things all have to be true at once:
//  - the draw-hand wrist has arrived near the face (mouth corners, falling back to the nose)
//  - the bow arm is substantially extended
//  - the two wrists are far apart (rules out the raise, where both hands are still up near
//    the head together, close before the arms are ever drawn apart)
//  - the draw wrist has stopped moving (rules out the raise, which is fast and continuous,
//    vs. anchor, which is a held pause)
// All distances are normalised by torso length so this works the same at any distance from
// the camera, and doesn't care which way the archer (or the mirrored camera) is facing.
// Returns false — never a guess — if a landmark it needs is uncertain.
function isAtFullDraw(landmarks, nowMs) {
  const drawWrist = rightHanded ? R_WRIST : L_WRIST;
  const bowShoulder = rightHanded ? L_SHOULDER : R_SHOULDER;
  const bowElbow = rightHanded ? L_ELBOW : R_ELBOW;
  const bowWrist = rightHanded ? L_WRIST : R_WRIST;
  const drawShoulder = rightHanded ? R_SHOULDER : L_SHOULDER;
  const drawHip = rightHanded ? R_HIP : L_HIP;
  const bowHip = rightHanded ? L_HIP : R_HIP;

  if (DEBUG) debugInfo = null; // cleared unless we make it all the way through below

  if (![drawWrist, bowShoulder, bowElbow, bowWrist].every((i) => visible(landmarks, i))) {
    return false;
  }

  let anchor;
  if (visible(landmarks, MOUTH_L) && visible(landmarks, MOUTH_R)) {
    anchor = {
      x: (landmarks[MOUTH_L].x + landmarks[MOUTH_R].x) / 2,
      y: (landmarks[MOUTH_L].y + landmarks[MOUTH_R].y) / 2,
    };
  } else if (visible(landmarks, NOSE)) {
    anchor = landmarks[NOSE];
  } else {
    return false;
  }

  const scale =
    torsoLength(landmarks, drawShoulder, drawHip) ?? torsoLength(landmarks, bowShoulder, bowHip);
  if (!scale) return false;

  const wrist = landmarks[drawWrist];
  const bowWristPos = landmarks[bowWrist];
  const anchorDist = Math.hypot(wrist.x - anchor.x, wrist.y - anchor.y) / scale;
  const handSep = Math.hypot(wrist.x - bowWristPos.x, wrist.y - bowWristPos.y) / scale;

  const bowArmAngle = angleAt(landmarks[bowShoulder], landmarks[bowElbow], landmarks[bowWrist]);
  if (bowArmAngle === null) return false;

  // Stillness: compare to where the draw wrist was last frame. Speed (distance moved per
  // second), not raw distance, so it doesn't depend on how often this happens to get called.
  // No previous frame yet means we can't know it's still, so treat that as "moving".
  const prev = lastDrawWrist;
  lastDrawWrist = { x: wrist.x, y: wrist.y, t: nowMs };
  const dtSec = prev ? (nowMs - prev.t) / 1000 : 0;
  const speed =
    prev && dtSec > 0 ? Math.hypot(wrist.x - prev.x, wrist.y - prev.y) / scale / dtSec : Infinity;

  const anchorOk = anchorDist <= FULL_DRAW_ANCHOR_MAX;
  const armOk = bowArmAngle >= FULL_DRAW_BOW_ARM_MIN;
  const sepOk = handSep >= FULL_DRAW_HAND_SEP_MIN;
  const stillOk = speed <= FULL_DRAW_STILL_MAX;

  if (DEBUG) debugInfo = { anchorDist, anchorOk, handSep, sepOk, bowArmAngle, armOk, speed, stillOk };

  return anchorOk && armOk && sepOk && stillOk;
}

// Pure state-machine transition for the auto-freeze logic — no DOM, no MediaPipe, easy to
// unit-test in isolation (see selfTest below). 'manual' is a dead end on purpose: this
// function must never be the thing that ends a manual freeze.
function nextFreezeState(state, atFullDraw, nowMs) {
  switch (state.kind) {
    case "manual":
      return state;
    case "armed":
      return atFullDraw ? { kind: "holding", since: nowMs } : state;
    case "holding":
      if (!atFullDraw) return { kind: "armed" };
      return nowMs - state.since >= FULL_DRAW_HOLD_MS ? { kind: "frozen", since: nowMs } : state;
    case "frozen":
      return nowMs - state.since >= AUTO_FREEZE_HOLD_MS ? { kind: "cooldown" } : state;
    case "cooldown":
      return atFullDraw ? state : { kind: "armed" };
    default:
      return { kind: "armed" };
  }
}

function drawSkeleton(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: "#00e5ff",
    lineWidth: 3,
  });
  drawingUtils.drawLandmarks(landmarks, { color: "#ffffff", radius: 4 });
}

function syncFreezeUI() {
  const shouldPause = isFrozen(freezeState);
  if (shouldPause && !video.paused) video.pause();
  if (!shouldPause && video.paused) video.play();

  btnFreeze.classList.toggle("active", shouldPause);
  btnFreeze.textContent = shouldPause ? "▶ Resume" : "⏸ Freeze";

  // Only the auto-frozen state gets a status message — a manual freeze needs no explanation.
  if (freezeState.kind === "frozen") {
    statusEl.classList.remove("hidden");
    statusEl.textContent = "Auto-froze at full draw";
  } else if (freezeState.kind !== "manual") {
    statusEl.classList.add("hidden");
  }
}

// ?debug-only readout of why the auto-freeze trigger is (or isn't) firing. No-op — not even
// a DOM lookup beyond the one at startup — when ?debug isn't in the URL.
function syncDebugOverlay() {
  if (!DEBUG) return;
  const d = debugInfo;
  const mark = (ok) => (ok ? "OK" : "FAIL");
  debugEl.textContent = !d
    ? `state: ${freezeState.kind}\n(no full-draw signal this frame)`
    : [
        `state: ${freezeState.kind}`,
        `anchor dist: ${d.anchorDist.toFixed(2)} ${mark(d.anchorOk)}`,
        `hand sep: ${d.handSep.toFixed(2)} ${mark(d.sepOk)}`,
        `bow arm: ${d.bowArmAngle.toFixed(0)}° ${mark(d.armOk)}`,
        `still: ${isFinite(d.speed) ? d.speed.toFixed(2) : "n/a"} ${mark(d.stillOk)}`,
      ].join("\n");
}

function renderLoop() {
  requestAnimationFrame(renderLoop);
  const now = performance.now();

  if (freezeState.kind === "manual") return; // holds until the button is tapped again

  if (isFrozen(freezeState)) {
    // Auto-frozen: no new landmarks to look at, just watch the clock for the auto-release.
    freezeState = nextFreezeState(freezeState, false, now);
    syncFreezeUI();
    syncDebugOverlay();
    return;
  }

  if (!poseLandmarker || video.readyState < 2) return;

  const result = poseLandmarker.detectForVideo(video, now);
  const landmarks = result.landmarks?.[0];
  let atFullDraw = false;

  if (!landmarks) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setReadout(readoutBowArm, valueBowArm, "— uncertain", "uncertain");
    setReadout(readoutElbow, valueElbow, "— uncertain", "uncertain");
    if (DEBUG) debugInfo = null;
  } else {
    drawSkeleton(landmarks);
    updateBowArmReadout(landmarks);
    updateDrawElbowReadout(landmarks);
    atFullDraw = isAtFullDraw(landmarks, now);
  }

  freezeState = nextFreezeState(freezeState, atFullDraw, now);
  syncFreezeUI();
  syncDebugOverlay();
}

function updateHandButtonLabel() {
  btnHand.textContent = rightHanded ? "🎯 Right-handed" : "🎯 Left-handed";
}

btnCamera.addEventListener("click", async () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  await startCamera();
});

btnHand.addEventListener("click", () => {
  rightHanded = !rightHanded;
  updateHandButtonLabel();
});

btnFreeze.addEventListener("click", () => {
  // Manual tap always wins: it resumes anything frozen (auto or manual), or freezes
  // indefinitely from live. Resuming goes to "cooldown" rather than "armed" so the auto
  // logic doesn't immediately re-trigger if the archer is still at full draw.
  freezeState = isFrozen(freezeState) ? { kind: "cooldown" } : { kind: "manual" };
  syncFreezeUI();
});

async function main() {
  try {
    statusEl.textContent = "Loading pose model…";
    await initPoseLandmarker();
    drawingUtils = new DrawingUtils(ctx);

    statusEl.textContent = "Starting camera…";
    await startCamera();

    statusEl.classList.add("hidden");
    updateHandButtonLabel();
    renderLoop();
  } catch (err) {
    statusEl.classList.remove("hidden");
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

// Plain-assert checks for the auto-freeze state machine — no framework, no fixtures.
// Open the page as ...index.html?selftest and read the console.
function selfTest() {
  const HOLD = FULL_DRAW_HOLD_MS;
  const FREEZE = AUTO_FREEZE_HOLD_MS;

  // Fires only after the hold duration.
  let s = nextFreezeState({ kind: "armed" }, true, 0);
  console.assert(s.kind === "holding", "full draw should start the hold, not fire immediately");
  s = nextFreezeState(s, true, HOLD - 1);
  console.assert(s.kind === "holding", "should not fire before the hold duration elapses");
  s = nextFreezeState(s, true, HOLD);
  console.assert(s.kind === "frozen", "should fire once the hold duration elapses");

  // Does not re-fire without leaving full draw first.
  let s2 = nextFreezeState(s, true, HOLD + FREEZE); // hold window ends, still at full draw
  console.assert(s2.kind === "cooldown", "should release into cooldown, not straight back to frozen");
  s2 = nextFreezeState(s2, true, HOLD + FREEZE + 1000);
  console.assert(s2.kind === "cooldown", "should stay in cooldown while still at full draw");
  s2 = nextFreezeState(s2, false, HOLD + FREEZE + 2000);
  console.assert(s2.kind === "armed", "should re-arm only after leaving full draw");

  // Releases after the hold window.
  let s3 = nextFreezeState({ kind: "frozen", since: 0 }, false, FREEZE - 1);
  console.assert(s3.kind === "frozen", "should still be frozen just before the hold window ends");
  s3 = nextFreezeState(s3, false, FREEZE);
  console.assert(s3.kind === "cooldown", "should release once the hold window ends");

  // Manual freeze is not overridden.
  const s4 = nextFreezeState({ kind: "manual" }, true, 999999);
  console.assert(s4.kind === "manual", "manual freeze must never be auto-released or overridden");

  // isAtFullDraw: the raise (bow arm straight, both hands up near the face together) must
  // NOT read as full draw — that was the field bug. Only real full draw (hands apart, held
  // still) should. Saves/restores the module state isAtFullDraw depends on so this doesn't
  // disturb the real app.
  const savedHanded = rightHanded;
  const savedLastWrist = lastDrawWrist;
  rightHanded = true;
  const mkLandmarks = (overrides) => {
    const lm = Array.from({ length: 25 }, () => ({ x: 0, y: 0, visibility: 1 }));
    for (const i in overrides) lm[i] = { ...lm[i], ...overrides[i] };
    return lm;
  };
  // Shared skeleton scale: shoulder-to-hip torso length of 0.3, bow arm dead straight.
  const base = {
    9: { x: 0.5, y: 0.3 }, // mouth L
    10: { x: 0.5, y: 0.3 }, // mouth R
    11: { x: 0.3, y: 0.3 }, // bow (left) shoulder
    12: { x: 0.5, y: 0.3 }, // draw (right) shoulder
    13: { x: 0.15, y: 0.3 }, // bow elbow
    23: { x: 0.3, y: 0.6 }, // bow hip
    24: { x: 0.5, y: 0.6 }, // draw hip
  };

  lastDrawWrist = null;
  const raise = mkLandmarks({ ...base, 15: { x: 0.48, y: 0.3 }, 16: { x: 0.52, y: 0.31 } });
  console.assert(
    isAtFullDraw(raise, 0) === false,
    "raise (hands together, bow arm straight) must not read as full draw"
  );

  lastDrawWrist = null;
  const drawn = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.52, y: 0.31 } });
  console.assert(
    isAtFullDraw(drawn, 0) === false,
    "first frame at full draw should read as still-moving (no prior position yet)"
  );
  console.assert(
    isAtFullDraw(drawn, 500) === true,
    "same position 500ms later (zero speed) should read as full draw"
  );
  const drifted = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.6, y: 0.31 } });
  console.assert(
    isAtFullDraw(drifted, 600) === false,
    "wrist jumping far in 100ms (fast) should not read as holding still"
  );

  rightHanded = savedHanded;
  lastDrawWrist = savedLastWrist;

  console.log("selfTest done — check above for any failed console.assert");
}

if (location.search.includes("selftest")) selfTest();

main();
