import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ===== CALIBRATE WITH COACH — placeholder target ranges, edit freely =====
const BOW_ARM_ANGLE_MIN = 165; // degrees at the bow elbow; 180 = perfectly straight
const BOW_ARM_ANGLE_MAX = 180;

const MIN_VISIBILITY = 0.6; // MediaPipe's 0–1 confidence per joint; below this we show "uncertain"

// Placeholders, compound-specific like the rest of this block — owner will tune these with a
// coach once there's real footage to compare against.
const SHOULDER_DROP_MIN_PCT = 45; // % of torso length; each shoulder's ear-to-shoulder gap must be at least this to count as "dropped" rather than "shrugged". Rough anatomy puts a relaxed adult around 45-50% and a shrugged one nearer 40%, so this sits at the boundary — but it is a desk estimate, not a measurement. Replace it with real numbers off the shot log after one session: shoot a few deliberately shrugged and a few deliberately dropped, and put this between the two clusters
const DRAW_ELBOW_ALIGN_MAX_DEVIATION = 8; // degrees the draw elbow may sit off the bow-wrist-to-draw-wrist line (extended backwards), either high or low, and still count as "in line" with the arrow

// These five are tuned for COMPOUND shooting (mechanical release aid, let-off held at full
// draw) — if the owner starts shooting recurve with this, expect to revisit all five, since a
// recurve archer anchors differently (fingers under the chin, not a release hand near the jaw)
// and can't hold nearly as steady (no let-off, fighting full poundage the whole time).
const FULL_DRAW_ANCHOR_MAX = 0.45; // draw-hand wrist must be this close to the mouth/nose, as a fraction of torso length, to count as "at anchor" — looser than you might expect because a release aid sits the hand further back near the jaw than fingers under the chin would; hand separation (below) does the real discriminating, this just filters out the grossly wrong
const FULL_DRAW_BOW_ARM_MIN = 150; // degrees; bow arm must be at least this straight to count as "drawn" (looser than the good-form target above — full draw should still be detected on so-so form)
const FULL_DRAW_HAND_SEP_MIN = 0.75; // the two wrists must be at least this far apart, as a fraction of torso length, to count as "drawn" — during the raise both hands are close together near the head, only at full draw are they a draw-length apart. THE key signal: a compound's draw length is fixed by a mechanical stop, so this is near-binary (mid-raise vs. hard against the wall) and can be set with confidence
const DRAW_ATTEMPT_MIN_SEP = 0.3; // fraction of torso length; hand separation must drop back below this (hands back together, at rest between shots) before the shot log below will treat the NEXT rise as a new attempt — this is what stops one long hold from being logged as several rows, and stops two separate shots from being merged into one
const FULL_DRAW_STILL_MAX = 0.35; // the draw wrist may drift at most this much (as a fraction of torso length) per second and still count as "holding still" — kept tight (not loosened) because a compound archer at let-off is genuinely steady, unlike the fast continuous motion of the raise

// Shot-log ATTEMPT GATING — decide whether a draw attempt plausibly happened at all, before it's
// allowed to become a row. Added after a field report: with only DRAW_ATTEMPT_MIN_SEP as a floor
// (15cm of hand separation), every hand movement past that — nocking an arrow, lowering the bow,
// adjusting a release aid, standing with arms slightly apart — was getting logged as a shot, and
// those phantom rows were dragging the session average around (see CLAUDE.md and README). These
// two are separate from DRAW_ATTEMPT_MIN_SEP above, which only decides when an in-progress
// attempt STARTS and ENDS, and separate from FULL_DRAW_HAND_SEP_MIN, which a shot still does NOT
// need to reach to be logged (see endAttempt) — a draw that fell short of full draw is still
// worth seeing, on purpose. These just rule out things that were never plausibly a draw at all.
const SHOT_MIN_PEAK_SEP_FRACTION = 0.8; // the attempt's peak hand separation must reach at least this fraction of FULL_DRAW_HAND_SEP_MIN (0.8 x 0.75 = 0.6 torso-lengths apart) to count as a real draw attempt — comfortably above the ~0.3-0.5 range nocking/lowering the bow produces, comfortably below the 0.75 that counts as full draw itself
const SHOT_MIN_DURATION_MS = 600; // an attempt must last at least this long, from when hands first cross DRAW_ATTEMPT_MIN_SEP to when they drop back below it, to count as a real draw rather than a brief noise spike (a hand passing near the body, a tracking glitch) — a real compound draw, even a rushed one, takes real time to raise, draw and settle

// Shot log display-only cutoffs — NOT form targets like the numbers above. These decide when a
// shot's own DEVIATION FROM THE OWNER'S OWN SESSION AVERAGE (see summarizeShots) gets coloured
// as "close to your average" vs "worth a look", nothing more. No coaching authority behind
// them either — picked as a plausible slice of a normal end's natural wobble — but unlike the
// block above, there's no "correct" value to eventually replace them with: they only ever judge
// a shot against the owner's own numbers, so tune by eye once there's a real session to look at.
const BOW_ARM_CONSISTENCY_MAX_DEVIATION = 3; // degrees from this session's average bow-arm angle
const SHOULDER_DROP_CONSISTENCY_MAX_DEVIATION = 8; // percentage points from this session's average
const ELBOW_ALIGN_CONSISTENCY_MAX_DEVIATION = 4; // degrees from this session's average
// ===========================================================================

// ===== SHOT CLIPS — recording settings. Not calibration like the block above, just knobs for
// clip length/size; change freely without asking a coach first. =====
const CLIP_TAIL_MS = 2500; // how long to keep recording AFTER endAttempt, so the release and follow-through make it into the clip, not just the draw
const CLIP_MAX_MS = 20000; // hard ceiling on one clip's length, so a full-draw detection that gets stuck "in progress" can't record forever
const CLIP_FRAME_RATE = 24; // fps requested from canvas.captureStream — modest on purpose, see CLIP_BITRATE
const CLIP_BITRATE = 1_500_000; // ~1.5 Mbps target — keeps a clip a couple of MB, not tens, since these live in memory for the whole session
// Tried in this order, first one the browser claims to support wins. iPhone Safari (the real
// target) only understands the mp4 entries; desktop Chrome (used for dev) only understands the
// webm ones — so both ends of that split need to be in the list, most-preferred first.
const CLIP_MIME_CANDIDATES = [
  "video/mp4;codecs=avc1.42E01E",
  "video/mp4",
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm",
];
// ===========================================================================

// ===== SMOOTHING — One Euro filter tunables for the pose landmarks. Performance/feel knobs,
// not calibration like the CALIBRATE WITH COACH block above: change these freely, no coach
// needed, just watch the skeleton and judge by eye (or check a recorded clip afterwards). See
// the OneEuroFilter class below for what these numbers actually do; in short, "how much to
// smooth a joint that's barely moving" (mincutoff) and "how fast to stop smoothing once a joint
// speeds up" (beta). =====
const SMOOTH_MIN_CUTOFF = 1.0; // Hz-scale smoothing floor, used when a joint is nearly still. LOWER this for a calmer, less jittery skeleton at full draw (where the archer is genuinely holding steady and every measurement gets taken); RAISE it if the skeleton starts to feel laggy even when barely moving. Picked at the paper's own default starting point (1.0), then left there — outdoor jitter is exactly the "nearly still, high-frequency noise" case this is built to kill.
const SMOOTH_BETA = 8; // how fast smoothing backs off as a joint speeds up. Landmark positions here are normalised to roughly 0–1 across the frame (not pixels), so a joint moving during a real raise or draw covers a big fraction of that range per second — this needs to be much bigger than the tiny beta values you'll see in mouse-pointer examples online, or the skeleton would lag behind the archer's arm during the raise. RAISE this if the skeleton looks like it's dragging behind a fast motion; LOWER it if fast motion still looks jittery instead of smooth.
const SMOOTH_DCUTOFF = 1.0; // smooths the estimated SPEED of a joint before beta reacts to it, so a single noisy frame can't fake a "fast motion" and prematurely unlock smoothing. Standard default from the paper; rarely worth touching — leave this one alone unless the adaptive behaviour itself (calm when still, responsive when moving) seems broken.
// ===========================================================================

// ===== POSE MODEL — performance knobs, not calibration; safe to change without a coach. =====
// MediaPipe offers a "lite" pose model (fast, less stable) and a "full" one (steadier landmarks,
// more GPU work per frame). This app starts on "full" for the steadier skeleton, then times
// itself for a short window right after startup and switches itself to "lite" if this phone
// can't keep full running at a usable frame rate. The owner never sees this decision happen —
// per CLAUDE.md they get one interaction, after they're done shooting — so the result (which
// model ended up running, and roughly how fast) is written into the shot log instead of shown
// live anywhere.
const MODEL_WARMUP_FRAMES = 10; // frames ignored before measuring starts — the first frames after startup are always slow (cold GPU pipeline, cold caches) and would make even a fast phone look like it needs the fallback
const MODEL_MEASURE_FRAMES = 20; // frames averaged together, right after warm-up, to make the one-time full-vs-lite decision
const MODEL_SLOW_FRAME_MS = 60; // average per-frame inference time above this triggers the switch to lite (60ms ≈ 16fps, noticeably behind a live camera feed). RAISE this to let the app stay on the steadier "full" model on a slower phone; LOWER it to fall back to "lite" more readily.
// ===========================================================================

const DEBUG = location.search.includes("debug"); // ?debug in the URL shows the live trigger-condition overlay

// MediaPipe pose landmark indices (33-point model)
const L_SHOULDER = 11, R_SHOULDER = 12;
const L_ELBOW = 13, R_ELBOW = 14;
const L_WRIST = 15, R_WRIST = 16;
const L_HIP = 23, R_HIP = 24;
const NOSE = 0, MOUTH_L = 9, MOUTH_R = 10;
const L_EAR = 7, R_EAR = 8;

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const btnCamera = document.getElementById("btn-camera");
const btnHand = document.getElementById("btn-hand");
const readoutBowArm = document.getElementById("readout-bowarm");
const valueBowArm = document.getElementById("value-bowarm");
const valueShoulderBow = document.getElementById("value-shoulder-bow");
const valueShoulderDraw = document.getElementById("value-shoulder-draw");
const readoutElbow = document.getElementById("readout-elbow");
const valueElbow = document.getElementById("value-elbow");
const debugEl = document.getElementById("debug");
if (DEBUG) debugEl.classList.remove("hidden");
const btnLog = document.getElementById("btn-log");
const shotLogEl = document.getElementById("shotlog");
const clipPlayerEl = document.getElementById("clipplayer");
const clipPlayerVideo = document.getElementById("clipplayer-video");
const clipPlayerClose = document.getElementById("clipplayer-close");
const clipPlayerRateBtns = document.querySelectorAll(".clipplayer-rate");

// Whether this browser can record a clip at all. Checked once, up front, rather than
// discovering it the first time an attempt starts — that way the "clips unavailable" banner
// (see markClipsUnavailable) can go up before the owner ever shoots, not after their first shot
// quietly has no video.
const CLIP_SUPPORTED = typeof MediaRecorder !== "undefined" && typeof canvas.captureStream === "function";

let poseLandmarker = null;
let stream = null;
let facingMode = "environment"; // rear camera first
let rightHanded = true;
let drawingUtils = null;

// Previous frame's draw-wrist position + timestamp, for the stillness check in isAtFullDraw
// below. Deliberately just one remembered frame, not a history buffer — cheap and enough.
let lastDrawWrist = null;

// Last frame's full-draw condition values, for the ?debug overlay only (see syncDebugOverlay).
// Stays null whenever isAtFullDraw bails out before it can compute them.
let debugInfo = null;

// Shot log: a persistent record the owner can check after they've finished shooting, because
// they cannot read the screen or tap anything while actually on the line (see CLAUDE.md). One
// row per draw attempt — the best (highest hand-separation) frame seen during it, whether or
// not it ever reached full draw — kept until the page reloads. No timer anywhere in this:
// entries never expire or get overwritten just because time passed, only because a newer
// attempt bumps an old one out of the last SHOT_LOG_MAX.
const SHOT_LOG_MAX = 10;
let shotCount = 0; // total LOGGED attempts this session (see endAttempt) — keeps counting even once the log above fills up
let log = []; // newest first
let attempt = null; // the attempt currently in progress, if any — see trackShotAttempt below

// How many draw attempts got discarded by the gates in endAttempt below — too short, or never
// got anywhere near full draw. The owner can't watch this filtering happen, so if the app is
// quietly discarding movement it has to say so every time the log is read, not just once: see
// the persistent line built from this in renderShotLog. Never reset mid-session, same treatment
// as shotCount above.
let rejectedAttemptCount = 0;

// The clip recording currently in progress or in its post-shot tail, if any — see
// startClipRecording below. Only ever one of these at a time; a new attempt starting while the
// previous clip is still in its tail cuts the old one short rather than running two at once.
let activeRecording = null;

// Set once, the first time clip recording turns out not to work in this browser (either it was
// never supported, or MediaRecorder threw when we tried to start it) — and never cleared. Shown
// as a persistent line at the top of the shot log for the rest of the session (see
// markClipsUnavailable / renderShotLog), because the owner can't be watching a console or a
// toast at the moment recording fails; they find out later, standing at the phone, so that's the
// only place this can usefully be said.
let clipsUnavailableReason = null;

// True only while selfTest() below is running. trackShotAttempt calls startClipRecording every
// time a fresh attempt begins, and selfTest drives trackShotAttempt directly with fake
// landmarks, many times, with no real camera behind canvas — so without this guard, plain logic
// tests would spin up real MediaRecorders against a 0×0 canvas and leave their timers running
// into the real session that follows (selfTest always runs before main(), never instead of it).
// The clip logic itself is tested separately below, by calling its pieces directly.
let selfTestInProgress = false;

// ===== One Euro filter — adaptive smoothing for the pose landmarks (Casiez/Roussel/Vogel 2012,
// "1€ Filter"). Deliberately implemented inline rather than pulled in as a dependency — it's
// short, and CLAUDE.md rules out adding a CDN dependency for this.
//
// Why not a plain moving average: a fixed amount of smoothing is a bad trade either way. Enough
// to kill outdoor jitter at full draw also blurs the fast raise into a laggy smear that trails
// behind the archer's real arm and throws off the full-draw-detection timing (see isAtFullDraw).
// Not enough to blur the raise leaves the jitter untouched at full draw — the exact moment every
// measurement is taken and jitter is most damaging. One Euro sidesteps the trade-off: it widens
// its own low-pass cutoff frequency in proportion to how fast the signal is currently moving, so
// it smooths hard when a joint is nearly still (full draw — a compound archer at let-off holds
// genuinely steady for seconds) and gets out of the way automatically the instant the joint
// speeds up (the raise). Pure function/class — no DOM, no MediaPipe — so selfTest can exercise
// it directly.
class OneEuroFilter {
  constructor(mincutoff, beta, dcutoff) {
    this.mincutoff = mincutoff;
    this.beta = beta;
    this.dcutoff = dcutoff;
    this.reset();
  }

  // Clears all state. After a reset, the next call to filter() returns its input completely
  // unsmoothed — see the tPrev === null branch below — so a stale position from before the reset
  // can never be blended into a fresh one.
  reset() {
    this.xPrev = null; // last filtered value
    this.dxPrev = 0; // last filtered speed estimate
    this.tPrev = null; // last timestamp (seconds), so dt is measured from REAL elapsed time, not an assumed frame rate
  }

  // The smoothing factor (0–1) for a first-order low-pass filter with the given cutoff frequency
  // (Hz) over an elapsed time dt (seconds). Standard formula from the One Euro paper.
  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  // Feeds one new raw sample through the filter and returns the smoothed value. `tSec` is a
  // timestamp in seconds on any consistent clock — the render loop passes performance.now()/1000.
  filter(value, tSec) {
    if (this.tPrev === null) {
      // First sample since construction (or since the last reset): nothing to smooth against
      // yet, so the output IS the input. This is what makes reset() clean rather than a source
      // of drag — see the "reset clears state" self-test below.
      this.xPrev = value;
      this.dxPrev = 0;
      this.tPrev = tSec;
      return value;
    }
    const dt = Math.max(tSec - this.tPrev, 1e-6); // guards divide-by-zero on a duplicate/out-of-order timestamp; real frames always have dt > 0
    // Estimate how fast the signal is currently moving, itself low-pass filtered at a fixed
    // dcutoff — an unfiltered derivative would be exactly as noisy as the raw signal and could
    // never be trusted to decide how much to smooth.
    const rawSpeed = (value - this.xPrev) / dt;
    const speedAlpha = OneEuroFilter.alpha(this.dcutoff, dt);
    const speed = this.dxPrev + speedAlpha * (rawSpeed - this.dxPrev);

    // The adaptive part: the faster the signal is moving, the higher this cutoff climbs, and the
    // less smoothing gets applied below.
    const cutoff = this.mincutoff + this.beta * Math.abs(speed);
    const a = OneEuroFilter.alpha(cutoff, dt);
    const smoothed = this.xPrev + a * (value - this.xPrev);

    this.xPrev = smoothed;
    this.dxPrev = speed;
    this.tPrev = tSec;
    return smoothed;
  }
}

// Runs one OneEuroFilter per landmark, per axis (x and y — NEVER visibility; visibility is a
// confidence score, not a position, and smoothing it would let a low-confidence joint's readout
// creep toward looking trustworthy over a few frames, which CLAUDE.md explicitly forbids — a
// joint's visibility must reflect THIS frame's confidence only). Pure array-in, array-out — no
// DOM, no MediaPipe — so selfTest can exercise it directly on plain fixture landmarks.
class LandmarkSmoother {
  constructor(mincutoff, beta, dcutoff) {
    this.mincutoff = mincutoff;
    this.beta = beta;
    this.dcutoff = dcutoff;
    this.filters = []; // filters[i] = { x: OneEuroFilter, y: OneEuroFilter }, created lazily per landmark index the first time it's seen
  }

  // Drops all filter state. Called whenever tracking is lost or the camera is switched, so a
  // position left over from before the gap can never get smoothed together with a fresh position
  // after it and drag the skeleton across the frame.
  reset() {
    this.filters = [];
  }

  // landmarks: MediaPipe's per-frame array of {x, y, z, visibility, ...}. tSec: timestamp in
  // seconds. Returns a NEW array — the input array is never mutated — with x/y replaced by their
  // smoothed values and every other field, visibility included, copied through untouched.
  smooth(landmarks, tSec) {
    return landmarks.map((lm, i) => {
      if (!this.filters[i]) {
        this.filters[i] = {
          x: new OneEuroFilter(this.mincutoff, this.beta, this.dcutoff),
          y: new OneEuroFilter(this.mincutoff, this.beta, this.dcutoff),
        };
      }
      const f = this.filters[i];
      return { ...lm, x: f.x.filter(lm.x, tSec), y: f.y.filter(lm.y, tSec) };
    });
  }
}

// One instance for the live session, built from the SMOOTHING constants above.
const landmarkSmoother = new LandmarkSmoother(SMOOTH_MIN_CUTOFF, SMOOTH_BETA, SMOOTH_DCUTOFF);
// ===========================================================================

// ===== Pose model selection — see the POSE MODEL constants above for the full explanation.
// These variables track the one-time "is full fast enough on this phone?" measurement and its
// outcome, and the persistent shot-log line that reports it (the owner can't watch this happen
// live, per CLAUDE.md).
const POSE_MODEL_URLS = {
  full: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
  lite: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
};
let visionFileset = null; // cached FilesetResolver result, so switching models doesn't re-fetch the wasm runtime a second time
let activePoseModel = "full"; // which model is actually loaded right now
let modelDecisionMade = false; // true once the one-time full-vs-lite decision has run — after that, measurePoseModelPerf stops doing any work, so the model never hunts back and forth mid-session
let modelWarmupSeen = 0; // frames seen since startup, counted toward MODEL_WARMUP_FRAMES then MODEL_MEASURE_FRAMES
let modelMeasureTotalMs = 0; // summed inference time across the measurement window (after warm-up)
// Persistent shot-log line recording which model ended up running and the measured frame rate —
// same "written once, shown on every render, never only live" treatment as clipsUnavailableReason
// below, and for the same reason: the owner is on the shooting line when this gets decided, not
// watching the phone.
let modelStatusLine = null;

async function getVisionFileset() {
  if (!visionFileset) {
    visionFileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
  }
  return visionFileset;
}

async function createPoseLandmarker(modelKey) {
  const vision = await getVisionFileset();
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: POSE_MODEL_URLS[modelKey],
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

async function initPoseLandmarker() {
  poseLandmarker = await createPoseLandmarker("full");
  activePoseModel = "full";
}

// Called once per frame from renderLoop with how long that frame's detectForVideo call took, in
// milliseconds. No-ops once modelDecisionMade is true — see that variable's comment. Skips the
// first MODEL_WARMUP_FRAMES entirely (cold-start frames are always slow and not representative),
// then averages the next MODEL_MEASURE_FRAMES to make the one-time decision.
function measurePoseModelPerf(inferenceMs) {
  if (modelDecisionMade) return;
  modelWarmupSeen++;
  if (modelWarmupSeen <= MODEL_WARMUP_FRAMES) return;
  modelMeasureTotalMs += inferenceMs;
  const measured = modelWarmupSeen - MODEL_WARMUP_FRAMES;
  if (measured < MODEL_MEASURE_FRAMES) return;

  modelDecisionMade = true;
  const avgMs = modelMeasureTotalMs / measured;
  const fps = 1000 / avgMs;
  if (avgMs > MODEL_SLOW_FRAME_MS && activePoseModel === "full") {
    switchToLitePoseModel(fps);
  } else {
    setModelStatusLine(fps);
  }
}

// Rebuilds the landmarker on the lighter "lite" model because the warm-up measurement found
// "full" running too slowly on this phone. Must never leave the app with no landmarker at all,
// and a failed rebuild must never interrupt tracking: if creating the new landmarker throws (a
// flaky fetch for the model file, most likely), the OLD "full" landmarker just keeps running —
// slower than ideal, but still tracking, which is what matters (see CLAUDE.md: pose tracking
// must never just stop).
async function switchToLitePoseModel(measuredFps) {
  try {
    const next = await createPoseLandmarker("lite");
    const old = poseLandmarker;
    poseLandmarker = next;
    activePoseModel = "lite";
    old?.close?.();
  } catch (err) {
    console.error("archery-form-coach: falling back to lite pose model failed, staying on full", err);
  }
  setModelStatusLine(measuredFps);
}

function setModelStatusLine(measuredFps) {
  const label =
    activePoseModel === "full"
      ? "full"
      : "lite (auto-switched — full ran too slow on this phone)";
  modelStatusLine = `Pose model: ${label} — about ${measuredFps.toFixed(1)} fps measured at startup.`;
  renderShotLog();
}
// ===========================================================================

async function startCamera() {
  // A landmark position smoothed from BEFORE a camera switch (different framing, possibly a
  // mirrored front camera) must never blend into positions AFTER it — that would drag the
  // skeleton across the frame on the very first frames of the new camera. See LandmarkSmoother.
  landmarkSmoother.reset();
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

// Like setReadout, but classes the value span itself rather than its containing panel — used
// for the shoulder-drop readout, which reports two independent numbers (bow/draw) in one panel
// and so needs two independent ok/warn/uncertain states rather than one shared by the panel.
function setValueState(valueEl, text, state) {
  valueEl.textContent = text;
  valueEl.classList.remove("ok", "warn", "uncertain");
  valueEl.classList.add(state);
}

// Bow-arm angle as a plain number (or null if we can't tell) — pulled out of the readout
// function below so the shot log can record the exact same value without duplicating the math.
function bowArmAngleOf(landmarks) {
  const bowShoulder = rightHanded ? L_SHOULDER : R_SHOULDER;
  const bowElbow = rightHanded ? L_ELBOW : R_ELBOW;
  const bowWrist = rightHanded ? L_WRIST : R_WRIST;
  if (![bowShoulder, bowElbow, bowWrist].every((i) => visible(landmarks, i))) return null;
  return angleAt(landmarks[bowShoulder], landmarks[bowElbow], landmarks[bowWrist]);
}

function updateBowArmReadout(landmarks) {
  const angle = bowArmAngleOf(landmarks);
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

// Shoulder drop for one shoulder: the vertical gap between that shoulder and its ear,
// normalised by torso length and given as a percentage — bigger number = shoulder sits
// further from the ear = more dropped, which is what "dropping my shoulders more" means.
// Reported per shoulder (not averaged, see updateShoulderDropReadout below) because the
// common compound fault is one shoulder — usually the bow shoulder, under load — creeping up
// while the other stays fine; an average would hide exactly that.
function shoulderDropOf(landmarks, shoulderIdx, sameEarIdx, otherEarIdx, ownHipIdx, otherShoulderIdx, otherHipIdx) {
  if (!visible(landmarks, shoulderIdx)) return null;

  // Side-on framing often means the far ear is occluded or low-confidence. Prefer the ear on
  // the same side as the shoulder being measured, but fall back to the other one — for a
  // purely vertical gap, which ear supplies the y-coordinate matters far less than having one.
  const earIdx = visible(landmarks, sameEarIdx) ? sameEarIdx : visible(landmarks, otherEarIdx) ? otherEarIdx : null;
  if (earIdx === null) return null;

  // Same "own side preferred, other side as fallback" torso-length convention used everywhere
  // else in this file.
  const scale = torsoLength(landmarks, shoulderIdx, ownHipIdx) ?? torsoLength(landmarks, otherShoulderIdx, otherHipIdx);
  if (!scale) return null;

  const shoulder = landmarks[shoulderIdx];
  const ear = landmarks[earIdx];
  // Image y grows downward, so the ear normally sits above the shoulder (smaller y). That gap
  // shrinks as the shoulder shrugs up toward the ear, and grows as it drops away from it.
  return ((shoulder.y - ear.y) / scale) * 100;
}

// Both shoulders' drop in one call, so the readout and the shot log stay in sync using exactly
// the same numbers.
function shoulderDropSampleOf(landmarks) {
  const bowShoulder = rightHanded ? L_SHOULDER : R_SHOULDER;
  const bowHip = rightHanded ? L_HIP : R_HIP;
  const bowEar = rightHanded ? L_EAR : R_EAR;
  const drawShoulder = rightHanded ? R_SHOULDER : L_SHOULDER;
  const drawHip = rightHanded ? R_HIP : L_HIP;
  const drawEar = rightHanded ? R_EAR : L_EAR;

  return {
    bow: shoulderDropOf(landmarks, bowShoulder, bowEar, drawEar, bowHip, drawShoulder, drawHip),
    draw: shoulderDropOf(landmarks, drawShoulder, drawEar, bowEar, drawHip, bowShoulder, bowHip),
  };
}

function updateShoulderDropReadout(landmarks) {
  const { bow, draw } = shoulderDropSampleOf(landmarks);
  for (const [pct, valueEl] of [[bow, valueShoulderBow], [draw, valueShoulderDraw]]) {
    if (pct === null) {
      setValueState(valueEl, "—", "uncertain");
    } else {
      setValueState(valueEl, `${Math.round(pct)}%`, pct >= SHOULDER_DROP_MIN_PCT ? "ok" : "warn");
    }
  }
}

// Draw-elbow alignment (replaces the old draw-elbow-height readout, which measured height off
// the shoulder — a proxy for the wrong thing). What actually matters: at full draw, bow hand,
// draw hand and draw elbow should form one straight line, the elbow sitting on the arrow's
// line extended back past the anchor. So: the angle at the draw wrist between the ray to the
// bow wrist and the ray to the draw elbow — 180° is perfectly in line.
//
// KNOWN LIMITATION: "in line with the arrow" is really three-dimensional. Side-on, the camera
// can see whether the elbow is too high or too low, but has no way to see whether it's flared
// out horizontally away from the arrow line — that component is pure depth from this angle.
// This measures a real and important half of the goal, not the whole thing. Do not mistake it
// for a complete check, and don't try to reconstruct the missing dimension from a single
// side-on camera — it isn't there to reconstruct.
function drawElbowAlignmentOf(landmarks) {
  const drawWrist = rightHanded ? R_WRIST : L_WRIST;
  const drawElbow = rightHanded ? R_ELBOW : L_ELBOW;
  const bowWrist = rightHanded ? L_WRIST : R_WRIST;

  if (![drawWrist, drawElbow, bowWrist].every((i) => visible(landmarks, i))) return null;

  const wrist = landmarks[drawWrist];
  const elbow = landmarks[drawElbow];
  const bow = landmarks[bowWrist];

  const angle = angleAt(bow, wrist, elbow);
  if (angle === null) return null;
  const deviation = 180 - angle; // 0 = dead in line with the arrow; bigger = further off, either way

  // Direction (high/low) comes from vertical position ONLY — never from the sign of a cross
  // product. A cross product's sign flips with the mirrored front camera and with the
  // handedness toggle, so a fixed "positive = high" rule would silently report high as low for
  // a left-handed archer, or on the front camera. Vertical position doesn't flip either way.
  // So: extend the bow-wrist -> draw-wrist line out to the elbow's x, and compare the elbow's
  // actual height to where that line would put it.
  const dx = wrist.x - bow.x;
  if (Math.abs(dx) < 1e-6) return null; // near-vertical line: can't tell high from low this way

  const t = (elbow.x - bow.x) / dx;
  const expectedY = bow.y + t * (wrist.y - bow.y);
  // Image y grows downward: a smaller actual y than expected means the elbow sits higher.
  const direction = elbow.y < expectedY ? "high" : elbow.y > expectedY ? "low" : "level";

  // Signed version of the same reading, for averaging (see summarizeShots below) — deviation +
  // direction on its own can't be averaged honestly, since "8 high" and "8 low" would both look
  // like +8 and cancel out into a fake "consistent" reading. High is positive, low is negative
  // (arbitrary choice, kept consistent everywhere this value is used); the plain deviation +
  // direction text above is untouched, so the live readout keeps reading e.g. "6° high".
  const signed = direction === "high" ? deviation : direction === "low" ? -deviation : 0;

  return { deviation, direction, signed };
}

function updateDrawElbowReadout(landmarks) {
  const result = drawElbowAlignmentOf(landmarks);
  if (result === null) {
    setReadout(readoutElbow, valueElbow, "— uncertain", "uncertain");
    return;
  }
  const rounded = Math.round(result.deviation);
  const text = rounded === 0 ? "in line" : `${rounded}° ${result.direction}`;
  const ok = result.deviation <= DRAW_ELBOW_ALIGN_MAX_DEVIATION;
  setReadout(readoutElbow, valueElbow, text, ok ? "ok" : "warn");
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

  const bowArmAngle = bowArmAngleOf(landmarks);
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

  const atFullDraw = anchorOk && armOk && sepOk && stillOk;

  // Feed the shot log regardless of ?debug — the owner needs shot numbers/form readouts
  // whether or not the diagnostic overlay is on; only the display of the extra fields below
  // is debug-gated (see renderShotLog). nowMs is threaded through rather than calling
  // performance.now() again in here, so selfTest can drive the shot-log timing deterministically.
  trackShotAttempt(
    {
      handSep,
      bowArmAngle,
      shoulderDrop: shoulderDropSampleOf(landmarks),
      elbowAlign: drawElbowAlignmentOf(landmarks),
      anchorOk,
      armOk,
      sepOk,
      stillOk,
      atFullDraw,
    },
    nowMs
  );

  return atFullDraw;
}

// Attempt-boundary rule for the shot log: a draw attempt is "in progress" for as long as hand
// separation stays at/above DRAW_ATTEMPT_MIN_SEP, tracking whichever frame in it had the
// highest separation so far. It ends — and gets logged — the moment separation drops back
// below that floor (hands back together at rest). This is the simplest rule that both (a)
// doesn't split one long hold into several rows and (b) doesn't merge two separate shots taken
// back-to-back into one. No timer involved: nothing here expires on its own, ever.
// nowMs threads through from isAtFullDraw's own nowMs parameter (renderLoop's `now`) — never
// performance.now() called fresh in here, so selfTest can drive attempt timing deterministically
// (see SHOT_MIN_DURATION_MS above and endAttempt below, which is what actually uses it).
function trackShotAttempt(sample, nowMs) {
  if (sample.handSep >= DRAW_ATTEMPT_MIN_SEP) {
    const isNewAttempt = !attempt; // hands just left the resting position — a fresh attempt, not a continuation
    // startMs (when this attempt began) and reachedFullDraw (whether ANY frame in it was a true
    // full draw, not just its best-hand-separation frame) both have to survive being carried
    // forward across frames even on frames that DON'T beat the current best — see below.
    const startMs = isNewAttempt ? nowMs : attempt.startMs;
    const reachedFullDraw = (isNewAttempt ? false : attempt.reachedFullDraw) || !!sample.atFullDraw;
    if (isNewAttempt || sample.handSep >= attempt.handSep) {
      attempt = { ...sample, startMs, reachedFullDraw };
    } else {
      attempt.startMs = startMs;
      attempt.reachedFullDraw = reachedFullDraw;
    }
    // Recording starts here, not in endAttempt, so the raise and draw are in the clip too — by
    // the time endAttempt fires the good part is already over.
    if (isNewAttempt) startClipRecording();
  } else {
    endAttempt(nowMs);
  }
}

// Ends whatever attempt is in progress (if any). Called when hand separation drops back to
// resting (from trackShotAttempt above, with the current timestamp) or when the pose is lost
// entirely mid-attempt (from renderLoop, with ITS current timestamp) — either way, whatever was
// going on has stopped, and it's judged the same way regardless of which of those two things
// ended it: tracking loss must not manufacture a shot that the same movement, ending normally,
// wouldn't have earned.
//
// Two gates, both against the attempt's own best frame — see SHOT_MIN_PEAK_SEP_FRACTION and
// SHOT_MIN_DURATION_MS above for why these two specifically. An attempt that fails either one
// gets thrown away, not logged: counted in rejectedAttemptCount instead, and any clip recording
// still running for it gets finalised (and therefore its capture track stopped) right now rather
// than left to expire on its own — see finalizeRecording, and CLAUDE.md on why a clip must never
// outlive the shot it belongs to.
function endAttempt(nowMs) {
  if (!attempt) return;
  const a = attempt;
  attempt = null; // clear first — logShot/finalizeRecording below must never see a stale in-progress attempt

  const gotDeepEnough = a.handSep >= SHOT_MIN_PEAK_SEP_FRACTION * FULL_DRAW_HAND_SEP_MIN;
  const lastedLongEnough = typeof nowMs === "number" && typeof a.startMs === "number" && nowMs - a.startMs >= SHOT_MIN_DURATION_MS;

  if (!gotDeepEnough || !lastedLongEnough) {
    rejectedAttemptCount++;
    finalizeRecording(activeRecording); // this attempt's clip (if any) never gets a shot number — drop it and stop its capture track now, not later
    renderShotLog(); // the "N movements ignored" line needs to move even when nothing gets logged
    return;
  }

  const shotNum = logShot(a);
  // The clip that's been recording since this attempt began now knows which shot it belongs to,
  // and can start counting down its post-release tail (see attachRecordingToShot).
  attachRecordingToShot(shotNum);
}

// ===== SHOT CLIPS — recording. One clip per draw attempt, covering raise through
// follow-through: starts in trackShotAttempt the moment an attempt begins, and is told which
// shot it belongs to here in endAttempt once that's known. Every function below fails safe: if
// anything here throws, the shot log and pose tracking must carry on exactly as if clips didn't
// exist (see CLAUDE.md and the brief this was built from) — so every entry point is wrapped in
// its own try/catch rather than trusting the caller to catch it.

// Picks a MIME type by trying CLIP_MIME_CANDIDATES in order and returning the first one this
// browser claims to support; null means "nothing on the list — let the browser pick its own
// default" rather than refusing to record at all.
function pickMimeType() {
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") return null;
  for (const type of CLIP_MIME_CANDIDATES) {
    try {
      if (MediaRecorder.isTypeSupported(type)) return type;
    } catch {
      // Some browsers throw on a type string they don't recognise rather than returning false —
      // treat that exactly like "not supported" and keep trying the rest of the list.
    }
  }
  return null;
}

// Latches the "clips don't work here" banner on, the first time it's true, and never clears it
// — a later clip succeeding doesn't erase the fact that one failed, and the owner needs to see
// this once they walk over, not have it silently disappear before then.
function markClipsUnavailable(reason) {
  if (clipsUnavailableReason) return;
  clipsUnavailableReason = reason;
  renderShotLog();
}

// Starts recording a new clip from the overlay canvas — called the instant an attempt begins.
// If a previous clip is still running (normally: still in its 2.5s post-shot tail, because a new
// attempt started before that timer fired), it gets cut short and finalised right now rather
// than left to keep running alongside a second recorder.
function startClipRecording() {
  if (selfTestInProgress || !CLIP_SUPPORTED) return; // banner already went up at startup; nothing more to do per attempt
  finalizeRecording(activeRecording);
  // Named clipStream, deliberately not `stream` — that name is already the module-level camera
  // MediaStream (see startCamera), and shadowing it here would be a trap for whoever next reads
  // a `stream.getTracks()...` call elsewhere in the file and assumes it means this one.
  let clipStream = null;
  try {
    const mimeType = pickMimeType();
    const options = { videoBitsPerSecond: CLIP_BITRATE };
    if (mimeType) options.mimeType = mimeType;
    clipStream = canvas.captureStream(CLIP_FRAME_RATE);
    const recorder = new MediaRecorder(clipStream, options);
    const rec = { recorder, clipStream, chunks: [], shotNum: null, finished: false, capTimer: null, tailTimer: null };
    recorder.ondataavailable = (ev) => {
      try {
        if (ev.data && ev.data.size > 0) rec.chunks.push(ev.data);
      } catch (err) {
        console.error("archery-form-coach: clip data handling failed", err);
      }
    };
    recorder.onerror = (ev) => console.error("archery-form-coach: clip recorder error", ev?.error ?? ev);
    // Stopping the recorder does NOT stop canvas.captureStream's tracks — they keep pulling
    // frames off the canvas at CLIP_FRAME_RATE forever unless something stops them explicitly,
    // which without this would mean one live, still-pulling capture track per shot for the rest
    // of the session (found in review: 3 shots -> 3 leaked live tracks, unbounded over an end).
    // Stopped here, in onstop, deliberately AFTER ondataavailable/onstop have already handed the
    // recorder its final data (see finishClipRecording) rather than the instant .stop() is
    // called — so cleaning up the source can never cost the recording its last frame.
    recorder.onstop = () => {
      stopClipStreamTracks(rec.clipStream);
      finishClipRecording(rec);
    };
    recorder.start();
    // Absolute ceiling from the moment recording starts, independent of whether/when the attempt
    // ever ends — this is what stops a stuck full-draw detection from recording forever.
    rec.capTimer = setTimeout(() => finalizeRecording(rec), CLIP_MAX_MS);
    activeRecording = rec;
  } catch (err) {
    stopClipStreamTracks(clipStream); // the capture stream may already exist even though start() (or the recorder itself) failed -- don't leak it
    activeRecording = null;
    markClipsUnavailable("Some shots couldn't be recorded — at least one clip failed this session.");
    console.error("archery-form-coach: clip recording failed to start", err);
  }
}

// Stops every track on a canvas capture stream, so it stops pulling frames off the canvas once
// its recording is done with it. Safe to call more than once on the same stream (or with null/
// undefined) — MediaStreamTrack.stop() on an already-stopped track is a harmless no-op, which
// matters here because more than one code path below can end up calling this for the same rec.
function stopClipStreamTracks(clipStream) {
  try {
    clipStream?.getTracks().forEach((t) => t.stop());
  } catch (err) {
    console.error("archery-form-coach: failed to stop clip capture tracks", err);
  }
}

// Called from endAttempt once a shot has just been logged and we know its number: tells the
// still-running recording which row it belongs to, and starts the post-shot tail timer so the
// release and follow-through get a couple more seconds of recording before it stops. If there is
// no active recording (recording never started, or a stuck-attempt cap already cut it off before
// this attempt ever ended), there's nothing to tell — this shot simply won't have a clip.
function attachRecordingToShot(shotNum) {
  if (!activeRecording || activeRecording.shotNum !== null) return;
  activeRecording.shotNum = shotNum;
  activeRecording.tailTimer = setTimeout(() => finalizeRecording(activeRecording), CLIP_TAIL_MS);
}

// Stops a recording (idempotent — safe to call twice, from both its cap timer and its tail timer
// racing, or from a fresh attempt cutting it short) and clears it from activeRecording. The
// actual blob only becomes available later, asynchronously, in the recorder's onstop handler —
// see finishClipRecording.
function finalizeRecording(rec) {
  if (!rec || rec.finished) return;
  rec.finished = true;
  clearTimeout(rec.capTimer);
  clearTimeout(rec.tailTimer);
  try {
    if (rec.recorder.state !== "inactive") {
      rec.recorder.stop(); // capture tracks get stopped from the onstop handler above, once the data is safely out
    } else {
      // Already inactive without us calling stop() here — onstop won't fire again on our
      // account, so nothing else is going to stop the capture tracks; do it ourselves.
      stopClipStreamTracks(rec.clipStream);
    }
  } catch (err) {
    console.error("archery-form-coach: failed to stop clip recording", err);
    stopClipStreamTracks(rec.clipStream); // stop() itself failed, so onstop may never fire -- don't leak the tracks over that
  }
  if (activeRecording === rec) activeRecording = null;
}

// Runs once a stopped recorder has finished handing over its data. If this clip never got a shot
// number (the stuck-attempt cap fired before the attempt ever ended) there's nothing to attach it
// to, so it's dropped — the rare cost of the CLIP_MAX_MS safety valve.
function finishClipRecording(rec) {
  try {
    if (!rec.chunks.length || rec.shotNum === null) return;
    const blob = new Blob(rec.chunks, { type: rec.recorder.mimeType || "video/webm" });
    if (blob.size === 0) return;
    attachClipToShot(rec.shotNum, blob);
  } catch (err) {
    console.error("archery-form-coach: failed to finalise clip", err);
  }
}

// Attaches a finished clip to its shot's row in the log, by shot number — reuniting the two,
// since the clip finishes recording well after logShot already ran. If that shot has since been
// bumped off the end of the log (SHOT_LOG_MAX newer attempts happened first), there is no row
// left to attach it to: the blob is simply dropped, and since no object URL was ever created for
// it, there's nothing to revoke either.
function attachClipToShot(shotNum, blob) {
  const entry = log.find((e) => e.shotNum === shotNum);
  if (!entry) return;
  entry.clipBlob = blob;
  entry.clipUrl = URL.createObjectURL(blob);
  entry.clipMimeType = blob.type;
  renderShotLog();
}

// Releases a row's clip — its object URL and the blob itself — called when that row is about to
// fall out of the log for good (see logShot below). Clips are memory-only, exactly like the log
// they ride along with: nothing here is ever written to disk, so this is the only cleanup needed.
function revokeClip(entry) {
  if (entry.clipUrl) URL.revokeObjectURL(entry.clipUrl);
}
// ===========================================================================

function logShot(entry) {
  shotCount++;
  const shotNum = shotCount;
  log.unshift({ ...entry, shotNum });
  const evicted = log.slice(SHOT_LOG_MAX);
  log = log.slice(0, SHOT_LOG_MAX);
  evicted.forEach(revokeClip);
  renderShotLog();
  return shotNum;
}

// One measure's stats across the shot log: the owner's own average, their spread (best-to-worst
// gap — the number that matters most, since every target range in this file is a desk guess but
// "did I repeat myself" needs no calibration at all, see CLAUDE.md), and each shot's deviation
// from that average. `getValue(entry)` pulls the one number this measure cares about (or
// null/undefined if that shot's landmark was below MIN_VISIBILITY) — those get excluded from
// the average, not silently treated as zero. Pure: no DOM, no MediaPipe, just arrays — so
// selfTest can check it directly.
function summarizeMeasure(entries, getValue) {
  const total = entries.length;
  const points = entries
    .map((e) => ({ shotNum: e.shotNum, value: getValue(e) }))
    .filter((p) => p.value != null);
  const n = points.length;

  // Fewer than two readings: there's nothing to spread across and no baseline worth deviating
  // from (a single point's "average" just equals itself). Report the plain average when there
  // is exactly one reading — it's still true — but no spread and no deviations, rather than
  // faking a "+0" or dividing by zero.
  if (n < 2) {
    return { n, total, average: n === 1 ? points[0].value : null, spread: null, deviations: {}, outlierShotNum: null };
  }

  const values = points.map((p) => p.value);
  const average = values.reduce((sum, v) => sum + v, 0) / n;
  const spread = Math.max(...values) - Math.min(...values);

  const deviations = {};
  let outlierShotNum = null;
  let worstAbsDev = -1;
  for (const p of points) {
    const dev = p.value - average;
    deviations[p.shotNum] = dev;
    if (Math.abs(dev) > worstAbsDev) {
      worstAbsDev = Math.abs(dev);
      outlierShotNum = p.shotNum;
    }
  }
  return { n, total, average, spread, deviations, outlierShotNum };
}

// The shot log's stats for all four numbers it tracks, in one call. Pure, same reason as above.
function summarizeShots(entries) {
  return {
    bowArm: summarizeMeasure(entries, (e) => e.bowArmAngle),
    shoulderBow: summarizeMeasure(entries, (e) => e.shoulderDrop?.bow ?? null),
    shoulderDraw: summarizeMeasure(entries, (e) => e.shoulderDrop?.draw ?? null),
    elbow: summarizeMeasure(entries, (e) => e.elbowAlign?.signed ?? null),
  };
}

// ===== SHOT LOG — plain-language consistency wording. The app has no idea what good archery
// form looks like: every target range in the CALIBRATE WITH COACH block at the top of this file
// is an explicitly labelled placeholder the owner hasn't tuned with a coach yet. So the log must
// never say anything that implies a judgement it can't back up — no "your bow arm is too bent",
// no scores, no pass/fail against an invented ideal. The one thing it CAN say honestly, needing
// no calibration at all, is whether the owner did the same thing twice — repeatability, measured
// only against his own other shots this session. See narrateMeasure below for exactly how.
//
// These two numbers are not target ranges like CALIBRATE WITH COACH above — there's no "correct"
// value here, only "how big a gap counts as worth mentioning", and that's answered relative to
// the session's OWN scatter, not a fixed number of degrees or percent. Tune by eye once there's
// real session wording to read against real shots.
const OUTLIER_SCATTER_FACTOR = 2.5; // a shot's own gap from the average must be more than this many times the OTHER shots' typical gap from average before it gets called out by name
const DRIFT_GAP_FACTOR = 1.5; // the gap between the first-half and second-half averages must be more than this many times the whole session's typical gap from average before it counts as drift rather than ordinary shot-to-shot wobble
// ===========================================================================

// Turns one measure's own numbers, across the shots in the log, into ONE plain-English headline
// sentence — or null if there's nothing honest to say. Priority order, checked in this order:
//   1. One shot sits clearly apart from the rest of them — name it, and say which way.
//   2. No single shot stands out, but there's a real trend from the early shots to the late ones
//      (compares the first half of the session against the second half) — most likely the archer
//      tiring or settling in over the end.
//   3. Neither of the above: every shot landed close to the others — "steady".
// Below three readings there's no honest consistency story to tell at all (a single point can't
// have a "spread", and two points are just two points), so this says that plainly instead of
// computing a spread from almost nothing. Below five, whatever it does say gets a short "early
// days" qualifier — three or four shots is still a small sample to be confident about.
//
// `wordFor(sign)` supplies the one word (or short phrase) that describes what a HIGHER reading
// of this particular measure means in plain terms — e.g. for bow-arm angle, higher = straighter;
// for shoulder drop, higher = sat lower (see shoulderDropOf's own comment: bigger % = further
// from the ear = more dropped). sign is +1 for a reading above the session average, -1 for below.
//
// Pure: entries (any order) + a value-getter + a label + a word function in, one result out —
// { text, outlierShotNum, outlierWord } (outlierShotNum/outlierWord are null unless case 1 above
// fired) or null. No DOM, no module state, so selfTest can drive it with plain fixture arrays.
function narrateMeasure(entries, getValue, label, wordFor) {
  const points = entries
    .map((e) => ({ shotNum: e.shotNum, value: getValue(e) }))
    .filter((p) => p.value != null)
    .sort((a, b) => a.shotNum - b.shotNum); // oldest first — shotNum order IS chronological order, needed for the drift check below
  const n = points.length;

  if (n === 0) return null; // every reading for this measure was uncertain this session — nothing honest to say at all
  if (n === 1) return { text: `${label} — only one shot logged, too early to say anything about consistency.`, outlierShotNum: null, outlierWord: null };
  if (n === 2) return { text: `${label} — only two shots logged, too early to call it steady or drifting.`, outlierShotNum: null, outlierWord: null };

  const average = points.reduce((sum, p) => sum + p.value, 0) / n;
  const deviations = points.map((p) => ({ shotNum: p.shotNum, dev: p.value - average }));
  const qualifier = n <= 4 ? ` Only ${n} shots so far — early days.` : "";

  // --- 1. Does one shot sit clearly apart from the REST of them? "Clearly apart" is measured
  // against how much the OTHER shots typically scatter, not a fixed number of degrees or
  // percent — so a tight, repeatable session and a naturally noisy one each get judged by their
  // own standard, instead of one arbitrary cutoff applied to every session forever.
  let worst = deviations[0];
  for (const d of deviations) if (Math.abs(d.dev) > Math.abs(worst.dev)) worst = d;
  const others = deviations.filter((d) => d.shotNum !== worst.shotNum);
  const othersScatter = others.reduce((sum, d) => sum + Math.abs(d.dev), 0) / others.length;
  const worstDev = Math.abs(worst.dev);
  // A real gap: either the other shots were essentially identical to each other and this one
  // wasn't (othersScatter ~0 but worstDev isn't), or this shot sits meaningfully farther out
  // than the others typically scatter (more than OUTLIER_SCATTER_FACTOR times their own average).
  const isOutlier = worstDev > 1e-6 && (othersScatter < 1e-6 || worstDev > OUTLIER_SCATTER_FACTOR * othersScatter);
  if (isOutlier) {
    return {
      text: `${label} — shot ${worst.shotNum} stands out: ${wordFor(Math.sign(worst.dev))} than your other ${others.length}.${qualifier}`,
      outlierShotNum: worst.shotNum,
      outlierWord: wordFor(Math.sign(worst.dev)),
    };
  }

  // --- 2. No single shot stands out — is there a trend from the early shots to the late ones?
  // Split chronologically in half (on an odd count, the middle shot sits out of both halves
  // rather than being forced into either) and compare the two halves' own averages against how
  // much this measure naturally varies overall, so a small, evenly-spread wobble doesn't read
  // as "drift" just because the last shot happened to land a hair higher than the first.
  const half = Math.floor(n / 2);
  const firstHalf = points.slice(0, half);
  const secondHalf = points.slice(n - half);
  const avgFirst = firstHalf.reduce((sum, p) => sum + p.value, 0) / half;
  const avgSecond = secondHalf.reduce((sum, p) => sum + p.value, 0) / half;
  const gap = avgSecond - avgFirst;
  const overallScatter = deviations.reduce((sum, d) => sum + Math.abs(d.dev), 0) / n;
  const isDrift = Math.abs(gap) > 1e-6 && (overallScatter < 1e-6 || Math.abs(gap) > DRIFT_GAP_FACTOR * overallScatter);
  if (isDrift) {
    const sign = Math.sign(gap); // +1 means the second half's own average sits above the first half's
    return {
      text: `${label} — drifted: ${wordFor(-sign)} on your first ${half}, ${wordFor(sign)} by the last ${half}.${qualifier}`,
      outlierShotNum: null,
      outlierWord: null,
    };
  }

  // --- 3. Neither: every shot landed close to the others.
  return { text: `${label} — steady. Every shot within a hair of the others.${qualifier}`, outlierShotNum: null, outlierWord: null };
}

// One word (or short phrase) per measure, describing what a HIGHER reading of that measure means
// in plain terms — shared between the outlier and drift sentences in narrateMeasure above, and
// between the headline and the per-row highlight note in renderShotRow below, so the wording is
// always consistent wherever a measure gets described. sign is +1 (above this session's own
// average) or -1 (below it) — see narrateMeasure's own comment for the full explanation.
const wordForBowArm = (sign) => (sign > 0 ? "straighter" : "more bent"); // bigger angle = straighter, see bowArmAngleOf
const wordForShoulder = (sign) => (sign > 0 ? "sat lower" : "sat higher"); // bigger % = further from the ear = more dropped, see shoulderDropOf
const wordForElbow = (sign) => (sign > 0 ? "higher" : "lower"); // positive signed deviation = high, see drawElbowAlignmentOf

// One shot's number rendered as "raw value (signed deviation)" — bare, no repeated "vs your
// average" text (that's said ONCE, under the summary block — see renderShotLog; saying it once
// per row on four different numbers was an unreadable wall of text on a phone screen). Coloured
// using the SAME ok/warn classes and --green/--amber variables the live readouts already use
// (see style.css). Falls back to the plain number, uncoloured, when there's no baseline yet
// (fewer than two readings) or this shot's own reading was uncertain (shown as "not measured",
// not a bare dash that reads like a typo mid-sentence).
//
// Returns { html, flagged } rather than just HTML: `flagged` says whether THIS shot is both the
// session's worst reading for this measure AND actually outside the display cutoff — the second
// half matters because summarizeMeasure always names an outlierShotNum once there are 2+
// readings ("someone is always the maximum"), even when every shot in a tight cluster is
// basically identical. Without the cutoff check, a shot 0.5 percentage points from its own
// average could get marked "most different" right next to text saying it's right on average —
// a real bug this replaced. renderShotRow uses `flagged` to show at most one outlier mark for
// the whole row, not one per measure.
function shotValueHtml(rawText, value, unit, shotNum, stats, maxDeviation) {
  if (value == null) return { html: `<span class="value uncertain">not measured</span>`, flagged: false };
  if (!(shotNum in stats.deviations)) return { html: `<span class="value">${rawText}</span>`, flagged: false };
  const dev = stats.deviations[shotNum];
  const rounded = Math.round(dev);
  const devText = rounded > 0 ? `+${rounded}` : `${rounded}`;
  const ok = Math.abs(dev) <= maxDeviation;
  const flagged = !ok && stats.outlierShotNum === shotNum;
  return { html: `<span class="value ${ok ? "ok" : "warn"}">${rawText} (${devText})</span>`, flagged };
}

// `outliers`/`words`: which shot number (if any) narrateMeasure named as standing out for each
// measure this session, and the word it used — computed once in renderShotLog and passed in here
// so all ten rows agree with the headline above them instead of each row re-deciding for itself.
function renderShotRow(e, stats, outliers, words) {
  const armText = e.bowArmAngle == null ? "—" : `${Math.round(e.bowArmAngle)}°`;
  const arm = shotValueHtml(armText, e.bowArmAngle, "°", e.shotNum, stats.bowArm, BOW_ARM_CONSISTENCY_MAX_DEVIATION);

  const bowDrop = e.shoulderDrop?.bow ?? null;
  const bowText = bowDrop == null ? "—" : `${Math.round(bowDrop)}%`;
  const bow = shotValueHtml(bowText, bowDrop, "%", e.shotNum, stats.shoulderBow, SHOULDER_DROP_CONSISTENCY_MAX_DEVIATION);

  const drawDrop = e.shoulderDrop?.draw ?? null;
  const drawText = drawDrop == null ? "—" : `${Math.round(drawDrop)}%`;
  const draw = shotValueHtml(drawText, drawDrop, "%", e.shotNum, stats.shoulderDraw, SHOULDER_DROP_CONSISTENCY_MAX_DEVIATION);

  const elbowText = !e.elbowAlign
    ? "—"
    : Math.round(e.elbowAlign.deviation) === 0
      ? "in line"
      : `${Math.round(e.elbowAlign.deviation)}° ${e.elbowAlign.direction}`;
  const elbow = shotValueHtml(elbowText, e.elbowAlign?.signed ?? null, "°", e.shotNum, stats.elbow, ELBOW_ALIGN_CONSISTENCY_MAX_DEVIATION);

  const debugBit = DEBUG
    ? `<span class="shotlog-debug">hand sep ${e.handSep.toFixed(2)} — anchor ${e.anchorOk ? "ok" : "fail"} · arm-check ${e.armOk ? "ok" : "fail"} · sep-check ${e.sepOk ? "ok" : "fail"} · still ${e.stillOk ? "ok" : "fail"}</span>`
    : "";
  // Every reading, in the same units as the live readouts, kept for whoever eventually tunes the
  // CALIBRATE WITH COACH constants against a real session — but demoted to small print, not the
  // headline: see the block comment above narrateMeasure for why the app can't put a judgement on
  // these numbers itself.
  const rawHtml = `<div class="shotlog-row-raw">bow arm ${arm.html} · bow shoulder ${bow.html} · draw shoulder ${draw.html} · elbow ${elbow.html}${debugBit}</div>`;

  // What (if anything) THIS shot was named for — reusing the exact same outlier picked out by
  // narrateMeasure for the headline above, so a row is only ever singled out here when the
  // headline already said so somewhere. A shot can be named for more than one measure at once;
  // list every one rather than picking just the first.
  const notes = [];
  if (outliers.bowArm === e.shotNum) notes.push(`bow arm ${words.bowArm} than your others`);
  if (outliers.shoulderBow === e.shotNum) notes.push(`bow shoulder ${words.shoulderBow} than your others`);
  if (outliers.shoulderDraw === e.shotNum) notes.push(`draw shoulder ${words.shoulderDraw} than your others`);
  if (outliers.elbow === e.shotNum) notes.push(`draw elbow ${words.elbow} than your others`);
  const highlightText = notes.length ? notes.join("; ") : "nothing stood out";

  // A logged attempt that never reached full draw is still worth keeping (see CLAUDE.md/README —
  // that's on purpose) but must read differently from one that did, at a glance, since it's the
  // only thing separating "he drew short of full draw" from "he drew all the way".
  const shortMark = e.reachedFullDraw === false ? ` <span class="shotlog-shortdraw">· short of full draw</span>` : "";

  // A big, obvious watch button when this shot has a clip; a plain "no clip" note when it
  // doesn't (recording unsupported, or it failed for just this one shot) — never nothing, so a
  // missing clip never reads as a missing shot. data-shot carries the shot number for the click
  // handler on shotLogEl (see openClipPlayer wiring) to look the entry back up by.
  const clipBit = e.clipUrl
    ? `<button type="button" class="shotlog-play" data-shot="${e.shotNum}">▶ Watch</button>`
    : `<span class="shotlog-noclip">no clip</span>`;

  return `<div class="shotlog-row"><div class="shotlog-row-main">Shot ${e.shotNum} — ${highlightText}${shortMark}${rawHtml}</div><div class="shotlog-row-clip">${clipBit}</div></div>`;
}

// Plain-language shot log — this is what the owner actually reads, standing at the phone after
// their end, not mid-shot, so it has to answer, in order: how many arrows did it see; was he
// consistent, and in what; which shot was the odd one out, and how (so he knows which clip to
// go watch). Nothing here judges his form against a target — see the block comment above
// narrateMeasure for why not. Raw degrees/percent are still there for whoever eventually tunes
// the CALIBRATE WITH COACH constants, just demoted to small print on each row, not the headline.
function renderShotLog() {
  // A clip-recording failure has to still be visible whenever the owner walks over and looks —
  // not just at the moment it happened — so this goes at the very top, above everything else,
  // every single render, for as long as clipsUnavailableReason is set (which is forever, once
  // it's set at all — see markClipsUnavailable).
  const banner = clipsUnavailableReason ? `<div class="shotlog-banner">${clipsUnavailableReason}</div>` : "";
  // Which pose model ended up running, and how fast — set once, shortly after startup (see
  // measurePoseModelPerf/setModelStatusLine), and shown on every render after that. Sits right
  // under the clip banner, same reasoning: the owner can't be watching when this gets decided.
  const modelBit = modelStatusLine ? `<div class="shotlog-modelinfo">${modelStatusLine}</div>` : "";
  // How many draw attempts got thrown out as noise this session (see endAttempt) — shown
  // whenever there's at least one, right alongside the two lines above, for the same reason:
  // this is diagnostic information the owner can only ever read after the fact. A big number
  // here next to a small arrow count is itself the useful signal — see CLAUDE.md.
  const rejectedBit =
    rejectedAttemptCount > 0
      ? `<div class="shotlog-rejected">${rejectedAttemptCount} movement${rejectedAttemptCount === 1 ? "" : "s"} ignored (too short, or never near full draw) — not counted as shots.</div>`
      : "";

  if (log.length === 0) {
    shotLogEl.innerHTML = `${banner}${modelBit}${rejectedBit}<div class="shotlog-empty">No shots recorded yet — draw once and this fills in.</div>`;
    return;
  }

  const arrowWord = shotCount === 1 ? "arrow" : "arrows";
  const shownNote = log.length < shotCount ? ` The consistency lines below are based on your most recent ${log.length}.` : "";
  const countLine = `<div class="shotlog-count">${shotCount} ${arrowWord} this session.${shownNote}</div>`;

  const bowArm = narrateMeasure(log, (e) => e.bowArmAngle, "Bow arm", wordForBowArm);
  const shoulderBow = narrateMeasure(log, (e) => e.shoulderDrop?.bow ?? null, "Bow shoulder", wordForShoulder);
  const shoulderDraw = narrateMeasure(log, (e) => e.shoulderDrop?.draw ?? null, "Draw shoulder", wordForShoulder);
  const elbow = narrateMeasure(log, (e) => e.elbowAlign?.signed ?? null, "Draw elbow", wordForElbow);

  const narrativeHtml = [bowArm, shoulderBow, shoulderDraw, elbow]
    .filter((r) => r) // a measure with every reading uncertain this session says nothing at all — see narrateMeasure
    .map((r) => `<div class="shotlog-narrative-row">${r.text}</div>`)
    .join("");

  // Which shot (if any) each measure named as standing out, and the word it used — shared with
  // every row below so each one can point back at exactly what the headline just said about it.
  const outliers = {
    bowArm: bowArm?.outlierShotNum ?? null,
    shoulderBow: shoulderBow?.outlierShotNum ?? null,
    shoulderDraw: shoulderDraw?.outlierShotNum ?? null,
    elbow: elbow?.outlierShotNum ?? null,
  };
  const words = {
    bowArm: bowArm?.outlierWord ?? null,
    shoulderBow: shoulderBow?.outlierWord ?? null,
    shoulderDraw: shoulderDraw?.outlierWord ?? null,
    elbow: elbow?.outlierWord ?? null,
  };

  const stats = summarizeShots(log); // still drives the demoted small-print numbers on each row, unchanged
  const rowsHtml = log.map((e) => renderShotRow(e, stats, outliers, words)).join("");

  shotLogEl.innerHTML = `${banner}${modelBit}${rejectedBit}${countLine}<div class="shotlog-narrative">${narrativeHtml}</div>${rowsHtml}`;
}

// Draws the current camera frame into the overlay canvas. Used to be just ctx.clearRect, leaving
// the canvas transparent so the <video> element underneath showed through on its own — visually
// identical, but it meant the canvas itself had no picture in it, only skeleton lines. Now that
// the canvas is what gets recorded (see startClipRecording — canvas.captureStream is the only
// way to bake the skeleton into a clip), the canvas needs its own copy of the video frame every
// time, landmarks or not, so a clip is never missing frames just because the pose was briefly
// lost. canvas.width/height are set to the video's native resolution in startCamera, so this
// plain draw lines up exactly with no cropping or letterboxing needed.
//
// Mirroring note: the front camera mirrors on-screen via a CSS transform on both #video and
// #overlay (see style.css), which only affects how the browser displays the elements — it does
// not touch the pixels drawn here. So a front-camera clip plays back unmirrored. That's fine
// (the owner shoots on the rear camera) and is NOT a bug to "fix" by mirroring the canvas draw —
// doing that would put every landmark coordinate on the wrong side of the frame.
function drawVideoFrame() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
}

function drawSkeleton(landmarks) {
  drawVideoFrame();
  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: "#00e5ff",
    lineWidth: 3,
  });
  drawingUtils.drawLandmarks(landmarks, { color: "#ffffff", radius: 4 });
}

// ?debug-only readout of why the full-draw trigger is (or isn't) firing. No-op — not even
// a DOM lookup beyond the one at startup — when ?debug isn't in the URL. Deliberately big and
// blunt: this has to be readable from ~5 metres away while the owner is mid-shot, not tidy.
// Hand separation gets top billing because it's the number most likely to need retuning. This
// is a LIVE readout only — for anything that has to survive past the instant it happens (which
// is everything the owner actually needs, per CLAUDE.md), see the shot log instead.
function syncDebugOverlay() {
  if (!DEBUG) return;
  const otherChecks = (s) => `anchor ${s.anchorOk ? "ok" : "fail"} · arm ${s.armOk ? "ok" : "fail"} · still ${s.stillOk ? "ok" : "fail"}`;

  const liveHtml = !debugInfo
    ? `<div class="debug-big debug-fail">hand sep: no pose seen</div>`
    : `<div class="debug-big ${debugInfo.sepOk ? "debug-ok" : "debug-fail"}">hand sep ${debugInfo.handSep.toFixed(2)} of ${FULL_DRAW_HAND_SEP_MIN} needed — ${debugInfo.sepOk ? "far enough apart" : "too close together"}</div>`;

  const checksHtml = debugInfo ? `<div class="debug-small">${otherChecks(debugInfo)}</div>` : "";

  debugEl.innerHTML = liveHtml + checksHtml;
}

function renderLoop() {
  requestAnimationFrame(renderLoop);
  const now = performance.now();

  if (!poseLandmarker || video.readyState < 2) return;

  const inferenceStart = performance.now();
  const result = poseLandmarker.detectForVideo(video, now);
  measurePoseModelPerf(performance.now() - inferenceStart);
  const rawLandmarks = result.landmarks?.[0];

  if (!rawLandmarks) {
    drawVideoFrame(); // keep the clip (and the on-screen view) showing the camera even without a skeleton
    setReadout(readoutBowArm, valueBowArm, "— uncertain", "uncertain");
    setValueState(valueShoulderBow, "—", "uncertain");
    setValueState(valueShoulderDraw, "—", "uncertain");
    setReadout(readoutElbow, valueElbow, "— uncertain", "uncertain");
    if (DEBUG) debugInfo = null;
    // Tracking just lost the archer entirely — whatever the filters were smoothing toward is now
    // stale. Reset so a fresh detection later starts clean rather than being dragged from
    // wherever the skeleton was last seen (see LandmarkSmoother).
    landmarkSmoother.reset();
    endAttempt(now); // pose lost mid-attempt counts as the attempt ending, same as hands relaxing
  } else {
    // Smoothed landmarks feed everything downstream — the skeleton drawing, all three readouts,
    // and the full-draw/shot-log sampling inside isAtFullDraw — so a shaky raw detection can't
    // show up in the numbers the owner reads later or the clip he watches back. Real elapsed time
    // (performance.now(), converted to seconds), not an assumed frame rate — see OneEuroFilter.
    const landmarks = landmarkSmoother.smooth(rawLandmarks, now / 1000);
    drawSkeleton(landmarks);
    updateBowArmReadout(landmarks);
    updateShoulderDropReadout(landmarks);
    updateDrawElbowReadout(landmarks);
    // Return value intentionally unused here — isAtFullDraw's real job on every frame is its
    // side effect, calling trackShotAttempt (below) to feed the shot log. It used to also drive
    // the auto-freeze state machine, which read the true/false result; that machine is gone, but
    // the shot log still depends on this call happening every frame, so it stays.
    isAtFullDraw(landmarks, now);
  }

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

// The one interaction the owner needs after they're done shooting: tap once to see everything
// that got recorded while they couldn't look. Tap again to put it away. Content is kept fresh
// as shots come in (see logShot), so there's nothing to render here beyond the toggle itself.
btnLog.addEventListener("click", () => {
  shotLogEl.classList.toggle("hidden");
});

// The clip player: a full-screen takeover opened by tapping "Watch" on a shot log row. One
// listener on the log's container (event delegation) rather than one per row, since rows are
// replaced wholesale on every renderShotLog — a per-row listener would need re-attaching every
// time and would leak the old ones.
shotLogEl.addEventListener("click", (ev) => {
  const btn = ev.target.closest(".shotlog-play");
  if (!btn) return;
  const shotNum = Number(btn.dataset.shot);
  const entry = log.find((e) => e.shotNum === shotNum);
  if (entry && entry.clipUrl) openClipPlayer(entry.clipUrl);
});

function openClipPlayer(url) {
  clipPlayerVideo.src = url;
  clipPlayerVideo.playbackRate = 1;
  clipPlayerRateBtns.forEach((b) => b.classList.toggle("active", b.dataset.rate === "1"));
  clipPlayerEl.classList.remove("hidden");
  clipPlayerVideo.play().catch(() => {}); // autoplay can be blocked; native controls let them press play themselves either way
}

// Closing returns to the live view — the camera and pose tracking underneath never stopped
// running while the player was open, so there's nothing to resume, just to uncover again.
function closeClipPlayer() {
  clipPlayerVideo.pause();
  clipPlayerEl.classList.add("hidden");
  clipPlayerVideo.removeAttribute("src");
  clipPlayerVideo.load();
}

clipPlayerClose.addEventListener("click", closeClipPlayer);
clipPlayerRateBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    clipPlayerVideo.playbackRate = Number(btn.dataset.rate);
    clipPlayerRateBtns.forEach((b) => b.classList.toggle("active", b === btn));
  });
});

if (!CLIP_SUPPORTED) {
  markClipsUnavailable("Clips unavailable in this browser — everything else still works, just no shot videos.");
}
renderShotLog(); // shows the "no shots yet" placeholder (and the banner above, if set) before the first shot comes in

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

// Plain-assert checks for the full-draw detection and shot log. No framework, no fixtures.
// Open the page as ...index.html?selftest and read the console.
function selfTest() {
  // isAtFullDraw: the raise (bow arm straight, both hands up near the face together) must
  // NOT read as full draw — that was the field bug. Only real full draw (hands apart, held
  // still) should. Saves/restores the module state isAtFullDraw depends on so this doesn't
  // disturb the real app.
  const savedHanded = rightHanded;
  const savedLastWrist = lastDrawWrist;
  const savedAttempt = attempt;
  const savedLog = log;
  const savedShotCount = shotCount;
  const savedRejectedAttemptCount = rejectedAttemptCount;
  selfTestInProgress = true; // see the flag's own comment — keeps trackShotAttempt below from spinning up real MediaRecorders
  rightHanded = true;
  const mkLandmarks = (overrides) => {
    const lm = Array.from({ length: 25 }, () => ({ x: 0, y: 0, visibility: 1 }));
    for (const i in overrides) lm[i] = { ...lm[i], ...overrides[i] };
    return lm;
  };
  // Shared skeleton scale: shoulder-to-hip torso length of 0.3.
  const base = {
    9: { x: 0.5, y: 0.3 }, // mouth L
    10: { x: 0.5, y: 0.3 }, // mouth R
    11: { x: 0.3, y: 0.3 }, // bow (left) shoulder
    12: { x: 0.5, y: 0.3 }, // draw (right) shoulder
    13: { x: 0.15, y: 0.3 }, // bow elbow — collinear with shoulder(0.3) and bow wrist(0.0) used by the "drawn"/"drifted"/mid-draw fixtures below: 180°, a genuinely straight arm
    23: { x: 0.3, y: 0.6 }, // bow hip
    24: { x: 0.5, y: 0.6 }, // draw hip
  };

  // --- Raise: bow arm straight, hands together up near the face. Must be rejected, and
  // rejected BY HAND SEPARATION SPECIFICALLY — not as a side effect of some other condition
  // also (accidentally) failing, which is exactly what let the original bug through: that
  // fixture's elbow was on the wrong side of the shoulder/wrist line, measuring 0° (folded)
  // instead of 180° (straight), so it was "rejected" by armOk failing instead, for real full
  // draw's actual bow arm angle, not the raise's.
  lastDrawWrist = null;
  const raise = mkLandmarks({
    ...base,
    13: { x: 0.41, y: 0.3 }, // bow elbow, positioned so shoulder(0.30)-elbow(0.41)-wrist(0.52) are collinear
    15: { x: 0.52, y: 0.3 }, // bow wrist
    16: { x: 0.52, y: 0.31 }, // draw wrist, right next to the bow wrist — hands together
  });
  const raiseArmAngle = angleAt(raise[L_SHOULDER], raise[L_ELBOW], raise[L_WRIST]);
  console.assert(
    Math.abs(raiseArmAngle - 180) < 0.01,
    "raise fixture's bow arm must actually measure as ~180° (straight), or this isn't testing the raise at all"
  );
  const raiseScale = torsoLength(raise, R_SHOULDER, R_HIP);
  const raiseAnchor = {
    x: (raise[MOUTH_L].x + raise[MOUTH_R].x) / 2,
    y: (raise[MOUTH_L].y + raise[MOUTH_R].y) / 2,
  };
  const raiseAnchorDist =
    Math.hypot(raise[R_WRIST].x - raiseAnchor.x, raise[R_WRIST].y - raiseAnchor.y) / raiseScale;
  const raiseHandSep =
    Math.hypot(raise[R_WRIST].x - raise[L_WRIST].x, raise[R_WRIST].y - raise[L_WRIST].y) / raiseScale;
  console.assert(
    raiseAnchorDist <= FULL_DRAW_ANCHOR_MAX,
    "raise fixture should pass the anchor-proximity check on its own"
  );
  console.assert(
    raiseHandSep < FULL_DRAW_HAND_SEP_MIN,
    "raise fixture's hands should be too close together to pass hand separation — the one thing meant to reject it"
  );
  console.assert(isAtFullDraw(raise, 0) === false, "raise (first frame) must not read as full draw");
  console.assert(
    isAtFullDraw(raise, 500) === false,
    "raise held steady for a second frame (passes stillness too, now) must still be rejected — by hand separation alone"
  );

  // --- Full draw, held: hands apart, near anchor, bow arm straight. Rejected on the first
  // frame only because there's no prior position yet to judge stillness from; reads true once
  // it's held for a frame.
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

  // --- Drawing in progress: hands already apart and already near anchor (so anchor, arm, and
  // separation would all pass) but the draw wrist is still travelling fast between frames —
  // must be rejected by stillness alone, not because it never got close enough.
  lastDrawWrist = null;
  const midDraw1 = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.4, y: 0.31 } });
  isAtFullDraw(midDraw1, 0); // seeds lastDrawWrist; this call's own result isn't the point
  const midDraw2 = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.5, y: 0.32 } });
  const midScale = torsoLength(midDraw2, R_SHOULDER, R_HIP);
  const midAnchor = {
    x: (midDraw2[MOUTH_L].x + midDraw2[MOUTH_R].x) / 2,
    y: (midDraw2[MOUTH_L].y + midDraw2[MOUTH_R].y) / 2,
  };
  const midAnchorDist =
    Math.hypot(midDraw2[R_WRIST].x - midAnchor.x, midDraw2[R_WRIST].y - midAnchor.y) / midScale;
  const midHandSep =
    Math.hypot(midDraw2[R_WRIST].x - midDraw2[L_WRIST].x, midDraw2[R_WRIST].y - midDraw2[L_WRIST].y) / midScale;
  console.assert(midAnchorDist <= FULL_DRAW_ANCHOR_MAX, "mid-draw fixture should pass the anchor check on its own");
  console.assert(
    midHandSep >= FULL_DRAW_HAND_SEP_MIN,
    "mid-draw fixture's hands should already be far enough apart on their own"
  );
  console.assert(
    isAtFullDraw(midDraw2, 50) === false,
    "wrist still travelling fast toward anchor (50ms, big jump) must not read as full draw yet — only stillness should be stopping it"
  );

  // --- Fast jump while otherwise at full draw: same idea as mid-draw above, kept as its own
  // check because it's the scenario closest to what a real archer's hand does in the instant
  // right before it settles at anchor.
  lastDrawWrist = null;
  const driftSeed = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.52, y: 0.31 } });
  isAtFullDraw(driftSeed, 500); // seeds lastDrawWrist at the same position/time as `drawn` above
  const drifted = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.6, y: 0.31 } });
  console.assert(
    isAtFullDraw(drifted, 600) === false,
    "wrist jumping far in 100ms (fast) should not read as holding still"
  );

  // --- Shot log attempt-boundary rule: an attempt is "in progress" for as long as hand
  // separation stays at/above DRAW_ATTEMPT_MIN_SEP, tracking its best (highest-separation)
  // frame; it ends when separation drops back below that floor OR the pose is lost — but only
  // gets LOGGED if it also cleared the SHOT_MIN_PEAK_SEP_FRACTION and SHOT_MIN_DURATION_MS gates
  // in endAttempt (added after the field report: without these, nocking an arrow or lowering
  // the bow was logging as a phantom shot). Reset to a clean slate here — the outer save/restore
  // above puts it all back afterwards regardless.
  attempt = null;
  log = [];
  shotCount = 0;
  rejectedAttemptCount = 0;
  const sample = (handSep, atFullDraw = false, extra = {}) => ({
    handSep,
    bowArmAngle: 178,
    atFullDraw,
    anchorOk: true,
    armOk: true,
    sepOk: handSep >= FULL_DRAW_HAND_SEP_MIN,
    stillOk: true,
    ...extra,
  });
  // SHOT_MIN_PEAK_SEP_FRACTION * FULL_DRAW_HAND_SEP_MIN = 0.8 * 0.75 = 0.6 — the peak-separation
  // gate every fixture below is deliberately placed clearly above or clearly below.

  trackShotAttempt(sample(0.1), 0); // resting, below the floor: no attempt yet
  console.assert(log.length === 0 && attempt === null, "resting below the floor should not start an attempt");

  // --- A short, shallow wiggle — nocking an arrow, lowering the bow, hands drifting apart for a
  // moment — must NOT be logged: this is the exact field bug. It crosses DRAW_ATTEMPT_MIN_SEP
  // (so it IS an "attempt" briefly), but its peak (0.45) never gets near the 0.6 gate, and it's
  // over in 250ms, well under SHOT_MIN_DURATION_MS (600ms).
  trackShotAttempt(sample(0.35), 1000); // crosses the floor
  trackShotAttempt(sample(0.45), 1100); // peak — shallow
  trackShotAttempt(sample(0.1), 1250); // hands back together 250ms after it started — ends the attempt
  console.assert(log.length === 0, "a brief, shallow wiggle must not be logged as a shot");
  console.assert(shotCount === 0, "shotCount must not advance for a rejected attempt");
  console.assert(rejectedAttemptCount === 1, "the rejected wiggle should be counted");

  // --- Long enough AND deep enough, but short of literal full draw: still logged, and marked as
  // such — the deliberate behaviour CLAUDE.md/README call out on purpose (a draw that fell short
  // of full draw is still worth seeing; only noise should be thrown away).
  trackShotAttempt(sample(0.4), 2000); // crosses the floor
  trackShotAttempt(sample(0.65), 2400); // peak: 0.65 clears the 0.6 gate, but never reaches FULL_DRAW_HAND_SEP_MIN (0.75)
  trackShotAttempt(sample(0.6), 2800); // held near peak
  trackShotAttempt(sample(0.05), 3200); // ends — 1200ms after it started, comfortably over SHOT_MIN_DURATION_MS
  console.assert(log.length === 1, "a long, near-full-draw attempt should be logged");
  console.assert(
    log[0].shotNum === 1,
    "the first LOGGED attempt should be shot 1 — the rejected wiggle above must not have consumed a shot number"
  );
  console.assert(log[0].handSep === 0.65, "the logged row should carry the attempt's peak hand separation");
  console.assert(log[0].reachedFullDraw === false, "an attempt that never reached full draw must be marked as such");

  // --- An attempt that DOES reach full draw (atFullDraw: true on its peak frame): logged, and
  // NOT marked as short of full draw.
  trackShotAttempt(sample(0.4), 4000);
  trackShotAttempt(sample(0.8, true), 4400); // peak, and a genuine full draw (all four isAtFullDraw gates true)
  trackShotAttempt(sample(0.78, true), 4800); // held at full draw
  trackShotAttempt(sample(0.05), 5200); // ends
  console.assert(log.length === 2, "a full-draw attempt should log as its own row");
  console.assert(log[0].shotNum === 2, "second logged attempt should be shot 2");
  console.assert(log[0].reachedFullDraw === true, "an attempt that reached full draw must not be marked as short of it");

  // --- Pose loss on an UNQUALIFIED attempt (short, shallow) discards it, same as one that ends
  // by hands relaxing — losing tracking must not manufacture a shot out of noise.
  trackShotAttempt(sample(0.35), 6000); // crosses the floor, shallow
  endAttempt(6100); // pose lost 100ms later — short AND shallow (renderLoop's !landmarks branch calls endAttempt directly)
  console.assert(log.length === 2, "pose loss on an attempt that never qualified must not log a phantom row");
  console.assert(rejectedAttemptCount === 2, "pose-loss discard should still count as a rejected attempt");
  console.assert(attempt === null, "ending an attempt by any path must clear it so the next rise starts fresh");

  // --- Pose loss on a QUALIFIED attempt (deep and long enough already) still logs it — losing
  // tracking is what ENDS the attempt, it doesn't retroactively disqualify one that already
  // earned its place.
  trackShotAttempt(sample(0.4), 7000);
  trackShotAttempt(sample(0.7), 7400); // deep enough (0.7 >= 0.6)
  endAttempt(7700); // pose lost 700ms after the attempt started — long enough (>= 600ms)
  console.assert(log.length === 3, "pose loss on a qualified attempt must still log it");
  console.assert(log[0].shotNum === 3, "third logged attempt should be shot 3");
  console.assert(log[0].handSep === 0.7, "the logged row should still carry the attempt's peak separation even though it ended via pose loss");

  // --- A further, separate attempt becomes its own row, not merged into the previous one.
  trackShotAttempt(sample(0.4), 8000);
  trackShotAttempt(sample(0.9), 8400);
  trackShotAttempt(sample(0.05), 9000); // 1000ms elapsed — ends
  console.assert(log.length === 4, "a further attempt must log as its own row, not merge into the previous one");
  console.assert(log[0].shotNum === 4 && log[1].shotNum === 3, "log should stay newest-first");

  // shotCount must track LOGGED attempts only, so the shot numbers stay contiguous even with two
  // rejections mixed in among them (checked incrementally above; restated here for the record).
  console.assert(shotCount === 4, "shotCount should equal the number of LOGGED attempts, not the number of attempts started");
  console.assert(rejectedAttemptCount === 2, "exactly the two discarded attempts (the wiggle and the short pose-loss) should be counted as rejected");

  // Cap: only the newest SHOT_LOG_MAX entries are kept, still newest-first. Every iteration here
  // is deep and long enough to qualify, so every one of them logs.
  let capT = 10000;
  for (let i = 0; i < SHOT_LOG_MAX + 3; i++) {
    trackShotAttempt(sample(0.4), capT);
    capT += 400;
    trackShotAttempt(sample(0.9), capT);
    capT += 400;
    trackShotAttempt(sample(0.05), capT); // ends — 800ms elapsed, over SHOT_MIN_DURATION_MS
    capT += 1000;
  }
  console.assert(log.length === SHOT_LOG_MAX, `log should cap at ${SHOT_LOG_MAX} entries, has ${log.length}`);
  console.assert(
    log[0].shotNum > log[SHOT_LOG_MAX - 1].shotNum,
    "log should stay newest-first even once it's capped"
  );

  // --- Feature A: shoulder drop = ear-to-shoulder gap, normalised by torso length, as a %.
  // Same-side ear preferred, falling back to the other ear (or to null) when it's occluded —
  // side-on framing means one ear is very often out of view.
  const dropHips = { 23: { x: 0.3, y: 0.7 }, 24: { x: 0.7, y: 0.7 } };
  const dropLm1 = mkLandmarks({
    ...dropHips,
    11: { x: 0.3, y: 0.4 }, // L shoulder
    7: { x: 0.3, y: 0.3 }, // L ear (same side) — gap 0.1, torso 0.3 -> 33.3%
    8: { x: 0.7, y: 0.35 }, // R ear (other side) — deliberately different; must NOT be used
  });
  const drop1 = shoulderDropOf(dropLm1, L_SHOULDER, L_EAR, R_EAR, L_HIP, R_SHOULDER, R_HIP);
  console.assert(
    drop1 !== null && Math.abs(drop1 - 33.33) < 0.1,
    "shoulder drop should be the same-side ear-to-shoulder gap as a % of torso length"
  );

  const dropLm2 = mkLandmarks({
    ...dropHips,
    11: { x: 0.3, y: 0.4 },
    7: { x: 0, y: 0, visibility: 0 }, // same-side ear occluded
    8: { x: 0.7, y: 0.25 }, // other-side ear — gap 0.15, torso 0.3 -> 50%
  });
  const drop2 = shoulderDropOf(dropLm2, L_SHOULDER, L_EAR, R_EAR, L_HIP, R_SHOULDER, R_HIP);
  console.assert(
    drop2 !== null && Math.abs(drop2 - 50) < 0.1,
    "shoulder drop should fall back to the other ear when the same-side one isn't visible"
  );

  const dropLm3 = mkLandmarks({
    ...dropHips,
    11: { x: 0.3, y: 0.4 },
    7: { x: 0, y: 0, visibility: 0 },
    8: { x: 0, y: 0, visibility: 0 },
  });
  console.assert(
    shoulderDropOf(dropLm3, L_SHOULDER, L_EAR, R_EAR, L_HIP, R_SHOULDER, R_HIP) === null,
    "shoulder drop should be uncertain (null), not a guess, when both ears are occluded"
  );

  const dropLm4 = mkLandmarks({
    ...dropHips,
    11: { x: 0, y: 0, visibility: 0 }, // shoulder itself occluded
    7: { x: 0.3, y: 0.3 },
    8: { x: 0.7, y: 0.3 },
  });
  console.assert(
    shoulderDropOf(dropLm4, L_SHOULDER, L_EAR, R_EAR, L_HIP, R_SHOULDER, R_HIP) === null,
    "shoulder drop should be uncertain (null) when the shoulder itself isn't visible"
  );

  // --- Feature B: draw-elbow alignment. angleAt gives the deviation magnitude; direction
  // (high/low) must come from vertical position only. The critical case: mirroring the whole
  // geometry left-right (as handedness or the front camera both do) must NOT flip high/low,
  // even though it would flip the sign of a cross product — which is exactly why this isn't
  // implemented with one.
  rightHanded = true;
  const inLine = mkLandmarks({ 15: { x: 0.0, y: 0.5 }, 16: { x: 0.5, y: 0.5 }, 14: { x: 1.0, y: 0.5 } });
  const inLineResult = drawElbowAlignmentOf(inLine);
  console.assert(
    inLineResult !== null && Math.abs(inLineResult.deviation) < 0.01,
    "elbow exactly on the extended bow-wrist -> draw-wrist line should read as ~0° deviation"
  );

  const highRH = mkLandmarks({ 15: { x: 0.0, y: 0.5 }, 16: { x: 0.5, y: 0.5 }, 14: { x: 1.0, y: 0.4 } });
  const highRHResult = drawElbowAlignmentOf(highRH);
  console.assert(
    highRHResult !== null && highRHResult.direction === "high" && highRHResult.deviation > 5,
    "elbow physically higher than the extended line (smaller y) should report direction: high"
  );

  const lowRH = mkLandmarks({ 15: { x: 0.0, y: 0.5 }, 16: { x: 0.5, y: 0.5 }, 14: { x: 1.0, y: 0.6 } });
  const lowRHResult = drawElbowAlignmentOf(lowRH);
  console.assert(
    lowRHResult !== null && lowRHResult.direction === "low" && lowRHResult.deviation > 5,
    "elbow physically lower than the extended line (bigger y) should report direction: low"
  );

  // Same two physical relationships, mirrored left-right (rightHanded: false swaps which wrist
  // is "bow" vs "draw", the same as flipping the whole scene) — vertical position is invariant
  // under a horizontal mirror, so the reported direction must not change.
  rightHanded = false;
  const highLH = mkLandmarks({
    16: { x: 1.0, y: 0.5 }, // bow wrist (R_WRIST now, since rightHanded is false)
    15: { x: 0.5, y: 0.5 }, // draw wrist (L_WRIST)
    13: { x: 0.0, y: 0.4 }, // draw elbow (L_ELBOW) — still physically higher
  });
  const highLHResult = drawElbowAlignmentOf(highLH);
  console.assert(
    highLHResult !== null && highLHResult.direction === "high",
    "mirrored (left-handed) geometry: elbow still physically higher must still report high, not low"
  );

  const lowLH = mkLandmarks({
    16: { x: 1.0, y: 0.5 },
    15: { x: 0.5, y: 0.5 },
    13: { x: 0.0, y: 0.6 }, // still physically lower
  });
  const lowLHResult = drawElbowAlignmentOf(lowLH);
  console.assert(
    lowLHResult !== null && lowLHResult.direction === "low",
    "mirrored (left-handed) geometry: elbow still physically lower must still report low, not high"
  );

  rightHanded = true;
  const vertical = mkLandmarks({ 15: { x: 0.5, y: 0.2 }, 16: { x: 0.5, y: 0.5 }, 14: { x: 0.5, y: 0.8 } });
  console.assert(
    drawElbowAlignmentOf(vertical) === null,
    "a near-vertical bow-wrist -> draw-wrist line can't tell high from low, so this should be uncertain (null), not a guess"
  );

  // --- Feature C: signed elbow deviation (high = positive, low = negative), so the shot log
  // summary can average it honestly instead of "8 high" and "8 low" cancelling out to ~0.
  console.assert(Math.abs(inLineResult.signed) < 0.01, "elbow dead in line should have ~0 signed deviation");
  console.assert(highRHResult.signed > 0, "a high elbow should have a positive signed deviation");
  console.assert(lowRHResult.signed < 0, "a low elbow should have a negative signed deviation");
  console.assert(
    Math.abs(highRHResult.signed - highRHResult.deviation) < 0.01,
    "signed deviation should equal the plain deviation magnitude when the direction is high"
  );
  console.assert(
    Math.abs(lowRHResult.signed + lowRHResult.deviation) < 0.01,
    "signed deviation should be the negative of the plain deviation magnitude when the direction is low"
  );

  // --- Feature D: summarizeShots — pure stats over the shot log (own average, own spread, each
  // shot's deviation from that average). No DOM, no module state, so plain fixture arrays are
  // enough; nothing here needs the save/restore dance the fixtures above use.
  const statFixture = [
    { shotNum: 1, bowArmAngle: 160, shoulderDrop: { bow: 40, draw: 50 }, elbowAlign: { signed: 2 } },
    { shotNum: 2, bowArmAngle: 170, shoulderDrop: { bow: 44, draw: null }, elbowAlign: { signed: -2 } },
    { shotNum: 3, bowArmAngle: null, shoulderDrop: { bow: 42, draw: 48 }, elbowAlign: null },
  ];
  const statResult = summarizeShots(statFixture);
  console.assert(
    statResult.bowArm.n === 2 && statResult.bowArm.total === 3,
    "bow arm average should count only the 2 non-null readings, out of 3 total attempts"
  );
  console.assert(Math.abs(statResult.bowArm.average - 165) < 0.001, "average of 160 and 170 should be 165");
  console.assert(Math.abs(statResult.bowArm.spread - 10) < 0.001, "spread should be the 10° gap between 160 and 170");
  console.assert(
    statResult.bowArm.deviations[1] === -5 && statResult.bowArm.deviations[2] === 5,
    "each shot's deviation should be its own value minus the average, signed"
  );
  console.assert(
    statResult.shoulderDraw.n === 2 && statResult.shoulderDraw.total === 3,
    "draw-shoulder average should exclude shot 2's null reading but still count 3 total attempts, not 2"
  );

  // Outlier picking: the shot furthest from the average, not just the first or last one.
  const outlierFixture = [
    { shotNum: 1, bowArmAngle: 160 },
    { shotNum: 2, bowArmAngle: 170 },
    { shotNum: 3, bowArmAngle: 168 },
  ];
  const outlierResult = summarizeShots(outlierFixture);
  console.assert(
    outlierResult.bowArm.outlierShotNum === 1,
    "shot 1 (160, furthest from the 166 average) should be flagged as the outlier, not shots 2 or 3"
  );

  // Thin samples must not throw, and must not fake a spread or a "+0" deviation.
  const oneShotResult = summarizeShots([
    { shotNum: 1, bowArmAngle: 170, shoulderDrop: { bow: null, draw: null }, elbowAlign: null },
  ]);
  console.assert(
    oneShotResult.bowArm.n === 1 && oneShotResult.bowArm.average === 170,
    "a single reading should still report a plain average"
  );
  console.assert(oneShotResult.bowArm.spread === null, "a single reading has no spread to report");
  console.assert(
    Object.keys(oneShotResult.bowArm.deviations).length === 0,
    "a single reading has no baseline to deviate from — must not fake a +0 deviation"
  );
  console.assert(oneShotResult.bowArm.outlierShotNum === null, "a single reading can't have an outlier");

  const emptyResult = summarizeShots([]);
  console.assert(
    emptyResult.bowArm.n === 0 && emptyResult.bowArm.total === 0 && emptyResult.bowArm.average === null,
    "an empty log must not throw, and must report nothing rather than a fake 0"
  );

  // --- Outlier gating: summarizeMeasure always names an outlierShotNum once n >= 2 (someone is
  // always the maximum), but shotValueHtml must only actually FLAG it when that shot's own
  // deviation exceeds the measure's display cutoff. This is the exact field bug: draw-shoulder
  // read 41% and 42% (half a point apart, well inside the ±8pp cutoff) and still showed "most
  // different this session" right next to "right on your average" — a self-contradiction.
  const tightFixture = [
    { shotNum: 1, shoulderDrop: { draw: 41 } },
    { shotNum: 2, shoulderDrop: { draw: 42 } },
  ];
  const tightStats = summarizeShots(tightFixture);
  console.assert(
    tightStats.shoulderDraw.outlierShotNum !== null,
    "sanity check: the pure function should still name a max-deviation shot even in a tight cluster"
  );
  const tightShot1 = shotValueHtml("41%", 41, "%", 1, tightStats.shoulderDraw, SHOULDER_DROP_CONSISTENCY_MAX_DEVIATION);
  const tightShot2 = shotValueHtml("42%", 42, "%", 2, tightStats.shoulderDraw, SHOULDER_DROP_CONSISTENCY_MAX_DEVIATION);
  console.assert(
    tightShot1.flagged === false && tightShot2.flagged === false,
    "a tight cluster, all readings well inside the display cutoff, must not flag anything as an outlier"
  );

  // Mirror case: a reading that genuinely exceeds the cutoff must still be flagged. Three
  // points, deliberately NOT symmetric — a two-point spread ties on |deviation| for both shots,
  // and the tie-break in summarizeMeasure (first one wins) would make this pass or fail
  // depending on which shotNum happens to go first, which isn't what's being tested here.
  const spreadFixture = [
    { shotNum: 1, shoulderDrop: { draw: 40 } },
    { shotNum: 2, shoulderDrop: { draw: 42 } },
    { shotNum: 3, shoulderDrop: { draw: 65 } }, // average 49, so shot 3 is +16, unambiguously furthest
  ];
  const spreadStats = summarizeShots(spreadFixture);
  const spreadShot3 = shotValueHtml("65%", 65, "%", 3, spreadStats.shoulderDraw, SHOULDER_DROP_CONSISTENCY_MAX_DEVIATION);
  console.assert(spreadShot3.flagged === true, "a reading that genuinely exceeds the display cutoff should still be flagged");

  // --- Feature E: narrateMeasure — the plain-English consistency wording that leads the log.
  // Pure: fixture arrays + a value-getter in, one sentence (or null) out. wordFor doesn't matter
  // for most of these checks, so a fixed stand-in is used except where the actual word matters.
  {
    const upDown = (sign) => (sign > 0 ? "up" : "down");

    // A tight cluster of shots must read as "steady" — never a fabricated drift or outlier claim
    // over noise this small. Spread here is +/-1 around an average of 170; OUTLIER_SCATTER_FACTOR
    // and DRIFT_GAP_FACTOR both need a MUCH bigger gap than this to fire.
    const steadyFixture = [
      { shotNum: 1, bowArmAngle: 170 },
      { shotNum: 2, bowArmAngle: 171 },
      { shotNum: 3, bowArmAngle: 169 },
      { shotNum: 4, bowArmAngle: 170.5 },
      { shotNum: 5, bowArmAngle: 169.5 },
    ];
    const steadyResult = narrateMeasure(steadyFixture, (e) => e.bowArmAngle, "Bow arm", upDown);
    console.assert(
      steadyResult && steadyResult.text.startsWith("Bow arm — steady"),
      `a tight cluster of shots should read as steady, got: ${steadyResult && steadyResult.text}`
    );
    console.assert(steadyResult.outlierShotNum === null, "a steady session must not name an outlier shot");

    // One shot sitting clearly apart from an otherwise tight cluster must be named — and named
    // by the RIGHT shot number. Shots 1-4 sit within +/-2 of each other; shot 5 (200) is wildly
    // farther out than the other four's own scatter, which is exactly what OUTLIER_SCATTER_FACTOR
    // is checking for.
    const outlierNarrFixture = [
      { shotNum: 1, bowArmAngle: 170 },
      { shotNum: 2, bowArmAngle: 169 },
      { shotNum: 3, bowArmAngle: 171 },
      { shotNum: 4, bowArmAngle: 170 },
      { shotNum: 5, bowArmAngle: 200 },
    ];
    const outlierNarrResult = narrateMeasure(outlierNarrFixture, (e) => e.bowArmAngle, "Bow arm", upDown);
    console.assert(
      outlierNarrResult && outlierNarrResult.outlierShotNum === 5,
      `shot 5 (200, wildly apart from the other four) should be named as the standout, got: ${outlierNarrResult && JSON.stringify(outlierNarrResult)}`
    );
    console.assert(
      outlierNarrResult.text.includes("shot 5 stands out"),
      `the sentence should name shot 5 by number, got: ${outlierNarrResult.text}`
    );

    // A real, monotonic trend across the end — values climbing steadily from the first shot to
    // the last — must read as drift. Same five values, chronological order (shotNum ascending):
    // 160, 163, 166, 169, 172.
    const driftValues = [160, 163, 166, 169, 172];
    const driftFixture = driftValues.map((v, i) => ({ shotNum: i + 1, bowArmAngle: v }));
    const driftResult = narrateMeasure(driftFixture, (e) => e.bowArmAngle, "Bow arm", upDown);
    console.assert(
      driftResult && driftResult.text.includes("drifted"),
      `a steadily climbing session should read as drift, got: ${driftResult && driftResult.text}`
    );
    console.assert(driftResult.outlierShotNum === null, "a drift result must not also name a single outlier shot");

    // The EXACT SAME five values, shuffled into an order where the trend disappears (first half
    // and second half average out close together) must NOT read as drift — proving the drift
    // check is genuinely order-sensitive, not just "the numbers happen to have this much spread".
    const shuffledValues = [166, 160, 172, 163, 169]; // same 5 numbers as driftValues, different assignment to shotNum
    const shuffledFixture = shuffledValues.map((v, i) => ({ shotNum: i + 1, bowArmAngle: v }));
    const shuffledResult = narrateMeasure(shuffledFixture, (e) => e.bowArmAngle, "Bow arm", upDown);
    console.assert(
      shuffledResult && !shuffledResult.text.includes("drifted"),
      `the same values in shuffled (non-trending) order must not read as drift, got: ${shuffledResult && shuffledResult.text}`
    );

    // One shot, or two shots: no honest consistency story to tell yet — must say so plainly and
    // never compute a fabricated steady/drift/outlier claim from almost nothing.
    const oneNarr = narrateMeasure([{ shotNum: 1, bowArmAngle: 170 }], (e) => e.bowArmAngle, "Bow arm", upDown);
    console.assert(
      oneNarr && oneNarr.text.includes("only one shot") && oneNarr.outlierShotNum === null,
      `a single shot must produce the honest one-shot wording, not a fabricated claim, got: ${oneNarr && oneNarr.text}`
    );
    const twoNarr = narrateMeasure(
      [{ shotNum: 1, bowArmAngle: 170 }, { shotNum: 2, bowArmAngle: 170 }],
      (e) => e.bowArmAngle,
      "Bow arm",
      upDown
    );
    console.assert(
      twoNarr && twoNarr.text.includes("only two shots") && twoNarr.outlierShotNum === null,
      `two shots must produce the honest two-shot wording, not a fabricated steady/drift claim, got: ${twoNarr && twoNarr.text}`
    );

    // A measure whose readings were ALL uncertain this session (getValue returns null every
    // time) must produce no statement at all — not an empty-string sentence, not a "no data"
    // line, nothing to render.
    const allUncertainFixture = [
      { shotNum: 1, shoulderDrop: { draw: null } },
      { shotNum: 2, shoulderDrop: { draw: null } },
      { shotNum: 3, shoulderDrop: { draw: null } },
    ];
    const allUncertainResult = narrateMeasure(allUncertainFixture, (e) => e.shoulderDrop?.draw ?? null, "Draw shoulder", upDown);
    console.assert(
      allUncertainResult === null,
      `a measure with every reading uncertain must produce no statement at all, got: ${JSON.stringify(allUncertainResult)}`
    );
  }

  // --- Shot clips: MIME selection order, attaching a blob to the right shot number, discarding
  // one whose shot has fallen off the log, and revoking a clip when eviction removes its row.
  // These call the clip functions directly rather than through startClipRecording, which is
  // deliberately suppressed during selfTest (see selfTestInProgress) — there's no real camera
  // here to record from, only the bookkeeping around a recording, which is what's under test.
  {
    if (typeof MediaRecorder !== "undefined" && typeof MediaRecorder.isTypeSupported === "function") {
      const savedIsTypeSupported = MediaRecorder.isTypeSupported;
      MediaRecorder.isTypeSupported = (t) => t === "video/webm;codecs=vp8";
      console.assert(
        pickMimeType() === "video/webm;codecs=vp8",
        "pickMimeType should return the first candidate (in CLIP_MIME_CANDIDATES order) the browser claims to support"
      );
      MediaRecorder.isTypeSupported = () => false;
      console.assert(
        pickMimeType() === null,
        "pickMimeType should return null (let the browser pick its own default) when nothing on the list is supported"
      );
      MediaRecorder.isTypeSupported = savedIsTypeSupported;
    } else {
      console.assert(pickMimeType() === null, "pickMimeType should return null when MediaRecorder isn't available at all");
    }

    // Attaching: a blob attaches to the log row with the matching shot number, and only that row.
    log = [{ shotNum: 5 }, { shotNum: 4 }];
    const blobA = new Blob(["a"], { type: "video/webm" });
    attachClipToShot(4, blobA);
    const rowFour = log.find((e) => e.shotNum === 4);
    const rowFive = log.find((e) => e.shotNum === 5);
    console.assert(
      rowFour.clipUrl && rowFour.clipBlob === blobA,
      "a clip should attach to the log row whose shotNum matches, gaining a clipBlob and a playable clipUrl"
    );
    console.assert(rowFive.clipUrl === undefined, "a clip attaching to one row must not touch any other row");
    URL.revokeObjectURL(rowFour.clipUrl); // tidy up the object URL this test itself created

    // Discarding: a shot number no longer in the log (already bumped off the end) gets nothing
    // attached, and nothing throws — this is the normal case for a clip that finishes recording
    // well after its shot, once SHOT_LOG_MAX newer attempts have already happened.
    log = [{ shotNum: 9 }];
    const blobB = new Blob(["b"], { type: "video/webm" });
    attachClipToShot(3, blobB); // shot 3 fell off the log already
    console.assert(
      log.length === 1 && log[0].clipUrl === undefined,
      "attaching a clip for a shot that's no longer in the log must be a silent no-op, not attach to an unrelated row"
    );

    // Revoking on eviction: logShot must revoke the object URL of any row it pushes off the end
    // of SHOT_LOG_MAX, so a clip's memory doesn't outlive the row it belonged to.
    const revokedUrls = [];
    const savedRevoke = URL.revokeObjectURL;
    URL.revokeObjectURL = (u) => revokedUrls.push(u);
    log = [];
    shotCount = 0;
    let evictT = 50000; // a fresh, well-separated block of the fake clock so this loop's timing can't collide with the timeline above
    for (let i = 0; i < SHOT_LOG_MAX; i++) {
      trackShotAttempt(sample(0.9), evictT); // deep enough to clear SHOT_MIN_PEAK_SEP_FRACTION
      evictT += 800; // long enough to clear SHOT_MIN_DURATION_MS
      trackShotAttempt(sample(0.05), evictT); // ends the attempt, logs it
      evictT += 200;
    }
    console.assert(log.length === SHOT_LOG_MAX, "setup: log should be exactly full before the eviction check");
    const oldestUrl = "blob:fake-oldest-for-selftest";
    log[log.length - 1].clipUrl = oldestUrl; // pretend the row about to be evicted has a clip attached
    trackShotAttempt(sample(0.9), evictT);
    evictT += 800;
    trackShotAttempt(sample(0.05), evictT); // one more attempt: pushes the oldest row (the one just tagged) off the end
    console.assert(
      revokedUrls.includes(oldestUrl),
      "evicting a row that has a clip attached must revoke its object URL"
    );
    URL.revokeObjectURL = savedRevoke;
  }

  // --- One Euro filter: pure logic, no DOM/MediaPipe involved, so these run straight against
  // the class with plain fixture numbers.
  {
    // A constant input must converge to that constant with no drift and no lasting offset —
    // once the filter has settled, feeding it the same value again and again should leave it
    // sitting on that value, not creeping away from it.
    const constFilter = new OneEuroFilter(SMOOTH_MIN_CUTOFF, SMOOTH_BETA, SMOOTH_DCUTOFF);
    let t = 0;
    let out = 0;
    for (let i = 0; i < 60; i++) {
      t += 1 / 30;
      out = constFilter.filter(5, t);
    }
    console.assert(
      Math.abs(out - 5) < 1e-6,
      `a constant input should converge exactly to that constant, got ${out} instead of 5`
    );

    // A noisy signal around a fixed value should come out with LESS variance than it went in —
    // that's the entire point of smoothing. Measured, not eyeballed: seeded pseudo-random noise
    // so the test is deterministic.
    let seed = 42;
    const pseudoRandom = () => {
      // simple deterministic LCG — good enough for a repeatable test fixture, no need for real
      // randomness here
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return (seed / 0x7fffffff) * 2 - 1; // -1..1
    };
    const noiseFilter = new OneEuroFilter(SMOOTH_MIN_CUTOFF, SMOOTH_BETA, SMOOTH_DCUTOFF);
    const raw = [];
    const smoothedOut = [];
    let nt = 0;
    for (let i = 0; i < 200; i++) {
      nt += 1 / 30; // a steady 30fps signal
      const noisy = 10 + pseudoRandom() * 0.05; // fixed value 10, ±0.05 noise
      raw.push(noisy);
      smoothedOut.push(noiseFilter.filter(noisy, nt));
    }
    const variance = (arr) => {
      const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
      return arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length;
    };
    // Drop the first few samples of the smoothed series — the filter is still settling from a
    // cold start right at the beginning, which isn't the steady-state behaviour under test.
    const rawVar = variance(raw.slice(20));
    const smoothedVar = variance(smoothedOut.slice(20));
    console.assert(
      smoothedVar < rawVar * 0.5,
      `smoothed variance (${smoothedVar.toFixed(6)}) should be well under half the raw variance (${rawVar.toFixed(6)}) around a fixed value`
    );

    // A fast ramp must be tracked with materially less lag than a heavy FIXED low-pass filter
    // (i.e. one that never adapts) would give — this is the actual point of using One Euro
    // instead of a plain moving average. Compare against a plain fixed-cutoff low-pass run over
    // the identical ramp.
    const fixedLowPass = (cutoff, samples, dtSec) => {
      let prev = null;
      let outVal = 0;
      for (const s of samples) {
        if (prev === null) {
          outVal = s;
        } else {
          outVal = outVal + OneEuroFilter.alpha(cutoff, dtSec) * (s - outVal);
        }
        prev = s;
      }
      return outVal;
    };
    const rampDt = 1 / 30;
    const rampSamples = [];
    for (let i = 0; i < 30; i++) rampSamples.push(i * 0.05); // a steady, fast ramp: 0 -> 1.45 over one second
    const rampTrue = rampSamples[rampSamples.length - 1];

    const adaptiveRampFilter = new OneEuroFilter(SMOOTH_MIN_CUTOFF, SMOOTH_BETA, SMOOTH_DCUTOFF);
    let rt = 0;
    let adaptiveOut = 0;
    for (const s of rampSamples) {
      rt += rampDt;
      adaptiveOut = adaptiveRampFilter.filter(s, rt);
    }
    const adaptiveLag = Math.abs(rampTrue - adaptiveOut);
    // A heavy fixed low-pass at the SAME cutoff the adaptive filter starts at when still
    // (SMOOTH_MIN_CUTOFF) — i.e. what you'd get WITHOUT the beta term ever kicking in.
    const fixedOut = fixedLowPass(SMOOTH_MIN_CUTOFF, rampSamples, rampDt);
    const fixedLag = Math.abs(rampTrue - fixedOut);
    console.assert(
      adaptiveLag < fixedLag * 0.5,
      `adaptive filter's lag on a fast ramp (${adaptiveLag.toFixed(4)}) should be well under a fixed heavy low-pass's lag (${fixedLag.toFixed(4)}) — otherwise beta isn't doing anything`
    );

    // Reset must clear state completely: the very next sample after a reset should come out
    // exactly equal to itself, not dragged toward wherever the filter was before the reset.
    const resetFilter = new OneEuroFilter(SMOOTH_MIN_CUTOFF, SMOOTH_BETA, SMOOTH_DCUTOFF);
    let rst = 0;
    for (let i = 0; i < 30; i++) {
      rst += 1 / 30;
      resetFilter.filter(0, rst); // settle near 0
    }
    resetFilter.reset();
    const firstAfterReset = resetFilter.filter(100, 0); // a totally fresh position/time after reset
    console.assert(
      firstAfterReset === 100,
      `first output after reset should equal the first new input exactly (100), got ${firstAfterReset} — state wasn't cleared`
    );

    // Elapsed time must be respected: the SAME sample sequence run at a different dt should
    // produce different smoothing, since a filter using an assumed frame rate instead of real
    // elapsed time would give the same answer regardless.
    const seq = [0, 1, 1, 1, 1, 1];
    const runAt = (dtSec) => {
      const f = new OneEuroFilter(SMOOTH_MIN_CUTOFF, SMOOTH_BETA, SMOOTH_DCUTOFF);
      let tt = 0;
      let last = 0;
      for (const v of seq) {
        tt += dtSec;
        last = f.filter(v, tt);
      }
      return last;
    };
    const outFast = runAt(1 / 60); // 60fps: less real time has passed by the same sample count
    const outSlow = runAt(1 / 10); // 10fps: more real time has passed by the same sample count
    console.assert(
      Math.abs(outFast - outSlow) > 1e-6,
      `identical sample sequence at different dt should NOT produce the same output (got ${outFast} and ${outSlow}) — dt is being ignored`
    );
  }

  // --- LandmarkSmoother: the array-level wrapper around OneEuroFilter that renderLoop actually
  // uses. Confirms it smooths x/y, leaves visibility completely untouched, and resets cleanly.
  {
    const smoother = new LandmarkSmoother(SMOOTH_MIN_CUTOFF, SMOOTH_BETA, SMOOTH_DCUTOFF);
    const mkFrame = (x, y, visibility) => [{ x, y, z: 0, visibility }];
    let st = 0;
    let lastFrame = null;
    for (let i = 0; i < 30; i++) {
      st += 1 / 30;
      // A still point plus a tiny wobble — same shape as outdoor camera jitter on a joint that
      // isn't actually moving.
      const wobble = i % 2 === 0 ? 0.01 : -0.01;
      lastFrame = smoother.smooth(mkFrame(0.5 + wobble, 0.5, 0.9), st);
    }
    console.assert(
      Math.abs(lastFrame[0].x - 0.5) < 0.01,
      "LandmarkSmoother should settle near the true still position despite per-frame wobble"
    );
    console.assert(
      lastFrame[0].visibility === 0.9,
      "LandmarkSmoother must pass visibility through completely untouched, never smoothed"
    );

    smoother.reset();
    const freshFrame = smoother.smooth(mkFrame(0.9, 0.1, 0.3), 0);
    console.assert(
      freshFrame[0].x === 0.9 && freshFrame[0].y === 0.1,
      "LandmarkSmoother.reset() should make the next frame come out completely unsmoothed"
    );
  }

  selfTestInProgress = false;
  rightHanded = savedHanded;
  lastDrawWrist = savedLastWrist;
  attempt = savedAttempt;
  log = savedLog;
  shotCount = savedShotCount;
  rejectedAttemptCount = savedRejectedAttemptCount;

  console.log("selfTest done — check above for any failed console.assert");
}

if (location.search.includes("selftest")) selfTest();

main();
