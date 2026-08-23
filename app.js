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

// ===== REGION-OF-INTEREST CROPPING — performance/accuracy knobs, not calibration; safe to
// change without a coach. This is the real fix for outdoor jitter, not just smoothing it away
// (see SMOOTHING above, which still runs on top of this). At five metres the archer is a small
// part of the camera frame, and MediaPipe itself resizes whatever image it's given down to its
// model's small square input before it ever looks at it — so without this, the model is making
// every joint guess from a version of the archer only a few dozen pixels tall. Instead of handing
// MediaPipe the whole video frame, the app remembers roughly where the archer was standing last
// frame, crops a generous box around that, and hands MediaPipe an upscaled close-up of just that
// box — same camera, same distance, far more pixels of archer reaching the model. See the "ROI
// CROPPING" runtime block further down for how the box is computed, and CLAUDE.md for the full
// write-up including the re-acquire-on-loss behaviour.
const ROI_CROPPING_ENABLED = true; // master on/off switch. Set to false to go back to whole-frame detection exactly like before this feature existed — flip this first if cropping is ever suspected of causing trouble in the field, since it rules the whole feature out in one line
const ROI_CANVAS_SIZE = 512; // pixel size (square) of the offscreen canvas the crop gets drawn into before MediaPipe sees it. BIGGER hands the model a more detailed close-up (steadier) but costs more GPU time per frame, same trade-off as the POSE MODEL block above; SMALLER is cheaper but starts to give back the pixels this whole feature exists to gain
const ROI_PADDING_FRACTION = 0.6; // extra space added around last frame's body box, as a fraction of that box's own size, on every side. BIGGER makes the crop more forgiving of a fast raise or a step sideways (less likely to lose the archer out of the box) but zooms in less; SMALLER zooms in more but risks the box not containing him next frame
const ROI_MIN_VISIBLE_LANDMARKS = 4; // fewer confidently-visible landmarks than this in a frame and the app treats the archer as "not really found" — it will not trust a crop box built from a couple of stray points, and re-acquires on the whole frame next frame instead (see "loss of tracking" below)
const ROI_SMOOTHING = 0.35; // how much the crop box itself resists moving, frame to frame — 0 means it jumps straight to chase wherever the body was JUST seen (can flicker/re-zoom on ordinary landmark noise); closer to 1 means it barely moves at all (steadier crop, but slower to follow a real step sideways). This is the hysteresis that stops the box — and therefore the zoom level and framing handed to the model — chasing noise every single frame
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
const btnMirror = document.getElementById("btn-mirror");
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

// Manual mirror toggle (🪞 button) — set at setup time, before the owner walks off, same as the
// handedness toggle. Deliberately NOT "mirrored: true/false" on its own: it means "flip away from
// whatever this camera's own default is", so switching cameras while the toggle is on still
// mirrors relative to the new camera's default rather than snapping to one fixed state. See
// effectiveMirror below for the actual combination of the two. Starts false — untouched, each
// camera just shows its own default (front mirrored, rear not), which is what the button already
// did before this toggle existed.
let mirrorToggled = false;

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
let shotCount = 0; // total attempts this session, keeps counting even once the log above fills up
let log = []; // newest first
let attempt = null; // the attempt currently in progress, if any — see trackShotAttempt below

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
let modelMeasureWindowStartMs = null; // wall-clock time (performance.now()) of the first frame counted toward the measurement window — lets measurePoseModelPerf report the REAL rendered frame rate over that window (drawing/smoothing/everything included), not just a number derived from inference time alone
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

// Called once per frame from renderLoop with how long that frame's detectForVideo call took (in
// milliseconds) and the frame's own timestamp (performance.now(), from the top of renderLoop —
// NOT re-read afterwards, so the measurement window's wall-clock length isn't inflated by this
// function's own work). No-ops once modelDecisionMade is true — see that variable's comment.
// Skips the first MODEL_WARMUP_FRAMES entirely (cold-start frames are always slow and not
// representative), then averages the next MODEL_MEASURE_FRAMES to make the one-time decision.
//
// Reports two different numbers, not one, because they answer different questions and conflating
// them was actively misleading: "avgMs" is purely how long pose detection itself took, the direct
// cost of this feature; "renderedFps" is how many whole frames (detection + smoothing + drawing +
// everything else in renderLoop) actually got through per second over the same window — the
// number that reflects what the owner's phone actually delivered. A frame rate computed from
// inference time alone (1000/avgMs) ignores every other cost in the frame and can look many times
// faster than the app really ran, which is exactly the bug this replaced (see CLAUDE.md).
function measurePoseModelPerf(inferenceMs, nowMs) {
  if (modelDecisionMade) return;
  modelWarmupSeen++;
  if (modelWarmupSeen <= MODEL_WARMUP_FRAMES) return;
  if (modelMeasureWindowStartMs === null) modelMeasureWindowStartMs = nowMs; // first frame actually counted toward the window
  modelMeasureTotalMs += inferenceMs;
  const measured = modelWarmupSeen - MODEL_WARMUP_FRAMES;
  if (measured < MODEL_MEASURE_FRAMES) return;

  modelDecisionMade = true;
  const avgMs = modelMeasureTotalMs / measured;
  const windowElapsedMs = Math.max(nowMs - modelMeasureWindowStartMs, 1e-6); // guards a divide-by-zero if somehow every measured frame landed on the same timestamp
  const renderedFps = (measured * 1000) / windowElapsedMs;
  if (avgMs > MODEL_SLOW_FRAME_MS && activePoseModel === "full") {
    switchToLitePoseModel(avgMs, renderedFps);
  } else {
    setModelStatusLine(avgMs, renderedFps);
  }
}

// Rebuilds the landmarker on the lighter "lite" model because the warm-up measurement found
// "full" running too slowly on this phone. Must never leave the app with no landmarker at all,
// and a failed rebuild must never interrupt tracking: if creating the new landmarker throws (a
// flaky fetch for the model file, most likely), the OLD "full" landmarker just keeps running —
// slower than ideal, but still tracking, which is what matters (see CLAUDE.md: pose tracking
// must never just stop).
async function switchToLitePoseModel(avgMs, renderedFps) {
  try {
    const next = await createPoseLandmarker("lite");
    const old = poseLandmarker;
    poseLandmarker = next;
    activePoseModel = "lite";
    old?.close?.();
  } catch (err) {
    console.error("archery-form-coach: falling back to lite pose model failed, staying on full", err);
  }
  setModelStatusLine(avgMs, renderedFps);
}

function setModelStatusLine(avgMs, renderedFps) {
  const label =
    activePoseModel === "full"
      ? "full"
      : "lite (auto-switched — full ran too slow on this phone)";
  modelStatusLine = `Pose model: ${label} — pose detection took about ${avgMs.toFixed(1)}ms/frame, ${renderedFps.toFixed(1)} fps actually rendered, measured at startup.`;
  renderShotLog();
}
// ===========================================================================

// ===== ROI CROPPING — runtime state and pure geometry for the region-of-interest crop described
// in the REGION-OF-INTEREST CROPPING constants above. The offscreen canvas below is never added
// to the page (document.createElement, not appendChild) — it exists purely as an intermediate
// image for MediaPipe to look at; it is NEVER what gets drawn to the visible #overlay canvas or
// recorded into a shot's clip (see drawVideoFrame/drawSkeleton further down — those always draw
// the full camera frame). Cropping is an inference-input concern only.
const roiCanvas = document.createElement("canvas");
roiCanvas.width = ROI_CANVAS_SIZE;
roiCanvas.height = ROI_CANVAS_SIZE;
const roiCtx = roiCanvas.getContext("2d");

// The crop box to use for THIS frame's detection, in VIDEO PIXEL space (not normalised) as
// {x, y, size} — always square, x/y is its top-left corner. null means "detect on the whole
// frame this frame": the starting state, and also what losing the archer resets to (see
// renderLoop). Kept in pixel space rather than normalised [0,1] coordinates specifically so the
// squareness below is a square in actual camera pixels — a video frame usually isn't square
// (e.g. 1280x720), so a box that was square in normalised x/y would actually be a rectangle on
// screen, and scaling a rectangle into the square offscreen canvas would stretch the archer,
// corrupting every angle this app measures.
let currentCropBox = null;

// The tight bounding box (in VIDEO PIXEL space) of every landmark confident enough to trust, or
// null if fewer than ROI_MIN_VISIBLE_LANDMARKS qualify — in which case there's nothing
// trustworthy to crop around, and the caller should fall back to whole-frame detection. Pure —
// no DOM — so selfTest can exercise it directly on fixture landmarks.
function boundingBoxOfLandmarks(landmarks, frameWidth, frameHeight) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  let seen = 0;
  for (const lm of landmarks) {
    if (!lm || (lm.visibility ?? 0) < MIN_VISIBILITY) continue;
    const px = lm.x * frameWidth;
    const py = lm.y * frameHeight;
    if (px < minX) minX = px;
    if (px > maxX) maxX = px;
    if (py < minY) minY = py;
    if (py > maxY) maxY = py;
    seen++;
  }
  if (seen < ROI_MIN_VISIBLE_LANDMARKS) return null;
  return { minX, minY, maxX, maxY };
}

// Pads a (minX,minY,maxX,maxY) box by ROI_PADDING_FRACTION, forces it to a square (so the crop
// never distorts the body when it's scaled into the square offscreen canvas — a squashed archer
// would corrupt every angle this app measures), and clamps it to fit inside the frame. Pure
// geometry, no DOM — so selfTest can check the clamping and squareness directly with hand-picked
// numbers, including a box that starts off-centre and not square.
function squareAndClampCropBox(minX, minY, maxX, maxY, frameWidth, frameHeight, paddingFraction) {
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const paddedSize = Math.max(maxX - minX, maxY - minY) * (1 + 2 * paddingFraction);
  // Never ask for a crop bigger than the frame itself — there'd be nothing left to zoom into.
  const size = Math.max(1, Math.min(paddedSize, frameWidth, frameHeight));
  let x = cx - size / 2;
  let y = cy - size / 2;
  // Slide the box back inside the frame rather than shrinking it further, so it keeps its full
  // size (and stays square) right up to the frame's edge — exactly what happens when the archer
  // is standing near the edge of the shot.
  x = Math.min(Math.max(x, 0), frameWidth - size);
  y = Math.min(Math.max(y, 0), frameHeight - size);
  return { x, y, size };
}

// Eases the crop box toward a freshly-computed one instead of snapping straight to it — the
// hysteresis described by ROI_SMOOTHING above, so ordinary landmark noise doesn't make the crop
// (and therefore the zoom level and framing MediaPipe sees) flicker frame to frame. No previous
// box (first acquisition, or just re-acquired after losing the archer) means nothing to ease
// from, so the fresh box is used outright.
function smoothCropBox(prevBox, freshBox, smoothing) {
  if (!prevBox) return freshBox;
  return {
    x: prevBox.x + (1 - smoothing) * (freshBox.x - prevBox.x),
    y: prevBox.y + (1 - smoothing) * (freshBox.y - prevBox.y),
    size: prevBox.size + (1 - smoothing) * (freshBox.size - prevBox.size),
  };
}

// The crop box to use NEXT frame, computed from THIS frame's (already full-frame-normalised —
// see mapCropLandmarkToFullFrame) landmarks. Returns null — meaning "detect on the whole frame
// next frame" — whenever there isn't a confident enough body to trust a box around, which is
// exactly the re-acquire-on-loss behaviour: a crop that no longer contains the archer must never
// get to persist, since detecting inside an empty box forever would be a hang the owner has no
// way to recover from (see CLAUDE.md).
function nextCropBox(landmarks, frameWidth, frameHeight, prevBox) {
  const bbox = boundingBoxOfLandmarks(landmarks, frameWidth, frameHeight);
  if (!bbox) return null;
  const fresh = squareAndClampCropBox(
    bbox.minX, bbox.minY, bbox.maxX, bbox.maxY, frameWidth, frameHeight, ROI_PADDING_FRACTION
  );
  return smoothCropBox(prevBox, fresh, ROI_SMOOTHING);
}

// Maps one landmark from crop-local normalised coordinates (0-1 across the ROI canvas, which is
// what MediaPipe hands back when it was given the crop) into full-frame normalised coordinates
// (0-1 across the actual camera frame — what EVERYTHING downstream of detection in this file,
// the angle maths, the torso-length scale reference, the skeleton drawing, the shot log, assumes
// it's working with). Getting this wrong would silently corrupt every number the owner is tuning,
// so it stays a small, pure, directly-testable function rather than being inlined into
// renderLoop. z is scaled by the same factor as x/y (crop size relative to frame width) so it
// stays roughly proportionate to the coordinate system everything else already assumes, even
// though nothing in this app currently reads z.
function mapCropLandmarkToFullFrame(lm, cropBox, frameWidth, frameHeight) {
  return {
    ...lm,
    x: (cropBox.x + lm.x * cropBox.size) / frameWidth,
    y: (cropBox.y + lm.y * cropBox.size) / frameHeight,
    z: (lm.z ?? 0) * (cropBox.size / frameWidth),
  };
}

// The exact inverse of mapCropLandmarkToFullFrame: a full-frame normalised point back into
// crop-local normalised coordinates. Not used by the live app (detection only ever needs the
// forward direction) — kept purely so selfTest can prove the mapping round-trips cleanly, the
// strongest guarantee that the forward map isn't quietly losing or shifting precision.
function mapFullFrameToCropLocal(lm, cropBox, frameWidth, frameHeight) {
  return {
    ...lm,
    x: (lm.x * frameWidth - cropBox.x) / cropBox.size,
    y: (lm.y * frameHeight - cropBox.y) / cropBox.size,
  };
}
// ===========================================================================

async function startCamera() {
  // A landmark position smoothed from BEFORE a camera switch (different framing, possibly a
  // mirrored front camera) must never blend into positions AFTER it — that would drag the
  // skeleton across the frame on the very first frames of the new camera. See LandmarkSmoother.
  landmarkSmoother.reset();
  // Same reasoning for the ROI crop box: a box computed against the OLD camera's framing means
  // nothing once the video source has changed underneath it, and could even be the wrong shape
  // for the new frame's resolution. Detect on the whole frame again until a fresh box is found.
  currentCropBox = null;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  // No CSS mirroring here any more — see effectiveMirror/withMirror below for why. The <video>
  // element itself is left completely alone (never flipped, never classed); the canvas painted
  // on top of it is opaque every frame (see drawVideoFrame) and is what actually gets mirrored,
  // in its pixels, when that's called for.
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

// Whether a given camera mirrors by default, before the owner's manual toggle is factored in.
// Front camera ("user") defaults to mirrored, matching what people expect of a selfie view; rear
// camera doesn't. Pure and camera-agnostic on purpose — this is the one place that convention
// lives, so selfTest can check it directly instead of poking at the DOM.
function defaultMirrorFor(facingMode) {
  return facingMode === "user";
}

// The actual on-screen (and in-clip, now that mirroring lives in the canvas pixels — see
// withMirror) mirror state: the camera's own default, flipped once more if the owner's manual
// toggle is on. Pure function of the two pieces of state that decide it, so every combination —
// which camera, toggle on or off — can be asserted directly in selfTest without touching the DOM
// or a real camera.
function effectiveMirror(facingMode, toggled) {
  return defaultMirrorFor(facingMode) !== toggled; // XOR: toggled flips whichever default applies
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

  // Feed the shot log regardless of ?debug — the owner needs shot numbers/form readouts
  // whether or not the diagnostic overlay is on; only the display of the extra fields below
  // is debug-gated (see renderShotLog).
  trackShotAttempt({
    handSep,
    bowArmAngle,
    shoulderDrop: shoulderDropSampleOf(landmarks),
    elbowAlign: drawElbowAlignmentOf(landmarks),
    anchorOk,
    armOk,
    sepOk,
    stillOk,
  });

  return anchorOk && armOk && sepOk && stillOk;
}

// Attempt-boundary rule for the shot log: a draw attempt is "in progress" for as long as hand
// separation stays at/above DRAW_ATTEMPT_MIN_SEP, tracking whichever frame in it had the
// highest separation so far. It ends — and gets logged — the moment separation drops back
// below that floor (hands back together at rest). This is the simplest rule that both (a)
// doesn't split one long hold into several rows and (b) doesn't merge two separate shots taken
// back-to-back into one. No timer involved: nothing here expires on its own, ever.
function trackShotAttempt(sample) {
  if (sample.handSep >= DRAW_ATTEMPT_MIN_SEP) {
    const isNewAttempt = !attempt; // hands just left the resting position — a fresh attempt, not a continuation
    if (isNewAttempt || sample.handSep >= attempt.handSep) {
      attempt = { ...sample };
    }
    // Recording starts here, not in endAttempt, so the raise and draw are in the clip too — by
    // the time endAttempt fires the good part is already over.
    if (isNewAttempt) startClipRecording();
  } else {
    endAttempt();
  }
}

// Ends whatever attempt is in progress (if any) and logs it. Called when hand separation drops
// back to resting (from trackShotAttempt above) or when the pose is lost entirely mid-attempt
// (from renderLoop) — either way, whatever was going on has stopped.
function endAttempt() {
  if (!attempt) return;
  const shotNum = logShot(attempt);
  attempt = null;
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

// Plain degrees, e.g. "168". Default average formatter for summaryLine below.
function fmtSignedDeg(v) {
  const r = Math.round(v);
  return r === 0 ? "in line" : `${Math.abs(r)}° ${r > 0 ? "high" : "low"}`;
}

// The one-line, plain-language summary for one measure, shown at the top of the log. Leads with
// the owner's own average and spread — nothing here is judged against a placeholder threshold —
// and says plainly when a reading is missing rather than quietly averaging over a smaller
// sample than the shot count suggests. `formatAvg` lets the elbow measure (signed, direction
// based) reuse this instead of duplicating the sentence structure.
function summaryLine(label, unit, s, formatAvg = (v) => `${Math.round(v)}${unit}`) {
  if (s.n === 0) return `${label} — no readable shots yet.`;
  const missing = s.n < s.total ? ` (based on ${s.n} of ${s.total} shots)` : "";
  const body =
    s.n === 1
      ? `${label} — only one shot so far, ${formatAvg(s.average)}, not enough yet to see how consistent you are`
      : `${label} — you averaged ${formatAvg(s.average)}, varying by ${Math.round(s.spread)}${unit}`;
  return `${body}${missing}.`;
}

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

function renderShotRow(e, stats) {
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

  // At most one outlier mark for the whole row (not one per measure, per the brief) — the reader
  // already knows from the summary note what a ⚠ means, so a bare mark next to the shot number
  // is enough.
  const outlierMark = [arm, bow, draw, elbow].some((v) => v.flagged) ? " ⚠" : "";
  const debugBit = DEBUG
    ? `<span class="shotlog-debug">hand sep ${e.handSep.toFixed(2)} — anchor ${e.anchorOk ? "ok" : "fail"} · arm-check ${e.armOk ? "ok" : "fail"} · sep-check ${e.sepOk ? "ok" : "fail"} · still ${e.stillOk ? "ok" : "fail"}</span>`
    : "";

  // A big, obvious watch button when this shot has a clip; a plain "no clip" note when it
  // doesn't (recording unsupported, or it failed for just this one shot) — never nothing, so a
  // missing clip never reads as a missing shot. data-shot carries the shot number for the click
  // handler on shotLogEl (see openClipPlayer wiring) to look the entry back up by.
  const clipBit = e.clipUrl
    ? `<button type="button" class="shotlog-play" data-shot="${e.shotNum}">▶ Watch</button>`
    : `<span class="shotlog-noclip">no clip</span>`;

  return `<div class="shotlog-row"><div class="shotlog-row-main">Shot ${e.shotNum}${outlierMark}<br>bow arm ${arm.html} · shoulders bow ${bow.html} / draw ${draw.html} · elbow ${elbow.html}${debugBit}</div><div class="shotlog-row-clip">${clipBit}</div></div>`;
}

// Plain-language shot log — this is what the owner actually reads, standing at the phone after
// their end, not mid-shot, so it can be a normal-sized list rather than the big blunt ?debug
// overlay. A summary block (own average + own spread, per measure) sits above the per-shot
// rows; a single note under it explains what the parenthesized numbers in every row below mean,
// so the rows themselves can stay compact — bare signed numbers, no repeated phrase. Extra
// per-shot detail (hand separation, the four trigger checks) only shows up when ?debug is on.
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

  if (log.length === 0) {
    shotLogEl.innerHTML = `${banner}${modelBit}<div class="shotlog-empty">No shots recorded yet — draw once and this fills in.</div>`;
    return;
  }

  const stats = summarizeShots(log);
  const summaryLines = [
    summaryLine("Bow arm", "°", stats.bowArm),
    summaryLine("Bow shoulder", "%", stats.shoulderBow),
    summaryLine("Draw shoulder", "%", stats.shoulderDraw),
    summaryLine("Elbow", "°", stats.elbow, fmtSignedDeg),
  ]
    .map((line) => `<div class="shotlog-summary-row">${line}</div>`)
    .join("");
  const note = `<div class="shotlog-summary-note">Numbers in parentheses below = that shot vs your session average above.</div>`;

  const rowsHtml = log.map((e) => renderShotRow(e, stats)).join("");

  shotLogEl.innerHTML = `${banner}${modelBit}<div class="shotlog-summary">${summaryLines}${note}</div>${rowsHtml}`;
}

// Draws the current camera frame into the overlay canvas. Used to be just ctx.clearRect, leaving
// the canvas transparent so the <video> element underneath showed through on its own — visually
// identical, but it meant the canvas itself had no picture in it, only skeleton lines. Now that
// the canvas is what gets recorded (see startClipRecording — canvas.captureStream is the only
// way to bake the skeleton into a clip), the canvas needs its own copy of the video frame every
// time, landmarks or not, so a clip is never missing frames just because the pose was briefly
// lost. canvas.width/height are set to the video's native resolution in startCamera, so this
// plain draw lines up exactly with no cropping or letterboxing needed. Always draws the RAW,
// unmirrored video frame — see withMirror below for where the flip actually happens; this
// function has no idea whether the current picture is mirrored or not, deliberately.
//
// This fully repaints canvas.width × canvas.height every call (clearRect then a drawImage that
// covers the same rectangle), so the <video> element underneath — which sits at the same CSS
// box (inset: 0, 100% × 100%) as this canvas — can never show through at an edge, even if the
// two elements' internal pixel dimensions differ (they can: canvas.width/height are the video's
// native capture resolution, while the CSS box both are stretched into is the on-screen viewport
// size). "Stretched into a differently-sized box" is not "gaps at the edges" — both elements
// always cover their entire box, just at different effective scales, so there is no seam for the
// unflipped video to leak through.
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

// Runs one frame's worth of canvas drawing (video frame, and the skeleton on top of it when
// there is one) inside a horizontal flip, when effectiveMirror says this frame should be
// mirrored. This is now the ONLY place mirroring happens in the whole app — replacing the old
// CSS `.mirrored` class on #video/#overlay, which only changed how the browser displayed those
// elements and never touched a single pixel. That mattered because clips are recorded straight
// off this canvas (canvas.captureStream, see startClipRecording): a CSS transform is invisible
// to a pixel-capture stream, so the old approach meant a mirrored on-screen view recorded an
// UNmirrored clip — the owner would watch back something that didn't match what he saw live.
// Doing the flip here instead means the canvas's own pixels are mirrored, so whatever the owner
// saw on screen is exactly what the recorder captured.
//
// The flip is pure presentation: it happens in ctx's transform, applied only to what gets drawn
// AFTER it, and is undone (ctx.restore) before this function returns. It never touches a
// landmark coordinate — bowArmAngleOf, shoulderDropSampleOf, drawElbowAlignmentOf, isAtFullDraw
// and trackShotAttempt all run on the same raw (smoothed) landmarks whether this frame is
// mirrored or not, so mirroring can never move a measured number or silently swap which arm the
// existing 🎯 handedness toggle is scoring. Video frame and skeleton are drawn inside the SAME
// save/restore pair (the caller passes both in one drawFn), not flipped separately, so the two
// can never end up mismatched by half a frame's worth of transform state.
function withMirror(drawFn) {
  const mirror = effectiveMirror(facingMode, mirrorToggled);
  ctx.save();
  if (mirror) {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }
  try {
    drawFn();
  } finally {
    ctx.restore();
  }
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

  const frameWidth = video.videoWidth;
  const frameHeight = video.videoHeight;

  // Region-of-interest cropping: if a crop box survived from last frame (and the feature isn't
  // switched off), draw just that box from the video into the small offscreen ROI canvas, scaled
  // up to fill it, and hand MediaPipe THAT instead of the whole video frame — same camera, same
  // distance, far more pixels of archer reaching the model. usedCropBox stays null (whole-frame
  // detection, exactly like before this feature existed) whenever cropping is off, there's no
  // box yet, or the video doesn't have real dimensions yet. See the ROI CROPPING constants and
  // runtime block above for the full reasoning.
  let usedCropBox = null;
  let detectionSource = video;
  if (ROI_CROPPING_ENABLED && currentCropBox && frameWidth && frameHeight) {
    usedCropBox = currentCropBox;
    roiCtx.drawImage(
      video,
      usedCropBox.x, usedCropBox.y, usedCropBox.size, usedCropBox.size,
      0, 0, ROI_CANVAS_SIZE, ROI_CANVAS_SIZE
    );
    detectionSource = roiCanvas;
  }

  const inferenceStart = performance.now();
  const result = poseLandmarker.detectForVideo(detectionSource, now);
  measurePoseModelPerf(performance.now() - inferenceStart, now);
  let rawLandmarks = result.landmarks?.[0];

  // The model just saw a close-up crop, so its landmarks came back in CROP-LOCAL normalised
  // coordinates (0-1 across the ROI canvas). Map them into full-frame normalised coordinates
  // (0-1 across the actual camera frame) right here, before anything else in this file ever sees
  // them — every readout, the shot log, and the skeleton drawing all assume full-frame
  // coordinates, and getting this wrong would silently corrupt every number the owner is tuning.
  if (rawLandmarks && usedCropBox) {
    rawLandmarks = rawLandmarks.map((lm) => mapCropLandmarkToFullFrame(lm, usedCropBox, frameWidth, frameHeight));
  }

  if (!rawLandmarks) {
    withMirror(drawVideoFrame); // keep the clip (and the on-screen view) showing the camera even without a skeleton
    setReadout(readoutBowArm, valueBowArm, "— uncertain", "uncertain");
    setValueState(valueShoulderBow, "—", "uncertain");
    setValueState(valueShoulderDraw, "—", "uncertain");
    setReadout(readoutElbow, valueElbow, "— uncertain", "uncertain");
    if (DEBUG) debugInfo = null;
    // Tracking just lost the archer entirely — whatever the filters were smoothing toward is now
    // stale. Reset so a fresh detection later starts clean rather than being dragged from
    // wherever the skeleton was last seen (see LandmarkSmoother).
    landmarkSmoother.reset();
    // Re-acquire on loss: whether this frame was cropped or not, no landmarks means whatever crop
    // box we had (if any) no longer contains the archer, or we never had one. Drop it so the very
    // next frame detects on the WHOLE frame again rather than continuing to stare into a box that
    // may no longer have anyone in it — a crop that can never find its way back would be exactly
    // the kind of hang the owner has no way to recover from (see CLAUDE.md).
    currentCropBox = null;
    endAttempt(); // pose lost mid-attempt counts as the attempt ending, same as hands relaxing
  } else {
    // Smoothed landmarks feed everything downstream — the skeleton drawing, all three readouts,
    // and the full-draw/shot-log sampling inside isAtFullDraw — so a shaky raw detection can't
    // show up in the numbers the owner reads later or the clip he watches back. Real elapsed time
    // (performance.now(), converted to seconds), not an assumed frame rate — see OneEuroFilter.
    // This runs on the MAPPED, full-frame coordinates above, never on crop-local ones — the
    // filter's whole job is smoothing real position over time, which only means something in a
    // coordinate system that doesn't itself change shape from frame to frame the way a moving
    // crop box would.
    const landmarks = landmarkSmoother.smooth(rawLandmarks, now / 1000);
    withMirror(() => drawSkeleton(landmarks));
    updateBowArmReadout(landmarks);
    updateShoulderDropReadout(landmarks);
    updateDrawElbowReadout(landmarks);
    // Return value intentionally unused here — isAtFullDraw's real job on every frame is its
    // side effect, calling trackShotAttempt (below) to feed the shot log. It used to also drive
    // the auto-freeze state machine, which read the true/false result; that machine is gone, but
    // the shot log still depends on this call happening every frame, so it stays.
    isAtFullDraw(landmarks, now);

    // Pick the crop box for NEXT frame from what was actually seen this frame (full-frame
    // coordinates, already mapped above if this frame itself was cropped). Recomputed from
    // scratch — not carried over — every frame cropping is on, so a body that walks away from
    // where it was is always followed; ROI_SMOOTHING (inside nextCropBox) is what stops that from
    // meaning the box chases ordinary landmark noise. Left null when cropping is switched off, so
    // the whole feature reduces to exactly the old whole-frame behaviour.
    currentCropBox = ROI_CROPPING_ENABLED
      ? nextCropBox(landmarks, frameWidth, frameHeight, usedCropBox)
      : null;
  }

  syncDebugOverlay();
}

function updateHandButtonLabel() {
  btnHand.textContent = rightHanded ? "🎯 Right-handed" : "🎯 Left-handed";
}

// Shows the CURRENT effective state, same convention as updateHandButtonLabel above — "what is
// true right now", not "what tapping this will do". The owner sets this up before walking away
// and can't come back to check it mid-shot, so at-a-glance current state is what matters.
function updateMirrorButtonLabel() {
  const mirrored = effectiveMirror(facingMode, mirrorToggled);
  btnMirror.textContent = mirrored ? "🪞 Mirrored" : "🪞 Not mirrored";
}

btnCamera.addEventListener("click", async () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  await startCamera();
  // Switching cameras changes the DEFAULT this toggle flips away from (see effectiveMirror), so
  // the label can change here even though the owner didn't touch the mirror button at all.
  updateMirrorButtonLabel();
});

btnHand.addEventListener("click", () => {
  rightHanded = !rightHanded;
  updateHandButtonLabel();
});

btnMirror.addEventListener("click", () => {
  mirrorToggled = !mirrorToggled;
  updateMirrorButtonLabel();
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
    updateMirrorButtonLabel();
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
  // frame; it ends and gets logged the moment separation drops back below that floor. Reset to
  // a clean slate here — the outer save/restore above puts it all back afterwards regardless.
  attempt = null;
  log = [];
  shotCount = 0;
  const sample = (handSep, extra = {}) => ({
    handSep,
    bowArmAngle: 178,
    elbowHeight: 3,
    anchorOk: true,
    armOk: true,
    sepOk: handSep >= FULL_DRAW_HAND_SEP_MIN,
    stillOk: true,
    ...extra,
  });

  trackShotAttempt(sample(0.1)); // resting, below the floor: no attempt yet
  console.assert(log.length === 0, "resting below the floor should not start an attempt");
  trackShotAttempt(sample(0.5)); // crosses the floor: attempt starts
  trackShotAttempt(sample(0.8));
  trackShotAttempt(sample(1.7)); // peak
  trackShotAttempt(sample(1.7)); // held steady — several frames at/near the peak
  trackShotAttempt(sample(1.6));
  console.assert(log.length === 0, "a hold still in progress must not be logged yet");
  console.assert(
    attempt && attempt.handSep === 1.7,
    "in-progress attempt should track its best (highest hand-sep) frame, not its latest"
  );
  trackShotAttempt(sample(0.05)); // hands come back together: attempt ends
  console.assert(log.length === 1, "one long hold must log exactly one row, not one per frame");
  console.assert(log[0].handSep === 1.7, "logged row should be the attempt's peak frame, not its last");
  console.assert(log[0].shotNum === 1, "first logged attempt should be shot 1");

  // A second, separate attempt must become its own row — including one that never gets close
  // to full draw at all, which is the exact case the log exists to capture.
  trackShotAttempt(sample(0.4));
  trackShotAttempt(sample(0.5)); // never reaches FULL_DRAW_HAND_SEP_MIN this time
  trackShotAttempt(sample(0.1)); // ends
  console.assert(log.length === 2, "a second attempt must log as its own row, not merge into the first");
  console.assert(
    log[0].shotNum === 2 && log[0].handSep === 0.5,
    "newest attempt should be first in the log, with its own (lower) peak — a near-miss is still recorded"
  );
  console.assert(log[1].shotNum === 1, "the earlier attempt should still be present, just not first");

  // Losing the pose entirely also ends whatever attempt was in progress (endAttempt, called
  // directly from renderLoop's !landmarks branch rather than through trackShotAttempt).
  trackShotAttempt(sample(0.6));
  endAttempt();
  console.assert(log.length === 3, "pose loss mid-attempt should still log it");
  console.assert(attempt === null, "ending an attempt must clear it so the next rise starts fresh");

  // Cap: only the newest SHOT_LOG_MAX entries are kept, still newest-first.
  for (let i = 0; i < SHOT_LOG_MAX + 3; i++) {
    trackShotAttempt(sample(0.5));
    trackShotAttempt(sample(0.05));
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
    for (let i = 0; i < SHOT_LOG_MAX; i++) {
      trackShotAttempt(sample(0.5));
      trackShotAttempt(sample(0.05)); // ends the attempt, logs it
    }
    console.assert(log.length === SHOT_LOG_MAX, "setup: log should be exactly full before the eviction check");
    const oldestUrl = "blob:fake-oldest-for-selftest";
    log[log.length - 1].clipUrl = oldestUrl; // pretend the row about to be evicted has a clip attached
    trackShotAttempt(sample(0.5));
    trackShotAttempt(sample(0.05)); // one more attempt: pushes the oldest row (the one just tagged) off the end
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

  // --- ROI cropping: pure geometry, no DOM/MediaPipe involved (same reasoning as the One Euro
  // tests above), so these run straight against the functions with plain fixture numbers. This
  // is the highest-risk part of the whole feature — get the crop-local <-> full-frame mapping
  // wrong and every angle the owner is calibrating gets silently corrupted — so it gets checked
  // directly here, not just eyeballed on a real camera.
  {
    // Mapping: a landmark at a KNOWN position inside a KNOWN crop box maps back to the correct
    // full-frame coordinate. The crop box here is deliberately off-centre (not the middle of the
    // frame) and not aligned to any nice fraction of it — a bug that only shows up for a crop
    // that isn't centred/aligned would sail straight through a test that only ever used one.
    const frameW1 = 1000, frameH1 = 800;
    const cropBox1 = { x: 200, y: 100, size: 300 }; // spans x 200-500, y 100-400 — well off-centre in an 800-tall frame
    const cropLm = { x: 0.25, y: 0.75, z: 0.1, visibility: 0.9 };
    const mapped = mapCropLandmarkToFullFrame(cropLm, cropBox1, frameW1, frameH1);
    // By hand: fullX = (200 + 0.25*300) / 1000 = 0.275; fullY = (100 + 0.75*300) / 800 = 0.40625
    console.assert(
      Math.abs(mapped.x - 0.275) < 1e-9 && Math.abs(mapped.y - 0.40625) < 1e-9,
      `a landmark at a known crop-local position should map to the known full-frame position, got (${mapped.x}, ${mapped.y})`
    );
    console.assert(
      Math.abs(mapped.z - 0.03) < 1e-9,
      `z should scale by the crop's size-to-frame-width ratio (300/1000 = 0.3), got ${mapped.z}`
    );
    console.assert(
      mapped.visibility === 0.9,
      "mapping crop-local to full-frame must pass visibility through untouched, same as LandmarkSmoother does for x/y"
    );

    // Round-trip: a full-frame point, taken crop-local (mapFullFrameToCropLocal) and mapped back
    // (mapCropLandmarkToFullFrame), must land back on the same point — the strongest available
    // check that the forward map isn't quietly shifting or losing precision. Different box, still
    // off-centre and not aligned to a round fraction of the frame, so this isn't just re-checking
    // the fixture above under a different name.
    const frameW2 = 1200, frameH2 = 900;
    const cropBox2 = { x: 50, y: 400, size: 250 };
    const fullPoint = { x: 0.63, y: 0.52, visibility: 1 };
    const cropLocal = mapFullFrameToCropLocal(fullPoint, cropBox2, frameW2, frameH2);
    const roundTripped = mapCropLandmarkToFullFrame(cropLocal, cropBox2, frameW2, frameH2);
    console.assert(
      Math.abs(roundTripped.x - fullPoint.x) < 1e-9 && Math.abs(roundTripped.y - fullPoint.y) < 1e-9,
      `full-frame -> crop-local -> full-frame should round-trip to the same point, got (${roundTripped.x}, ${roundTripped.y}) from (${fullPoint.x}, ${fullPoint.y})`
    );

    // Clamping + squareness: a body bounding box near the LEFT edge of the frame, padded out,
    // would want to extend past x=0 — it must instead slide back inside the frame (not shrink)
    // and stay exactly square, the same way a real archer standing near the edge of the shot
    // would be handled.
    const edgeBox = squareAndClampCropBox(0, 200, 40, 280, 640, 480, ROI_PADDING_FRACTION);
    console.assert(
      edgeBox.x >= 0 && edgeBox.y >= 0 && edgeBox.x + edgeBox.size <= 640 && edgeBox.y + edgeBox.size <= 480,
      `a crop box near the frame's edge must be clamped fully inside it, got x=${edgeBox.x} y=${edgeBox.y} size=${edgeBox.size} in a 640x480 frame`
    );
    console.assert(
      edgeBox.x === 0,
      `a box that would overhang the LEFT edge should slide flush against it (x=0), not shrink — got x=${edgeBox.x}`
    );
    // "Square" here just means one shared `size` drives both dimensions (see squareAndClampCropBox
    // — there is no separate width/height to disagree), so the real thing worth checking is that
    // clamping didn't accidentally shrink it into something degenerate.
    console.assert(edgeBox.size > 0, "a clamped crop box must still have a positive size, not collapse to nothing");

    // A body box bigger than the frame itself (padding pushed it past both dimensions) must be
    // capped to fit inside the frame entirely — never asked to crop something larger than the
    // source image.
    const oversizeBox = squareAndClampCropBox(10, 10, 260, 110, 300, 300, ROI_PADDING_FRACTION);
    console.assert(
      oversizeBox.size <= 300 && oversizeBox.x === 0 && oversizeBox.y === 0,
      `a box whose padded size exceeds the frame must be capped to fill the frame (x=0,y=0,size<=300), got x=${oversizeBox.x} y=${oversizeBox.y} size=${oversizeBox.size}`
    );

    // Loss of tracking resets to whole-frame detection: with too few confidently-visible
    // landmarks, boundingBoxOfLandmarks (and therefore nextCropBox) must return null — the
    // signal renderLoop uses to detect on the whole frame again next frame, rather than trust a
    // box built from a couple of stray points or keep cropping into a region that may no longer
    // have anyone in it.
    const frameW3 = 640, frameH3 = 480;
    const sparseLandmarks = [
      { x: 0.5, y: 0.5, visibility: 0.9 },
      { x: 0.5, y: 0.5, visibility: 0.9 },
      { x: 0.5, y: 0.5, visibility: 0.1 }, // below MIN_VISIBILITY — must not count
      { x: 0.5, y: 0.5, visibility: 0 },
    ];
    console.assert(
      boundingBoxOfLandmarks(sparseLandmarks, frameW3, frameH3) === null,
      "fewer than ROI_MIN_VISIBLE_LANDMARKS confidently-visible landmarks should yield no bounding box at all"
    );
    console.assert(
      nextCropBox(sparseLandmarks, frameW3, frameH3, { x: 100, y: 100, size: 200 }) === null,
      "losing confident tracking must return null (whole-frame detection next frame), even if a crop box was active going into this frame"
    );

    // Contrast case: enough confidently-visible landmarks spread across a real body pose DOES
    // produce a usable box — confirms the null results above are really about confidence/count,
    // not a bug that makes this always return null.
    const confidentLandmarks = [
      { x: 0.45, y: 0.30, visibility: 1 }, // head-ish
      { x: 0.40, y: 0.45, visibility: 1 }, // shoulder
      { x: 0.60, y: 0.45, visibility: 1 }, // other shoulder
      { x: 0.35, y: 0.60, visibility: 1 }, // hip
      { x: 0.65, y: 0.60, visibility: 1 }, // other hip
    ];
    const acquired = nextCropBox(confidentLandmarks, frameW3, frameH3, null);
    console.assert(
      acquired !== null && acquired.size > 0,
      "enough confidently-visible landmarks should produce a real crop box for next frame, not null"
    );

    // Hysteresis: smoothCropBox should ease toward a fresh box rather than snap straight to it —
    // the result must land strictly between the previous and fresh boxes (given a smoothing
    // factor strictly between 0 and 1), and exactly on the fresh box when there's no previous one
    // to ease from (first acquisition / just re-acquired).
    const prevSmoothBox = { x: 0, y: 0, size: 100 };
    const freshSmoothBox = { x: 100, y: 100, size: 200 };
    const eased = smoothCropBox(prevSmoothBox, freshSmoothBox, ROI_SMOOTHING);
    console.assert(
      eased.x > prevSmoothBox.x && eased.x < freshSmoothBox.x && eased.size > prevSmoothBox.size && eased.size < freshSmoothBox.size,
      "smoothing a crop box toward a fresh one should land strictly between the two, not jump straight to the fresh box"
    );
    const firstAcquisition = smoothCropBox(null, freshSmoothBox, ROI_SMOOTHING);
    console.assert(
      firstAcquisition.x === freshSmoothBox.x && firstAcquisition.size === freshSmoothBox.size,
      "with no previous box to ease from, smoothCropBox should use the fresh box outright"
    );
  }

  // --- Mirror toggle: effectiveMirror (and the defaultMirrorFor it's built on) are pure
  // functions of facingMode + the toggle, never touched by module state, so every combination
  // can be checked directly rather than by poking classList/DOM. This is also the exact table
  // the brief asked for: default state per camera, and what the toggle does to each.
  console.assert(defaultMirrorFor("user") === true, "front camera should default to mirrored");
  console.assert(defaultMirrorFor("environment") === false, "rear camera should default to unmirrored");

  console.assert(
    effectiveMirror("environment", false) === false,
    "rear camera, toggle off, should be unmirrored (matches its default)"
  );
  console.assert(
    effectiveMirror("environment", true) === true,
    "rear camera, toggle on, should flip to mirrored (away from its unmirrored default)"
  );
  console.assert(
    effectiveMirror("user", false) === true,
    "front camera, toggle off, should be mirrored (matches its default)"
  );
  console.assert(
    effectiveMirror("user", true) === false,
    "front camera, toggle on, should flip to unmirrored (away from its mirrored default)"
  );

  selfTestInProgress = false;
  rightHanded = savedHanded;
  lastDrawWrist = savedLastWrist;
  attempt = savedAttempt;
  log = savedLog;
  shotCount = savedShotCount;

  console.log("selfTest done — check above for any failed console.assert");
}

if (location.search.includes("selftest")) selfTest();

main();
