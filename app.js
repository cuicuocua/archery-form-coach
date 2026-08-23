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
let shotCount = 0; // total attempts this session, keeps counting even once the log above fills up
let log = []; // newest first
let attempt = null; // the attempt currently in progress, if any — see trackShotAttempt below

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
    if (!attempt || sample.handSep >= attempt.handSep) {
      attempt = { ...sample };
    }
  } else {
    endAttempt();
  }
}

// Ends whatever attempt is in progress (if any) and logs it. Called when hand separation drops
// back to resting (from trackShotAttempt above) or when the pose is lost entirely mid-attempt
// (from renderLoop) — either way, whatever was going on has stopped.
function endAttempt() {
  if (!attempt) return;
  logShot(attempt);
  attempt = null;
}

function logShot(entry) {
  shotCount++;
  log.unshift({ ...entry, shotNum: shotCount });
  log = log.slice(0, SHOT_LOG_MAX);
  renderShotLog();
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
  return `<div class="shotlog-row">Shot ${e.shotNum}${outlierMark}<br>bow arm ${arm.html} · shoulders bow ${bow.html} / draw ${draw.html} · elbow ${elbow.html}${debugBit}</div>`;
}

// Plain-language shot log — this is what the owner actually reads, standing at the phone after
// their end, not mid-shot, so it can be a normal-sized list rather than the big blunt ?debug
// overlay. A summary block (own average + own spread, per measure) sits above the per-shot
// rows; a single note under it explains what the parenthesized numbers in every row below mean,
// so the rows themselves can stay compact — bare signed numbers, no repeated phrase. Extra
// per-shot detail (hand separation, the four trigger checks) only shows up when ?debug is on.
function renderShotLog() {
  if (log.length === 0) {
    shotLogEl.innerHTML = `<div class="shotlog-empty">No shots recorded yet — draw once and this fills in.</div>`;
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

  shotLogEl.innerHTML = `<div class="shotlog-summary">${summaryLines}${note}</div>${rowsHtml}`;
}

function drawSkeleton(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
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

  const result = poseLandmarker.detectForVideo(video, now);
  const landmarks = result.landmarks?.[0];

  if (!landmarks) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setReadout(readoutBowArm, valueBowArm, "— uncertain", "uncertain");
    setValueState(valueShoulderBow, "—", "uncertain");
    setValueState(valueShoulderDraw, "—", "uncertain");
    setReadout(readoutElbow, valueElbow, "— uncertain", "uncertain");
    if (DEBUG) debugInfo = null;
    endAttempt(); // pose lost mid-attempt counts as the attempt ending, same as hands relaxing
  } else {
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
renderShotLog(); // shows the "no shots yet" placeholder before the first one comes in

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

  rightHanded = savedHanded;
  lastDrawWrist = savedLastWrist;
  attempt = savedAttempt;
  log = savedLog;
  shotCount = savedShotCount;

  console.log("selfTest done — check above for any failed console.assert");
}

if (location.search.includes("selftest")) selfTest();

main();
