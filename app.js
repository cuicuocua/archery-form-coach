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
// ===========================================================================

// MediaPipe pose landmark indices (33-point model)
const L_SHOULDER = 11, R_SHOULDER = 12;
const L_ELBOW = 13, R_ELBOW = 14;
const L_WRIST = 15, R_WRIST = 16;
const L_HIP = 23, R_HIP = 24;

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

let poseLandmarker = null;
let stream = null;
let facingMode = "environment"; // rear camera first
let rightHanded = true;
let frozen = false;
let drawingUtils = null;

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

function drawSkeleton(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: "#00e5ff",
    lineWidth: 3,
  });
  drawingUtils.drawLandmarks(landmarks, { color: "#ffffff", radius: 4 });
}

function renderLoop() {
  requestAnimationFrame(renderLoop);
  if (frozen || !poseLandmarker || video.readyState < 2) return;

  const result = poseLandmarker.detectForVideo(video, performance.now());
  const landmarks = result.landmarks?.[0];

  if (!landmarks) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setReadout(readoutBowArm, valueBowArm, "— uncertain", "uncertain");
    setReadout(readoutElbow, valueElbow, "— uncertain", "uncertain");
    return;
  }

  drawSkeleton(landmarks);
  updateBowArmReadout(landmarks);
  updateDrawElbowReadout(landmarks);
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
  frozen = !frozen;
  btnFreeze.classList.toggle("active", frozen);
  btnFreeze.textContent = frozen ? "▶ Resume" : "⏸ Freeze";
  if (frozen) {
    video.pause();
  } else {
    video.play();
  }
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

main();
