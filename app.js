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

// A shot's logged numbers are the MEDIAN of each measure across every eligible (settled — see
// PIPELINE SETTLING above) frame of the hold, computed independently per measure — not a single
// "best" frame any more. See the block comment above medianSampleOf below for why this replaced
// picking the frame with the highest hand separation: that rule turned out to be a biased sample,
// not just an arbitrary one. MEDIAN_SAMPLE_CAP bounds how many eligible frames one attempt keeps
// in memory — a stuck full-draw held near CLIP_MAX_MS (20s) at 60fps could otherwise see north of
// 1000 frames land in one attempt's array, an unbounded growth this constant rules out. Frames are
// kept via reservoir sampling (reservoirAdd below), not "first N seen" or "most recent N" — either
// of those would quietly bias the median toward one part of the hold (the raise-to-hold transition
// for "first N", late-session fatigue for "most recent N"); reservoir sampling gives every eligible
// frame of the WHOLE hold an equal chance of being kept, so the cap changes precision, never which
// part of the hold the median leans toward. 200 is comfortably above how many eligible frames a
// normal 1-4 second hold produces at a typical 20-60fps (so most real holds are never subsampled at
// all) while still bounding memory hard for the pathological case.
const MEDIAN_SAMPLE_CAP = 200;

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
const CLIP_STOP_TIMEOUT_MS = 4000; // how long finalizeRecording waits for MediaRecorder's onstop to actually fire after .stop() is called before giving up on it and cleaning up anyway (see resolveClipOutcome) — canvas-stream recording on some browsers has been seen to leave a recorder stuck mid-stop with onstop never firing at all, which without this bound would both leak that clip's capture track forever and leave its shot's row showing a bare "no clip" with no explanation
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

// ===== PIPELINE SETTLING — performance/precision knob, not calibration; safe to change without
// a coach, same family as SMOOTH_*/ROI_* above. Not about the archer at all: a statement about
// how long THIS PIPELINE itself takes to settle after it starts fresh, before a frame's numbers
// are trustworthy enough to become a shot's LOGGED sample.
//
// Right after landmarkSmoother resets (session start, tracking lost and re-found, camera
// switched) and right after the ROI crop box is first (re)established, both mechanisms are still
// catching up rather than reporting the archer's true position: One Euro's very first sample
// comes back completely unsmoothed (see OneEuroFilter.filter's tPrev===null branch), and — with
// cropping on — the very first frame(s) after a reset run whole-frame, at lower effective
// resolution, until a box exists. A shot logged from one of those frames is measured through a
// different pipeline than every later shot, corrupting exactly the shot-to-shot comparison the
// log exists to make. Caught in testing: on a synthetic body that never moved at all, the very
// first logged draw of a session read several degrees off from the rest of an otherwise
// identical session, purely from this — nothing to do with the archer, everything to do with
// measuring the first shot through a colder pipeline than the others.
//
// SETTLE_FRAMES_REQUIRED is sized off the SMOOTHING filter, the slower of the two mechanisms to
// settle: One Euro's time constant at SMOOTH_MIN_CUTOFF (1.0Hz — the calmest case, used when a
// joint is nearly still, exactly the condition here) is tau = 1/(2*pi*1.0) ~= 0.16s; three time
// constants (~0.48s, ~95% converged) is a reasonable settle target. At a modest ~20fps (this
// file already assumes a phone can run this slow — see MODEL_SLOW_FRAME_MS above), that target
// is about 10 frames; this constant sits well above that for margin, since (see below) the cost
// of waiting a little longer is nearly always zero.
//
// A crop box merely EXISTING is not the same as it having finished moving. smoothCropBox eases
// the box toward its target every frame (ROI_SMOOTHING above) rather than snapping to it, and
// the box's own size sets the scale everything gets resolved at — a landmark read through a
// still-shrinking-or-growing box is not on the same footing as one read through a settled box,
// even once SETTLE_FRAMES_REQUIRED frames have passed since the last reset. This matters most
// exactly at the moment a shot is worth logging: the box's fresh target jumps bigger the instant
// the archer's arms reach full extension, right as the raise turns into the hold — precisely
// when the highest-hand-separation frame (the one that gets scored) is likely to occur. Caught
// in testing: adding a deliberate multi-second pause on top of the frame-count gate alone still
// cut the residual bias on shoulder drop and elbow alignment by more than half, which a
// sufficient frame count on its own should have made no difference to.
//
// CROP_BOX_STABLE_MAX_DELTA is derived from ROI_SMOOTHING the same way SETTLE_FRAMES_REQUIRED is
// derived from SMOOTH_MIN_CUTOFF: each frame smoothCropBox closes (1 - ROI_SMOOTHING) = 65% of
// the remaining gap to a static target, so the residual gap after n frames is ROI_SMOOTHING^n of
// wherever it started. Even a full box-size jump (100% — roughly resting posture to a fully
// extended full-draw silhouette, about the largest real jump this app will ever see) decays
// below a 2% residual in n where 0.35^n <= 0.02, i.e. n >= 4 frames — fast, but real, and exactly
// the handful of frames right at the raise-to-hold transition that a frame-count gate alone
// cannot see. Below this fraction, frame-to-frame change is treated as the crop box's own
// ordinary steady-state jitter (landmark noise still moves the fresh target a little every
// frame, even standing still), not genuine settling in progress.
const SETTLE_FRAMES_REQUIRED = 30; // consecutive good-tracking frames needed after a reset before a frame's numbers are trusted enough to log. RAISE for more margin (safer, but the gate takes longer to clear after every reset — still normally invisible, see above); LOWER only with real measured convergence data in hand, not a guess
const CROP_BOX_STABLE_MAX_DELTA = 0.02; // the crop box's size AND position must each change by less than this fraction of its own size, frame to frame, to count as settled rather than still catching up — see the derivation above. RAISE if measurements still look inflated right after a raise-to-hold transition (rare box shapes may need more than 4 frames to fully decay); LOWER only with real measured box-jitter data in hand, not a guess — too low and ordinary steady-state box jitter (never truly zero) could block eligibility forever
// ===========================================================================

// ===== ROUTINE-START ATTENTION GATING — performance/battery knob, not calibration; safe to
// change without a coach, same family as ROI_*/SETTLE_*. Owner's own field report: "between
// shots I move around a lot even just to nock the next arrow and I'd like to contain the app
// functionality to only when I'm actually shooting." Right now the app runs the full pipeline
// (pose detection at full rate, plus everything downstream) continuously, whether he's mid-shot
// or nocking, turning to the quiver, or twenty feet away collecting arrows. This block lets the
// app run pose detection at a slower, cheaper rate whenever things look plainly calm (see
// attentionIsClearlyCalm below) and switch back to full rate the instant they don't — WITHOUT
// ever fully turning detection off, which is what makes recovery structural rather than
// something a threshold has to get right (see ATTENTION_GATING_ENABLED's own note, and the
// runtime block further down, for the "cannot latch off" guarantee in detail).
//
// THE OWNER'S EXPLICIT INSTRUCTION, and the one rule everything below is built around: when this
// detector is unsure, it must fail toward RECORDING, not toward idling. "A phantom row is
// visible and he can tell me about it; a missed arrow is invisible and he never knows." In code
// terms that becomes a deliberate asymmetry: allowing the app to go idle requires POSITIVE proof
// of calm (attentionIsClearlyCalm returning true, held continuously for ATTENTION_IDLE_AFTER_MS)
// — anything ambiguous keeps it engaged. Waking back up needs only the ABSENCE of that same
// proof on a single sample — anything ambiguous wakes it up. The two checks share the exact same
// function on purpose (see updateAttentionState): "should I stay idle" and "should I idle in the
// first place" are literally the same question, asked with opposite defaults baked into which
// side of the boolean each caller trusts.
//
// GEOMETRY-FIX NOTE (see the PM's brief this was built from): ATTENTION_REST_HAND_SEP_MAX and
// ATTENTION_REST_MOVE_MAX_PER_SEC are both torso-length-scaled distances, computed the exact same
// Math.hypot(...)/torsoLength(...) way as DRAW_ATTEMPT_MIN_SEP and FULL_DRAW_STILL_MAX already
// are elsewhere in this file — deliberately, so that when the in-progress fix to this app's
// coordinate maths lands (MediaPipe's x/y are normalised to frame WIDTH/HEIGHT separately, not
// one shared scale, so today's Euclidean distances are stretched on a non-square frame), these
// two constants inherit the correction automatically through the shared torsoLength() function,
// the same way DRAW_ATTEMPT_MIN_SEP will. They were NOT tuned against today's distorted numbers —
// they're set relative to DRAW_ATTEMPT_MIN_SEP/FULL_DRAW_STILL_MAX's own existing values instead
// (comfortably below/above them) specifically so they stay sensibly related after that fix, but
// whoever does that retuning should still sanity-check these two alongside it.
const ATTENTION_GATING_ENABLED = true; // master on/off switch. Set to false to go back to full-rate detection on every frame, exactly like before this feature existed — flip this first if the detector is ever suspected of causing trouble in the field. Structural safety net even while true: idle NEVER means "stopped detecting", only "detecting less often" (see the runtime block below) — so this switch changes cost, never correctness or recoverability
const ATTENTION_IDLE_SAMPLE_INTERVAL_MS = 150; // while idle, pose detection still runs this often — this (plus one frame's own processing time) is the WORST-CASE delay between the owner actually starting a routine and the app noticing and returning to full rate. LOWER for a faster reaction (less battery/heat saved); RAISE for more battery/heat saved (slower reaction) — keep it well under SHOT_MIN_DURATION_MS (600ms) so a genuine draw can never start and finish inside one blind gap; see the selfTest assertion enforcing that relationship
const ATTENTION_IDLE_AFTER_MS = 1500; // how long the calm condition below must hold, continuously, before the app allows itself to idle at all — a single calm-looking frame proves nothing (an archer pausing mid-routine for a second looks identical for one frame), so this requires a genuine held stretch of stillness, not just one lucky sample. RAISE to make the app more reluctant to idle (safer, more battery used); LOWER to idle sooner
const ATTENTION_REST_HAND_SEP_MAX = 0.2; // hand separation (torso-length fraction) at/below which the hands count as "relaxed" for the calm check — deliberately well BELOW DRAW_ATTEMPT_MIN_SEP (0.3, the floor that starts a tracked attempt), so hands can never be resting by this measure's standard while also being in the middle of a real draw attempt; see the invariant assertion in selfTest. RAISE cautiously (never above DRAW_ATTEMPT_MIN_SEP) if ordinary at-rest hand position sits higher than expected; LOWER for a stricter definition of "relaxed"
const ATTENTION_REST_MOVE_MAX_PER_SEC = 0.5; // how fast the body's reference point (hip midpoint, see bodyReferencePoint) may drift, in torso-lengths per second, and still count as "not walking/stepping". Ordinary standing sway is a small fraction of this; a real step or a walk toward/away from the line covers far more than a torso-length in a second. RAISE if ordinary standing sway is ever mistaken for movement (blocks idling, safe but wastes battery); LOWER if genuine walking is ever mistaken for standing still (also safe on its own — it only delays idling — but defeats the point of this feature if it happens often)
// ===========================================================================

// ===== STARTUP — timeouts, not calibration; safe to change without a coach. The owner cannot
// read a console or tap anything mid-session (see CLAUDE.md's "one interaction" rule), so an app
// that silently sits on "Starting camera…" forever is the worst failure mode there is for him —
// he can't tell "still loading" from "dead". These constants exist so startup can never hang
// silently: VIDEO_READY_TIMEOUT_MS bounds one internal wait (see waitForVideoReady), and
// STARTUP_WATCHDOG_MS/STARTUP_MODEL_WATCHDOG_MS together bound the whole of main() (see
// armStartupWatchdog) before the status text is replaced with a plain-English line naming
// whatever step didn't finish.
const VIDEO_READY_TIMEOUT_MS = 5000; // how long ONE wait attempt for the video's real width/height runs before re-arming rather than staying stuck on that one attempt forever (see the retry loop in startCamera) — bounds each attempt, not the overall wait; STARTUP_WATCHDOG_MS below is what bounds the overall wait and tells the owner if dimensions never show up at all
const STARTUP_WATCHDOG_MS = 15000; // if a startup step OTHER than the pose-model download (see STARTUP_MODEL_WATCHDOG_MS below) hasn't finished within this long, treat it as stuck and report it — see armStartupWatchdog/startupStuckMessage. Also doubles as the "this is taking a while" checkpoint for the model-download step itself: past this point on a slow connection the status text says so, calmly, without raising the alarm yet.
const STARTUP_MODEL_WATCHDOG_MS = 45000; // how long the pose-model DOWNLOAD step specifically gets before the watchdog gives up on it and raises the same alarm the other steps raise at STARTUP_WATCHDOG_MS. Field bug this fixes: the pose model is a multi-megabyte file (see POSE MODEL above), and on ordinary phone data at a shooting range, downloading it can genuinely take longer than STARTUP_WATCHDOG_MS while completely healthy — the owner saw exactly this: the "never finished loading" alarm fired, then cleared itself once the download landed and tracking started fine. A slow download is not the same failure as a step that will never finish; treating it as one made him restart the app several times, each restart throwing away whatever download progress the previous attempt had made. This is deliberately many times STARTUP_WATCHDOG_MS so an ordinary slow connection is never mistaken for a dead one — but it is still a bound, not a disabled watchdog: a model that truly never lands (bad URL, blocked host, no signal at all) is still caught and reported once this elapses.
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
// renderShotLog() writes into this inner scrolling element, never into shotLogEl itself — the
// close button lives outside it in the DOM (see index.html) specifically so a long log can never
// scroll it out of reach. See HANDOVER.md Stage 1a.
const shotLogContentEl = document.getElementById("shotlog-content");
const shotLogCloseBtn = document.getElementById("shotlog-close");
const clipPlayerEl = document.getElementById("clipplayer");
const clipPlayerVideo = document.getElementById("clipplayer-video");
const clipPlayerClose = document.getElementById("clipplayer-close");
const clipPlayerRateBtns = document.querySelectorAll(".clipplayer-rate");

// Whether this browser can record a clip at all. Checked once, up front, rather than
// discovering it the first time an attempt starts — that way the "clips unavailable" banner
// (see markClipsUnavailable) can go up before the owner ever shoots, not after their first shot
// quietly has no video.
//
// IMPORTANT LIMITATION, found investigating a field report where this read true, no banner ever
// showed, and yet neither of two real shots produced a clip: this is plain feature detection — it
// only proves the two functions EXIST, never that a real recording started through them will
// actually come back with usable video. That gap is real on iOS Safari specifically, where
// MediaRecorder-from-canvas.captureStream is a thin, quirky combination (as opposed to recording
// straight from a camera stream): a MIME type can report itself "supported" and still fail to mux,
// `ondataavailable` can fire nothing but empty chunks, and a recorder can simply never call
// `onstop`. None of that can be caught by asking two functions whether they exist — it only shows
// up once a real recording is actually attempted, which is now checked and explained per-attempt
// instead (see resolveClipOutcome / explainClipFailure further down) rather than assumed safe here.
const CLIP_SUPPORTED = typeof MediaRecorder !== "undefined" && typeof canvas.captureStream === "function";

let poseLandmarker = null;
let stream = null;
let facingMode = "environment"; // rear camera first
let rightHanded = true;
let drawingUtils = null;

// Which step of startup is currently in flight, and when it started — read by the startup
// watchdog (see STARTUP above) so its message can name the actual stuck step rather than a
// generic one, and so each step is judged against its OWN allowance rather than one deadline for
// the whole of main() (a slow step must not leave the next step looking instantly overdue the
// moment it begins). Only meaningful while main() is still running; set together, at each
// transition, by setStartupStep below — never assigned directly — and never read again once
// startup finishes.
let startupStep = "loading the pose model";
let stepStartedAt = 0;

function setStartupStep(step) {
  startupStep = step;
  stepStartedAt = performance.now();
}

// Set (see recordStartupProblem, further down) when startup stalls past STARTUP_WATCHDOG_MS or
// fails outright — the persistent, plain-English record of a startup failure that renderShotLog
// puts at the very top of the shot log, above even the clip-recording banner: if this is set,
// nothing else in the log can be trusted to mean much, since tracking may never have properly
// started this session. Cleared automatically if a watchdog-triggered stall goes on to recover on
// its own (see clearStartupProblem) — an eventual success must never leave this kind of residue
// behind for the owner to find later.
let startupProblem = null;

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
// row per draw attempt, whether or not it ever reached full draw — kept until the page reloads.
// Each measure on that row is the MEDIAN of that measure across every eligible frame of the
// hold, not a single "best" frame (see medianSampleOf's own block comment for why — a real,
// measured bias, not just a simplification). No timer anywhere in this: entries never expire or
// get overwritten just because time passed, only because a newer attempt bumps an old one out of
// the last SHOT_LOG_MAX.
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

// How many real draw attempts (deep and long enough — never counted in rejectedAttemptCount
// above) had every single frame land before the pipeline had settled (see PIPELINE SETTLING and
// endAttempt), so there was no honest reading to log. A DIFFERENT claim from rejectedAttemptCount
// on purpose: that one means "this wasn't a real draw"; this one means "this really was a draw,
// the app just couldn't get a settled enough look at it" — the owner needs to be able to tell the
// two apart, not read one combined "something got lost" number. See its own line in renderShotLog.
let unsettledAttemptCount = 0;

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

// Carries forward the reason a clip recording failed when it finished (or was forcibly given up
// on — see resolveClipOutcome) BEFORE the attempt it belonged to had actually logged a shot — the
// one case where the failure has no row yet to explain itself on. That only happens when the
// CLIP_MAX_MS safety cap cuts a stuck-in-progress recording off before endAttempt ever runs (see
// finalizeRecording's capTimer). The very next call to attachRecordingToShot (from the endAttempt
// that follows moments later) consumes this and writes it onto that shot's own row, so the owner
// still gets a specific reason instead of a bare, unexplained "no clip". Cleared the instant it's
// consumed; never carries over to a later, unrelated shot.
let pendingClipNote = null;

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

// ===== PIPELINE SETTLING — runtime state and the functions that implement the PIPELINE
// SETTLING constants above. See SETTLE_FRAMES_REQUIRED's and CROP_BOX_STABLE_MAX_DELTA's own
// comments for the full reasoning; this is just the bookkeeping.
let settledFrames = 0; // consecutive good-tracking frames seen since landmarkSmoother last reset
let prevUsedCropBox = null; // the crop box actually used on the PREVIOUS good-tracking frame — compared against THIS frame's own box below to tell "settled" apart from "merely present"

// Pure geometry: has the crop box stopped meaningfully moving/resizing between two consecutive
// frames? No prior box to compare against (the very first frame after acquisition, or no box at
// all) can never count as stable — there is nothing yet to prove it has stopped changing. Pure —
// no DOM, no module state — so selfTest can drive it directly with fixture boxes.
function cropBoxIsStable(box, prevBox) {
  if (!box || !prevBox) return false;
  const sizeDelta = Math.abs(box.size - prevBox.size) / prevBox.size;
  const posDelta = Math.hypot(box.x - prevBox.x, box.y - prevBox.y) / prevBox.size;
  return sizeDelta <= CROP_BOX_STABLE_MAX_DELTA && posDelta <= CROP_BOX_STABLE_MAX_DELTA;
}

// Called once per frame that has valid landmarks (never on pose loss — a lost frame has no
// numbers to be trustworthy about, and ends whatever attempt was in progress anyway). Advances
// the settling counter and reports whether THIS frame's own numbers are trustworthy enough to
// log: takes the crop state as plain values rather than reading module state directly, so
// selfTest can drive it deterministically without a real camera or crop box.
function advanceSettling(cropBoxUsedThisFrame, cropBoxStableThisFrame) {
  settledFrames++;
  return (
    settledFrames >= SETTLE_FRAMES_REQUIRED &&
    (!ROI_CROPPING_ENABLED || (cropBoxUsedThisFrame && cropBoxStableThisFrame))
  );
}

// Re-arms the gate. Called from every place landmarkSmoother.reset() is called (pose lost,
// camera switched) — same reset points, same reasoning: a frame right after either of those is
// exactly as untrustworthy as one right after startup, and must be re-gated the same way. Also
// drops prevUsedCropBox — a box from before the reset must never be compared against a fresh one
// after it, which would falsely read as "stable" when nothing has actually settled.
function resetSettling() {
  settledFrames = 0;
  prevUsedCropBox = null;
}
// ===========================================================================

// ===== ROUTINE-START ATTENTION GATING — runtime state and the logic behind the constants above.
// See that block's own comment for the full reasoning; this is the mechanism.

let attentionEngaged = true; // true = full rate (every frame gets pose detection); false = idle (throttled, see ATTENTION_IDLE_SAMPLE_INTERVAL_MS in renderLoop). Starts true: fail toward recording from the very first frame, same as everything else in this feature — the owner may start shooting right after the page loads, and there is no "calm" evidence yet either way to justify starting idle
let attentionCalmSinceMs = null; // wall-clock time the CURRENT unbroken streak of "clearly calm" samples began, or null if the most recent sample wasn't calm. Only meaningful while engaged — see updateAttentionState
let attentionLastIdleSampleMs = null; // performance.now() of the last frame that actually ran detection while idle — renderLoop throttles against this, not a frame counter, so the pacing is real elapsed time regardless of the device's actual frame rate
let attentionLastEvalMs = null; // performance.now() of the last sample actually fed to attentionIsClearlyCalm (engaged frame or idle sample alike) — the basis for the body-stillness speed calculation below, which needs real elapsed time between two ACTUAL samples, not between two rAF ticks (most of which are skipped while idle)
let attentionPrevRef = null; // bodyReferencePoint() from the last actual sample — compared against this frame's own to judge "not walking", same hysteresis-free instantaneous-speed approach isAtFullDraw already uses for wrist stillness
// Transparency counters — the owner can't watch this decision happen live any more than any of
// the other automatic decisions in this file, so what it did has to be readable afterwards, in
// its own words, distinguishable from rejectedAttemptCount/unsettledAttemptCount (see
// renderShotLog): those two are about attempts that WERE tracked and then judged; these two are
// about the attention layer's own behaviour, never reset mid-session, same treatment as every
// other running total in this file.
let attentionIdlePeriods = 0; // how many times this session the app allowed itself to idle between shots
let attentionLateWakeCount = 0; // how many of those idle periods ended on a sample where hand separation was ALREADY past DRAW_ATTEMPT_MIN_SEP — meaning the real start of that movement happened sometime during the idle gap just slept through, not on the exact frame that noticed it. See its own comment at the increment site below for what this does and doesn't mean

// Two wrists' separation, scaled by torso length — the same "how far apart are the hands" signal
// isAtFullDraw uses internally (see FULL_DRAW_HAND_SEP_MIN/DRAW_ATTEMPT_MIN_SEP above), but kept
// as its own small function here rather than reaching into isAtFullDraw's own computation, which
// is being reworked by a different engineer in parallel right now (the geometry-maths fix) — this
// stays deliberately independent so the two changes land on different lines and merge cleanly.
// Same "never guess" convention as the rest of the file: null if either wrist isn't confidently
// visible, or there's no usable torso-length scale reference.
function handSeparationForAttention(landmarks) {
  const bowWrist = rightHanded ? L_WRIST : R_WRIST;
  const drawWrist = rightHanded ? R_WRIST : L_WRIST;
  if (!visible(landmarks, bowWrist) || !visible(landmarks, drawWrist)) return null;
  const scale = attentionScale(landmarks);
  if (!scale) return null;
  const a = landmarks[bowWrist];
  const b = landmarks[drawWrist];
  return Math.hypot(a.x - b.x, a.y - b.y) / scale;
}

// Shared torso-length scale reference for this block — draw side preferred, bow side as
// fallback, the same convention used everywhere else in this file (isAtFullDraw, shoulderDropOf).
function attentionScale(landmarks) {
  const drawShoulder = rightHanded ? R_SHOULDER : L_SHOULDER;
  const drawHip = rightHanded ? R_HIP : L_HIP;
  const bowShoulder = rightHanded ? L_SHOULDER : R_SHOULDER;
  const bowHip = rightHanded ? L_HIP : R_HIP;
  return torsoLength(landmarks, drawShoulder, drawHip) ?? torsoLength(landmarks, bowShoulder, bowHip);
}

// The midpoint between the two hips — a stable whole-body reference point that isn't a hand and
// doesn't itself move as part of a normal draw, used to tell "standing settled" apart from
// "walking/stepping" (see ATTENTION_REST_MOVE_MAX_PER_SEC). Null if either hip isn't confidently
// visible — same "never guess" convention as the rest of the file.
function bodyReferencePoint(landmarks) {
  if (!visible(landmarks, L_HIP) || !visible(landmarks, R_HIP)) return null;
  return {
    x: (landmarks[L_HIP].x + landmarks[R_HIP].x) / 2,
    y: (landmarks[L_HIP].y + landmarks[R_HIP].y) / 2,
  };
}

// THE core signal, and the one function both directions of the state machine below share (see
// the constants block's own comment for why sharing it is the whole point). Pure — no module
// state read or written — so selfTest can drive it directly with fixture landmarks. Returns
// whether THIS ONE sample looks clearly calm: no landmarks at all (nobody there to be shooting —
// unambiguous), or landmarks present with both hands confidently relaxed together AND (when
// there's a previous sample to compare against) the body not stepping/walking. Anything else —
// a wrist not visible, no usable scale, hands apart, or the body moving — returns false. This is
// deliberately NOT symmetric with "is a draw happening": it only ever answers "is there positive
// proof of calm", which is exactly the bar fail-toward-recording needs on both ends (see
// updateAttentionState): allowing idle requires this to be true; staying idle requires it to
// keep being true; anything else, on either end, defaults toward engaged.
function attentionIsClearlyCalm(landmarks, prevRef, dtSec) {
  if (!landmarks) return true;
  const handSep = handSeparationForAttention(landmarks);
  if (handSep === null || handSep > ATTENTION_REST_HAND_SEP_MAX) return false;
  const ref = bodyReferencePoint(landmarks);
  if (ref && prevRef && dtSec > 0) {
    const scale = attentionScale(landmarks);
    if (!scale) return false;
    const speed = Math.hypot(ref.x - prevRef.x, ref.y - prevRef.y) / scale / dtSec;
    if (speed > ATTENTION_REST_MOVE_MAX_PER_SEC) return false;
  }
  return true;
}

// Runs once per sample (every engaged frame, or one idle-rate sample while idle — see
// renderLoop) and updates attentionEngaged for the FRAMES THAT FOLLOW. Deliberately called with
// RAW (crop-mapped, pre-smoothing) landmarks, never the smoothed ones — this decision needs to
// react as fast as possible, and One Euro's whole job is adding a little lag in exchange for
// steadiness, which is exactly what this must NOT have.
//
// `gatingEnabled` and `modelReady` default to the module constant/flag they mirror, but are
// threaded through explicitly (like `nowMs`/`frameEligible` already are elsewhere in this file)
// so selfTest can prove the master-switch and warm-up-guard behaviour directly, without needing
// a mutable module-level constant or having to actually run the pose-model warm-up first.
//
// PIPELINE SETTLING, worked through as the brief asked: the moment this function decides to
// re-engage (idle -> engaged), it resets landmarkSmoother, the settling counter, and the ROI crop
// box — the exact same three things every other recovery point in this file resets (session
// start, tracking lost, camera switched; see PIPELINE SETTLING above) — and it does so BEFORE
// returning, so the render loop's own smoothing/settling/isAtFullDraw calls for THIS SAME frame
// run against the fresh state, not the frame after. That ordering matters: if the reset happened
// one frame later, this frame's own advanceSettling() call could read a settledFrames count left
// over from a streak that ended seconds ago (whatever it was before the app went idle), letting a
// frame right after a long idle gap read as "already settled" — exactly the bias the PIPELINE
// SETTLING work already fixed once, in a new shape. Going idle itself does NOT reset anything:
// there is nothing worth resetting FOR while idle (attempt is null by construction the entire
// time — see the hard rule below — so nothing idle samples see ever reaches the shot log), and
// resetting on the way in would just mean paying the same reset twice for one idle stretch.
function updateAttentionState(nowMs, landmarks, gatingEnabled = ATTENTION_GATING_ENABLED, modelReady = modelDecisionMade) {
  if (!gatingEnabled) {
    attentionEngaged = true;
    return;
  }
  // The one-time pose-model warm-up measurement (see POSE MODEL above) needs every one of its
  // frames spaced at the pipeline's real full-rate cadence to mean anything — an idle throttle
  // landing mid-measurement would stretch the window's wall-clock length without changing the
  // frame count, quietly deflating the reported "rendered fps" without that being a real
  // slowdown. The owner is very likely standing calmly at exactly this moment (still setting the
  // phone up), which makes this a real, not just theoretical, collision. So: no idling at all
  // until that one-time decision has been made.
  if (!modelReady) {
    attentionEngaged = true;
    return;
  }

  const dtSec = attentionLastEvalMs === null ? 0 : (nowMs - attentionLastEvalMs) / 1000;
  const calm = attentionIsClearlyCalm(landmarks, attentionPrevRef, dtSec);
  attentionPrevRef = landmarks ? bodyReferencePoint(landmarks) : null;
  attentionLastEvalMs = nowMs;

  // Hard rule, checked first, that nothing below is allowed to override: an attempt in progress
  // must never see the pipeline idle out from under it. In practice this can never actually fire
  // — ATTENTION_REST_HAND_SEP_MAX sits below DRAW_ATTEMPT_MIN_SEP by construction (see that
  // constant's own comment and the invariant selfTest checks), so hands cannot be "calm" by this
  // function's own standard while an attempt is open — but it stays here, explicit, as a second
  // independent guarantee that doesn't rely on remembering that numeric relationship correctly
  // forever, and it also documents the intent directly rather than leaving it implicit.
  if (attempt) {
    attentionCalmSinceMs = null;
    attentionEngaged = true;
    return;
  }

  if (attentionEngaged) {
    if (!calm) {
      attentionCalmSinceMs = null;
      return; // stays engaged
    }
    if (attentionCalmSinceMs === null) attentionCalmSinceMs = nowMs;
    if (nowMs - attentionCalmSinceMs >= ATTENTION_IDLE_AFTER_MS) {
      attentionEngaged = false;
      attentionIdlePeriods++;
    }
    return;
  }

  // Currently idle. Fail-toward-recording in its most literal form: the instant this sample is
  // NOT clearly calm any more, engage — no second confirming sample, no cooldown, nothing else
  // has to also be true. A landmarks-null sample (still nobody there) or a relaxed-and-still
  // sample keeps it idle; anything else at all wakes it up.
  if (!calm) {
    attentionEngaged = true;
    attentionCalmSinceMs = null;
    // If hands are ALREADY past the attempt-start floor on the very sample that woke this up,
    // the real start of the movement happened sometime during the idle gap just slept through,
    // not on this exact sample — the ATTENTION_IDLE_SAMPLE_INTERVAL_MS worst case made real. This
    // does NOT mean the shot was missed: trackShotAttempt still starts a fresh attempt this same
    // frame and measures forward from here exactly as it always does; it means the small window
    // documented above (see the constants block's GEOMETRY-FIX note and ATTENTION_IDLE_SAMPLE_
    // INTERVAL_MS's own comment) applied this time, so this attempt's very first instant, and any
    // clip recording it, may start a beat later than the real movement did. Counted separately
    // from rejectedAttemptCount/unsettledAttemptCount on purpose — this is neither "that wasn't a
    // real draw" nor "the pipeline wasn't settled"; it's "the attention layer noticed a moment
    // later than an always-on pipeline would have."
    const wokeHandSep = landmarks ? handSeparationForAttention(landmarks) : null;
    if (wokeHandSep !== null && wokeHandSep >= DRAW_ATTEMPT_MIN_SEP) attentionLateWakeCount++;
    // Re-engaging is exactly like every other recovery point in PIPELINE SETTLING (session
    // start, tracking lost, camera switched) — see this function's own top comment for why the
    // reset happens HERE, synchronously, before this same frame's landmarks get smoothed/settled.
    landmarkSmoother.reset();
    resetSettling();
    currentCropBox = null;
  }
}
// ===========================================================================

async function startCamera() {
  // A landmark position smoothed from BEFORE a camera switch (different framing, possibly a
  // mirrored front camera) must never blend into positions AFTER it — that would drag the
  // skeleton across the frame on the very first frames of the new camera. See LandmarkSmoother.
  landmarkSmoother.reset();
  resetSettling(); // both mechanisms are starting fresh — the next frames are exactly as unsettled as at session start, see PIPELINE SETTLING above
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

  // Call play() because some browsers need it, but never await it. The <video> element already
  // carries autoplay/playsinline/muted (see index.html), so the picture showing up does not
  // depend on this promise settling — and on iOS Safari it sometimes doesn't: field bug, this
  // await used to hang here forever ("Starting camera…" stuck, picture visible, no skeleton, see
  // CLAUDE.md/README). A rejection here isn't a real error either — swallow it rather than let it
  // fall into main()'s catch block and report a false "Error:" for a camera that is actually fine.
  video.play().catch(() => {});

  setStartupStep("waiting for the camera picture");
  // Each single call is bounded (VIDEO_READY_TIMEOUT_MS) so this can never hang the way `await
  // video.play()` used to — but a video that STILL has no real dimensions after one bounded wait
  // must not be treated as ready either, or startCamera() would return "successfully" with a 0×0
  // canvas and nothing to detect on, and main() below would hide the status text and declare
  // victory over a camera that never actually sent a picture. So: keep re-waiting (each attempt
  // still bounded) until real dimensions show up. This never spins hot — each iteration is just
  // re-arming one "loadedmetadata" listener and a timer — and if dimensions genuinely never
  // arrive, it keeps main() from ever reaching its success path, which is exactly what leaves
  // STARTUP_WATCHDOG_MS (see main()) as the one honest backstop that tells the owner instead of
  // the app quietly pretending to have started.
  while (!(video.videoWidth > 0 && video.videoHeight > 0)) {
    await waitForVideoReady();
  }

  // No CSS mirroring here any more — see effectiveMirror/withMirror below for why. The <video>
  // element itself is left completely alone (never flipped, never classed); the canvas painted
  // on top of it is opaque every frame (see drawVideoFrame) and is what actually gets mirrored,
  // in its pixels, when that's called for.
  sizeCanvasToVideo();
}

// Resolves once the video element has real pixel dimensions, OR VIDEO_READY_TIMEOUT_MS has passed
// without them — whichever comes first. This is what replaced `await video.play()` as the thing
// startCamera() actually waits on: real dimensions are what canvas sizing and detection need, and
// — unlike play()'s promise — a single call here can never hang, so a slow-to-report camera can't
// wedge this the way play() did. It does NOT by itself mean the video is ready: the caller
// (startCamera(), above) loops on this until dimensions are real, so a timeout here just means
// "check again" rather than "give up" — see that loop's own comment for why giving up here would
// be the wrong call. sizeCanvasToVideo's standing listener (below) is a second, independent path
// to the same outcome, for the camera-switch case and any other caller that isn't looping on this.
function waitForVideoReady() {
  if (video.videoWidth > 0 && video.videoHeight > 0) return Promise.resolve();
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      video.removeEventListener("loadedmetadata", onMeta);
      clearTimeout(timer);
      resolve();
    };
    const onMeta = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) finish();
    };
    video.addEventListener("loadedmetadata", onMeta);
    const timer = setTimeout(finish, VIDEO_READY_TIMEOUT_MS);
  });
}

// Keeps the overlay canvas sized to match the video frame. Called directly once startCamera()
// has finished waiting above, AND wired as a standing listener (right below, attached once at
// module scope) for the rare case dimensions arrive only after VIDEO_READY_TIMEOUT_MS gave up
// waiting — without that second path a slow-to-report camera would leave the canvas at 0×0
// forever, and nothing would ever draw even once tracking does start.
function sizeCanvasToVideo() {
  if (video.videoWidth > 0 && video.videoHeight > 0) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}
video.addEventListener("loadedmetadata", sizeCanvasToVideo); // module-scope, attached once — covers first startup, every later camera switch, and late-arriving dimensions alike

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

// MediaPipe hands back x and y normalised INDEPENDENTLY on each axis — x is a fraction of the
// video frame's WIDTH, y a fraction of its HEIGHT. Those two fractions only mean the same
// physical distance when the frame is square. An iPhone held upright records portrait video
// (something like 720 wide by 1280 tall), so a step of 0.1 in x is a much shorter real-world
// distance than a step of 0.1 in y — yet every distance and angle in this file was, until this
// fix, computed with plain Math.hypot/atan-style vector maths directly on those raw x/y values,
// silently stretching every measurement along whichever axis happens to be the frame's narrower
// one. Multiplying x by the frame's pixel WIDTH and y by its pixel HEIGHT turns both axes back
// into the same units (video pixels) before any geometry happens, so a physically straight arm
// reads as straight and a real distance compares honestly to another real distance, regardless
// of whether the phone is held upright or sideways. This is a GEOMETRY-ONLY conversion: nothing
// that ends up on screen or in a recorded clip may use it (see mapCropLandmarkToFullFrame's own
// comment) — drawing stays in MediaPipe's normalised space throughout, exactly as before.
function toPixelSpace(lm, frameWidth, frameHeight) {
  return { x: lm.x * frameWidth, y: lm.y * frameHeight, visibility: lm.visibility };
}

function angleAt(a, b, c) {
  // angle at point b, between rays b->a and b->c, in degrees. Pure vector maths — agnostic to
  // whatever coordinate space a/b/c are already in, so callers are responsible for passing
  // points already converted to a physically-honest space (see toPixelSpace above) when the
  // frame isn't square. Never called directly on raw normalised landmarks for that reason.
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
// frameWidth/frameHeight are the video's own pixel dimensions — required so the shoulder/elbow/
// wrist points can be converted to a physically-honest space (see toPixelSpace) before angleAt
// measures the angle between them; skipping this would report a distorted angle on any
// non-square frame (i.e. basically every phone video).
function bowArmAngleOf(landmarks, frameWidth, frameHeight) {
  const bowShoulder = rightHanded ? L_SHOULDER : R_SHOULDER;
  const bowElbow = rightHanded ? L_ELBOW : R_ELBOW;
  const bowWrist = rightHanded ? L_WRIST : R_WRIST;
  if (![bowShoulder, bowElbow, bowWrist].every((i) => visible(landmarks, i))) return null;
  return angleAt(
    toPixelSpace(landmarks[bowShoulder], frameWidth, frameHeight),
    toPixelSpace(landmarks[bowElbow], frameWidth, frameHeight),
    toPixelSpace(landmarks[bowWrist], frameWidth, frameHeight)
  );
}

function updateBowArmReadout(landmarks, frameWidth, frameHeight) {
  const angle = bowArmAngleOf(landmarks, frameWidth, frameHeight);
  if (angle === null) {
    setReadout(readoutBowArm, valueBowArm, "— uncertain", "uncertain");
    return;
  }
  const ok = angle >= BOW_ARM_ANGLE_MIN && angle <= BOW_ARM_ANGLE_MAX;
  setReadout(readoutBowArm, valueBowArm, `${Math.round(angle)}°`, ok ? "ok" : "warn");
}

// The scale reference every ratio-based measure in this file divides by. Converted to pixel
// space (see toPixelSpace) before the hypot — shoulder-to-hip is close to purely vertical in a
// side-on view, but "close to" isn't "exactly", and this must be correct even for an archer who
// isn't perfectly upright, so the conversion is never skipped just because it would often cancel
// out anyway.
function torsoLength(landmarks, shoulderIdx, hipIdx, frameWidth, frameHeight) {
  if (!visible(landmarks, shoulderIdx) || !visible(landmarks, hipIdx)) return null;
  const s = toPixelSpace(landmarks[shoulderIdx], frameWidth, frameHeight);
  const h = toPixelSpace(landmarks[hipIdx], frameWidth, frameHeight);
  return Math.hypot(s.x - h.x, s.y - h.y);
}

// Shoulder drop for one shoulder: the vertical gap between that shoulder and its ear,
// normalised by torso length and given as a percentage — bigger number = shoulder sits
// further from the ear = more dropped, which is what "dropping my shoulders more" means.
// Reported per shoulder (not averaged, see updateShoulderDropReadout below) because the
// common compound fault is one shoulder — usually the bow shoulder, under load — creeping up
// while the other stays fine; an average would hide exactly that.
function shoulderDropOf(landmarks, shoulderIdx, sameEarIdx, otherEarIdx, ownHipIdx, otherShoulderIdx, otherHipIdx, frameWidth, frameHeight) {
  if (!visible(landmarks, shoulderIdx)) return null;

  // Side-on framing often means the far ear is occluded or low-confidence. Prefer the ear on
  // the same side as the shoulder being measured, but fall back to the other one — for a
  // purely vertical gap, which ear supplies the y-coordinate matters far less than having one.
  const earIdx = visible(landmarks, sameEarIdx) ? sameEarIdx : visible(landmarks, otherEarIdx) ? otherEarIdx : null;
  if (earIdx === null) return null;

  // Same "own side preferred, other side as fallback" torso-length convention used everywhere
  // else in this file.
  const scale =
    torsoLength(landmarks, shoulderIdx, ownHipIdx, frameWidth, frameHeight) ??
    torsoLength(landmarks, otherShoulderIdx, otherHipIdx, frameWidth, frameHeight);
  if (!scale) return null;

  // Converted to pixel space (see toPixelSpace) before the subtraction, same as everywhere else
  // — the ear-to-shoulder gap is close to purely vertical in a side-on view, but not exactly,
  // and this needs to stay correct for an archer who's leaning or turned slightly too.
  const shoulder = toPixelSpace(landmarks[shoulderIdx], frameWidth, frameHeight);
  const ear = toPixelSpace(landmarks[earIdx], frameWidth, frameHeight);
  // Image y grows downward, so the ear normally sits above the shoulder (smaller y). That gap
  // shrinks as the shoulder shrugs up toward the ear, and grows as it drops away from it.
  return ((shoulder.y - ear.y) / scale) * 100;
}

// Both shoulders' drop in one call, so the readout and the shot log stay in sync using exactly
// the same numbers.
function shoulderDropSampleOf(landmarks, frameWidth, frameHeight) {
  const bowShoulder = rightHanded ? L_SHOULDER : R_SHOULDER;
  const bowHip = rightHanded ? L_HIP : R_HIP;
  const bowEar = rightHanded ? L_EAR : R_EAR;
  const drawShoulder = rightHanded ? R_SHOULDER : L_SHOULDER;
  const drawHip = rightHanded ? R_HIP : L_HIP;
  const drawEar = rightHanded ? R_EAR : L_EAR;

  return {
    bow: shoulderDropOf(landmarks, bowShoulder, bowEar, drawEar, bowHip, drawShoulder, drawHip, frameWidth, frameHeight),
    draw: shoulderDropOf(landmarks, drawShoulder, drawEar, bowEar, drawHip, bowShoulder, bowHip, frameWidth, frameHeight),
  };
}

function updateShoulderDropReadout(landmarks, frameWidth, frameHeight) {
  const { bow, draw } = shoulderDropSampleOf(landmarks, frameWidth, frameHeight);
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
function drawElbowAlignmentOf(landmarks, frameWidth, frameHeight) {
  const drawWrist = rightHanded ? R_WRIST : L_WRIST;
  const drawElbow = rightHanded ? R_ELBOW : L_ELBOW;
  const bowWrist = rightHanded ? L_WRIST : R_WRIST;

  if (![drawWrist, drawElbow, bowWrist].every((i) => visible(landmarks, i))) return null;

  // Converted to pixel space up front (see toPixelSpace) — both the angle below AND the
  // high/low line-interpolation further down mix x and y together, so both need every point in
  // the same physically-honest units, not just the angle.
  const wrist = toPixelSpace(landmarks[drawWrist], frameWidth, frameHeight);
  const elbow = toPixelSpace(landmarks[drawElbow], frameWidth, frameHeight);
  const bow = toPixelSpace(landmarks[bowWrist], frameWidth, frameHeight);

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
  // near-vertical line: can't tell high from low this way. wrist/bow are pixel-space now (see
  // above), not the [0,1] normalised fractions this epsilon was originally written against, so
  // the threshold scales with frameWidth to mean the same tiny fraction-of-frame it always did.
  if (Math.abs(dx) < 1e-6 * frameWidth) return null;

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

function updateDrawElbowReadout(landmarks, frameWidth, frameHeight) {
  const result = drawElbowAlignmentOf(landmarks, frameWidth, frameHeight);
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
// frameEligible: whether THIS frame's own reading is settled enough to become a shot's logged
// sample (see PIPELINE SETTLING above) — required, not optional, so every caller has to make a
// deliberate choice rather than accidentally defaulting to "trust everything".
// frameWidth/frameHeight: the video's own pixel dimensions, needed so every distance below can
// be computed in a physically-honest space (see toPixelSpace) rather than raw normalised x/y,
// which stretch distances along whichever axis the frame happens to be narrower on.
function isAtFullDraw(landmarks, nowMs, frameEligible, frameWidth, frameHeight) {
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

  let anchorNorm;
  if (visible(landmarks, MOUTH_L) && visible(landmarks, MOUTH_R)) {
    anchorNorm = {
      x: (landmarks[MOUTH_L].x + landmarks[MOUTH_R].x) / 2,
      y: (landmarks[MOUTH_L].y + landmarks[MOUTH_R].y) / 2,
    };
  } else if (visible(landmarks, NOSE)) {
    anchorNorm = landmarks[NOSE];
  } else {
    return false;
  }
  // anchorNorm above is built by averaging two normalised points (still normalised, fine to mix
  // before converting) or taken straight from a landmark; either way it's still in MediaPipe's
  // normalised space at this point, so it gets the same toPixelSpace treatment as everything
  // else below before any distance touches it.
  const anchor = toPixelSpace(anchorNorm, frameWidth, frameHeight);

  const scale =
    torsoLength(landmarks, drawShoulder, drawHip, frameWidth, frameHeight) ??
    torsoLength(landmarks, bowShoulder, bowHip, frameWidth, frameHeight);
  if (!scale) return false;

  const wrist = toPixelSpace(landmarks[drawWrist], frameWidth, frameHeight);
  const bowWristPos = toPixelSpace(landmarks[bowWrist], frameWidth, frameHeight);
  const anchorDist = Math.hypot(wrist.x - anchor.x, wrist.y - anchor.y) / scale;
  const handSep = Math.hypot(wrist.x - bowWristPos.x, wrist.y - bowWristPos.y) / scale;

  const bowArmAngle = bowArmAngleOf(landmarks, frameWidth, frameHeight);
  if (bowArmAngle === null) return false;

  // Stillness: compare to where the draw wrist was last frame. Speed (distance moved per
  // second), not raw distance, so it doesn't depend on how often this happens to get called.
  // No previous frame yet means we can't know it's still, so treat that as "moving". lastDrawWrist
  // is kept in the same normalised space landmarks always arrive in (matching every other piece
  // of state this file carries frame-to-frame) and converted to pixel space at compare time,
  // using THIS frame's dimensions for both points — correct as long as the frame size hasn't
  // changed since the previous frame, true for every frame within one camera stream.
  const prevNorm = lastDrawWrist;
  lastDrawWrist = { x: landmarks[drawWrist].x, y: landmarks[drawWrist].y, t: nowMs };
  const dtSec = prevNorm ? (nowMs - prevNorm.t) / 1000 : 0;
  const prev = prevNorm ? toPixelSpace(prevNorm, frameWidth, frameHeight) : null;
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
      shoulderDrop: shoulderDropSampleOf(landmarks, frameWidth, frameHeight),
      elbowAlign: drawElbowAlignmentOf(landmarks, frameWidth, frameHeight),
      anchorOk,
      armOk,
      sepOk,
      stillOk,
      atFullDraw,
      eligible: frameEligible,
    },
    nowMs
  );

  return atFullDraw;
}

// Attempt-boundary rule for the shot log: a draw attempt is "in progress" for as long as hand
// separation stays at/above DRAW_ATTEMPT_MIN_SEP. It ends — and gets judged for logging — the
// moment separation drops back below that floor (hands back together at rest). This is the
// simplest rule that both (a) doesn't split one long hold into several rows and (b) doesn't
// merge two separate shots taken back-to-back into one. No timer involved: nothing here expires
// on its own, ever.
//
// An in-progress attempt tracks TWO separate things, deliberately kept apart:
//   - peakHandSep: the best hand separation seen on ANY frame, eligible or not. This is what
//     the SHOT_MIN_PEAK_SEP_FRACTION gate in endAttempt below judges — whether the archer really
//     drew the bow is a fact about what his hands did, unaffected by whether the pipeline had
//     finished settling yet. UNCHANGED by the move to medians below: "did he draw far enough to
//     count as a shot at all" is still, correctly, a question about the single most extreme
//     moment of the attempt, not an average across it.
//   - eligibleFrames: every ELIGIBLE frame's sample (frameEligible, see PIPELINE SETTLING above
//     and isAtFullDraw), kept via reservoirAdd so the array stays bounded (see
//     MEDIAN_SAMPLE_CAP above) without biasing which part of the hold survives. THIS is what
//     endAttempt below draws the shot's logged numbers from, each measure medianed
//     independently — see medianSampleOf. An unsettled frame's own numbers must never become
//     part of the shot's reading, even though the attempt around it is completely real.
// startMs is tracked from the very first frame regardless of eligibility too, for the same
// reason as peakHandSep — SHOT_MIN_DURATION_MS is about how long the draw actually took, not
// about when the pipeline happened to finish settling.
//
// nowMs threads through from isAtFullDraw's own nowMs parameter (renderLoop's `now`) — never
// performance.now() called fresh in here, so selfTest can drive attempt timing deterministically
// (see SHOT_MIN_DURATION_MS above and endAttempt below, which is what actually uses it).
function trackShotAttempt(sample, nowMs) {
  if (sample.handSep >= DRAW_ATTEMPT_MIN_SEP) {
    const isNewAttempt = !attempt; // hands just left the resting position — a fresh attempt, not a continuation
    if (isNewAttempt) {
      attempt = { startMs: nowMs, peakHandSep: sample.handSep, eligibleFrames: [], eligibleSeen: 0, reachedFullDraw: false };
      // Recording starts here, not in endAttempt, so the raise and draw are in the clip too — by
      // the time endAttempt fires the good part is already over. Starts regardless of this
      // frame's eligibility — the clip is a recording of what happened, not a measurement.
      startClipRecording();
    } else if (sample.handSep > attempt.peakHandSep) {
      attempt.peakHandSep = sample.handSep;
    }
    if (sample.eligible) {
      reservoirAdd(attempt, sample);
      // "Did any settled frame reach true full draw" — a plain OR over every eligible frame this
      // attempt has seen, independent of which frames the reservoir above happened to keep (a
      // frame that gets evicted from the reservoir must not un-say that full draw was reached;
      // the reservoir bounds MEMORY for the medians, it must never bound what this flag can see).
      attempt.reachedFullDraw = attempt.reachedFullDraw || !!sample.atFullDraw;
    }
  } else {
    endAttempt(nowMs);
  }
}

// Reservoir sampling (Algorithm R): keeps an UNBIASED random subset of up to MEDIAN_SAMPLE_CAP
// items from a stream of unknown/unbounded length, without ever seeing the whole stream at once.
// The first MEDIAN_SAMPLE_CAP items are kept outright; after that, the n-th item (1-indexed via
// eligibleSeen) replaces a uniformly-random existing slot with probability MEDIAN_SAMPLE_CAP/n —
// the standard proof that every item ends up equally likely to survive to the end, regardless of
// when it arrived. That property is the whole point here: "first N" would bias toward the
// raise-to-hold transition, "most recent N" would bias toward late-hold fatigue, but reservoir
// sampling can't lean toward any part of the hold, so capping memory this way can only ever cost
// precision, never introduce a NEW selection bias of the kind this file just spent Part 1 finding.
function reservoirAdd(attempt, sample) {
  attempt.eligibleSeen++;
  if (attempt.eligibleFrames.length < MEDIAN_SAMPLE_CAP) {
    attempt.eligibleFrames.push(sample);
  } else {
    const j = Math.floor(Math.random() * attempt.eligibleSeen);
    if (j < MEDIAN_SAMPLE_CAP) attempt.eligibleFrames[j] = sample;
  }
}

// Plain median of a list of numbers — the middle value once sorted, or the average of the two
// middle values on an even count. Returns null on an empty list rather than NaN, so callers can
// tell "no readable frames for this measure" apart from a real zero. Pure, no module state — so
// selfTest can check it directly.
function median(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Turns an attempt's kept eligible frames (see reservoirAdd above) into ONE shot entry, shaped
// exactly like the single-frame `sample` object endAttempt used to log directly — everything
// downstream (logShot, summarizeShots, renderShotRow, narrateMeasure) reads a shot entry the same
// way regardless of which of the two ever produced it, so this is the only place that needed to
// change to fix the bias Part 1 measured.
//
// THE FIX ITSELF: Part 1 found that scoring a shot from "the single frame with the highest hand
// separation" was not just an arbitrary choice among equally-good frames — it was a BIASED one.
// Hand separation is a distance divided by an estimated torso length, and torso length jitters
// frame to frame (it's built from noisy landmarks too), so the frame that happens to look most
// "drawn" is disproportionately the frame where the torso-length ESTIMATE happened to land small,
// inflating the ratio. Shoulder drop divides by that same torso-length estimate, so whichever
// shoulder shares hand-separation's own scale reference (draw-side, by default — see
// isAtFullDraw) inherits the bias directly: measured, it ran the logged shoulder-drop reading
// about half a percentage point high across repeated trials, consistently in the same direction
// almost every time, not the coin-flip a merely arbitrary choice would produce. Bow-arm angle and
// elbow alignment are plain angles at a joint — no division by torso length anywhere in their
// maths — and measured close to zero, no consistent direction. That contrast is exactly what
// confirms it's the division, not "any single frame is untrustworthy": a single frame is a fine
// sample, but hand separation is a bad SELECTOR for which one to trust, because it's built from
// the same shaky denominator as one of the numbers being reported.
//
// The fix is not "pick a better frame" — any selection rule based on a ratio that shares a
// denominator with a reported measure would reintroduce the same problem in a new shape. Instead:
// stop selecting a frame at all. Take the median of each measure independently across every
// eligible frame of the hold. A median is robust to exactly the kind of one-frame excursion a
// peak-selection rule goes looking for, and since no single frame gets to "win," there is no
// longer a selection process for measurement noise to bias.
//
// Each measure's median is computed from only the frames where THAT measure was actually
// readable (not null, not below MIN_VISIBILITY — see shoulderDropOf/bowArmAngleOf/
// drawElbowAlignmentOf, which already return null rather than a guess) — one measure being
// unreadable on some frames must never shrink another measure's sample, and a measure that was
// unreadable on EVERY eligible frame comes back null here, which renders exactly like a single
// uncertain reading always has (see updateBowArmReadout etc. and shotValueHtml).
//
// Elbow alignment is medianed via its SIGNED value (see drawElbowAlignmentOf's own comment: "8
// high" and "8 low" must not be treated as the same number) and only turned back into a plain
// deviation + high/low/level direction afterwards, for exactly the same reason the live readout
// and the session average both already use `signed` rather than averaging deviation and
// direction separately.
//
// handSep and the four instantaneous trigger flags (anchorOk/armOk/sepOk/stillOk) are diagnostic,
// ?debug-only fields, not one of the four measures the owner reads (see renderShotRow's debugBit)
// — but there is no longer a single "the" frame to read them off either. handSep gets the same
// median treatment as the real measures, for consistency; the four booleans get a MAJORITY vote
// (true if more than half the eligible frames were true) rather than either a median (undefined
// for booleans) or reviving a selection rule to pick one frame's flags — "was this typically true
// across the hold" is the honest question a debug line summarizing many frames can actually answer.
function medianSampleOf(frames) {
  const nums = (getValue) => frames.map(getValue).filter((v) => v != null);
  const majority = (getValue) => {
    const bools = frames.map(getValue).filter((v) => v != null);
    if (bools.length === 0) return null;
    return bools.filter(Boolean).length * 2 > bools.length;
  };

  const bowArmAngle = median(nums((f) => f.bowArmAngle));
  const shoulderBow = median(nums((f) => f.shoulderDrop?.bow ?? null));
  const shoulderDraw = median(nums((f) => f.shoulderDrop?.draw ?? null));
  const elbowSigned = median(nums((f) => f.elbowAlign?.signed ?? null));
  const elbowAlign =
    elbowSigned == null
      ? null
      : {
          deviation: Math.abs(elbowSigned),
          direction: elbowSigned > 0 ? "high" : elbowSigned < 0 ? "low" : "level",
          signed: elbowSigned,
        };

  return {
    handSep: median(nums((f) => f.handSep)),
    bowArmAngle,
    shoulderDrop: { bow: shoulderBow, draw: shoulderDraw },
    elbowAlign,
    anchorOk: majority((f) => f.anchorOk),
    armOk: majority((f) => f.armOk),
    sepOk: majority((f) => f.sepOk),
    stillOk: majority((f) => f.stillOk),
  };
}

// Ends whatever attempt is in progress (if any). Called when hand separation drops back to
// resting (from trackShotAttempt above, with the current timestamp) or when the pose is lost
// entirely mid-attempt (from renderLoop, with ITS current timestamp) — either way, whatever was
// going on has stopped, and it's judged the same way regardless of which of those two things
// ended it: tracking loss must not manufacture a shot that the same movement, ending normally,
// wouldn't have earned.
//
// Two gates first, both against the attempt's own peak (ALL frames, eligible or not — see
// trackShotAttempt above) — see SHOT_MIN_PEAK_SEP_FRACTION and SHOT_MIN_DURATION_MS above for
// why these two specifically. An attempt that fails either one gets thrown away, not logged:
// counted in rejectedAttemptCount, and any clip recording still running for it gets DISCARDED
// (see discardRecording) right now rather than left to expire on its own.
//
// A THIRD, separate case, checked only once the attempt has cleared both gates above: a real
// draw attempt whose every single frame happened to be unsettled (see PIPELINE SETTLING above)
// has no eligible frames to log — a.eligibleFrames is still empty. This is NOT the same claim as
// rejectedAttemptCount (noise that was never plausibly a draw at all): the owner really did draw
// the bow here, the app just never got a settled enough look at it to log a number. Counted and
// reported separately — unsettledAttemptCount, its own line in the log — so the two can never be
// confused for each other; see renderShotLog. Its recording is discarded the same way as the
// rejected case above — no shot ever gets logged for this attempt either.
//
// Both of these throw the ATTEMPT away, not just its recording — no shot ever gets logged for
// it, so its clip (if any) must never raise the "at least one clip failed" banner or explain
// itself on some unrelated later row (see discardRecording / resolveClipOutcome's own discarded
// check). A field bug caught this the hard way: a single landmark-noise blip lasting one or two
// frames is easily long enough to cross DRAW_ATTEMPT_MIN_SEP and start a real recording, but far
// too short for canvas.captureStream/MediaRecorder to ever encode a single frame of video before
// it's thrown away here — that recording resolving with zero chunks used to read as a genuine
// recording FAILURE and raise the banner, even in a session where the one real shot recorded
// perfectly.
function endAttempt(nowMs) {
  if (!attempt) return;
  const a = attempt;
  attempt = null; // clear first — logShot/finalizeRecording below must never see a stale in-progress attempt

  const gotDeepEnough = a.peakHandSep >= SHOT_MIN_PEAK_SEP_FRACTION * FULL_DRAW_HAND_SEP_MIN;
  const lastedLongEnough = typeof nowMs === "number" && typeof a.startMs === "number" && nowMs - a.startMs >= SHOT_MIN_DURATION_MS;

  if (!gotDeepEnough || !lastedLongEnough) {
    rejectedAttemptCount++;
    discardRecording(activeRecording); // this attempt's clip (if any) never gets a shot number, AND must never be reported as a clip failure — see discardRecording
    renderShotLog(); // the "N movements ignored" line needs to move even when nothing gets logged
    return;
  }

  if (a.eligibleFrames.length === 0) {
    unsettledAttemptCount++;
    discardRecording(activeRecording); // same reasoning as the rejected case above — never leave a clip with no shot to attach to, and never report it as a failure either
    renderShotLog();
    return;
  }

  // Each of the four real measures (plus the diagnostic handSep/flags) is the MEDIAN of its own
  // eligible frames, computed independently — see medianSampleOf's own comment for the full
  // reasoning and what this replaced.
  const shotNum = logShot({ ...medianSampleOf(a.eligibleFrames), startMs: a.startMs, reachedFullDraw: a.reachedFullDraw });
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
//
// FIELD BUG this section was rewritten for: the owner shot two arrows on his iPhone and both rows
// showed a bare "no clip" — with NO "clips unavailable" banner at all. That absence is the whole
// clue: CLIP_SUPPORTED only checks that `MediaRecorder` and `canvas.captureStream` exist as
// functions, and startClipRecording's try/catch only fires if `.start()` throws synchronously.
// Neither of those is what actually happened — recording believed it had started fine and then
// produced nothing, silently, which is a failure mode plain feature-detection cannot see and the
// old code had no way to report even if it noticed. iOS Safari recording a canvas.captureStream
// through MediaRecorder is a known-thin combination (as opposed to recording straight from a
// camera stream) — MIME types can come back "supported" from isTypeSupported and still fail to
// actually mux frames, `ondataavailable` can fire with only zero-byte blobs, `onerror` can fire
// with nothing listening for it (the old code only console.error'd it), and a recorder can get
// stuck never firing `onstop` at all. This sandbox cannot reproduce Safari itself, so every one of
// those specific shapes is simulated below (see the verification notes this shipped with) and each
// is now given its own explanation instead of a silent "no clip" — see `explainClipFailure`.

// Picks a MIME type by trying CLIP_MIME_CANDIDATES in order and returning the first one this
// browser claims to support; null means "nothing on the list — let the browser pick its own
// default" rather than refusing to record at all. NOTE: this only reflects what
// `MediaRecorder.isTypeSupported` CLAIMS, which on Safari has historically been thin/optimistic —
// a type can come back "supported" here and still fail to actually produce video once recording
// starts. That downstream failure is caught and explained where it actually shows up (see
// resolveClipOutcome below), not here — isTypeSupported has no way to predict it in advance.
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
    // `failReason`: the first specific explanation something below finds for why this recording
    // is going to come back empty, if any does — surfaced to the owner by resolveClipOutcome
    // rather than staying a console-only message. `settled`: true once this recording's outcome
    // has been fully resolved, one way or another (real onstop, or the stop-watchdog giving up on
    // it) — guards resolveClipOutcome against running twice for the same recording, since more
    // than one of those can end up racing to call it for the same rec.
    const rec = {
      recorder, clipStream, chunks: [], shotNum: null,
      finished: false, settled: false, failReason: null, discarded: false,
      capTimer: null, tailTimer: null, stopWatchdog: null,
    };
    recorder.ondataavailable = (ev) => {
      try {
        if (ev.data && ev.data.size > 0) rec.chunks.push(ev.data);
        // A zero-size chunk isn't an error by itself (MediaRecorder can emit one on an empty
        // interval) — it's just not kept. If EVERY chunk this recording ever produces turns out
        // to be zero-size, rec.chunks stays empty and resolveClipOutcome's generic "came out
        // empty" explanation below is exactly the right, honest thing to tell the owner.
      } catch (err) {
        console.error("archery-form-coach: clip data handling failed", err);
      }
    };
    // A MediaRecorder error mid-recording used to only reach console.error — invisible to the
    // owner, who cannot watch a console (see CLAUDE.md). Now it both stops this recording right
    // away (a recorder that has errored can't be trusted to keep going) and raises the SAME
    // "at least one clip failed this session" banner a synchronous start() failure already raises
    // below — from the owner's side, a recorder that errors out mid-shot is exactly that claim.
    recorder.onerror = (ev) => {
      console.error("archery-form-coach: clip recorder error", ev?.error ?? ev);
      rec.failReason = rec.failReason || "no clip — recorder error";
      markClipsUnavailable("Some shots couldn't be recorded — at least one clip failed this session.");
      finalizeRecording(rec);
    };
    // Stopping the recorder does NOT stop canvas.captureStream's tracks — they keep pulling
    // frames off the canvas at CLIP_FRAME_RATE forever unless something stops them explicitly,
    // which without this would mean one live, still-pulling capture track per shot for the rest
    // of the session (found in review: 3 shots -> 3 leaked live tracks, unbounded over an end).
    // Stopped here, in onstop, deliberately AFTER ondataavailable/onstop have already handed the
    // recorder its final data (see resolveClipOutcome) rather than the instant .stop() is
    // called — so cleaning up the source can never cost the recording its last frame.
    recorder.onstop = () => {
      clearTimeout(rec.stopWatchdog); // it actually stopped -- the stop-watchdog below doesn't need to force anything
      stopClipStreamTracks(rec.clipStream);
      resolveClipOutcome(rec);
    };
    // Each capture track ending on its own (not because WE stopped it) is a distinct failure
    // shape from anything above: the source of frames vanished out from under a recorder that's
    // otherwise still "running" and may never call onstop or onerror about it at all. Caught here
    // so it can't turn into a silently-hung recording (see the stop-watchdog below, which is the
    // other half of this same protection) or a leaked track sitting in "ended" limbo forever.
    clipStream.getTracks().forEach((track) => {
      track.addEventListener("ended", () => {
        if (rec.settled) return; // recording already wrapped up through the normal path -- this is just the track's own cleanup firing after the fact, not news
        rec.failReason = rec.failReason || "no clip — camera feed cut out";
        finalizeRecording(rec);
      });
    });
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
// this attempt ever ended), there's usually nothing to tell — this shot simply won't have a clip
// — UNLESS that earlier ending left a reason behind in pendingClipNote (see its own comment), in
// which case this shot's row gets to explain itself instead of showing a bare "no clip".
function attachRecordingToShot(shotNum) {
  if (!activeRecording || activeRecording.shotNum !== null) {
    if (pendingClipNote) {
      explainClipFailure(shotNum, pendingClipNote);
      pendingClipNote = null;
    }
    return;
  }
  activeRecording.shotNum = shotNum;
  activeRecording.tailTimer = setTimeout(() => finalizeRecording(activeRecording), CLIP_TAIL_MS);
}

// Stops a recording (idempotent — safe to call twice, from both its cap timer and its tail timer
// racing, or from a fresh attempt cutting it short) and clears it from activeRecording. The
// actual blob only becomes available later, asynchronously, in the recorder's onstop handler —
// see resolveClipOutcome. Arms a bounded stop-watchdog alongside the real stop() call: iOS Safari
// has been seen to leave a MediaRecorder recording from a canvas stream stuck mid-stop with
// onstop never firing at all, which — left unbounded — would both leak that recording's capture
// track forever (see stopClipStreamTracks) and leave its shot's row silently showing "no clip"
// with no explanation, forever, since nothing would ever run to say why.
function finalizeRecording(rec) {
  if (!rec || rec.finished) return;
  rec.finished = true;
  clearTimeout(rec.capTimer);
  clearTimeout(rec.tailTimer);
  try {
    if (rec.recorder.state !== "inactive") {
      rec.recorder.stop(); // onstop above (or the watchdog below, if onstop never comes) is what actually resolves this recording's outcome
      rec.stopWatchdog = setTimeout(() => {
        rec.failReason = rec.failReason || "no clip — recorder never finished";
        stopClipStreamTracks(rec.clipStream); // onstop is the thing that was supposed to do this -- it hasn't, so do it ourselves rather than leave the track running
        resolveClipOutcome(rec);
      }, CLIP_STOP_TIMEOUT_MS);
    } else {
      // Already inactive without us calling stop() here — onstop won't fire again on our
      // account, so nothing else is going to stop the capture tracks; do it ourselves.
      stopClipStreamTracks(rec.clipStream);
      resolveClipOutcome(rec); // safety net: covers the rare case where the recorder never even reached "recording" (e.g. start() silently no-op'd) and onstop was never going to fire at all
    }
  } catch (err) {
    console.error("archery-form-coach: failed to stop clip recording", err);
    stopClipStreamTracks(rec.clipStream); // stop() itself failed, so onstop may never fire -- don't leak the tracks over that
    rec.failReason = rec.failReason || "no clip — recorder failed to stop";
    resolveClipOutcome(rec);
  }
  if (activeRecording === rec) activeRecording = null;
}

// Marks a recording as belonging to an attempt the app has just decided to THROW AWAY (failed
// SHOT_MIN_PEAK_SEP_FRACTION/SHOT_MIN_DURATION_MS, or never produced a single settled frame) —
// see endAttempt's two early-return branches, the only callers. This is a genuinely different
// claim from a recording that failed: the app is not reporting "this clip broke", it's reporting
// "there was never going to be a shot here to have a clip at all." Set BEFORE finalizeRecording
// runs, so that whichever path eventually resolves this recording's outcome (a real onstop, or
// the stop-watchdog) can see it via resolveClipOutcome's own discarded check below and skip every
// bit of owner-facing reporting for it — no banner, no per-row reason, no pendingClipNote left
// for some later, unrelated shot to inherit.
//
// Why this needs its own flag rather than just leaving shotNum null (which resolveClipOutcome
// already treated as "not a failure" for the CLIP_MAX_MS case): a discarded attempt's recording
// can genuinely have FAILED too (chunks.length === 0) — most commonly because the discarded
// attempt itself was too brief for canvas.captureStream/MediaRecorder to ever encode a single
// frame before it got thrown away, which is not a malfunction, just an attempt that never
// deserved a recording in the first place. Without distinguishing this from a real failure, that
// empty-chunks outcome used to fall straight into resolveClipOutcome's generic "recording came
// out empty" branch and raise the "at least one clip failed" banner — a false positive even in a
// session where the one real, logged shot recorded perfectly. Found and fixed after a false
// "clips failed" banner surfaced on a run with exactly one successful, fully-attached clip.
function discardRecording(rec) {
  if (rec) rec.discarded = true;
  finalizeRecording(rec);
}

// Runs once a recording's outcome is actually known — either the real onstop fired, or the
// stop-watchdog above gave up waiting for one. Guarded by rec.settled so whichever of those two
// gets here first is the one that counts; the other is a harmless no-op (both can legitimately
// fire for the same rec — e.g. a late onstop arriving just after the watchdog already forced
// cleanup — and processing a recording's outcome twice could double-attach or double-count it).
//
// A clip with real, non-empty video attaches to its shot as before. Everything else is a failure
// that used to just `return` — the owner would see an unexplained "no clip" and have no way to
// tell "recording never worked at all" apart from "this one shot's recording came back empty"
// apart from "the clip arrived after its row was already gone". Those are different claims (see
// CLAUDE.md on why clipsUnavailableReason's own two messages stay distinct) and now say so:
// explainClipFailure writes a specific reason onto the row itself rather than a bare "no clip".
//
// EXCEPT for a recording marked `discarded` (see discardRecording above): the app itself threw
// that attempt away, so there is no shot for a clip to have been owed to. Whatever this recording
// did or didn't produce is simply not a story the owner needs told — bail out before any of the
// chunk/blob inspection below, so a discarded attempt's recording can never raise the banner,
// write a per-row reason, or leave a pendingClipNote for some unrelated later shot to inherit.
function resolveClipOutcome(rec) {
  if (rec.settled) return;
  rec.settled = true;
  if (rec.discarded) return; // the attempt itself was thrown away -- see discardRecording; nothing here is ever a failure worth reporting
  try {
    if (rec.chunks.length > 0) {
      const blob = new Blob(rec.chunks, { type: rec.recorder.mimeType || "video/webm" });
      if (blob.size > 0) {
        if (rec.shotNum === null) {
          // The video itself is genuinely fine — this is the CLIP_MAX_MS safety cap cutting a
          // stuck-in-progress recording off before endAttempt ever ran, so there's no shot number
          // yet to attach it to, and there never will be one this recording still exists to hear
          // about. Remembered as a pending note (see explainClipFailure/pendingClipNote) so that
          // shot's row — once endAttempt logs it moments later — can still explain why it has no
          // clip, instead of a bare, unexplained one.
          explainClipFailure(null, "no clip — recording hit the 20-second limit before the shot ended");
          return;
        }
        attachClipToShot(rec.shotNum, blob);
        return;
      }
    }
    // Nothing usable came out of this recording. This is the shape the field bug itself actually
    // took — recording believed it had started fine (no synchronous throw, see startClipRecording's
    // own catch) and simply never produced real video — so it gets its own honest explanation
    // rather than silently falling through to a bare "no clip", and it also raises the same
    // session-level banner a synchronous start() failure or a mid-recording onerror would.
    markClipsUnavailable("Some shots couldn't be recorded — at least one clip failed this session.");
    explainClipFailure(rec.shotNum, rec.failReason || "no clip — recording came out empty");
  } catch (err) {
    console.error("archery-form-coach: failed to finalise clip", err);
    explainClipFailure(rec.shotNum, "no clip — failed to save");
  }
}

// Writes a specific failure reason onto a shot's row (rendered in place of the plain "no clip"
// note — see renderShotRow), so a failed clip explains itself instead of just being absent. If
// this recording's attempt hasn't actually logged a shot yet — shotNum is still null, which only
// happens when the CLIP_MAX_MS safety cap cuts a stuck recording off before endAttempt has run —
// there's no row to write to yet, so the reason is remembered in pendingClipNote instead, for the
// next call to attachRecordingToShot (the endAttempt that follows moments later) to pick up. Never
// overwrites a reason a row (or the pending note) already has — the FIRST cause found is the one
// that actually explains what happened; whatever ran on to fail again after that is noise on top.
function explainClipFailure(shotNum, reason) {
  if (shotNum === null) {
    if (!pendingClipNote) pendingClipNote = reason;
    return;
  }
  const entry = log.find((e) => e.shotNum === shotNum);
  if (!entry || entry.clipFailReason) return;
  entry.clipFailReason = reason;
  renderShotLog();
}

// Attaches a finished clip to its shot's row in the log, by shot number — reuniting the two,
// since the clip finishes recording well after logShot already ran. Two reasons there might be
// no row to attach to: this shot has since been bumped off the end of the log (SHOT_LOG_MAX newer
// attempts happened first), or shotNum is still null because the attempt this clip belongs to
// hasn't finished (and logged a shot) yet. Either way the blob is simply dropped — and since no
// object URL was ever created for it, there's nothing to revoke either.
function attachClipToShot(shotNum, blob) {
  if (shotNum === null) return;
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

// ===== MEASUREMENT PRECISION FLOORS — a second, ABSOLUTE bar under the relative one above.
// Different kind of number entirely: OUTLIER_SCATTER_FACTOR/DRIFT_GAP_FACTOR ask "is this gap
// big compared to how this session scattered?", which has a blind spot — a very tight, repeatable
// session has very little scatter to compare against, so a gap of a fraction of a degree can look
// "statistically" huge next to it. Caught in the field: a synthetic body held completely still
// (only ±0.004 of normalised-coordinate jitter, representative of the noise still left after the
// One Euro filter — see SMOOTH_* above) produced a 1-2 point wobble in shoulder-drop that the
// relative test alone called "drifted". Nothing drifted; the pipeline just can't measure a joint
// more precisely than this, and the tighter the archer shoots, the more confidently it would have
// narrated that noise back at him — the same phantom-shot failure one level up, an untrue claim
// stated with unearned confidence. So every claim narrateMeasure can make (outlier AND drift) must
// now clear BOTH bars: big relative to this session's own scatter, AND bigger than the app's own
// measurement floor below. Below the floor, the honest answer is "steady", because as far as this
// app can actually tell, it was.
//
// NOT a form target — nothing here implies any judgement of the owner's archery, so it doesn't
// belong in CALIBRATE WITH COACH and doesn't need a coach to set it. It's a statement about how
// precisely THIS PIPELINE can measure a joint from a phone camera at five metres — same family as
// SMOOTH_* above: a characteristic of the measurement pipeline, not of the archer. Four separate
// constants, one per measure, because they're in different units (degrees vs. percentage points)
// and shouldn't be compared as if they were the same number wearing different labels.
//
// Where these numbers come from: each measure's own geometry, run through roughly ±0.004 of
// normalised-coordinate jitter per landmark (the residual noise level the field case above was
// built from, and the rough order of magnitude One Euro is tuned to leave behind on a joint that
// isn't actually moving). Shoulder drop is a vertical gap between two independently-jittered
// points (shoulder, ear) divided by torso length (~0.3 in a typical side-on frame, the same
// figure used throughout this file's own test fixtures): two jittered points can disagree by up
// to ~2x0.004, so (0.008 / 0.3) x 100 ~= 2.7 percentage points of possible noise on its own —
// rounded up for margin. The two angle measures move a joint sideways off a limb segment roughly
// 0.15 units long (again, this file's own fixture scale); a jitter of 0.004 perpendicular to that
// segment shifts the angle by roughly atan(0.004 / 0.15) ~= 1.5-2 degrees, and two joints can each
// contribute — rounded up for the same reason. RAISE any of these to make the app more reluctant
// to call anything a trend or a standout shot (more noise gets called "steady"); LOWER them to
// make it more talkative — and more likely to chase noise, which is exactly the failure this
// block exists to prevent, so lower with real recorded jitter in hand, not a guess.
const BOW_ARM_CONSISTENCY_FLOOR_DEG = 2; // degrees — a bow-arm angle claim (outlier or drift) must clear this on top of the relative test
const ELBOW_CONSISTENCY_FLOOR_DEG = 2; // degrees — same reasoning as bow arm above; both are angles measured off similarly-sized limb segments
const SHOULDER_BOW_CONSISTENCY_FLOOR_PCT = 3; // percentage points — bow-side shoulder drop
const SHOULDER_DRAW_CONSISTENCY_FLOOR_PCT = 3; // percentage points — draw-side shoulder drop; same value as bow-side today (same measurement, same units, same noise budget) but named and set separately in case a real session ever shows the two sides need different floors
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
// `floor`: this measure's own MEASUREMENT PRECISION FLOOR (see the block above) — an absolute
// number, in this measure's own units, that a claim must ALSO clear on top of the relative
// (OUTLIER_SCATTER_FACTOR / DRIFT_GAP_FACTOR) test. Required, not optional: the relative test
// alone has no way to tell "genuinely different" apart from "this session happened to be very
// tight, so ordinary noise looks huge by comparison" — see that block's comment for the field
// case this caught.
//
// Pure: entries (any order) + a value-getter + a label + a word function + a floor in, one
// result out — { text, outlierShotNum, outlierWord } (outlierShotNum/outlierWord are null unless
// case 1 above fired) or null. No DOM, no module state, so selfTest can drive it with plain
// fixture arrays.
function narrateMeasure(entries, getValue, label, wordFor, floor) {
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
  // than the others typically scatter (more than OUTLIER_SCATTER_FACTOR times their own average)
  // — AND, regardless of how that comparison comes out, worstDev has to clear this measure's own
  // measurement-precision floor. Without that second bar, a very tight session (small
  // othersScatter) lets a gap of a fraction of the app's own noise floor look "statistically"
  // enormous and get called out by name — exactly the false claim MEASUREMENT PRECISION FLOORS
  // above exists to block.
  const isOutlier = worstDev > floor && (othersScatter < 1e-6 || worstDev > OUTLIER_SCATTER_FACTOR * othersScatter);
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
  // Same two-bar rule as the outlier check above: big relative to this session's own scatter,
  // AND bigger than this measure's own measurement-precision floor. A tight, repeatable session
  // has very little scatter to divide by, so without the floor a gap of noise-scale size can
  // clear the relative bar easily and read as "drift" when nothing actually changed.
  const isDrift = Math.abs(gap) > floor && (overallScatter < 1e-6 || Math.abs(gap) > DRIFT_GAP_FACTOR * overallScatter);
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

  // A big, obvious watch button when this shot has a clip; otherwise a "no clip" note — never
  // nothing, so a missing clip never reads as a missing shot. When something specific is known
  // about WHY (see explainClipFailure — a recorder error, an empty recording, one that arrived
  // too late, etc.), that reason is shown instead of the bare word "no clip", since a non-coder
  // owner standing at the phone is the only person who will ever see this and has no console to
  // check instead. data-shot carries the shot number for the click handler on shotLogEl (see
  // openClipPlayer wiring) to look the entry back up by.
  const clipBit = e.clipUrl
    ? `<button type="button" class="shotlog-play" data-shot="${e.shotNum}">▶ Watch</button>`
    : `<span class="shotlog-noclip">${e.clipFailReason || "no clip"}</span>`;

  return `<div class="shotlog-row"><div class="shotlog-row-main">Shot ${e.shotNum} — ${highlightText}${shortMark}${rawHtml}</div><div class="shotlog-row-clip">${clipBit}</div></div>`;
}

// Plain-language shot log — this is what the owner actually reads, standing at the phone after
// their end, not mid-shot, so it has to answer, in order: how many arrows did it see; was he
// consistent, and in what; which shot was the odd one out, and how (so he knows which clip to
// go watch). Nothing here judges his form against a target — see the block comment above
// narrateMeasure for why not. Raw degrees/percent are still there for whoever eventually tunes
// the CALIBRATE WITH COACH constants, just demoted to small print on each row, not the headline.
function renderShotLog() {
  // A startup failure outranks every other banner below — if this is set, tracking may never
  // have properly run this session at all, which makes even the clip-availability banner beside
  // it secondary. See recordStartupProblem/clearStartupProblem for when this is set and cleared.
  const startupBit = startupProblem ? `<div class="shotlog-banner">${startupProblem}</div>` : "";
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
  // A DIFFERENT claim from rejectedBit above, said in different words on purpose — see
  // unsettledAttemptCount's own comment. This one means a real draw happened but the app hadn't
  // finished getting ready yet, not that it decided the movement wasn't a shot.
  const unsettledBit =
    unsettledAttemptCount > 0
      ? `<div class="shotlog-rejected">${unsettledAttemptCount} arrow${unsettledAttemptCount === 1 ? "" : "s"} drawn before the app finished settling weren't recorded — give it a few seconds after starting before your first shot.</div>`
      : "";
  // ROUTINE-START ATTENTION GATING transparency — see that block's own comment for why this
  // exists at all: the owner can't watch the app decide to pause between shots any more than he
  // can watch anything else in this file happen live. Shown only once it's actually done
  // something (idled at least once), same >0 gating as the two lines above, and worded
  // differently from both on purpose: this isn't a claim about whether a movement was a real
  // draw (rejectedBit) or whether the pipeline was ready to measure one (unsettledBit) — it's a
  // plain statement that the app throttled itself to save battery between shots, and always came
  // back before anything was lost.
  const attentionBit =
    ATTENTION_GATING_ENABLED && attentionIdlePeriods > 0
      ? `<div class="shotlog-rejected">Paused full tracking between shots ${attentionIdlePeriods} time${attentionIdlePeriods === 1 ? "" : "s"} to save battery, checking every ${ATTENTION_IDLE_SAMPLE_INTERVAL_MS}ms for movement.${
          attentionLateWakeCount > 0
            ? ` ${attentionLateWakeCount} of those ${attentionLateWakeCount === 1 ? "time" : "times"} noticed the next movement only after it had already started — that shot (and its clip, if it has one) may be missing its very first instant.`
            : ""
        }</div>`
      : "";

  if (log.length === 0) {
    shotLogContentEl.innerHTML = `${startupBit}${banner}${modelBit}${rejectedBit}${unsettledBit}${attentionBit}<div class="shotlog-empty">No shots recorded yet — draw once and this fills in.</div>`;
    return;
  }

  const arrowWord = shotCount === 1 ? "arrow" : "arrows";
  const shownNote = log.length < shotCount ? ` The consistency lines below are based on your most recent ${log.length}.` : "";
  const countLine = `<div class="shotlog-count">${shotCount} ${arrowWord} this session.${shownNote}</div>`;

  const bowArm = narrateMeasure(log, (e) => e.bowArmAngle, "Bow arm", wordForBowArm, BOW_ARM_CONSISTENCY_FLOOR_DEG);
  const shoulderBow = narrateMeasure(log, (e) => e.shoulderDrop?.bow ?? null, "Bow shoulder", wordForShoulder, SHOULDER_BOW_CONSISTENCY_FLOOR_PCT);
  const shoulderDraw = narrateMeasure(log, (e) => e.shoulderDrop?.draw ?? null, "Draw shoulder", wordForShoulder, SHOULDER_DRAW_CONSISTENCY_FLOOR_PCT);
  const elbow = narrateMeasure(log, (e) => e.elbowAlign?.signed ?? null, "Draw elbow", wordForElbow, ELBOW_CONSISTENCY_FLOOR_DEG);

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

  shotLogContentEl.innerHTML = `${startupBit}${banner}${modelBit}${rejectedBit}${unsettledBit}${attentionBit}${countLine}<div class="shotlog-narrative">${narrativeHtml}</div>${rowsHtml}`;
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

  // ROUTINE-START ATTENTION GATING: while idle, pose detection (the expensive step) only runs
  // every ATTENTION_IDLE_SAMPLE_INTERVAL_MS — everything below this point in renderLoop is
  // skipped on the frames in between. This is the ONLY thing idle changes: detection still runs,
  // just less often, so there is no state in which the app has stopped watching for the owner to
  // start (see ATTENTION_GATING_ENABLED's own comment for why that's structural, not a promise).
  // While engaged, this never skips anything — every frame behaves exactly as it did before this
  // feature existed.
  if (ATTENTION_GATING_ENABLED && !attentionEngaged) {
    const dueForIdleSample =
      attentionLastIdleSampleMs === null || now - attentionLastIdleSampleMs >= ATTENTION_IDLE_SAMPLE_INTERVAL_MS;
    if (!dueForIdleSample) {
      withMirror(drawVideoFrame); // keep the on-screen view (and a clip, if one were somehow recording) alive even on a skipped frame
      syncDebugOverlay();
      return;
    }
    attentionLastIdleSampleMs = now; // this frame IS the idle sample — the next one is due no sooner than a full interval from now
  }

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

  // Decide whether the app should be engaged (full rate) or idle for the frames that follow —
  // see updateAttentionState's own comment for why this runs on the RAW, pre-smoothing landmarks
  // and BEFORE the smoothing/settling below, not after: if this call is about to re-engage from
  // idle, it resets landmarkSmoother/settledFrames/currentCropBox synchronously, so everything
  // from here to the end of this same frame already runs against the fresh, reset state.
  updateAttentionState(now, rawLandmarks);

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
    resetSettling(); // the archer was just lost — whatever comes back next is exactly as unsettled as session start, see PIPELINE SETTLING above
    // Re-acquire on loss: whether this frame was cropped or not, no landmarks means whatever crop
    // box we had (if any) no longer contains the archer, or we never had one. Drop it so the very
    // next frame detects on the WHOLE frame again rather than continuing to stare into a box that
    // may no longer have anyone in it — a crop that can never find its way back would be exactly
    // the kind of hang the owner has no way to recover from (see CLAUDE.md).
    currentCropBox = null;
    endAttempt(now); // pose lost mid-attempt counts as the attempt ending, same as hands relaxing
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
    updateBowArmReadout(landmarks, frameWidth, frameHeight);
    updateShoulderDropReadout(landmarks, frameWidth, frameHeight);
    updateDrawElbowReadout(landmarks, frameWidth, frameHeight);
    // Return value intentionally unused here — isAtFullDraw's real job on every frame is its
    // side effect, calling trackShotAttempt (below) to feed the shot log. It used to also drive
    // the auto-freeze state machine, which read the true/false result; that machine is gone, but
    // the shot log still depends on this call happening every frame, so it stays.
    //
    // frameEligible: is THIS frame's own reading settled enough to become a shot's logged sample
    // (see PIPELINE SETTLING above)? Computed here, not inside isAtFullDraw, because only
    // renderLoop knows whether THIS frame actually used an established crop box, and whether
    // that box has actually stopped moving — a box merely being present isn't the same as it
    // having settled; see cropBoxIsStable. Compared against the PREVIOUS frame's box before that
    // variable gets overwritten for next frame, then threaded through as an explicit argument
    // (like nowMs already is) so selfTest can drive it too.
    const cropBoxStableThisFrame = cropBoxIsStable(usedCropBox, prevUsedCropBox);
    prevUsedCropBox = usedCropBox;
    const frameEligible = advanceSettling(!!usedCropBox, cropBoxStableThisFrame);
    isAtFullDraw(landmarks, now, frameEligible, frameWidth, frameHeight);

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

// Fixes HANDOVER.md Stage 1a: the log used to be the only way to dismiss itself, and it covered
// the very 📋 button that does that. closeShotLog is now the one place "put the log away" happens,
// called from the always-visible #shotlog-close button (never inside the scrolling content, so a
// long log can't carry it out of reach) and from the tap-outside-to-close handler below.
function closeShotLog() {
  shotLogEl.classList.add("hidden");
}

shotLogCloseBtn.addEventListener("click", closeShotLog);

// Convenience, not the fix itself (the close button above is): tapping anywhere outside the log
// panel while it's open also dismisses it. Skips btn-log itself so this can never race that
// button's own open/close toggle — without the guard, a tap on 📋 to close the log would first
// close it here, then immediately reopen it when the toggle handler ran on the same click.
//
// Also skips anything inside the clip player. #clipplayer is a DOM sibling of #shotlog, not a
// descendant, but it opens ON TOP of the log (z-index 20 vs 10 — see style.css) whenever the
// owner taps "Watch" on a row, and stays a sibling the whole time it's open. Without this, every
// tap inside the player while reviewing a clip — its own ✕ Close, the video's own transport
// controls, a scrub — reads as "outside the log" and closes the log behind it, so closing the
// clip would dump the owner back at the camera view instead of back at the log he was reading.
// Checked by DOM containment rather than the player's hidden state, so it can't race the
// player's own close handler: containment still holds even after that handler has already
// hidden the player earlier in the same click's bubble phase. #clipplayer is the only thing that
// currently opens above the log's z-index; give any future full-screen overlay the same
// treatment here.
document.addEventListener("click", (ev) => {
  if (shotLogEl.classList.contains("hidden")) return;
  if (shotLogEl.contains(ev.target) || clipPlayerEl.contains(ev.target) || ev.target === btnLog) return;
  closeShotLog();
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

// Plain-English description of whatever startupStep hasn't finished yet — the one place that
// wording lives, shared by the transient status message and the persistent shot-log record below
// so the two surfaces never describe the same stall two different ways. Deliberately says WHICH
// thing didn't happen ("the camera started but never sent a picture") rather than a generic
// "camera error" — that distinction is the whole point of tracking startupStep at all, see
// CLAUDE.md's "one interaction" rule and its knock-on: whatever this says has to be enough on its
// own for a non-coder to relay back, since he can't describe a stack trace he never saw.
function startupStepProblem(step) {
  switch (step) {
    case "loading the pose model":
      return "The pose tracker never finished loading";
    case "starting the camera":
      return "The camera never started";
    case "waiting for the camera picture":
      return "The camera started but never sent a picture";
    case "starting tracking":
      return "The camera was ready, but tracking never started";
    default:
      return "Startup never finished";
  }
}

// The transient status-overlay line for whichever step the watchdog below caught still in
// flight. Short and non-technical — the owner reads this from ~5 metres and has one option: come
// over and restart the app (see CLAUDE.md's "one interaction" rule).
function startupStuckMessage(step) {
  return `${startupStepProblem(step)}. Close the app and reopen it.`;
}

// Writes (or updates) the persistent startup-problem banner at the top of the shot log — the
// DURABLE record of a startup failure, alongside (never instead of) the transient status-overlay
// message above. The status pill is easy to miss: the owner may not walk over for minutes, the
// screen could have dimmed, and nothing about a small pill at the top says "look at me" the way
// the shot log — the one place he's guaranteed to check when he's done shooting — already does
// for a failed clip recording (clipsUnavailableReason) or the pose-model choice (modelStatusLine).
// This gets the same treatment. `detail`, if given, is a technical aside (how long startup had
// been stuck, or the browser's own error text) — kept in small print, after the plain-English
// sentence, never standing in for it.
function recordStartupProblem(headline, detail) {
  startupProblem = detail ? `${headline}<div class="shotlog-startup-detail">${detail}</div>` : headline;
  renderShotLog();
}

// Clears the startup-problem banner — used only when startup goes on to succeed AFTER the
// watchdog already fired (the stuck step finished on its own before the owner ever got to the
// phone). A successful start must leave no scary residue in the one place he's going to check
// afterward; see recordStartupProblem above for why the shot log is that place.
function clearStartupProblem() {
  if (startupProblem === null) return;
  startupProblem = null;
  renderShotLog();
}

// Starts the startup watchdog: if main() hasn't finished (disarmed via the returned function)
// within a step's own allowance, the status text is replaced with a message naming whatever step
// was still in flight (startupStep), AND the same fact is written to the persistent shot log (see
// recordStartupProblem) so it is still there whenever the owner actually walks over. This is the
// fix for the deeper failure, not just the video.play() hang above — ANY step that stalls (a
// slow/broken model fetch, a camera permission prompt the owner never sees, a picture that never
// arrives) leaves no exception for main()'s own catch block to report, since a promise that never
// settles never throws.
//
// Polls once a second (cheap; nothing here needs finer resolution) rather than firing once at a
// single deadline, because — per setStartupStep — each step now gets judged against its own
// allowance, and the pose-model download step gets TWO checkpoints, not one:
//   - past STARTUP_WATCHDOG_MS: might just be a slow connection (see STARTUP_MODEL_WATCHDOG_MS's
//     own comment for the field bug this fixes), so the status text says so CALMLY — no alarm, no
//     "close the app and reopen it", nothing written to the persistent shot-log banner.
//   - past STARTUP_MODEL_WATCHDOG_MS: only now is it treated the same as every other stuck step.
// Every other step keeps the original single-checkpoint behaviour at STARTUP_WATCHDOG_MS — there
// is no large download to explain a slow one, so a stall there is reported immediately, same as
// before this fix. Per CLAUDE.md the owner may see whatever is on screen at one instant and never
// again before he walks over, so a message that will go on to retract itself must never be the
// alarming one — that's the whole reason the calm checkpoint exists as a separate step from the
// alarm, rather than just pushing the single alarm threshold out to STARTUP_MODEL_WATCHDOG_MS.
//
// Returns a function that disarms the watchdog; call it as soon as startup actually finishes
// (success OR error) so it can never fire after the fact and stomp on a status the app has already
// resolved on its own.
function armStartupWatchdog() {
  let announcedSlowModel = false; // so the calm "still loading" checkpoint only overwrites the status text once, not every poll
  const interval = setInterval(() => {
    const step = startupStep;
    const elapsedMs = performance.now() - stepStartedAt;
    if (step === "loading the pose model") {
      if (elapsedMs < STARTUP_MODEL_WATCHDOG_MS) {
        if (!announcedSlowModel && elapsedMs >= STARTUP_WATCHDOG_MS) {
          announcedSlowModel = true;
          statusEl.classList.remove("hidden");
          statusEl.textContent = "Still loading the pose tracker — this can take a while on a slow connection…";
        }
        return; // within this step's generous allowance — slow, not stuck, nothing alarming shown
      }
    } else if (elapsedMs < STARTUP_WATCHDOG_MS) {
      return; // within this step's ordinary allowance
    }
    clearInterval(interval);
    statusEl.classList.remove("hidden");
    statusEl.textContent = startupStuckMessage(step);
    recordStartupProblem(
      `${startupStepProblem(step)} — nothing was tracked this session.`,
      `Startup hadn't finished after ${Math.round(elapsedMs / 1000)} seconds.`
    );
  }, 1000);
  return () => clearInterval(interval);
}

async function main() {
  setStartupStep("loading the pose model");
  const disarmWatchdog = armStartupWatchdog();
  try {
    statusEl.textContent = "Loading pose model…";
    await initPoseLandmarker();
    drawingUtils = new DrawingUtils(ctx);

    setStartupStep("starting the camera");
    statusEl.textContent = "Starting camera…";
    await startCamera(); // updates startupStep to "waiting for the camera picture" partway through, see startCamera

    setStartupStep("starting tracking");
    disarmWatchdog();
    clearStartupProblem(); // startup made it through after all — remove any watchdog banner left by a step that stalled earlier but then recovered on its own
    statusEl.classList.add("hidden");
    updateHandButtonLabel();
    updateMirrorButtonLabel();
    renderLoop();
  } catch (err) {
    disarmWatchdog();
    statusEl.classList.remove("hidden");
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
    // Unlike the watchdog case above, main() gives up entirely here — there is no later success
    // to clear this away, so it's written once and (like clipsUnavailableReason) stands for the
    // rest of the session.
    recordStartupProblem(`${startupStepProblem(startupStep)}.`, err.message);
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
  const savedUnsettledAttemptCount = unsettledAttemptCount;
  const savedSettledFrames = settledFrames;
  const savedPrevUsedCropBox = prevUsedCropBox;
  const savedAttentionEngaged = attentionEngaged;
  const savedAttentionCalmSinceMs = attentionCalmSinceMs;
  const savedAttentionLastIdleSampleMs = attentionLastIdleSampleMs;
  const savedAttentionLastEvalMs = attentionLastEvalMs;
  const savedAttentionPrevRef = attentionPrevRef;
  const savedAttentionIdlePeriods = attentionIdlePeriods;
  const savedAttentionLateWakeCount = attentionLateWakeCount;
  // The clip-failure tests further down drive resolveClipOutcome/explainClipFailure directly
  // (see their own comments) and, in doing so, touch these same two module-level variables the
  // real app uses to remember an unresolved clip problem across renders. Saved/restored here —
  // once, for the whole run — for the same reason as everything else on this list: a diagnostic
  // run must never leave the app started while corrupting the exact state a diagnostic run
  // exists to double check.
  const savedClipsUnavailableReason = clipsUnavailableReason;
  const savedPendingClipNote = pendingClipNote;
  selfTestInProgress = true; // see the flag's own comment — keeps trackShotAttempt below from spinning up real MediaRecorders
  rightHanded = true;
  // Every fixture below except the dedicated ASPECT-RATIO CORRECTNESS section further down was
  // written (and its expected numbers hand-checked) as plain [0,1] normalised coordinates, with
  // no camera frame involved. Every geometry function now needs a frame width/height to convert
  // into (see toPixelSpace) — passing 1x1 here is a literal no-op multiplication (every landmark
  // times 1 is itself), so these fixtures keep measuring exactly what they always measured. It is
  // NOT a claim that real phone video is 1x1 or square — quite the opposite, see below.
  const NOOP_W = 1, NOOP_H = 1;
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
  const raiseScale = torsoLength(raise, R_SHOULDER, R_HIP, NOOP_W, NOOP_H);
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
  console.assert(isAtFullDraw(raise, 0, true, NOOP_W, NOOP_H) === false, "raise (first frame) must not read as full draw");
  console.assert(
    isAtFullDraw(raise, 500, true, NOOP_W, NOOP_H) === false,
    "raise held steady for a second frame (passes stillness too, now) must still be rejected — by hand separation alone"
  );

  // --- Full draw, held: hands apart, near anchor, bow arm straight. Rejected on the first
  // frame only because there's no prior position yet to judge stillness from; reads true once
  // it's held for a frame.
  lastDrawWrist = null;
  const drawn = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.52, y: 0.31 } });
  console.assert(
    isAtFullDraw(drawn, 0, true, NOOP_W, NOOP_H) === false,
    "first frame at full draw should read as still-moving (no prior position yet)"
  );
  console.assert(
    isAtFullDraw(drawn, 500, true, NOOP_W, NOOP_H) === true,
    "same position 500ms later (zero speed) should read as full draw"
  );

  // --- Drawing in progress: hands already apart and already near anchor (so anchor, arm, and
  // separation would all pass) but the draw wrist is still travelling fast between frames —
  // must be rejected by stillness alone, not because it never got close enough.
  lastDrawWrist = null;
  const midDraw1 = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.4, y: 0.31 } });
  isAtFullDraw(midDraw1, 0, true, NOOP_W, NOOP_H); // seeds lastDrawWrist; this call's own result isn't the point
  const midDraw2 = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.5, y: 0.32 } });
  const midScale = torsoLength(midDraw2, R_SHOULDER, R_HIP, NOOP_W, NOOP_H);
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
    isAtFullDraw(midDraw2, 50, true, NOOP_W, NOOP_H) === false,
    "wrist still travelling fast toward anchor (50ms, big jump) must not read as full draw yet — only stillness should be stopping it"
  );

  // --- Fast jump while otherwise at full draw: same idea as mid-draw above, kept as its own
  // check because it's the scenario closest to what a real archer's hand does in the instant
  // right before it settles at anchor.
  lastDrawWrist = null;
  const driftSeed = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.52, y: 0.31 } });
  isAtFullDraw(driftSeed, 500, true, NOOP_W, NOOP_H); // seeds lastDrawWrist at the same position/time as `drawn` above
  const drifted = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.6, y: 0.31 } });
  console.assert(
    isAtFullDraw(drifted, 600, true, NOOP_W, NOOP_H) === false,
    "wrist jumping far in 100ms (fast) should not read as holding still"
  );

  // --- Shot log attempt-boundary rule: an attempt is "in progress" for as long as hand
  // separation stays at/above DRAW_ATTEMPT_MIN_SEP, collecting its eligible frames along the way
  // (see reservoirAdd); it ends when separation drops back below that floor OR the pose is lost —
  // but only gets LOGGED if it also cleared the SHOT_MIN_PEAK_SEP_FRACTION and SHOT_MIN_DURATION_MS
  // gates in endAttempt (added after the field report: without these, nocking an arrow or lowering
  // the bow was logging as a phantom shot). Reset to a clean slate here — the outer save/restore
  // above puts it all back afterwards regardless.
  attempt = null;
  log = [];
  shotCount = 0;
  rejectedAttemptCount = 0;
  unsettledAttemptCount = 0;
  const sample = (handSep, atFullDraw = false, eligible = true, extra = {}) => ({
    handSep,
    bowArmAngle: 178,
    atFullDraw,
    eligible,
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
  // Eligible handSeps for this attempt were 0.4, 0.65, 0.6 — the MEDIAN (middle of the sorted
  // three) is 0.6, not the peak (0.65, which only the SHOT_MIN_PEAK_SEP_FRACTION gate above uses).
  console.assert(log[0].handSep === 0.6, "the logged row should carry the median hand separation across eligible frames, not the peak");
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
  // Eligible handSeps were 0.4 and 0.7 — median of an even count is the average of the two middle
  // values: (0.4 + 0.7) / 2 = 0.55.
  console.assert(log[0].handSep === 0.55, "the logged row should carry the median hand separation across eligible frames even though the attempt ended via pose loss");

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

  // --- PIPELINE SETTLING: advanceSettling/resetSettling/cropBoxIsStable, the pure bookkeeping
  // behind SETTLE_FRAMES_REQUIRED and CROP_BOX_STABLE_MAX_DELTA — see those constants' own
  // comments for the reasoning. Reset first so this doesn't inherit settledFrames from whatever
  // earlier fixtures left behind. Every advanceSettling call here passes `true` for the
  // stability argument unless the test is specifically about stability, so frame-count behaviour
  // stays isolated from box-stability behaviour.
  {
    resetSettling();
    // A frame before the threshold cannot become a logged sample: advanceSettling must read
    // false for every one of the first SETTLE_FRAMES_REQUIRED - 1 frames, cropping satisfied on
    // every one of them (present AND stable), so it's the FRAME COUNT specifically being tested
    // here, not either crop condition.
    let sawTrueEarly = false;
    for (let i = 0; i < SETTLE_FRAMES_REQUIRED - 1; i++) {
      if (advanceSettling(true, true)) sawTrueEarly = true;
    }
    console.assert(!sawTrueEarly, "no frame before SETTLE_FRAMES_REQUIRED should ever read as settled");
    // One after (the SETTLE_FRAMES_REQUIRED-th good-tracking frame) can: this is the frame that
    // crosses the threshold.
    console.assert(
      advanceSettling(true, true) === true,
      "the SETTLE_FRAMES_REQUIRED-th consecutive good-tracking frame should read as settled"
    );
    // Once settled, later frames stay settled (the counter doesn't wrap or reset on its own).
    console.assert(advanceSettling(true, true) === true, "settling should stay settled on the next frame too, not flicker back off");

    // The crop-PRESENT requirement is independent of frame count: even with far more than
    // SETTLE_FRAMES_REQUIRED good-tracking frames, a frame that never actually used an
    // established crop box must never read as settled while ROI_CROPPING_ENABLED is on.
    resetSettling();
    let sawTrueWithoutCrop = false;
    for (let i = 0; i < SETTLE_FRAMES_REQUIRED + 20; i++) {
      if (advanceSettling(false, false)) sawTrueWithoutCrop = true;
    }
    console.assert(
      !ROI_CROPPING_ENABLED || !sawTrueWithoutCrop,
      "with cropping on, a frame that never used an established crop box must never read as settled, no matter how many frames have passed"
    );

    // The crop-STABLE requirement is ALSO independent of frame count: a box that is present
    // every frame but never stops moving/resizing must never read as settled either — a box
    // merely existing is not the same as it having finished catching up (see
    // CROP_BOX_STABLE_MAX_DELTA's own comment: the raise-to-hold transition is exactly the case
    // this is for).
    resetSettling();
    let sawTrueUnstable = false;
    for (let i = 0; i < SETTLE_FRAMES_REQUIRED + 20; i++) {
      if (advanceSettling(true, false)) sawTrueUnstable = true;
    }
    console.assert(
      !ROI_CROPPING_ENABLED || !sawTrueUnstable,
      "with cropping on, a crop box that never stabilises must never read as settled, no matter how many frames have passed"
    );
    // Once the box actually stabilises (with the frame count already satisfied from a prior
    // unstable stretch), eligibility can be reached on the very next frame.
    console.assert(
      advanceSettling(true, true) === true,
      "once the box stabilises, with the frame count already well past the threshold, the next frame should read as settled"
    );

    // A reset mid-session re-arms the gate: get to settled, reset, and confirm the very next
    // frame goes back to reading unsettled — the counter doesn't remember where it was before.
    resetSettling();
    for (let i = 0; i < SETTLE_FRAMES_REQUIRED; i++) advanceSettling(true, true);
    console.assert(advanceSettling(true, true) === true, "setup: should be settled before the reset");
    resetSettling();
    console.assert(advanceSettling(true, true) === false, "a reset mid-session must re-arm the gate — the very next frame should read as unsettled again");
  }

  // --- cropBoxIsStable: pure geometry, no dependency on advanceSettling's frame counting.
  {
    const prevBox = { x: 100, y: 100, size: 200 };
    console.assert(cropBoxIsStable(prevBox, null) === false, "no prior box to compare against must never read as stable");
    console.assert(cropBoxIsStable(null, prevBox) === false, "no current box must never read as stable");
    console.assert(
      cropBoxIsStable({ x: 100, y: 100, size: 200 }, prevBox) === true,
      "an identical box (zero change) must read as stable"
    );
    console.assert(
      cropBoxIsStable({ x: 100, y: 100, size: 202 }, prevBox) === true, // 2/200 = 1% size change — comfortably under CROP_BOX_STABLE_MAX_DELTA (2%)
      "a small size change, well under the threshold, must still read as stable"
    );
    console.assert(
      cropBoxIsStable({ x: 100, y: 100, size: 210 }, prevBox) === false, // 10/200 = 5% size change — over the threshold
      "a size change over CROP_BOX_STABLE_MAX_DELTA must read as unstable"
    );
    console.assert(
      cropBoxIsStable({ x: 102, y: 100, size: 200 }, prevBox) === true, // 2/200 = 1% position change
      "a small position change, well under the threshold, must still read as stable"
    );
    console.assert(
      cropBoxIsStable({ x: 110, y: 100, size: 200 }, prevBox) === false, // 10/200 = 5% position change
      "a position change over CROP_BOX_STABLE_MAX_DELTA must read as unstable, even with size unchanged"
    );
  }

  // --- PIPELINE SETTLING, integration: trackShotAttempt/endAttempt must never log a frame's own
  // numbers unless that frame was eligible (sample.eligible), even though the attempt boundary
  // itself (start/end timing, the peak-separation gate) is judged from ALL frames regardless.
  attempt = null;
  log = [];
  shotCount = 0;
  rejectedAttemptCount = 0;
  unsettledAttemptCount = 0;

  // A real draw (deep and long enough) made ENTIRELY of unsettled frames: nothing here is noise
  // — sample(0.9, false, false) clears both SHOT_MIN_PEAK_SEP_FRACTION and, given the timing
  // below, SHOT_MIN_DURATION_MS — but every frame is ineligible, so there is no honest reading to
  // log. Must be counted as unsettled, NOT rejected, and must not appear in the log.
  trackShotAttempt(sample(0.4, false, false), 20000); // crosses the floor, unsettled
  trackShotAttempt(sample(0.9, false, false), 20400); // peak, still unsettled
  trackShotAttempt(sample(0.05, false, false), 20900); // ends — 900ms elapsed, well over SHOT_MIN_DURATION_MS
  console.assert(log.length === 0, "an attempt made entirely of unsettled frames must not be logged");
  console.assert(shotCount === 0, "shotCount must not advance for an unsettled attempt");
  console.assert(rejectedAttemptCount === 0, "an unsettled-but-otherwise-real attempt must NOT be counted as rejected (noise) — those are different claims");
  console.assert(unsettledAttemptCount === 1, "the unsettled attempt should be counted in unsettledAttemptCount specifically");

  // A real draw where the app was STILL unsettled during the raise but settles partway through:
  // the logged row's median must be computed from the eligible frame(s) only — the two ineligible
  // frames (including the higher-handSep one) must never enter the median at all, not even diluted
  // in with the eligible ones.
  trackShotAttempt(sample(0.4, false, false), 21000); // crosses the floor, unsettled
  trackShotAttempt(sample(0.95, false, false), 21400); // higher handSep, but STILL unsettled — must never be logged
  trackShotAttempt(sample(0.8, true, true, { bowArmAngle: 171 }), 21800); // settles now — the ONLY eligible frame this attempt has
  trackShotAttempt(sample(0.05), 22400); // ends — 1400ms elapsed
  console.assert(log.length === 1, "an attempt that settles partway through should still log, once it has an eligible frame");
  // Exactly one eligible frame — the median of a single value is that value itself, so this also
  // covers "a single-frame attempt still works" from a real trackShotAttempt/endAttempt run, not
  // just a direct call into medianSampleOf.
  console.assert(log[0].handSep === 0.8, "the logged reading must come from the eligible frame (0.8), not the higher-handSep unsettled one (0.95)");
  console.assert(log[0].bowArmAngle === 171, "the logged reading's other fields must also come from the eligible frame, not an unsettled one");
  console.assert(log[0].reachedFullDraw === true, "reachedFullDraw should reflect the eligible frame's own atFullDraw reading");
  console.assert(unsettledAttemptCount === 1, "a settled-in-time attempt must not also be counted as unsettled");

  // --- Shot log MEDIAN aggregation: median(), reservoirAdd(), and medianSampleOf() — the fix
  // Part 1's bias measurement led to. See medianSampleOf's own block comment for the full
  // reasoning; these check the specific properties the fix promises.
  {
    // median(): plain odd/even/empty behaviour.
    console.assert(median([3, 1, 2]) === 2, "median of an odd count should be the sorted middle value");
    console.assert(median([1, 2, 3, 4]) === 2.5, "median of an even count should average the two middle values");
    console.assert(median([5]) === 5, "median of a single value should be that value");
    console.assert(median([]) === null, "median of an empty list should be null, not NaN or 0");

    // medianSampleOf: each measure's null frames are excluded from THAT measure's median only —
    // one measure being unreadable on a frame must never shrink another measure's sample. Built
    // directly (not through trackShotAttempt) so the fixture can hand-place nulls precisely.
    const mixedFrames = [
      { handSep: 0.8, bowArmAngle: 170, shoulderDrop: { bow: 40, draw: null }, elbowAlign: { signed: 2 }, anchorOk: true, armOk: true, sepOk: true, stillOk: true, atFullDraw: true },
      { handSep: 0.82, bowArmAngle: null, shoulderDrop: { bow: 42, draw: 50 }, elbowAlign: { signed: -2 }, anchorOk: true, armOk: false, sepOk: true, stillOk: true, atFullDraw: false },
      { handSep: 0.78, bowArmAngle: 174, shoulderDrop: { bow: 44, draw: 52 }, elbowAlign: null, anchorOk: false, armOk: true, sepOk: false, stillOk: true, atFullDraw: false },
    ];
    const mixedResult = medianSampleOf(mixedFrames);
    console.assert(mixedResult.handSep === 0.8, "handSep median should use all 3 frames (0.78, 0.8, 0.82 -> 0.8)");
    console.assert(mixedResult.bowArmAngle === 172, "bowArmAngle median should skip the 1 null frame and use the other 2 (170, 174 -> 172)");
    console.assert(mixedResult.shoulderDrop.bow === 42, "bow-shoulder median should use all 3 frames (40, 42, 44 -> 42)");
    console.assert(mixedResult.shoulderDrop.draw === 51, "draw-shoulder median should skip its 1 null frame and use the other 2 (50, 52 -> 51)");
    console.assert(mixedResult.elbowAlign.signed === 0, "elbow median should skip its 1 null frame and use the other 2 (2, -2 -> 0)");
    console.assert(mixedResult.elbowAlign.direction === "level", "an elbow median of exactly 0 should report direction: level, same convention as a live 0deg reading");
    // Majority vote on the boolean flags: anchorOk true on 2/3, armOk true on 2/3, sepOk true on
    // 2/3, stillOk true on 3/3 — each independently, not tied to any one frame's own flag set.
    console.assert(mixedResult.anchorOk === true && mixedResult.armOk === true && mixedResult.sepOk === true && mixedResult.stillOk === true, "boolean flags should be a majority vote across the eligible frames, computed independently per flag");

    // A measure with NO readable frames at all must come back null — renders exactly like a
    // single uncertain reading already does (see updateBowArmReadout/shotValueHtml), never a
    // fabricated number and never NaN.
    const allNullBowArm = [
      { handSep: 0.8, bowArmAngle: null, shoulderDrop: { bow: 40, draw: 50 }, elbowAlign: { signed: 1 }, anchorOk: true, armOk: true, sepOk: true, stillOk: true, atFullDraw: true },
      { handSep: 0.8, bowArmAngle: null, shoulderDrop: { bow: 41, draw: 51 }, elbowAlign: { signed: 1 }, anchorOk: true, armOk: true, sepOk: true, stillOk: true, atFullDraw: true },
    ];
    const allNullResult = medianSampleOf(allNullBowArm);
    console.assert(allNullResult.bowArmAngle === null, "a measure that was unreadable on every eligible frame must come back null, not skip the other measures");
    console.assert(allNullResult.shoulderDrop.bow === 40.5, "a sibling measure with real readings must still get its own median even when bowArmAngle is entirely null");

    // The actual bias fix, demonstrated directly: an obvious one-frame excursion (the shape of a
    // torso-length underestimate spiking hand separation for a single frame — see Part 1) must
    // NOT move the reported value the way the old peak-selection rule did. Sixteen frames sitting
    // tight around bowArmAngle 170, plus ONE wild outlier frame at 200 with the highest handSep in
    // the set (so the OLD rule would have picked exactly that frame and reported 200).
    const excursionFrames = [];
    for (let i = 0; i < 16; i++) excursionFrames.push({ handSep: 0.8, bowArmAngle: 170, shoulderDrop: { bow: 45, draw: 45 }, elbowAlign: { signed: 0 }, anchorOk: true, armOk: true, sepOk: true, stillOk: true, atFullDraw: true });
    excursionFrames.push({ handSep: 0.99, bowArmAngle: 200, shoulderDrop: { bow: 80, draw: 80 }, elbowAlign: { signed: 30 }, anchorOk: true, armOk: true, sepOk: true, stillOk: true, atFullDraw: true });
    const oldRuleWouldPick = excursionFrames.reduce((best, f) => (f.handSep >= best.handSep ? f : best));
    console.assert(oldRuleWouldPick.bowArmAngle === 200, "sanity check: the fixture's excursion frame really is the one the old peak-handSep rule would have picked");
    const excursionResult = medianSampleOf(excursionFrames);
    console.assert(excursionResult.bowArmAngle === 170, `the median must land on the tight cluster (170), not the excursion (200) the old rule would have reported, got ${excursionResult.bowArmAngle}`);
    console.assert(excursionResult.shoulderDrop.bow === 45, "same for shoulder drop: median stays at the cluster (45), not the excursion (80)");

    // reservoirAdd: bounds memory (never grows past MEDIAN_SAMPLE_CAP) no matter how many eligible
    // frames one attempt sees, while eligibleSeen keeps an honest count of the true total.
    const reservoirAttempt = { eligibleFrames: [], eligibleSeen: 0 };
    const totalFed = MEDIAN_SAMPLE_CAP + 137;
    for (let i = 0; i < totalFed; i++) reservoirAdd(reservoirAttempt, { handSep: i / totalFed, bowArmAngle: 170, shoulderDrop: { bow: 45, draw: 45 }, elbowAlign: { signed: 0 } });
    console.assert(reservoirAttempt.eligibleFrames.length === MEDIAN_SAMPLE_CAP, `reservoir must cap at MEDIAN_SAMPLE_CAP (${MEDIAN_SAMPLE_CAP}) frames, has ${reservoirAttempt.eligibleFrames.length}`);
    console.assert(reservoirAttempt.eligibleSeen === totalFed, "eligibleSeen must keep counting every frame offered, even ones the reservoir didn't keep");
    // Every kept frame's handSep is a real value from the fed stream (0..1, i/totalFed) — the
    // simplest smoke check that eviction swaps in real items rather than corrupting slots.
    console.assert(
      reservoirAttempt.eligibleFrames.every((f) => f.handSep >= 0 && f.handSep < 1),
      "every frame the reservoir kept should still be one of the real fed samples, not a corrupted or missing slot"
    );
  }

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
  const drop1 = shoulderDropOf(dropLm1, L_SHOULDER, L_EAR, R_EAR, L_HIP, R_SHOULDER, R_HIP, NOOP_W, NOOP_H);
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
  const drop2 = shoulderDropOf(dropLm2, L_SHOULDER, L_EAR, R_EAR, L_HIP, R_SHOULDER, R_HIP, NOOP_W, NOOP_H);
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
    shoulderDropOf(dropLm3, L_SHOULDER, L_EAR, R_EAR, L_HIP, R_SHOULDER, R_HIP, NOOP_W, NOOP_H) === null,
    "shoulder drop should be uncertain (null), not a guess, when both ears are occluded"
  );

  const dropLm4 = mkLandmarks({
    ...dropHips,
    11: { x: 0, y: 0, visibility: 0 }, // shoulder itself occluded
    7: { x: 0.3, y: 0.3 },
    8: { x: 0.7, y: 0.3 },
  });
  console.assert(
    shoulderDropOf(dropLm4, L_SHOULDER, L_EAR, R_EAR, L_HIP, R_SHOULDER, R_HIP, NOOP_W, NOOP_H) === null,
    "shoulder drop should be uncertain (null) when the shoulder itself isn't visible"
  );

  // --- Feature B: draw-elbow alignment. angleAt gives the deviation magnitude; direction
  // (high/low) must come from vertical position only. The critical case: mirroring the whole
  // geometry left-right (as handedness or the front camera both do) must NOT flip high/low,
  // even though it would flip the sign of a cross product — which is exactly why this isn't
  // implemented with one.
  rightHanded = true;
  const inLine = mkLandmarks({ 15: { x: 0.0, y: 0.5 }, 16: { x: 0.5, y: 0.5 }, 14: { x: 1.0, y: 0.5 } });
  const inLineResult = drawElbowAlignmentOf(inLine, NOOP_W, NOOP_H);
  console.assert(
    inLineResult !== null && Math.abs(inLineResult.deviation) < 0.01,
    "elbow exactly on the extended bow-wrist -> draw-wrist line should read as ~0° deviation"
  );

  const highRH = mkLandmarks({ 15: { x: 0.0, y: 0.5 }, 16: { x: 0.5, y: 0.5 }, 14: { x: 1.0, y: 0.4 } });
  const highRHResult = drawElbowAlignmentOf(highRH, NOOP_W, NOOP_H);
  console.assert(
    highRHResult !== null && highRHResult.direction === "high" && highRHResult.deviation > 5,
    "elbow physically higher than the extended line (smaller y) should report direction: high"
  );

  const lowRH = mkLandmarks({ 15: { x: 0.0, y: 0.5 }, 16: { x: 0.5, y: 0.5 }, 14: { x: 1.0, y: 0.6 } });
  const lowRHResult = drawElbowAlignmentOf(lowRH, NOOP_W, NOOP_H);
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
  const highLHResult = drawElbowAlignmentOf(highLH, NOOP_W, NOOP_H);
  console.assert(
    highLHResult !== null && highLHResult.direction === "high",
    "mirrored (left-handed) geometry: elbow still physically higher must still report high, not low"
  );

  const lowLH = mkLandmarks({
    16: { x: 1.0, y: 0.5 },
    15: { x: 0.5, y: 0.5 },
    13: { x: 0.0, y: 0.6 }, // still physically lower
  });
  const lowLHResult = drawElbowAlignmentOf(lowLH, NOOP_W, NOOP_H);
  console.assert(
    lowLHResult !== null && lowLHResult.direction === "low",
    "mirrored (left-handed) geometry: elbow still physically lower must still report low, not high"
  );

  rightHanded = true;
  const vertical = mkLandmarks({ 15: { x: 0.5, y: 0.2 }, 16: { x: 0.5, y: 0.5 }, 14: { x: 0.5, y: 0.8 } });
  console.assert(
    drawElbowAlignmentOf(vertical, NOOP_W, NOOP_H) === null,
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
    const steadyResult = narrateMeasure(steadyFixture, (e) => e.bowArmAngle, "Bow arm", upDown, BOW_ARM_CONSISTENCY_FLOOR_DEG);
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
    const outlierNarrResult = narrateMeasure(outlierNarrFixture, (e) => e.bowArmAngle, "Bow arm", upDown, BOW_ARM_CONSISTENCY_FLOOR_DEG);
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
    const driftResult = narrateMeasure(driftFixture, (e) => e.bowArmAngle, "Bow arm", upDown, BOW_ARM_CONSISTENCY_FLOOR_DEG);
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
    const shuffledResult = narrateMeasure(shuffledFixture, (e) => e.bowArmAngle, "Bow arm", upDown, BOW_ARM_CONSISTENCY_FLOOR_DEG);
    console.assert(
      shuffledResult && !shuffledResult.text.includes("drifted"),
      `the same values in shuffled (non-trending) order must not read as drift, got: ${shuffledResult && shuffledResult.text}`
    );

    // --- MEASUREMENT PRECISION FLOOR: the field bug this was built to fix. A body held
    // completely still still shows a point or two of noise (see the MEASUREMENT PRECISION
    // FLOORS block above) — these are the coordinator's own reported numbers, bow-side shoulder
    // drop reading 42, 41, 40, 40 with nothing actually happening. Relative to THIS session's own
    // (tiny) scatter, that gap clears DRIFT_GAP_FACTOR easily; it must still come out "steady"
    // because it never clears SHOULDER_BOW_CONSISTENCY_FLOOR_PCT.
    const noiseOnlyFixture = [
      { shotNum: 1, shoulderDrop: { bow: 42 } },
      { shotNum: 2, shoulderDrop: { bow: 41 } },
      { shotNum: 3, shoulderDrop: { bow: 40 } },
      { shotNum: 4, shoulderDrop: { bow: 40 } },
    ];
    const noiseOnlyStats = narrateMeasure(noiseOnlyFixture, (e) => e.shoulderDrop?.bow ?? null, "Bow shoulder", upDown, SHOULDER_BOW_CONSISTENCY_FLOOR_PCT);
    console.assert(
      noiseOnlyStats && noiseOnlyStats.text.startsWith("Bow shoulder — steady"),
      `a 1-2 point wobble on a tight, unmoving session must read as steady, not drift, got: ${noiseOnlyStats && noiseOnlyStats.text}`
    );
    console.assert(noiseOnlyStats.outlierShotNum === null, "measurement noise must not get named as an outlier shot either");
    // Sanity check on the fixture itself: confirm the RELATIVE test alone (no floor) really would
    // have called this drift, so the assertion above is actually exercising the floor and not
    // just a fixture that never triggered the relative test in the first place.
    const noiseOnlyNoFloor = narrateMeasure(noiseOnlyFixture, (e) => e.shoulderDrop?.bow ?? null, "Bow shoulder", upDown, 0);
    console.assert(
      noiseOnlyNoFloor && noiseOnlyNoFloor.text.includes("drifted"),
      `sanity check failed: this fixture should clear the relative drift test on its own (floor 0) — got: ${noiseOnlyNoFloor && noiseOnlyNoFloor.text}, so the floor test above isn't testing what it claims to`
    );

    // Same idea, the outlier side: one shot a hair off a tight cluster clears the relative
    // outlier test but must still be called steady once it can't clear the floor either.
    const noiseOutlierFixture = [
      { shotNum: 1, shoulderDrop: { bow: 40 } },
      { shotNum: 2, shoulderDrop: { bow: 40 } },
      { shotNum: 3, shoulderDrop: { bow: 40 } },
      { shotNum: 4, shoulderDrop: { bow: 40.75 } },
    ];
    const noiseOutlierStats = narrateMeasure(noiseOutlierFixture, (e) => e.shoulderDrop?.bow ?? null, "Bow shoulder", upDown, SHOULDER_BOW_CONSISTENCY_FLOOR_PCT);
    console.assert(
      noiseOutlierStats && noiseOutlierStats.text.startsWith("Bow shoulder — steady") && noiseOutlierStats.outlierShotNum === null,
      `a sub-point gap on a tight session must not be named as an outlier shot, got: ${noiseOutlierStats && JSON.stringify(noiseOutlierStats)}`
    );
    const noiseOutlierNoFloor = narrateMeasure(noiseOutlierFixture, (e) => e.shoulderDrop?.bow ?? null, "Bow shoulder", upDown, 0);
    console.assert(
      noiseOutlierNoFloor && noiseOutlierNoFloor.outlierShotNum === 4,
      `sanity check failed: this fixture should clear the relative outlier test on its own (floor 0) — got: ${noiseOutlierNoFloor && JSON.stringify(noiseOutlierNoFloor)}, so the floor test above isn't testing what it claims to`
    );

    // The floor must not swallow a REAL trend or a REAL outlier, only noise-scale ones — both
    // driftResult and outlierNarrResult above already exercise this (they pass a real,
    // production floor and still correctly report drift/outlier), restated explicitly here.
    console.assert(
      driftResult.text.includes("drifted"),
      "a genuine drift that clears the floor must still be reported as drift, not swallowed by it"
    );
    console.assert(
      outlierNarrResult.outlierShotNum === 5,
      "a genuine outlier that clears the floor must still be named, not swallowed by it"
    );

    // One shot, or two shots: no honest consistency story to tell yet — must say so plainly and
    // never compute a fabricated steady/drift/outlier claim from almost nothing.
    const oneNarr = narrateMeasure([{ shotNum: 1, bowArmAngle: 170 }], (e) => e.bowArmAngle, "Bow arm", upDown, BOW_ARM_CONSISTENCY_FLOOR_DEG);
    console.assert(
      oneNarr && oneNarr.text.includes("only one shot") && oneNarr.outlierShotNum === null,
      `a single shot must produce the honest one-shot wording, not a fabricated claim, got: ${oneNarr && oneNarr.text}`
    );
    const twoNarr = narrateMeasure(
      [{ shotNum: 1, bowArmAngle: 170 }, { shotNum: 2, bowArmAngle: 170 }],
      (e) => e.bowArmAngle,
      "Bow arm",
      upDown,
      BOW_ARM_CONSISTENCY_FLOOR_DEG
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
    const allUncertainResult = narrateMeasure(allUncertainFixture, (e) => e.shoulderDrop?.draw ?? null, "Draw shoulder", upDown, SHOULDER_DRAW_CONSISTENCY_FLOOR_PCT);
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

  // --- Clip failure explaining: every distinct way a recording can come back with nothing
  // usable now writes a specific reason onto its shot's row (see explainClipFailure) instead of
  // silently leaving a bare "no clip" — the field bug this rewrite was built from (two real shots
  // on the owner's iPhone, both silently "no clip", no banner at all) is exactly what a silent
  // return used to produce. Exercised here by calling resolveClipOutcome / explainClipFailure /
  // attachRecordingToShot directly against fabricated `rec` objects — same "test the bookkeeping,
  // not a real MediaRecorder" approach as the block above, and for the same reason (no real camera
  // or recorder here, and startClipRecording is deliberately suppressed during selfTest).
  {
    const fakeRec = (overrides) => ({
      recorder: { mimeType: "video/webm" },
      chunks: [],
      shotNum: null,
      settled: false,
      failReason: null,
      ...overrides,
    });

    // A successful recording (real, non-empty chunks, a known shot number) attaches normally,
    // leaves no failure reason behind, and must never raise the clips-unavailable banner. Reset
    // to a known-clean null here as test SETUP, not a save/restore -- the whole function's outer
    // savedClipsUnavailableReason (see the top of selfTest) is what actually restores the real
    // original value once, at the very end; nothing in between needs to preserve it, only start
    // each sub-test from a clean slate so its own assertions mean what they say.
    log = [{ shotNum: 40 }];
    clipsUnavailableReason = null;
    resolveClipOutcome(fakeRec({ chunks: [new Blob(["x"], { type: "video/webm" })], shotNum: 40 }));
    const row40 = log.find((e) => e.shotNum === 40);
    console.assert(row40.clipUrl && !row40.clipFailReason, "a recording with real data and a known shot number should attach a clip, not a failure reason");
    console.assert(clipsUnavailableReason === null, "a successful recording must never raise the clips-unavailable banner");
    URL.revokeObjectURL(row40.clipUrl);

    // Empty chunks -- the field bug's actual shape: recording believed it started fine and simply
    // never produced usable video -- gets its own explanation on the row, AND raises the same
    // session banner a synchronous recorder failure already raises (see markClipsUnavailable);
    // from the owner's side a clip that silently comes back empty IS a clip that failed.
    log = [{ shotNum: 41 }];
    resolveClipOutcome(fakeRec({ chunks: [], shotNum: 41 }));
    const row41 = log.find((e) => e.shotNum === 41);
    console.assert(
      row41.clipFailReason === "no clip — recording came out empty",
      `an empty recording should explain itself on its row, got: ${row41.clipFailReason}`
    );
    console.assert(
      clipsUnavailableReason === "Some shots couldn't be recorded — at least one clip failed this session.",
      "a recording that came back empty should raise the same banner a synchronous recorder failure would"
    );

    // A more specific reason set earlier (e.g. by the track-ended listener or onerror in
    // startClipRecording) wins over the generic "came out empty" fallback, and the FIRST reason
    // found for a shot is the one that sticks -- a later cause must never overwrite it.
    log = [{ shotNum: 42 }];
    resolveClipOutcome(fakeRec({ chunks: [], shotNum: 42, failReason: "no clip — camera feed cut out" }));
    console.assert(
      log.find((e) => e.shotNum === 42).clipFailReason === "no clip — camera feed cut out",
      "a specific failure reason set before resolveClipOutcome runs should win over the generic empty-recording message"
    );
    explainClipFailure(42, "no clip — recorder error");
    console.assert(
      log.find((e) => e.shotNum === 42).clipFailReason === "no clip — camera feed cut out",
      "the first failure reason found for a shot must stick -- a later cause must not overwrite it"
    );

    // resolveClipOutcome must be idempotent: a real onstop firing and the stop-watchdog giving up
    // can both end up calling it for the same recording (see finalizeRecording), and it must only
    // ever be processed once.
    log = [{ shotNum: 43 }];
    const rec43 = fakeRec({ chunks: [], shotNum: 43 });
    resolveClipOutcome(rec43);
    log.find((e) => e.shotNum === 43).clipFailReason = "sentinel"; // stand-in for whatever the row legitimately held after the first call
    resolveClipOutcome(rec43); // rec43.settled is already true -- this must be a complete no-op
    console.assert(
      log.find((e) => e.shotNum === 43).clipFailReason === "sentinel",
      "resolveClipOutcome must not process the same recording's outcome twice"
    );

    // A recording that resolves before its attempt has logged a shot (shotNum still null -- only
    // happens when CLIP_MAX_MS's safety cap cuts a stuck recording off before endAttempt has run)
    // has no row yet to explain itself on: the reason is remembered in pendingClipNote, and the
    // NEXT shot logged (via attachRecordingToShot, exactly as endAttempt calls it) picks it up
    // instead of that shot showing a bare, unexplained "no clip".
    pendingClipNote = null;
    resolveClipOutcome(fakeRec({ chunks: [], shotNum: null }));
    console.assert(
      pendingClipNote === "no clip — recording came out empty",
      `a recording resolving before its shot is logged should leave a pending note, got: ${pendingClipNote}`
    );
    log = [{ shotNum: 44 }];
    attachRecordingToShot(44); // activeRecording is null here -- the "recording already ended" path
    console.assert(
      log.find((e) => e.shotNum === 44).clipFailReason === "no clip — recording came out empty",
      "a pending note left by an earlier-resolved recording should attach to the next shot logged"
    );
    console.assert(pendingClipNote === null, "a pending note should be consumed, not reused, once it's attached to a shot");
  }

  // --- Clip false-positive lock-down: a discarded (thrown-away) attempt's recording must NEVER
  // look like a clip failure. Written after a real false positive was caught in review: a single
  // clean, successful shot raised the "at least one clip failed" banner anyway. Root cause: a
  // landmark-noise blip lasting one or two frames is easily long enough to cross
  // DRAW_ATTEMPT_MIN_SEP and start a real recording, but far too short for
  // canvas.captureStream/MediaRecorder to ever encode a single frame before endAttempt's own
  // gates threw the attempt away — and that empty-chunks outcome used to read as a genuine
  // recording failure. These two cases are exactly what the coordinator asked this fix be locked
  // down against, so a regression here fails loudly instead of quietly shipping the same bug
  // back to the owner's phone.
  {
    // Case 1: one clean, successful shot, nothing else in the session. Must produce no failure
    // text anywhere -- no banner, no per-row reason. clipsUnavailableReason is reset to null here
    // as test SETUP, not a local save/restore -- see selfTest's own outer
    // savedClipsUnavailableReason, which is what restores the true original value once, at the
    // very end of the whole run (a local save/restore here is exactly the pattern that let this
    // bug slip through the first time: it only ever covered part of a block).
    log = [{ shotNum: 60 }];
    clipsUnavailableReason = null;
    resolveClipOutcome({
      recorder: { mimeType: "video/webm" }, chunks: [new Blob(["a"], { type: "video/webm" })], shotNum: 60,
      settled: false, failReason: null, discarded: false,
    });
    const row60 = log.find((e) => e.shotNum === 60);
    console.assert(row60.clipUrl && !row60.clipFailReason, "a single successful shot must attach a clip with no failure reason on its row");
    console.assert(clipsUnavailableReason === null, "a single successful shot must never raise the clips-unavailable banner");
    URL.revokeObjectURL(row60.clipUrl);

    // Case 2: a rejected movement (its recording resolves with EMPTY chunks -- the realistic
    // shape, since a movement too brief/shallow to count as a shot is also too brief for the
    // recorder to have encoded anything) alongside a real, successful shot in the same session.
    // Neither the discarded recording NOR the real one may show any failure text, and the real
    // shot's clip must attach exactly as if the discarded one never existed.
    log = [{ shotNum: 61 }];
    clipsUnavailableReason = null;
    pendingClipNote = null;
    resolveClipOutcome({
      recorder: { mimeType: "video/webm" }, chunks: [], shotNum: null,
      settled: false, failReason: null, discarded: true, // marked by discardRecording -- see endAttempt's rejected/unsettled branches
    });
    console.assert(clipsUnavailableReason === null, "a discarded (thrown-away) attempt's empty recording must never raise the clips-unavailable banner");
    console.assert(pendingClipNote === null, "a discarded attempt's recording must never leave a pending note behind for a later, unrelated shot to inherit");
    resolveClipOutcome({
      recorder: { mimeType: "video/webm" }, chunks: [new Blob(["b"], { type: "video/webm" })], shotNum: 61,
      settled: false, failReason: null, discarded: false,
    });
    const row61 = log.find((e) => e.shotNum === 61);
    console.assert(
      row61.clipUrl && !row61.clipFailReason,
      "a real shot's clip must attach normally even when a rejected movement's recording resolved earlier in the same session"
    );
    console.assert(clipsUnavailableReason === null, "a session with one rejected movement and one real, successful shot must show no clips-unavailable banner");
    URL.revokeObjectURL(row61.clipUrl);
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

  // --- ASPECT-RATIO CORRECTNESS: the bug all the toPixelSpace conversions above exist to fix.
  // MediaPipe normalises x by frame WIDTH and y by frame HEIGHT independently, so raw x/y are
  // only the same physical units when the frame is square — and a phone's camera frame never is
  // (e.g. roughly 720x1280 held upright). Left uncorrected, every distance and angle in this file
  // is stretched along whichever axis the frame happens to be narrower on: a field shot log once
  // showed hand separation reading 2.31-2.32 torso-lengths, ~3x a physically possible full draw,
  // traced to exactly this. This section builds ONE synthetic archer directly in real-world-
  // proportional units (no camera involved — this IS the ground truth, not a MediaPipe fixture),
  // projects it into normalised landmarks the way a real camera+MediaPipe would for two very
  // different frame shapes (a phone held upright, and the same phone on its side), and checks
  // that every measure this file reports (a) matches the known-correct answer computed straight
  // from those real-world units, in BOTH orientations, and (b) — the actual property that broke
  // in the field — reads the SAME regardless of which way the phone was held.
  {
    // Real-world-proportional coordinates (one arbitrary consistent unit — only ratios and
    // angles between these points are ever checked, never their absolute size). Convention
    // matches the rest of this file: x grows toward the bow-arm/target side, y grows DOWNWARD
    // (image convention), so a SMALLER y is physically HIGHER up.
    const shoulder = { x: 0, y: 0 }; // bow shoulder and draw shoulder both land here — the same "shoulder line is nearly degenerate" simplification CLAUDE.md describes for a true side-on view
    const hip = { x: 0, y: 0.5 }; // torso length 0.5 units, purely vertical — bow hip and draw hip both here too
    const bowElbow = { x: 0.3, y: 0 }; // collinear with shoulder and bowWrist below: a genuinely straight bow arm, 180° by construction
    const bowWrist = { x: 0.6, y: 0 };
    const drawWrist = { x: 0.05, y: -0.18 }; // anchor near the face: close to the centreline, higher than the shoulder
    const mouth = { x: 0.0, y: -0.2 }; // near, not on top of, drawWrist — a small but real (nonzero) anchor distance
    const bowEar = { x: 0, y: -0.1 };
    const drawEar = { x: 0, y: -0.15 };
    // Deliberately NOT on the bow-wrist -> draw-wrist line: physically ABOVE it by a known
    // amount, so both the deviation magnitude and the high/low direction are real, checkable
    // facts about this fixture, not accidents of whatever numbers happened to be picked.
    const drawElbow = { x: -0.225, y: -0.35 };

    // Ground truth: plain vector maths on the real-world coordinates above, no camera or
    // normalisation involved anywhere — this is what a perfect measurement would report, and
    // exactly what the corrected functions below are held to, in both orientations.
    const trueTorso = Math.hypot(shoulder.x - hip.x, shoulder.y - hip.y);
    const trueBowArmAngle = angleAt(shoulder, bowElbow, bowWrist);
    console.assert(Math.abs(trueBowArmAngle - 180) < 1e-9, "fixture sanity check: the bow arm was meant to be built perfectly straight");
    const trueHandSepRatio = Math.hypot(bowWrist.x - drawWrist.x, bowWrist.y - drawWrist.y) / trueTorso;
    const trueAnchorRatio = Math.hypot(drawWrist.x - mouth.x, drawWrist.y - mouth.y) / trueTorso;
    const trueBowDropPct = ((shoulder.y - bowEar.y) / trueTorso) * 100;
    const trueDrawDropPct = ((shoulder.y - drawEar.y) / trueTorso) * 100;
    const trueElbowAngle = angleAt(bowWrist, drawWrist, drawElbow);
    const trueElbowDeviation = 180 - trueElbowAngle;
    const elbowDx = drawWrist.x - bowWrist.x;
    const elbowT = (drawElbow.x - bowWrist.x) / elbowDx;
    const elbowExpectedY = bowWrist.y + elbowT * (drawWrist.y - bowWrist.y);
    const trueElbowDirection = drawElbow.y < elbowExpectedY ? "high" : drawElbow.y > elbowExpectedY ? "low" : "level";
    console.assert(trueElbowDirection === "high", "fixture sanity check: the elbow was meant to be built physically higher than the extended line");

    // Projects one real-world point into normalised [0,1] landmark coordinates for a given frame
    // size, the way a real camera + MediaPipe would: the SAME physical scale (pixels per unit
    // distance) on BOTH axes — exactly what a real lens does at a given distance from the subject
    // — then normalised independently by the frame's own width and height. That last step, and
    // only that last step, is what makes a non-square frame distort raw x/y differently — it's
    // the entire bug toPixelSpace exists to undo, reproduced here on purpose so this test can
    // catch a regression of it.
    const ASPECT_PX_PER_UNIT = 400; // keeps every projected point comfortably inside [0,1] for both frame shapes below
    function projectPhysical(p, frameWidth, frameHeight) {
      return {
        x: (frameWidth / 2 + p.x * ASPECT_PX_PER_UNIT) / frameWidth,
        y: (frameHeight / 2 + p.y * ASPECT_PX_PER_UNIT) / frameHeight,
        visibility: 1,
      };
    }
    function buildAspectLandmarks(frameWidth, frameHeight) {
      const lm = Array.from({ length: 25 }, () => ({ x: 0.5, y: 0.5, visibility: 0 }));
      const put = (idx, p) => { lm[idx] = projectPhysical(p, frameWidth, frameHeight); };
      put(L_SHOULDER, shoulder); put(R_SHOULDER, shoulder);
      put(L_HIP, hip); put(R_HIP, hip);
      put(L_ELBOW, bowElbow); put(R_ELBOW, drawElbow);
      put(L_WRIST, bowWrist); put(R_WRIST, drawWrist);
      put(MOUTH_L, mouth); put(MOUTH_R, mouth);
      put(L_EAR, bowEar); put(R_EAR, drawEar);
      return lm;
    }

    // A phone held upright (portrait: narrower than it is tall) and the same phone on its side
    // (landscape) — the two shapes an iPhone's camera actually produces, at roughly the same
    // pixel count either way. rightHanded is already true from earlier in selfTest.
    const portrait = buildAspectLandmarks(720, 1280);
    const landscape = buildAspectLandmarks(1280, 720);

    const TOL = 1e-4; // far tighter than the 30-300%+ distortion this section exists to catch; comfortably loose for ordinary floating-point rounding

    for (const [label, lm, w, h] of [["portrait", portrait, 720, 1280], ["landscape", landscape, 1280, 720]]) {
      const bowArm = bowArmAngleOf(lm, w, h);
      console.assert(
        Math.abs(bowArm - 180) < TOL,
        `${label}: a physically straight bow arm must still read as ~180° once corrected for aspect ratio, got ${bowArm}`
      );

      const scale = torsoLength(lm, R_SHOULDER, R_HIP, w, h);
      const wristPx = toPixelSpace(lm[R_WRIST], w, h);
      const bowWristPx = toPixelSpace(lm[L_WRIST], w, h);
      const mouthPx = toPixelSpace(lm[MOUTH_L], w, h); // MOUTH_L === MOUTH_R here, either works
      const handSepRatio = Math.hypot(wristPx.x - bowWristPx.x, wristPx.y - bowWristPx.y) / scale;
      const anchorRatio = Math.hypot(wristPx.x - mouthPx.x, wristPx.y - mouthPx.y) / scale;
      console.assert(
        Math.abs(handSepRatio - trueHandSepRatio) < TOL,
        `${label}: corrected hand-separation ratio should match the real-world ground truth (${trueHandSepRatio.toFixed(6)}), got ${handSepRatio.toFixed(6)}`
      );
      console.assert(
        Math.abs(anchorRatio - trueAnchorRatio) < TOL,
        `${label}: corrected anchor-distance ratio should match the real-world ground truth (${trueAnchorRatio.toFixed(6)}), got ${anchorRatio.toFixed(6)}`
      );

      const { bow: bowDrop, draw: drawDrop } = shoulderDropSampleOf(lm, w, h);
      console.assert(
        Math.abs(bowDrop - trueBowDropPct) < TOL,
        `${label}: corrected bow-shoulder drop should match ground truth (${trueBowDropPct.toFixed(6)}%), got ${bowDrop}`
      );
      console.assert(
        Math.abs(drawDrop - trueDrawDropPct) < TOL,
        `${label}: corrected draw-shoulder drop should match ground truth (${trueDrawDropPct.toFixed(6)}%), got ${drawDrop}`
      );

      const elbow = drawElbowAlignmentOf(lm, w, h);
      console.assert(
        elbow !== null && Math.abs(elbow.deviation - trueElbowDeviation) < TOL,
        `${label}: corrected elbow deviation should match ground truth (${trueElbowDeviation.toFixed(6)}°), got ${elbow && elbow.deviation}`
      );
      console.assert(
        elbow !== null && elbow.direction === trueElbowDirection,
        `${label}: corrected elbow direction should match ground truth (${trueElbowDirection}), got ${elbow && elbow.direction}`
      );
    }

    // The actual property that broke in the field: the SAME real body, measured through a
    // portrait frame and through a landscape frame, must read the SAME — not two different
    // numbers depending on which way the phone happened to be held. (Each side already matched
    // ground truth above; this restates the comparison directly between the two orientations so
    // a future regression that broke both sides identically — e.g. a wrong-but-consistent
    // conversion — still gets caught here even though it wouldn't be caught by comparing each
    // side to ground truth alone... except ground truth is independent and correct, so in
    // practice the loop above already proves this. Kept anyway as the single most direct
    // statement of the property this whole fix is for.)
    const portraitBowArm = bowArmAngleOf(portrait, 720, 1280);
    const landscapeBowArm = bowArmAngleOf(landscape, 1280, 720);
    console.assert(
      Math.abs(portraitBowArm - landscapeBowArm) < TOL,
      `bow-arm angle must be orientation-invariant: portrait ${portraitBowArm}, landscape ${landscapeBowArm}`
    );

    // Before/after, on this exact same body, in the portrait frame specifically (the shape that
    // triggered the field bug): the UNCORRECTED formula — raw Math.hypot straight on normalised
    // x/y, exactly what every measure in this file used to do — reproduced here only for this
    // side-by-side comparison, since the corrected functions above no longer have a way to skip
    // the conversion.
    const buggyPortraitTorso = Math.hypot(
      portrait[R_SHOULDER].x - portrait[R_HIP].x,
      portrait[R_SHOULDER].y - portrait[R_HIP].y
    );
    const buggyPortraitHandSepRatio =
      Math.hypot(portrait[R_WRIST].x - portrait[L_WRIST].x, portrait[R_WRIST].y - portrait[L_WRIST].y) / buggyPortraitTorso;
    const correctedPortraitHandSepRatio =
      Math.hypot(
        toPixelSpace(portrait[R_WRIST], 720, 1280).x - toPixelSpace(portrait[L_WRIST], 720, 1280).x,
        toPixelSpace(portrait[R_WRIST], 720, 1280).y - toPixelSpace(portrait[L_WRIST], 720, 1280).y
      ) / torsoLength(portrait, R_SHOULDER, R_HIP, 720, 1280);
    console.log(
      `ASPECT-RATIO fix, same synthetic body, 720x1280 portrait frame: hand-separation ratio was ${buggyPortraitHandSepRatio.toFixed(3)} (uncorrected) -> ${correctedPortraitHandSepRatio.toFixed(3)} (corrected); real-world ground truth is ${trueHandSepRatio.toFixed(3)}.`
    );
    console.assert(
      buggyPortraitHandSepRatio > correctedPortraitHandSepRatio * 1.3,
      `the uncorrected ratio should be substantially inflated relative to the corrected one in a portrait frame — got uncorrected ${buggyPortraitHandSepRatio.toFixed(3)} vs corrected ${correctedPortraitHandSepRatio.toFixed(3)}`
    );
    console.assert(
      Math.abs(correctedPortraitHandSepRatio - trueHandSepRatio) < TOL,
      "corrected portrait hand-separation ratio should match ground truth"
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

  // --- ROUTINE-START ATTENTION GATING: attentionIsClearlyCalm first, as a pure function, then
  // updateAttentionState as the state machine built on top of it. Reuses mkLandmarks/base from
  // above — same shared skeleton scale (torso length 0.3), same rightHanded === true.
  {
    const restLandmarks = mkLandmarks({
      ...base,
      15: { x: 0.4, y: 0.3 }, // bow wrist
      16: { x: 0.42, y: 0.3 }, // draw wrist, right next to it — hands relaxed together
    });
    console.assert(
      attentionIsClearlyCalm(null, null, 0) === true,
      "no landmarks at all (nobody in frame) should read as clearly calm — nothing to be shooting"
    );
    console.assert(
      attentionIsClearlyCalm(restLandmarks, null, 0) === true,
      "relaxed hands, with no previous reference point to judge stillness against yet, should read as clearly calm"
    );

    const drawnLandmarks = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.52, y: 0.31 } });
    console.assert(
      attentionIsClearlyCalm(drawnLandmarks, null, 0) === false,
      "hands far apart (a real draw) must never read as clearly calm"
    );

    const noWristLandmarks = mkLandmarks({ ...base, 15: { x: 0.4, y: 0.3, visibility: 0 }, 16: { x: 0.42, y: 0.3 } });
    console.assert(
      attentionIsClearlyCalm(noWristLandmarks, null, 0) === false,
      "an invisible wrist can't confirm the hands are relaxed — must not read as clearly calm"
    );

    const calmRef = bodyReferencePoint(restLandmarks);
    console.assert(
      attentionIsClearlyCalm(restLandmarks, calmRef, 1) === true,
      "an unmoved body reference point over a full second should still read as clearly calm"
    );
    const farRef = { x: calmRef.x + 1, y: calmRef.y }; // ~3+ torso-lengths of drift in one second — unmistakably walking
    console.assert(
      attentionIsClearlyCalm(restLandmarks, farRef, 1) === false,
      "a body reference point that moved far in one second (walking) must not read as clearly calm, even with relaxed hands"
    );

    // Numeric invariants the state machine's hard rule and structural safety depend on — see
    // updateAttentionState's own comments for what each protects.
    console.assert(
      ATTENTION_REST_HAND_SEP_MAX < DRAW_ATTEMPT_MIN_SEP,
      "ATTENTION_REST_HAND_SEP_MAX must stay below DRAW_ATTEMPT_MIN_SEP, or a real in-progress attempt could read as 'calm' and be allowed to idle"
    );
    console.assert(
      ATTENTION_IDLE_SAMPLE_INTERVAL_MS < SHOT_MIN_DURATION_MS,
      "ATTENTION_IDLE_SAMPLE_INTERVAL_MS must stay below SHOT_MIN_DURATION_MS — otherwise a genuine draw could start and finish entirely inside a single blind idle-sampling gap and never be noticed by any sample at all"
    );

    // --- updateAttentionState: the state machine. Fresh, isolated state for this block; restored
    // (like everything else in this file's selfTest) from the outer save/restore at the very end.
    attempt = null;
    attentionEngaged = true;
    attentionCalmSinceMs = null;
    attentionLastIdleSampleMs = null;
    attentionLastEvalMs = null;
    attentionPrevRef = null;
    attentionIdlePeriods = 0;
    attentionLateWakeCount = 0;
    settledFrames = SETTLE_FRAMES_REQUIRED; // pretend already settled, so the reset-on-re-engage effect below is actually observable
    currentCropBox = { x: 10, y: 10, size: 50 };
    prevUsedCropBox = { x: 10, y: 10, size: 50 };

    const calmLm = restLandmarks;
    const drawLm = drawnLandmarks;
    // Thin wrapper so the state-machine tests below don't have to pass `true` for `modelReady`
    // on every call — this whole block is about the idle/engage logic itself, not the pose-model
    // warm-up guard (covered separately: production renderLoop always defaults modelReady to the
    // real modelDecisionMade flag). gatingEnabled still defaults to true and can be overridden
    // per call, same as updateAttentionState's own default.
    const callAttention = (nowMs, lm, gatingEnabled = true) => updateAttentionState(nowMs, lm, gatingEnabled, true);

    let t = 0;
    callAttention(t, calmLm);
    console.assert(attentionEngaged === true, "a single calm sample must not idle immediately — ATTENTION_IDLE_AFTER_MS hasn't elapsed yet");

    t += ATTENTION_IDLE_AFTER_MS - 1;
    callAttention(t, calmLm);
    console.assert(attentionEngaged === true, "calm held for just under the idle threshold must still be engaged");

    t += 2;
    callAttention(t, calmLm);
    console.assert(attentionEngaged === false, "calm held continuously past ATTENTION_IDLE_AFTER_MS should allow the app to idle");
    console.assert(attentionIdlePeriods === 1, "going idle should count as exactly one idle period");

    // --- Recovery ("it recovers from every idle state"): the very next sample that isn't clearly
    // calm must engage immediately, with no extra delay beyond this one sample.
    callAttention(t + 50, drawLm);
    console.assert(attentionEngaged === true, "a not-calm sample while idle must engage on that very sample — the routine-start case");

    // --- PIPELINE SETTLING interaction, worked through as the brief asked: re-engaging must
    // reset settling/smoothing/crop state, so a frame right after a long idle stretch is treated
    // exactly like a fresh reset — not as already settled just because it happened to be settled
    // before the app went idle.
    console.assert(settledFrames === 0, "re-engaging from idle should reset the settling counter, same as any other PIPELINE SETTLING recovery point");
    console.assert(currentCropBox === null, "re-engaging from idle should drop the crop box, same as any other recovery point");
    console.assert(prevUsedCropBox === null, "re-engaging from idle should clear prevUsedCropBox too — resetSettling's own job, reused here");

    // --- Cannot latch off: drive several idle/wake cycles back to back and confirm engaged is
    // ALWAYS true the instant a not-calm sample arrives, regardless of how many cycles came
    // before — no counter or cooldown anywhere in this function can make a wake-up "reluctant".
    let tt = t + 1000;
    for (let cycle = 0; cycle < 5; cycle++) {
      for (let i = 0; i < 5; i++) {
        tt += 400;
        callAttention(tt, calmLm);
      }
      tt += ATTENTION_IDLE_AFTER_MS + 100;
      callAttention(tt, calmLm);
      console.assert(attentionEngaged === false, `cycle ${cycle}: sustained calm should reach idle`);
      tt += 10;
      callAttention(tt, drawLm);
      console.assert(attentionEngaged === true, `cycle ${cycle}: the very next not-calm sample must engage — no latch-off, ever`);
    }

    // --- Fail toward recording: a genuinely AMBIGUOUS movement — hands apart enough not to be
    // relaxed, but nowhere near a real draw attempt (DRAW_ATTEMPT_MIN_SEP) — is exactly the kind
    // of thing nocking, adjusting a release aid, or turning partway could look like. It must
    // still engage, never idle through it: "treated as shooting, not discarded."
    const ambiguousLm = mkLandmarks({ ...base, 15: { x: 0.4, y: 0.3 }, 16: { x: 0.48, y: 0.32 } });
    const ambigScale = torsoLength(ambiguousLm, R_SHOULDER, R_HIP);
    const ambigHandSep = Math.hypot(ambiguousLm[16].x - ambiguousLm[15].x, ambiguousLm[16].y - ambiguousLm[15].y) / ambigScale;
    console.assert(
      ambigHandSep > ATTENTION_REST_HAND_SEP_MAX && ambigHandSep < DRAW_ATTEMPT_MIN_SEP,
      "the ambiguous fixture should sit strictly between 'clearly relaxed' and 'a real draw attempt', or this isn't testing the ambiguous case at all"
    );
    attentionEngaged = true;
    attentionCalmSinceMs = null;
    attentionLateWakeCount = 0; // the "cannot latch off" cycles above woke on drawLm repeatedly, each a late wake on its own terms — reset so this check is about THIS wake only
    let t2 = tt + 1000;
    callAttention(t2, calmLm);
    t2 += ATTENTION_IDLE_AFTER_MS + 50;
    callAttention(t2, calmLm);
    console.assert(attentionEngaged === false, "setup: should be idle before the ambiguous sample");
    callAttention(t2 + 10, ambiguousLm);
    console.assert(
      attentionEngaged === true,
      "an ambiguous movement — neither clearly relaxed nor a clean full draw — must engage, not stay idle: fail toward recording"
    );
    console.assert(attentionLateWakeCount === 0, "the ambiguous fixture is below DRAW_ATTEMPT_MIN_SEP, so waking on it must not count as a late wake");

    // --- Late-wake counting: a sample that wakes the app up with hands ALREADY past
    // DRAW_ATTEMPT_MIN_SEP means the real start of that movement happened sometime during the
    // idle gap just slept through — counted separately from rejectedAttemptCount/
    // unsettledAttemptCount (see the constant's own comment), so it must go up here.
    attentionEngaged = true;
    attentionCalmSinceMs = null;
    attentionLateWakeCount = 0; // isolate this specific wake's contribution from the ambiguous-wake check just above
    let t3 = t2 + 1000;
    callAttention(t3, calmLm);
    t3 += ATTENTION_IDLE_AFTER_MS + 50;
    callAttention(t3, calmLm);
    console.assert(attentionEngaged === false, "setup: should be idle again before the late-wake sample");
    callAttention(t3 + 10, drawLm);
    console.assert(attentionEngaged === true, "waking on an already-past-floor sample should still engage");
    console.assert(attentionLateWakeCount === 1, "waking up on a sample where hands are already past DRAW_ATTEMPT_MIN_SEP should count as exactly one late wake");

    // --- Hard rule: an attempt in progress must never be allowed to idle, no matter what a
    // single sample looks like. Can't actually happen given the ATTENTION_REST_HAND_SEP_MAX <
    // DRAW_ATTEMPT_MIN_SEP invariant above, but this is the explicit, defensive guarantee, not
    // one relying on remembering that numeric relationship forever.
    attentionEngaged = true;
    attentionCalmSinceMs = null;
    attempt = { startMs: 0, peakHandSep: 0.5, eligibleFrames: [], eligibleSeen: 0, reachedFullDraw: false };
    let t4 = t3 + 1000;
    for (let i = 0; i < 20; i++) {
      t4 += ATTENTION_IDLE_AFTER_MS / 4;
      callAttention(t4, calmLm); // feeding CALM landmarks on purpose — the hard rule must win regardless of the sample
      console.assert(attentionEngaged === true, "an attempt in progress must never be allowed to idle, no matter what the sample looks like");
    }
    attempt = null;

    // --- Master switch: gatingEnabled=false must force full-rate engagement unconditionally,
    // even starting from idle — proven directly via the explicit parameter (see
    // updateAttentionState's own comment on why it's threaded through rather than read from the
    // module constant, the same convention nowMs/frameEligible already use elsewhere).
    attentionEngaged = false;
    callAttention(t4 + 100, calmLm, false);
    console.assert(attentionEngaged === true, "gatingEnabled=false must force engaged, even starting from idle");
  }

  // Every one of these was BORROWED for the run, on the promise that it comes back exactly as it
  // went in — most of them (log, shotCount, lastDrawWrist, settledFrames, unsettledAttemptCount,
  // attempt...) are deliberately left "dirty" by the test blocks above right up until this point,
  // by design, so restoring is genuinely a one-shot operation here, not an invariant kept
  // throughout the run.
  selfTestInProgress = false;
  rightHanded = savedHanded;
  lastDrawWrist = savedLastWrist;
  attempt = savedAttempt;
  log = savedLog;
  shotCount = savedShotCount;
  rejectedAttemptCount = savedRejectedAttemptCount;
  unsettledAttemptCount = savedUnsettledAttemptCount;
  settledFrames = savedSettledFrames;
  prevUsedCropBox = savedPrevUsedCropBox;
  attentionEngaged = savedAttentionEngaged;
  attentionCalmSinceMs = savedAttentionCalmSinceMs;
  attentionLastIdleSampleMs = savedAttentionLastIdleSampleMs;
  attentionLastEvalMs = savedAttentionLastEvalMs;
  attentionPrevRef = savedAttentionPrevRef;
  attentionIdlePeriods = savedAttentionIdlePeriods;
  attentionLateWakeCount = savedAttentionLateWakeCount;
  clipsUnavailableReason = savedClipsUnavailableReason;
  pendingClipNote = savedPendingClipNote;

  // Lock-down: after the restore above, every borrowed variable must read back EXACTLY as it was
  // saved at the top of this function — a diagnostic mode that quietly leaves the live app
  // mutated is worse than no diagnostic mode at all, because it makes the app lie in precisely
  // the tool used to check whether it's lying. Written after a real leak reached exactly that
  // state: ?selftest left a false "clips failed" banner standing for the rest of that page load,
  // because clipsUnavailableReason had no restore line here at all — only a couple of local,
  // partial ones scattered inside individual test blocks, one of which covered only PART of its
  // own block and left a later sub-test's own re-triggering of the banner with nothing left to
  // undo it. These assertions won't catch a mid-run leak that the restore above also happens to
  // paper over (nothing here could, from JS — a straight assignment always "succeeds"), but they
  // do catch the much more likely future mistake: a new piece of borrowed state that gets a
  // `savedX` capture but no matching restore line above, or a restore line that assigns the wrong
  // saved value to the wrong variable. Add both a restore line above AND an assertion here
  // whenever a future test starts borrowing another piece of module state.
  console.assert(rightHanded === savedHanded, "selfTest leaked: rightHanded was not restored to its original value");
  console.assert(lastDrawWrist === savedLastWrist, "selfTest leaked: lastDrawWrist was not restored to its original value");
  console.assert(attempt === savedAttempt, "selfTest leaked: attempt was not restored to its original value");
  console.assert(log === savedLog, "selfTest leaked: log was not restored to its original value");
  console.assert(shotCount === savedShotCount, "selfTest leaked: shotCount was not restored to its original value");
  console.assert(rejectedAttemptCount === savedRejectedAttemptCount, "selfTest leaked: rejectedAttemptCount was not restored to its original value");
  console.assert(unsettledAttemptCount === savedUnsettledAttemptCount, "selfTest leaked: unsettledAttemptCount was not restored to its original value");
  console.assert(settledFrames === savedSettledFrames, "selfTest leaked: settledFrames was not restored to its original value");
  console.assert(prevUsedCropBox === savedPrevUsedCropBox, "selfTest leaked: prevUsedCropBox was not restored to its original value");
  console.assert(
    clipsUnavailableReason === savedClipsUnavailableReason,
    `selfTest leaked: clipsUnavailableReason was not restored to its original value (left as ${JSON.stringify(clipsUnavailableReason)}) — a diagnostic run must never leave a false "clips failed" banner behind`
  );
  console.assert(pendingClipNote === savedPendingClipNote, "selfTest leaked: pendingClipNote was not restored to its original value");

  console.log("selfTest done — check above for any failed console.assert");
}

// ===== TEST HOOKS — read-only introspection for automated (Playwright) verification of the
// ROUTINE-START ATTENTION GATING feature, behind an explicit URL flag so this never activates in
// normal use and never ships anything extra to the owner's phone in practice. Exposes exactly the
// module state an outside test needs to read — nothing this file doesn't already track for
// itself — and nothing to WRITE with: no setters, no way for a test to drive the app any
// differently than a real session would. window.__testHooks.getState() is a plain snapshot,
// safe to call as often as a test likes (e.g. to poll for a state transition).
const TEST_HOOKS = location.search.includes("testhooks");
if (TEST_HOOKS) {
  window.__testHooks = {
    getState: () => ({
      shotCount,
      logLength: log.length,
      log: log.map((e) => ({ shotNum: e.shotNum, handSep: e.handSep, reachedFullDraw: e.reachedFullDraw, hasClip: !!e.clipUrl })),
      rejectedAttemptCount,
      unsettledAttemptCount,
      attemptInProgress: attempt !== null,
      attentionEngaged,
      attentionIdlePeriods,
      attentionLateWakeCount,
      settledFrames,
      hasCropBox: currentCropBox !== null,
      attemptEligibleFrames: attempt ? attempt.eligibleFrames.length : null,
      attemptPeakHandSep: attempt ? attempt.peakHandSep : null,
    }),
  };
}
// ===========================================================================

if (location.search.includes("selftest")) selfTest();

main();
