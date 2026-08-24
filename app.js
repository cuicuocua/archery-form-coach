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
const FULL_DRAW_ANCHOR_MAX = 0.3; // draw-hand wrist must be this close to the mouth/nose, as a fraction of torso length, to count as "at anchor". MEASURED, not guessed: the owner read 0.24 at his own real anchor on 2026-08-24 using the ?triggertest inspector, and this leaves ~25% headroom over that. Was 0.45, which he reported as firing "even when i'm nowhere near my mouth" — 0.45 is a radius of roughly 22cm on a typical torso, out past the cheek. The old value was loose ON PURPOSE, on the assumption that hand separation did the real discriminating and this only filtered out the grossly wrong; that assumption is now dead (see the hand-separation Key decision in CLAUDE.md — SEP fires on a resting body, as does ARM), so this gate has to carry weight it was never sized for. Tightening here fails toward a MISLABELLED row, not a lost arrow: anchorOk feeds reachedFullDraw, never whether an attempt is logged, so a draw that misses this still appears in the log marked short of full draw

// ANCHOR is a DISTANCE only — a circle around the mouth — which accepts anatomically impossible
// positions along with real ones (in front of the mouth, above the nose). Owner, 2026-08-24: "i
// cannot anchor ahead of my mouth. it'll always be below or slightly backwards." These two add
// DIRECTION on top of FULL_DRAW_ANCHOR_MAX's existing distance gate — they never replace it, and
// FULL_DRAW_ANCHOR_MAX itself is untouched (it's the one constant in this file measured off a real
// reading; changing it needs a new reading, not a desk decision).
//
// MEASURED 2026-08-24, same session and same ?triggertest ANCHOR screen as FULL_DRAW_ANCHOR_MAX's
// own reading, both replacing the unmeasured placeholders they launched with (0.25 / -0.1). As with
// every other reading in this file, this is a SINGLE SAMPLE, not a spread across several shots —
// treat it as a real anchor, not as proof of how much it varies shot to shot. Loosening either
// value needs a session showing REAL anchors getting rejected (several shots, not one), the same
// bar every other calibration constant here is held to; a single field report of a false reject
// would justify investigating, not retuning on the spot.
const FULL_DRAW_ANCHOR_ABOVE_MAX = 0.05; // fraction of torso length the draw wrist may sit ABOVE the anchor point (mouth-midpoint, or nose if the mouth isn't visible) and still pass — image y grows downward, so this bounds how far UP is still plausible, which is the "above the nose" failure mode reported. Was 0.25 (unmeasured; loose enough to permit the hand up around eyebrow height). Owner's own reading: his draw hand sits 0.08 torso-lengths BELOW the mouth at real anchor (i.e. anchorVerticalOffset ≈ +0.08, the opposite side of this limit entirely), so 0.05 of allowed room ABOVE the mouth leaves him roughly 0.13 torso-lengths (~6cm on his own build) of margin before this would ever reject a real anchor
const FULL_DRAW_ANCHOR_BACKWARD_MIN = 0; // the draw wrist's position along the anchor-point→draw-ear axis, in torso-lengths (see isAtFullDraw's own comment for why the draw-side ear, not the shoulder line, defines "backwards" here) — positive means toward the ear (backwards, as the owner described), negative means toward the front of the face. Was -0.1 (unmeasured; permitted the hand a tenth of a torso-length IN FRONT of the mouth — exactly the case the owner reported as wrong). Owner's own words settle this at exactly 0, not some small positive tolerance: "i cannot anchor ahead of my mouth. it'll always be below or slightly backwards" — a compound anchor is never in front, so anything at or behind the mouth (>= 0) is anatomically the only possibility, and anything negative is definitionally wrong. His own reading was 0.17, comfortably clear of this line
const FULL_DRAW_BOW_ARM_MIN = 140; // degrees; bow arm must be at least this straight to count as "drawn". Was 150 — loosened 2026-08-24 after the owner reported "sometimes i shoot with my arm not 100% straight and the slight bend doesn't pass the trigger." Not a measurement, a deliberate response to a false-reject field report: the new ARM CONE check below (FULL_DRAW_ARM_CONE_APERTURE_DEG) now shares the job of ruling out "arm isn't really extended toward the shot" that straightness alone used to carry by itself, so straightness itself can afford to be laxer
// ARM CONE — owner, 2026-08-24, replacing an earlier "reuse the raise-height signal" instruction
// with his own better one: "create a cone in front of my shoulder joint and check for the arm to
// be placed there and extended rather than only extended." An arm can be perfectly straight
// (passing FULL_DRAW_BOW_ARM_MIN) while hanging at the archer's side — armOk never checked WHERE
// the straight arm pointed. This constant is the cone's aperture; see bowArmElevationOf's own
// comment for why the cone's axis is horizontal ELEVATION rather than an attempted 3D "in front of
// the body" direction. UNMEASURED PLACEHOLDER, defaulted permissively (wide) — 45° comfortably
// passes ordinary shooting form (a drawn bow arm sits close to horizontal) while still failing an
// arm hanging at the side (~90° off horizontal) outright, which is the one case this exists to
// catch. Read the live elevation angle off the ?triggertest ARM screen to tighten this once real
// numbers exist, same loop as every other constant in this block.
const FULL_DRAW_ARM_CONE_APERTURE_DEG = 45;
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
//
// KNOWN RESIDUAL CASE, left as-is on purpose: hand separation is measured assuming a side-on
// framing, where the two arms genuinely overlap in the camera's view at rest. Someone standing
// face-on or back-on to the camera (see the RAISE TRIGGER's own bow-arm-height signal below,
// which doesn't have this problem) can read as "hands apart" from ordinary standing posture alone
// — proven with a static, motionless test pose, see the diagnosis this shipped with. A long
// enough pause in that orientation can still clear these two gates and open/log a row via hand
// separation alone (the raise trigger's fallback path, kept deliberately — see its own comment:
// a missed raise must still be able to log a real shot). Do NOT read that as license to tighten
// SHOT_MIN_PEAK_SEP_FRACTION/DRAW_ATTEMPT_MIN_SEP against it — with the reachedFullDraw distinction
// below (see fullDrawShotCount, endAttempt's signalOutcome call, and renderShotLog/buildShareText),
// this case can no longer fake a green "arrow counted" flash or corrupt the consistency numbers:
// anchorOk fails for it every time (nobody's hand is near their face while just standing there),
// so it logs a marked, excluded, "seen but not confirmed" row — a correct, acceptable outcome for
// a resting posture, not a bug to chase by making these two constants stricter.
const SHOT_MIN_PEAK_SEP_FRACTION = 0.8; // the attempt's peak hand separation must reach at least this fraction of FULL_DRAW_HAND_SEP_MIN (0.8 x 0.75 = 0.6 torso-lengths apart) to count as a real draw attempt — comfortably above the ~0.3-0.5 range nocking/lowering the bow produces, comfortably below the 0.75 that counts as full draw itself
const SHOT_MIN_DURATION_MS = 600; // an attempt must last at least this long, from when hands first cross DRAW_ATTEMPT_MIN_SEP to when they drop back below it, to count as a real draw rather than a brief noise spike (a hand passing near the body, a tracking glitch) — a real compound draw, even a rushed one, takes real time to raise, draw and settle

// RAISE TRIGGER — owner's own proposal, field-tested reasoning: "every time i raise my left arm
// at shoulder height trigger a record because the shooting routine has started." Raising the bow
// arm is the first deliberate movement of every shot, happens once, and isn't affected by
// occlusion the way the draw hand's anchor position can be. This is an EARLIER, ADDITIONAL way to
// start watching a draw attempt — see trackShotAttempt — never a replacement for the existing
// hand-separation trigger (DRAW_ATTEMPT_MIN_SEP above), which still opens an attempt on its own if
// a raise is ever missed (occlusion, tracking blip): fail toward recording, per CLAUDE.md.
// Deliberately does NOT change how a shot's numbers are measured — see bowArmRaiseHeight and
// trackShotAttempt's own comments for how the raise is kept out of the median.
const RAISE_TRIGGER_UP_FRACTION = 0; // bow wrist at or above bow-shoulder height — (shoulder.y - wrist.y)/torsoLength >= this — arms the "watching" state. 0 = exactly level with the shoulder; the first instant the bow arm reaches shoulder height on the way up
const RAISE_TRIGGER_DOWN_FRACTION = -0.3; // the bow wrist must drop back down at least this far below shoulder height before a NEW raise can trigger again — well below RAISE_TRIGGER_UP_FRACTION (hysteresis), so ordinary landmark noise right at shoulder height can't chatter the trigger on and off every frame the arm happens to hover near it
// Safety valve, not a coach knob (same family as CLIP_MAX_MS) — a raise that never turns into a
// real draw (he lowers the bow, changes his mind, adjusts something with the arm still up) closes
// on its own after this long, rejected like any other attempt that didn't earn a row, rather than
// leaving the app "watching" for the rest of the session.
const RAISE_ATTEMPT_TIMEOUT_MS = 12000;

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

// ===== CALIBRATION — owner's body-proportion reference (HANDOVER.md Stage 4). NOT a form target
// like CALIBRATE WITH COACH above — nothing here judges his archery. This is a reference for what
// HIS body's own proportions actually are, captured passively while he's standing calmly in frame
// before he ever picks up the bow, so the app has something honest to check a measurement against.
// It once reported his wrists 2.3 torso-lengths apart — physically impossible — and had no way to
// know that was nonsense; this is what lets it know.
//
// Runs with NO button and no explicit trigger, per the owner's own decision — it should cost him
// nothing to remember. It reuses the ATTENTION GATING calm detector above (attentionEngaged /
// attentionIsClearlyCalm) as its "is he standing there, readable" signal rather than building a
// second one: see sampleForCalibration further down for exactly which flags it reads.
const CALIBRATION_MIN_SAMPLES = 15; // how many good (all-needed-landmarks-visible) frames one calm stretch needs before a calibration is trusted — see medianCalibrationOf. At a typical idle sampling rate this is a couple of seconds of ordinary standing-still, nothing he'd notice
const MIN_SHOULDER_TO_TORSO = 0.2; // a shoulder-width reading below this fraction of torso length is treated as "the camera's basically side-on right now, this number is unreliable" rather than a genuine narrow-shouldered reading — see CLAUDE.md's own note that the shoulder line goes nearly degenerate in a side-on view. Below this floor, calibration produces NO reading for that frame rather than a garbage-small ratio nobody could ever match again
const CALIBRATION_PLAUSIBILITY_SLACK = 1.15; // extra margin (15%) on top of the exact physical reach bound in handSepIsPlausible below, to absorb ordinary calibration/measurement noise without crying wolf — not a fudge on the physics, just room for the fact that neither calibration nor a shot's own reading is perfectly noise-free (though both are already medians over many frames, not single noisy ones)
const CALIBRATION_AGREEMENT_TOLERANCE = 0.15; // relative difference (15%) between a stored ratio and a freshly re-measured one before it counts as a real disagreement worth telling him about, rather than ordinary measurement noise. A desk estimate, like SHOULDER_DROP_MIN_PCT above — not yet tuned against a real two-session comparison
const CALIBRATION_STORAGE_KEY = "archery-calibration-v1"; // localStorage key — deliberately its own small, independent use of storage (see HANDOVER.md), not shared with whatever the shot-log persistence work elsewhere uses

// FRAMING SIGNATURE — an OPTIONAL add-on to the calibration above, for a session-tracking feature
// the owner is considering but hasn't committed to building. It answers a different question from
// the body-proportion ratios above: not "is this reading physically possible for him", but "was
// he set up the same way relative to the camera as last time" — stand a metre further back, or
// tilt the phone, and every measurement shifts for reasons that have nothing to do with his form.
// Nothing anywhere in this app depends on this existing: if it can't be computed (or there's no
// stored one to compare against yet), calibration/detection/logging/clips carry on exactly as if
// this feature didn't exist — see framingChangeMessage's own null-safety.
const FRAMING_SIZE_TOLERANCE = 0.2; // relative — apparent size (torso length as a fraction of frame height) allowed to differ by this much, session to session, before it's worth a word. A desk estimate, like CALIBRATION_AGREEMENT_TOLERANCE above — loose enough that standing a step closer or further doesn't cry wolf
const FRAMING_POSITION_TOLERANCE = 0.12; // absolute — fraction of the frame's own width/height the body's reference point (hip midpoint) may drift before "you're standing in a different part of the frame" is worth saying. Ordinary "didn't plant his feet in the exact same spot" variation is well under this
const FRAMING_SQUARENESS_TOLERANCE = 0.15; // absolute, in torso-lengths — how much the shoulder/hip horizontal-separation proxy (see framingSignatureOf) may differ before "how square to the camera he is" counts as having actually changed, not just ordinary stance noise
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

// ?triggertest in the URL shows a live, single-trigger inspector instead of the multi-trigger
// ?debug grid — see the TRIGGER TEST block further down (near buildTriggerTestPanel) for the
// panel itself. Kept as its own flag, independent of DEBUG, so ?triggertest works completely on
// its own (no need to also pass ?debug) and ?debug's own panel/layout stay exactly as they were —
// see that block's own comment for the couple of `if (DEBUG)` display-only capture points this
// flag also has to fire, since this panel needs the exact same real computed numbers (debugInfo,
// debugRaiseHeight, lastPoseSeen, lastCropBoxStable) those already exist to capture, just shown
// one trigger at a time instead of all at once. Read-only in every case: none of those capture
// points ever feed back into detection, gating, measurement or logging — only into what a
// diagnostic panel displays.
const TRIGGERTEST = location.search.includes("triggertest");

// ===== DEBUG OVERLAY REFRESH — performance knob, not calibration; safe to change without a
// coach, same family as SMOOTHING/POSE MODEL/ROI CROPPING above (kept here rather than inside
// CALIBRATE WITH COACH, which is reserved for archery-form thresholds — see README's "The numbers
// are placeholders"). syncDebugOverlay used to rebuild this panel's DOM (innerHTML) on every
// single rendered frame, 30-60 times a second — no one can read a number changing that fast, so
// that cost bought nothing, and measured (see PM's brief) it was landing on the exact main thread
// the video now depends on to keep presenting smoothly on its own. This constant just puts a
// floor under how often the panel's DOM actually gets rewritten; debugInfo itself is still
// recomputed fresh every frame underneath, so what full draw needs to trigger, what gets logged
// and what a clip records are all untouched — only how often the ?debug TEXT gets repainted.
const DEBUG_OVERLAY_REFRESH_MS = 150; // minimum time between ?debug panel rewrites — about 6-7 times a second, plenty to read. RAISE to spend even less on this diagnostic-only panel (choppier to watch, never affects detection/logging); LOWER only if it ever needs to feel snappier than this, never down near a single frame's own interval

// A momentary event (an attempt opening, the raise trigger firing, a shot getting logged, a
// single frame being eligible to log from) can be true for exactly one frame at 30-60fps. With
// the panel above already throttled to DEBUG_OVERLAY_REFRESH_MS, anything shorter than that
// refresh isn't just hard to see — the render can land on a tick that never sampled the frame it
// happened on at all, so the annunciator for it would never light. Each such event instead
// records a timestamp (see debugEvents below) the instant it happens, and its lamp stays lit for
// this long afterwards — long enough that the panel's own throttled refresh is guaranteed to
// catch it at least once. Display-only: nothing anywhere reads these timestamps except
// syncDebugOverlay, so this can never feed back into detection, gating, measurement or logging.
const DEBUG_EVENT_LATCH_MS = 400;

// MediaPipe pose landmark indices (33-point model)
const L_SHOULDER = 11, R_SHOULDER = 12;
const L_ELBOW = 13, R_ELBOW = 14;
const L_WRIST = 15, R_WRIST = 16;
const L_HIP = 23, R_HIP = 24;
const NOSE = 0, MOUTH_L = 9, MOUTH_R = 10;
const L_EAR = 7, R_EAR = 8;
const L_ANKLE = 27, R_ANKLE = 28; // only used by the calibration framing check (bothAnklesVisible) — "are you fully in frame, or cut off at the legs"

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const stageEl = document.getElementById("stage");
const cameraBoxEl = document.getElementById("camerabox");
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
// Element references into the panel buildDebugPanel builds below, keyed by each node's own
// data-x attribute. Declared up here (not down with the rest of the ?debug runtime state,
// further below) specifically so buildDebugPanel — called a few lines down, before any of that
// later state exists yet — can assign to it without a temporal-dead-zone error.
let dbgRefs = null;
if (DEBUG) {
  debugEl.classList.remove("hidden");
  document.body.classList.add("debug-mode"); // see the ?debug LAYOUT block in style.css — the only thing that turns the two-column grid on
  buildDebugPanel(); // built once, here, before any frame renders — see its own comment for why
}

// ?triggertest — live single-trigger inspector, independent of ?debug (see the TRIGGERTEST
// constant's own comment and the TRIGGER TEST block further down, near buildTriggerTestPanel).
const triggerTestEl = document.getElementById("triggertest");
let ttRefs = null;
if (TRIGGERTEST) {
  triggerTestEl.classList.remove("hidden");
  document.body.classList.add("triggertest-mode"); // see the ?triggertest LAYOUT block in style.css
  // buildTriggerTestPanel() itself is NOT called here, unlike buildDebugPanel() above — it reads
  // TRIGGER_DEFS, a `const` defined much further down in this file (in the TRIGGER TEST block),
  // and calling it this early would hit that `const`'s temporal dead zone (module top-level
  // execution hasn't reached its initialiser yet). buildTriggerTestPanel() is a function
  // DECLARATION so it hoists fine and can be defined down there — it just can't be CALLED until
  // TRIGGER_DEFS itself has actually run, so the call site lives right after that block instead,
  // gated on this same TRIGGERTEST flag. See TRIGGER_DEFS's own comment.
}
// PROVISIONAL — Stage 4 calibration/framing status line (see HANDOVER.md and the CALIBRATION
// block further down). Belongs on Stage 3's future Setup screen; lives here, minimally, until
// that's built. Independent of the ?debug panel above (which gets its own calibration row — see
// buildDebugPanel/syncDebugOverlay's s-calibration entries) — this element is the normal-UI line,
// shown in every build, not just ?debug.
const calibrationStatusEl = document.getElementById("calibration-status");
const btnLog = document.getElementById("btn-log");
const shotLogEl = document.getElementById("shotlog");
// renderShotLog() writes into this inner scrolling element, never into shotLogEl itself — the
// close button lives outside it in the DOM (see index.html) specifically so a long log can never
// scroll it out of reach. See HANDOVER.md Stage 1a.
const shotLogContentEl = document.getElementById("shotlog-content");
const shotLogCloseBtn = document.getElementById("shotlog-close");
const shotLogShareBtn = document.getElementById("shotlog-share");
const shotLogShareTextEl = document.getElementById("shotlog-sharetext");
const clipPlayerEl = document.getElementById("clipplayer");
const clipPlayerVideo = document.getElementById("clipplayer-video");
const clipPlayerClose = document.getElementById("clipplayer-close");
const clipPlayerRateBtns = document.querySelectorAll(".clipplayer-rate");
const cueEl = document.getElementById("cue"); // see the SHOOTING CUES block further down, and its own comment in style.css

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

// Builds a fresh "nothing computed yet/this bail" debugInfo object — the ?debug panel's
// NEVER-BLANK guarantee lives here. `reason` is a plain-language sentence naming why isAtFullDraw
// couldn't get all the way through this frame (or null on a frame that succeeded); every other
// field defaults to null (shown as "—" in the panel) rather than being left undefined, so the
// panel's rendering code never has to guess which fields exist on a given call. debugInfo is
// ALWAYS one of these objects, never bare null — see isAtFullDraw and renderLoop's pose-lost
// branch, both of which only ever call this rather than assigning null directly.
function emptyDebugInfo(reason) {
  return {
    reason,
    handSep: null, sepOk: null,
    anchorDist: null, anchorOk: null,
    anchorVerticalOffset: null, anchorVerticalOk: null,
    anchorBackward: null, anchorBackwardOk: null,
    bowArmAngle: null, armStraightOk: null,
    armElevation: null, armConeOk: null, armOk: null,
    speed: null, stillOk: null,
    // Pixel-space points/scale, filled in only on a frame that got all the way through — used
    // exclusively by drawTriggerTestOverlay to draw the acceptance regions on top of the camera
    // picture without recomputing any geometry a second time (see that function's own comment).
    // Never read by any detection/gating/measurement code, same as the rest of this object.
    anchorPx: null, drawWristPx: null, bowWristPx: null, bowShoulderPx: null,
    anchorEarPx: null, scale: null,
  };
}

// Last frame's full-draw condition values, for the ?debug panel only (see syncDebugOverlay).
// Always a real object (see emptyDebugInfo) — never bare null, even on a frame isAtFullDraw
// bails out of early, so the panel always has a `reason` to show instead of going blank.
let debugInfo = emptyDebugInfo("not started yet");

// Names which of a set of landmarks aren't confidently visible, in plain language, for the
// ?debug panel's never-blank bail-reason text (see isAtFullDraw). `checks` is a list of
// [landmarkIndex, plainName] pairs. Pure — no module state — so selfTest can call it directly.
function describeMissingLandmarks(landmarks, checks) {
  return checks.filter(([idx]) => !visible(landmarks, idx)).map(([, name]) => name);
}

// Last time (performance.now()) the ?debug panel's DOM was actually rewritten — see
// DEBUG_OVERLAY_REFRESH_MS and syncDebugOverlay. -Infinity so the very first frame always paints.
let lastDebugRenderMs = -Infinity;

// ===== ?debug PANEL — runtime state, all of it display-only. Nothing in this block is ever read
// by any detection, gating, measurement or logging code — only by buildDebugPanel/syncDebugOverlay
// further down. Kept together so it's obvious at a glance which module-level variables exist
// purely to feed the diagnostic panel and can never influence what the app actually does.
//
// debugEvents: performance.now() timestamps of the most recent occurrence of each MOMENTARY event
// (see DEBUG_EVENT_LATCH_MS above for why these need a timestamp rather than a plain boolean) — 0
// means "never happened this session". isDebugEventLit (below) is the only thing that reads these.
let debugEvents = { attemptStarted: 0, raiseFired: 0, frameEligible: 0, shotLogged: 0 };
// Live value of the raise trigger's own signal (see bowArmRaiseHeight/updateRaiseTrigger) —
// captured every frame that can read it at all, purely so the TRIGGERS/STATE section can show the
// actual number next to RAISE_TRIGGER_UP_FRACTION, not just the armed/unarmed boolean.
let debugRaiseHeight = null;
// ?triggertest-only display state for the ATTN screen (see TRIGGER TEST further down). Nothing
// in updateAttentionState/attentionIsClearlyCalm was changed to add this — both are already pure,
// side-effect-free functions (see attentionIsClearlyCalm's own comment), so renderLoop just calls
// them a second time, purely for display, with a snapshot of the previous-frame state taken
// before updateAttentionState overwrites it. Never read by anything except the trigger-test panel.
let debugAttnCalm = null; // the calm/not-calm verdict attentionIsClearlyCalm reached this frame
let debugAttnHandSep = null; // hand separation as ATTN sees it (handSeparationForAttention) — a different formula from the SEP gate's own handSep, see ATTENTION_REST_HAND_SEP_MAX's own comment
let debugAttnSpeed = null; // body-reference-point drift speed, torso-lengths/second, as ATTN sees it
// Whether the crop box used THIS frame was stable (see cropBoxIsStable) — captured in renderLoop
// right where that's already computed for frameEligible, so the panel can show it as its own
// continuous lamp without recomputing crop-box geometry a second time.
let lastCropBoxStable = false;
// Whether a pose was actually found the last time detection genuinely ran (full-rate frame, or an
// idle-rate sample) — NOT updated on an idle-throttle tick that skipped detection entirely, so the
// panel correctly keeps showing the last real answer through an idle gap instead of flickering to
// "not seen" for a reason that has nothing to do with whether the archer is there.
let lastPoseSeen = false;
// How long the most recent detectForVideo call took, and an instantaneous rendered-frame-rate
// figure recomputed every renderLoop call — a live companion to the one-time startup measurement
// already reported in modelStatusLine (see POSE MODEL above), for the SESSION section's tracking-
// health line. debugLastFrameTs is the bookkeeping timestamp debugInstantFps is derived from.
let lastInferenceMs = null;
let debugLastFrameTs = null;
let debugInstantFps = null;
// (dbgRefs itself is declared up near debugEl, before buildDebugPanel's own call site — see that
// declaration's comment for why it has to live there instead of here with the rest of this state.)

// Is a momentary event's lamp still lit? True from the instant its timestamp is recorded through
// DEBUG_EVENT_LATCH_MS afterwards; 0 (never happened) is always unlit. Pure — no module state —
// so selfTest can drive it directly with fixture timestamps, including proving it goes dark once
// the window passes.
function isDebugEventLit(eventTs, nowMs) {
  return eventTs !== 0 && nowMs - eventTs <= DEBUG_EVENT_LATCH_MS;
}
// ===========================================================================

// Shot log: a persistent record the owner can check after they've finished shooting, because
// they cannot read the screen or tap anything while actually on the line (see CLAUDE.md). One
// row per draw attempt, whether or not it ever reached full draw — kept until the page reloads.
// Each measure on that row is the MEDIAN of that measure across every eligible frame of the
// hold, not a single "best" frame (see medianSampleOf's own block comment for why — a real,
// measured bias, not just a simplification). No timer anywhere in this: entries never expire or
// get overwritten just because time passed, only because a newer attempt bumps an old one out of
// the last SHOT_LOG_MAX.
const SHOT_LOG_MAX = 10;
let shotCount = 0; // total LOGGED attempts this session (see endAttempt) — keeps counting even once the log above fills up. Includes rows that never reached full draw (see fullDrawShotCount below) — this is "how many rows exist", not "how many arrows", and drives shotNum numbering, so it must never skip a value just because a row turned out short of full draw
let fullDrawShotCount = 0; // of the LOGGED rows above, how many actually reached full draw (anchorOk/armOk/sepOk/stillOk all true on at least one eligible frame — see reachedFullDraw). THIS is "arrows" for the headline count and the green cue (see logShot/endAttempt) — a row can be logged, numbered and kept without ever incrementing this. Same "keeps counting past SHOT_LOG_MAX" treatment as shotCount, for the same reason: the headline arrow count must stay true to the whole session, not just whatever's still in the capped log
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
// recorded into a shot's clip (see paintCanvas/drawVideoFrame further down — those always draw
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

// ===== CALIBRATION — runtime state. Fed passively by sampleForCalibration (called from
// renderLoop whenever landmarks come back), which reuses attentionEngaged above as its "is he
// standing there, readable" signal rather than building a second detector — see that function's
// own comment. One verdict per session, same "set once, then quiet" convention as
// modelStatusLine/clipsUnavailableReason elsewhere in this file.
let calibrationSamples = []; // proportions collected during the CURRENT unbroken calm stretch — cleared the instant he moves (attentionEngaged goes true), so a calibration is never built from samples spanning "calm, then walked around, then calm again"
let calibrationCalmSinceMs = null; // wall-clock start of the current calm stretch — mirrors attentionCalmSinceMs's own bookkeeping, used only to give the framing check below a real held stretch to judge rather than one lucky frame
let calibrationDone = false; // this session's one calibration verdict has already been reached — stop sampling/comparing for the rest of the session
let calibrationStatusLine = null; // plain-English disagreement note — set ONLY when today's fresh reading disagrees with what's stored; stays null (nothing shown, nothing said) on agreement, per the owner's explicit "silent unless it disagrees" decision
let framingStatusLine = null; // plain-English framing note ("legs may be outside the frame") — set only if the ankles stay missing across a real held calm stretch; stays null when he's fully in frame
// The calibration this SESSION is actually using for the shot-log nonsense check below
// (handSepIsPlausible). Starts from whatever was REMEMBERED (loadStoredCalibration) rather than
// null — "remember it" (the owner's own words) means the app should keep working from last
// session's numbers from the very first frame, not go dark for the whole session waiting on a
// fresh re-check that (see the PM's review) may never actually complete if he never happens to
// face the camera. Replaced with the fresh reading the moment one lands (see
// finishPassiveCalibration) — "recheck it every time" — whether that reading agrees or not.
let activeCalibration = loadStoredCalibration();
let framingSamples = []; // FRAMING SIGNATURE (optional add-on, see the constants block above) — collected alongside calibrationSamples during the same calm stretch, cleared/reset the same way
let framingChangeStatusLine = null; // plain-English note that THIS SESSION's framing looks different from the stored one — set only on a real difference; stays null (nothing shown) when it matches, or when there's nothing to compare against yet

// Two wrists' separation, scaled by torso length — the same "how far apart are the hands" signal
// isAtFullDraw uses internally (see FULL_DRAW_HAND_SEP_MIN/DRAW_ATTEMPT_MIN_SEP above), but kept
// as its own small function here rather than reaching into isAtFullDraw's own computation, which
// is being reworked by a different engineer in parallel right now (the geometry-maths fix) — this
// stays deliberately independent so the two changes land on different lines and merge cleanly.
// Same "never guess" convention as the rest of the file: null if either wrist isn't confidently
// visible, or there's no usable torso-length scale reference. frameWidth/frameHeight: same
// physically-honest pixel-space conversion (see toPixelSpace) every other distance in this file
// uses — required here for exactly the same reason, not optional polish.
function handSeparationForAttention(landmarks, frameWidth, frameHeight) {
  const bowWrist = rightHanded ? L_WRIST : R_WRIST;
  const drawWrist = rightHanded ? R_WRIST : L_WRIST;
  if (!visible(landmarks, bowWrist) || !visible(landmarks, drawWrist)) return null;
  const scale = attentionScale(landmarks, frameWidth, frameHeight);
  if (!scale) return null;
  const a = toPixelSpace(landmarks[bowWrist], frameWidth, frameHeight);
  const b = toPixelSpace(landmarks[drawWrist], frameWidth, frameHeight);
  return Math.hypot(a.x - b.x, a.y - b.y) / scale;
}

// Shared torso-length scale reference for this block — draw side preferred, bow side as
// fallback, the same convention used everywhere else in this file (isAtFullDraw, shoulderDropOf).
function attentionScale(landmarks, frameWidth, frameHeight) {
  const drawShoulder = rightHanded ? R_SHOULDER : L_SHOULDER;
  const drawHip = rightHanded ? R_HIP : L_HIP;
  const bowShoulder = rightHanded ? L_SHOULDER : R_SHOULDER;
  const bowHip = rightHanded ? L_HIP : R_HIP;
  return (
    torsoLength(landmarks, drawShoulder, drawHip, frameWidth, frameHeight) ??
    torsoLength(landmarks, bowShoulder, bowHip, frameWidth, frameHeight)
  );
}

// The midpoint between the two hips — a stable whole-body reference point that isn't a hand and
// doesn't itself move as part of a normal draw, used to tell "standing settled" apart from
// "walking/stepping" (see ATTENTION_REST_MOVE_MAX_PER_SEC). Null if either hip isn't confidently
// visible — same "never guess" convention as the rest of the file. Returns NORMALISED coordinates
// (same convention as lastDrawWrist in isAtFullDraw) — converted to pixel space at compare time,
// using that frame's own dimensions, by whoever compares two of these against each other.
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
function attentionIsClearlyCalm(landmarks, prevRef, dtSec, frameWidth, frameHeight) {
  if (!landmarks) return true;
  const handSep = handSeparationForAttention(landmarks, frameWidth, frameHeight);
  if (handSep === null || handSep > ATTENTION_REST_HAND_SEP_MAX) return false;
  const ref = bodyReferencePoint(landmarks);
  if (ref && prevRef && dtSec > 0) {
    const scale = attentionScale(landmarks, frameWidth, frameHeight);
    if (!scale) return false;
    // ref/prevRef are still normalised (see bodyReferencePoint) — converted to pixel space here,
    // at compare time, using THIS frame's dimensions for both points, same convention isAtFullDraw
    // uses for lastDrawWrist.
    const refPx = toPixelSpace(ref, frameWidth, frameHeight);
    const prevRefPx = toPixelSpace(prevRef, frameWidth, frameHeight);
    const speed = Math.hypot(refPx.x - prevRefPx.x, refPx.y - prevRefPx.y) / scale / dtSec;
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
// frameWidth/frameHeight are required (no default, same convention as isAtFullDraw) — every
// distance this function's own calm check touches needs the physically-honest pixel-space
// conversion (see toPixelSpace) every other measure in this file already gets.
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
function updateAttentionState(nowMs, landmarks, frameWidth, frameHeight, gatingEnabled = ATTENTION_GATING_ENABLED, modelReady = modelDecisionMade) {
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
  const calm = attentionIsClearlyCalm(landmarks, attentionPrevRef, dtSec, frameWidth, frameHeight);
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
      saveSessionToStorage(); // this counter has its own line in the log/share text — see SESSION PERSISTENCE. Fires once per idle period, not per frame.
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
    const wokeHandSep = landmarks ? handSeparationForAttention(landmarks, frameWidth, frameHeight) : null;
    if (wokeHandSep !== null && wokeHandSep >= DRAW_ATTEMPT_MIN_SEP) {
      attentionLateWakeCount++;
      saveSessionToStorage(); // same reasoning as attentionIdlePeriods above — fires once per late wake, not per frame
    }
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

  // Nothing to do for mirroring here — see effectiveMirror/withMirror/syncMirrorClasses for how
  // it actually gets applied (a CSS class kept in sync every frame from paintCanvas), not
  // anything startCamera itself needs to set up.
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
  layoutCameraBox(); // the on-screen box's shape depends on these same dimensions — see its own comment
}
video.addEventListener("loadedmetadata", sizeCanvasToVideo); // module-scope, attached once — covers first startup, every later camera switch, and late-arriving dimensions alike

// Sizes #camerabox (in index.html/style.css) to the largest box of the camera's own aspect ratio
// that fits inside #stage, centred — the letterbox fix. Without this, #camerabox (and therefore
// #video/#overlay, which always fill it at 100%x100%) was stretched to fill #stage's own box
// (100vw x 100dvh, i.e. whatever shape the screen happens to be), and a <canvas> has no
// `object-fit` of its own to correct that the way <video> does — so on any screen shape that
// doesn't already match the camera's native aspect ratio, the picture visibly stretched. That
// mismatch is small with the browser address bar taking up part of the screen, and much more
// visible with it gone (launched from the Home Screen, `display-mode: standalone`) — see
// HANDOVER.md. This ONLY changes the on-screen CSS box: canvas.width/height (the actual pixel
// buffer every measurement and every recorded clip uses — see toPixelSpace and
// startClipRecording's canvas.captureStream) are untouched, set purely from
// video.videoWidth/videoHeight exactly as before. #video and #overlay both stay at 100%x100% of
// this box (see style.css), so they are always scaled by the exact same factor and stay
// pixel-registered with each other — there is no separate letterboxing calculation for each.
function layoutCameraBox() {
  if (!(video.videoWidth > 0 && video.videoHeight > 0)) return;
  const stageWidth = stageEl.clientWidth;
  const stageHeight = stageEl.clientHeight;
  if (!stageWidth || !stageHeight) {
    // #stage hasn't been laid out yet — seen right after a fresh page load, before the browser's
    // first layout/paint pass, which is exactly when startCamera() calls this directly (found by
    // testing, not assumed: a headless run hit this race on its very first call). Nothing sane to
    // compute yet, but this must not just give up — the only other callers are event listeners
    // (loadedmetadata/resize/orientationchange) that aren't guaranteed to fire again on their own,
    // so a box left unsized here could stay unsized (and therefore still stretched, the exact bug
    // this function exists to fix) for the rest of the session. Try again next frame instead.
    requestAnimationFrame(layoutCameraBox);
    return;
  }
  const cameraAspect = video.videoWidth / video.videoHeight;
  const stageAspect = stageWidth / stageHeight;
  let boxWidth, boxHeight;
  if (cameraAspect > stageAspect) {
    // Camera is relatively wider than the screen: full width, letterbox bars top/bottom.
    boxWidth = stageWidth;
    boxHeight = boxWidth / cameraAspect;
  } else {
    // Camera is relatively taller (or equal): full height, letterbox bars left/right.
    boxHeight = stageHeight;
    boxWidth = boxHeight * cameraAspect;
  }
  cameraBoxEl.style.width = `${boxWidth}px`;
  cameraBoxEl.style.height = `${boxHeight}px`;
}
// Re-run whenever the SCREEN's shape can change, not just the camera's — the camera's own
// dimensions don't change without a loadedmetadata event (already handled by sizeCanvasToVideo
// above), but the viewport does: rotating the phone, and — the specific case from HANDOVER.md —
// iOS Safari's address bar showing or hiding, which changes the available height without any
// camera event firing at all. `resize`/`orientationchange` alone can miss that address-bar case on
// some iOS versions, so `visualViewport`'s own resize event (where available) is also wired up as
// a second path to the same call — cheap and idempotent either way if both happen to fire.
window.addEventListener("resize", layoutCameraBox);
window.addEventListener("orientationchange", layoutCameraBox);
if (window.visualViewport) {
  window.visualViewport.addEventListener("resize", layoutCameraBox);
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

// ===== RAISE TRIGGER — see the RAISE_TRIGGER_UP_FRACTION/RAISE_TRIGGER_DOWN_FRACTION constants
// above for the full reasoning. How far the BOW wrist sits above (positive) or below (negative)
// the BOW shoulder, in torso-lengths — the one signal the raise trigger watches. Own-side torso
// scale only (no draw-side fallback, unlike isAtFullDraw's hand-separation scale): this measures
// the bow arm specifically, so the bow arm's own shoulder/hip is the physically correct reference,
// not a borrowed one from the other side of the body. Image y grows downward, so a wrist ABOVE the
// shoulder has a SMALLER y — hence shoulder.y - wrist.y, not the other way round. Same "never
// guess" convention as every other measure here: null if a needed landmark isn't confidently
// visible or there's no usable torso-length scale. Follows the handedness toggle via rightHanded,
// exactly like bowArmAngleOf/shoulderDropSampleOf — "bow arm", never "left arm".
function bowArmRaiseHeight(landmarks, frameWidth, frameHeight) {
  const bowShoulder = rightHanded ? L_SHOULDER : R_SHOULDER;
  const bowWrist = rightHanded ? L_WRIST : R_WRIST;
  const bowHip = rightHanded ? L_HIP : R_HIP;
  if (!visible(landmarks, bowShoulder) || !visible(landmarks, bowWrist)) return null;
  const scale = torsoLength(landmarks, bowShoulder, bowHip, frameWidth, frameHeight);
  if (!scale) return null;
  const shoulder = toPixelSpace(landmarks[bowShoulder], frameWidth, frameHeight);
  const wrist = toPixelSpace(landmarks[bowWrist], frameWidth, frameHeight);
  return (shoulder.y - wrist.y) / scale;
}

// true from the moment the bow wrist crosses RAISE_TRIGGER_UP_FRACTION going up until it drops
// back through RAISE_TRIGGER_DOWN_FRACTION — latched with hysteresis (see those constants' own
// comments) so noise right at shoulder height can't fire this every frame. Read by
// trackShotAttempt below to decide whether a draw attempt should be open; never touched by the
// four full-draw conditions or the median measures — the raise only decides WHEN to watch, never
// what gets measured.
let raiseArmed = false;

// Updates raiseArmed for THIS frame from the bow wrist's current height. An unreadable frame
// (bow shoulder/wrist not confidently visible, or no torso scale) can't fire OR clear the trigger
// — never guess, same convention as every other signal in this file — so raiseArmed simply holds
// whatever it was. Pure state transition only; trackShotAttempt below decides what a raise
// actually DOES (open an attempt, start a clip recording).
// nowMs defaults to a real clock read so every existing call site (several in selfTest) keeps
// working unchanged — it only matters for the ?debug panel's momentary RAISE FIRED lamp (see
// DEBUG_EVENT_LATCH_MS above), which isAtFullDraw's own call passes its real nowMs into instead.
function updateRaiseTrigger(landmarks, frameWidth, frameHeight, nowMs = performance.now()) {
  const height = bowArmRaiseHeight(landmarks, frameWidth, frameHeight);
  if (DEBUG || TRIGGERTEST) debugRaiseHeight = height; // display-only — see debugRaiseHeight's own comment
  if (height === null) return;
  if (!raiseArmed && height >= RAISE_TRIGGER_UP_FRACTION) {
    raiseArmed = true;
    if (DEBUG || TRIGGERTEST) debugEvents.raiseFired = nowMs; // display-only latch, see DEBUG_EVENT_LATCH_MS
  } else if (raiseArmed && height <= RAISE_TRIGGER_DOWN_FRACTION) {
    raiseArmed = false;
  }
}
// ===========================================================================

// ===== ARM CONE — see FULL_DRAW_ARM_CONE_APERTURE_DEG's own comment for the field report this
// exists to fix (a bow arm hanging straight at the archer's side passes plain straightness).
//
// The cone's axis is horizontal ELEVATION (the bow shoulder→wrist line's angle off horizontal),
// not an attempted 3D "pointing in front of the body" direction — a single camera's 2D projection
// cannot recover that reliably, and picking the wrong axis produces a check that works from one
// camera position and silently fails from another. CLAUDE.md already records that exact class of
// bug twice (the aspect-ratio distortion toPixelSpace fixes, and hand separation collapsing from
// 1.571 to 0.247 under pure body rotation). Elevation avoids it on three counts: it's measured
// against gravity (the image's y-axis), so it doesn't depend on which way the archer faces, the
// mirror toggle, or the handedness toggle — the three things that have broken directional logic in
// this file before; it kills the reported failure mode directly (an arm hanging at the side reads
// ~90° off horizontal, nowhere near a realistic aperture); and it needs no shoulder-line reference,
// which CLAUDE.md separately records as nearly degenerate in this app's side-on framing (the same
// reason drawElbowAlignmentOf was written to avoid it).
//
// dx is folded through Math.abs() specifically so this reads the same magnitude whether the bow
// arm extends to the camera's left or right — which side depends only on facing direction and the
// mirror toggle, neither of which may move a measured number (see CLAUDE.md's mirroring Key
// decision: mirroring never touches a landmark coordinate). Positive = wrist above shoulder,
// negative = below, 0 = level — ordinary atan2 sign convention flipped once for image y growing
// downward.
//
// DELIBERATELY NOT IMPLEMENTED: a horizontal half of the cone (front-of-body vs. behind-the-back),
// which the owner's own phrasing ("in front of my shoulder") and instructions explicitly allowed
// for IF it could be made reliable. The only candidate reference for "which way is forward" in a
// single 2D frame is the shoulder-to-shoulder line — exactly the reference CLAUDE.md already
// documents as nearly degenerate in this app's side-on framing (both shoulders project close
// together in x). Building the horizontal half on that reference would reintroduce the same
// unreliable-axis problem elevation was chosen specifically to avoid, so it's left out rather than
// shipped silently wrong. Elevation alone still catches the reported case (arm at the side) outright.
//
// Uses toPixelSpace (see that function's own comment) so this is computed in a physically honest
// space — an aspect-ratio-distorted frame would report a false angle here exactly like it once did
// for bowArmAngleOf, before that fix. Never guesses: null if the bow shoulder or bow wrist isn't
// confidently visible, same convention as bowArmRaiseHeight just above.
function bowArmElevationOf(landmarks, frameWidth, frameHeight) {
  const bowShoulder = rightHanded ? L_SHOULDER : R_SHOULDER;
  const bowWrist = rightHanded ? L_WRIST : R_WRIST;
  if (!visible(landmarks, bowShoulder) || !visible(landmarks, bowWrist)) return null;
  const shoulder = toPixelSpace(landmarks[bowShoulder], frameWidth, frameHeight);
  const wrist = toPixelSpace(landmarks[bowWrist], frameWidth, frameHeight);
  const dx = wrist.x - shoulder.x;
  const dy = wrist.y - shoulder.y;
  return -(Math.atan2(dy, Math.abs(dx)) * 180) / Math.PI;
}
// ===========================================================================

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

// ===== CALIBRATION — measurement. Unlike every other measure in this file, this doesn't care
// which arm is "bow" or "draw" — it's a fact about the archer's own body, not his stance — so it
// just prefers whichever side's landmarks happen to be clearer, same "own side, other side as
// fallback" convention used everywhere else (torsoLength, shoulderDropOf).

// Arm length (shoulder to wrist, via the elbow) for one side, or null if any of the three joints
// on that side isn't confidently visible. Sum of the two segments, not the straight shoulder-to-
// wrist distance — he's just standing there during calibration, not holding a T-pose, and a
// slightly bent elbow must not read as a shorter arm than he actually has.
function armLengthOf(landmarks, shoulderIdx, elbowIdx, wristIdx, frameWidth, frameHeight) {
  if (![shoulderIdx, elbowIdx, wristIdx].every((i) => visible(landmarks, i))) return null;
  const s = toPixelSpace(landmarks[shoulderIdx], frameWidth, frameHeight);
  const e = toPixelSpace(landmarks[elbowIdx], frameWidth, frameHeight);
  const w = toPixelSpace(landmarks[wristIdx], frameWidth, frameHeight);
  return Math.hypot(s.x - e.x, s.y - e.y) + Math.hypot(e.x - w.x, e.y - w.y);
}

// Both ankles visible — the calibration framing check ("fully in frame" vs. "cut off at the
// legs"). BOTH, not either: side-on, one ankle can legitimately hide behind the other, but
// calibration happens standing roughly toward the phone before he's walked to the line, so seeing
// neither ankle is the honest sign his feet are outside the frame, not just self-occluded.
function bothAnklesVisible(landmarks) {
  return visible(landmarks, L_ANKLE) && visible(landmarks, R_ANKLE);
}

// One frame's calibration proportions, or null if this frame can't support a trustworthy reading
// — a missing/low-confidence landmark, a torso estimate to divide by, or a near side-on shoulder
// projection (see MIN_SHOULDER_TO_TORSO). Ratios only, never an absolute size — he stands at a
// different distance from the phone every time; ratios are the only thing that survives that
// (dividing by frameWidth/frameHeight via toPixelSpace, then again by torso length, cancels out
// both his distance from the camera AND the frame's own resolution). Averages both sides' arm
// length when both are visible, for a steadier number; either measure alone still counts.
function bodyProportionsOf(landmarks, frameWidth, frameHeight) {
  const torso =
    torsoLength(landmarks, L_SHOULDER, L_HIP, frameWidth, frameHeight) ??
    torsoLength(landmarks, R_SHOULDER, R_HIP, frameWidth, frameHeight);
  if (!torso) return null;

  const arms = [
    armLengthOf(landmarks, L_SHOULDER, L_ELBOW, L_WRIST, frameWidth, frameHeight),
    armLengthOf(landmarks, R_SHOULDER, R_ELBOW, R_WRIST, frameWidth, frameHeight),
  ].filter((v) => v != null);
  if (arms.length === 0) return null;
  const armLength = arms.reduce((a, b) => a + b, 0) / arms.length;

  if (!visible(landmarks, L_SHOULDER) || !visible(landmarks, R_SHOULDER)) return null;
  const ls = toPixelSpace(landmarks[L_SHOULDER], frameWidth, frameHeight);
  const rs = toPixelSpace(landmarks[R_SHOULDER], frameWidth, frameHeight);
  const shoulderWidth = Math.hypot(ls.x - rs.x, ls.y - rs.y);
  // Near side-on, both shoulders project close together in x — see CLAUDE.md's own note on this
  // — and the resulting tiny "width" is a projection artefact, not a real measurement. Below
  // MIN_SHOULDER_TO_TORSO this frame contributes NO calibration at all, rather than a garbage
  // ratio nobody could ever match again.
  if (shoulderWidth / torso < MIN_SHOULDER_TO_TORSO) return null;

  const earIdx = visible(landmarks, L_EAR) ? L_EAR : visible(landmarks, R_EAR) ? R_EAR : null;
  if (earIdx === null || !visible(landmarks, NOSE)) return null;
  const nose = toPixelSpace(landmarks[NOSE], frameWidth, frameHeight);
  const ear = toPixelSpace(landmarks[earIdx], frameWidth, frameHeight);
  const headSize = Math.hypot(nose.x - ear.x, nose.y - ear.y); // nose-to-ear: stays readable from a side-on view, unlike ear-to-ear (see the shoulder-width note above — the same degeneracy would apply)

  return {
    armToTorso: armLength / torso,
    shoulderToTorso: shoulderWidth / torso,
    headToTorso: headSize / torso,
  };
}

// Turns a calm stretch's per-frame proportions into ONE calibration — each ratio medianed
// independently, reusing the exact same median() function and reasoning a shot's own numbers
// already use (see medianSampleOf above): a calibration built from a single lucky/unlucky frame
// is exactly the kind of noise this whole feature exists to avoid trusting. Returns null if there
// weren't enough good frames yet (see CALIBRATION_MIN_SAMPLES) — the caller (sampleForCalibration)
// just keeps collecting rather than treating this as a final failure.
function medianCalibrationOf(samples) {
  if (samples.length < CALIBRATION_MIN_SAMPLES) return null;
  return {
    armToTorso: median(samples.map((s) => s.armToTorso)),
    shoulderToTorso: median(samples.map((s) => s.shoulderToTorso)),
    headToTorso: median(samples.map((s) => s.headToTorso)),
  };
}

// ===== FRAMING SIGNATURE — measurement (optional add-on, see its own constants-block comment
// above). Same pure, no-DOM discipline as bodyProportionsOf above, and shares its torso-length
// scale reference, but answers a different question: not "what are his body's proportions" but
// "how was he set up relative to the camera" — apparent size, roughly where in the frame he sits,
// and how square-on he was. Returns null (no signature at all, never a guessed one) if the
// landmarks it needs aren't confidently visible.
function framingSignatureOf(landmarks, frameWidth, frameHeight) {
  const torso =
    torsoLength(landmarks, L_SHOULDER, L_HIP, frameWidth, frameHeight) ??
    torsoLength(landmarks, R_SHOULDER, R_HIP, frameWidth, frameHeight);
  if (!torso || !frameHeight) return null;

  // Hip midpoint, already normalised [0,1] fractions of the frame — MediaPipe's own coordinate
  // convention, no conversion needed — reused as-is from the ATTENTION GATING block above rather
  // than inventing a second "roughly where he's standing" reference point.
  const ref = bodyReferencePoint(landmarks);
  if (!ref) return null;
  if (!visible(landmarks, L_SHOULDER) || !visible(landmarks, R_SHOULDER)) return null;

  const ls = toPixelSpace(landmarks[L_SHOULDER], frameWidth, frameHeight);
  const rs = toPixelSpace(landmarks[R_SHOULDER], frameWidth, frameHeight);
  const lh = toPixelSpace(landmarks[L_HIP], frameWidth, frameHeight);
  const rh = toPixelSpace(landmarks[R_HIP], frameWidth, frameHeight);

  return {
    // How big he is in frame — DELIBERATELY not a torso-length ratio like the body-proportion
    // measures above; this one is SUPPOSED to change with how far he stands from the camera, not
    // cancel it out. That's the "am I standing where I stood last time" signal.
    apparentSize: torso / frameHeight,
    // Roughly where in the frame he's standing — a moved or re-aimed phone shows up here.
    frameX: ref.x,
    frameY: ref.y,
    // How square-on to the camera he is: viewed properly side-on, the two shoulders (and the two
    // hips) project almost on top of each other, so their HORIZONTAL-ONLY separation (not the
    // full 2D distance shoulderToTorso above uses) is small; turning toward the camera grows it.
    // Normalised by torso length so it doesn't also mean "how big he is in frame" — the variable
    // most likely to corrupt the elbow/shoulder-drop geometry if it changes between sessions.
    shoulderSquareness: Math.abs(ls.x - rs.x) / torso,
    hipSquareness: Math.abs(lh.x - rh.x) / torso,
  };
}

// Same median-of-a-calm-stretch treatment as medianCalibrationOf above, same reasoning, same
// minimum sample bar — kept as its own function (not folded into medianCalibrationOf) because this
// is optional and independent: a session where this never reaches enough good frames must not stop
// the real calibration above from completing.
function medianFramingOf(samples) {
  if (samples.length < CALIBRATION_MIN_SAMPLES) return null;
  return {
    apparentSize: median(samples.map((s) => s.apparentSize)),
    frameX: median(samples.map((s) => s.frameX)),
    frameY: median(samples.map((s) => s.frameY)),
    shoulderSquareness: median(samples.map((s) => s.shoulderSquareness)),
    hipSquareness: median(samples.map((s) => s.hipSquareness)),
  };
}

// Plain-English DESCRIPTION of what changed, never a DIAGNOSIS of why — "you look smaller in frame
// than last time" is something this measurement actually supports; "the camera is lower" isn't (a
// vertical-position change could be camera height, camera tilt, or where he stood, and this can't
// tell those apart). A confident wrong explanation is worse than a vague right one. Returns null
// (say nothing) the moment nothing crosses its tolerance — same "never cry wolf" standard as
// describeFraming/calibrationVerdict above.
function describeFramingChange(stored, fresh) {
  const notes = [];
  if (Math.abs(fresh.apparentSize - stored.apparentSize) / stored.apparentSize > FRAMING_SIZE_TOLERANCE) {
    notes.push(fresh.apparentSize > stored.apparentSize ? "you look bigger in frame than last time" : "you look smaller in frame than last time");
  }
  if (Math.hypot(fresh.frameX - stored.frameX, fresh.frameY - stored.frameY) > FRAMING_POSITION_TOLERANCE) {
    notes.push("you're standing in a different part of the frame than last time");
  }
  const squarenessDelta = Math.max(
    Math.abs(fresh.shoulderSquareness - stored.shoulderSquareness),
    Math.abs(fresh.hipSquareness - stored.hipSquareness)
  );
  if (squarenessDelta > FRAMING_SQUARENESS_TOLERANCE) {
    notes.push("you're angled differently toward the camera than last time");
  }
  if (notes.length === 0) return null;
  return `Framing looks different from your last calibration: ${notes.join("; ")} — form numbers may not compare well session to session until this matches.`;
}

// The null-safe entry point finishPassiveCalibration actually calls — every "optional, nothing
// depends on it" guarantee lives here in one place: no stored signature yet, or this session's own
// reading never got enough good frames, and this says nothing at all, silently, forever (until a
// calibration exists on both ends to compare).
function framingChangeMessage(storedFraming, freshFraming) {
  if (!storedFraming || !freshFraming) return null;
  return describeFramingChange(storedFraming, freshFraming);
}
// ===========================================================================

// Use #1 (HANDOVER.md's own order of value): can a reading like this ever actually happen for
// THIS archer's body? PROVEN, not just plausible — for any two wrist positions, each at most
// armToTorso torso-lengths from its OWN shoulder, and the two shoulders at most shoulderToTorso
// torso-lengths apart, the triangle inequality puts a hard ceiling on how far apart the wrists can
// ever be: armToTorso + shoulderToTorso + armToTorso — reached only in the limiting (never actually
// drawn) case of both arms pointing straight away from each other in a dead line through both
// shoulders. CALIBRATION_PLAUSIBILITY_SLACK adds a little room on top for calibration/measurement
// noise — both of which are already medians over many frames, not single noisy ones — not because
// the physics has any give in it. Returns null (no verdict, never a false "fine") when there's
// nothing to check against — no calibration yet, or this shot's own hand-sep reading was itself
// uncertain.
function maxPlausibleHandSep(calibration) {
  return CALIBRATION_PLAUSIBILITY_SLACK * (2 * calibration.armToTorso + calibration.shoulderToTorso);
}

function handSepIsPlausible(handSep, calibration) {
  if (handSep == null || calibration == null) return null;
  return handSep <= maxPlausibleHandSep(calibration);
}

// Use #2: is the calibration on file still describing the archer standing there right now?
// Per-ratio relative comparison — deliberately NOT resolved automatically either way (see
// HANDOVER.md and the owner's own words: "remember it but recheck it everytime against remembered
// data"). Pure: two calibration objects in, which (if any) ratios disagree beyond
// CALIBRATION_AGREEMENT_TOLERANCE out — calibrationVerdict below decides what to actually do
// about it.
function compareCalibrations(stored, fresh, tolerance = CALIBRATION_AGREEMENT_TOLERANCE) {
  const names = ["armToTorso", "shoulderToTorso", "headToTorso"];
  const mismatches = names.filter((name) => Math.abs(stored[name] - fresh[name]) / stored[name] > tolerance);
  return { agrees: mismatches.length === 0, mismatches };
}

const CALIBRATION_LABELS = { armToTorso: "arm length", shoulderToTorso: "shoulder width", headToTorso: "head size" };

// Plain-English disagreement note — what the owner actually reads, live on screen and later in the
// shot log. Pure: a comparison + the two calibration objects in, one sentence out, same "no DOM, no
// module state" convention as narrateMeasure/buildShareText above, so selfTest can check the
// wording directly. Only ever called when comparison.agrees is false (see calibrationVerdict) —
// agreement says NOTHING, per the owner's explicit "silent unless it disagrees" decision.
function describeCalibrationResult(comparison, stored, fresh) {
  const names = comparison.mismatches.map((n) => CALIBRATION_LABELS[n]).join(", ");
  return `Today's ${names} measurement${comparison.mismatches.length === 1 ? "" : "s"} don't match your saved calibration — using today's numbers for this session, but not overwriting what's saved. Recalibrate on purpose (or check you're standing the way you usually do) if this keeps happening.`;
}

// The actual decision, pulled out as its own pure function so "does this stay silent" is directly
// testable without touching localStorage or any module state (see finishPassiveCalibration, the
// only caller). No stored calibration yet: nothing to disagree with, save it, say nothing. Agrees:
// save the refreshed reading, say nothing — the whole point of running unprompted is that
// there is nothing new for him to remember. Disagrees: do NOT overwrite what's on file, and say so.
function calibrationVerdict(stored, fresh) {
  if (!stored) return { save: true, message: null };
  const comparison = compareCalibrations(stored, fresh);
  if (comparison.agrees) return { save: true, message: null };
  return { save: false, message: describeCalibrationResult(comparison, stored, fresh) };
}

// Told to the owner directly, live and in the shot log, when calibration hasn't managed to
// confirm anything this session — see calibrationStatusText below for why this exists as its own
// state rather than staying silent like agreement does.
const CALIBRATION_NOT_YET_MESSAGE =
  "Calibration hasn't run yet this session — stand where you shoot and face the camera for a couple of seconds when you get the chance.";

// THREE states, not two — the gap this fixes (PM review, 2026-08-23): agreement and "never
// managed to measure you" both used to render as the same thing, silence, and those are not the
// same situation. He shoots side-on, and a side-on shoulder line is genuinely too degenerate to
// calibrate from (see MIN_SHOULDER_TO_TORSO/bodyProportionsOf — that floor is protecting against
// real garbage, not being weakened to force a reading through) — so if his routine never happens
// to include a moment facing the camera, `calibrationDone` can stay false all session, and before
// this fix that looked identical, forever, to everything being fine. `calibrationDone` false is
// the ONLY thing distinguishing this from the other two states, deliberately: it's a fact about
// THIS SESSION, not about whether a calibration exists at all (see calibrationShareLine below for
// that half of the picture, which this function doesn't need). Neutral tone on purpose — nothing
// is broken, there's just something worth doing when he gets the chance; renderCalibrationStatus
// is what keeps this from LOOKING like the amber "worth flagging" warnings elsewhere in this file.
function calibrationStatusText(calibrationDone, disagreementMessage) {
  if (!calibrationDone) return { text: CALIBRATION_NOT_YET_MESSAGE, tone: "neutral" };
  if (disagreementMessage) return { text: disagreementMessage, tone: "warn" };
  return null; // agrees — silent, per the owner's own "silent unless it disagrees" decision
}

// What the PM actually needs from a shared session (see the same review): not just today's
// verdict, but whether a calibration exists AT ALL and when it was last confirmed — without that,
// "no calibration line in the share text" is indistinguishable from "calibration has never once
// run", exactly the ambiguity calibrationStatusText above exists to remove for the owner. Pure:
// the stored record (or null) + today's outcome in, one line out — no Date-object comparisons in
// selfTest, since `stored.takenAt` is a plain epoch-ms number formatted here, not compared.
function calibrationShareLine(stored, calibrationDone, disagreementMessage) {
  const lastLine = stored?.takenAt ? `last confirmed ${new Date(stored.takenAt).toLocaleString()}` : "never calibrated";
  const todayLine = !calibrationDone ? "not yet confirmed this session" : disagreementMessage ? "disagreed with what was stored" : "agreed with what was stored";
  return `Calibration: ${lastLine} — today: ${todayLine}`;
}

// Use #3: a plain framing note, only when there's actually something to say — same "never cry
// wolf" standard as everything else here. legsVisible: whether bothAnklesVisible held true across
// the whole calm stretch checked so far (see sampleForCalibration) — nothing to say when it did.
function describeFraming(legsVisible) {
  return legsVisible
    ? null
    : "Your legs may be outside the frame — stand back so your whole body, head to feet, fits before you shoot.";
}

// ===== CALIBRATION — storage. Deliberately separate from any other persistence in this file (see
// HANDOVER.md) — its own small, independent use of localStorage under its own key, not a shared
// abstraction with whatever the shot-log persistence work elsewhere ends up using. try/catch
// because localStorage can throw (private browsing, storage disabled) and calibration failing to
// persist must never take the rest of the app down with it.
function loadStoredCalibration() {
  try {
    const raw = localStorage.getItem(CALIBRATION_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveCalibration(calibration) {
  try {
    localStorage.setItem(CALIBRATION_STORAGE_KEY, JSON.stringify(calibration));
  } catch {
    // storage unavailable — calibration just won't survive a reload this session; nothing else
    // depends on it, so this fails silently rather than raising an alarm about it
  }
}
// ===========================================================================

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

  // NEVER-BLANK default: cleared to a fresh "nothing computed yet this frame" object rather than
  // bare null, so that even if a future bail path below forgets to set a more specific reason,
  // the panel still shows something honest ("full draw check not yet run this frame") instead of
  // going blank — see emptyDebugInfo's own comment. Every bail branch below overwrites this with
  // its own specific reason; only a path nobody has written yet would ever leave this generic one
  // standing, and even that is a real, readable sentence rather than silence.
  if (DEBUG || TRIGGERTEST) debugInfo = emptyDebugInfo("full draw check not yet run this frame");

  // RAISE TRIGGER: independent of the stricter visibility this function's own full-draw checks
  // need below — see bowArmRaiseHeight's own comment. Updated (and fed to trackShotAttempt via
  // sample.raiseArmed, both here and below) on every frame this CAN be read at all, even one
  // where the rest of this function can't proceed (e.g. the draw elbow is briefly occluded) — the
  // raise must fire as easily as possible: fail toward recording.
  updateRaiseTrigger(landmarks, frameWidth, frameHeight, nowMs);

  if (![drawWrist, bowShoulder, bowElbow, bowWrist].every((i) => visible(landmarks, i))) {
    if (DEBUG || TRIGGERTEST) {
      const missing = describeMissingLandmarks(landmarks, [
        [drawWrist, "draw wrist"],
        [bowShoulder, "bow shoulder"],
        [bowElbow, "bow elbow"],
        [bowWrist, "bow wrist"],
      ]);
      debugInfo = emptyDebugInfo(`${missing.join(", ")} not confidently visible`);
    }
    trackShotAttempt({ handSep: null, raiseArmed, atFullDraw: false, eligible: false }, nowMs);
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
    if (DEBUG || TRIGGERTEST) debugInfo = emptyDebugInfo("no anchor landmark — mouth and nose not confidently visible");
    trackShotAttempt({ handSep: null, raiseArmed, atFullDraw: false, eligible: false }, nowMs);
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
  if (!scale) {
    if (DEBUG || TRIGGERTEST) debugInfo = emptyDebugInfo("no torso scale — hip not confidently visible on either side");
    trackShotAttempt({ handSep: null, raiseArmed, atFullDraw: false, eligible: false }, nowMs);
    return false;
  }

  const wrist = toPixelSpace(landmarks[drawWrist], frameWidth, frameHeight);
  const bowWristPos = toPixelSpace(landmarks[bowWrist], frameWidth, frameHeight);
  const anchorDist = Math.hypot(wrist.x - anchor.x, wrist.y - anchor.y) / scale;
  const handSep = Math.hypot(wrist.x - bowWristPos.x, wrist.y - bowWristPos.y) / scale;

  // ANCHOR DIRECTION — see FULL_DRAW_ANCHOR_ABOVE_MAX/FULL_DRAW_ANCHOR_BACKWARD_MIN's own comments
  // for why a plain distance circle isn't enough. Both measured from the SAME anchor point
  // anchorDist already uses (mouth-midpoint, or nose if the mouth isn't visible), not a second,
  // possibly-disagreeing reference — so "how far" and "which direction" are always talking about
  // the same point.
  //
  // Vertical: image y grows downward, so a wrist ABOVE the anchor has a SMALLER y — hence
  // wrist.y - anchor.y (positive = below, negative = above), the same sign convention
  // bowArmRaiseHeight already uses for shoulder/wrist. Needs only the anchor point and the draw
  // wrist, both already confirmed above — never unmeasurable once anchorDist itself is.
  const anchorVerticalOffset = (wrist.y - anchor.y) / scale;
  const anchorVerticalOk = anchorVerticalOffset >= -FULL_DRAW_ANCHOR_ABOVE_MAX;

  // Backward: "behind the mouth, toward the draw side" has no meaning without a body-relative
  // axis, and the shoulder line is nearly degenerate in this app's side-on framing (see CLAUDE.md
  // — the same reason drawElbowAlignmentOf avoids it) so it cannot supply one safely. The
  // anchor-point→draw-ear vector can: it's a real, local direction on the head itself, present in
  // every frame the anchor check already needs the head confidently visible for.
  //
  // Prefer the draw-side ear (physically where a release-aid hand ends up, so it best captures
  // "backwards" from the owner's perspective); fall back to the bow-side ear if the draw-side one
  // is occluded — a side-on archer often has one ear or the other hidden from the camera, and the
  // same "own side preferred, other side as fallback" convention already used by
  // shoulderDropOf's ear lookup applies here for the same reason (running off whichever ear IS
  // visible beats not running at all). If NEITHER ear is confidently visible, this cannot be
  // measured at all — never guess a direction from a landmark that isn't there: anchorBackwardOk
  // stays null (not true), so the combined anchorOk below cannot pass on an unconfirmed direction.
  const drawEar = rightHanded ? R_EAR : L_EAR;
  const bowEarForAnchor = rightHanded ? L_EAR : R_EAR;
  const anchorEarIdx = visible(landmarks, drawEar) ? drawEar : visible(landmarks, bowEarForAnchor) ? bowEarForAnchor : null;
  let anchorBackward = null, anchorBackwardOk = null, anchorEarPx = null;
  if (anchorEarIdx !== null) {
    const ear = toPixelSpace(landmarks[anchorEarIdx], frameWidth, frameHeight);
    anchorEarPx = ear;
    const axisX = ear.x - anchor.x, axisY = ear.y - anchor.y;
    const axisLen = Math.hypot(axisX, axisY);
    if (axisLen > 0) {
      // Signed component of the anchor->wrist vector along the anchor->ear axis, normalised to
      // the same torso-length units anchorDist/handSep already use — positive means toward the
      // ear (backwards), negative means toward the front of the face.
      const wristVecX = wrist.x - anchor.x, wristVecY = wrist.y - anchor.y;
      anchorBackward = (wristVecX * axisX + wristVecY * axisY) / axisLen / scale;
      anchorBackwardOk = anchorBackward >= FULL_DRAW_ANCHOR_BACKWARD_MIN;
    }
  }

  const anchorDistOk = anchorDist <= FULL_DRAW_ANCHOR_MAX;
  const anchorOk = anchorDistOk && anchorVerticalOk && anchorBackwardOk === true;

  const bowArmAngle = bowArmAngleOf(landmarks, frameWidth, frameHeight);
  if (bowArmAngle === null) {
    // handSep is already known at this point (computed above) even though the bow arm's own
    // angle isn't — no reason to throw away a real reading here, including on the ?debug panel.
    if (DEBUG || TRIGGERTEST) debugInfo = { ...emptyDebugInfo("bow-arm angle unavailable — shoulder/elbow/wrist are degenerate (identical positions)"), handSep };
    trackShotAttempt({ handSep, raiseArmed, atFullDraw: false, eligible: false }, nowMs);
    return false;
  }

  // ARM CONE — see bowArmElevationOf's own comment for the geometry and why elevation (not a 3D
  // "in front of the body" direction) is what's measured. Only the shoulder/wrist visibility
  // bowArmElevationOf checks for itself is required, both already confirmed above, so this can
  // never be unmeasurable once bowArmAngle itself is.
  const armElevation = bowArmElevationOf(landmarks, frameWidth, frameHeight);
  const armStraightOk = bowArmAngle >= FULL_DRAW_BOW_ARM_MIN;
  const armConeOk = armElevation !== null && Math.abs(armElevation) <= FULL_DRAW_ARM_CONE_APERTURE_DEG;
  const armOk = armStraightOk && armConeOk;

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

  const sepOk = handSep >= FULL_DRAW_HAND_SEP_MIN;
  const stillOk = speed <= FULL_DRAW_STILL_MAX;

  if (DEBUG || TRIGGERTEST) {
    debugInfo = {
      reason: null,
      anchorDist, anchorOk, anchorVerticalOffset, anchorVerticalOk, anchorBackward, anchorBackwardOk,
      handSep, sepOk,
      bowArmAngle, armStraightOk, armElevation, armConeOk, armOk,
      speed, stillOk,
      // Pixel-space points/scale for drawTriggerTestOverlay — see emptyDebugInfo's own comment.
      anchorPx: anchor, drawWristPx: wrist, bowWristPx: bowWristPos,
      bowShoulderPx: toPixelSpace(landmarks[bowShoulder], frameWidth, frameHeight),
      anchorEarPx, scale,
    };
  }

  const atFullDraw = anchorOk && armOk && sepOk && stillOk;

  // Feed the shot log regardless of ?debug — the owner needs shot numbers/form readouts
  // whether or not the diagnostic overlay is on; only the display of the extra fields below
  // is debug-gated (see renderShotLog). nowMs is threaded through rather than calling
  // performance.now() again in here, so selfTest can drive the shot-log timing deterministically.
  trackShotAttempt(
    {
      handSep,
      raiseArmed,
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
// separation stays at/above DRAW_ATTEMPT_MIN_SEP, OR the raise trigger is armed (see RAISE
// TRIGGER above) — whichever comes first. It ends — and gets judged for logging — once BOTH are
// false again (hands back together AND the bow arm back down). Two independent ways in, because
// occlusion can take either one away on any given attempt (fail toward recording, per CLAUDE.md):
// a raise that's missed still gets caught once hands separate for real; a full draw whose raise
// was missed (bow arm never confidently visible, say) still gets caught by hand separation, same
// as before this feature existed. This is the simplest rule that both (a) doesn't split one long
// hold into several rows and (b) doesn't merge two separate shots taken back-to-back into one.
// RAISE_ATTEMPT_TIMEOUT_MS is the one timer involved, and only for a raise that never turns into a
// real draw (see below) — nothing else here expires on its own.
//
// The raise decides WHEN to watch; it must never change what gets MEASURED. So a fresh attempt
// opened by the raise alone starts with startMs left null — "has the real draw actually started"
// — and peakHandSep/eligibleFrames/reachedFullDraw stay untouched until hand separation itself
// crosses DRAW_ATTEMPT_MIN_SEP, at which point startMs is set (once) and every existing gate,
// median and duration check runs exactly as it did before the raise trigger existed. A raise-phase
// frame (arm up, hands not yet apart) never contributes a sample to the median — see
// medianSampleOf — because it's never added to eligibleFrames in the first place.
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
  const drawing = sample.handSep !== null && sample.handSep >= DRAW_ATTEMPT_MIN_SEP;
  const inMotion = drawing || sample.raiseArmed;

  if (inMotion) {
    const isNewAttempt = !attempt; // the raise armed, or hands just left the resting position — a fresh attempt, not a continuation
    if (isNewAttempt) {
      attempt = {
        startMs: null, // set below the first time `drawing` is actually true — a raise alone must never start the SHOT_MIN_DURATION_MS clock, see this function's own block comment
        watchStartedAt: nowMs, // when THIS attempt object was created (raise or hand-separation, whichever fired first) — only used by RAISE_ATTEMPT_TIMEOUT_MS below
        peakHandSep: 0,
        eligibleFrames: [],
        eligibleSeen: 0,
        reachedFullDraw: false,
      };
      if (DEBUG || TRIGGERTEST) debugEvents.attemptStarted = nowMs; // display-only latch, see DEBUG_EVENT_LATCH_MS
      // Recording starts here, not in endAttempt, so the raise and draw are in the clip too — by
      // the time endAttempt fires the good part is already over. Starts regardless of this
      // frame's eligibility — the clip is a recording of what happened, not a measurement.
      startClipRecording();
    }
    if (drawing) {
      if (attempt.startMs === null) attempt.startMs = nowMs; // the real draw has now started — see this function's own block comment
      if (sample.handSep > attempt.peakHandSep) attempt.peakHandSep = sample.handSep;
      if (sample.eligible) {
        reservoirAdd(attempt, sample);
        // "Did any settled frame reach true full draw" — a plain OR over every eligible frame this
        // attempt has seen, independent of which frames the reservoir above happened to keep (a
        // frame that gets evicted from the reservoir must not un-say that full draw was reached;
        // the reservoir bounds MEMORY for the medians, it must never bound what this flag can see).
        attempt.reachedFullDraw = attempt.reachedFullDraw || !!sample.atFullDraw;
      }
    } else if (attempt.startMs === null && nowMs - attempt.watchStartedAt >= RAISE_ATTEMPT_TIMEOUT_MS) {
      // Armed by a raise that never turned into hands actually separating — closes cleanly as a
      // rejected attempt (peakHandSep is still 0, so endAttempt's own gate throws it out) rather
      // than leaving the app "watching" for the rest of the session. See RAISE_ATTEMPT_TIMEOUT_MS.
      endAttempt(nowMs);
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
    saveSessionToStorage(); // this counter has its own line in the log/share text — see SESSION PERSISTENCE
    signalOutcome(false); // "seen but rejected" — a draw was open a moment ago (see the watching cue), it just didn't earn a row
    return;
  }

  if (a.eligibleFrames.length === 0) {
    unsettledAttemptCount++;
    discardRecording(activeRecording); // same reasoning as the rejected case above — never leave a clip with no shot to attach to, and never report it as a failure either
    renderShotLog();
    saveSessionToStorage(); // same reasoning as the rejected branch above
    signalOutcome(false); // same cue as the rejected case above — the owner asked for "seen but not logged", not a separate state per reason why
    return;
  }

  // Each of the four real measures (plus the diagnostic handSep/flags) is the MEDIAN of its own
  // eligible frames, computed independently — see medianSampleOf's own comment for the full
  // reasoning and what this replaced.
  const medianSample = medianSampleOf(a.eligibleFrames);
  // CALIBRATION nonsense-check (HANDOVER.md's most valuable use of the three): is this hand-sep
  // reading even physically possible for this archer, as calibrated? `=== false` specifically,
  // not just falsy — handSepIsPlausible returns null (no verdict) whenever there's no calibration
  // yet or the reading itself was uncertain, and null must never render as a flag (see
  // handSepIsPlausible's own comment and the "never cry wolf" rule in HANDOVER.md).
  const implausible = handSepIsPlausible(medianSample.handSep, activeCalibration) === false;
  const shotNum = logShot({ ...medianSample, startMs: a.startMs, reachedFullDraw: a.reachedFullDraw, implausible });
  if (DEBUG) debugEvents.shotLogged = nowMs; // display-only latch, see DEBUG_EVENT_LATCH_MS
  // The clip that's been recording since this attempt began now knows which shot it belongs to,
  // and can start counting down its post-release tail (see attachRecordingToShot).
  attachRecordingToShot(shotNum);
  // Green means an arrow, not "a row got added". "Fail toward recording" says keep the evidence
  // when unsure — it does not say call it an arrow: a logged draw that never reached full draw
  // (anchorOk/armOk/stillOk never all true — see isAtFullDraw) is real, worth keeping, and marked
  // as short of full draw right on its row (see renderShotRow's shortMark) — but flashing the
  // "that arrow counted" cue for it would be a false statement delivered as reassurance to someone
  // who cannot check it, at the one moment that distinction matters most. See a.reachedFullDraw
  // (an OR over every eligible frame, set in trackShotAttempt) and signalOutcome's own comment.
  signalOutcome(a.reachedFullDraw); // true only for a real arrow; a logged-but-short draw gets the same "seen, not confirmed" cue as a rejected one
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
  saveSessionToStorage(); // latches once per session — see SESSION PERSISTENCE
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
  saveSessionToStorage(); // a row's clipFailReason is one of the small facts a restore keeps — see SESSION PERSISTENCE
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
  if (entry.reachedFullDraw) fullDrawShotCount++; // see that counter's own comment — a row that never reached full draw still gets a shotNum, just never counts as an arrow
  log.unshift({ ...entry, shotNum });
  const evicted = log.slice(SHOT_LOG_MAX);
  log = log.slice(0, SHOT_LOG_MAX);
  evicted.forEach(revokeClip);
  renderShotLog();
  saveSessionToStorage(); // a shot just got added — see SESSION PERSISTENCE below
  return shotNum;
}

// ===== SESSION PERSISTENCE — survives a reload, not a shutdown of the app for the day. =====
// The shot log above lives in a plain in-memory array, and until now died the instant the page
// reloaded — which iOS Safari does aggressively to a backgrounded tab (he locks the phone, takes
// a call, switches apps to check something) or to this app relaunched from the Home Screen. He
// can't watch for that and can't stop it; he'd just walk back to the phone and find an empty log,
// with no way to tell "nothing happened" from "it happened and got lost." This is a deliberate,
// narrow exception to CLAUDE.md's "no persistence across sessions" rule — the same kind the
// calibration routine gets (see HANDOVER.md) — made to survive a crash/reload mid-session, not to
// carry one range trip into the next.
//
// What gets saved: the log itself, plus the running counters the log and Share text actually
// read (shotCount, rejectedAttemptCount, unsettledAttemptCount, the two attention counters,
// clipsUnavailableReason). What never gets saved: video clips. A clip is a Blob living only in
// this tab's memory, referenced by a blob: object URL created fresh each page load — there is no
// way to put a Blob in localStorage that survives a reload, and CLAUDE.md rules out adding
// anything heavy enough to try (e.g. IndexedDB video storage). So a restored row says its clip is
// gone in plain language instead of offering a Watch button that would open a dead link.
const SHOT_SESSION_STORAGE_KEY = "archery-form-coach:shot-session:v1";
const SHOT_SESSION_FORMAT_VERSION = 1;

// How stale a saved session may be before a reload starts fresh instead of restoring it — the
// guard that keeps "survive a reload" from quietly becoming "remember forever" (CLAUDE.md's rule
// still stands; this is a narrow, timestamped exception to it, not a repeal). Three hours
// comfortably covers one sitting at the range — shooting an end, walking to pull arrows, a water
// break, even a long lunch mid-session — while staying well short of "he put the phone down and
// came back the next day," which is exactly the case this must NOT resurrect.
const SESSION_RESTORE_MAX_AGE_MS = 3 * 60 * 60 * 1000; // 3 hours

// True for the rest of this page load once a saved session has actually been restored into the
// live state below — drives the "session recovered" notice in renderShotLog/buildShareText. Set
// in exactly one place (restoreSessionFromStorage) and never cleared: once he's seen the notice,
// everything logged for the rest of this load is added on top of a recovered session, and the
// share text should keep saying so.
let sessionWasRestored = false;

// Strips a log entry down to what a save/reload round trip can honestly keep. clipBlob and
// clipUrl both die with the page (see this section's own top comment) and must never be written
// to storage — but whether this row DID have a clip before the reload is a small, plain fact
// worth keeping, so a restored row can say "clip lost" rather than looking like it never had one.
// Pure — no DOM, no URL.* calls — safe to run on an entry still live in `log`.
function entryForPersistence(e) {
  const { clipBlob, clipUrl, ...rest } = e;
  return { ...rest, hadClip: !!clipUrl };
}

// Builds the plain, JSON-safe object written to storage. Pure — takes the pieces of module state
// it needs as arguments rather than reading globals itself — so selfTest can round-trip it
// directly without touching the real log or localStorage.
function serializeShotSession(state, nowMs) {
  return {
    v: SHOT_SESSION_FORMAT_VERSION,
    savedAt: nowMs,
    log: state.log.map(entryForPersistence),
    shotCount: state.shotCount,
    fullDrawShotCount: state.fullDrawShotCount,
    rejectedAttemptCount: state.rejectedAttemptCount,
    unsettledAttemptCount: state.unsettledAttemptCount,
    attentionIdlePeriods: state.attentionIdlePeriods,
    attentionLateWakeCount: state.attentionLateWakeCount,
    clipsUnavailableReason: state.clipsUnavailableReason,
  };
}

// The other half of the round trip: takes whatever came back out of storage (already
// JSON.parsed) and returns either a usable state object or null. Null covers every way this can
// go wrong — nothing saved yet, a corrupt/partial/hand-edited payload, a future format version
// this code doesn't know, or a payload that parsed fine but is simply too old (see
// SESSION_RESTORE_MAX_AGE_MS) — so the caller never has to tell those cases apart, only "restore
// this" or "start fresh," exactly like today's behaviour when nothing was ever saved at all. Pure
// and defensive on purpose: this runs on whatever a previous (possibly different) version of this
// file left behind, so it must never throw, no matter how mangled that is.
function deserializeShotSession(raw, nowMs, maxAgeMs) {
  if (!raw || typeof raw !== "object") return null;
  if (raw.v !== SHOT_SESSION_FORMAT_VERSION) return null;
  if (typeof raw.savedAt !== "number" || !Number.isFinite(raw.savedAt)) return null;
  if (raw.savedAt > nowMs || nowMs - raw.savedAt > maxAgeMs) return null; // too old, or a clock-skewed future timestamp — either way, don't trust it
  if (!Array.isArray(raw.log)) return null;

  // Each entry just needs a real shot number to be worth keeping — the four measures are already
  // allowed to be null in a perfectly ordinary LIVE entry (an uncertain joint, see
  // MIN_VISIBILITY), so a restored one can't be held to a stricter standard than a fresh one ever
  // is. A clip that existed before the restart (hadClip) becomes clipLostOnRestore instead of a
  // real clipUrl — see renderShotRow/shareLineForEntry — and clipBlob/clipUrl are never trusted
  // out of storage even if a hand-edited payload somehow includes them.
  const log = [];
  for (const e of raw.log) {
    if (!e || typeof e !== "object" || typeof e.shotNum !== "number" || !Number.isFinite(e.shotNum)) return null;
    const { clipBlob, clipUrl, hadClip, ...rest } = e;
    log.push({ ...rest, clipLostOnRestore: !!hadClip });
  }

  const num = (v, fallback) => (typeof v === "number" && Number.isFinite(v) ? v : fallback);
  return {
    log: log.slice(0, SHOT_LOG_MAX), // defensive re-cap — the live log is always already within this, but never trust a stored payload more than a fresh computation
    shotCount: num(raw.shotCount, 0),
    // Fallback isn't a flat 0 like the others: a payload saved by an older build (before this
    // counter existed) still has reachedFullDraw on every entry it kept, so counting those is a
    // real answer, not a guess — better than silently undercounting a session already in progress.
    fullDrawShotCount: num(raw.fullDrawShotCount, log.filter((e) => e.reachedFullDraw).length),
    rejectedAttemptCount: num(raw.rejectedAttemptCount, 0),
    unsettledAttemptCount: num(raw.unsettledAttemptCount, 0),
    attentionIdlePeriods: num(raw.attentionIdlePeriods, 0),
    attentionLateWakeCount: num(raw.attentionLateWakeCount, 0),
    clipsUnavailableReason: typeof raw.clipsUnavailableReason === "string" ? raw.clipsUnavailableReason : null,
  };
}

// Writes the current session to storage. Called only from the handful of places that actually
// change something this restores (a shot logged, a movement rejected, an idle period, clip
// recording found to be unavailable) — never from the render loop, which is performance-sensitive
// (see the brief this shipped from) — so this runs at most a few times per shot, not per frame.
// Wrapped end to end: Safari Private Browsing throws on setItem (it still has localStorage, just
// a zero quota), and any other storage failure must degrade to exactly today's in-memory-only
// behaviour, never break the very shot it exists to protect.
function saveSessionToStorage() {
  try {
    const payload = serializeShotSession(
      { log, shotCount, fullDrawShotCount, rejectedAttemptCount, unsettledAttemptCount, attentionIdlePeriods, attentionLateWakeCount, clipsUnavailableReason },
      Date.now()
    );
    localStorage.setItem(SHOT_SESSION_STORAGE_KEY, JSON.stringify(payload));
  } catch (err) {
    // Silent on purpose — see this function's own comment. There's nothing the owner can act on
    // mid-shot, and the app must carry on exactly as if persistence didn't exist.
    console.error("archery-form-coach: could not save session", err);
  }
}

// Reads whatever's in storage and, if it's usable, restores it into the live module state before
// the first frame of this page load is ever rendered — see the call site near the bottom of this
// file. Every failure mode (nothing saved, JSON.parse throwing on corrupt text, storage
// unavailable at all) falls through to doing nothing, i.e. exactly today's behaviour: an empty
// in-memory log.
function restoreSessionFromStorage() {
  let raw = null;
  try {
    const text = localStorage.getItem(SHOT_SESSION_STORAGE_KEY);
    if (text) raw = JSON.parse(text);
  } catch (err) {
    console.error("archery-form-coach: could not read saved session", err);
    return;
  }
  const restored = deserializeShotSession(raw, Date.now(), SESSION_RESTORE_MAX_AGE_MS);
  if (!restored) return;

  log = restored.log;
  shotCount = restored.shotCount;
  fullDrawShotCount = restored.fullDrawShotCount;
  rejectedAttemptCount = restored.rejectedAttemptCount;
  unsettledAttemptCount = restored.unsettledAttemptCount;
  attentionIdlePeriods = restored.attentionIdlePeriods;
  attentionLateWakeCount = restored.attentionLateWakeCount;
  clipsUnavailableReason = restored.clipsUnavailableReason;
  sessionWasRestored = true;
}
// ===========================================================================

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
    ? `<span class="shotlog-debug">hand sep ${e.handSep == null ? "uncertain" : e.handSep.toFixed(2)} — anchor ${e.anchorOk ? "ok" : "fail"} · arm-check ${e.armOk ? "ok" : "fail"} · sep-check ${e.sepOk ? "ok" : "fail"} · still ${e.stillOk ? "ok" : "fail"}</span>`
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

  // CALIBRATION nonsense-check (see handSepIsPlausible/endAttempt) — a data-quality flag about
  // the MEASUREMENT, never a judgement of his form. Only ever set true; never set for a shot
  // logged before any calibration existed this session (see endAttempt's own `=== false` check).
  const implausibleMark = e.implausible
    ? ` <span class="shotlog-implausible">⚠ hands measured farther apart than your calibrated reach allows — check calibration or framing</span>`
    : "";

  // A big, obvious watch button when this shot has a clip; otherwise a "no clip" note — never
  // nothing, so a missing clip never reads as a missing shot. When something specific is known
  // about WHY (see explainClipFailure — a recorder error, an empty recording, one that arrived
  // too late, etc.), that reason is shown instead of the bare word "no clip", since a non-coder
  // owner standing at the phone is the only person who will ever see this and has no console to
  // check instead. A row restored after a reload (clipLostOnRestore — see SESSION PERSISTENCE)
  // gets its own honest wording too: it really did have a clip before the restart, so "no clip"
  // would be a lie, but the clip itself is a Blob that died with the old page and cannot be
  // brought back — never offer a Watch button whose URL is already dead. data-shot carries the
  // shot number for the click handler on shotLogEl (see openClipPlayer wiring) to look the entry
  // back up by.
  const clipBit = e.clipUrl
    ? `<button type="button" class="shotlog-play" data-shot="${e.shotNum}">▶ Watch</button>`
    : `<span class="shotlog-noclip">${e.clipLostOnRestore ? "clip lost — the app restarted" : e.clipFailReason || "no clip"}</span>`;

  return `<div class="shotlog-row"><div class="shotlog-row-main">Shot ${e.shotNum} — ${highlightText}${shortMark}${implausibleMark}${rawHtml}</div><div class="shotlog-row-clip">${clipBit}</div></div>`;
}

// Plain-language shot log — this is what the owner actually reads, standing at the phone after
// their end, not mid-shot, so it has to answer, in order: how many arrows did it see; was he
// consistent, and in what; which shot was the odd one out, and how (so he knows which clip to
// go watch). Nothing here judges his form against a target — see the block comment above
// narrateMeasure for why not. Raw degrees/percent are still there for whoever eventually tunes
// the CALIBRATE WITH COACH constants, just demoted to small print on each row, not the headline.
function renderShotLog() {
  // The very first thing on the page whenever a reload recovered a session (see
  // sessionWasRestored / SESSION PERSISTENCE) — he needs to be able to tell "the app restarted
  // and kept my shots" apart from "the app restarted and lost them" the moment he looks, not
  // after reading everything else first. Set once, at startup, and never cleared for the rest of
  // this page load — everything logged from here on is added on top of a recovered session.
  const restoredBit = sessionWasRestored
    ? `<div class="shotlog-restored">Session recovered after the app restarted. Shots below may include some from before that — their video clips didn't survive, but their numbers did.</div>`
    : "";
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
  // CALIBRATION (HANDOVER.md Stage 4) — same "recorded and still there later" rule as everything
  // else here: renderCalibrationStatus already shows this live, but the owner (or, more likely
  // here, the PM reading a shared log) may only ever check this, so whatever it had to say — a
  // stored-vs-fresh disagreement, calibration never having confirmed itself this session, a
  // legs-cut-off framing note, or a framing-signature change — needs to still be readable after
  // the fact too. Only ever silent on the one state that's actually fine: agreement.
  const calibrationLine = calibrationStatusText(calibrationDone, calibrationStatusLine);
  const calibrationParts = [calibrationLine?.text, framingStatusLine, framingChangeStatusLine].filter(Boolean);
  const calibrationBit = calibrationParts.length ? `<div class="shotlog-modelinfo">${calibrationParts.join(" ")}</div>` : "";
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
    shotLogContentEl.innerHTML = `${restoredBit}${startupBit}${banner}${modelBit}${calibrationBit}${rejectedBit}${unsettledBit}${attentionBit}<div class="shotlog-empty">No shots recorded yet — draw once and this fills in.</div>`;
    return;
  }

  // "Arrows" means confirmed full draw — see fullDrawShotCount's own comment and endAttempt's
  // signalOutcome call. A row that never reached full draw still exists (shotCount, shotNum,
  // its own row below, still marked "short of full draw") but must never inflate this headline.
  const arrowWord = fullDrawShotCount === 1 ? "arrow" : "arrows";
  const shownNote = log.length < shotCount ? ` The consistency lines below are based on your most recent ${log.length}.` : "";
  // The consistency numbers (narrateMeasure/summarizeShots below) must be built ONLY from
  // confirmed arrows — see the coordinator's own reasoning: a phantom's junk numbers averaged
  // into "steady / drifted / stood out" would corrupt the one claim CLAUDE.md lets this app make,
  // silently, which is worse than a visible phantom row. Said explicitly, same convention as
  // shownNote above, rather than just quietly narrowing the population.
  const fullDrawLog = log.filter((e) => e.reachedFullDraw);
  const excludedNote =
    fullDrawLog.length < log.length
      ? ` ${log.length - fullDrawLog.length} shown row${log.length - fullDrawLog.length === 1 ? "" : "s"} never reached full draw and ${log.length - fullDrawLog.length === 1 ? "isn't" : "aren't"} included in the consistency numbers below.`
      : "";
  const countLine = `<div class="shotlog-count">${fullDrawShotCount} ${arrowWord} this session.${shownNote}${excludedNote}</div>`;

  const bowArm = narrateMeasure(fullDrawLog, (e) => e.bowArmAngle, "Bow arm", wordForBowArm, BOW_ARM_CONSISTENCY_FLOOR_DEG);
  const shoulderBow = narrateMeasure(fullDrawLog, (e) => e.shoulderDrop?.bow ?? null, "Bow shoulder", wordForShoulder, SHOULDER_BOW_CONSISTENCY_FLOOR_PCT);
  const shoulderDraw = narrateMeasure(fullDrawLog, (e) => e.shoulderDrop?.draw ?? null, "Draw shoulder", wordForShoulder, SHOULDER_DRAW_CONSISTENCY_FLOOR_PCT);
  const elbow = narrateMeasure(fullDrawLog, (e) => e.elbowAlign?.signed ?? null, "Draw elbow", wordForElbow, ELBOW_CONSISTENCY_FLOOR_DEG);

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

  // Built from fullDrawLog, same reasoning as narrateMeasure above — a short-draw row's own raw
  // numbers still print (renderShotRow shows them regardless), but shotValueHtml only colours/
  // compares a shotNum that's actually a key in stats.deviations; a shotNum excluded here simply
  // renders as a plain, uncoloured number instead of a fake "(+N)" comparison against a
  // population it was never part of.
  const stats = summarizeShots(fullDrawLog);
  const rowsHtml = log.map((e) => renderShotRow(e, stats, outliers, words)).join("");

  shotLogContentEl.innerHTML = `${restoredBit}${startupBit}${banner}${modelBit}${calibrationBit}${rejectedBit}${unsettledBit}${attentionBit}${countLine}<div class="shotlog-narrative">${narrativeHtml}</div>${rowsHtml}`;
}

// ===== SHARE — the owner's only route to a full session's numbers. He can't remember to add
// ?debug before he starts shooting (he isn't even holding the phone by then), and a screenshot
// loses whatever's off the bottom of the screen and isn't machine-readable at the other end. So
// Share always includes everything ?debug shows on a row (hand sep, the four trigger booleans)
// PLUS the session-level lines renderShotLog already computes — regardless of ?debug.
//
// buildShareText is the ONE pure function that does the actual text generation — entries (a
// log-shaped array, any order) and a plain counters object in, one string out. No DOM, no
// navigator, no module state read directly, so selfTest can assert on it exactly like
// summarizeShots/narrateMeasure above. `entries` is deliberately just "whatever log currently
// holds", the same SHOT_LOG_MAX-capped list renderShotLog reads — counters.shotCount is the true
// session total, and the two can disagree (see the truncation notice below) whenever more than
// SHOT_LOG_MAX arrows have been shot.
function buildShareText(entries, counters) {
  const {
    shotCount,
    fullDrawShotCount,
    rejectedAttemptCount,
    unsettledAttemptCount,
    attentionIdlePeriods,
    attentionLateWakeCount,
    clipsUnavailableReason,
    modelStatusLine,
    rightHanded,
    mirrored,
    cameraWidth,
    cameraHeight,
    sessionWasRestored,
    // Defaulted (not required) so every existing caller/fixture that doesn't know about
    // calibration yet keeps working unchanged — see calibrationShareLine's own comment for why
    // the PM needs this line at all.
    storedCalibration = null,
    calibrationDone = false,
    calibrationStatusLine: calibrationDisagreement = null,
  } = counters;

  const lines = [];
  lines.push("Archery form coach — session share");
  // A restored session is a slightly different animal from a clean one — whoever reads this
  // needs to know some of the shots below may predate an app restart, and that any of them could
  // be missing a clip that genuinely existed once (see SESSION PERSISTENCE / clipLostOnRestore
  // below). Right under the title, same "first thing read" placement as the on-screen notice.
  if (sessionWasRestored) {
    lines.push("NOTE: recovered after the app restarted mid-session — some shots below are from before that, and their video clips did not survive the restart.");
  }
  // Context that changes how every number below should be read: which arm is "bow", whether the
  // picture (and therefore the clips) are mirrored, and the actual capture resolution — geometry
  // fixes in this app have already been sensitive to exactly these settings (see CLAUDE.md).
  lines.push(`Camera ${cameraWidth ?? "?"}x${cameraHeight ?? "?"} · ${rightHanded ? "right-handed" : "left-handed"} · mirror ${mirrored ? "on" : "off"}`);
  if (modelStatusLine) lines.push(modelStatusLine);
  // Whether calibration exists at all, when it was last confirmed, and what today's check found —
  // see calibrationShareLine's own comment. Always present (never gated on a truthy check like
  // the other optional lines above/below) for exactly the reason the PM's review raised: its
  // absence must never be mistaken for "calibration is fine."
  lines.push(calibrationShareLine(storedCalibration, calibrationDone, calibrationDisagreement));

  lines.push("");
  // "Arrows" means confirmed full draw, same distinction as the on-screen count line — see
  // fullDrawShotCount's own comment. shotCount (every logged row, short draws included) is still
  // in here too, explicitly, rather than left for whoever reads this to infer from the per-draw
  // fullDraw=yes/no field below.
  lines.push(`Arrows this session: ${fullDrawShotCount}`);
  lines.push(`Logged draws this session (arrows + short-of-full-draw): ${shotCount}`);
  // The log only ever keeps the newest SHOT_LOG_MAX entries — without this line, a 30-arrow
  // session sharing only its last 10 draws would silently look like a complete 10-arrow session,
  // and threshold retuning done against it would be working from a biased sample without knowing.
  if (entries.length < shotCount) {
    lines.push(`NOTE: only the most recent ${entries.length} of ${shotCount} draws are included below — the log only keeps the last ${entries.length}.`);
  }
  lines.push(`Movements ignored (not real draws): ${rejectedAttemptCount}`);
  lines.push(`Arrows drawn before settling (not recorded): ${unsettledAttemptCount}`);
  lines.push(
    `Attention idle periods: ${attentionIdlePeriods}${attentionLateWakeCount ? ` (${attentionLateWakeCount} late wake${attentionLateWakeCount === 1 ? "" : "s"})` : ""}`
  );
  if (clipsUnavailableReason) lines.push(`Clip recording: ${clipsUnavailableReason}`);

  // Same consistency wording renderShotLog puts in the app itself (narrateMeasure) — reusing it
  // rather than re-deriving a second version keeps the shared text and the on-screen log always
  // in agreement. Built from fullDrawEntries only, same reasoning as renderShotLog: a short-draw
  // row's junk numbers must never dilute the one claim this app is allowed to make.
  lines.push("");
  const fullDrawEntries = entries.filter((e) => e.reachedFullDraw);
  const shortCount = entries.length - fullDrawEntries.length;
  lines.push("Consistency:");
  if (shortCount > 0) {
    lines.push(`  (based on the ${fullDrawEntries.length} of ${entries.length} shown draws that reached full draw — ${shortCount} excluded as short of full draw)`);
  }
  const measures = [
    narrateMeasure(fullDrawEntries, (e) => e.bowArmAngle, "Bow arm", wordForBowArm, BOW_ARM_CONSISTENCY_FLOOR_DEG),
    narrateMeasure(fullDrawEntries, (e) => e.shoulderDrop?.bow ?? null, "Bow shoulder", wordForShoulder, SHOULDER_BOW_CONSISTENCY_FLOOR_PCT),
    narrateMeasure(fullDrawEntries, (e) => e.shoulderDrop?.draw ?? null, "Draw shoulder", wordForShoulder, SHOULDER_DRAW_CONSISTENCY_FLOOR_PCT),
    narrateMeasure(fullDrawEntries, (e) => e.elbowAlign?.signed ?? null, "Draw elbow", wordForElbow, ELBOW_CONSISTENCY_FLOOR_DEG),
  ].filter(Boolean);
  if (measures.length === 0) lines.push("  Not enough shots yet for a consistency read.");
  else measures.forEach((m) => lines.push(`  ${m.text}`));

  // One draw per line, oldest first (entries itself is newest-first, same order the log/render
  // read it in) — chronological reads naturally top-to-bottom for a session review, and matches
  // shot numbering order.
  lines.push("");
  lines.push("Draws (oldest first):");
  const chronological = [...entries].sort((a, b) => a.shotNum - b.shotNum);
  if (chronological.length === 0) lines.push("  (none)");
  else chronological.forEach((e) => lines.push(shareLineForEntry(e)));

  const clipFailures = chronological.filter((e) => e.clipFailReason);
  if (clipFailures.length) {
    lines.push("");
    lines.push("Clip failures:");
    clipFailures.forEach((e) => lines.push(`  Shot ${e.shotNum}: ${e.clipFailReason}`));
  }

  return lines.join("\n");
}

// Formatters shared by shareLineForEntry below — each renders a measure's own null (uncertain,
// below MIN_VISIBILITY — see shoulderDropOf/bowArmAngleOf/drawElbowAlignmentOf) as the honest word
// "uncertain", never a fake 0/0%/pass, so a retuning session can't mistake "wasn't measured" for
// "measured as zero".
function shareDeg(v) { return v == null ? "uncertain" : `${Math.round(v)}deg`; }
function shareSignedDeg(v) { return v == null ? "uncertain" : `${v >= 0 ? "+" : ""}${Math.round(v)}deg`; }
function sharePct(v) { return v == null ? "uncertain" : `${Math.round(v)}%`; }
function sharePassFail(v) { return v == null ? "unknown" : v ? "pass" : "fail"; }

// One logged draw as one labelled line: shot number, all four form readouts, hand separation (the
// key figure the threshold retune needs — see CLAUDE.md/HANDOVER.md), the pass/fail of each of the
// four full-draw trigger conditions, whether it reached true full draw, and whether it has a clip.
// Always includes everything a ?debug row shows, regardless of whether ?debug is set — that gate
// only applies to the LIVE on-screen row (see renderShotRow's debugBit); it never applies here.
function shareLineForEntry(e) {
  const recorded = e.clipUrl ? "yes" : e.clipLostOnRestore ? "lost-on-restart" : "no";
  const failBit = e.clipFailReason ? ` clipFailReason="${e.clipFailReason}"` : "";
  return (
    `shot=${e.shotNum} bowArm=${shareDeg(e.bowArmAngle)} shoulderBow=${sharePct(e.shoulderDrop?.bow ?? null)} ` +
    `shoulderDraw=${sharePct(e.shoulderDrop?.draw ?? null)} elbow=${shareSignedDeg(e.elbowAlign?.signed ?? null)} ` +
    `handSep=${e.handSep == null ? "uncertain" : e.handSep.toFixed(3)} anchor=${sharePassFail(e.anchorOk)} arm=${sharePassFail(e.armOk)} ` +
    `sep=${sharePassFail(e.sepOk)} still=${sharePassFail(e.stillOk)} fullDraw=${e.reachedFullDraw ? "yes" : "no"} recorded=${recorded}${failBit}`
  );
}

// Gets buildShareText's string off the phone. Three steps, each only tried if the one before it
// is unavailable or actually failed — the owner must never tap Share and get nothing with no idea
// why. (1) The iOS share sheet — the natural route on his device: one tap, then AirDrop to his
// Mac, or into Notes/Messages. Unsupported on desktop Safari and some contexts, and it throws if
// not called from a genuine user gesture (which this always is — only ever called from the Share
// button's own click handler). (2) The clipboard. (3) A selectable on-screen text block he can
// copy by hand — the guaranteed-to-work last resort.
async function shareSessionText(text) {
  if (navigator.share) {
    try {
      await navigator.share({ title: "Archery form coach — session share", text });
      return;
    } catch (err) {
      // AbortError means the owner opened the share sheet and backed out himself — that's not a
      // failure to fall back from, he just changed his mind. Anything else (unsupported context,
      // not a real user gesture, a share target rejecting it) falls through to the clipboard.
      if (err && err.name === "AbortError") return;
    }
  }
  if (navigator.clipboard && navigator.clipboard.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      flashShareButton("Copied!");
      return;
    } catch (err) {
      // fall through to the manual copy block below
    }
  }
  showShareFallbackText(text);
}

// Briefly relabels the Share button to confirm the clipboard copy actually happened — the owner
// gets no OS-level confirmation for navigator.clipboard the way he does for the native share
// sheet, so without this a silent success and a silent failure would look identical to him.
function flashShareButton(label) {
  const original = shotLogShareBtn.textContent;
  shotLogShareBtn.textContent = label;
  setTimeout(() => { shotLogShareBtn.textContent = original; }, 1500);
}

// Last resort: neither the share sheet nor the clipboard worked. A plain, pre-selected, readonly
// textarea the owner can press-and-hold to copy by hand — guaranteed to work with no permissions
// or APIs at all, which is the whole point of it being the last step in the chain.
function showShareFallbackText(text) {
  shotLogShareTextEl.value = text;
  shotLogShareTextEl.classList.remove("hidden");
  shotLogShareTextEl.focus();
  shotLogShareTextEl.select();
}

// ===== SHOOTING CUES — HANDOVER.md Stage 2. He can't touch or read the phone while shooting, and
// lost freeze meant he had no way to tell mid-end whether the app was even working. Four states,
// exactly the four he asked for (see HANDOVER.md "Answered by the owner" — he named two things
// unprompted, then confirmed a third when asked directly, and ruled out a fifth): not seeing him,
// seeing him and calm, watching a live draw, and the one-shot outcome once a draw ends. No new
// detection logic here — every state is read straight off state this file already tracks for
// other reasons (tracking loss, the module-level `attempt`, and endAttempt's own verdict). Lives
// on #cue, a plain DOM element covering the whole #stage — deliberately NOT #camerabox: anchored
// to the camera box, the border lands off-screen entirely on any axis where the camera and screen
// aspect ratios agree and no letterbox gutter exists, silently hiding the most important state.
// See index.html/style.css. Never drawn on
// #overlay: #overlay is what canvas.captureStream() records into every clip (see
// startClipRecording), so anything painted there would be baked into the owner's saved footage
// without him ever finding out until he watched one back.
const CUE_OUTCOME_MS = 900; // how long the one-shot logged/rejected flash shows before handing back to whatever's actually true right now (see revertCueToCurrent) — matches the longer of the two flash animations in style.css (cue-flash-rejected) so the class is never swapped mid-animation
let cueOutcomeTimer = null; // non-null while a logged/rejected flash is playing; blocks updateCue from overwriting it early (see updateCue below)
let cueLastLost = true; // last known "not seeing you" state, kept up to date even while a flash is covering the display, so the flash can hand back to the truth once it ends rather than to a stale guess. Starts true: before the camera/pose model are ready, nobody has been seen yet — the honest state to show
let cueLastWatching = false; // same idea, for "a draw is currently open"
cueEl.className = "cue-lost"; // match the DOM to cueLastLost's own initial value from the very first paint — before the camera/pose model are ready there is genuinely nobody being seen yet, so this is accurate, not just a placeholder

// Sets #cue's class, with one wrinkle: cue-logged/cue-rejected are one-shot CSS animations, and
// assigning a className the DOM already has is a no-op — so two rejected draws close together
// (plausible: nocking, then lowering the bow, each briefly crossing the attempt-start floor) would
// only flash once. Clearing the class and forcing a layout first (void .offsetWidth) makes the
// browser treat the very next assignment as a fresh start every time. Cheap: this only runs once
// per draw attempt, never per frame.
function applyCueClass(cls) {
  if (cls === "cue-logged" || cls === "cue-rejected") {
    cueEl.className = "";
    void cueEl.offsetWidth;
  }
  cueEl.className = cls;
}

// Called once per rendered frame (renderLoop) with what's true RIGHT NOW: is tracking lost, and
// is a draw attempt currently open. Always remembers both (cueLastLost/cueLastWatching), even
// while an outcome flash is covering the display, so revertCueToCurrent below can hand back to
// the real current state rather than whatever was true when the flash started. If a flash IS
// playing, the display itself is left alone here — it gets to finish on its own timer, never cut
// short by the very next frame.
function updateCue(lost, watching) {
  cueLastLost = lost;
  cueLastWatching = watching;
  if (cueOutcomeTimer) return;
  applyCueClass(lost ? "cue-lost" : watching ? "cue-watching" : "cue-resting");
}

// Hands the display back to whichever persistent state (lost/watching/resting) is actually true
// right now, once an outcome flash has run its course. A separate, directly-callable function
// (rather than inlined in the setTimeout below) purely so selfTest can drive this exact hand-back
// logic deterministically, without waiting on a real timer.
function revertCueToCurrent() {
  cueOutcomeTimer = null;
  applyCueClass(cueLastLost ? "cue-lost" : cueLastWatching ? "cue-watching" : "cue-resting");
}

// Called from endAttempt the instant a draw's fate is decided — a confirmed arrow, or seen but
// not confirmed (the owner explicitly asked these be told apart, separately from "watching this
// draw" above). `confirmed` is NOT "was a row logged": a draw that got a row but never reached
// full draw (anchorOk/armOk/stillOk never all true) still gets the cue-rejected flash here, same
// as a draw that earned no row at all — green is a claim that an arrow was actually shot, and
// that claim needs to be true every time, not just usually, since the owner is five metres away
// and cannot check it. See endAttempt's own comment at its call site. A brief, self-clearing
// flash either way, never something to dismiss (see CLAUDE.md) — CUE_OUTCOME_MS later it hands
// back to whatever updateCue has most recently observed to be true, which may itself have
// changed WHILE the flash was playing (tracking lost the instant after a draw is rejected, say),
// so the cue always lands back on the truth, never a stale guess.
function signalOutcome(confirmed) {
  clearTimeout(cueOutcomeTimer);
  applyCueClass(confirmed ? "cue-logged" : "cue-rejected");
  cueOutcomeTimer = setTimeout(revertCueToCurrent, CUE_OUTCOME_MS);
}
// ===========================================================================

// Draws the current camera frame into the overlay canvas, unmirrored. Only used while a clip is
// actively recording (see paintCanvas below) — canvas.captureStream is the only way to bake the
// picture into a clip, so the canvas needs its own copy of the video frame for exactly that
// window, landmarks or not, so a clip is never missing frames just because the pose was briefly
// lost. canvas.width/height are set to the video's native resolution in startCamera, so this
// plain draw lines up exactly with no cropping or letterboxing needed. See withMirror below for
// where the flip actually happens; this function has no idea whether the current picture is
// mirrored or not, deliberately.
//
// This fully repaints canvas.width × canvas.height every call (clearRect then a drawImage that
// covers the same rectangle) — the same box <video> sits in underneath (inset: 0, 100% × 100%),
// so while this is drawing there is no seam at any edge for the unflipped video to leak through.
function drawVideoFrame() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
}

// The skeleton lines/dots only — no video frame. Pulled out on its own so paintCanvas can draw
// it either on top of a freshly-composited video frame (recording) or straight onto a transparent
// canvas (not recording, <video> shows through on its own) without duplicating the two
// drawingUtils calls in both places.
function drawSkeletonLines(landmarks) {
  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: "#00e5ff",
    lineWidth: 3,
  });
  drawingUtils.drawLandmarks(landmarks, { color: "#ffffff", radius: 4 });
}

// Runs one frame's worth of canvas drawing inside a horizontal flip, when effectiveMirror says
// this frame should be mirrored. Only used while a clip is recording (see paintCanvas) — the rest
// of the time mirroring is a CSS class instead (see syncMirrorClasses/style.css), because a CSS
// transform is invisible to canvas.captureStream: a clip is recorded straight off this canvas's
// own pixels, so whatever the owner sees mirrored on screen while recording has to be mirrored IN
// THE CANVAS, not just in CSS, or the clip he watches back won't match what he saw live.
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

// Pure: should the #overlay canvas element carry the CSS .mirrored class (see style.css) this
// frame? Only when the picture should appear mirrored AND nothing is currently baking a
// pixel-level mirror into the canvas itself (see withMirror) — stacking both would flip an
// already-mirrored recording's picture back to unmirrored on screen. Kept pure and tiny so
// selfTest can check the full truth table directly, the same convention effectiveMirror uses.
function canvasShouldMirrorViaCss(mirror, recording) {
  return mirror && !recording;
}

// Keeps #video and #overlay's CSS mirror state in sync with effectiveMirror (facingMode/
// mirrorToggled) and with whether a clip is currently recording (activeRecording) — see
// canvasShouldMirrorViaCss. Called every frame from paintCanvas below rather than from every
// place any of those three things can change: a missed call site there would show the wrong
// picture with nothing to catch it, where a classList.toggle a frame late costs nothing visible.
function syncMirrorClasses() {
  const mirror = effectiveMirror(facingMode, mirrorToggled);
  const recording = !!activeRecording;
  video.classList.toggle("mirrored", mirror);
  canvas.classList.toggle("mirrored", canvasShouldMirrorViaCss(mirror, recording));
}

// One frame's worth of drawing onto the overlay canvas — the single place renderLoop now goes to
// put anything on screen. landmarks is this frame's smoothed landmarks, or null when there's
// nothing to draw a skeleton from (pose lost, or an idle-throttled tick that skipped detection).
//
// FIELD BUG this replaced: the canvas used to be repainted with an opaque copy of the video frame
// every single call, which is the only thing that made <video> (sitting underneath, same box)
// invisible — so the picture the owner saw was a canvas repaint gated behind the ENTIRE rest of
// this function's caller (renderLoop), including the synchronous, blocking pose-detection call.
// The visible image could therefore never update faster than one inference pass per frame — at
// the owner's own measured ~27.8ms/frame, that is the ~28.6fps his own shot log already reported,
// and any slowdown in inference (a re-acquisition after losing the archer, a hot phone throttling
// itself) showed up directly as the PICTURE stuttering, not just the skeleton lagging behind it.
// Confirmed by measurement (see the PM's brief): with detection artificially delayed by a fixed
// amount, the old code's on-screen repaint rate tracked 1/delay almost exactly, in lockstep with
// the number of detection calls — not an assumption, a measured 1:1 relationship.
//
// The fix: while a clip is recording (see startClipRecording — canvas.captureStream is the only
// way to bake the skeleton into a clip), the canvas still needs the full picture, pixel-mirrored,
// exactly as this app has always recorded it. The rest of the time, <video> is what the owner
// actually sees — playing at the camera's own native frame rate, entirely independent of how fast
// pose detection is keeping up — and the canvas only ever holds the (transparent-background,
// unmirrored-in-pixels) skeleton lines on top of it; the CSS .mirrored class (syncMirrorClasses)
// flips both elements together so they stay pixel-registered with each other either way.
function paintCanvas(landmarks) {
  syncMirrorClasses();
  if (activeRecording) {
    withMirror(() => {
      drawVideoFrame();
      if (landmarks) drawSkeletonLines(landmarks);
    });
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (landmarks) drawSkeletonLines(landmarks);
  }
}

// Builds the ?debug panel's entire DOM ONCE, at startup — see DEBUG_OVERLAY_REFRESH_MS's own
// comment for why rebuilding this via innerHTML every frame was a real, measured stutter bug.
// After this call, syncDebugOverlay only ever touches textContent/className/classList on the
// nodes captured into dbgRefs here — strictly less DOM work per refresh than the panel this
// replaced, which rebuilt via innerHTML on every throttled tick.
//
// Four sections, in the order the owner asked for: FULL DRAW (the four gates plus the summary),
// TRIGGERS/STATE (what the app is doing right now, as an annunciator-style lamp grid per the
// owner's own follow-up brief — dark-and-outlined when off, solid green when lit), MEASURES (the
// same numbers #readouts shows), SESSION (running totals plus tracking health). One static
// template literal, not per-field createElement calls — cheap to build once, and every value node
// carries a data-x attribute this function uses to populate dbgRefs right afterwards.
function buildDebugPanel() {
  debugEl.innerHTML = `
    <section class="dbg-section">
      <h2 class="dbg-heading">Full draw</h2>
      <div class="dbg-reason" data-x="reason"></div>
      <div class="dbg-row">
        <span class="dbg-name">Anchor dist</span>
        <span class="dbg-val" data-x="g-anchor-val">—</span>
        <span class="dbg-thresh" data-x="g-anchor-thresh"></span>
        <span class="dbg-lamp" data-x="g-anchor-lamp">ANCHOR</span>
      </div>
      <div class="dbg-row">
        <span class="dbg-name">Bow-arm angle</span>
        <span class="dbg-val" data-x="g-arm-val">—</span>
        <span class="dbg-thresh" data-x="g-arm-thresh"></span>
        <span class="dbg-lamp" data-x="g-arm-lamp">ARM</span>
      </div>
      <div class="dbg-row">
        <span class="dbg-name">Hand separation</span>
        <span class="dbg-val" data-x="g-sep-val">—</span>
        <span class="dbg-thresh" data-x="g-sep-thresh"></span>
        <span class="dbg-lamp" data-x="g-sep-lamp">SEP</span>
      </div>
      <div class="dbg-row">
        <span class="dbg-name">Stillness (speed)</span>
        <span class="dbg-val" data-x="g-still-val">—</span>
        <span class="dbg-thresh" data-x="g-still-thresh"></span>
        <span class="dbg-lamp" data-x="g-still-lamp">STILL</span>
      </div>
      <div class="dbg-summary">
        <span class="dbg-name">AT FULL DRAW</span>
        <span class="dbg-lamp" data-x="g-fulldraw-lamp" style="min-width:7em">FULL DRAW</span>
      </div>
    </section>

    <section class="dbg-section">
      <h2 class="dbg-heading">Triggers / state</h2>
      <div class="dbg-lamps">
        <div class="dbg-lampcell"><span class="dbg-lamp" data-x="t-attention-lamp">ATTN</span><div class="dbg-lampcaption">attention engaged</div></div>
        <div class="dbg-lampcell"><span class="dbg-lamp" data-x="t-raise-lamp">RAISE</span><div class="dbg-lampcaption">raise armed</div></div>
        <div class="dbg-lampcell"><span class="dbg-lamp" data-x="t-attempt-lamp">OPEN</span><div class="dbg-lampcaption">attempt open</div></div>
        <div class="dbg-lampcell"><span class="dbg-lamp" data-x="t-cropstable-lamp">STABLE</span><div class="dbg-lampcaption">crop box stable</div></div>
        <div class="dbg-lampcell"><span class="dbg-lamp" data-x="t-pose-lamp">POSE</span><div class="dbg-lampcaption">pose seen</div></div>
      </div>
      <div class="dbg-lamps">
        <div class="dbg-lampcell"><span class="dbg-lamp dbg-lamp-event" data-x="t-ev-attempt">START</span><div class="dbg-lampcaption">attempt started</div></div>
        <div class="dbg-lampcell"><span class="dbg-lamp dbg-lamp-event" data-x="t-ev-raise">FIRED</span><div class="dbg-lampcaption">raise fired</div></div>
        <div class="dbg-lampcell"><span class="dbg-lamp dbg-lamp-event" data-x="t-ev-eligible">ELIGIBLE</span><div class="dbg-lampcaption">frame eligible</div></div>
        <div class="dbg-lampcell"><span class="dbg-lamp dbg-lamp-event" data-x="t-ev-shot">LOGGED</span><div class="dbg-lampcaption">shot logged</div></div>
      </div>
      <div class="dbg-row">
        <span class="dbg-name">Raise height</span>
        <span class="dbg-val" data-x="t-raise-val">—</span>
        <span class="dbg-thresh" data-x="t-raise-thresh"></span>
        <span></span>
      </div>
      <div class="dbg-row dbg-row-2col">
        <span class="dbg-name">Attempt</span>
        <span class="dbg-val" data-x="t-attempt-detail">none</span>
      </div>
      <div class="dbg-row dbg-row-2col">
        <span class="dbg-name">Settling</span>
        <span class="dbg-val" data-x="t-settle-detail">—</span>
      </div>
      <div class="dbg-row">
        <span class="dbg-name">Hand sep vs attempt floor</span>
        <span class="dbg-val" data-x="t-boundary-val">—</span>
        <span class="dbg-thresh" data-x="t-boundary-thresh"></span>
        <span class="dbg-pf" data-x="t-boundary-pf"></span>
      </div>
    </section>

    <section class="dbg-section">
      <h2 class="dbg-heading">Measures</h2>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Bow-arm line</span><span class="dbg-val" data-x="m-bowarm">—</span></div>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Shoulder drop (bow)</span><span class="dbg-val" data-x="m-shoulder-bow">—</span></div>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Shoulder drop (draw)</span><span class="dbg-val" data-x="m-shoulder-draw">—</span></div>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Elbow ↔ arrow line</span><span class="dbg-val" data-x="m-elbow">—</span></div>
    </section>

    <section class="dbg-section">
      <h2 class="dbg-heading">Session</h2>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Arrows</span><span class="dbg-val" data-x="s-arrows">0</span></div>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Rows logged</span><span class="dbg-val" data-x="s-rows">0</span></div>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Movements ignored</span><span class="dbg-val" data-x="s-ignored">0</span></div>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Unsettled attempts</span><span class="dbg-val" data-x="s-unsettled">0</span></div>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Attention idle periods</span><span class="dbg-val" data-x="s-idle">0</span></div>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Calibration</span><span class="dbg-val" data-x="s-calibration">—</span></div>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Pose</span><span class="dbg-val" data-x="s-pose">—</span></div>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Torso scale</span><span class="dbg-val" data-x="s-scale">—</span></div>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">ROI cropping</span><span class="dbg-val" data-x="s-roi">—</span></div>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Model</span><span class="dbg-val" data-x="s-model">measuring…</span></div>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Last inference</span><span class="dbg-val" data-x="s-inference">—</span></div>
      <div class="dbg-row dbg-row-2col"><span class="dbg-name">Rendered fps</span><span class="dbg-val" data-x="s-fps">—</span></div>
    </section>
  `;
  dbgRefs = {};
  debugEl.querySelectorAll("[data-x]").forEach((el) => {
    dbgRefs[el.dataset.x] = el;
  });
}

// ?debug panel's per-refresh update. No-op (not even a DOM lookup) when ?debug isn't in the URL.
// Throttled to DEBUG_OVERLAY_REFRESH_MS exactly as before — only TEXT/CLASS updates happen on the
// nodes buildDebugPanel already created, never a DOM rebuild (see that function's own comment).
//
// `landmarks` has three distinct meanings, not two — this is the mechanism behind the panel's
// "never go blank, and never lie about being fresh" behaviour during an idle gap:
//   - a real array: a pose was found this frame — MEASURES/torso-scale are recomputed fresh.
//   - null: detection genuinely ran this frame and found nobody — MEASURES show "uncertain" and
//     lastPoseSeen goes false (set by the caller, not here).
//   - undefined (the argument simply omitted): this call came from an attention-gating idle tick
//     that skipped detection entirely — MEASURES/torso-scale are left exactly as last drawn,
//     since there is no fresher truth to show and redrawing "uncertain" would be a false claim
//     that tracking was lost, not merely un-sampled this instant.
// The FULL DRAW section and most of TRIGGERS/STATE don't need this distinction at all: they read
// straight from debugInfo/module state, which idle ticks simply never touch, so they already hold
// their last real value through an idle gap for free.
function syncDebugOverlay(nowMs, landmarks, frameWidth, frameHeight) {
  if (!DEBUG) return;
  if (nowMs - lastDebugRenderMs < DEBUG_OVERLAY_REFRESH_MS) return;
  lastDebugRenderMs = nowMs;
  const r = dbgRefs;
  const d = debugInfo; // always a real object — see emptyDebugInfo's own comment

  const fmt = (v, digits = 2) => (v == null ? "—" : v.toFixed(digits));
  const fmtDeg = (v) => (v == null ? "—" : `${Math.round(v)}°`);
  const setLamp = (el, lit) => el.classList.toggle("lit", !!lit);
  const okClass = (ok) => (ok == null ? "uncertain" : ok ? "ok" : "warn");

  // ----- FULL DRAW: the never-blank reason line, then the four gates plus their summary lamp. -----
  r["reason"].textContent = d.reason ?? "reading normally — full draw check completed this frame";
  r["reason"].classList.toggle("dbg-reason-bail", !!d.reason);
  r["reason"].classList.toggle("dbg-reason-ok", !d.reason);

  const setGate = (key, value, ok, threshText, fmtFn) => {
    r[`${key}-val`].textContent = fmtFn(value);
    r[`${key}-val`].className = `dbg-val ${okClass(ok)}`;
    r[`${key}-thresh`].textContent = threshText;
    setLamp(r[`${key}-lamp`], ok === true);
  };
  setGate("g-anchor", d.anchorDist, d.anchorOk, `≤ ${FULL_DRAW_ANCHOR_MAX}`, fmt);
  setGate("g-arm", d.bowArmAngle, d.armOk, `≥ ${FULL_DRAW_BOW_ARM_MIN}°`, fmtDeg);
  setGate("g-sep", d.handSep, d.sepOk, `≥ ${FULL_DRAW_HAND_SEP_MIN}`, fmt);
  setGate("g-still", d.speed, d.stillOk, `≤ ${FULL_DRAW_STILL_MAX}`, fmt);
  setLamp(r["g-fulldraw-lamp"], d.anchorOk === true && d.armOk === true && d.sepOk === true && d.stillOk === true);

  // ----- TRIGGERS / STATE -----
  setLamp(r["t-attention-lamp"], attentionEngaged);
  setLamp(r["t-raise-lamp"], raiseArmed);
  setLamp(r["t-attempt-lamp"], attempt !== null);
  setLamp(r["t-cropstable-lamp"], lastCropBoxStable);
  setLamp(r["t-pose-lamp"], lastPoseSeen);

  setLamp(r["t-ev-attempt"], isDebugEventLit(debugEvents.attemptStarted, nowMs));
  setLamp(r["t-ev-raise"], isDebugEventLit(debugEvents.raiseFired, nowMs));
  setLamp(r["t-ev-eligible"], isDebugEventLit(debugEvents.frameEligible, nowMs));
  setLamp(r["t-ev-shot"], isDebugEventLit(debugEvents.shotLogged, nowMs));

  r["t-raise-val"].textContent = fmt(debugRaiseHeight);
  r["t-raise-thresh"].textContent = `≥ ${RAISE_TRIGGER_UP_FRACTION} up / ≤ ${RAISE_TRIGGER_DOWN_FRACTION} down`;

  if (attempt) {
    const drawing = attempt.startMs !== null;
    const elapsedMs = nowMs - (drawing ? attempt.startMs : attempt.watchStartedAt);
    r["t-attempt-detail"].textContent =
      `open ${(elapsedMs / 1000).toFixed(1)}s (${drawing ? "drawing" : "raise phase"}) — peak sep ${fmt(attempt.peakHandSep)} — real-draw clock ${drawing ? "started" : "not started yet"}`;
  } else {
    r["t-attempt-detail"].textContent = "none";
  }

  r["t-settle-detail"].textContent =
    `${settledFrames}/${SETTLE_FRAMES_REQUIRED} frames` +
    (ROI_CROPPING_ENABLED ? `, crop box ${lastCropBoxStable ? "stable" : "not stable"}` : "");

  const boundaryOk = d.handSep == null ? null : d.handSep >= DRAW_ATTEMPT_MIN_SEP;
  r["t-boundary-val"].textContent = fmt(d.handSep);
  r["t-boundary-thresh"].textContent = `≥ ${DRAW_ATTEMPT_MIN_SEP}`;
  r["t-boundary-pf"].textContent = boundaryOk == null ? "—" : boundaryOk ? "PASS" : "FAIL";
  r["t-boundary-pf"].className = `dbg-pf ${okClass(boundaryOk)}`;

  // ----- MEASURES + torso scale — only when THIS call actually carries a fresh sample (see this
  // function's own comment on the three meanings of `landmarks`). An idle tick (landmarks
  // undefined) leaves these exactly as last drawn rather than redrawing a false "uncertain". -----
  if (landmarks !== undefined) {
    const setMeasure = (key, value, ok, fmtFn) => {
      r[key].textContent = value == null ? "— uncertain" : fmtFn(value);
      r[key].className = `dbg-val ${okClass(value == null ? null : ok)}`;
    };
    const bowArmAngle = landmarks ? bowArmAngleOf(landmarks, frameWidth, frameHeight) : null;
    setMeasure("m-bowarm", bowArmAngle, bowArmAngle != null && bowArmAngle >= BOW_ARM_ANGLE_MIN && bowArmAngle <= BOW_ARM_ANGLE_MAX, fmtDeg);
    const shoulders = landmarks ? shoulderDropSampleOf(landmarks, frameWidth, frameHeight) : { bow: null, draw: null };
    setMeasure("m-shoulder-bow", shoulders.bow, shoulders.bow != null && shoulders.bow >= SHOULDER_DROP_MIN_PCT, (v) => `${Math.round(v)}%`);
    setMeasure("m-shoulder-draw", shoulders.draw, shoulders.draw != null && shoulders.draw >= SHOULDER_DROP_MIN_PCT, (v) => `${Math.round(v)}%`);
    const elbow = landmarks ? drawElbowAlignmentOf(landmarks, frameWidth, frameHeight) : null;
    setMeasure(
      "m-elbow",
      elbow,
      elbow != null && elbow.deviation <= DRAW_ELBOW_ALIGN_MAX_DEVIATION,
      (v) => (Math.round(v.deviation) === 0 ? "in line" : `${Math.round(v.deviation)}° ${v.direction}`)
    );

    const bowShoulder = rightHanded ? L_SHOULDER : R_SHOULDER;
    const bowHip = rightHanded ? L_HIP : R_HIP;
    const drawShoulder = rightHanded ? R_SHOULDER : L_SHOULDER;
    const drawHip = rightHanded ? R_HIP : L_HIP;
    const torsoScale = landmarks
      ? torsoLength(landmarks, drawShoulder, drawHip, frameWidth, frameHeight) ?? torsoLength(landmarks, bowShoulder, bowHip, frameWidth, frameHeight)
      : null;
    r["s-scale"].textContent = torsoScale == null ? "—" : `${torsoScale.toFixed(1)}px`;
  }

  // ----- SESSION — running totals and pure module state; always safe to refresh regardless of
  // whether this call carries a fresh landmarks sample. -----
  r["s-arrows"].textContent = String(fullDrawShotCount);
  r["s-rows"].textContent = String(shotCount);
  r["s-ignored"].textContent = String(rejectedAttemptCount);
  r["s-unsettled"].textContent = String(unsettledAttemptCount);
  r["s-idle"].textContent = String(attentionIdlePeriods);
  // CALIBRATION (HANDOVER.md Stage 4) — the debug panel's own view of the same state
  // renderCalibrationStatus/renderShotLog already surface elsewhere (see #calibration-status and
  // the shot log's calibrationBit), reusing calibrationStatusText rather than re-deriving the
  // wording a second time. Unlike those two, this panel is a diagnostic instrument that never
  // goes quiet on purpose (see buildDebugPanel's own comment) — so the one state they render as
  // silence (calibrationDone && no disagreement) gets an explicit "agrees with stored calibration"
  // line here instead, matching the never-blank convention the rest of this panel already follows.
  const calibText = calibrationStatusText(calibrationDone, calibrationStatusLine);
  const calibParts = [calibText?.text, framingStatusLine, framingChangeStatusLine].filter(Boolean);
  r["s-calibration"].textContent = calibParts.length ? calibParts.join(" ") : "agrees with stored calibration";
  r["s-calibration"].className = `dbg-val ${
    calibText?.tone === "warn" || framingStatusLine || framingChangeStatusLine ? "warn" : calibText?.tone === "neutral" ? "uncertain" : "ok"
  }`;
  r["s-pose"].textContent = lastPoseSeen ? "seen" : "not seen";
  r["s-roi"].textContent = !ROI_CROPPING_ENABLED ? "disabled" : currentCropBox ? "active" : "no box yet";
  r["s-model"].textContent = modelStatusLine ?? "measuring…";
  r["s-inference"].textContent = lastInferenceMs == null ? "—" : `${lastInferenceMs.toFixed(1)}ms`;
  r["s-fps"].textContent = debugInstantFps == null ? "—" : debugInstantFps.toFixed(1);
}

// ===== TRIGGER TEST — ?triggertest live single-trigger inspector. Built after the owner used
// ?debug for a real session and asked for something different: "make it similar to debug except
// there is only one trigger on the side and there's an explanation of what it checks for and the
// values that it's measuring so i understand what it's expecting of me and i can monitor if it's
// getting the correct input." This is a different PRESENTATION of the same real state ?debug's
// TRIGGERS/STATE section already shows — one trigger at a time, in depth, with the plain-language
// explanation and the raw landmark inputs ?debug never had room for — not a second measurement
// pipeline. He cycles through the ten screens with the one big button below (or Space/→/← on a
// desktop), standing back far enough for the camera to see his whole body to read a trigger, then
// walking up to switch — see buildTriggerTestPanel's button for why it's one big control, not a
// row of small ones: every tap costs him a walk, so it has to be unmissable from a few metres.
//
// READ-ONLY, same discipline as ?debug: every number on this panel is read straight off state the
// real detection pipeline already computed this frame (debugInfo, raiseArmed, attempt,
// settledFrames, lastCropBoxStable, lastPoseSeen, attentionEngaged, debugAttnCalm/HandSep/Speed —
// see those variables' own declarations for the handful of places TRIGGERTEST had to join DEBUG
// to populate them, and the ATTN block in renderLoop for why attentionIsClearlyCalm could just be
// called a second time instead). Nothing here computes a NEW measurement or reimplements any
// geometry — the four full-draw gates' own numbers (anchorDist/handSep/bowArmAngle/speed) come
// straight out of isAtFullDraw's own debugInfo object, not a second calculation of that formula.
//
// Do NOT tune any of the text below to make a trigger look like it's behaving well. SEP in
// particular is proven (see CLAUDE.md) to light up on an ordinary resting stance — the note on
// that screen says so plainly, and stays even if that's uncomfortable to read while testing.

// Landmark role labels, keyed by CURRENT handedness — rightHanded can change any time via the 🎯
// button, so this is computed fresh on every call rather than once, exactly like isAtFullDraw's
// own role lookups.
function triggerTestRoleLabels() {
  return {
    bowShoulder: rightHanded ? L_SHOULDER : R_SHOULDER,
    drawShoulder: rightHanded ? R_SHOULDER : L_SHOULDER,
    bowElbow: rightHanded ? L_ELBOW : R_ELBOW,
    drawElbow: rightHanded ? R_ELBOW : L_ELBOW,
    bowWrist: rightHanded ? L_WRIST : R_WRIST,
    drawWrist: rightHanded ? R_WRIST : L_WRIST,
    bowHip: rightHanded ? L_HIP : R_HIP,
    drawHip: rightHanded ? R_HIP : L_HIP,
  };
}

// One row describing a single landmark's live visibility/confidence — the raw input the owner
// asked for, so he can tell "this trigger is reading bad input" apart from "this trigger is
// reading good input and firing/not firing correctly". `landmarks` may be a real array, null (a
// frame where detection ran and found nobody), or undefined (an idle-throttle tick that never ran
// detection at all — see syncDebugOverlay's own comment on that three-way distinction, which this
// mirrors); all three safely fall through to "not seen this frame" here. Smoothing never touches
// `visibility` (see CLAUDE.md), so reading it off the smoothed landmarks this panel already has is
// exactly the same number MediaPipe itself reported — not a second, different confidence figure.
function ttInputRow(landmarks, idx, label) {
  const lm = landmarks ? landmarks[idx] : null;
  return {
    label,
    seen: !!lm,
    confidence: lm ? lm.visibility ?? 0 : null,
    ok: !!landmarks && visible(landmarks, idx),
  };
}

const ttFmt = (v, digits = 2) => (v == null ? "—" : v.toFixed(digits));
const ttFmtDeg = (v) => (v == null ? "—" : `${Math.round(v)}°`);

// ===== TRIGGER TEST OVERLAY DRAWING — the acceptance region for the CURRENTLY SELECTED trigger,
// drawn straight onto the camera picture. Owner, after using the numeric panel above: "i'd like
// to see the cones and the boxes and the other artifacts like these" — a number says a check
// failed, a drawn region shows WHY (a cone pointing somewhere daft, an anchor region sitting off
// his face), which a number alone cannot.
//
// Colour is the only language this uses: green (TT_OVERLAY_PASS/_FILL) when that specific
// sub-condition is currently met, a neutral off-white (TT_OVERLAY_NEUTRAL/_FILL) when it isn't or
// can't be told — never red, and never a third colour, so "green" keeps meaning one single thing
// everywhere on this overlay. Thin strokes, translucent fills, on purpose — the owner needs to see
// his own body through the shape, not have it painted over.
//
// Each TRIGGER_DEFS entry's own `draw(landmarks, frameWidth, frameHeight)` (added below, only on
// screens with real geometry to show — see the per-entry comments) reads pixel-space points
// straight off debugInfo where isAtFullDraw already computed them (anchorPx, drawWristPx,
// bowWristPx, bowShoulderPx, anchorEarPx, scale) rather than recomputing any geometry a second
// time, same "read real state, never a second calculation" discipline as read() above — the
// numbers on the panel and the shape on screen must never be able to disagree. RAISE is the one
// exception (see its own draw() below) since bowArmRaiseHeight's visibility requirements are
// looser than isAtFullDraw's and debugInfo may not have populated its pixel fields at all while a
// raise is being demonstrated on its own.
//
// Screens with no sensible geometry (FULLDRAW, OPEN, ELIGIBLE, POSE, ATTN) simply have no `draw`
// — drawTriggerTestOverlay no-ops for those rather than inventing a decorative shape.
const TT_OVERLAY_PASS = "rgba(0, 230, 118, 0.95)";
const TT_OVERLAY_PASS_FILL = "rgba(0, 230, 118, 0.16)";
const TT_OVERLAY_NEUTRAL = "rgba(255, 255, 255, 0.75)";
const TT_OVERLAY_NEUTRAL_FILL = "rgba(255, 255, 255, 0.08)";
const ttStrokeColor = (ok) => (ok === true ? TT_OVERLAY_PASS : TT_OVERLAY_NEUTRAL);
const ttFillColor = (ok) => (ok === true ? TT_OVERLAY_PASS_FILL : TT_OVERLAY_NEUTRAL_FILL);

function drawTtCircle(center, radius, ok) {
  ctx.beginPath();
  ctx.arc(center.x, center.y, Math.max(radius, 1), 0, Math.PI * 2);
  ctx.fillStyle = ttFillColor(ok);
  ctx.fill();
  ctx.strokeStyle = ttStrokeColor(ok);
  ctx.stroke();
}

function drawTtLine(a, b, ok) {
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.strokeStyle = ttStrokeColor(ok);
  ctx.stroke();
}
// ===========================================================================

// One entry per screen, in cycle order. `read(landmarks, frameWidth, frameHeight)` returns
// everything renderCurrentTrigger needs to paint THIS trigger, computed from real, live module
// state — never a new calculation. `sentence`/`doThis`/`note` are static, plain-language text
// written for a non-coder standing in front of the camera, not for a developer. `draw`, where
// present, paints that same trigger's acceptance region onto the camera canvas — see the TRIGGER
// TEST OVERLAY DRAWING block just above for the shared conventions.
const TRIGGER_DEFS = [
  {
    id: "anchor",
    label: "ANCHOR",
    sentence: "Is your string hand right up at your anchor point — near your mouth or jaw, at or below that level, and behind it toward your draw side — the way it sits at a real full draw?",
    doThis: "Draw back to your normal anchor position to light this. Move your draw hand away from your face — resting position, reaching for an arrow, scratching your nose — to make it go dark. The three rows below split this into distance, height and front/back, so you can see exactly which one is holding it back if the lamp won't light.",
    note: null,
    read(landmarks) {
      const { bowShoulder, bowElbow, bowWrist, drawShoulder, drawHip, bowHip, drawWrist } = triggerTestRoleLabels();
      const d = debugInfo;
      return {
        lamp: d.anchorOk === true,
        reason: d.reason,
        rows: [
          { label: "Distance from anchor", value: `${ttFmt(d.anchorDist)} × torso`, threshold: `needs ≤ ${FULL_DRAW_ANCHOR_MAX}`, ok: d.anchorDist == null ? null : d.anchorDist <= FULL_DRAW_ANCHOR_MAX },
          { label: "Height (below mouth level)", value: `${ttFmt(d.anchorVerticalOffset)} × torso`, threshold: `needs ≥ ${(-FULL_DRAW_ANCHOR_ABOVE_MAX).toFixed(2)} (positive = below)`, ok: d.anchorVerticalOk },
          { label: "Behind the mouth (toward draw ear)", value: `${ttFmt(d.anchorBackward)} × torso`, threshold: `needs ≥ ${FULL_DRAW_ANCHOR_BACKWARD_MIN} — "—" means neither ear is visible enough to tell`, ok: d.anchorBackwardOk },
        ],
        inputs: [
          ttInputRow(landmarks, drawWrist, "draw wrist"),
          ttInputRow(landmarks, MOUTH_L, "mouth (one side)"),
          ttInputRow(landmarks, MOUTH_R, "mouth (other side)"),
          ttInputRow(landmarks, NOSE, "nose — backup anchor point if the mouth isn't visible"),
          ttInputRow(landmarks, rightHanded ? R_EAR : L_EAR, "draw-side ear — defines \"backwards\", preferred"),
          ttInputRow(landmarks, rightHanded ? L_EAR : R_EAR, "bow-side ear — backup for \"backwards\" if the draw-side ear is occluded"),
          ttInputRow(landmarks, drawShoulder, "draw shoulder — for the torso-length scale"),
          ttInputRow(landmarks, drawHip, "draw hip — for the torso-length scale"),
          ttInputRow(landmarks, bowShoulder, "bow shoulder — backup scale, and required before any of the four full-draw checks can run at all"),
          ttInputRow(landmarks, bowHip, "bow hip — backup scale"),
          ttInputRow(landmarks, bowElbow, "bow elbow — not used by THIS check directly, but required before any of the four full-draw checks can run at all"),
          ttInputRow(landmarks, bowWrist, "bow wrist — same as above, a shared prerequisite"),
        ],
      };
    },
    // The acceptance region as three shapes, each coloured by its OWN sub-check (not the combined
    // anchorOk) so the owner can see spatially which one is failing, matching the three rows
    // above: a circle at the distance limit, a horizontal line at the highest the wrist may sit
    // (the reported "above the nose" boundary), and — only when an ear was confidently visible
    // enough to compute it (see isAtFullDraw's own comment) — a line perpendicular to the
    // anchor→ear axis marking the backward cutoff. Plus the actual anchor→wrist line, so he can
    // see the distance he's being judged on, not just read it as a number.
    draw() {
      const d = debugInfo;
      if (!d.anchorPx || !d.scale) return;
      drawTtCircle(d.anchorPx, FULL_DRAW_ANCHOR_MAX * d.scale, d.anchorDist == null ? null : d.anchorDist <= FULL_DRAW_ANCHOR_MAX);

      const cutoffY = d.anchorPx.y - FULL_DRAW_ANCHOR_ABOVE_MAX * d.scale;
      const halfW = d.scale * 0.6;
      drawTtLine({ x: d.anchorPx.x - halfW, y: cutoffY }, { x: d.anchorPx.x + halfW, y: cutoffY }, d.anchorVerticalOk);

      if (d.anchorEarPx) {
        const axisX = d.anchorEarPx.x - d.anchorPx.x, axisY = d.anchorEarPx.y - d.anchorPx.y;
        const axisLen = Math.hypot(axisX, axisY);
        if (axisLen > 0) {
          const ux = axisX / axisLen, uy = axisY / axisLen;
          const px = -uy, py = ux; // perpendicular unit vector
          const originX = d.anchorPx.x + ux * FULL_DRAW_ANCHOR_BACKWARD_MIN * d.scale;
          const originY = d.anchorPx.y + uy * FULL_DRAW_ANCHOR_BACKWARD_MIN * d.scale;
          const halfLen = d.scale * 0.5;
          drawTtLine(
            { x: originX - px * halfLen, y: originY - py * halfLen },
            { x: originX + px * halfLen, y: originY + py * halfLen },
            d.anchorBackwardOk
          );
        }
      }

      if (d.drawWristPx) drawTtLine(d.anchorPx, d.drawWristPx, d.anchorOk);
    },
  },
  {
    id: "arm",
    label: "ARM",
    sentence: "Is your bow arm — the one holding the bow — straight, AND pointed out roughly level from your shoulder rather than hanging or raised?",
    doThis: "Extend your bow arm fully straight toward the target, roughly level with your shoulder, to light this. Bend your bow elbow, or drop the whole arm to your side, to make it go dark — the two rows below show straightness and direction separately, so you can see which one is failing.",
    note: null,
    read(landmarks) {
      const { bowShoulder, bowElbow, bowWrist, drawWrist } = triggerTestRoleLabels();
      const d = debugInfo;
      return {
        lamp: d.armOk === true,
        reason: d.reason,
        rows: [
          { label: "Bow-arm angle (straightness)", value: ttFmtDeg(d.bowArmAngle), threshold: `needs ≥ ${FULL_DRAW_BOW_ARM_MIN}°`, ok: d.armStraightOk },
          { label: "Elevation off horizontal (the cone)", value: ttFmtDeg(d.armElevation), threshold: `needs within ±${FULL_DRAW_ARM_CONE_APERTURE_DEG}° of level`, ok: d.armConeOk },
        ],
        inputs: [
          ttInputRow(landmarks, bowShoulder, "bow shoulder"),
          ttInputRow(landmarks, bowElbow, "bow elbow"),
          ttInputRow(landmarks, bowWrist, "bow wrist"),
          ttInputRow(landmarks, drawWrist, "draw wrist — not used by THIS check directly, but required before any of the four full-draw checks can run at all"),
        ],
      };
    },
    // The cone as a filled wedge from the bow shoulder (two rays at ±the aperture from
    // horizontal, opening toward whichever side the wrist is actually on), coloured by armConeOk
    // alone — then the real shoulder→wrist ray drawn on top, thicker, coloured by the COMBINED
    // armOk (straight AND in the cone), so a straight arm pointing the wrong way is visibly
    // "outside the wedge" even though the ray itself isn't bent.
    draw() {
      const d = debugInfo;
      if (!d.bowShoulderPx || !d.bowWristPx) return;
      const shoulder = d.bowShoulderPx, wrist = d.bowWristPx;
      const armLen = Math.hypot(wrist.x - shoulder.x, wrist.y - shoulder.y);
      const coneLen = armLen > 0 ? armLen * 1.2 : (d.scale || 80);
      const sign = wrist.x >= shoulder.x ? 1 : -1;
      const apertureRad = (FULL_DRAW_ARM_CONE_APERTURE_DEG * Math.PI) / 180;
      const dxCone = sign * coneLen * Math.cos(apertureRad);
      const topEnd = { x: shoulder.x + dxCone, y: shoulder.y - coneLen * Math.sin(apertureRad) };
      const bottomEnd = { x: shoulder.x + dxCone, y: shoulder.y + coneLen * Math.sin(apertureRad) };

      ctx.beginPath();
      ctx.moveTo(shoulder.x, shoulder.y);
      ctx.lineTo(topEnd.x, topEnd.y);
      ctx.lineTo(bottomEnd.x, bottomEnd.y);
      ctx.closePath();
      ctx.fillStyle = ttFillColor(d.armConeOk);
      ctx.fill();
      ctx.strokeStyle = ttStrokeColor(d.armConeOk);
      ctx.stroke();

      ctx.lineWidth = 4;
      drawTtLine(shoulder, wrist, d.armOk);
      ctx.lineWidth = 2;
    },
  },
  {
    id: "sep",
    label: "SEP",
    sentence: "Are your two hands far enough apart to look like a real full draw — roughly three-quarters of your own torso length or more?",
    doThis: "Draw the bow all the way back to light this. Bring your hands back together to make it go dark.",
    note: "Known issue, proven against a real session (see CLAUDE.md): standing normally with your arms relaxed at your sides is often enough on its own to light this up — your shoulders alone are usually wider than this threshold. If this stays lit while you're just standing there doing nothing, that's a known limitation of this check, not something wrong with your stance — it's one of the reasons this inspector exists.",
    read(landmarks) {
      const { bowWrist, drawWrist, bowShoulder, bowElbow } = triggerTestRoleLabels();
      const d = debugInfo;
      return {
        lamp: d.sepOk === true,
        reason: d.reason,
        rows: [{ label: "Hand separation", value: `${ttFmt(d.handSep)} × torso`, threshold: `needs ≥ ${FULL_DRAW_HAND_SEP_MIN}`, ok: d.sepOk }],
        inputs: [
          ttInputRow(landmarks, bowWrist, "bow wrist"),
          ttInputRow(landmarks, drawWrist, "draw wrist"),
          ttInputRow(landmarks, bowShoulder, "bow shoulder — required before any of the four full-draw checks can run at all"),
          ttInputRow(landmarks, bowElbow, "bow elbow — same as above"),
        ],
      };
    },
    // The wrist-to-wrist line, coloured by sepOk, with a perpendicular tick mark showing where
    // FULL_DRAW_HAND_SEP_MIN falls along it (measured from the bow wrist) — so the owner can see
    // how far short of (or past) the required separation his hands actually are, not just read it.
    draw() {
      const d = debugInfo;
      if (!d.bowWristPx || !d.drawWristPx || !d.scale) return;
      drawTtLine(d.bowWristPx, d.drawWristPx, d.sepOk);
      const dx = d.drawWristPx.x - d.bowWristPx.x, dy = d.drawWristPx.y - d.bowWristPx.y;
      const len = Math.hypot(dx, dy);
      if (len > 0) {
        const reqLen = FULL_DRAW_HAND_SEP_MIN * d.scale;
        const ux = dx / len, uy = dy / len;
        const markX = d.bowWristPx.x + ux * reqLen, markY = d.bowWristPx.y + uy * reqLen;
        const px = -uy, py = ux;
        const tick = d.scale * 0.08;
        drawTtLine({ x: markX - px * tick, y: markY - py * tick }, { x: markX + px * tick, y: markY + py * tick }, d.sepOk);
      }
    },
  },
  {
    id: "still",
    label: "STILL",
    sentence: "Has your draw hand stopped moving — are you holding steady?",
    doThis: "Hold your position still for a moment to light this. Keep moving your draw arm/hand to make it go dark.",
    note: null,
    read(landmarks) {
      const { drawWrist } = triggerTestRoleLabels();
      const d = debugInfo;
      return {
        lamp: d.stillOk === true,
        reason: d.reason,
        rows: [{ label: "Draw-wrist speed", value: `${ttFmt(d.speed)} × torso/second`, threshold: `needs ≤ ${FULL_DRAW_STILL_MAX}`, ok: d.stillOk }],
        inputs: [
          ttInputRow(landmarks, drawWrist, "draw wrist — compared against where it was on the previous frame; there's no second landmark for this one"),
        ],
      };
    },
    // A circle around the draw wrist sized to the allowed per-second drift radius — coloured by
    // stillOk. Not a motion trail (that would need extra state this file doesn't otherwise keep,
    // see the comment on lastDrawWrist being a single remembered frame, not a history buffer) —
    // just "this is the tolerance you're being held to right now".
    draw() {
      const d = debugInfo;
      if (!d.drawWristPx || !d.scale) return;
      drawTtCircle(d.drawWristPx, FULL_DRAW_STILL_MAX * d.scale, d.stillOk);
    },
  },
  {
    id: "fulldraw",
    label: "AT FULL DRAW",
    sentence: "Are all four full-draw checks true at the same time — anchored, arm straight, hands apart, and holding still?",
    doThis: "Draw fully to your anchor point and hold still to light this. Check the sub-checks below to see exactly which one is holding it back if it won't light.",
    note: null,
    read() {
      const d = debugInfo;
      const allFour = d.anchorOk === true && d.armOk === true && d.sepOk === true && d.stillOk === true;
      return {
        lamp: allFour,
        reason: d.reason,
        subLamps: [
          { label: "ANCHOR", ok: d.anchorOk },
          { label: "ARM", ok: d.armOk },
          { label: "SEP", ok: d.sepOk },
          { label: "STILL", ok: d.stillOk },
        ],
        rows: [],
        inputs: [],
      };
    },
  },
  {
    id: "raise",
    label: "RAISE",
    sentence: "Has your bow arm come up to shoulder height or higher — the first deliberate movement of your shot routine?",
    doThis: "Raise your bow arm up to shoulder height or above to light this. Lower it back down well below shoulder height to make it go dark — there's deliberate hysteresis built in, so it won't flicker right at shoulder height.",
    note: null,
    read(landmarks) {
      const { bowShoulder, bowWrist, bowHip } = triggerTestRoleLabels();
      return {
        lamp: raiseArmed,
        rows: [{
          label: "Bow-wrist height above shoulder",
          value: `${ttFmt(debugRaiseHeight)} × torso`,
          threshold: `≥ ${RAISE_TRIGGER_UP_FRACTION} lights it, ≤ ${RAISE_TRIGGER_DOWN_FRACTION} clears it`,
          ok: raiseArmed,
        }],
        inputs: [
          ttInputRow(landmarks, bowShoulder, "bow shoulder"),
          ttInputRow(landmarks, bowWrist, "bow wrist"),
          ttInputRow(landmarks, bowHip, "bow hip — for the torso-length scale"),
        ],
      };
    },
    // A horizontal line at bow-shoulder height — the RAISE_TRIGGER_UP_FRACTION (0) threshold —
    // coloured by raiseArmed. Computed straight from the raw landmark here rather than through
    // debugInfo (see this block's own comment): bowArmRaiseHeight's visibility needs are looser
    // than isAtFullDraw's own full-draw gate, so a raise can be live and armed while debugInfo's
    // pixel fields are still sitting on a bail from a frame before the draw wrist was visible.
    draw(landmarks, frameWidth, frameHeight) {
      const { bowShoulder } = triggerTestRoleLabels();
      if (!visible(landmarks, bowShoulder)) return;
      const shoulder = toPixelSpace(landmarks[bowShoulder], frameWidth, frameHeight);
      const halfW = (debugInfo.scale || 80) * 0.8;
      drawTtLine({ x: shoulder.x - halfW, y: shoulder.y }, { x: shoulder.x + halfW, y: shoulder.y }, raiseArmed);
    },
  },
  {
    id: "open",
    label: "OPEN",
    sentence: "Has the app decided a shot attempt is currently in progress — either because your bow arm came up to shoulder height, or your hands separated?",
    doThis: "Raise your bow arm to shoulder height, OR pull your hands apart, to open it. Relax completely — arm down, hands together — and hold still for a moment to close it.",
    note: null,
    read(landmarks) {
      const { bowShoulder, bowWrist, drawWrist } = triggerTestRoleLabels();
      const d = debugInfo;
      const isOpen = attempt !== null;
      let detail = "no attempt open";
      if (attempt) {
        const drawing = attempt.startMs !== null;
        const elapsedMs = performance.now() - (drawing ? attempt.startMs : attempt.watchStartedAt);
        detail = `open ${(elapsedMs / 1000).toFixed(1)}s (${drawing ? "drawing" : "raise phase"}) — peak hand sep ${ttFmt(attempt.peakHandSep)}`;
      }
      return {
        lamp: isOpen,
        rows: [
          {
            label: "Hand separation",
            value: `${ttFmt(d.handSep)} × torso`,
            threshold: `≥ ${DRAW_ATTEMPT_MIN_SEP} opens it`,
            ok: d.handSep == null ? null : d.handSep >= DRAW_ATTEMPT_MIN_SEP,
          },
          { label: "Raise armed", value: raiseArmed ? "yes" : "no", threshold: "also opens it", ok: raiseArmed },
          { label: "Attempt state", value: detail, threshold: "", ok: null },
        ],
        inputs: [
          ttInputRow(landmarks, bowShoulder, "bow shoulder — for RAISE"),
          ttInputRow(landmarks, bowWrist, "bow wrist — for RAISE and SEP"),
          ttInputRow(landmarks, drawWrist, "draw wrist — for SEP"),
        ],
      };
    },
  },
  {
    id: "eligible",
    label: "ELIGIBLE",
    sentence: "Has the tracking pipeline been reading you steadily for long enough — and, if the zoomed-in tracking box has stopped resizing — that THIS frame's numbers are trustworthy enough to log?",
    doThis: "This one isn't about your pose at all — it's about how long the app has been tracking you continuously. Stand still and stay in frame for a couple of seconds after the app starts (or after it loses and re-finds you) to light this. Step out of frame, or move enough that the tracking box keeps jumping around, to make it go dark.",
    note: null,
    read(landmarks) {
      const NAMED = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP, NOSE, MOUTH_L, MOUTH_R, L_EAR, R_EAR];
      const visibleCount = landmarks ? NAMED.filter((i) => visible(landmarks, i)).length : 0;
      const fired = isDebugEventLit(debugEvents.frameEligible, performance.now());
      return {
        lamp: fired,
        rows: [
          {
            label: "Consecutive good-tracking frames",
            value: `${settledFrames}`,
            threshold: `needs ≥ ${SETTLE_FRAMES_REQUIRED}`,
            ok: settledFrames >= SETTLE_FRAMES_REQUIRED,
          },
          {
            label: "Tracking box stable",
            value: ROI_CROPPING_ENABLED ? (lastCropBoxStable ? "yes" : "no") : "n/a — cropping is switched off",
            threshold: "",
            ok: ROI_CROPPING_ENABLED ? lastCropBoxStable : null,
          },
          {
            label: "Confidently-visible landmarks this frame",
            value: `${visibleCount}`,
            threshold: `≥ ${ROI_MIN_VISIBLE_LANDMARKS} keeps the tracking box locked on`,
            ok: visibleCount >= ROI_MIN_VISIBLE_LANDMARKS,
          },
        ],
        inputs: [], // this trigger is about frame HISTORY, not any one landmark — see the rows above instead
      };
    },
  },
  {
    id: "pose",
    label: "POSE",
    sentence: "Is a body being detected in the camera frame at all, right now?",
    doThis: "Stand where the camera can see your torso to light this. Step out of frame, turn away completely, or block the camera to make it go dark.",
    note: null,
    read(landmarks) {
      const NAMED = [
        [L_SHOULDER, "left shoulder"], [R_SHOULDER, "right shoulder"],
        [L_ELBOW, "left elbow"], [R_ELBOW, "right elbow"],
        [L_WRIST, "left wrist"], [R_WRIST, "right wrist"],
        [L_HIP, "left hip"], [R_HIP, "right hip"],
        [NOSE, "nose"], [MOUTH_L, "mouth (one side)"], [MOUTH_R, "mouth (other side)"],
        [L_EAR, "left ear"], [R_EAR, "right ear"],
        [L_ANKLE, "left ankle"], [R_ANKLE, "right ankle"],
      ];
      return {
        lamp: lastPoseSeen,
        rows: [{ label: "Body detected", value: lastPoseSeen ? "yes" : "no", threshold: "", ok: lastPoseSeen }],
        inputs: NAMED.map(([idx, label]) => ttInputRow(landmarks, idx, label)),
      };
    },
  },
  {
    id: "attn",
    label: "ATTN",
    sentence: "Is the app currently watching you at full speed — rather than idling to save battery between shots?",
    doThis: "Move — raise your arm, separate your hands, or just walk around — to light this (engaged). Stand completely relaxed and still for about a second and a half to let it go dark (idle). Nothing ever stops the app watching for good; idle just means it checks less often.",
    note: null,
    read(landmarks) {
      const { bowWrist, drawWrist } = triggerTestRoleLabels();
      return {
        lamp: attentionEngaged,
        rows: [
          {
            label: "Calm signal",
            value: debugAttnCalm == null ? "—" : debugAttnCalm ? "calm" : "not calm",
            threshold: "must hold calm for 1.5s to idle",
            ok: debugAttnCalm,
          },
          {
            label: "Hand separation",
            value: `${ttFmt(debugAttnHandSep)} × torso`,
            threshold: `needs ≤ ${ATTENTION_REST_HAND_SEP_MAX} to count as relaxed`,
            ok: debugAttnHandSep == null ? null : debugAttnHandSep <= ATTENTION_REST_HAND_SEP_MAX,
          },
          {
            label: "Body movement",
            value: `${ttFmt(debugAttnSpeed)} × torso/second`,
            threshold: `needs ≤ ${ATTENTION_REST_MOVE_MAX_PER_SEC} to count as still`,
            ok: debugAttnSpeed == null ? null : debugAttnSpeed <= ATTENTION_REST_MOVE_MAX_PER_SEC,
          },
        ],
        inputs: [
          ttInputRow(landmarks, bowWrist, "bow wrist"),
          ttInputRow(landmarks, drawWrist, "draw wrist"),
          ttInputRow(landmarks, L_HIP, "left hip — body reference point"),
          ttInputRow(landmarks, R_HIP, "right hip — body reference point"),
        ],
      };
    },
  },
];

let ttIndex = 0; // which of TRIGGER_DEFS is currently shown — advanced only by the big cycle button or its keyboard equivalents, never automatically
// Last REAL landmarks snapshot syncTriggerTestPanel saw (array, or null on genuine pose loss) —
// kept separate from the per-frame `landmarks` argument, which can also be undefined on an
// idle-throttle tick that skipped detection entirely (see syncDebugOverlay's own comment on that
// three-way distinction) — so the cycle button can re-render immediately with the last REAL
// answer, never a stale placeholder, without waiting for the next real frame.
let ttLastLandmarks;
let ttLastFrameW = 0, ttLastFrameH = 0;
let lastTtRenderMs = -Infinity; // throttle bookkeeping, same convention and constant as lastDebugRenderMs/DEBUG_OVERLAY_REFRESH_MS

// Builds the panel's DOM once (see buildDebugPanel's own comment for why this pattern — build
// once, only text/classes touched after) and wires the one big cycle button plus its keyboard
// equivalents. No-op, not even a DOM lookup, unless ?triggertest is in the URL (only ever called
// from the TRIGGERTEST bootstrap block near buildDebugPanel's own call site).
function buildTriggerTestPanel() {
  triggerTestEl.innerHTML = `
    <div class="tt-position" data-x="tt-position"></div>
    <div class="tt-title" data-x="tt-title"></div>
    <div class="tt-lampwrap"><span class="dbg-lamp tt-lamp" data-x="tt-lamp">DARK</span></div>
    <button type="button" class="tt-next-btn" data-x="tt-next">
      Next trigger →
      <span class="tt-next-sub">tap here, or press space / → on a keyboard</span>
    </button>
    <div class="dbg-reason" data-x="tt-reason" style="display:none"></div>
    <section class="dbg-section" data-x="tt-sublamps-section" style="display:none">
      <h2 class="dbg-heading">Which part is holding it back</h2>
      <div class="dbg-lamps" data-x="tt-sublamps"></div>
    </section>
    <section class="dbg-section">
      <h2 class="dbg-heading">What it checks</h2>
      <p class="tt-text" data-x="tt-sentence"></p>
    </section>
    <section class="dbg-section">
      <h2 class="dbg-heading">What it wants from you</h2>
      <p class="tt-text" data-x="tt-dothis"></p>
      <p class="tt-text tt-note" data-x="tt-note" style="display:none"></p>
    </section>
    <section class="dbg-section">
      <h2 class="dbg-heading">Live values</h2>
      <div data-x="tt-rows"></div>
    </section>
    <section class="dbg-section">
      <h2 class="dbg-heading">Raw inputs it's reading</h2>
      <div data-x="tt-inputs"></div>
    </section>
    <button type="button" class="tt-reload-btn" data-x="tt-reload">⟳ Hard reload (clears cached app.js)</button>
  `;
  ttRefs = {};
  triggerTestEl.querySelectorAll("[data-x]").forEach((el) => {
    ttRefs[el.dataset.x] = el;
  });

  const advance = (delta) => {
    ttIndex = (ttIndex + delta + TRIGGER_DEFS.length) % TRIGGER_DEFS.length;
    renderCurrentTrigger();
  };
  ttRefs["tt-next"].addEventListener("click", () => advance(1));
  // HARD RELOAD — owner's own request while testing a build that changes frequently: a plain
  // location.reload() is not enough here, because ES modules are cached per exact URL and a
  // reload of the SAME url can still hand back the OLD app.js (see CLAUDE.md's Testing section,
  // "ES modules cache per origin", and the bootstrap loader in index.html this depends on). The
  // fix has to change app.js's own URL, not just re-request this page — so this sets a fresh `cb`
  // (cache-bust) value, which index.html's bootstrap script reads and appends to app.js's own src,
  // guaranteeing a real network fetch instead of a cache hit. Every OTHER existing param
  // (?triggertest, ?debug, ?selftest, ...) is carried over unchanged via URLSearchParams, so the
  // owner lands back in the exact mode he was testing, not a bare reload to the plain app.
  ttRefs["tt-reload"].addEventListener("click", () => {
    const params = new URLSearchParams(location.search);
    params.set("cb", String(Date.now()));
    location.href = `${location.pathname}?${params.toString()}`;
  });
  // Keyboard equivalents cost nothing and help anyone testing on a desktop — Space/Enter/→ mirror
  // the one big button (forward, wrapping); ← is a bonus back-step the on-screen button
  // deliberately doesn't offer (the owner asked for ONE big forward button, not a pair).
  document.addEventListener("keydown", (e) => {
    if (!TRIGGERTEST) return;
    if (e.key === " " || e.key === "Enter" || e.key === "ArrowRight") {
      e.preventDefault();
      advance(1);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      advance(-1);
    }
  });

  renderCurrentTrigger(); // paint something immediately, before the first real frame arrives
}

// Repaints the currently-selected trigger's screen from ttLastLandmarks/ttLastFrameW/ttLastFrameH
// and whatever live module state that trigger's own `read` needs (debugInfo, raiseArmed, attempt,
// settledFrames, attentionEngaged, etc.) — never throttled itself (see syncTriggerTestPanel for
// the throttle), so the cycle button always feels instant even though the per-frame refresh below
// it is deliberately rate-limited like ?debug's own panel.
function renderCurrentTrigger() {
  if (!TRIGGERTEST || !ttRefs) return;
  const def = TRIGGER_DEFS[ttIndex];
  const result = def.read(ttLastLandmarks, ttLastFrameW, ttLastFrameH);
  const r = ttRefs;

  r["tt-position"].textContent = `${ttIndex + 1} of ${TRIGGER_DEFS.length}`;
  r["tt-title"].textContent = def.label;
  r["tt-lamp"].textContent = result.lamp ? "LIT" : "DARK";
  r["tt-lamp"].classList.toggle("lit", !!result.lamp);

  if (result.reason) {
    r["tt-reason"].textContent = result.reason;
    r["tt-reason"].classList.add("dbg-reason-bail");
    r["tt-reason"].style.display = "";
  } else {
    // Clear text/class too, not just display — a screen with no reason must leave nothing for a
    // later check (or a future CSS change) to find. Bug found in review: this branch used to only
    // hide the element, so a stale reason from a PREVIOUS screen sat in the DOM (invisible, but
    // still there) until the next screen that has one overwrote it.
    r["tt-reason"].textContent = "";
    r["tt-reason"].classList.remove("dbg-reason-bail");
    r["tt-reason"].style.display = "none";
  }

  if (result.subLamps) {
    r["tt-sublamps-section"].style.display = "";
    r["tt-sublamps"].innerHTML = result.subLamps
      .map((s) => {
        const state = s.ok === true ? "lit" : s.ok === false ? "dark" : "not evaluated yet";
        return `<div class="dbg-lampcell"><span class="dbg-lamp ${s.ok === true ? "lit" : ""}">${s.label}</span><div class="dbg-lampcaption">${state}</div></div>`;
      })
      .join("");
  } else {
    r["tt-sublamps-section"].style.display = "none";
    r["tt-sublamps"].innerHTML = "";
  }

  r["tt-sentence"].textContent = def.sentence;
  r["tt-dothis"].textContent = def.doThis;
  if (def.note) {
    r["tt-note"].textContent = def.note;
    r["tt-note"].style.display = "";
  } else {
    // Clear text too, not just display — same bug and same fix as tt-reason above. This was the
    // reported one: SEP's "known issue" paragraph stayed in the DOM after moving to STILL/AT FULL
    // DRAW/RAISE (none of which set a note), invisible only because display:none happened to be
    // set correctly — a false alarm waiting to happen the moment anything reads text instead of
    // computed style, which is exactly how this was caught.
    r["tt-note"].textContent = "";
    r["tt-note"].style.display = "none";
  }

  r["tt-rows"].innerHTML = result.rows
    .map((row) => {
      const cls = row.ok == null ? "uncertain" : row.ok ? "ok" : "warn";
      const thresh = row.threshold ? ` <span class="dbg-thresh">(${row.threshold})</span>` : "";
      return `<div class="dbg-row dbg-row-2col"><span class="dbg-name">${row.label}</span><span class="dbg-val ${cls}">${row.value}${thresh}</span></div>`;
    })
    .join("");

  r["tt-inputs"].innerHTML = result.inputs
    .map((inp) => {
      const cls = !inp.seen ? "uncertain" : inp.ok ? "ok" : "warn";
      const confText =
        inp.confidence == null
          ? "not seen this frame"
          : `${Math.round(inp.confidence * 100)}% confident (needs ≥ ${Math.round(MIN_VISIBILITY * 100)}%)`;
      return `<div class="dbg-row dbg-row-2col"><span class="dbg-name">${inp.label}</span><span class="dbg-val ${cls}">${confText}</span></div>`;
    })
    .join("");
}

// Per-frame hook, called from renderLoop exactly like syncDebugOverlay (see both call sites there
// for the idle-tick vs real-frame distinction `landmarks` carries). No-op — not even a DOM lookup
// — unless ?triggertest is in the URL. Updates the "last known real landmarks" snapshot on every
// call that carries one (so the cycle button can always re-render instantly from real data), but
// only repaints the DOM at DEBUG_OVERLAY_REFRESH_MS at most, same throttle and same reasoning as
// ?debug's own panel — nobody can read a number changing 30-60 times a second, and rebuilding this
// panel's rows every frame would cost real main-thread time for no benefit.
function syncTriggerTestPanel(nowMs, landmarks, frameWidth, frameHeight) {
  if (!TRIGGERTEST) return;
  if (landmarks !== undefined) {
    ttLastLandmarks = landmarks;
    ttLastFrameW = frameWidth;
    ttLastFrameH = frameHeight;
  }
  if (nowMs - lastTtRenderMs < DEBUG_OVERLAY_REFRESH_MS) return;
  lastTtRenderMs = nowMs;
  renderCurrentTrigger();
}

// Paints the CURRENTLY SELECTED trigger's acceptance region on top of whatever paintCanvas already
// drew this frame (skeleton included) — see the TRIGGER TEST OVERLAY DRAWING block above
// TRIGGER_DEFS for the shared drawing primitives and colour convention.
//
// CLIP SAFETY — the `activeRecording` check below is the ONLY thing that guarantees this, and it
// has to be a real runtime check, not just "called from a different code path than the recording
// branch": canvas.captureStream samples whatever pixels are actually sitting on the shared
// `canvas` element when the browser next grabs a frame for the stream, regardless of which
// function last drew to it or when — drawing this overlay onto the same canvas at any point while
// a clip is recording risks it landing in that clip, even if paintCanvas itself took its
// non-recording branch this same tick. So this bails out immediately whenever a clip is active,
// full stop, before touching the canvas at all.
//
// No-ops immediately when there's no fresh pose (landmarks null/undefined) or the current screen
// has no `draw` at all (FULLDRAW, OPEN, ELIGIBLE, POSE, ATTN) — nothing decorative gets invented
// for those, per the brief.
function drawTriggerTestOverlay(landmarks, frameWidth, frameHeight) {
  if (!TRIGGERTEST || !landmarks || activeRecording) return;
  const def = TRIGGER_DEFS[ttIndex];
  if (!def.draw) return;
  ctx.save();
  ctx.lineWidth = 2;
  try {
    def.draw(landmarks, frameWidth, frameHeight);
  } finally {
    ctx.restore();
  }
}

// Built here, not in the early bootstrap block near triggerTestEl's own declaration — see that
// block's comment for why (TRIGGER_DEFS's temporal dead zone). This is the actual "turn it on"
// call for the whole feature; everything above this point in the file is just declarations.
if (TRIGGERTEST) buildTriggerTestPanel();
// ===========================================================================

// ===== CALIBRATION — passive capture, wired into renderLoop below. No button, no explicit
// trigger — the owner's own decision (see HANDOVER.md Stage 4). "Is he standing there, readable"
// reuses signals the pipeline already computes, rather than a new detector:
//   - attentionEngaged === false — the ATTENTION GATING calm detector (attentionIsClearlyCalm)
//     has already decided hands are relaxed AND the body isn't moving, continuously, for at least
//     ATTENTION_IDLE_AFTER_MS. That IS "a stable pose" in this app's own existing terms.
//   - MIN_VISIBILITY, via bodyProportionsOf/visible() — every ratio below is null unless the
//     landmarks it needs are confidently visible.
//   - bothAnklesVisible — "body fully in frame", for the framing note specifically.
// Called every frame landmarks come back (renderLoop), so while idle this runs at the throttled
// idle-sample rate — plenty to gather CALIBRATION_MIN_SAMPLES good frames in a couple of seconds
// of ordinary standing-still. While an attempt is open, updateAttentionState's own hard rule forces
// attentionEngaged true, so this always resets rather than sampling — a real shot's landmarks can
// never contaminate a calibration.
function sampleForCalibration(landmarks, frameWidth, frameHeight, nowMs) {
  if (calibrationDone) return; // this session's one verdict is already in

  if (attentionEngaged) {
    // Moving, or not yet held calm long enough (see ATTENTION GATING above) — whatever was being
    // built up during a PREVIOUS calm stretch doesn't carry over into this one; a calibration
    // built from samples either side of him walking around isn't a reading of one steady pose.
    calibrationSamples = [];
    framingSamples = [];
    calibrationCalmSinceMs = null;
    return;
  }
  if (calibrationCalmSinceMs === null) calibrationCalmSinceMs = nowMs;

  const sample = bodyProportionsOf(landmarks, frameWidth, frameHeight);
  if (sample) calibrationSamples.push(sample);

  // FRAMING SIGNATURE (optional add-on) — collected alongside, from the same frames, but never
  // gates anything: if this never fills up, calibration above still completes on its own schedule.
  const framingSample = framingSignatureOf(landmarks, frameWidth, frameHeight);
  if (framingSample) framingSamples.push(framingSample);

  // Framing note: only after a real held stretch of calm (not one blip — see
  // attentionIsClearlyCalm's own "positive proof" standard, which this borrows) and only once,
  // ever, per session. Independent of whether calibration itself succeeds — the ratios above
  // never need the ankles at all, so a cut-off framing problem still gets its own word even on a
  // session where calibration completes fine.
  if (
    !framingStatusLine &&
    nowMs - calibrationCalmSinceMs >= ATTENTION_IDLE_AFTER_MS &&
    !bothAnklesVisible(landmarks)
  ) {
    framingStatusLine = describeFraming(false);
    renderCalibrationStatus();
  }

  if (calibrationSamples.length >= CALIBRATION_MIN_SAMPLES) finishPassiveCalibration();
}

function finishPassiveCalibration() {
  const fresh = medianCalibrationOf(calibrationSamples);
  const freshFraming = medianFramingOf(framingSamples); // optional — may be null; see medianFramingOf's own comment
  calibrationSamples = [];
  framingSamples = [];
  if (!fresh) return; // not enough GOOD frames even after CALIBRATION_MIN_SAMPLES pushes (e.g. persistently near side-on) — keep trying on the next calm stretch, calibrationDone stays false

  calibrationDone = true;
  activeCalibration = fresh; // this session's nonsense-check always uses today's fresh reading — see handSepIsPlausible and the owner's own "recheck it every time" decision
  const stored = loadStoredCalibration();
  const verdict = calibrationVerdict(stored, fresh);
  // One record, one write — the framing signature (optional) rides along with the same
  // proportions it was measured alongside, never a second storage mechanism (see HANDOVER.md).
  // takenAt records when the CURRENTLY-TRUSTED calibration was last confirmed — read back by
  // calibrationShareLine so the PM can tell from shared text whether calibration ever ran at all,
  // not just guess from its absence (see the PM's own review of this gap).
  if (verdict.save) saveCalibration({ ...fresh, framing: freshFraming, takenAt: Date.now() });
  if (verdict.message) calibrationStatusLine = verdict.message; // disagreement only — see calibrationVerdict; agreement (or first-ever calibration) says nothing, on purpose

  // FRAMING SIGNATURE comparison — entirely independent of the verdict above (see
  // framingChangeMessage's own null-safety): compares against whatever was stored BEFORE this
  // write, regardless of whether the proportions above agreed or not.
  const framingMessage = framingChangeMessage(stored?.framing ?? null, freshFraming);
  if (framingMessage) framingChangeStatusLine = framingMessage;

  renderCalibrationStatus();
}

// Thin DOM wrapper — PROVISIONAL presentation (see HANDOVER.md Stage 3/4): a plain status line
// that sets itself, same shape as modelStatusLine/clipsUnavailableReason, shown live (so the
// framing note is actionable before he walks off) and folded into the shot log below (so it's
// still there afterwards too — see CLAUDE.md's "recorded and still there later" rule). Stage 3's
// Setup screen is its real home; move it there rather than duplicating it when that lands.
function renderCalibrationStatus() {
  const calibrationState = calibrationStatusText(calibrationDone, calibrationStatusLine);
  const parts = [calibrationState?.text, framingStatusLine, framingChangeStatusLine].filter(Boolean);
  if (parts.length === 0) {
    calibrationStatusEl.classList.add("hidden");
    return;
  }
  calibrationStatusEl.textContent = parts.join(" ");
  // Neutral ("hasn't run yet") is information, not a warning (see calibrationStatusText's own
  // comment) — it must never look like something is broken, because nothing is. Only apply the
  // quiet styling when nothing ELSE on the line is an actual warning (the legs-cut-off or
  // framing-changed notes, which are).
  calibrationStatusEl.classList.toggle("neutral", calibrationState?.tone === "neutral" && !framingStatusLine && !framingChangeStatusLine);
  calibrationStatusEl.classList.remove("hidden");
}
// ===========================================================================

function renderLoop() {
  requestAnimationFrame(renderLoop);
  const now = performance.now();

  // Display-only: a live "rendered fps" figure for the ?debug SESSION section, recomputed from
  // consecutive renderLoop calls — this callback runs every rAF tick regardless of attention
  // gating (only the detection work below is ever skipped), so this is the same "whole frame,
  // drawing included" quantity the one-time POSE MODEL warm-up measurement reports, just live.
  // Guarded by DEBUG so it costs nothing outside ?debug.
  if (DEBUG) {
    if (debugLastFrameTs !== null) debugInstantFps = 1000 / (now - debugLastFrameTs);
    debugLastFrameTs = now;
  }

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
      paintCanvas(null); // <video> keeps playing on its own; this only matters if a clip happens to be recording (see paintCanvas)
      // landmarks argument omitted (undefined), not null: this tick never ran detection at all —
      // see syncDebugOverlay's own comment on the difference — so the panel keeps showing its
      // last real reading through the idle gap instead of flashing "not seen" for a reason that
      // has nothing to do with whether the archer is actually there.
      syncDebugOverlay(now);
      syncTriggerTestPanel(now);
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
  const inferenceMs = performance.now() - inferenceStart;
  if (DEBUG) lastInferenceMs = inferenceMs; // display-only, see its own comment
  measurePoseModelPerf(inferenceMs, now);
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
  // TRIGGER TEST — snapshot the previous-frame ATTN state BEFORE updateAttentionState below
  // overwrites it, so the recompute after the real call can reconstruct exactly what it saw. Two
  // plain local variables, read only a few lines down; never assigned anywhere else.
  const ttPrevAttnRef = TRIGGERTEST ? attentionPrevRef : null;
  const ttPrevAttnEvalMs = TRIGGERTEST ? attentionLastEvalMs : null;

  updateAttentionState(now, rawLandmarks, frameWidth, frameHeight);

  // TRIGGER TEST — display-only. Calls the exact same pure functions updateAttentionState just
  // used internally (attentionIsClearlyCalm, handSeparationForAttention, bodyReferencePoint,
  // attentionScale, toPixelSpace) a second time, with the snapshot taken above, purely so the ATTN
  // screen can show the live numbers behind the engaged/idle decision. Read-only: this can never
  // feed back into attentionEngaged or anything else the real app depends on.
  if (TRIGGERTEST) {
    const ttDtSec = ttPrevAttnEvalMs === null ? 0 : (now - ttPrevAttnEvalMs) / 1000;
    debugAttnCalm = attentionIsClearlyCalm(rawLandmarks, ttPrevAttnRef, ttDtSec, frameWidth, frameHeight);
    debugAttnHandSep = rawLandmarks ? handSeparationForAttention(rawLandmarks, frameWidth, frameHeight) : null;
    const ttRef = rawLandmarks ? bodyReferencePoint(rawLandmarks) : null;
    const ttScale = rawLandmarks ? attentionScale(rawLandmarks, frameWidth, frameHeight) : null;
    if (ttRef && ttPrevAttnRef && ttDtSec > 0 && ttScale) {
      const a = toPixelSpace(ttRef, frameWidth, frameHeight);
      const b = toPixelSpace(ttPrevAttnRef, frameWidth, frameHeight);
      debugAttnSpeed = Math.hypot(a.x - b.x, a.y - b.y) / ttScale / ttDtSec;
    } else {
      debugAttnSpeed = null;
    }
  }

  // Display-only: this frame's smoothed landmarks (or null, on genuine pose loss), for
  // syncDebugOverlay's MEASURES/torso-scale section at the bottom of this function — declared
  // out here because the success branch below sets it inside its own block scope. Never read by
  // anything except the ?debug panel.
  let landmarksForDebug = null;

  if (!rawLandmarks) {
    updateCue(true, false); // "not seeing you" — set before endAttempt below, so a rejected/logged flash this same frame knows to hand back to "lost", not "resting", once it clears
    paintCanvas(null); // no skeleton to draw; <video> keeps the on-screen view alive on its own, and paintCanvas still bakes the picture in if a clip is recording
    setReadout(readoutBowArm, valueBowArm, "— uncertain", "uncertain");
    setValueState(valueShoulderBow, "—", "uncertain");
    setValueState(valueShoulderDraw, "—", "uncertain");
    setReadout(readoutElbow, valueElbow, "— uncertain", "uncertain");
    if (DEBUG || TRIGGERTEST) {
      debugInfo = emptyDebugInfo("no pose detected this frame");
      lastPoseSeen = false;
    }
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
    landmarksForDebug = landmarks; // display-only — see its own declaration and syncDebugOverlay's call at the bottom of this function
    if (DEBUG || TRIGGERTEST) lastPoseSeen = true;
    paintCanvas(landmarks);
    updateBowArmReadout(landmarks, frameWidth, frameHeight);
    updateShoulderDropReadout(landmarks, frameWidth, frameHeight);
    updateDrawElbowReadout(landmarks, frameWidth, frameHeight);
    sampleForCalibration(landmarks, frameWidth, frameHeight, now);
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
    if (DEBUG || TRIGGERTEST) lastCropBoxStable = cropBoxStableThisFrame; // display-only, see its own declaration
    prevUsedCropBox = usedCropBox;
    const frameEligible = advanceSettling(!!usedCropBox, cropBoxStableThisFrame);
    if ((DEBUG || TRIGGERTEST) && frameEligible) debugEvents.frameEligible = now; // display-only latch, see DEBUG_EVENT_LATCH_MS
    isAtFullDraw(landmarks, now, frameEligible, frameWidth, frameHeight);
    // `attempt` (module-level, see trackShotAttempt) is already up to date for THIS frame — the
    // isAtFullDraw call above just ran it. Read it straight rather than threading a return value
    // through isAtFullDraw/trackShotAttempt purely for this: "watching this draw" IS "an attempt
    // is open", the exact same fact the shot log and clip recording already key off.
    updateCue(false, attempt !== null);

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

  syncDebugOverlay(now, landmarksForDebug, frameWidth, frameHeight);
  syncTriggerTestPanel(now, landmarksForDebug, frameWidth, frameHeight);
  drawTriggerTestOverlay(landmarksForDebug, frameWidth, frameHeight);
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
  // Also updates the CSS mirror classes (see syncMirrorClasses) — every call site here changes
  // effectiveMirror's inputs, and paintCanvas only re-syncs those once renderLoop is running, so
  // without this the picture would show unmirrored for a moment right after startup or a camera
  // switch, before the first frame paints.
  syncMirrorClasses();
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
  shotLogShareTextEl.classList.add("hidden"); // don't leave the manual-copy fallback showing next time the log opens
}

shotLogCloseBtn.addEventListener("click", closeShotLog);

// Share: gathers the context that changes how the numbers should be read (camera resolution,
// handedness, mirror state — all live DOM/module state, read here rather than inside
// buildShareText so that function stays pure) and hands the rest to buildShareText/
// shareSessionText. video.videoWidth/Height can briefly be 0 before the camera's ready; buildShareText
// already renders that honestly as "?x?" rather than "0x0".
shotLogShareBtn.addEventListener("click", () => {
  const text = buildShareText(log, {
    shotCount,
    fullDrawShotCount,
    rejectedAttemptCount,
    unsettledAttemptCount,
    attentionIdlePeriods,
    attentionLateWakeCount,
    clipsUnavailableReason,
    modelStatusLine,
    rightHanded,
    mirrored: effectiveMirror(facingMode, mirrorToggled),
    cameraWidth: video.videoWidth || null,
    cameraHeight: video.videoHeight || null,
    sessionWasRestored,
    storedCalibration: loadStoredCalibration(), // read fresh, not activeCalibration — reflects what's actually on file right now, whether or not this session's own check has landed
    calibrationDone,
    calibrationStatusLine,
  });
  shareSessionText(text);
});

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

// Must run before anything below that can call saveSessionToStorage (markClipsUnavailable
// included) — otherwise this session's own empty starting state would overwrite the very save
// being restored, before it was ever read. See SESSION PERSISTENCE above.
restoreSessionFromStorage();

if (!CLIP_SUPPORTED) {
  markClipsUnavailable("Clips unavailable in this browser — everything else still works, just no shot videos.");
}
renderShotLog(); // shows the "no shots yet" placeholder (and the banner above, if set) before the first shot comes in — or the restored log, if one was found

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
  // Shown from the very first moment, not just once a frame has run through sampleForCalibration —
  // "not yet confirmed" is the honest state from startup, and silence before then would be exactly
  // the ambiguity the PM's review flagged (see calibrationStatusText's own comment).
  renderCalibrationStatus();
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
  const savedFullDrawShotCount = fullDrawShotCount;
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
  const savedCueOutcomeTimer = cueOutcomeTimer;
  const savedCueLastLost = cueLastLost;
  const savedCueLastWatching = cueLastWatching;
  const savedCueClassName = cueEl.className;
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
    const lm = Array.from({ length: 25 }, () => ({ x: 0, y: 0, visibility: 0 }));
    // A landmark this fixture never mentions at all defaults to invisible (visibility: 0) — see
    // the FIXTURE BUG comment on `base` below for why: an untouched slot sitting at the frame's
    // corner, marked fully confident, is a landmark that isn't really there pretending to be one.
    // A landmark the fixture DOES mention gets visibility: 1 by default — nearly every override in
    // this file is written as just `{ x, y }` with no visibility key, on the long-standing
    // convention that placing a real coordinate there means "this joint is visible here"; that
    // default is applied FIRST so an override that explicitly sets visibility (occlusion fixtures
    // like `{ x, y, visibility: 0 }`) still wins over it.
    for (const i in overrides) lm[i] = { visibility: 1, ...overrides[i] };
    return lm;
  };
  // Shared skeleton scale: shoulder-to-hip torso length of 0.3.
  //
  // FIXTURE BUG FOUND THIS TASK (2026-08-24), fixed at TWO levels. `base` set no ears (7/8), so any
  // fixture built on it that reached the ANCHOR DIRECTION backward-check (see
  // FULL_DRAW_ANCHOR_BACKWARD_MIN below) was silently measuring the anchor->ear axis off whatever
  // mkLandmarks defaulted an unmentioned landmark to — back then, { x: 0, y: 0, visibility: 1 },
  // the frame's top-left corner marked fully confident. The shared `drawn` full-draw fixture read a
  // NEGATIVE (in-front-of-the-mouth) backward value off that corner, which only passed because the
  // placeholder threshold (-0.1) was loose enough to swallow nonsense. Real ear positions below
  // close that hole directly: any fixture spreading `base` and not overriding 7/8 now gets a
  // genuine head, not a corner.
  //
  // The wider fix, applied after auditing every fixture in this file for the same class of defect:
  // mkLandmarks itself now defaults an untouched landmark to visibility: 0 (see its own comment
  // above), so a fixture that forgets to set a landmark the code under test actually needs gets a
  // correctly-invisible slot instead of a corner point pretending to be a confident real one — this
  // class of bug is now structurally caught by the pipeline's own visibility gates, rather than
  // relying on the next engineer to notice a specific coordinate looks wrong. That sweep found one
  // more real instance beyond `base`'s ears: the `slightBend` fixture further down (search for its
  // own FIXTURE BUG comment) never set a draw wrist at all, and was silently passing isAtFullDraw's
  // visibility gate off the same corner default.
  //
  // Anthropometric reasoning for where the ears actually go, at this rig's own 0.3 torso-length
  // scale: this is a side-on view with the bow arm reaching toward decreasing x (see the
  // `raise`/`drawn` fixtures below: bow wrist sits at x=0.0, far from the body) and the draw hand
  // coming back near the mouth at higher x — so the archer's face points toward LOW x, and the
  // ears (back of the head) sit toward HIGH x, same side as the mouth but further out.
  // - Horizontal (mouth-to-ear, tragus to lip corner): roughly 8-9cm on an adult, against a
  //   roughly 45-50cm shoulder-to-hip torso -> ratio ~0.17-0.19. At this rig's 0.3 torso scale
  //   that's an offset of ~0.05-0.06; used 0.06.
  // - Vertical (tragus sits slightly above mouth level, not level with it): roughly 3-4cm on an
  //   adult against the same ~45-50cm torso -> ratio ~0.07-0.08, i.e. ~0.02-0.025 at this scale;
  //   used 0.025 (image y grows downward, so "above" is a SMALLER y).
  // Both ears are placed at the SAME point: a true side-on profile projects the near and far ear
  // to almost the same 2D position (the far one occluded behind the head, differing mainly in
  // real-world visibility, which is a separate axis fixtures below override explicitly — e.g.
  // drawEarOccluded/bothEarsOccluded — when they specifically need one ear hidden).
  const base = {
    9: { x: 0.5, y: 0.3 }, // mouth L
    10: { x: 0.5, y: 0.3 }, // mouth R
    7: { x: 0.56, y: 0.275 }, // L_EAR — see anthropometric reasoning above
    8: { x: 0.56, y: 0.275 }, // R_EAR — same point; a real side-on profile projects both ears together
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
  //
  // midDraw2's wrist position was corrected this task (2026-08-24), same audit as base's ear fix
  // above: it used to sit at (0.5, 0.32), directly below the mouth with zero component toward the
  // (previously corner-default) ear axis, so once base got a real head this fixture's own
  // anchorBackward went NEGATIVE — it would have failed the direction check that didn't exist when
  // this fixture was written, silently turning "rejected by stillness alone" into "rejected by
  // stillness AND direction", which breaks the isolation this test claims in its own comment above
  // (a broken stillness check could then hide behind the direction check still failing, and this
  // assertion would keep passing for the wrong reason). Moved to (0.56, 0.30) — toward the real
  // draw ear, same side base's fix uses — so anchor/arm/separation all genuinely pass on their own
  // and stillness is once again the only thing left to reject it.
  lastDrawWrist = null;
  const midDraw1 = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.4, y: 0.31 } });
  isAtFullDraw(midDraw1, 0, true, NOOP_W, NOOP_H); // seeds lastDrawWrist; this call's own result isn't the point
  const midDraw2 = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.56, y: 0.3 } });
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

  // --- ?debug NEVER-BLANK: isAtFullDraw must leave debugInfo as a real, non-null object naming a
  // plain-language reason on every bail path it can take, never bare null or an object with no
  // reason — a blank ?debug panel was exactly the bug this redesign exists to fix. debugInfo is
  // only ever written `if (DEBUG)` (see isAtFullDraw itself), so this whole block is correctly a
  // no-op — not a false pass — on a page loaded without ?debug in the URL.
  if (DEBUG) {
    lastDrawWrist = null;
    const missingDrawWrist = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.52, y: 0.31, visibility: 0 } });
    isAtFullDraw(missingDrawWrist, 0, true, NOOP_W, NOOP_H);
    console.assert(
      debugInfo !== null && typeof debugInfo.reason === "string" && debugInfo.reason.length > 0,
      "a missing required landmark must leave debugInfo with a real, non-empty reason — never blank"
    );
    console.assert(
      debugInfo.reason.includes("draw wrist"),
      `the missing-landmark reason should name the actual joint that's missing (got ${JSON.stringify(debugInfo.reason)})`
    );

    lastDrawWrist = null;
    const noAnchor = mkLandmarks({
      ...base,
      9: { x: 0.5, y: 0.3, visibility: 0 }, // mouth L hidden
      10: { x: 0.5, y: 0.3, visibility: 0 }, // mouth R hidden
      0: { x: 0.5, y: 0.25, visibility: 0 }, // nose hidden too — no fallback anchor left at all
      15: { x: 0.0, y: 0.3 },
      16: { x: 0.52, y: 0.31 },
    });
    isAtFullDraw(noAnchor, 0, true, NOOP_W, NOOP_H);
    console.assert(
      debugInfo.reason && debugInfo.reason.includes("anchor"),
      `losing both mouth and nose should report a no-anchor reason (got ${JSON.stringify(debugInfo.reason)})`
    );

    lastDrawWrist = null;
    const noScale = mkLandmarks({
      ...base,
      23: { x: 0.3, y: 0.6, visibility: 0 }, // bow hip hidden
      24: { x: 0.5, y: 0.6, visibility: 0 }, // draw hip hidden too — no fallback scale left either
      15: { x: 0.0, y: 0.3 },
      16: { x: 0.52, y: 0.31 },
    });
    isAtFullDraw(noScale, 0, true, NOOP_W, NOOP_H);
    console.assert(
      debugInfo.reason && debugInfo.reason.includes("torso scale"),
      `losing both hips should report a no-torso-scale reason (got ${JSON.stringify(debugInfo.reason)})`
    );

    // A genuine full draw, for contrast: the never-blank fix must not turn a perfectly good,
    // fully-readable frame into a fake bail — reason must clear back to null and the real numbers
    // must still be there.
    lastDrawWrist = null;
    isAtFullDraw(drawn, 500, true, NOOP_W, NOOP_H);
    console.assert(debugInfo.reason === null, "a fully-readable full-draw frame must clear the bail reason, not invent one");
    console.assert(
      typeof debugInfo.handSep === "number" && typeof debugInfo.bowArmAngle === "number",
      "a fully-readable frame's debugInfo must carry its real computed numbers"
    );
  }

  // --- ?debug momentary-event lamps (see DEBUG_EVENT_LATCH_MS): a same-frame event must still
  // read as lit by the time the panel's own throttled refresh gets to it, and must go dark again
  // once the latch window has genuinely passed — the whole point of latching at all. Pure
  // function, no ?debug/module state involved, so this runs unconditionally either way.
  console.assert(isDebugEventLit(0, 1000) === false, "an event that never happened (timestamp 0) must never read as lit");
  console.assert(isDebugEventLit(1000, 1000) === true, "an event that just happened this instant must be lit");
  console.assert(
    isDebugEventLit(1000, 1000 + DEBUG_EVENT_LATCH_MS) === true,
    "an event must still be lit exactly at the edge of its own latch window"
  );
  console.assert(
    isDebugEventLit(1000, 1000 + DEBUG_EVENT_LATCH_MS + 1) === false,
    "an event must go dark once its latch window has genuinely passed — a latch that never turns off would hide every OTHER momentary event by staying lit forever"
  );

  // --- ?triggertest TRIGGER_DEFS: structural shape first (runs unconditionally — doesn't depend
  // on debugInfo being populated), then each screen's `read()` driven with real fixtures through
  // the real pipeline functions, gated `if (DEBUG || TRIGGERTEST)` for the same reason as the
  // ?debug NEVER-BLANK block above: debugInfo/debugRaiseHeight/lastPoseSeen/lastCropBoxStable are
  // only populated under one of those two flags (see their assignments), so this is a correct
  // no-op — not a false pass — on a page loaded with neither.
  console.assert(TRIGGER_DEFS.length === 10, `TRIGGER_DEFS must have exactly 10 screens, got ${TRIGGER_DEFS.length}`);
  console.assert(
    TRIGGER_DEFS.map((d) => d.label).join(",") === "ANCHOR,ARM,SEP,STILL,AT FULL DRAW,RAISE,OPEN,ELIGIBLE,POSE,ATTN",
    `TRIGGER_DEFS order/labels must match the brief exactly, got ${TRIGGER_DEFS.map((d) => d.label).join(",")}`
  );
  const ttById = Object.fromEntries(TRIGGER_DEFS.map((d) => [d.id, d]));

  if (DEBUG || TRIGGERTEST) {
    // A genuine held full draw (reusing the `drawn` fixture from earlier) must light ANCHOR, ARM,
    // SEP, STILL and the AT FULL DRAW composite all at once — the SHOULD-FIRE case for all five
    // full-draw screens, and the never-blank reason must have cleared (see debugInfo.reason).
    lastDrawWrist = null;
    isAtFullDraw(drawn, 0, true, NOOP_W, NOOP_H);
    isAtFullDraw(drawn, 500, true, NOOP_W, NOOP_H); // second frame: zero speed, so STILL passes too
    console.assert(ttById.anchor.read(drawn).lamp === true, "ANCHOR screen must read lit on a genuine held full draw");
    console.assert(ttById.arm.read(drawn).lamp === true, "ARM screen must read lit on a genuine held full draw");
    console.assert(ttById.sep.read(drawn).lamp === true, "SEP screen must read lit on a genuine held full draw");
    console.assert(ttById.still.read(drawn).lamp === true, "STILL screen must read lit on a genuine held full draw");
    const fdOnDraw = ttById.fulldraw.read(drawn);
    console.assert(fdOnDraw.lamp === true, "AT FULL DRAW screen must read lit once all four sub-checks are true");
    console.assert(fdOnDraw.subLamps.every((s) => s.ok === true), "AT FULL DRAW's own sub-lamps must all show lit when the composite is lit");

    // THE proven bug this whole tool exists to surface (see CLAUDE.md): a realistic body standing
    // at rest, arms hanging naturally at the sides — NOT drawing at all — must make SEP read lit
    // anyway, because an ordinary adult's shoulder width alone is comfortably past
    // FULL_DRAW_HAND_SEP_MIN as a fraction of torso length. Anthropometric assumptions (standard
    // adult anthropometry figures, as fractions of standing height): biacromial (shoulder) width
    // ≈0.26x height, shoulder-to-hip (torso) length ≈0.29x height. On this file's shared 0.3
    // torso-length scale (see `base` above), that puts the shoulders ≈0.26 apart (0.26/0.29 × 0.3),
    // not `base`'s own narrower 0.2 — `base` was never meant to be an anthropometrically realistic
    // body, just a convenient shared rig for angle fixtures, so this trigger-test fixture overrides
    // shoulder width rather than reusing it. Hanging arms, both wrists at the same height straight
    // down from their own shoulder, then separate horizontally by that same ≈0.26/0.3 ≈ 0.87
    // torso-lengths — comfortably past the 0.75 threshold.
    lastDrawWrist = null;
    const restingArmsAtSides = mkLandmarks({
      11: { x: 0.3, y: 0.3 }, // bow shoulder
      12: { x: 0.56, y: 0.3 }, // draw shoulder — 0.26 apart, real anthropometric shoulder width at this torso scale (see comment above)
      13: { x: 0.3, y: 0.45 }, // bow elbow — hangs straight down from the bow shoulder
      14: { x: 0.56, y: 0.45 }, // draw elbow — hangs straight down from the draw shoulder
      15: { x: 0.3, y: 0.58 }, // bow wrist — further down, same x as its own shoulder (hanging straight down, not out)
      16: { x: 0.56, y: 0.58 }, // draw wrist — same
      23: { x: 0.3, y: 0.6 }, // bow hip
      24: { x: 0.56, y: 0.6 }, // draw hip
      9: { x: 0.43, y: 0.3 }, // mouth — centred over the torso, near the head, far from either resting hand
      10: { x: 0.43, y: 0.3 },
      // Ears — added in the same audit as `base`'s own ear fix above. This fixture doesn't spread
      // `base`, so it had the identical corner-default gap: isAtFullDraw's ANCHOR DIRECTION check
      // (see FULL_DRAW_ANCHOR_BACKWARD_MIN) would have silently measured off a (0,0) landmark if
      // anything here ever isolated it. Currently harmless in practice — this fixture's own
      // anchorDist is already ~1.0 (its whole point: the resting hand is nowhere near the mouth),
      // so anchorOk fails on distance alone regardless of direction — but a real head costs nothing
      // and means that stays true by the geometry, not by an unrelated gate hiding it. Same offset
      // convention as `base` (~0.06 out, ~0.025 up from the mouth); this fixture has no strong
      // facing direction of its own (arms hang straight down, not reaching for a bow), so both ears
      // sit symmetrically either side of the mouth rather than both toward one side.
      7: { x: 0.37, y: 0.275 }, // L_EAR
      8: { x: 0.49, y: 0.275 }, // R_EAR
    });
    const restScale = torsoLength(restingArmsAtSides, R_SHOULDER, R_HIP, NOOP_W, NOOP_H);
    const restHandSep =
      Math.hypot(
        restingArmsAtSides[R_WRIST].x - restingArmsAtSides[L_WRIST].x,
        restingArmsAtSides[R_WRIST].y - restingArmsAtSides[L_WRIST].y
      ) / restScale;
    console.assert(
      restHandSep >= FULL_DRAW_HAND_SEP_MIN,
      `resting-arms-at-sides fixture must itself measure past FULL_DRAW_HAND_SEP_MIN (${FULL_DRAW_HAND_SEP_MIN}) for this to be a fair test of the known bug — got ${restHandSep.toFixed(3)}`
    );
    isAtFullDraw(restingArmsAtSides, 0, true, NOOP_W, NOOP_H);
    isAtFullDraw(restingArmsAtSides, 500, true, NOOP_W, NOOP_H); // standing still — isolates the finding to SEP/ANCHOR/ARM specifically, not STILL
    console.assert(
      ttById.sep.read(restingArmsAtSides).lamp === true,
      "PROVEN BUG (see CLAUDE.md): SEP must read lit on an ordinary resting stance — if this ever reads dark, either the underlying bug was fixed (update the report) or this fixture stopped being realistic"
    );
    console.assert(
      ttById.anchor.read(restingArmsAtSides).lamp === false,
      "ANCHOR must correctly stay dark at rest (the draw hand is nowhere near the face) — this is what keeps AT FULL DRAW honest even while SEP alone is wrong"
    );
    console.assert(
      ttById.fulldraw.read(restingArmsAtSides).lamp === false,
      "AT FULL DRAW must stay dark at rest even though SEP alone incorrectly fires — ANCHOR failing must be enough to keep the composite honest"
    );
    // ARM CONE FIX (this task, 2026-08-24): a relaxed hanging arm doesn't buckle at the elbow
    // under just its own weight, so it still measures as very nearly straight (armStraightOk true
    // on its own) — this fixture used to be the "additional finding" proving ARM had the same
    // shape of bug as SEP, firing on an ordinary resting stance. bowArmElevationOf now measures
    // this exact arm at ~-90° off horizontal (straight down), nowhere near the
    // FULL_DRAW_ARM_CONE_APERTURE_DEG (45°) aperture, so armConeOk — and therefore the combined
    // armOk — now correctly rejects it. THIS is the headline assertion: an arm hanging at the
    // side must fail armOk, and this exact fixture is proven (see the PM's own verification run,
    // reverting FULL_DRAW_ARM_CONE_APERTURE_DEG's check) to have read `true` before this fix.
    console.assert(
      ttById.arm.read(restingArmsAtSides).lamp === false,
      "ARM CONE FIX: an arm hanging straight at the archer's side must NOT read as armOk even though it measures as straight — this is the headline case FULL_DRAW_ARM_CONE_APERTURE_DEG exists to catch"
    );

    // ===== ANCHOR DIRECTION — this task (2026-08-24). Owner: "i cannot anchor ahead of my mouth.
    // it'll always be below or slightly backwards." Fixtures below probe each new sub-check in
    // isolation. Shared head geometry: mouth at (0.40, 0.20); draw-side ear (R_EAR, since
    // rightHanded is true throughout selfTest) straight out to the side at the SAME height — a
    // deliberately simple axis (pure +x) so anchorBackward reduces to a plain x-comparison and
    // every number below can be hand-checked.
    const anchorHead = {
      9: { x: 0.4, y: 0.2 }, 10: { x: 0.4, y: 0.2 }, // mouth
      [R_EAR]: { x: 0.5, y: 0.2 }, // draw-side ear when rightHanded
      [L_EAR]: { x: 0.3, y: 0.2 }, // bow-side ear (fallback)
    };

    // Correct anchor: right distance, at/below mouth level, behind it toward the draw ear — all
    // three sub-checks, and the combined anchorOk, must pass.
    lastDrawWrist = null;
    const goodAnchor = mkLandmarks({ ...base, ...anchorHead, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.42, y: 0.22 } });
    isAtFullDraw(goodAnchor, 0, true, NOOP_W, NOOP_H);
    console.assert(debugInfo.anchorVerticalOk === true, "correct anchor: vertical sub-check must pass (wrist at/below mouth level)");
    console.assert(debugInfo.anchorBackwardOk === true, "correct anchor: backward sub-check must pass (wrist toward the draw ear)");
    console.assert(debugInfo.anchorOk === true, "correct anchor: combined anchorOk must pass when all three sub-checks do");

    // Hand above the nose: same horizontal position as the mouth, far enough above it to clear
    // FULL_DRAW_ANCHOR_ABOVE_MAX while staying inside the plain distance circle — isolates the
    // failure to the vertical sub-check, proving it does real work beyond what the distance circle
    // alone already covers (the "above the nose" failure the owner reported).
    lastDrawWrist = null;
    const aboveNose = mkLandmarks({ ...base, ...anchorHead, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.4, y: 0.12 } });
    isAtFullDraw(aboveNose, 0, true, NOOP_W, NOOP_H);
    console.assert(debugInfo.anchorDist <= FULL_DRAW_ANCHOR_MAX, "above-nose fixture must stay within the plain distance circle, or this isn't isolating the vertical check");
    console.assert(debugInfo.anchorVerticalOk === false, "above-nose fixture: vertical sub-check must fail");
    console.assert(debugInfo.anchorOk === false, "above-nose fixture: combined anchorOk must fail even though distance alone would have passed");

    // Hand in front of the mouth, AT THE CORRECT DISTANCE: isolates the failure to the backward
    // sub-check — directly the owner's own complaint ("i cannot anchor ahead of my mouth"), and
    // also the "correct distance but wrong side" case from the brief.
    lastDrawWrist = null;
    const inFrontOfMouth = mkLandmarks({ ...base, ...anchorHead, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.34, y: 0.22 } });
    isAtFullDraw(inFrontOfMouth, 0, true, NOOP_W, NOOP_H);
    console.assert(debugInfo.anchorDist <= FULL_DRAW_ANCHOR_MAX, "in-front-of-mouth fixture must stay within the plain distance circle, or this isn't isolating the backward check");
    console.assert(debugInfo.anchorVerticalOk === true, "in-front-of-mouth fixture: vertical sub-check must pass on its own (this fixture isolates backward, not vertical)");
    console.assert(debugInfo.anchorBackwardOk === false, "in-front-of-mouth fixture: backward sub-check must fail");
    console.assert(debugInfo.anchorOk === false, "in-front-of-mouth fixture: combined anchorOk must fail even at the correct distance and height");

    // Ear-visibility fallback, one ear only: the draw-side ear (the preferred one) is occluded —
    // a side-on archer often has one ear or the other hidden from the camera. The backward check
    // must still run off the bow-side ear rather than going silent just because the preferred one
    // is missing.
    lastDrawWrist = null;
    const drawEarOccluded = mkLandmarks({
      ...base, ...anchorHead,
      [R_EAR]: { x: 0.5, y: 0.2, visibility: 0 }, // draw-side ear hidden
      15: { x: 0.0, y: 0.3 }, 16: { x: 0.42, y: 0.22 },
    });
    isAtFullDraw(drawEarOccluded, 0, true, NOOP_W, NOOP_H);
    console.assert(
      debugInfo.anchorBackward !== null,
      "the backward check must still run (fall back to the bow-side ear), not go null, just because the preferred draw-side ear is occluded"
    );

    // Ear-visibility fallback, BOTH hidden: the backward direction genuinely cannot be measured —
    // must degrade to "not confirmed" (null), never guess a pass. anchorOk must stay false even
    // though distance and vertical both pass on their own — an unconfirmed direction is not the
    // same thing as a confirmed one, and the combined check must not treat it as a pass.
    lastDrawWrist = null;
    const bothEarsOccluded = mkLandmarks({
      ...base, ...anchorHead,
      [L_EAR]: { x: 0.3, y: 0.2, visibility: 0 },
      [R_EAR]: { x: 0.5, y: 0.2, visibility: 0 },
      15: { x: 0.0, y: 0.3 }, 16: { x: 0.42, y: 0.22 },
    });
    isAtFullDraw(bothEarsOccluded, 0, true, NOOP_W, NOOP_H);
    console.assert(debugInfo.anchorBackward === null, "with both ears occluded, anchorBackward must stay null — never guess a direction with no ear to measure it from");
    console.assert(debugInfo.anchorBackwardOk === null, "with both ears occluded, anchorBackwardOk must be null (unconfirmed), never true");
    console.assert(
      debugInfo.anchorVerticalOk === true && debugInfo.anchorDist <= FULL_DRAW_ANCHOR_MAX,
      "sanity check: this fixture's distance and vertical position are otherwise a real anchor, isolating the missing-ear case specifically"
    );
    console.assert(debugInfo.anchorOk === false, "with the backward direction unconfirmed, combined anchorOk must not pass on distance and vertical alone");

    // ===== ARM CONE — additional fixtures beyond restingArmsAtSides (the headline hanging-arm
    // case, already asserted above).

    // Properly extended: the existing `drawn` fixture's bow arm (shoulder->elbow->wrist collinear,
    // pointing level out to the side) should measure ~0° elevation and pass both ARM sub-checks.
    console.assert(
      Math.abs(bowArmElevationOf(drawn, NOOP_W, NOOP_H)) < 1e-6,
      "a bow arm extended level with the shoulder should measure ~0° elevation"
    );
    lastDrawWrist = null;
    isAtFullDraw(drawn, 0, true, NOOP_W, NOOP_H);
    isAtFullDraw(drawn, 500, true, NOOP_W, NOOP_H);
    console.assert(
      debugInfo.armStraightOk === true && debugInfo.armConeOk === true && debugInfo.armOk === true,
      "a properly extended, level bow arm must pass both ARM sub-checks"
    );

    // Slightly bent, now passing: bent enough to measure ~146° at the elbow — below the OLD 150°
    // threshold (would have failed) but above the NEW, loosened 140° threshold (must now pass) —
    // while staying level enough to clear the cone easily. Owner's own field report: "sometimes i
    // shoot with my arm not 100% straight and the slight bend doesn't pass the trigger."
    //
    // FIXTURE BUG FOUND THIS TASK (2026-08-24), same audit as `base`'s ear fix and the mkLandmarks
    // default-visibility fix above: this fixture never set a draw wrist (16) at all, so before
    // mkLandmarks's untouched-landmark default was changed to invisible, isAtFullDraw's own
    // initial visibility gate ([drawWrist, bowShoulder, bowElbow, bowWrist].every(visible)) was
    // silently passing off the OLD corner-default (0,0, visibility:1) standing in for the draw
    // wrist — this fixture was never actually exercising isAtFullDraw's arm checks at all until
    // the default-visibility fix made that gate correctly reject it. Given a real draw wrist here,
    // matching `drawn`'s own anchor position, so this is now a genuine full-draw pose with only
    // the bow arm bent, same as the comment above always claimed it to be.
    lastDrawWrist = null;
    const slightBend = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.4 }, 16: { x: 0.52, y: 0.31 } }); // bow wrist dropped; bow elbow stays at base's (0.15, 0.3); draw wrist at a real anchor position
    const slightBendAngle = angleAt(slightBend[L_SHOULDER], slightBend[L_ELBOW], slightBend[L_WRIST]);
    console.assert(
      slightBendAngle > FULL_DRAW_BOW_ARM_MIN && slightBendAngle < 150,
      `slight-bend fixture must land strictly between the new (${FULL_DRAW_BOW_ARM_MIN}°) and old (150°) thresholds to be a fair test, got ${slightBendAngle.toFixed(1)}°`
    );
    isAtFullDraw(slightBend, 0, true, NOOP_W, NOOP_H);
    console.assert(
      debugInfo.armStraightOk === true,
      `a ${slightBendAngle.toFixed(1)}° bend must pass the loosened FULL_DRAW_BOW_ARM_MIN (${FULL_DRAW_BOW_ARM_MIN}°) — it would have failed the old 150° threshold`
    );
    console.assert(debugInfo.armConeOk === true, "the slight-bend fixture's arm should still be level enough to clear the cone");

    // Pure bowArmElevationOf checks: raised overhead (~+90°) and never-guess on a missing wrist.
    const overheadArm = mkLandmarks({ ...base, 15: { x: 0.3, y: 0.0 } }); // bow wrist straight above the bow shoulder
    console.assert(
      Math.abs(bowArmElevationOf(overheadArm, NOOP_W, NOOP_H) - 90) < 1e-6,
      "a bow arm raised straight overhead should measure ~+90° elevation"
    );
    const noWristForElevation = mkLandmarks({ ...base, 15: { x: 0.3, y: 0.0, visibility: 0 } });
    console.assert(
      bowArmElevationOf(noWristForElevation, NOOP_W, NOOP_H) === null,
      "bowArmElevationOf must never guess — null when the bow wrist isn't confidently visible"
    );

    // RAISE / OPEN: a dedicated arm-raised fixture (the earlier `raise` fixture's bow arm points
    // sideways to test hand separation, not upward, so it doesn't exercise bowArmRaiseHeight).
    raiseArmed = false;
    lastDrawWrist = null;
    const armRaised = mkLandmarks({
      ...base,
      13: { x: 0.3, y: 0.1 }, // bow elbow, above the bow shoulder
      15: { x: 0.3, y: 0.0 }, // bow wrist, well above the bow shoulder (y=0.3) — comfortably past shoulder height
      16: { x: 0.5, y: 0.6 }, // draw wrist stays down at rest — this is a raise, not a draw
    });
    const raisedHeight = bowArmRaiseHeight(armRaised, NOOP_W, NOOP_H);
    console.assert(
      raisedHeight >= RAISE_TRIGGER_UP_FRACTION,
      `arm-raised fixture must itself clear RAISE_TRIGGER_UP_FRACTION for this to be a fair test — got ${raisedHeight}`
    );
    updateRaiseTrigger(armRaised, NOOP_W, NOOP_H, 0);
    console.assert(ttById.raise.read(armRaised).lamp === true, "RAISE screen must read lit once the bow wrist clears shoulder height");
    attempt = null;
    trackShotAttempt({ handSep: null, raiseArmed, atFullDraw: false, eligible: false }, 0);
    console.assert(ttById.open.read(armRaised).lamp === true, "OPEN screen must read lit once the raise alone has opened an attempt");

    const armDown = mkLandmarks({ ...base, 13: { x: 0.3, y: 0.45 }, 15: { x: 0.3, y: 0.58 }, 16: { x: 0.5, y: 0.58 } });
    updateRaiseTrigger(armDown, NOOP_W, NOOP_H, 1000);
    console.assert(ttById.raise.read(armDown).lamp === false, "RAISE screen must read dark once the bow wrist drops back down well below shoulder height");
    trackShotAttempt({ handSep: 0, raiseArmed, atFullDraw: false, eligible: false }, 1000);
    console.assert(ttById.open.read(armDown).lamp === false, "OPEN screen must read dark once the attempt has closed (hands relaxed, raise cleared)");

    // ELIGIBLE: resetSettling() must show 0 settled frames and a dark (unlit) lamp; enough
    // consecutive advanceSettling() calls must clear the frame-count gate and light the momentary
    // ELIGIBLE lamp — cropBoxStableThisFrame passed as true throughout so this isolates the frame-
    // count gate specifically, regardless of ROI_CROPPING_ENABLED's real value (advanceSettling
    // ignores that argument entirely when cropping is off).
    resetSettling();
    console.assert(ttById.eligible.read(null).rows[0].value === "0", "ELIGIBLE's settled-frame count must read 0 right after a reset");
    console.assert(ttById.eligible.read(null).lamp === false, "ELIGIBLE lamp must be dark right after a reset — no frame has been eligible yet");
    let eligibleNow = false;
    for (let i = 0; i < SETTLE_FRAMES_REQUIRED; i++) eligibleNow = advanceSettling(true, true);
    console.assert(eligibleNow === true, "advanceSettling must itself return true once enough frames have passed, or this isn't a fair test of ELIGIBLE");
    debugEvents.frameEligible = performance.now();
    console.assert(ttById.eligible.read(null).lamp === true, "ELIGIBLE screen must read lit the instant a frame clears the settling gate");

    // POSE: a direct, honest reflection of lastPoseSeen — set here exactly as renderLoop's own two
    // branches set it, not recomputed independently.
    lastPoseSeen = true;
    console.assert(ttById.pose.read(drawn).lamp === true, "POSE screen must read lit when lastPoseSeen is true");
    console.assert(ttById.pose.read(drawn).inputs.length === 15, "POSE screen must list all 15 named landmarks this app tracks");
    lastPoseSeen = false;
    console.assert(ttById.pose.read(null).lamp === false, "POSE screen must read dark when lastPoseSeen is false");

    // ATTN: attentionIsClearlyCalm decides calm/not-calm per sample; updateAttentionState's state
    // machine decides engaged/idle from a HELD stretch of that signal. Drive both directly with a
    // calm, motionless fixture and confirm the ATTN screen only goes dark (idle) once the hold has
    // actually lasted ATTENTION_IDLE_AFTER_MS, and instantly back to lit (engaged) the moment a
    // single sample stops being calm — the fail-toward-recording asymmetry CLAUDE.md describes.
    attentionEngaged = true;
    attentionCalmSinceMs = null;
    attentionPrevRef = null;
    attentionLastEvalMs = null;
    attempt = null;
    const calmFixture = mkLandmarks({ ...base, 15: { x: 0.3, y: 0.58 }, 16: { x: 0.32, y: 0.58 } }); // hands relaxed together, well under ATTENTION_REST_HAND_SEP_MAX
    updateAttentionState(0, calmFixture, NOOP_W, NOOP_H, true, true);
    console.assert(ttById.attn.read(calmFixture).lamp === true, "ATTN must still read lit (engaged) on the very first calm sample — one sample is never enough to idle");
    updateAttentionState(ATTENTION_IDLE_AFTER_MS + 1, calmFixture, NOOP_W, NOOP_H, true, true);
    console.assert(ttById.attn.read(calmFixture).lamp === false, "ATTN must read dark (idle) once the calm condition has held continuously for ATTENTION_IDLE_AFTER_MS");
    const movingFixture = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.52, y: 0.31 } }); // hands apart — not calm
    updateAttentionState(ATTENTION_IDLE_AFTER_MS + 100, movingFixture, NOOP_W, NOOP_H, true, true);
    console.assert(ttById.attn.read(movingFixture).lamp === true, "ATTN must read lit (engaged) again the instant a single sample stops being calm — no confirmation delay");
  }

  // --- ?triggertest DOM RESIDUE: found in review. `renderCurrentTrigger` sets tt-note/tt-reason
  // text only inside the `if (has one)` branch and only cleared DISPLAY (not text) in the `else`
  // — so a note/reason from a PREVIOUS screen stayed sitting in the DOM (invisible via
  // display:none, but still readable as text) until a later screen overwrote it. The bug is
  // ORDER-DEPENDENT: it never shows up checking one screen fresh off a page load, only checking a
  // note-less screen AFTER a screen that had one — so that's exactly what this checks. Only
  // meaningful once the real panel exists (buildTriggerTestPanel/ttRefs — see TRIGGERTEST's own
  // comment for why this needs the flag itself, not just DEBUG).
  if (TRIGGERTEST) {
    const savedTtIndex = ttIndex;
    const sepIndex = TRIGGER_DEFS.findIndex((d) => d.id === "sep"); // has a note
    const stillIndex = TRIGGER_DEFS.findIndex((d) => d.id === "still"); // note: null
    ttIndex = sepIndex;
    renderCurrentTrigger();
    console.assert(
      ttRefs["tt-note"].textContent.includes("Known issue"),
      "sanity check: SEP's own note must actually render, or the ordering check below proves nothing"
    );
    ttIndex = stillIndex;
    renderCurrentTrigger();
    console.assert(
      ttRefs["tt-note"].textContent === "",
      `a screen with note:null must leave tt-note's TEXT empty, not just hidden, once a PREVIOUS screen has set one — got ${JSON.stringify(ttRefs["tt-note"].textContent)}`
    );

    // Same class of bug, same fix, for the reason line: drive isAtFullDraw into a bail state (a
    // real reason) on one screen, then switch to a screen whose `read()` never returns a `reason`
    // key at all (raise) and confirm the text — not just the display — actually clears.
    lastDrawWrist = null;
    const missingForReason = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3, visibility: 0 }, 16: { x: 0.52, y: 0.31 } });
    isAtFullDraw(missingForReason, 0, true, NOOP_W, NOOP_H);
    const anchorIndex = TRIGGER_DEFS.findIndex((d) => d.id === "anchor");
    ttIndex = anchorIndex;
    renderCurrentTrigger();
    console.assert(
      ttRefs["tt-reason"].textContent.length > 0,
      "sanity check: a real bail reason must actually render, or the ordering check below proves nothing"
    );
    const raiseIndex = TRIGGER_DEFS.findIndex((d) => d.id === "raise"); // read() never sets `reason`
    ttIndex = raiseIndex;
    renderCurrentTrigger();
    console.assert(
      ttRefs["tt-reason"].textContent === "",
      `a screen whose read() carries no reason must leave tt-reason's TEXT empty, not just hidden, once a PREVIOUS screen has set one — got ${JSON.stringify(ttRefs["tt-reason"].textContent)}`
    );

    ttIndex = savedTtIndex;
    renderCurrentTrigger();
  }

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
  fullDrawShotCount = 0;
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
  console.assert(cueEl.className === "cue-rejected", "endAttempt's rejected path should flash the cue-rejected cue (HANDOVER.md Stage 2: seen but rejected)");

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
  // Green means an arrow, not "a row got added" — see endAttempt's own comment at its
  // signalOutcome call. A logged-but-short-of-full-draw row is real and stays in the log (see the
  // two assertions above), but must never flash the confirmed-arrow cue: that would be telling the
  // owner, at the one moment he can't check it, that an arrow was shot when the app never actually
  // saw one.
  console.assert(cueEl.className === "cue-rejected", "a logged row that never reached full draw must flash the same 'seen, not confirmed' cue as a rejected attempt, not the confirmed-arrow cue");
  console.assert(fullDrawShotCount === 0, "a row short of full draw must not count toward fullDrawShotCount (the 'arrows' headline)");

  // --- An attempt that DOES reach full draw (atFullDraw: true on its peak frame): logged, NOT
  // marked as short of full draw, DOES flash the confirmed-arrow cue, and DOES count as an arrow.
  trackShotAttempt(sample(0.4), 4000);
  trackShotAttempt(sample(0.8, true), 4400); // peak, and a genuine full draw (all four isAtFullDraw gates true)
  trackShotAttempt(sample(0.78, true), 4800); // held at full draw
  trackShotAttempt(sample(0.05), 5200); // ends
  console.assert(log.length === 2, "a full-draw attempt should log as its own row");
  console.assert(log[0].shotNum === 2, "second logged attempt should be shot 2");
  console.assert(log[0].reachedFullDraw === true, "an attempt that reached full draw must not be marked as short of it");
  console.assert(cueEl.className === "cue-logged", "a genuine full-draw attempt must still flash the confirmed-arrow cue");
  console.assert(fullDrawShotCount === 1, "a genuine full draw must count toward fullDrawShotCount");

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

  // --- RAISE TRIGGER: bowArmRaiseHeight (pure geometry) and updateRaiseTrigger's hysteresis.
  // Reuses the `base` fixture's torso (bow shoulder (0.3,0.3), bow hip (0.3,0.6) — torso length
  // 0.3) so a wrist AT shoulder height is y=0.3, and RAISE_TRIGGER_DOWN_FRACTION's -0.3 lands
  // exactly at wrist y=0.39 (0.3 + 0.3*0.3).
  {
    const armLm = (wristY) => mkLandmarks({ ...base, 15: { x: 0.3, y: wristY } });
    console.assert(
      Math.abs(bowArmRaiseHeight(armLm(0.6), NOOP_W, NOOP_H) - -1) < 1e-9,
      "wrist at hip level (arm hanging at rest) should read a full torso-length below the shoulder"
    );
    console.assert(
      Math.abs(bowArmRaiseHeight(armLm(0.3), NOOP_W, NOOP_H) - 0) < 1e-9,
      "wrist exactly at shoulder height should read 0 — RAISE_TRIGGER_UP_FRACTION's own boundary"
    );
    console.assert(bowArmRaiseHeight(armLm(0.2), NOOP_W, NOOP_H) > 0, "wrist above the shoulder should read positive");
    console.assert(
      bowArmRaiseHeight(mkLandmarks({ ...base, 15: { x: 0.3, y: 0.3, visibility: 0 } }), NOOP_W, NOOP_H) === null,
      "an unreadable bow wrist must never guess a height"
    );

    raiseArmed = false;
    updateRaiseTrigger(armLm(0.6), NOOP_W, NOOP_H); // arm hanging at rest, well below the trigger
    console.assert(raiseArmed === false, "an arm hanging at rest must not arm the raise trigger");
    updateRaiseTrigger(armLm(0.3), NOOP_W, NOOP_H); // crosses RAISE_TRIGGER_UP_FRACTION (0) going up
    console.assert(raiseArmed === true, "the bow wrist reaching shoulder height should arm the raise trigger");
    updateRaiseTrigger(armLm(0.36), NOOP_W, NOOP_H); // height -0.2 — below shoulder, but NOT past RAISE_TRIGGER_DOWN_FRACTION (-0.3)
    console.assert(raiseArmed === true, "hysteresis: dipping below shoulder height without clearing RAISE_TRIGGER_DOWN_FRACTION must not disarm");
    updateRaiseTrigger(armLm(0.39), NOOP_W, NOOP_H); // height exactly -0.3 — at the down threshold
    console.assert(raiseArmed === false, "dropping to RAISE_TRIGGER_DOWN_FRACTION should disarm the trigger");
    updateRaiseTrigger(mkLandmarks({ ...base, 15: { x: 0.3, y: 0.3, visibility: 0 } }), NOOP_W, NOOP_H); // unreadable
    console.assert(raiseArmed === false, "an unreadable frame must hold the previous state, never guess a re-arm");
  }

  // --- RAISE TRIGGER, integration with trackShotAttempt/endAttempt. The owner's own proposal
  // (see RAISE TRIGGER constants above): the raise opens an attempt EARLIER than hand separation
  // alone would, but must never change what a logged shot's numbers are measured from.
  {
    attempt = null;
    log = [];
    shotCount = 0;
    fullDrawShotCount = 0;
    rejectedAttemptCount = 0;
    unsettledAttemptCount = 0;
    const raiseSample = (handSep, extra = {}) => sample(handSep, false, true, { raiseArmed: true, ...extra });

    // The raise alone opens a "watching" attempt, before hands have separated at all — and must
    // NOT start the SHOT_MIN_DURATION_MS clock (startMs stays null) until hands actually do.
    trackShotAttempt(raiseSample(0.05), 0);
    console.assert(attempt !== null, "the raise alone should open an attempt, before hand separation ever crosses DRAW_ATTEMPT_MIN_SEP");
    console.assert(attempt.startMs === null, "a raise with hands still together must not start the real-draw duration clock");

    // He changes his mind and lowers the bow without ever drawing: must close as a REJECTED row
    // (peakHandSep never left 0), never a stuck "watching" state.
    trackShotAttempt(sample(0.05, false, true, { raiseArmed: false }), 200);
    console.assert(attempt === null, "a raise that never turns into a draw must close, not hang open");
    console.assert(rejectedAttemptCount === 1, "an abandoned raise should be counted as a rejected attempt, same as any other non-draw");

    // A real raise-then-draw: the raise-phase frames (hands still together, arm just going up)
    // must never contribute to the shot's logged numbers — only once hand separation itself
    // crosses DRAW_ATTEMPT_MIN_SEP should eligible frames start counting. Deliberately wrong
    // bowArmAngle values on the raise-phase frames (20°, 25° — nowhere near a real 178° full
    // draw) so contamination would be obvious if this regressed.
    trackShotAttempt(raiseSample(0.05, { bowArmAngle: 20 }), 1000); // raised, hands together, mid-raise
    trackShotAttempt(raiseSample(0.15, { bowArmAngle: 25 }), 1100); // still short of DRAW_ATTEMPT_MIN_SEP (0.3)
    trackShotAttempt(raiseSample(0.8, { bowArmAngle: 178, atFullDraw: true }), 1700); // now genuinely drawing
    trackShotAttempt(sample(0.05, false, true, { raiseArmed: false }), 2400); // ends — 700ms of real draw, over SHOT_MIN_DURATION_MS
    console.assert(shotCount === 1, "a real raise-then-draw should log exactly one shot");
    console.assert(
      log[0].bowArmAngle === 178,
      `the logged bow-arm angle must come only from the real full-draw frame (178), not the raise-phase frames (20/25) — got ${log[0]?.bowArmAngle}`
    );
    console.assert(log[0].startMs === 1700, "duration must be measured from when hands actually separated, not from the raise");

    // A raise that's held indefinitely without ever drawing or lowering must not watch forever —
    // RAISE_ATTEMPT_TIMEOUT_MS closes it as rejected.
    attempt = null;
    rejectedAttemptCount = 0;
    trackShotAttempt(raiseSample(0.05), 5000);
    console.assert(attempt !== null, "setup: the raise should still be open just before the timeout");
    trackShotAttempt(raiseSample(0.05), 5000 + RAISE_ATTEMPT_TIMEOUT_MS - 100);
    console.assert(attempt !== null, "an armed raise under RAISE_ATTEMPT_TIMEOUT_MS old must still be open");
    trackShotAttempt(raiseSample(0.05), 5000 + RAISE_ATTEMPT_TIMEOUT_MS + 100);
    console.assert(attempt === null, "an armed raise older than RAISE_ATTEMPT_TIMEOUT_MS that never drew must close on its own");
    console.assert(rejectedAttemptCount === 1, "a timed-out raise should be counted as a rejected attempt");

    // Fallback, unaffected: with the raise never armed (occlusion, say), hand separation alone
    // must still open, measure and log an attempt exactly as it did before this feature existed.
    attempt = null;
    log = [];
    shotCount = 0;
    fullDrawShotCount = 0;
    trackShotAttempt(sample(0.4, false, true, { raiseArmed: false, bowArmAngle: 179 }), 8000);
    trackShotAttempt(sample(0.8, true, true, { raiseArmed: false, bowArmAngle: 179 }), 8400);
    trackShotAttempt(sample(0.05, false, true, { raiseArmed: false }), 9000);
    console.assert(shotCount === 1, "hand separation alone (raise missed) must still be able to open and log a shot, unchanged");
    console.assert(log[0].bowArmAngle === 179, "a fallback-only shot's numbers must be exactly what the pre-raise-trigger logic would have produced");

    attempt = null;
    log = [];
    shotCount = 0;
    fullDrawShotCount = 0;
    rejectedAttemptCount = 0;
    unsettledAttemptCount = 0;
  }

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
  fullDrawShotCount = 0;
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
  console.assert(cueEl.className === "cue-rejected", "an unsettled attempt is still \"seen but rejected\" from the owner's point of view — same cue as a genuinely rejected one, not a fifth state");

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

    // A row that never reached full draw must be invisible to the consistency machinery —
    // renderShotLog/buildShareText both filter to reachedFullDraw entries (fullDrawLog/
    // fullDrawEntries) before ever calling summarizeShots/narrateMeasure. Proven directly: a
    // wildly different short-of-full-draw reading must neither move the average nor get named as
    // the outlier once excluded — and the sanity check right after confirms it WOULD have done
    // both if it hadn't been.
    // Reuses the exact fixture shape already proven above to reliably trigger the outlier path
    // (outlierNarrFixture: four consistent shots, one wildly apart) — shot 5's 200 becomes a
    // short-of-full-draw row here instead of a real one.
    const exclusionFixture = [
      { shotNum: 1, bowArmAngle: 170, reachedFullDraw: true },
      { shotNum: 2, bowArmAngle: 169, reachedFullDraw: true },
      { shotNum: 3, bowArmAngle: 171, reachedFullDraw: true },
      { shotNum: 4, bowArmAngle: 170, reachedFullDraw: true },
      { shotNum: 5, bowArmAngle: 200, reachedFullDraw: false }, // a short draw, not a real reading of steady form
    ];
    const fullDrawOnly = exclusionFixture.filter((e) => e.reachedFullDraw);
    const excludedStats = summarizeShots(fullDrawOnly);
    console.assert(
      !(5 in excludedStats.bowArm.deviations),
      "a row excluded from the consistency population must not appear in stats.deviations — shotValueHtml relies on exactly this to render it as a plain, uncoloured number instead of a fake comparison"
    );
    console.assert(
      Math.abs(excludedStats.bowArm.average - 170) < 1e-9,
      `excluding the short-draw row (200) must leave the average computed from the four real full draws only (170), got ${excludedStats.bowArm.average}`
    );
    const excludedNarr = narrateMeasure(fullDrawOnly, (e) => e.bowArmAngle, "Bow arm", upDown, BOW_ARM_CONSISTENCY_FLOOR_DEG);
    console.assert(
      excludedNarr && excludedNarr.text.startsWith("Bow arm — steady") && excludedNarr.outlierShotNum === null,
      `four consistent full draws, short-draw row excluded, should read steady with no outlier named, got: ${excludedNarr && JSON.stringify(excludedNarr)}`
    );
    const includedNarr = narrateMeasure(exclusionFixture, (e) => e.bowArmAngle, "Bow arm", upDown, BOW_ARM_CONSISTENCY_FLOOR_DEG);
    console.assert(
      includedNarr && includedNarr.outlierShotNum === 5,
      `sanity check failed: shot 5's 200° should stand out when included — got: ${includedNarr && JSON.stringify(includedNarr)}, so the exclusion test above isn't proving anything`
    );
  }

  // --- Share text: buildShareText is the one pure function behind the Share button (see its own
  // comment) — entries + counters in, one string out, so it's checked directly here exactly like
  // summarizeShots/narrateMeasure above, no DOM or navigator involved.
  {
    const shareEntries = [
      {
        shotNum: 3,
        bowArmAngle: 172,
        shoulderDrop: { bow: 46, draw: 44 },
        elbowAlign: { signed: 2, deviation: 2, direction: "high" },
        handSep: 0.81,
        anchorOk: true, armOk: true, sepOk: true, stillOk: true,
        reachedFullDraw: true,
        clipUrl: "blob:fake-3",
      },
      // Deliberately the messiest of the three: an uncertain (below-MIN_VISIBILITY) bow-arm and
      // draw-shoulder reading, a failed trigger condition, a short-of-full-draw draw, AND a
      // failed clip — proves null-handling, trigger pass/fail, fullDraw, and clip failure all
      // come through honestly on the same row rather than only being exercised in isolation.
      {
        shotNum: 2,
        bowArmAngle: null,
        shoulderDrop: { bow: 45, draw: null },
        elbowAlign: null,
        handSep: 0.62,
        anchorOk: true, armOk: false, sepOk: true, stillOk: true,
        reachedFullDraw: false,
        clipFailReason: "no clip — recording came out empty",
      },
      {
        shotNum: 1,
        bowArmAngle: 170,
        shoulderDrop: { bow: 44, draw: 45 },
        elbowAlign: { signed: -1, deviation: 1, direction: "low" },
        handSep: 0.79,
        anchorOk: true, armOk: true, sepOk: true, stillOk: true,
        reachedFullDraw: true,
        clipUrl: "blob:fake-1",
      },
    ];
    const shareCounters = {
      shotCount: 3,
      fullDrawShotCount: 2, // shots 1 and 3 reached full draw; shot 2 (armOk: false) did not — see shareEntries above
      rejectedAttemptCount: 1,
      unsettledAttemptCount: 0,
      attentionIdlePeriods: 0,
      attentionLateWakeCount: 0,
      clipsUnavailableReason: null,
      modelStatusLine: "Pose model: full — pose detection took about 12.3ms/frame, 47.1 fps actually rendered, measured at startup.",
      rightHanded: true,
      mirrored: false,
      cameraWidth: 720,
      cameraHeight: 1280,
    };
    const shareText = buildShareText(shareEntries, shareCounters);

    console.assert(shareText.includes("Arrows this session: 2"), "share text's arrow count must be confirmed full draws only (shots 1 and 3), not every logged row");
    console.assert(shareText.includes("Logged draws this session (arrows + short-of-full-draw): 3"), "share text must still state the true total row count somewhere, separate from the arrow count");
    console.assert(shareText.includes("Movements ignored (not real draws): 1"), "share text must include rejectedAttemptCount even though a rejected movement never becomes its own row");
    console.assert(shareText.includes("Camera 720x1280"), "share text header must carry the camera resolution");
    console.assert(shareText.includes("right-handed") && shareText.includes("mirror off"), "share text header must carry handedness and mirror state");

    // One labelled line per draw, in stable shot=N order, oldest first — chronological even
    // though the input array (like the real log) was newest-first.
    const drawLines = shareText.split("\n").filter((l) => l.startsWith("shot="));
    console.assert(drawLines.length === 3, `share text must include exactly one line per logged draw, got ${drawLines.length}`);
    console.assert(
      drawLines[0].startsWith("shot=1") && drawLines[1].startsWith("shot=2") && drawLines[2].startsWith("shot=3"),
      `draw lines must be chronological (oldest first) regardless of input order, got: ${drawLines.map((l) => l.slice(0, 8)).join(", ")}`
    );

    // Nulls/uncertain readings must be represented honestly, never as a fake 0 that would look
    // like a real (if unusually low) measurement to whoever eventually tunes the thresholds.
    console.assert(drawLines[1].includes("bowArm=uncertain"), "a null bow-arm reading must render as 'uncertain', not a fake number");
    console.assert(drawLines[1].includes("shoulderDraw=uncertain"), "a null draw-shoulder reading must render as 'uncertain', not a fake number");
    console.assert(!drawLines[1].includes("bowArm=0") && !drawLines[1].includes("shoulderDraw=0"), "a null reading must never render as a fake zero");

    // Every trigger condition's pass/fail must come through, per shot — this is the whole point:
    // the owner's only route to real hand-sep/trigger figures without remembering ?debug.
    console.assert(drawLines[1].includes("arm=fail"), "a failed trigger condition must show as fail on its shot's line");
    console.assert(drawLines[0].includes("arm=pass") && drawLines[0].includes("still=pass"), "passing trigger conditions must show as pass");
    console.assert(drawLines[1].includes("handSep=0.620"), "hand separation must be included per shot, at real precision, for threshold retuning");

    // Short-of-full-draw and clip status/failure reason must both be visible on the row.
    console.assert(drawLines[1].includes("fullDraw=no"), "a draw that fell short of full draw must say so on its own line");
    console.assert(drawLines[0].includes("fullDraw=yes") && drawLines[0].includes("recorded=yes"), "a full draw with a clip must say so on its own line");
    console.assert(
      drawLines[1].includes("recorded=no") && drawLines[1].includes('clipFailReason="no clip — recording came out empty"'),
      "a failed clip must show recorded=no plus its stated reason on its own shot's line"
    );
    console.assert(shareText.includes('Shot 2: no clip — recording came out empty'), "a clip failure must also appear in the session-level Clip failures section");

    // Consistency lines must be present and must reuse narrateMeasure's own wording, not a
    // second, possibly-diverging implementation — computed from the SAME reachedFullDraw-only
    // population buildShareText itself must use (see the excludedNote assertion right below).
    console.assert(shareText.includes("Consistency:"), "share text must include the consistency section");
    console.assert(
      shareText.includes("2 of 3 shown draws that reached full draw — 1 excluded as short of full draw"),
      "share text must say how many shown draws were excluded from the consistency numbers, and why"
    );
    const fullDrawShareEntries = shareEntries.filter((e) => e.reachedFullDraw);
    const bowArmShareLine = narrateMeasure(fullDrawShareEntries, (e) => e.bowArmAngle, "Bow arm", wordForBowArm, BOW_ARM_CONSISTENCY_FLOOR_DEG);
    console.assert(
      bowArmShareLine && shareText.includes(bowArmShareLine.text),
      "share text's consistency section must match narrateMeasure's own wording exactly, not a re-derived copy"
    );

    // The truncation notice: only 2 of these 3 entries "survive" as if the log had capped them —
    // simulates a 30-arrow session sharing only its last few draws. Must say so explicitly rather
    // than silently presenting the partial sample as the whole session.
    const truncatedText = buildShareText(shareEntries.slice(0, 2), { ...shareCounters, shotCount: 30 });
    console.assert(
      truncatedText.includes("only the most recent 2 of 30 draws are included"),
      "share text must explicitly say when fewer draws are included than actually happened"
    );
    const completeText = buildShareText(shareEntries, { ...shareCounters, shotCount: 3 });
    console.assert(
      !completeText.includes("only the most recent"),
      "share text must NOT show a truncation notice when every draw that happened is included"
    );

    // Empty session: no shots yet, must say so plainly rather than rendering an empty or broken
    // draws section.
    const emptyShareText = buildShareText([], { ...shareCounters, shotCount: 0, fullDrawShotCount: 0, rejectedAttemptCount: 0 });
    console.assert(emptyShareText.includes("Arrows this session: 0"), "an empty session must still state the (zero) arrow count");
    console.assert(emptyShareText.includes("(none)"), "an empty session's draws section must say plainly that there are none, not render blank");

    // shareCounters above (an existing fixture, predating calibration) omits every calibration
    // field on purpose — buildShareText must still include a calibration line via its own
    // defaults, never throw, and never claim more than "never calibrated" for a caller that never
    // supplied anything.
    console.assert(shareText.includes("Calibration:") && shareText.includes("never calibrated"), "buildShareText must include a calibration line even when the caller supplies no calibration fields at all, defaulting honestly to 'never calibrated'");
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
    fullDrawShotCount = 0;
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

      // ROUTINE-START ATTENTION GATING's own hand-separation signal shares this same wrist pair
      // and the same aspect-ratio hazard (see handSeparationForAttention/attentionScale) — proven
      // here directly against the same ground truth, in both orientations, rather than trusted to
      // follow along just because it's built the same way.
      const attnHandSep = handSeparationForAttention(lm, w, h);
      console.assert(
        attnHandSep !== null && Math.abs(attnHandSep - trueHandSepRatio) < TOL,
        `${label}: handSeparationForAttention should match the real-world ground truth (${trueHandSepRatio.toFixed(6)}), got ${attnHandSep}`
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

  // --- canvasShouldMirrorViaCss: the #overlay canvas gets the CSS mirror flip only when the
  // picture should look mirrored AND nothing is currently baking a pixel-level mirror into the
  // canvas itself (see withMirror/paintCanvas) — stacking both would flip an already-mirrored
  // recording's picture back to unmirrored on screen. Full truth table, since this is exactly the
  // kind of two-boolean interaction a single missed case would silently get backwards.
  console.assert(
    canvasShouldMirrorViaCss(true, false) === true,
    "mirrored + not recording: canvas should get the CSS flip"
  );
  console.assert(
    canvasShouldMirrorViaCss(true, true) === false,
    "mirrored + recording: canvas pixels are already mirrored (withMirror) — CSS flip must be withheld, or the picture flips back to unmirrored"
  );
  console.assert(
    canvasShouldMirrorViaCss(false, false) === false,
    "not mirrored + not recording: no CSS flip needed"
  );
  console.assert(
    canvasShouldMirrorViaCss(false, true) === false,
    "not mirrored + recording: no CSS flip either way"
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
      attentionIsClearlyCalm(null, null, 0, NOOP_W, NOOP_H) === true,
      "no landmarks at all (nobody in frame) should read as clearly calm — nothing to be shooting"
    );
    console.assert(
      attentionIsClearlyCalm(restLandmarks, null, 0, NOOP_W, NOOP_H) === true,
      "relaxed hands, with no previous reference point to judge stillness against yet, should read as clearly calm"
    );

    const drawnLandmarks = mkLandmarks({ ...base, 15: { x: 0.0, y: 0.3 }, 16: { x: 0.52, y: 0.31 } });
    console.assert(
      attentionIsClearlyCalm(drawnLandmarks, null, 0, NOOP_W, NOOP_H) === false,
      "hands far apart (a real draw) must never read as clearly calm"
    );

    const noWristLandmarks = mkLandmarks({ ...base, 15: { x: 0.4, y: 0.3, visibility: 0 }, 16: { x: 0.42, y: 0.3 } });
    console.assert(
      attentionIsClearlyCalm(noWristLandmarks, null, 0, NOOP_W, NOOP_H) === false,
      "an invisible wrist can't confirm the hands are relaxed — must not read as clearly calm"
    );

    const calmRef = bodyReferencePoint(restLandmarks);
    console.assert(
      attentionIsClearlyCalm(restLandmarks, calmRef, 1, NOOP_W, NOOP_H) === true,
      "an unmoved body reference point over a full second should still read as clearly calm"
    );
    const farRef = { x: calmRef.x + 1, y: calmRef.y }; // ~3+ torso-lengths of drift in one second — unmistakably walking
    console.assert(
      attentionIsClearlyCalm(restLandmarks, farRef, 1, NOOP_W, NOOP_H) === false,
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
    const callAttention = (nowMs, lm, gatingEnabled = true) => updateAttentionState(nowMs, lm, NOOP_W, NOOP_H, gatingEnabled, true);

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
    // Computed via the real production function (not a hand-rolled reimplementation) so this
    // check can never drift out of sync with what attentionIsClearlyCalm itself actually measures.
    const ambigHandSep = handSeparationForAttention(ambiguousLm, NOOP_W, NOOP_H);
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

  // --- CALIBRATION (HANDOVER.md Stage 4): body-proportion measurement, the physical-plausibility
  // check it enables, the stored-vs-fresh comparison, and the optional framing-signature add-on.
  // All pure functions — no module state touched, so nothing here needs saving/restoring.
  {
    const near = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;

    // A simple standing body, arms hanging at the sides, facing roughly toward the camera (NOT
    // side-on — calibration happens before he's turned to shoot). Round numbers on purpose:
    // torso 0.3, each arm's two segments 0.15+0.15=0.3 (armToTorso 1.0), shoulder width 0.3
    // (shoulderToTorso 1.0), nose-to-ear 0.06 (headToTorso 0.2).
    const standingBody = mkLandmarks({
      0: { x: 0.5, y: 0.1 }, // nose
      7: { x: 0.44, y: 0.1 }, // left ear
      11: { x: 0.35, y: 0.3 }, // left shoulder
      12: { x: 0.65, y: 0.3 }, // right shoulder
      13: { x: 0.35, y: 0.45 }, // left elbow
      14: { x: 0.65, y: 0.45 }, // right elbow
      15: { x: 0.35, y: 0.6 }, // left wrist
      16: { x: 0.65, y: 0.6 }, // right wrist
      23: { x: 0.35, y: 0.6 }, // left hip
      24: { x: 0.65, y: 0.6 }, // right hip
      27: { x: 0.35, y: 0.9 }, // left ankle
      28: { x: 0.65, y: 0.9 }, // right ankle
    });

    const proportions = bodyProportionsOf(standingBody, NOOP_W, NOOP_H);
    console.assert(proportions !== null, "a fully-visible standing body should produce a calibration reading");
    console.assert(near(proportions.armToTorso, 1.0), `armToTorso should read 1.0 for this fixture, got ${proportions?.armToTorso}`);
    console.assert(near(proportions.shoulderToTorso, 1.0), `shoulderToTorso should read 1.0 for this fixture, got ${proportions?.shoulderToTorso}`);
    console.assert(near(proportions.headToTorso, 0.2), `headToTorso should read 0.2 for this fixture, got ${proportions?.headToTorso}`);

    // --- Scale invariance: THE property this whole feature rests on (see bodyProportionsOf's own
    // comment). Same body, standing twice as far from the camera — every landmark scaled by 0.5
    // toward the frame centre, exactly like walking backward — must produce the SAME ratios, not
    // smaller ones.
    const scaleToward = (pt, c, k) => ({ x: c.x + k * (pt.x - c.x), y: c.y + k * (pt.y - c.y) });
    const center = { x: 0.5, y: 0.5 };
    const farOverrides = {};
    for (const i of [0, 7, 11, 12, 13, 14, 15, 16, 23, 24, 27, 28]) farOverrides[i] = scaleToward(standingBody[i], center, 0.5);
    const standingBodyFar = mkLandmarks(farOverrides);
    const proportionsFar = bodyProportionsOf(standingBodyFar, NOOP_W, NOOP_H);
    console.assert(proportionsFar !== null, "the same body, standing closer/farther, should still produce a calibration reading");
    console.assert(
      near(proportions.armToTorso, proportionsFar.armToTorso) &&
        near(proportions.shoulderToTorso, proportionsFar.shoulderToTorso) &&
        near(proportions.headToTorso, proportionsFar.headToTorso),
      "body proportions must be scale-invariant — the same body at a different apparent size must read the same ratios (the property the whole calibration feature rests on)"
    );

    // --- Missing/low-confidence landmarks must produce NO calibration, never a bad one.
    const noWrists = mkLandmarks({
      ...standingBody,
      15: { ...standingBody[15], visibility: 0 },
      16: { ...standingBody[16], visibility: 0 },
    });
    console.assert(bodyProportionsOf(noWrists, NOOP_W, NOOP_H) === null, "invisible wrists on both sides (no arm can be measured) must produce no calibration at all");

    const sideOnShoulders = mkLandmarks({ ...standingBody, 11: { x: 0.5, y: 0.3 }, 12: { x: 0.5, y: 0.3 } });
    console.assert(
      bodyProportionsOf(sideOnShoulders, NOOP_W, NOOP_H) === null,
      "shoulders projecting on top of each other (side-on) must produce no calibration — a near-zero shoulder width is a projection artefact, not a real reading (see MIN_SHOULDER_TO_TORSO)"
    );

    // --- medianCalibrationOf: too few good frames must produce no calibration at all.
    console.assert(
      medianCalibrationOf([proportions, proportions, proportions]) === null,
      "fewer than CALIBRATION_MIN_SAMPLES good frames must produce no calibration, however consistent they are"
    );
    const enoughSamples = Array.from({ length: CALIBRATION_MIN_SAMPLES }, () => proportions);
    const calibrationFromMedian = medianCalibrationOf(enoughSamples);
    console.assert(
      calibrationFromMedian &&
        near(calibrationFromMedian.armToTorso, 1.0) &&
        near(calibrationFromMedian.shoulderToTorso, 1.0) &&
        near(calibrationFromMedian.headToTorso, 0.2),
      "enough identical good frames should produce exactly their own values as the calibration"
    );

    // --- handSepIsPlausible: the triangle-inequality reach bound. maxPlausibleHandSep here is
    // 1.15 x (2x1.0 + 1.0) = 3.45.
    console.assert(handSepIsPlausible(1.0, calibrationFromMedian) === true, "a typical full-draw hand-separation reading (~1 torso-length) must not be flagged");
    console.assert(handSepIsPlausible(4.0, calibrationFromMedian) === false, "a hand-separation reading beyond what this archer's own calibrated reach could ever produce must be flagged");
    console.assert(handSepIsPlausible(null, calibrationFromMedian) === null, "no hand-sep reading this shot (uncertain) must give no verdict, never a false 'fine'");
    console.assert(handSepIsPlausible(1.0, null) === null, "no calibration yet must give no verdict, never a false 'fine' or a false alarm");
    // The field bug this whole feature exists for, reproduced with REPRESENTATIVE (not the
    // owner's own measured) adult body-proportion ratios — arms noticeably longer than the torso,
    // shoulders somewhat narrower than it, both plausible for a real adult build. This proves the
    // mechanism catches an implausible reading under realistic numbers; whether it would have
    // caught the OWNER's actual historical 2.3 reading depends on his real measured proportions,
    // which this fixture does not claim to reproduce.
    const representativeAdult = { armToTorso: 0.7, shoulderToTorso: 0.35, headToTorso: 0.2 };
    console.assert(
      handSepIsPlausible(2.3, representativeAdult) === false,
      "under representative adult body-proportion ratios, a 2.3-torso-length hand separation (the app's own historical impossible reading) should be flagged"
    );

    // --- compareCalibrations / calibrationVerdict: agreement must stay SILENT — not just "not
    // flagged as different" but literally no message at all — across several slightly-varying
    // re-measurements. This matters more now than a one-off trigger would: with no button behind
    // it, this comparison runs every single time the app calibrates, so the bar for saying
    // anything has to hold up under ordinary repeated use, not just a single lucky test case.
    for (const jitter of [0, 0.03, -0.05, 0.08, -0.1]) {
      const fresh = {
        armToTorso: calibrationFromMedian.armToTorso * (1 + jitter),
        shoulderToTorso: calibrationFromMedian.shoulderToTorso * (1 - jitter),
        headToTorso: calibrationFromMedian.headToTorso * (1 + jitter / 2),
      };
      const verdict = calibrationVerdict(calibrationFromMedian, fresh);
      console.assert(
        verdict.save === true && verdict.message === null,
        `agreeing calibrations (jitter ${jitter}) must stay completely silent — no message at all, not even a mild one`
      );
    }
    const firstEverVerdict = calibrationVerdict(null, calibrationFromMedian);
    console.assert(
      firstEverVerdict.save === true && firstEverVerdict.message === null,
      "the very first calibration ever (nothing stored yet) must save silently — nothing to disagree with"
    );

    // --- A real disagreement must be surfaced, named correctly, and must NOT overwrite storage.
    const armChanged = {
      armToTorso: calibrationFromMedian.armToTorso * 1.6,
      shoulderToTorso: calibrationFromMedian.shoulderToTorso,
      headToTorso: calibrationFromMedian.headToTorso,
    };
    const disagreeVerdict = calibrationVerdict(calibrationFromMedian, armChanged);
    console.assert(disagreeVerdict.save === false, "a real disagreement must not overwrite what's stored");
    console.assert(
      typeof disagreeVerdict.message === "string" && disagreeVerdict.message.includes("arm length"),
      "a disagreement in arm-to-torso specifically should name 'arm length' in the message"
    );
    console.assert(
      !disagreeVerdict.message.includes("shoulder width") && !disagreeVerdict.message.includes("head size"),
      "a disagreement in only ONE ratio must not also name the ratios that actually agreed"
    );

    // --- calibrationStatusText: THREE states, not two (PM review, 2026-08-23) — agreement stays
    // silent, a session with no successful calibration gets its own neutral line, and a
    // disagreement keeps the existing wording. Each must be independently distinguishable: a
    // "not yet" session must never look like an agreeing one (both used to render as nothing at
    // all), and neither may be confused with an actual disagreement.
    const neverState = calibrationStatusText(false, null);
    console.assert(
      neverState !== null && neverState.tone === "neutral" && typeof neverState.text === "string" && neverState.text.length > 0,
      "calibrationDone === false must produce its own neutral, non-null line — silence here is exactly the ambiguity being fixed"
    );
    const agreedState = calibrationStatusText(true, null);
    console.assert(agreedState === null, "calibrationDone === true with no disagreement message must stay completely silent — agreement is the one state allowed to say nothing");
    const disagreedState = calibrationStatusText(true, disagreeVerdict.message);
    console.assert(
      disagreedState !== null && disagreedState.tone === "warn" && disagreedState.text === disagreeVerdict.message,
      "calibrationDone === true with a disagreement message must surface that exact message, tagged as a real warning, not the neutral 'not yet' wording"
    );
    // The three states must actually be distinguishable from one another, not just individually
    // non-crashing — this is the literal bug being fixed (agreement and "never measured" both
    // rendered as nothing).
    console.assert(neverState?.text !== disagreedState?.text, "the 'not yet calibrated' line and a real disagreement must never read as the same sentence");

    // --- calibrationShareLine: the PM's window into whether calibration ran at all, when, and
    // what today found. Must distinguish "never calibrated, ever" from "calibrated before, not
    // reconfirmed today" from "reconfirmed and agreed" from "reconfirmed and disagreed" — four
    // combinations of the same two facts (does a stored record exist, did today's check land).
    const neverCalibratedLine = calibrationShareLine(null, false, null);
    console.assert(neverCalibratedLine.includes("never calibrated") && neverCalibratedLine.includes("not yet confirmed"), "no stored record and no fresh one this session must say plainly that calibration has never run");

    const staleLine = calibrationShareLine({ ...calibrationFromMedian, takenAt: Date.now() - 86400000 }, false, null);
    console.assert(
      staleLine.includes("last confirmed") && !staleLine.includes("never calibrated") && staleLine.includes("not yet confirmed"),
      "a stored record from a past session, not yet reconfirmed today, must say when it was last confirmed AND that today hasn't checked in yet — not silently imply today already agreed"
    );

    const agreedLine = calibrationShareLine({ ...calibrationFromMedian, takenAt: Date.now() }, true, null);
    console.assert(agreedLine.includes("last confirmed") && agreedLine.includes("agreed with what was stored"), "a session where today's check agreed must say so explicitly, not just stay quiet the way the live status line does — the PM needs this even when the owner doesn't");

    const disagreedLine = calibrationShareLine(calibrationFromMedian, true, disagreeVerdict.message);
    console.assert(disagreedLine.includes("disagreed with what was stored"), "a session where today's check disagreed must say so explicitly in the shared text");

    // --- Framing (legs-cut-off) note: silent when fully in frame, plain when not.
    console.assert(describeFraming(true) === null, "fully in frame must say nothing at all");
    console.assert(typeof describeFraming(false) === "string" && describeFraming(false).length > 0, "legs missing from frame for a real stretch must say something plain");

    // --- FRAMING SIGNATURE (optional add-on): the squareness proxy responds to rotation toward
    // the camera, and stays put under pure scaling (a distance change, not a rotation).
    const framing = framingSignatureOf(standingBody, NOOP_W, NOOP_H);
    const framingFar = framingSignatureOf(standingBodyFar, NOOP_W, NOOP_H);
    console.assert(framing !== null && framingFar !== null, "a fully-visible standing body should produce a framing signature");
    console.assert(
      near(framing.shoulderSquareness, framingFar.shoulderSquareness) && near(framing.hipSquareness, framingFar.hipSquareness),
      "the squareness proxy must stay put when the body is only scaled (a distance change), not rotated"
    );

    const fullySideOn = mkLandmarks({
      ...standingBody,
      11: { x: 0.5, y: 0.3 },
      12: { x: 0.5, y: 0.3 },
      23: { x: 0.5, y: 0.6 },
      24: { x: 0.5, y: 0.6 },
    });
    const moreFrontal = mkLandmarks({ ...standingBody, 11: { x: 0.2, y: 0.3 }, 12: { x: 0.8, y: 0.3 } });
    const framingSideOn = framingSignatureOf(fullySideOn, NOOP_W, NOOP_H);
    const framingFrontal = framingSignatureOf(moreFrontal, NOOP_W, NOOP_H);
    console.assert(framingSideOn !== null && near(framingSideOn.shoulderSquareness, 0), "shoulders projecting on top of each other (dead side-on) should read ~0 squareness");
    console.assert(
      framingSideOn.shoulderSquareness < framing.shoulderSquareness && framing.shoulderSquareness < framingFrontal.shoulderSquareness,
      "the squareness proxy must actually respond to rotation toward the camera — side-on < standingBody's own angle < more frontal"
    );

    // --- describeFramingChange: silent across ordinary, slightly-varying setups; speaks up on a
    // real difference, describing WHAT changed without diagnosing WHY.
    for (const jitter of [0, 0.05, -0.08, 0.1]) {
      const freshFraming = {
        apparentSize: framing.apparentSize * (1 + jitter * 0.5),
        frameX: framing.frameX + jitter * 0.02,
        frameY: framing.frameY - jitter * 0.02,
        shoulderSquareness: framing.shoulderSquareness + jitter * 0.05,
        hipSquareness: framing.hipSquareness + jitter * 0.05,
      };
      console.assert(
        describeFramingChange(framing, freshFraming) === null,
        `ordinary session-to-session framing variation (jitter ${jitter}) must stay silent`
      );
    }
    const closerFraming = { ...framing, apparentSize: framing.apparentSize * 1.6 };
    const closerMsg = describeFramingChange(framing, closerFraming);
    console.assert(
      typeof closerMsg === "string" && closerMsg.includes("bigger in frame"),
      "standing noticeably closer should say he looks bigger in frame, not diagnose a cause"
    );
    console.assert(!closerMsg.toLowerCase().includes("camera"), "the framing-change message must describe what changed, never diagnose a cause like the camera moving");

    console.assert(framingChangeMessage(null, framing) === null, "no stored framing signature yet must say nothing — optional, never a false claim");
    console.assert(framingChangeMessage(framing, null) === null, "no fresh framing signature this session must say nothing — optional, never a false claim");
  }

  // --- SHOOTING CUES (HANDOVER.md Stage 2): updateCue must not let the very next frame cut a
  // logged/rejected flash short, and once the flash ends it must hand back to whatever is
  // ACTUALLY true at that moment, not whatever was true when the flash started — the exact
  // sequence a real session can produce (a shot logs, and the very next instant tracking is
  // lost because he's already lowering the bow and stepping out of frame).
  {
    clearTimeout(cueOutcomeTimer); // in case an earlier block in this run left one pending
    cueOutcomeTimer = null;

    updateCue(false, false);
    console.assert(cueEl.className === "cue-resting", "updateCue(false, false) should show the calm resting cue");

    updateCue(true, false);
    console.assert(cueEl.className === "cue-lost", "updateCue(true, ...) should show the lost cue regardless of the watching flag");

    updateCue(false, true);
    console.assert(cueEl.className === "cue-watching", "updateCue(false, true) should show the watching cue");

    signalOutcome(true);
    console.assert(cueEl.className === "cue-logged", "signalOutcome(true) should flash the logged cue");
    updateCue(true, false); // simulates tracking being lost the instant after a shot logs — a real sequence, not a contrived one
    console.assert(cueEl.className === "cue-logged", "a live updateCue call during the flash window must not cut the flash short");
    revertCueToCurrent();
    console.assert(cueEl.className === "cue-lost", "once the flash ends it must hand back to what's ACTUALLY true now (lost), not what was true when the flash started (watching)");

    signalOutcome(false);
    console.assert(cueEl.className === "cue-rejected", "signalOutcome(false) should flash the rejected cue");
    revertCueToCurrent();
    console.assert(cueEl.className === "cue-lost", "still lost after the flash — a rejected draw must not accidentally imply tracking recovered");

    clearTimeout(cueOutcomeTimer);
    cueOutcomeTimer = null;
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
  fullDrawShotCount = savedFullDrawShotCount;
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
  clearTimeout(cueOutcomeTimer); // never let a test's own outcome flash fire for real after this function has already handed module state back
  cueOutcomeTimer = savedCueOutcomeTimer;
  cueLastLost = savedCueLastLost;
  cueLastWatching = savedCueLastWatching;
  cueEl.className = savedCueClassName;

  // ===== SESSION PERSISTENCE — serialize/deserialize round trip, and what a restored row looks
  // like. Deliberately tests only the PURE functions (serializeShotSession/deserializeShotSession/
  // entryForPersistence) rather than the real saveSessionToStorage/restoreSessionFromStorage —
  // those two touch the real localStorage key a real session on this page might already be using,
  // and a diagnostic run must never clobber or read that (same "leaves the live app exactly as it
  // found it" rule as everything else in selfTest). No savedX/restore bookkeeping needed here for
  // that reason: nothing below reads or writes actual module state (log, shotCount, etc.).
  {
    const now = 1_000_000_000;
    const fixtureEntry = {
      shotNum: 1,
      bowArmAngle: 171,
      shoulderDrop: { bow: 40, draw: 45 },
      elbowAlign: { deviation: 3, direction: "high", signed: 3 },
      handSep: 0.8,
      anchorOk: true, armOk: true, sepOk: true, stillOk: true,
      startMs: 500,
      reachedFullDraw: true,
      clipFailReason: null,
      clipUrl: "blob:fake-had-a-clip", // simulates a shot that DID record a clip before the reload
      clipBlob: { fake: true },
    };
    const state = {
      log: [fixtureEntry],
      shotCount: 1,
      fullDrawShotCount: 1,
      rejectedAttemptCount: 2,
      unsettledAttemptCount: 1,
      attentionIdlePeriods: 3,
      attentionLateWakeCount: 1,
      clipsUnavailableReason: "Some shots couldn't be recorded — at least one clip failed this session.",
    };

    // Round trip preserves every field. Goes through an actual JSON.stringify/parse, not just the
    // plain object serializeShotSession returns, because that's what a real save/load does (see
    // saveSessionToStorage/restoreSessionFromStorage) and JSON silently drops things a plain object
    // copy wouldn't (undefined values, for one).
    const payload = serializeShotSession(state, now);
    console.assert(payload.v === SHOT_SESSION_FORMAT_VERSION, "serializeShotSession should stamp the current format version");
    console.assert(payload.log[0].hadClip === true, "serializeShotSession should record that a clipped entry HAD a clip");
    console.assert(payload.log[0].clipUrl === undefined && payload.log[0].clipBlob === undefined, "serializeShotSession must never write clipUrl/clipBlob — see this section's own top comment");

    const roundTripped = JSON.parse(JSON.stringify(payload));
    const restored = deserializeShotSession(roundTripped, now + 1000, SESSION_RESTORE_MAX_AGE_MS);
    console.assert(restored !== null, "a fresh, well-formed payload must be accepted");
    if (restored) {
      const r = restored.log[0];
      console.assert(r.shotNum === 1 && r.bowArmAngle === 171 && r.handSep === 0.8 && r.startMs === 500 && r.reachedFullDraw === true, "every plain field on a restored entry must match what was saved");
      console.assert(r.shoulderDrop.bow === 40 && r.shoulderDrop.draw === 45, "restored shoulderDrop must round-trip both sides");
      console.assert(r.elbowAlign.signed === 3 && r.elbowAlign.direction === "high", "restored elbowAlign must round-trip");
      console.assert(r.anchorOk === true && r.armOk === true && r.sepOk === true && r.stillOk === true, "restored debug-flag fields must round-trip");
      console.assert(r.clipUrl === undefined && r.clipBlob === undefined, "a restored entry must never carry a clipUrl/clipBlob — both died with the old page");
      console.assert(r.clipLostOnRestore === true, "an entry that HAD a clip before the reload must come back marked clipLostOnRestore, so its row can say so honestly");
      console.assert(
        restored.shotCount === 1 && restored.fullDrawShotCount === 1 && restored.rejectedAttemptCount === 2 && restored.unsettledAttemptCount === 1 &&
        restored.attentionIdlePeriods === 3 && restored.attentionLateWakeCount === 1 && restored.clipsUnavailableReason === state.clipsUnavailableReason,
        "every session counter must round-trip alongside the log"
      );

      // A payload saved by an older build (before fullDrawShotCount existed) is missing the
      // field entirely — must fall back to counting reachedFullDraw in the restored log itself,
      // never a silent 0 that would undercount a session already in progress. See
      // deserializeShotSession's own comment.
      const { fullDrawShotCount: _drop, ...payloadWithoutFullDrawCount } = roundTripped;
      const restoredOldFormat = deserializeShotSession(payloadWithoutFullDrawCount, now + 1000, SESSION_RESTORE_MAX_AGE_MS);
      console.assert(
        restoredOldFormat && restoredOldFormat.fullDrawShotCount === 1,
        "a payload missing fullDrawShotCount (an older build's save) must fall back to counting reachedFullDraw entries in its own restored log, not default to 0"
      );

      // A restored entry with a dead clip must render and share without throwing — the whole
      // point of clipLostOnRestore is that a restored row degrades honestly instead of offering a
      // Watch button whose blob: URL is already dead (see renderShotRow's clipBit).
      let rowHtml = null;
      try {
        rowHtml = renderShotRow(r, summarizeShots([r]), { bowArm: null, shoulderBow: null, shoulderDraw: null, elbow: null }, { bowArm: null, shoulderBow: null, shoulderDraw: null, elbow: null });
      } catch (err) {
        console.assert(false, `renderShotRow must not throw on a restored entry with a dead clip (threw: ${err.message})`);
      }
      if (rowHtml !== null) {
        console.assert(rowHtml.includes("clip lost"), "a restored row with a dead clip should say the clip is gone, in plain language");
        console.assert(!rowHtml.includes("shotlog-play"), "a restored row must never offer a Watch button for a clip that can't possibly play");
      }

      let shareText = null;
      try {
        shareText = buildShareText([r], {
          shotCount: restored.shotCount,
          fullDrawShotCount: restored.fullDrawShotCount,
          rejectedAttemptCount: restored.rejectedAttemptCount,
          unsettledAttemptCount: restored.unsettledAttemptCount,
          attentionIdlePeriods: restored.attentionIdlePeriods,
          attentionLateWakeCount: restored.attentionLateWakeCount,
          clipsUnavailableReason: restored.clipsUnavailableReason,
          modelStatusLine: null,
          rightHanded: true,
          mirrored: false,
          cameraWidth: 720,
          cameraHeight: 1280,
          sessionWasRestored: true,
        });
      } catch (err) {
        console.assert(false, `buildShareText must not throw on a restored entry with a dead clip (threw: ${err.message})`);
      }
      if (shareText !== null) {
        console.assert(shareText.includes("recovered after the app restarted"), "shared text must say the session was recovered, same fact the on-screen notice shows");
        console.assert(shareText.includes("recorded=lost-on-restart"), "a restored entry's share line must say its clip was lost on restart, not the plain recorded=no a shot that simply never had one would get");
      }
    }

    // A stale payload must be rejected — this is the whole guard against resurrecting an old
    // session (see SESSION_RESTORE_MAX_AGE_MS's own comment). "Now" here is moved past the
    // cutoff instead of shrinking the cutoff itself, so this exercises the real constant.
    const stalePayload = serializeShotSession(state, now);
    const staleResult = deserializeShotSession(stalePayload, now + SESSION_RESTORE_MAX_AGE_MS + 1, SESSION_RESTORE_MAX_AGE_MS);
    console.assert(staleResult === null, "a payload older than SESSION_RESTORE_MAX_AGE_MS must be rejected, not restored");

    // Corrupt/partial/hand-edited payloads must be rejected WITHOUT throwing — this runs on
    // whatever a previous, possibly different version of this file left behind (see
    // deserializeShotSession's own comment), so it has to survive being handed garbage.
    const garbagePayloads = [
      null,
      undefined,
      "just a string, not an object",
      {}, // missing everything
      { v: SHOT_SESSION_FORMAT_VERSION }, // missing savedAt/log
      { v: SHOT_SESSION_FORMAT_VERSION + 1, savedAt: now, log: [] }, // a future format version
      { v: SHOT_SESSION_FORMAT_VERSION, savedAt: "not a number", log: [] },
      { v: SHOT_SESSION_FORMAT_VERSION, savedAt: now, log: "not an array" },
      { v: SHOT_SESSION_FORMAT_VERSION, savedAt: now, log: [{ bowArmAngle: 171 }] }, // entry with no shotNum
      { v: SHOT_SESSION_FORMAT_VERSION, savedAt: now, log: [null] },
    ];
    garbagePayloads.forEach((g, i) => {
      let result = "not called";
      let threw = false;
      try {
        result = deserializeShotSession(g, now, SESSION_RESTORE_MAX_AGE_MS);
      } catch (err) {
        threw = true;
      }
      console.assert(!threw, `deserializeShotSession must never throw on garbage payload #${i}`);
      console.assert(result === null, `deserializeShotSession must reject garbage payload #${i} (got ${JSON.stringify(result)})`);
    });
  }
  // ===========================================================================

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
  console.assert(fullDrawShotCount === savedFullDrawShotCount, "selfTest leaked: fullDrawShotCount was not restored to its original value");
  console.assert(rejectedAttemptCount === savedRejectedAttemptCount, "selfTest leaked: rejectedAttemptCount was not restored to its original value");
  console.assert(unsettledAttemptCount === savedUnsettledAttemptCount, "selfTest leaked: unsettledAttemptCount was not restored to its original value");
  console.assert(settledFrames === savedSettledFrames, "selfTest leaked: settledFrames was not restored to its original value");
  console.assert(prevUsedCropBox === savedPrevUsedCropBox, "selfTest leaked: prevUsedCropBox was not restored to its original value");
  console.assert(
    clipsUnavailableReason === savedClipsUnavailableReason,
    `selfTest leaked: clipsUnavailableReason was not restored to its original value (left as ${JSON.stringify(clipsUnavailableReason)}) — a diagnostic run must never leave a false "clips failed" banner behind`
  );
  console.assert(pendingClipNote === savedPendingClipNote, "selfTest leaked: pendingClipNote was not restored to its original value");
  console.assert(cueLastLost === savedCueLastLost, "selfTest leaked: cueLastLost was not restored to its original value");
  console.assert(cueLastWatching === savedCueLastWatching, "selfTest leaked: cueLastWatching was not restored to its original value");
  console.assert(cueEl.className === savedCueClassName, "selfTest leaked: #cue's class was not restored to its original value");

  console.log("selfTest done — check above for any failed console.assert");
}

if (location.search.includes("selftest")) selfTest();

main();
