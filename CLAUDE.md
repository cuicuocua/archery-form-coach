# CLAUDE.md

This file provides guidance to Claude Code when working in this project.

## Project Purpose

A browser-based archery form coach prototype: opens in iPhone Safari, uses the
phone camera to run live body pose tracking, and overlays a skeleton plus two
form readouts (bow-arm straightness, draw-elbow height) while the owner
practices archery. Built for a non-coder owner — explain changes in plain
language, avoid unexplained jargon.

## Working method — PM and engineering team

**Claude on Opus 5 acts as project manager.** It plans, scopes, assigns, reviews
the returned work, and reports to the owner in plain language. It does not write
feature code itself.

**All development is done by sub-agents running Sonnet 5**, briefed as senior
engineers. The PM spawns them with the Agent tool using `model: "sonnet"` — no
manual model switching by the owner is needed, the PM stays on Opus while the
engineers run on Sonnet.

Rules the PM follows:

- Each agent gets a self-contained brief: the goal, the files it may touch, the
  constraints from this file, and what "done" looks like. Agents do not see the
  owner's conversation.
- Agents run in parallel only when their work touches genuinely independent
  areas. `app.js` is one small file — concurrent edits to it are serialised, or
  isolated in worktrees when parallelism is actually worth it.
- The PM reviews every returned diff before reporting it as done, and runs the
  check itself rather than trusting the agent's summary.
- The owner is told what changed and what to test, never handed a raw agent
  report.

## Stack and constraints

- Plain HTML/CSS/JS, no build step, no npm/frameworks. Libraries load from CDN
  (jsDelivr) at runtime.
- Pose tracking: MediaPipe Tasks Vision `PoseLandmarker` (GPU delegate),
  imported directly from jsDelivr as an ES module in `app.js`. Starts on the
  `pose_landmarker_full` model for steadier landmarks; auto-falls-back to
  `pose_landmarker_lite` if a short warm-up measurement finds this phone can't
  keep full running fast enough (see Key decisions below). Fallback considered
  but not built: MoveNet/TensorFlow.js, only if MediaPipe itself proves
  unreliable on iPhone Safari.
- Camera access (`getUserMedia`) requires a secure context (HTTPS or
  localhost). Local dev/testing uses `python3 -m http.server` + a Cloudflare
  quick tunnel (`cloudflared tunnel --url http://localhost:8000`) for a
  temporary public HTTPS URL — free, no account. The tunnel URL changes every
  time it's restarted; that's expected for this prototype stage.

## Files

- `index.html` — page structure: video element, drawing canvas, status text,
  two readout panels, four control buttons (camera flip, handedness, mirror
  toggle, shot log), and the full-screen clip player (video +
  slow-motion/close buttons) that opens over everything else when a shot log
  row's clip is tapped.
- `style.css` — full-screen dark UI, large tap targets, green/amber/grey
  color states for readouts, plus the shot log's clip button/no-clip note and
  the clip player's own styling.
- `app.js` — all logic: MediaPipe setup (with the full→lite pose-model
  fallback), camera start/switch, region-of-interest cropping of the video
  before each frame is handed to MediaPipe (a small offscreen canvas, never
  shown or recorded — see "Region-of-interest cropping" below), per-frame
  pose detection, One Euro landmark smoothing, skeleton drawing (via
  MediaPipe's own `DrawingUtils`) burned into the overlay canvas together
  with the raw camera frame — that combined canvas (always the full camera
  view, never the crop) is what per-shot clip recording actually captures —
  the two angle calculations, the shot log, clip recording/playback, and the
  constants blocks (calibration, smoothing, pose model, ROI cropping) at the
  top of the file.

## Key decisions

- **Handedness convention**: right-handed archer → bow arm = left, draw arm =
  right (and swapped for left-handed). Toggle button, default right-handed.
- **Bow-arm line** = angle at the bow elbow between shoulder→elbow and
  elbow→wrist vectors (180° = straight arm).
- **Draw-elbow height** = NOT a raw angle off the shoulder line — the shoulder
  line is nearly degenerate in a side-on view (both shoulders project close
  together in x), so instead we measure the draw elbow's vertical pixel offset
  from the draw shoulder and express it as `atan2(verticalOffset, torsoLength)`
  in degrees, using shoulder-to-hip distance as the scale reference (draw side
  preferred, bow side as fallback). This stays meaningful even when the
  shoulder line itself is unreliable. Documented inline in `app.js`.
- **The owner cannot touch or read the phone while shooting.** They are on a
  shooting line five metres away, holding a drawn compound bow, looking at a
  target. They cannot tap a button, read a number, or take a screenshot at any
  particular moment. This has broken three designs so far (the manual freeze
  button, the live debug overlay, and a "walk over and screenshot it" test
  protocol). **Anything the app needs to convey must be recorded by the app and
  still be there later** — never shown only in the instant it occurs, and never
  gated behind an action taken at a specific time. Assume exactly one
  interaction per session: the owner walks over when they are done and looks.

- **Bow type: compound only, for now.** The owner shoots both recurve and
  compound; this prototype is tuned for **compound**. That is not cosmetic —
  compound has mechanical let-off (the archer holds full draw comfortably for
  seconds, so hold-and-stillness thresholds can be generous), a fixed draw stop
  (draw length is highly repeatable shot to shot, so hand separation is a near-
  binary signal), and a release aid anchoring at or behind the jaw rather than
  fingers under the chin. Recurve would want looser hold/stillness thresholds
  and a different anchor position. **Do not build a bow-type toggle or a second
  set of constants until the owner actually starts shooting recurve with this** —
  it is speculative until then.

- **Confidence handling**: any landmark with MediaPipe `visibility` below
  `MIN_VISIBILITY` (constant, currently 0.6) makes its readout show grey
  "— uncertain" instead of a number/color — never show a confident-looking
  number from a low-confidence joint.
- **Calibration constants are placeholders.** `BOW_ARM_ANGLE_MIN/MAX`,
  `DRAW_ELBOW_HEIGHT_MIN/MAX`, and `MIN_VISIBILITY` live in one clearly labeled
  block at the top of `app.js` — owner will tune these with a coach.
  Three other labelled blocks near the same spot (`SMOOTH_*`, `MODEL_*`, and
  `ROI_*`) hold a completely different kind of constant — performance/feel
  knobs, not form targets. Nothing in any of the three needs a coach: change
  them freely and judge by eye (watch the skeleton, or a recorded clip
  afterwards).

- **Landmark smoothing: a One Euro filter, not a plain moving average.** Raw
  MediaPipe output is jittery outdoors — the archer occupies a small part of
  the frame at five metres, so each frame's joint estimate lands slightly
  differently even standing still. A fixed amount of smoothing is the wrong
  trade either way: enough to kill that jitter at full draw also blurs the
  fast raise into a laggy skeleton that trails behind the archer's real
  position and throws off full-draw-detection timing; not enough to blur the
  raise leaves full draw jittery. One Euro adapts instead — it smooths hard
  when a joint is nearly still and lets go automatically the instant the joint
  speeds up. That matters specifically because full draw (a compound archer
  holding steady at let-off) is both the stillest moment of a shot AND the
  exact moment every measurement gets taken — the case this filter is built
  to help most. Implemented inline in `app.js` (`OneEuroFilter` /
  `LandmarkSmoother` classes, no dependency added), filtering `x`/`y` per
  landmark — never `visibility`, which is a confidence score, not a position,
  and smoothing it would let a low-confidence joint's readout creep toward
  looking trustworthy over a few frames. Tunables: `SMOOTH_MIN_CUTOFF` (lower
  = calmer at rest), `SMOOTH_BETA` (higher = less lag when moving fast),
  `SMOOTH_DCUTOFF` (rarely needs touching). Filter state resets whenever
  tracking is lost or the camera is switched, so a stale position can never
  smooth into a fresh one and drag the skeleton across the frame. Smoothed
  landmarks feed everything downstream: the skeleton drawing (and therefore
  the recorded clips), all three readouts, and the shot-log sampling.

- **Pose model: `full` by default, with an automatic fallback to `lite`.**
  `pose_landmarker_full` tracks more steadily than `lite` but costs more GPU
  time per frame. Rather than guess which a given phone can afford, the app
  times a short warm-up window right after startup (skipping the first
  `MODEL_WARMUP_FRAMES`, since cold-start frames are never representative) and
  switches itself to `lite` if the average frame took longer than
  `MODEL_SLOW_FRAME_MS`. A failed rebuild never interrupts tracking — the app
  just keeps running on whichever landmarker it already has. The owner cannot
  watch this decision happen (see "owner cannot touch or read the phone"
  above), so it is never shown live: instead, a small persistent line at the
  top of the shot log (same spot as the clip-recording banner) records which
  model ended up running, the measured per-frame pose-detection cost in
  milliseconds, and the real rendered frame rate over that same window (drawing
  and smoothing included, not just inference) — reported as two separate
  figures, deliberately, because a frame rate derived from inference time
  alone ignores everything else a frame costs and can look many times faster
  than the app actually ran; that was a real bug this replaced.

- **Region-of-interest cropping — the real fix for distance-driven jitter, not
  just smoothing it away.** At five metres the archer occupies a small part of
  the camera frame, and MediaPipe itself resizes whatever image it's given
  down to the model's small square input before it ever looks at it — so
  without this, every joint guess is being made from a version of the archer
  only a few dozen pixels tall, no matter how good the camera actually is. The
  One Euro smoothing above treats the resulting jitter as noise to average
  out after the fact; this treats the cause instead. Each frame, the app draws
  just a square region around where the archer was last seen — expanded by a
  generous padding margin, forced square, clamped to the video's bounds — into
  a small offscreen canvas (`document.createElement("canvas")`, never added to
  the page, never shown or recorded), scaled up to fill it, and hands
  MediaPipe *that* instead of the whole video frame. Same camera, same
  distance, far more pixels of archer reaching the model. The landmarks
  MediaPipe returns are in that crop's own local coordinates, so they are
  mapped back into full-frame normalised coordinates immediately, before
  anything downstream (angle maths, torso-length scale reference, drawing, the
  shot log) ever sees them — everything in this app assumes full-frame
  coordinates, and getting that mapping wrong would silently corrupt every
  number the owner is calibrating. The One Euro smoothing filter runs on
  those mapped, full-frame coordinates, exactly as before this feature
  existed.
  Tunables live in one labelled `ROI_*` block: `ROI_CROPPING_ENABLED` is the
  master on/off switch (set it `false` to go back to exactly the old
  whole-frame behaviour, e.g. to rule the feature out if it ever misbehaves in
  the field); `ROI_CANVAS_SIZE` is the offscreen canvas's pixel size;
  `ROI_PADDING_FRACTION` controls how forgiving the crop is of a fast raise or
  a step sideways versus how much it zooms in; `ROI_MIN_VISIBLE_LANDMARKS`
  sets how many confidently-visible landmarks it takes to trust a crop box at
  all; `ROI_SMOOTHING` is hysteresis on the box itself (an exponential ease,
  not a hard "only move past X" threshold) so ordinary landmark noise doesn't
  make the crop — and therefore the zoom level and framing MediaPipe sees —
  flicker frame to frame.
  **Recovery**: whenever a frame comes back with no landmarks, or too few
  confidently-visible ones to trust a crop box, the app drops the crop
  entirely and detects on the whole frame again starting next frame — it never
  keeps cropping into a region that may no longer contain the archer. A crop
  that could get stuck staring at an empty patch of grass, with no landmarks
  ever coming back to trigger a whole-frame retry, would be exactly the kind
  of hang the owner has no way to recover from mid-session (see "owner cannot
  touch or read the phone" above) — this is why re-acquisition is unconditional
  and immediate, not something that waits or times out.
  Cropping is an inference-input concern only: the visible `#overlay` canvas,
  and therefore every recorded clip, always shows the full camera frame with
  the skeleton drawn over it — the crop is never drawn to anything the owner
  sees.

- **Freeze button: removed.** The prototype originally had a manual freeze
  button plus an auto-freeze that triggered at full draw and released itself
  after a few seconds. Both are gone — the owner confirmed in the field that
  neither helps: he cannot tap a button while holding a drawn bow five metres
  away, and a frame that freezes for four seconds and releases itself is gone
  by the time he walks over (see "owner cannot touch or read the phone" above).
  `isAtFullDraw` still runs every frame — it now exists purely to feed the shot
  log via `trackShotAttempt`, and its boolean return value is otherwise unused
  by the app (selfTest still reads it directly). The `<video>` element is never
  paused after startup any more.

- **Per-shot clips, not one long session recording.** What replaced the freeze
  button: every draw attempt records its own short video (the overlay canvas —
  camera frame plus skeleton — via `canvas.captureStream`, through
  `MediaRecorder`), from the moment hand separation shows the raise starting
  to about 2.5 seconds after the shot ends, so release and follow-through are
  in it too. A single continuous recording of the whole session was
  deliberately rejected: the owner would have to scrub through minutes of
  video standing at the phone to find the ten seconds that mattered, which is
  exactly the kind of live interaction this app exists to avoid (see "owner
  cannot touch or read the phone" above) — a *recorded* thing he can review
  is fine, a live thing he has to operate is not. Each finished clip attaches
  to its shot's row in the log by shot number once ready; a row with no clip
  (recording unsupported, or it failed for that one shot) still shows its
  numbers, with a plain "no clip" note in place of the watch button. Clips are
  capped at 20 seconds each as a safety valve, live in memory only exactly
  like the shot log, and get `URL.revokeObjectURL`'d the moment their row
  falls off the end of `SHOT_LOG_MAX`. Every `canvas.captureStream` track this
  creates is explicitly stopped (in the recorder's `onstop`, once the blob's
  data is safely out) once its clip is done — `MediaRecorder.stop()` does not
  stop the capture tracks feeding it, and leaving those running would mean one
  live, still-pulling track left behind per shot for the rest of the session.
  If recording doesn't work at all in the browser, or throws on start, a
  persistent line goes up at the top of the shot log saying so — never just a
  console message, for the same "the owner finds out later, not in the
  moment" reason as everything else here. The wording is deliberately
  different depending on which happened: never-supported-at-all (checked once
  at startup) says plainly that clips don't work here; a `MediaRecorder` that
  threw partway through a session says only that at least one shot's clip
  failed, since later attempts still retry and can succeed — the two are not
  the same claim, and telling the owner "recording doesn't work" while half
  the log has working Watch buttons would be actively misleading. Either
  message latches once set and is never cleared or overwritten by the other.

- **Mirroring is done in the canvas pixels, not CSS.** The front camera has
  always needed to mirror on-screen (the normal selfie convention), and there
  is now a manual 🪞 toggle on top of that (set-up-time only, same as
  handedness — the owner can't reach the button once he's walked to the
  line). Both used to be a CSS `transform: scaleX(-1)` on `#video`/`#overlay`,
  which only changes how the browser *displays* those elements — it never
  touched a pixel. That was a real bug once clips existed: `canvas.captureStream`
  reads the canvas's actual pixel buffer, which the CSS transform never
  touched, so a mirrored on-screen view recorded an *unmirrored* clip — what
  the owner watched back didn't match what he saw live. The fix: mirroring now
  happens by wrapping the canvas draw itself (video frame and skeleton
  together, in one `ctx.save()`/`translate`/`scale(-1,1)`/`restore()`) in
  `withMirror`, so the canvas's own pixels are what's flipped — on-screen view
  and recorded clip are the same pixels, so they can never disagree again.
  This is now the *only* place mirroring happens anywhere in the app — the old
  CSS class is gone, so there is no way for the picture to end up
  double-mirrored or mirrored in one place and not the other. Effective
  mirror state is `defaultMirrorFor(facingMode) XOR mirrorToggled` — a pure
  function, checked directly in `selfTest`. Mirroring only ever changes how a
  frame is drawn; it never touches a landmark coordinate, so it cannot move a
  measured number or interfere with the handedness toggle's job of deciding
  which arm is which.

- **Shot log attempt gating: a draw attempt must plausibly BE a draw before it
  gets logged.** Field report from the owner: "there are many more arrows than
  I've actually shot." `DRAW_ATTEMPT_MIN_SEP` (0.3 torso-lengths, ~15cm) alone
  was only ever meant to mark when an attempt starts/ends, not to decide
  whether one deserved a row — but with nothing else gating it, nocking an
  arrow, lowering the bow, adjusting a release aid, or the camera briefly
  losing tracking (`endAttempt` also fires on pose loss) all crossed that
  floor and got logged as phantom shots, dragging `summarizeShots`'s session
  average around for every real shot compared against it. Two more gates,
  checked once an attempt ends (by hands relaxing OR by tracking loss — same
  check either way, so losing the archer can never manufacture a shot the
  same movement ending normally wouldn't have earned): the attempt's peak
  hand separation must clear `SHOT_MIN_PEAK_SEP_FRACTION` of
  `FULL_DRAW_HAND_SEP_MIN`, and it must last at least `SHOT_MIN_DURATION_MS`
  from start to end. A draw that fell short of literal full draw is still
  logged and still worth seeing — that specific behaviour is deliberate and
  unchanged, see the shot log bullet below — the new gates only throw out
  things that were never plausibly a draw attempt at all. Every discarded
  attempt increments `rejectedAttemptCount`, shown as a persistent "N
  movements ignored" line in the log once it's above zero, for the same
  reason as the clip-recording banner and pose-model line: the owner can't
  watch this filtering happen, so if the app is quietly discarding movement
  it has to say so later, not just get it right silently. `shotCount` only
  increments for attempts that actually get logged, so the numbers the owner
  reads match the arrows he shot.

- **The shot log claims repeatability, never form quality.** The app has no
  idea what good archery form looks like — every target range under
  CALIBRATE WITH COACH is an explicitly labelled placeholder nobody has tuned
  with a coach yet. So the log must never say anything that implies a
  judgement it can't back up: no "your bow arm is too bent", no scores, no
  pass/fail against an invented ideal. The one thing it CAN say honestly,
  needing no calibration at all, is whether the owner did the same thing
  twice — measured only against his own other shots that session, nothing
  else. `narrateMeasure` in `app.js` turns each measurement's own numbers
  into one plain-English line: whether it was steady, whether it drifted from
  the early shots to the late ones, or whether one shot stood out (named by
  number, so the owner knows which clip to go watch) — each judged against
  how much the OTHER shots in the same session happened to scatter, never
  against a fixed number of degrees or percent. Below three shots it says
  plainly that there's no honest consistency story to tell yet, rather than
  computing a spread from almost nothing; below five, whatever it does say
  carries an "early days" qualifier. **Do not add a coaching verdict, a score,
  or a fixed good/bad threshold to this wording** — if a future change wants
  to judge the owner's form against a target, that target needs to come from
  an actual coach into the CALIBRATE WITH COACH block first, not get invented
  in the log's language. Raw degrees/percent are still shown, in small print
  on each row, for whoever eventually tunes those placeholder constants
  against a real session — they're just never the headline.

- **Consistency claims need an absolute floor under the relative one, or a
  tight session narrates its own noise back at the owner.** Caught in
  testing: a synthetic body held completely still, with only the residual
  jitter smoothing leaves behind (~±0.004 of normalised-coordinate noise per
  landmark), still produced a 1-2 point wobble in shoulder drop — and because
  `narrateMeasure`'s drift/outlier checks were purely RELATIVE (a gap judged
  only against how much this session happened to scatter), a very tight,
  repeatable session has almost no scatter to divide by, so that tiny wobble
  looked "statistically" huge and got reported as drift. The tighter the
  archer shoots, the more confidently the app would narrate pure noise as a
  finding — the phantom-shot bug one level up, restated as a phantom trend.
  Fix: `BOW_ARM_CONSISTENCY_FLOOR_DEG`, `ELBOW_CONSISTENCY_FLOOR_DEG`,
  `SHOULDER_BOW_CONSISTENCY_FLOOR_PCT`, `SHOULDER_DRAW_CONSISTENCY_FLOOR_PCT`
  (one per measure, since degrees and percentage points aren't the same
  number wearing different labels) — an absolute, in-that-measure's-own-units
  floor a claim must ALSO clear, on top of the relative test, before
  `narrateMeasure` will call it drift or name a shot as standing out. Below
  the floor the honest answer is "steady", because that's the most this
  pipeline can actually tell apart from its own noise. **Not a calibration
  constant** — nothing about it judges the owner's archery, so it does not
  belong in CALIBRATE WITH COACH and needs no coach to set: it is a statement
  about how precisely this pipeline can measure a joint from a phone camera
  at five metres, same family as `SMOOTH_*`. Defaults are derived, not
  guessed, from each measure's own geometry at that ~±0.004 jitter level
  (worked through in the constant block's own comment in `app.js`) — a
  degree or two for the two angle measures, a few percentage points for
  shoulder drop, matching the actual field numbers this bug was found from.

- **The first shot of a session (or of any tracking dropout) was being measured
  through a colder pipeline than the rest, and that turned out to be most of
  what the floors above were papering over.** Investigated after the floor fix:
  the same static-body test kept naming shot 1 specifically, not spread evenly
  across shots the way ordinary noise would. Root cause, confirmed by
  comparing shot 1's readings against later shots' with and without a
  deliberate settle pause first: `OneEuroFilter` returns its very first
  sample completely unsmoothed and takes several frames to converge, and (see
  region-of-interest cropping above) `currentCropBox` runs whole-frame,
  lower-resolution detection until a box is first acquired. Both reset at the
  same three points — session start, tracking lost, camera switched — so a
  shot logged from any of those windows was measured differently from every
  other shot, a real difference in the numbers, not a wording problem. Fix:
  `SETTLE_FRAMES_REQUIRED` (a new PIPELINE SETTLING constant, same
  performance-knob family as `SMOOTH_*`/`ROI_*` — not calibration, no coach
  needed) counts consecutive good-tracking frames since the last reset;
  `trackShotAttempt`/`endAttempt` now track an attempt's true peak hand
  separation (all frames, for the existing gates — untouched) separately from
  its ELIGIBLE frames (settled frames only — see `advanceSettling` /
  `resetSettling`), and only ever log a shot's numbers from the latter (see
  the median-of-eligible-frames bullet below for what "log a shot's numbers
  from" means today — it changed after this fix shipped, but the ELIGIBLE-only
  requirement it's built on here did not). An attempt that's real
  (clears the existing gates) but has no eligible frame at all is neither
  logged nor silently dropped: it's counted in `unsettledAttemptCount`, its
  own persistent line, worded differently on purpose from the
  `rejectedAttemptCount` "movements ignored" line — one means "that wasn't a
  real draw", the other means "that was a real draw, the app just wasn't
  ready to measure it yet", and the owner needs to be able to tell them
  apart. In practice this gate is invisible in normal use — the owner has
  many seconds walking to the line before his first real shot — and only
  bites right after a mid-session tracking dropout, exactly the case where an
  unsettled reading would otherwise have been the risky one to log.
  Re-measured after this fix, at the same jitter level the bug was found at:
  shot 1 vs. later shots' readings converged to a sub-half-degree gap
  (previously several degrees), and the false-claim rate on a static body
  dropped from roughly 3 claims in 16 measure-lines to 0 in 200 — with the
  floor constants above left untouched, since nothing survived to need them
  raised further.

- **A shot's logged numbers are now the MEDIAN of each measure across the
  whole hold, not a single "best" frame — because that single-frame rule
  turned out to be a biased sample, not just an arbitrary one.** Until this
  fix, `endAttempt` logged whichever eligible frame had the HIGHEST hand
  separation (the two wrists' distance apart, divided by an estimated torso
  length). Investigated on suspicion, then measured directly: torso length is
  itself estimated from noisy landmarks, so a frame's hand-separation ratio
  can read high either because the archer's hands really were far apart THAT
  frame, or because the noisy torso-length ESTIMATE happened to come out
  small that frame and inflate the ratio — the peak-selection rule can't tell
  those two apart, and prefers both equally. That would be harmless on its
  own, except shoulder drop is ALSO a distance divided by that same estimated
  torso length, so the frame most likely to "win" on hand separation is
  disproportionately likely to also over-report shoulder drop, by the exact
  same mechanism. A synthetic-archer measurement (32 attempts, ~4,600 eligible
  frames, two independent trial runs, full pipeline — ROI cropping, One Euro
  smoothing, pipeline settling — all running for real, only the pose model
  itself replaced) confirmed it: torso-length estimates on the selected frame
  ran below that attempt's own mean in every single attempt measured (32/32);
  draw-side shoulder drop — which shares hand separation's own default torso
  scale (see `isAtFullDraw`: draw-side torso preferred, bow-side as fallback)
  — read on the order of half a percentage point high on the selected frame,
  consistently in the same direction (28/32 attempts), not the coin-flip a
  merely arbitrary choice would produce. Bow-side shoulder drop, which uses a
  DIFFERENT (independently-noisy) torso-length estimate, showed no consistent
  direction — and neither did bow-arm angle or draw-elbow alignment, both
  plain angles with no division by torso length anywhere in their maths. That
  contrast (strong + consistent where a measure shares hand separation's own
  denominator, absent where it doesn't or can't) is what confirms the bias is
  the division, not merely "one frame is untrustworthy." Fix: `endAttempt` no
  longer selects a frame at all. Each measure — bow-arm angle, bow- and
  draw-side shoulder drop, draw-elbow alignment — is now the MEDIAN of that
  measure across every ELIGIBLE frame of the hold, computed independently per
  measure (`medianSampleOf` in `app.js`; see its own block comment for the
  full reasoning and the null-handling rules). A median can't be dragged
  toward one noisy frame's ratio the way a peak-selection rule can, and
  because no single frame "wins," there is no longer a selection process for
  measurement noise to bias. Re-measured after the fix, same synthetic
  archer, same probe: the shipped, real logged shoulder-drop readings sit
  within noise of the attempt's own population mean, no consistent direction
  left. **Unaffected by this change, on purpose**: `peakHandSep` (all frames,
  eligible or not) still decides `SHOT_MIN_PEAK_SEP_FRACTION` — "did he draw
  far enough to count as a shot at all" is correctly a question about the
  single most extreme moment of the attempt, not an average across it — and
  `reachedFullDraw` still means "did any eligible frame reach true full
  draw." Memory is bounded per attempt (`MEDIAN_SAMPLE_CAP`, 200 eligible
  frames) via reservoir sampling, not "first N" or "most recent N" — either
  of those would bias the median toward one part of the hold; reservoir
  sampling gives every eligible frame of the whole hold an equal chance of
  surviving the cap, so bounding memory this way costs precision, never
  introduces a new selection bias of the same shape as the one just fixed.

## Not built / explicitly out of scope for this prototype

- No build tooling, no bundler, no TypeScript.
- No server-side component — everything runs client-side in the browser.
- No persistence/history of readings across sessions — and, per the same rule,
  no persistence of clips either. Both are memory-only and gone on reload.
- No automated tests (this is a visual/physical prototype — verification is
  "open it and watch the skeleton track your body correctly").

## Testing

Manual only: run locally via `python3 -m http.server`, check in a desktop
browser first (camera + toggles + shot log), then via the Cloudflare tunnel URL
on iPhone Safari for the real side-on shooting test. See README.md for exact
commands.
