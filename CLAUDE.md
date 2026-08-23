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
- Pose tracking: MediaPipe Tasks Vision `PoseLandmarker` (`pose_landmarker_lite`
  model, GPU delegate), imported directly from jsDelivr as an ES module in
  `app.js`. Fallback considered but not built: MoveNet/TensorFlow.js, only if
  MediaPipe proves unreliable on iPhone Safari.
- Camera access (`getUserMedia`) requires a secure context (HTTPS or
  localhost). Local dev/testing uses `python3 -m http.server` + a Cloudflare
  quick tunnel (`cloudflared tunnel --url http://localhost:8000`) for a
  temporary public HTTPS URL — free, no account. The tunnel URL changes every
  time it's restarted; that's expected for this prototype stage.

## Files

- `index.html` — page structure: video element, drawing canvas, status text,
  two readout panels, three control buttons (camera flip, handedness, shot
  log).
- `style.css` — full-screen dark UI, large tap targets, green/amber/grey
  color states for readouts.
- `app.js` — all logic: MediaPipe setup, camera start/switch, per-frame pose
  detection, skeleton drawing (via MediaPipe's own `DrawingUtils`), the two
  angle calculations, the shot log, and the calibration constants block at
  the top of the file.

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

## Not built / explicitly out of scope for this prototype

- No build tooling, no bundler, no TypeScript.
- No server-side component — everything runs client-side in the browser.
- No persistence/history of readings across sessions.
- No automated tests (this is a visual/physical prototype — verification is
  "open it and watch the skeleton track your body correctly").

## Testing

Manual only: run locally via `python3 -m http.server`, check in a desktop
browser first (camera + toggles + shot log), then via the Cloudflare tunnel URL
on iPhone Safari for the real side-on shooting test. See README.md for exact
commands.
