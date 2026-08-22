# CLAUDE.md

This file provides guidance to Claude Code when working in this project.

## Project Purpose

A browser-based archery form coach prototype: opens in iPhone Safari, uses the
phone camera to run live body pose tracking, and overlays a skeleton plus two
form readouts (bow-arm straightness, draw-elbow height) while the owner
practices archery. Built for a non-coder owner — explain changes in plain
language, avoid unexplained jargon.

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
  two readout panels, three control buttons (camera flip, handedness, freeze).
- `style.css` — full-screen dark UI, large tap targets, green/amber/grey
  color states for readouts.
- `app.js` — all logic: MediaPipe setup, camera start/switch, per-frame pose
  detection, skeleton drawing (via MediaPipe's own `DrawingUtils`), the two
  angle calculations, freeze/resume, and the calibration constants block at
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
- **Confidence handling**: any landmark with MediaPipe `visibility` below
  `MIN_VISIBILITY` (constant, currently 0.6) makes its readout show grey
  "— uncertain" instead of a number/color — never show a confident-looking
  number from a low-confidence joint.
- **Calibration constants are placeholders.** `BOW_ARM_ANGLE_MIN/MAX`,
  `DRAW_ELBOW_HEIGHT_MIN/MAX`, and `MIN_VISIBILITY` live in one clearly labeled
  block at the top of `app.js` — owner will tune these with a coach.
- **Freeze button** pauses the `<video>` element itself (not just the canvas
  overlay), so the frozen skeleton and the frozen video frame stay in sync.

## Not built / explicitly out of scope for this prototype

- No build tooling, no bundler, no TypeScript.
- No server-side component — everything runs client-side in the browser.
- No persistence/history of readings across sessions.
- No automated tests (this is a visual/physical prototype — verification is
  "open it and watch the skeleton track your body correctly").

## Testing

Manual only: run locally via `python3 -m http.server`, check in a desktop
browser first (camera + toggles + freeze), then via the Cloudflare tunnel URL
on iPhone Safari for the real side-on shooting test. See README.md for exact
commands.
