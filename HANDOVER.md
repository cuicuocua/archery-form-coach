# Handover — archery form coach

Paste this whole file into a new chat, or point that chat at `HANDOVER.md` in
the repo. It is written to be self-contained: a new assistant should not need
the conversation that produced it.

**Repo**: https://github.com/cuicuocua/archery-form-coach
**Live app**: https://cuicuocua.github.io/archery-form-coach/ (GitHub Pages, serves `main`)
**Branch**: work on `main`; it is what deploys.

**Read first, in this order**: `CLAUDE.md` (binding project rules and the full
decision history), then `README.md` (what the app does from the owner's point
of view), then `app.js` (~4500 lines, heavily commented — the comments explain
*why*, and are the real documentation).

---

## Who this is for

A compound archer. He props his phone side-on about five metres away, walks to
the shooting line, shoots an end, and walks back.

**He cannot touch or read the phone while shooting.** This single constraint has
killed several features already (a manual freeze button, an auto-freeze that
released itself, a live debug overlay). Assume exactly one interaction per
session: he walks over when he is done and looks. Anything the app needs to
tell him must be *recorded and still there later* — never shown only in the
instant it happens.

He is not a programmer. Explain changes in plain language.

---

## Working method

Claude on Opus acts as project manager: plans, scopes, writes brief, reviews
returned diffs, runs the verification itself rather than trusting an agent's
summary, and reports to the owner in plain language. Feature code is written by
sub-agents on Sonnet, briefed as senior engineers, each with a self-contained
brief. `app.js` is one file — serialise edits, or isolate them in git worktrees
when parallelism is worth it. See CLAUDE.md for the full statement of this.

---

## Current state

Everything below is live on `main` and deployed.

- Freeze removed; per-shot video clips with 1× / 0.5× / 0.25× playback
- Shot log written in plain English (steady / drifted / which shot stood out),
  judged only against the owner's own shots that session — never against an
  invented ideal
- Phantom-shot gating: a movement must get near full draw and last long enough
- Jitter work: One Euro smoothing, region-of-interest cropping, `full` pose
  model with automatic fallback to `lite`
- Left–right mirror toggle (flips canvas pixels, so clips match the view)
- Measurement fidelity: pipeline settling, crop-box stability, median-of-hold
  scoring, and a fix for aspect-ratio distortion in all geometry
- Startup watchdog that names the stuck step instead of hanging silently
- Clip failures now explain themselves on the row
- Routine-start attention gating (idles between arrows, fails toward recording).
  **Note: this was inert from the day it merged until 2026-08-23** — see the
  merge-bug lesson below. Any earlier claim of battery savings was not measuring
  what it thought it was.

---

## The plan

### Stage 1 — Two bugs, ship first — ✅ DONE 2026-08-23

Independent of everything else. Small. The owner hits 1a every time he checks
the log.

**1a. The shot log traps him.** It is a full-screen overlay that covers the 📋
button that dismisses it. Give it its own always-reachable close control, keep
the control row above it, and consider dismiss-on-tap-outside. Verify on a
phone-sized viewport: open the log, close it.

**1b. The picture stretches when launched from the Home Screen.** The canvas
draws into a buffer at the camera's native resolution; CSS then scales that
buffer to whatever box it is given, and a canvas has no `object-fit`, so a
mismatched box distorts the image. Without the browser address bar the box is
taller and the mismatch shows. Fix: constrain the displayed box to the camera's
aspect ratio and letterbox the remainder.

Two things must be checked, because this is where a careless fix does damage:
the skeleton must stay registered with the body after letterboxing, and clips
must still record the full frame undistorted. Measurements should be unaffected
(geometry runs in pixel space) — confirm rather than assume.

### Stage 2 — Recording feedback

**The gap left when freeze was removed.** He has no way to tell, mid-end,
whether a shot was recorded. He asked for visual cues and suggested the routine
detector could drive them. He is right: it is the component that knows when a
shot starts.

Two pieces of information, and probably only two:

- **"It can see me"** — a persistent, glanceable indicator that tracking is alive.
- **"That arrow counted"** — an unmistakable change when a shot enters the log.

Constraints: readable at five metres in sunlight, so screen-edge colour rather
than text or small icons; never something he must respond to; the log remains
the authoritative record, so this is reassurance only. Distinguish "recording
this draw" from "logged it" — he will want to know a draw was *seen* even if it
is later rejected.

Depends on the routine detector, which is already built and live.

### Stage 3 — Interface rethink

The substantial piece. Do it once, deliberately.

**The framing: the app has three modes and currently pretends it has one.**

- **Setup** — at the phone, can read and tap. Handedness, mirror, framing check,
  and later the calibration routine. Wants controls and detail.
- **Shooting** — five metres away, glancing only. Wants near-nothing: the Stage 2
  indicators, large. The three readout panels do not belong here; nobody can
  read `165°` from the shooting line.
- **Review** — back at the phone. Log, clips, numbers, the consistency lines.

Everything currently shares one cluttered screen, which is why it reads as a
prototype: each mode is compromised by the others' furniture.

**DESIGN SETTLED 2026-08-23** — proposal reviewed and approved by the owner. Decisions:

- **One boolean, not a three-mode enum.** Review already exists: `#shotlog` is a
  full-screen overlay with its own close control and tap-outside-to-dismiss.
  Setup is the resting state, Review is the log open over it, Shooting is one
  new `shootingMode` flag. Do NOT thread a `mode` value through `renderLoop`,
  `updateCue`, or any measurement function — if a signature has to change for
  this, the design has drifted from "purely presentational".
- **Modes are presentational only.** Detection, `trackShotAttempt`, clip
  recording and the cue state machine run identically in all three. No
  detection/logging/recording path may ever branch on the current mode. This is
  what makes a wrong mode cost a confusing screen rather than a lost arrow.
- **`#cue` stays visible in every mode.** Not Shooting-only. If the owner props
  the phone and forgets to tap "Start shooting", a Shooting-only cue would leave
  him with no signal for a whole end — and its absence looks identical to the
  calm state from five metres. Always-on deletes that failure class for free.
- **`#readouts` move into Setup** as a pre-flight check (is the camera reading
  me confidently, is handedness right), hidden while Shooting and while
  reviewing. **The owner confirmed he shoots solo** — no coach watches the phone
  mid-end, so there is no live-numbers-for-a-coach case to design for. If that
  ever changes it is a separate scoped feature, not a tweak to Shooting mode.
- **Entry/exit:** a "Start shooting" button in Setup; tap anywhere on the
  Shooting screen to return to Setup. **The button must be inert until startup
  completes** (same readiness signal `#status` uses) or an early tap drops him
  into a Shooting screen with no live cue.
- **Calibration is automatic** (owner's choice): measured passively while he is
  stood in Setup, silent when it agrees with the stored value, speaking up only
  on a mismatch. No "Calibrate now" button. It and the framing check live in
  Setup as plain-English status lines, following the shot log's existing banner
  pattern.

Original brief follows.

**Deliverable: a design proposal first** — how modes are entered and left, what
each shows, what happens to the readout panels — approved by the owner before
any code is written.

### Stage 4 — Calibration routine (owner's idea)

At startup, before he walks off, capture his body **proportions** without the
bow: arm length against torso, shoulder width against torso, head size against
torso. Proportions, not absolute sizes — he moves closer and further, and
absolute pixel measurements do not survive that.

Three uses, in order of value:

1. **The app can catch its own nonsense.** It once reported his wrists 2.3
   torso-lengths apart and had no idea that was impossible. A reference lets it
   say so in the log instead of silently recording garbage.
2. **A stable scale.** Every ratio divides by a torso length re-estimated each
   frame from noisy landmarks. That noise caused a real measurement bias (see
   the median-of-hold decision in CLAUDE.md). A reference bounds it.
3. **A framing check before he walks away** — "you are fully in frame" or "your
   legs are cut off" — at the one moment he can act on it.

**Framing signature (decided 2026-08-23, in build)**: calibration also records
an optional *framing signature* — apparent size in frame, position in frame, and
a squareness proxy (side-on, the two shoulders project nearly on top of each
other; that separation grows as the archer rotates toward the camera). Compared
against the stored one each session so the app can say "you are set up
differently from last time" at the one moment he can still fix it.

Why it exists: the owner is considering tracking form **across** sessions. The
blocker there is not storage, it is comparability — a metre further back, a
higher phone, or 10° off perpendicular moves every number for reasons unrelated
to his form. Without a controlled setup, a cross-session trend line narrates the
tripod back to him with total confidence, which is the cry-wolf failure again in
a more persuasive costume. Note also that within-session **spread** survives the
trip across sessions far better than absolute values do, because it is his shots
measured against each other and setup differences largely cancel — so any future
tracking should lead with repeatability, not averages. Cross-session tracking is
NOT built and may never be; the signature must be worth having on its own.

**Decided**: store it, and re-measure each session to check against the stored
value rather than trusting it blindly (see "Answered by the owner" below). This
is the first thing the app persists across reloads — a deliberate exception to
the no-persistence rule, not an erosion of it. Nothing else should follow it
across without the same explicit decision.

Belongs inside setup mode, so build it with or after Stage 3 rather than twice.

### Sequencing

```
Stage 1  bugs             ── start now, independent
Stage 2  feedback         ── routine detector already live
Stage 3  interface        ── design approval before code
Stage 4  calibration      ── folds into setup mode
```

Stage 1 can run in parallel with anything. Stages 2–4 are best done in order;
3 absorbs 2's surfaces and 4 lives inside 3.

---

## Also outstanding

- ~~**Remove the `?testhooks` block from `app.js`.**~~ ✅ Done 2026-08-23 — 29 lines, nothing depended on it. Test scaffolding shipped into
  the app by one engineer. Inert in normal use, but every other engineer
  verified from outside without it, and CLAUDE.md keeps this file minimal.
- **Retune the detection thresholds.** `FULL_DRAW_HAND_SEP_MIN`,
  `DRAW_ATTEMPT_MIN_SEP`, `FULL_DRAW_ANCHOR_MAX`, `FULL_DRAW_STILL_MAX`,
  `SHOT_MIN_PEAK_SEP_FRACTION`, and the two `ATTENTION_REST_*` constants were
  all chosen against the old, distorted geometry. **Do not guess at new values.**
  Get a real session's `hand sep` figures from the shot log's small print first
  — it should now read near 1, not the 2.3 that exposed the bug.
- **Clips may still not record on his iPhone.** Failures now name themselves on
  the row. The next field report tells you which limitation is being hit. If
  every row says the same thing, canvas-stream recording is likely unviable on
  that Safari version, and the fallback (record the plain camera stream, losing
  the skeleton overlay) becomes a real design decision for the owner.

---

## Answered by the owner (2026-08-23)

1. **Mid-end, he needs exactly two things**: *"it can see me"* and *"that arrow
   counted"*. He explicitly ruled out a running arrow count and anything else.
   **The shooting mode can therefore be almost empty** — two indicators, large,
   and nothing else. Stage 3 should not try to keep the readout panels alive in
   some reduced form; they belong to review.
2. **Calibration should persist, AND be re-measured every session and checked
   against the stored value.** His words: "remember it but recheck it everytime
   against remembered data." This is better than either option offered — it
   keeps the reference stable across reloads while removing the risk a stale
   measurement is silently in use. Treat a disagreement between the fresh
   measurement and the stored one as something to surface, not to resolve
   silently: it means either the calibration was bad or the framing has changed,
   and both are worth telling him at the one moment he is standing at the phone.

---

## Hard-won lessons — do not repeat these

**The test environment does not resemble the device.** Everything here runs
headless Chromium on a desktop viewport. **Correction (2026-08-23): the claim
that the sandbox cannot reach `cdn.jsdelivr.net` and stubs MediaPipe is wrong
for the current environment** — three engineers independently loaded the real
library (`Graph successfully started running`). What is actually unavailable is
the *camera*: `getUserMedia` is denied outright, so anything past it must be
driven with a stub or an injected stream. Check for yourself rather than
inheriting either claim. The owner shoots on a *portrait*
iPhone with the real library. Two serious bugs lived entirely in that gap: a
startup hang on real Safari, and aspect-ratio distortion invisible to fixtures
built in the same distorted space. **Test in portrait. State plainly which
claims are proven and which are reasoned.**

**A clean merge is not a correct merge.** The most expensive bug found so far
was written by nobody: a merge combined two individually-correct branches into a
call site passing three arguments to `torsoLength`, which the other branch had
changed to need five. The result was `NaN`, which is falsy, so a guard clause
swallowed it and the whole attention-gating feature silently switched itself off
for weeks with no error anywhere. Git reported no conflict. **After any merge
that touches `app.js`, re-run `?selftest` against the merged result and check
the arity of every call to functions either branch changed.** Four separate
worktrees were merged on 2026-08-23 and verified this way.

**Verify which port your server actually bound.** Several servers run in this
sandbox. One that fails to bind leaves you testing another directory's files —
this produced a "both builds pass" result that was impossible, and nearly hid a
real bug. Fetch a known string from your own port before trusting any result.

**Never let a component cry wolf.** The owner restarted the app repeatedly
because a watchdog told him it was broken when it was not. A false alarm costs
more than a missing one: it destroys trust in every other line the log shows.
Any new warning needs a test proving it stays silent when nothing is wrong.

**No test hooks in `app.js`.** Verify from outside — hook browser prototypes,
read the DOM, drive the real pipeline with a stub that returns crop-relative
coordinates (the app crops before inference).

**The app must never claim to judge form.** Every target range under CALIBRATE
WITH COACH is an unvalidated placeholder. The log may only report repeatability
against the owner's own shots. No scores, no verdicts, no invented thresholds.
This is a standing rule in CLAUDE.md, not a preference.

**Fail toward recording.** The owner's explicit instruction for the routine
detector: a phantom row is visible and he can report it; a missed arrow is
invisible and he never knows. Keep the ignored count visible. Do not quietly
reverse this.
