# Archery Form Coach

A web page that watches you shoot and tells you three things about your form.
Point your phone at yourself, shoot as normal, and check the numbers afterwards.
Nothing to install.

**https://cuicuocua.github.io/archery-form-coach/**

Open that in Safari on your iPhone. Add it to your home screen and it launches
like an app. The link never changes.

Currently set up for **compound** shooting.

---

## Using it at the range

**Set the phone up.** Side-on to you — square to your shooting line, so the
camera sees your profile. Your whole body in frame, feet to head. If you're cut
off at the knees the measurements go wrong, because everything is scaled against
your torso length.

**Check the handedness button** matches how you shoot, before your first arrow.
It decides which arm is scored as the bow arm and which as the draw arm. Get it
backwards and every number is nonsense.

**Then shoot.** Don't touch the phone — you can't, mid-shot, and the app knows
it. There used to be a freeze button here; it's gone, because you're five
metres away holding a drawn bow and can never tap it, and by the time you walk
over, whatever it froze has long since released itself. Everything the app
needs you to see instead gets recorded to the shot log as it happens — including
a short video of every draw, so you can watch the shot itself, not just the
numbers it produced.

**Afterwards, tap 📋 Log.** That's the one interaction you need, and you do it
standing at the phone when you've finished. It lists your last ten draws.

---

## The three readouts

**Bow-arm line** — the angle at your bow elbow. 180° is a perfectly straight
arm.

**Shoulder drop** — how far each shoulder sits below the ear on that side, as a
percentage of your torso length. Bigger is more dropped. Shown separately for
bow side and draw side, because the usual fault is one shoulder creeping up
under load while the other stays put.

**Elbow ↔ arrow line** — whether your draw elbow sits on the line running from
your bow hand through your draw hand and out behind you. Reads "in line", or
tells you how far off and which way: "6° high", "4° low".

Note what this one can't see. From side-on the camera knows whether your elbow
is too high or too low, but it cannot tell whether it's flared out sideways from
the arrow line — that's depth, and it's invisible from that angle. It measures
a real half of the problem, not all of it.

**Green or amber** tells you whether a number is inside its target range.
**Grey "— uncertain"** means the camera couldn't get a confident read on that
joint — common side-on when one arm hides behind the other. Trust the skeleton
and your own eyes over a guessed number; the app will never show you a confident
figure it isn't sure of.

---

## The shot log

Every draw is recorded, including ones where you never reached full draw at
all. That's deliberate — a near-miss is exactly the case worth knowing about,
and an empty log would teach you nothing.

Each row gives the shot number and all three readouts from the moment you were
most fully drawn.

The log holds the last ten draws and lasts until you reload the page. Reloading
clears it. There's no history between sessions.

---

## Watching a shot back

Every row that has a clip gets a green **▶ Watch** button. Tap it and the clip
opens full-screen: the exact video the app was recording, skeleton and all,
covering the whole shot — raise, draw, hold, release, follow-through. Three
buttons let you slow it down: **1×**, **0.5×**, **0.25×**. Full speed is
basically useless for checking your own form; slow it down. **✕ Close** takes
you back — the camera never stopped running behind the player, so nothing
needs restarting.

One clip per shot, not one long recording of the whole session. That's on
purpose: a single running video would mean scrubbing through several minutes
to find the ten seconds that mattered, standing at the phone, which is exactly
the kind of thing this app exists to avoid.

A row without a clip shows a plain **no clip** note instead of the button —
your numbers are still there either way. That happens if your browser can't
record video at all (rare) or if recording happened to fail for just that one
shot; either way it never hides the shot itself.

**Clips live in memory only**, exactly like the log — they never touch disk,
and reloading the page loses them along with everything else. Watch them
before you reload.

If clip recording runs into trouble, an amber line appears at the top of the
log saying so, and it stays there for the rest of the session — not just a
message that flashes by while you're not looking. It says one of two things,
depending on what actually went wrong: if your browser can't record video at
all, it says that plainly, and no row will ever have a clip; if recording
generally works but one shot's clip failed, it says a shot or two might be
missing a clip rather than claiming nothing works, since later shots can
still succeed. Either way, everything else (the readouts, the shot log, the
numbers) keeps working regardless.

---

## Buttons

- **🔄 Camera** — rear or front camera.
- **🎯 Right-handed / Left-handed** — which arm is which. Set before shooting.
- **📋 Log** — show or hide the shot log.

---

## The numbers are placeholders

The green/amber ranges are estimates, not coaching standards. They live in one
labelled block at the top of `app.js` called **CALIBRATE WITH COACH**. Nothing
else in the file needs touching to change them.

Two other labelled blocks sit right next to it — **SMOOTHING** and **POSE
MODEL** — but those are a different thing entirely: performance knobs, not
form targets. No coach needed for those; change them and judge by watching
the skeleton (live or in a recorded clip). See "Steadying the skeleton"
below.

The shoulder-drop threshold in particular is a guess made at a desk from human
proportions. To set it properly: in one session, shoot a few shots with your
shoulders deliberately shrugged and a few with them deliberately dropped as far
as you can. Read both off the log. The real threshold sits between the two
clusters.

---

## Steadying the skeleton

The camera picks up real jitter, especially outdoors at five metres — you take
up a small part of the frame, so the pose model's guess for a joint wobbles a
little frame to frame even when you're holding dead still. The app smooths
this out itself before it draws the skeleton, feeds a readout, or logs a shot,
using a filter that eases off automatically the moment you actually move —
so it can smooth hard while you're motionless at full draw (exactly when
every number gets read) without dragging the skeleton behind you during the
raise.

The app also tries the steadier of MediaPipe's two pose models by default, and
quietly switches to the lighter one on its own if your phone can't keep it
running fast enough — you'll never see this happen, but the shot log always
shows a small line saying which one ended up running and roughly how many
frames per second it managed. If you ever see this line say "lite
(auto-switched)", it's telling you the full model was too much for that
phone on that day; nothing to act on, just useful to know.

---

## Diagnostic mode

Add `?debug` to the address:

```
https://cuicuocua.github.io/archery-form-coach/?debug
```

This shows a large panel with the live values behind full-draw detection —
how far apart your hands are, how close your draw hand is to your face, how
straight your bow arm is, how still you are — each marked pass or fail. The shot
log gains the same detail per row.

Use it when full draw is being detected at the wrong moment or not at all. The
numbers say which condition is the problem, which turns "it didn't work" into a
specific threshold to change.

---

## For whoever works on the code

Plain HTML, CSS and JavaScript. No build step, no npm, no frameworks. Libraries
load from a CDN at runtime. Everything is in `index.html`, `style.css` and
`app.js`.

Pushing to `main` publishes to the live URL above via GitHub Pages, usually
within a minute or two.

**Self-check:** add `?selftest` to the URL and open the browser console. It runs
assertions over the full-draw detection, the shot-log attempt boundaries, and
the two form calculations. Silence means they passed. There is no test
framework and no test files — this is the whole of it.

**Running it locally**, if you're changing something and don't want to publish
first. Camera access needs HTTPS or localhost, so a plain file won't do:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`. To reach it from a phone before publishing,
put a tunnel in front of it:

```bash
cloudflared tunnel --url http://localhost:8000
```

That prints a temporary `https://` address, different every time. You only need
this for testing unpublished changes on a real phone — for normal use, the
permanent link at the top of this file is the whole story.
