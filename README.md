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

**Each shot's numbers are a typical value across the whole hold, not one
instant.** Earlier versions picked the single moment your hands looked
furthest apart and reported the numbers from exactly that moment. That turned
out to be a bad idea: how far apart your hands look and how dropped your
shoulder looks are both worked out using the same rough measurement of your
torso, and on any one frame that measurement can come out a little too small
or a little too large just from camera jitter — so the "hands furthest apart"
frame was quietly more likely to also be the "shoulder looks more dropped"
frame, for no reason to do with your actual form. Now the app takes the
middle value of each of the four numbers across every good frame of your
hold and reports that instead, which can't get pulled around by one noisy
frame the same way. Nothing about how the numbers themselves are calculated
changed, and nothing about which draws count as a real shot changed either —
just what one number represents: not a single snapshot, but what your hold
looked like typically.

Only real draws get a row. The app watches for hand movement that's deep
enough and lasts long enough to plausibly be a draw before it logs anything —
nocking an arrow, lowering the bow, adjusting a release aid, or the camera
briefly losing you mid-movement all get thrown away instead of logged as a
phantom shot. If any movement gets thrown away, a line at the top of the log
says how many: **"N movements ignored (too short, or never near full draw)."**
A big number there next to a small arrow count is itself worth noticing.

The app also needs a moment right after it starts (or right after it briefly
loses track of you) before its readings are trustworthy enough to log — it's
still catching up on how steady you're actually holding. In practice you'll
never notice: you start the app, prop the phone, and walk to the line, which
is already more time than it needs. If a draw somehow lands entirely inside
that catching-up window, it says so in its own line — **"N arrows drawn
before the app finished settling weren't recorded"** — worded differently
from "movements ignored" on purpose, because it means something different:
that was a real arrow, not noise, the app just wasn't ready to measure it.

A draw that never quite reached full draw is still logged, and still worth
seeing — that part hasn't changed — it's just marked **short of full draw**
so you can tell it apart from one that went all the way.

**What the log tells you, and what it deliberately doesn't.** The app has no
way to know what good archery form looks like — every green/amber target
range it uses is a placeholder nobody has tuned with a coach yet (see "The
numbers are placeholders" below). So the log never judges your form. The one
thing it can say with real confidence, needing no calibration at all, is
whether you did the same thing twice — compared only against your own other
shots that end, nothing else.

At the top it says how many arrows it counted, then one line per measurement
telling you plainly whether it was **steady**, whether it **drifted** over
the course of the end (a shoulder creeping up as you tire is a classic one),
or whether **one shot stood out**, named by number so you know exactly which
clip to go watch. With fewer than three shots it says there isn't enough yet
to call anything steady or drifting, rather than guessing from almost
nothing. If a measurement's readings were all "uncertain" that session, it
says nothing about it at all rather than reaching for a claim it can't back
up.

The camera itself has a little wobble in it even when nothing is moving —
that's just how a phone camera at five metres works. The app knows roughly
how much, and won't call that wobble a trend or single out a shot over it:
if a number's own session-to-session variation could plausibly just be the
camera's own noise, it gets called steady, not drifting. The very best
sessions you shoot are exactly where this matters most — a session with
almost no real variation is also the one where ordinary camera noise looks
biggest by comparison.

Each row underneath names the shot, says in plain English what — if anything
— stood out about it, and gives you the Watch button. The raw degrees and
percent are still there too, in small print under each shot, for whoever
eventually tunes those placeholder target ranges against a real session; they
just aren't the headline any more.

The log holds the last ten draws and lasts until you reload the page. Reloading
clears it. There's no history between sessions.

---

## Watching a shot back

Every row that has a clip gets a green **▶ Watch** button. Tap it and the clip
opens full-screen: the exact video the app was recording, skeleton and all,
covering the whole shot — raise, draw, hold, release, follow-through. If the
🪞 mirror toggle was on when you shot, the clip is mirrored too — whatever
you saw live is exactly what got recorded, not a mismatched flip of it. Three
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
- **🪞 Mirrored / Not mirrored** — flips the picture left-right, like looking in a mirror. The
  front camera starts mirrored by default (the normal selfie convention), the rear camera starts
  un-mirrored; this button flips whichever camera you're currently on away from its own default.
  Like the handedness button, it always shows what's true right now, not what tapping it would
  do — set it before you shoot, since you can't reach the phone once you've walked to the line.
- **📋 Log** — show or hide the shot log.

---

## The numbers are placeholders

The green/amber ranges are estimates, not coaching standards. They live in one
labelled block at the top of `app.js` called **CALIBRATE WITH COACH**. Nothing
else in the file needs touching to change them.

Three other labelled blocks sit right next to it — **SMOOTHING**, **POSE
MODEL**, and **REGION-OF-INTEREST CROPPING** — but those are a different
thing entirely: performance knobs, not form targets. No coach needed for
those; change them and judge by watching the skeleton (live or in a recorded
clip). See "Steadying the skeleton" below.

The shoulder-drop threshold in particular is a guess made at a desk from human
proportions. To set it properly: in one session, shoot a few shots with your
shoulders deliberately shrugged and a few with them deliberately dropped as far
as you can. Read both off the log. The real threshold sits between the two
clusters.

---

## Steadying the skeleton

The camera picks up real jitter, especially outdoors at five metres — you take
up a small part of the frame, so the pose model's guess for a joint wobbles a
little frame to frame even when you're holding dead still.

The main fix for that: before the app hands each frame to the pose model, it
crops in on roughly where you were standing last frame — a generously padded
box around you, not a tight one — and zooms that up before the model ever
looks at it. Same camera, same distance, but the model is now looking at a
close-up of you instead of a wide shot with you as a small part of it, so its
guess for each joint is working from far more detail. You never see this
happen; the crop only ever feeds the pose model, never what's drawn on screen
or saved in a clip — both of those always show the whole camera view. If you
ever lose full-draw detection unexpectedly for a step or two, it's likely
because you moved out of the cropped box and the app is re-finding you on the
whole frame again — that recovers on its own within a frame or two, nothing
to do about it.

On top of that, the app also smooths the landmarks it gets back, using a
filter that eases off automatically the moment you actually move — so it can
smooth hard while you're motionless at full draw (exactly when every number
gets read) without dragging the skeleton behind you during the raise.

The app also tries the steadier of MediaPipe's two pose models by default, and
quietly switches to the lighter one on its own if your phone can't keep it
running fast enough — you'll never see this happen, but the shot log always
shows a small line saying which one ended up running, about how many
milliseconds each frame's pose detection took, and the real frame rate the
app actually managed (drawing and everything else included, not just the pose
model). If you ever see this line say "lite (auto-switched)", it's telling
you the full model was too much for that phone on that day; nothing to act
on, just useful to know.

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
assertions over the full-draw detection, the shot-log attempt boundaries and
gating (including that noise gets thrown away and real draws don't), the
plain-language consistency wording, and the two form calculations. Silence
means they passed. There is no test framework and no test files — this is the
whole of it.

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
