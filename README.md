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

**Then shoot.** Don't touch the phone. It freezes itself when you reach full
draw, holds the frame about four seconds so the skeleton and the video stay
together, then releases and gets ready for the next arrow.

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

Every draw is recorded, including ones where the auto-freeze never fired. That's
deliberate — "it didn't fire" is exactly the case worth knowing about, and an
empty log would teach you nothing.

Each row gives the shot number, all three readouts from the moment you were most
fully drawn, and whether it froze.

The log holds the last ten draws and lasts until you reload the page. Reloading
clears it. There's no history between sessions.

---

## Buttons

- **🔄 Camera** — rear or front camera.
- **🎯 Right-handed / Left-handed** — which arm is which. Set before shooting.
- **⏸ Freeze** — manual freeze, if you want to hold a frame yourself. It
  overrides the automatic one: tap it and the app won't take the frame back off
  you until you tap again.
- **📋 Log** — show or hide the shot log.

---

## The numbers are placeholders

The green/amber ranges are estimates, not coaching standards. They live in one
labelled block at the top of `app.js` called **CALIBRATE WITH COACH**. Nothing
else in the file needs touching to change them.

The shoulder-drop threshold in particular is a guess made at a desk from human
proportions. To set it properly: in one session, shoot a few shots with your
shoulders deliberately shrugged and a few with them deliberately dropped as far
as you can. Read both off the log. The real threshold sits between the two
clusters.

---

## Diagnostic mode

Add `?debug` to the address:

```
https://cuicuocua.github.io/archery-form-coach/?debug
```

This shows a large panel with the live values behind the auto-freeze trigger —
how far apart your hands are, how close your draw hand is to your face, how
straight your bow arm is, how still you are — each marked pass or fail. The shot
log gains the same detail per row.

Use it when the freeze is firing at the wrong moment or not at all. The numbers
say which condition is the problem, which turns "it didn't work" into a specific
threshold to change.

---

## For whoever works on the code

Plain HTML, CSS and JavaScript. No build step, no npm, no frameworks. Libraries
load from a CDN at runtime. Everything is in `index.html`, `style.css` and
`app.js`.

Pushing to `main` publishes to the live URL above via GitHub Pages, usually
within a minute or two.

**Self-check:** add `?selftest` to the URL and open the browser console. It runs
assertions over the auto-freeze state machine, the full-draw detection, the
shot-log attempt boundaries, and the two form calculations. Silence means they
passed. There is no test framework and no test files — this is the whole of it.

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
