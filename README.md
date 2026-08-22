# Archery Form Coach

A prototype web page that uses your iPhone's camera to draw a skeleton over your
body while you shoot, plus two big form readouts: bow-arm straightness and
draw-elbow height. No app to install — it runs in Safari.

## Every time you want to use it

You need **two things running at once**: a little local web server (serves the
page from your Mac) and a tunnel (gives that page a temporary `https://` web
address your iPhone can reach). Open **two Terminal windows/tabs** for this.

### Terminal window 1 — serve the files
```
cd /Users/teo/Claude/archery-form-coach
python3 -m http.server 8000
```
Leave this running. (To stop it later: click into that window and press `Control+C`.)

### Terminal window 2 — start the tunnel
```
cloudflared tunnel --url http://localhost:8000
```
Wait a few seconds. It will print a line that looks like:
```
https://some-random-words.trycloudflare.com
```
That's your link for this session — copy it.

*(First time only: if `cloudflared` isn't installed yet, run `brew install cloudflared`
once, then the command above.)*

### On your iPhone
1. Open **Safari** (must be Safari, not Chrome, for the camera to work reliably here).
2. Type or AirDrop yourself the `https://...trycloudflare.com` link from above and open it.
3. Tap **Allow** when it asks for camera access.
4. You should see your camera feed with a skeleton overlay and two readouts at the top.

### Controls on the page
- **🔄 Camera** — switches between rear and front camera.
- **🎯 Right-handed / Left-handed** — tap to match how you shoot. This decides which
  arm is scored as the bow arm vs. the draw arm.
- **⏸ Freeze** — holds the current frame so you can study your position. Tap **▶ Resume**
  to continue.

### When you're done
Stop both Terminal windows with `Control+C`. The link stops working once the tunnel
closes — that's expected, you'll get a new one next time.

## Notes
- The link changes every time you start a new tunnel — that's normal for this free,
  no-account tunnel service.
- If a readout shows a grey **"— uncertain"**, the camera couldn't get a confident
  read on that arm (common when one arm hides the other from a side angle) — trust
  the skeleton and your own eyes over a guessed number.
- The green/amber target ranges are **placeholders** — see the "CALIBRATE WITH
  COACH" block at the top of `app.js` to adjust them once you've worked out real
  targets with a coach.
