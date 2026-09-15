# Cron backstop

GitHub's own `schedule` trigger drops runs under load rather than queueing them — measured on
this repo at a 2-in-6 fire rate on 2026-09-14/15 (`python tools/schedule_report.py`). Its
`workflow_dispatch` event has never dropped once in the same history. This is a small Cloudflare
Worker that calls `workflow_dispatch` on `predict-early.yml`, `predict-late.yml` and `grade.yml`
on a schedule GitHub does not control, so a week only fails to publish if *every* slot in *both*
systems misses. See `worker.js` for the full reasoning and `wrangler.toml` for exactly when it
fires.

It does not replace anything — the staggered crons in `.github/workflows/` keep running exactly
as before. This is a second, independent path to the same safe, idempotent workflows.

## One-time setup

You need three things: a Cloudflare account, `wrangler` (Cloudflare's CLI), and a GitHub token
scoped to this repo only. None of it touches the Python package or the existing workflows.

### 1. Cloudflare account

Free tier is enough — 5 Cron Triggers per account, 100k requests/day, this Worker uses 5 and a
few dozen requests a week. Sign up at <https://dash.cloudflare.com/sign-up> if you don't have one.

### 2. `wrangler`

```
npm install -g wrangler
wrangler login
```

`login` opens a browser to authorize the CLI against your Cloudflare account. No install needed
beyond Node — nothing here is added to `pyproject.toml`, this is a separate deployable.

### 3. A GitHub token, scoped to this repo only

Go to <https://github.com/settings/personal-access-tokens/new> (fine-grained, not classic —
classic tokens can't be scoped to one repo).

- **Resource owner:** your account
- **Repository access:** "Only select repositories" → `nfl-game-predictor`
- **Permissions:** Repository permissions → **Actions: Read and write**. That's the only
  permission this token needs or should have — it cannot read code, write commits, or touch any
  other repo.
- **Expiration:** fine-grained tokens cannot auto-renew. Pick something you'll remember to
  rotate — 90 days is reasonable — and put the date in `docs/LOG.md` when you set it up, the same
  way the odds API key's checks are logged. A silently expired token fails the same way GitHub's
  own dropped crons did: quietly, with nothing red until `tools/schedule_report.py` or the
  `/dispatch-now` check below is run by hand.

Copy the token now; GitHub only shows it once.

### 4. Wire the token into the Worker

```
cd tools/cloudflare-worker
wrangler secret put GITHUB_TOKEN
```

Paste the token when prompted. This stores it in Cloudflare, not in any file in this repo —
`wrangler.toml` deliberately has no `[vars]` section for it. Same rule as `ODDS_API_KEY`: it
never lives in a committed file.

### 5. Deploy

```
wrangler deploy
```

Prints a URL like `https://nfl-predict-cron.<your-subdomain>.workers.dev`. That URL is live
immediately; the cron schedule from `wrangler.toml` is registered at the same time.

### 6. Verify it actually works, without waiting for a cron tick

Visit `<your-worker-url>/dispatch-now` in a browser, or:

```
curl https://nfl-predict-cron.<your-subdomain>.workers.dev/dispatch-now
```

You should get back JSON with three entries, each `"ok": true`, and — within a minute or two —
three new `workflow_dispatch` runs in the repo's Actions tab (`predict-early`, `predict-late`,
`grade`, each showing `workflow_dispatch` as the trigger, not `schedule`). If any entry is not
`ok`, the response includes GitHub's own error text — almost always a token permission problem
that redoing step 3 fixes.

`/dispatch-now` bypasses the September–February season guard (see `worker.js`), so this is safe
to run for a wiring check at any time of year — it does real work only if there's a game left to
predict or a result left to grade; otherwise it's the same safe no-op the GitHub crons already
rely on.

## Ongoing

Nothing. The five triggers in `wrangler.toml` fire on their own from here. The only thing that
will ever pull you back into this directory is the token expiring — and `tools/schedule_report.py`
run against `predict-early`/`predict-late`/`grade` will start showing a widening gap between
`workflow_dispatch` events (still arriving from GitHub's own manual runs, if any) and the
Worker's ticks going quiet, if you want to check without waiting for the calendar reminder.

## Updating

Edit `worker.js` or `wrangler.toml`, then `wrangler deploy` again from this directory. Nothing
here is deployed by any GitHub Actions workflow — it is intentionally outside the system it is
backstopping, so a broken deploy pipeline in this repo can never also take out the backstop.
