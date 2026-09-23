/**
 * A trigger for the publishing workflows that GitHub's own scheduler cannot drop.
 *
 * Measured on this repo 2026-09-14/15 (see docs/LOG.md and tools/schedule_report.py): GitHub's
 * `schedule` event fired 2 of its 6 attempts that day — the rest were dropped outright, not
 * delayed, with no trace left anywhere in the Actions tab. In the same window, every single
 * `workflow_dispatch` call — every manual run and every one made from this Worker's own testing
 * — fired immediately. GitHub sheds *scheduled* runs under load; it queues dispatched ones like
 * any other event. So this calls `workflow_dispatch` on a schedule GitHub does not control
 * (Cloudflare's Cron Triggers), instead of trying to out-guess which minute is safer to schedule
 * on. It does not replace the GitHub-native crons in .github/workflows/ — both keep running, and
 * a week only fails to publish if every slot in both systems misses.
 *
 * Each tick dispatches exactly ONE workflow — the one CRON_TARGETS maps its own cron expression
 * to — not all three. An earlier version dispatched all three on every tick, on the theory that
 * predict.py/grade.py's own no-op gates made triggering the "wrong" one at the "wrong" time
 * harmless. That missed a real interaction: all three workflows share the `data-writes`
 * concurrency group (so a predict pass and a grade run can never race on `git push`), and GitHub
 * Actions concurrency groups hold at most one running run plus one queued run — a *third*
 * simultaneous dispatch in the same group is cancelled outright, not queued. Verified live: the
 * first `/dispatch-now` test after deploying fired all three, and `grade` came back
 * `"conclusion": "cancelled"` with zero jobs ever created — cancelled before it started, on
 * every single tick, not just that one test. CRON_TARGETS removes the contention instead of
 * living with it: nothing this Worker fires ever collides with anything else it fires.
 *
 * TUESDAY EARLY-OPENER TICK ("35 18 * * 3"): predict-early.yml's Tuesday safety net (for
 * Thanksgiving/Christmas/Wednesday-opener weeks) is the one failure mode in this whole system
 * with no fallback if GitHub drops it — a missed early game is gone from the record permanently,
 * unlike a missed grade (self-heals) or a missed regular Thursday pass (has 6 GitHub attempts
 * plus this Worker's own Thursday tick behind it). This tick closes that gap, timed after
 * GitHub's own four Tuesday attempts as the final catch-all, same reasoning as the grade tick
 * below. It needs `CRON_INPUTS` because a plain workflow_dispatch call has no
 * `github.event.schedule` to match against predict-early.yml's Tuesday-detection logic — without
 * the `only_early_openers: true` input, this tick would hit the *default* branch instead and
 * publish the full week unconditionally, every single Tuesday, not just early-opener ones.
 *
 * SEASON GUARD: dispatching outside Sep-Feb would hit predict.py's `next_week_target`, which
 * raises when there is no game left in the loaded schedule — a red run, every tick, for seven
 * months a year, for no reason. Skipped here rather than in Python so the existing GitHub-native
 * crons (which have the same latent gap, already accepted before this Worker existed — see
 * docs/LOG.md) are left alone; fixing that gap generally is a separate, un-scoped piece of work.
 *
 * Setup: see README.md in this directory.
 */

const OWNER = "Kanishk234";
const REPO = "nfl-game-predictor";
const REF = "main";

// Every cron in wrangler.toml, mapped to the one workflow it exists to trigger — see the header
// comment above for why this is a 1:1 map rather than "dispatch all three every tick". Kept as
// filenames, not workflow IDs, so a rename in .github/workflows/ is easy to catch; kept as exact
// cron-string keys (Cloudflare's own weekday convention, matching wrangler.toml verbatim) so
// tests/test_cloudflare_worker.py can assert this map and wrangler.toml's `crons` list never
// drift apart — a cron declared in one without an entry in the other is either a silent gap (a
// tick that dispatches nothing) or dead code (a mapping nothing ever triggers).
const CRON_TARGETS = {
  "11 20 * * 5": "predict-early.yml",
  "35 9 * * 1": "predict-late.yml",
  "51 14 * * 1": "predict-late.yml",
  "13 16 * * 3": "grade.yml",
  "35 18 * * 3": "predict-early.yml",
};

// Extra `workflow_dispatch` inputs for the one tick that needs them, keyed the same as
// CRON_TARGETS. Without this, "35 18 * * 3" would hit predict-early.yml's default branch (a
// full, unconditional publish) instead of the Tuesday-only-if-the-week-opens-early gate — see
// predict-early.yml's own `only_early_openers` input for why plain workflow_dispatch calls
// can't reach that gate any other way (there is no `github.event.schedule` to match on).
const CRON_INPUTS = {
  "35 18 * * 3": { only_early_openers: "true" },
};

// The full set, for the /dispatch-now manual check only — see fetch() below for why that path
// still fires all three despite the contention CRON_TARGETS exists to avoid on real ticks.
const ALL_WORKFLOWS = ["predict-early.yml", "predict-late.yml", "grade.yml"];

/** True for the months an NFL season can have a game still to predict: kickoff week 1 in
 * September through the Super Bowl in February. */
function inSeason(now) {
  const month = now.getUTCMonth() + 1; // getUTCMonth is 0-indexed
  return month >= 9 || month <= 2;
}

async function dispatchOne(env, workflow, inputs = {}) {
  const url = `https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${workflow}/dispatches`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "nfl-predict-cron-worker",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: REF, inputs }),
  });
  // A successful dispatch is 204 No Content and has no body to read.
  const detail = res.ok ? null : await res.text();
  return { workflow, status: res.status, ok: res.status === 204, detail };
}

async function dispatchMany(env, workflows) {
  return Promise.all(workflows.map((w) => dispatchOne(env, w)));
}

function logResults(results) {
  for (const r of results) {
    if (r.ok) {
      console.log(`dispatched ${r.workflow}`);
    } else {
      console.error(`dispatch failed: ${r.workflow} -> HTTP ${r.status} ${r.detail ?? ""}`);
    }
  }
}

export default {
  async scheduled(event, env, ctx) {
    if (!inSeason(new Date(event.scheduledTime))) {
      console.log("off-season; skipping dispatch");
      return;
    }
    // event.cron is the exact matched expression from wrangler.toml. A miss here means the two
    // files have drifted (see tests/test_cloudflare_worker.py) — dispatch nothing rather than
    // guess and reintroduce the concurrency contention CRON_TARGETS exists to avoid.
    const workflow = CRON_TARGETS[event.cron];
    if (!workflow) {
      console.error(`no CRON_TARGETS entry for cron "${event.cron}" — dispatching nothing`);
      return;
    }
    const inputs = CRON_INPUTS[event.cron] ?? {};
    ctx.waitUntil(dispatchOne(env, workflow, inputs).then((r) => logResults([r])));
  },

  // A plain GET so the wiring (token, permissions, workflow names) can be checked from a
  // browser without waiting for a cron tick. Ignores the season guard on purpose — this is a
  // manual, deliberate call, the same as clicking "Run workflow" in the GitHub UI. Fires all
  // three at once, unlike a real tick: this is a one-off wiring check, not a production
  // dispatch, so the `data-writes` concurrency group cancelling one of the three (see the header
  // comment) is an expected, harmless side effect here — what this endpoint verifies is that all
  // three dispatch *calls* succeed (right token, right permissions, right workflow names), not
  // that all three runs complete.
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/dispatch-now") {
      return new Response(
        "nfl-predict cron worker.\nGET /dispatch-now to trigger predict-early, predict-late " +
          "and grade by hand (bypasses the season guard, same as a manual run in the GitHub UI; " +
          "may show one of the three as GitHub-cancelled afterward — that is the shared " +
          "concurrency group, not a dispatch failure — see worker.js).\n",
        { status: 200 },
      );
    }
    const results = await dispatchMany(env, ALL_WORKFLOWS);
    return new Response(JSON.stringify(results, null, 2), {
      status: results.every((r) => r.ok) ? 200 : 502,
      headers: { "Content-Type": "application/json" },
    });
  },
};
