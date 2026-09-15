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
 * All three publishing pipelines are dispatched on every tick, not mapped one cron-to-one
 * workflow. This is deliberate, not lazy: predict.py and grade.py are already gated to no-op
 * when there is nothing to do (predict.py, `pred_path.exists()`; the late pass additionally
 * checks LATE_PASS_LEAD_LIMIT; grade.py rewrites nothing when no game has finished), so
 * triggering the "wrong" one at the "wrong" time costs a few seconds of a no-op job, not a bug.
 * Encoding day-of-week logic here as well as in predict.py's own gate would just be two places
 * that can drift out of sync with each other.
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

// The three workflows that write to data/ and publish the site. Kept as filenames, not workflow
// IDs, so this stays readable next to .github/workflows/ and a rename there is easy to catch —
// tests/test_cloudflare_worker.py asserts every name here is a real file in that directory.
const WORKFLOWS = ["predict-early.yml", "predict-late.yml", "grade.yml"];

/** True for the months an NFL season can have a game still to predict: kickoff week 1 in
 * September through the Super Bowl in February. */
function inSeason(now) {
  const month = now.getUTCMonth() + 1; // getUTCMonth is 0-indexed
  return month >= 9 || month <= 2;
}

async function dispatchOne(env, workflow) {
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
    body: JSON.stringify({ ref: REF }),
  });
  // A successful dispatch is 204 No Content and has no body to read.
  const detail = res.ok ? null : await res.text();
  return { workflow, status: res.status, ok: res.status === 204, detail };
}

async function dispatchAll(env) {
  return Promise.all(WORKFLOWS.map((w) => dispatchOne(env, w)));
}

export default {
  async scheduled(event, env, ctx) {
    if (!inSeason(new Date(event.scheduledTime))) {
      console.log("off-season; skipping dispatch");
      return;
    }
    ctx.waitUntil(
      (async () => {
        const results = await dispatchAll(env);
        for (const r of results) {
          if (r.ok) {
            console.log(`dispatched ${r.workflow}`);
          } else {
            console.error(`dispatch failed: ${r.workflow} -> HTTP ${r.status} ${r.detail ?? ""}`);
          }
        }
      })(),
    );
  },

  // A plain GET so the wiring (token, permissions, workflow names) can be checked from a
  // browser without waiting for a cron tick. Ignores the season guard on purpose — this is a
  // manual, deliberate call, the same as clicking "Run workflow" in the GitHub UI.
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/dispatch-now") {
      return new Response(
        "nfl-predict cron worker.\nGET /dispatch-now to trigger predict-early, predict-late " +
          "and grade by hand (bypasses the season guard, same as a manual run in the GitHub UI).\n",
        { status: 200 },
      );
    }
    const results = await dispatchAll(env);
    return new Response(JSON.stringify(results, null, 2), {
      status: results.every((r) => r.ok) ? 200 : 502,
      headers: { "Content-Type": "application/json" },
    });
  },
};
