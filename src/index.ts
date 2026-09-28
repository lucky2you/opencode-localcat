/**
 * opencode-localcat — OpenCode V2 plugin entry point.
 *
 * Registers three OpenAI-compatible providers (llama.cpp, litellm,
 * workbuddy-wild) into OpenCode's provider registry on startup, then
 * re-fetches each backend's /v1/models on a periodic interval so newly
 * loaded llama.cpp presets or newly added litellm/workbuddy-wild models
 * appear in /models without an OpenCode restart.
 *
 * Configuration lives in opencode.jsonc → plugins[].options:
 *
 *   {
 *     "plugins": [
 *       {
 *         "package": "file:///path/to/opencode-localcat",
 *         "options": {
 *           "refreshIntervalMs": 300000,
 *           "backends": {
 *             "llama-cpp":      { "baseURL": "http://127.0.0.1:8080/v1" },
 *             "litellm":        { "baseURL": "http://127.0.0.1:4000/v1", "apiKey": "..." },
 *             "workbuddy-wild": { "baseURL": "http://127.0.0.1:7863/v1", "apiKey": "WorkBuddy2API" }
 *           }
 *         }
 *       }
 *     ]
 *   }
 *
 * A backend that's down or returns no models is skipped silently (warn-logged);
 * the other two still register. See README.md for full docs.
 */

import { Plugin } from "@opencode/plugin";

import { discoverAll } from "./discovery.js";
import { toProviderRecord } from "./normalize.js";
import type { PluginOptions } from "./types.js";

const DEFAULT_REFRESH_MS = 5 * 60_000; // 5 minutes

export default Plugin.define({
  id: "local-models",
  async setup(ctx) {
    const opts: PluginOptions = (ctx.options as PluginOptions | undefined) ?? {};
    const refreshMs = typeof opts.refreshIntervalMs === "number" && opts.refreshIntervalMs > 0
      ? opts.refreshIntervalMs
      : DEFAULT_REFRESH_MS;

    // First pass: discover all configured backends. We capture the result in
    // a mutable `source` object so the periodic refresh can swap it out.
    const source = { records: await discoverAll(opts) };

    // Register everything we found. editor.add() is idempotent within a
    // single transform replay, so re-running this transform (after reload)
    // produces the same provider set.
    await ctx.provider.transform((editor) => {
      for (const backend of source.records) {
        if (!backend.online || backend.modelIds.length === 0) continue;
        const { info, models } = toProviderRecord(backend);
        editor.add({ info, models });
      }
    });

    // Sanity check: log what we registered so a missing /models entry is
    // easy to trace in the OpenCode log.
    for (const b of source.records) {
      if (b.online) {
        console.log(`[local-models] ${b.id}: ${b.modelIds.length} models from ${b.baseURL}`);
      } else if (b.error) {
        console.warn(`[local-models] ${b.id} offline: ${b.error}`);
      }
    }

    // Periodic refresh. We re-discover all backends and call ctx.provider.reload()
    // so OpenCode replays every active transform with the new model list.
    // The interval is intentionally not aligned to wall-clock time — drift is
    // fine, and a startup surge that delays the first refresh by a few seconds
    // is acceptable.
    const timer = setInterval(async () => {
      try {
        const next = await discoverAll(opts);
        const prevById = new Map(source.records.map((r) => [r.id, r]));
        // Only reload if something actually changed — avoids churn when
        // llama.cpp has been quietly serving the same model all day.
        if (!recordsEqual(prevById, next)) {
          source.records = next;
          await ctx.provider.reload();
          for (const b of next) {
            if (b.online) {
              console.log(`[local-models] refresh ${b.id}: ${b.modelIds.length} models`);
            }
          }
        }
      } catch (err) {
        // Never let a refresh failure kill the timer — log and keep going.
        console.warn(
          `[local-models] refresh failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }, refreshMs);

    // If the user unloads the plugin (or OpenCode exits), stop the timer.
    // Returning a cleanup function is the V2 plugin contract for teardown.
    return () => {
      clearInterval(timer);
    };
  },
});

/**
 * Compare two discovery result sets by id+modelIds. Used to decide whether
 * a refresh actually warrants a provider.reload() call. We deliberately do
 * NOT compare extended metadata — if the model list is unchanged, the
 * capability details are assumed unchanged too (and OpenCode already has
 * them from the previous transform).
 */
function recordsEqual(
  prev: Map<string, { id: string; modelIds: string[]; online: boolean }>,
  next: readonly { id: string; modelIds: string[]; online: boolean }[],
): boolean {
  if (prev.size !== next.length) return false;
  for (const n of next) {
    const p = prev.get(n.id);
    if (!p) return false;
    if (p.online !== n.online) return false;
    if (p.modelIds.length !== n.modelIds.length) return false;
    for (let i = 0; i < p.modelIds.length; i++) {
      if (p.modelIds[i] !== n.modelIds[i]) return false;
    }
  }
  return true;
}

