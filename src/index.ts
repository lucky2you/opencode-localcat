/**
 * opencode-localcat — plugin entry point supporting both the opencode V2 and
 * V1 plugin APIs from a single default export.
 *
 * V2 reads the spread `Plugin.define({ id, setup })` and calls `setup(ctx)`,
 * contributing providers through `ctx.provider.transform`.
 *
 * V1 ignores `id`/`setup` and calls `server(ctx, options)`, using the returned
 * `config` hook to inject `provider.<id>` blocks.
 *
 * Both paths discover the same three OpenAI-compatible backends (llama.cpp,
 * litellm, workbuddy-wild) and share one in-process discovery cache.
 *
 *   V2 opencode.jsonc:
 *     {
 *       "plugins": [
 *         { "package": "opencode-localcat",
 *           "options": { "backends": { "llama-cpp": { "baseURL": "http://127.0.0.1:8080/v1" } } } }
 *       ]
 *     }
 *
 *   V1 opencode.jsonc:
 *     {
 *       "plugin": [
 *         ["opencode-localcat",
 *          { "backends": { "llama-cpp": { "baseURL": "http://127.0.0.1:8080/v1" } } }]
 *       ]
 *     }
 *
 * A backend that is down or returns no usable models is skipped
 * (warn-logged); the others still register. Discovery is cached per distinct
 * options object for the process lifetime; to pick up models added to a
 * backend after opencode started, restart opencode. See README.md for docs.
 */

import { Plugin } from "@opencode/plugin";
import { discoverAll } from "./discovery.js";
import { toProviderConfig, toProviderRegistration } from "./normalize.js";
import type { ConfigProvider, ProviderRegistration } from "./normalize.js";
import type { DiscoveredProvider, PluginOptions } from "./types.js";

// ---------------------------------------------------------------------------
// V1 plugin API types (structural; not imported from the V1 package)
// ---------------------------------------------------------------------------

/** Minimal structural view of the opencode config we mutate in V1. */
interface OpenCodeConfig {
  provider?: Record<string, ConfigProvider>;
  [key: string]: unknown;
}

interface PluginHooks {
  config: (config: OpenCodeConfig) => Promise<void>;
}

// ---------------------------------------------------------------------------
// Shared discovery (one probe per distinct options object)
// ---------------------------------------------------------------------------

/**
 * Cache discovery by serialized options. opencode may run setup/config more
 * than once per process (e.g. per location/instance); a backend probe can take
 * several seconds, so we only pay for it once. Stores the online backends
 * (each with at least one model); each API path then maps them to its own shape.
 */
const discoveryCache = new Map<string, Promise<DiscoveredProvider[]>>();

function discoverOnlineBackends(opts: PluginOptions): Promise<DiscoveredProvider[]> {
  const key = JSON.stringify(opts ?? {});
  const cached = discoveryCache.get(key);
  if (cached) return cached;

  const pending = (async () => {
    const records = await discoverAll(opts);
    const online: DiscoveredProvider[] = [];
    for (const backend of records) {
      if (!backend.online || backend.modelIds.length === 0) continue;
      online.push(backend);
      console.log(
        `[local-models] ${backend.id}: ${backend.modelIds.length} models from ${backend.baseURL}`,
      );
    }
    return online;
  })();

  discoveryCache.set(key, pending);
  return pending;
}

// ---------------------------------------------------------------------------
// V2 plugin API
// ---------------------------------------------------------------------------

const v2 = Plugin.define({
  id: "local-models",

  async setup(ctx) {
    const opts = ctx.options as PluginOptions;

    let registrations: ProviderRegistration[] = [];
    try {
      const backends = await discoverOnlineBackends(opts);
      registrations = backends.map(toProviderRegistration);
    } catch (err) {
      // A discovery/parse failure must never prevent opencode from loading —
      // log and continue with whatever the user already has.
      console.warn(
        `[local-models] discovery failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
    if (registrations.length === 0) return;

    // Transforms are replayable, synchronous edits; discovery already ran above
    // so we capture the result and only feed `editor.add` in the callback.
    await ctx.provider.transform((editor) => {
      for (const registration of registrations) {
        editor.add(registration);
      }
    });
  },
});

// ---------------------------------------------------------------------------
// Default export: V2 reads `id`/`setup`, V1 calls `server`
// ---------------------------------------------------------------------------

export default {
  ...v2,

  async server(_ctx: unknown, options?: PluginOptions): Promise<PluginHooks> {
    const opts: PluginOptions = options ?? {};

    return {
      async config(config: OpenCodeConfig): Promise<void> {
        try {
          const backends = await discoverOnlineBackends(opts);
          if (backends.length === 0) return;
          config.provider = config.provider ?? {};
          for (const backend of backends) {
            config.provider[backend.id] = toProviderConfig(backend);
          }
        } catch (err) {
          // A discovery/parse failure must never prevent opencode from loading
          // its config — log and continue with whatever the user already has.
          console.warn(
            `[local-models] config hook failed: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
      },
    };
  },
};
