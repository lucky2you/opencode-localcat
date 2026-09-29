/**
 * opencode-localcat — plugin entry point for the opencode 1.x plugin API.
 *
 * Registers three OpenAI-compatible providers (llama.cpp, litellm,
 * workbuddy-wild) by discovering each backend's `/v1/models` (plus any
 * extended metadata) during config load, then injecting them into opencode's
 * `provider` config via the `config` hook.
 *
 * opencode 1.x calls a plugin as:
 *
 *   export default { id, server(ctx, options) => hooks }
 *
 * and the `config` hook receives the live merged config object, which we
 * mutate in place. Plugin options are passed through the config tuple:
 *
 *   {
 *     "plugin": [
 *       ["file:///path/to/opencode-localcat", {
 *         "backends": {
 *           "llama-cpp":      { "baseURL": "http://127.0.0.1:8080/v1" },
 *           "litellm":        { "baseURL": "http://127.0.0.1:4000/v1", "apiKey": "..." },
 *           "workbuddy-wild": { "baseURL": "http://127.0.0.1:7863/v1", "apiKey": "WorkBuddy2API" }
 *         }
 *       }]
 *     ]
 *   }
 *
 * A backend that is down or returns no usable models is skipped
 * (warn-logged); the others still register. Discovery is cached per distinct
 * options object for the process lifetime. Unlike the V2 API there is no
 * `provider.reload()`, so newly-added backend models appear after an opencode
 * restart. See README.md for full docs.
 */

import { discoverAll } from "./discovery.js";
import { toProviderConfig } from "./normalize.js";
import type { ConfigProvider } from "./normalize.js";
import type { PluginOptions } from "./types.js";

/** Minimal structural view of the opencode config we mutate. */
interface OpenCodeConfig {
  provider?: Record<string, ConfigProvider>;
  [key: string]: unknown;
}

interface PluginHooks {
  config: (config: OpenCodeConfig) => Promise<void>;
}

interface DiscoveredProviderBlock {
  id: string;
  block: ConfigProvider;
}

/**
 * Cache discovery by serialized options. opencode may invoke the config hook
 * more than once per process (e.g. per location/instance); a backend probe
 * can take several seconds, so we only pay for it once.
 */
const discoveryCache = new Map<string, Promise<DiscoveredProviderBlock[]>>();

function discoverProviderBlocks(opts: PluginOptions): Promise<DiscoveredProviderBlock[]> {
  const key = JSON.stringify(opts ?? {});
  const cached = discoveryCache.get(key);
  if (cached) return cached;

  const pending = (async () => {
    const records = await discoverAll(opts);
    const blocks: DiscoveredProviderBlock[] = [];
    for (const backend of records) {
      if (!backend.online || backend.modelIds.length === 0) continue;
      blocks.push({ id: backend.id, block: toProviderConfig(backend) });
      console.log(
        `[local-models] ${backend.id}: ${backend.modelIds.length} models from ${backend.baseURL}`,
      );
    }
    return blocks;
  })();

  discoveryCache.set(key, pending);
  return pending;
}

export default {
  id: "local-models",

  async server(_ctx: unknown, options?: PluginOptions): Promise<PluginHooks> {
    const opts: PluginOptions = options ?? {};

    return {
      async config(config: OpenCodeConfig): Promise<void> {
        try {
          const blocks = await discoverProviderBlocks(opts);
          if (blocks.length === 0) return;
          config.provider = config.provider ?? {};
          for (const { id, block } of blocks) {
            config.provider[id] = block;
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
