/**
 * Static descriptors for the three backends supported by this plugin.
 *
 * Each descriptor is purely declarative — it does not perform I/O. Discovery
 * lives in discovery.ts; this file just gives the plugin a stable identity
 * (display name, default port, OpenAI-compatible package) per backend.
 *
 * All three backends speak the OpenAI Chat Completions protocol on
 * `/v1/chat/completions`, so the same package routes each of them:
 *   - opencode V1 uses the `@ai-sdk/openai-compatible` npm package
 *     (`packageName`, emitted as `provider.<id>.npm`).
 *   - opencode V2 uses the built-in `@opencode/ai/providers/openai-compatible`
 *     runtime package (see `V2_PACKAGE` in normalize.ts).
 */

import type { BackendConfig, BackendID, ModelOverride } from "./types.js";

export interface BackendDescriptor {
  id: BackendID;
  /** Human-readable name shown by OpenCode in /providers. */
  displayName: string;
  /** Default port if the user omits it from baseURL. */
  defaultPort: number;
  /** Default baseURL prefix used when the user sets enabled:true but omits baseURL. */
  defaultBaseURL: string;
  /** Sentinel used when apiKey is omitted. llama.cpp accepts any string here; workbuddy-wild requires a real key. */
  defaultApiKey: string;
  /** opencode V1 npm package used to route chat completions (`provider.<id>.npm`). */
  packageName: string;
  /** Whether this backend requires a non-trivial apiKey (workbuddy-wild enforces it; llama.cpp typically doesn't). */
  requiresApiKey: boolean;
  /** Default streamed-delta field carrying reasoning tokens. Overridable per backend via options.reasoningField. */
  reasoningField: string;
}

export const BACKENDS: readonly BackendDescriptor[] = [
  {
    id: "llama-cpp",
    displayName: "llama.cpp",
    defaultPort: 8080,
    defaultBaseURL: "http://127.0.0.1:8080/v1",
    defaultApiKey: "no-key-required",
    packageName: "@ai-sdk/openai-compatible",
    requiresApiKey: false,
    reasoningField: "reasoning_content",
  },
  {
    id: "litellm",
    displayName: "LiteLLM",
    defaultPort: 4000,
    defaultBaseURL: "http://127.0.0.1:4000/v1",
    defaultApiKey: "sk-litellm-proxy",
    packageName: "@ai-sdk/openai-compatible",
    requiresApiKey: true,
    reasoningField: "reasoning_content",
  },
  {
    id: "workbuddy-wild",
    displayName: "WorkBuddy-Wild",
    defaultPort: 7863,
    defaultBaseURL: "http://127.0.0.1:7863/v1",
    defaultApiKey: "WorkBuddy2API",
    packageName: "@ai-sdk/openai-compatible",
    requiresApiKey: true,
    reasoningField: "reasoning_content",
  },
] as const;

/** Lookup by ID. */
export function getBackend(id: BackendID): BackendDescriptor {
  const found = BACKENDS.find((b) => b.id === id);
  if (!found) {
    throw new Error(`[local-models] unknown backend id: ${id}`);
  }
  return found;
}

/**
 * Resolve a user-provided BackendConfig against a backend's defaults.
 * baseURL falls back to the descriptor's defaultBaseURL; apiKey falls back to
 * the descriptor's defaultApiKey (which is "no-key-required" for llama.cpp).
 */
export function resolveConfig(
  descriptor: BackendDescriptor,
  raw: BackendConfig | undefined,
): BackendConfig & { baseURL: string; apiKey: string; timeoutMs: number; reasoningField: string; models?: Record<string, ModelOverride> } {
  const baseURL = (raw?.baseURL ?? descriptor.defaultBaseURL).replace(/\/+$/, "");
  const apiKey = raw?.apiKey ?? descriptor.defaultApiKey;
  const timeoutMs = raw?.timeoutMs ?? 5_000;
  const enabled = raw?.enabled ?? true;
  const headers = raw?.headers;
  const reasoningField = raw?.reasoningField ?? descriptor.reasoningField;
  const models = raw?.models;
  return { baseURL, apiKey, enabled, timeoutMs, headers, reasoningField, models };
}

/**
 * Required model ID prefix for workbuddy-wild. The server rejects most models
 * without one of these prefixes with HTTP 400, so we filter them out at
 * discovery time to avoid showing dead entries in /models.
 */
export const WORKBUDDY_WILD_PREFIXES = ["workbuddy/", "traework/"] as const;

/**
 * Synthetic "auto" aliases workbuddy-wild accepts despite having no prefix.
 * These are `owned_by: "synthetic"` aliases the proxy resolves to a concrete
 * model server-side. `auto-free` picks the cheapest (incl. free) model and
 * `auto-nofree` the cheapest paid one; both are valid, callable model ids.
 *
 * A bare id that fails at chat time fails with a transient account error
 * (HTTP 503 "all accounts unavailable (cooling/disabled)"), NOT an invalid-model
 * error — so these are genuine models, not dead entries. We can't drop the
 * prefix filter entirely because truly invalid ids do exist; instead we accept
 * anything marked `owned_by: "synthetic"`, with this list as a fallback for
 * responses that omit `owned_by`.
 */
export const WORKBUDDY_WILD_BARE_ID_ALLOWLIST = ["auto-free", "auto-nofree"] as const;

/** Owner value workbuddy-wild uses for its synthetic auto-select aliases. */
export const WORKBUDDY_WILD_SYNTHETIC_OWNER = "synthetic";

/** Returns true iff `id` starts with one of the workbuddy-wild prefixes. */
export function hasWorkBuddyPrefix(id: string): boolean {
  return WORKBUDDY_WILD_PREFIXES.some((p) => id.startsWith(p));
}

/**
 * Returns true iff `id` is a model workbuddy-wild will actually accept:
 * it carries a required prefix, OR it is a known synthetic alias (either
 * flagged `owned_by: "synthetic"` or on the bare-id allowlist).
 */
export function isWorkBuddyModelId(id: string, ownedBy?: string): boolean {
  if (hasWorkBuddyPrefix(id)) return true;
  if (ownedBy === WORKBUDDY_WILD_SYNTHETIC_OWNER) return true;
  return (WORKBUDDY_WILD_BARE_ID_ALLOWLIST as readonly string[]).includes(id);
}
