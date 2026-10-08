/**
 * Static descriptors for the three backends supported by this plugin.
 *
 * Each descriptor is purely declarative — it does not perform I/O. Discovery
 * lives in discovery.ts; this file just gives the plugin a stable identity
 * (display name, default port, OpenAI-compatible package) per backend.
 *
 * The `package` field must be `@ai-sdk/openai-compatible` for all three
 * backends, because llama.cpp, litellm, and workbuddy-wild all speak the
 * OpenAI Chat Completions protocol on `/v1/chat/completions`.
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
  /** The npm provider package OpenCode should use to route chat completions. */
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
 * Required model ID prefix for workbuddy-wild. The server rejects any model
 * without one of these prefixes with HTTP 400, so we filter them out at
 * discovery time to avoid showing dead entries in /models.
 */
export const WORKBUDDY_WILD_PREFIXES = ["workbuddy/", "traework/"] as const;

/** Returns true iff `id` starts with one of the workbuddy-wild prefixes. */
export function hasWorkBuddyPrefix(id: string): boolean {
  return WORKBUDDY_WILD_PREFIXES.some((p) => id.startsWith(p));
}
