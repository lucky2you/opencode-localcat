/**
 * Normalization: convert each backend's raw + extended metadata into the
 * opencode `provider.<id>` config block shape used by the 1.x plugin API.
 *
 * Capability precedence (highest to lowest):
 *   1. Extended-endpoint metadata (llama.cpp /props, litellm model_info,
 *      workbuddy-wild context_length/max_output_tokens) — produced by discovery.ts.
 *   2. Name-pattern heuristics — capabilities.ts.
 *   3. opencode's own defaults for unknown models (ctx=200K, output=32K, tools on).
 *
 * For limit fields (context/output/input), extended metadata wins outright
 * when present; otherwise we fall through to opencode's default.
 */

import { getBackend } from "./backends.js";
import { DEFAULT_CAPABILITIES, inferCapabilities, mergeCapabilities } from "./capabilities.js";
import type { DiscoveredProvider } from "./types.js";

/** opencode's fallback assumptions for an unknown model. */
const DEFAULT_CONTEXT = 200_000;
const DEFAULT_OUTPUT = 32_000;

/** Per-model entry inside `provider.<id>.models`. */
export interface ConfigModel {
  name?: string;
  /** Model accepts non-text input (images/pdf/audio) as attachments. */
  attachment?: boolean;
  /** Model emits reasoning traces. */
  reasoning?: boolean;
  /**
   * Where interleaved reasoning tokens live in each streamed delta. The
   * workbuddy-wild backend puts them in `reasoning_content` (not the
   * OpenAI-default `reasoning`), so we must declare it explicitly.
   * Without this, opencode falls back to a default that renders each
   * reasoning delta as its own block instead of one aggregated block.
   *
   * Must use the object form `{ field: "reasoning_content" }`; the bare
   * string form is accepted by the JSON schema but does not trigger
   * aggregation in the opencode runtime.
   */
  interleaved?: { field: string };
  /** Model supports tool/function calling. */
  tool_call?: boolean;
  limit: {
    context: number;
    output: number;
    input?: number;
  };
}

/** `provider.<id>` config block. */
export interface ConfigProvider {
  name: string;
  /** npm package opencode uses to talk to this provider. */
  npm: string;
  options: {
    baseURL: string;
    apiKey?: string;
  };
  models: Record<string, ConfigModel>;
}

/** Build the config entry for a single discovered model (null if id is empty). */
export function toModelConfig(
  backend: DiscoveredProvider,
  modelID: string,
): ConfigModel | null {
  if (!modelID) return null;

  const descriptor = getBackend(backend.id);
  const ext = backend.extended[modelID];
  const heuristic = inferCapabilities(modelID);

  // Capabilities: extended metadata overrides heuristic field-by-field.
  const caps = ext ? mergeCapabilities(heuristic, ext.capabilities) : heuristic;

  // Limits: extended metadata wins outright when > 0; otherwise default.
  const context = positive(ext?.limits.context) ?? DEFAULT_CONTEXT;
  const output = Math.min(positive(ext?.limits.output) ?? DEFAULT_OUTPUT, context);
  const limit: ConfigModel["limit"] = { context, output };
  const input = positive(ext?.limits.input);
  if (input !== undefined) limit.input = Math.min(input, context);

  // Display name: for workbuddy-wild we keep the full id (with its
  // required `workbuddy/` or `traework/` prefix) so it matches the server's
  // id, e.g. "workbuddy/hy3". For other backends we use the leaf of the id
  // plus the backend's display name in parens, e.g. "hy3 (LiteLLM)".
  const name =
    descriptor.id === "workbuddy-wild"
      ? modelID
      : (() => {
          const leaf = modelID.includes("/") ? modelID.split("/").slice(-1)[0] ?? modelID : modelID;
          return `${leaf} (${descriptor.displayName})`;
        })();

  // Per-model override: different models on the same backend may stream
  // reasoning under different field names. Check `modelOverrides[modelID]`
  // first, then fall back to the backend-level `reasoningField`.
  const reasoningField = backend.modelOverrides?.[modelID] ?? backend.reasoningField;

  return {
    name,
    attachment: caps.input.some((m) => m !== "text"),
    reasoning: caps.reasoning,
    // Tell opencode where each backend actually streams its reasoning tokens.
    // Without this, opencode can't aggregate the trace and renders every
    // streamed delta as its own "Thought" block — the per-token spam reported.
    //
    // We set `interleaved` for EVERY model of a backend that declares a
    // reasoningField, not only those flagged `reasoning`. Rationale: the
    // capability heuristics under-detect reasoning (e.g. most workbuddy-wild
    // and many litellm models stream `reasoning_content` but aren't on the
    // name-pattern whitelist), so gating on `caps.reasoning` leaves those
    // models un-aggregated. A false positive here is harmless — if the model
    // doesn't emit reasoning, the field is simply absent and opencode ignores
    // it. A false negative, by contrast, is the exact bug we're fixing.
    //
    // The value MUST be the object form { field: "..." }, not a bare string.
    // opencode's runtime only aggregates reasoning into a single block when
    // `interleaved` is an object with a `field` key; a bare string is accepted
    // by the JSON schema but does not trigger aggregation, causing each
    // streamed reasoning delta to render as its own "Thought" block.
    interleaved: reasoningField ? { field: reasoningField } : undefined,
    tool_call: caps.tools,
    limit,
  };
}

/**
 * Convert a full DiscoveredProvider into a `provider.<id>` config block.
 * Callers should only invoke this for online backends with at least one model.
 */
export function toProviderConfig(backend: DiscoveredProvider): ConfigProvider {
  const descriptor = getBackend(backend.id);

  const models: Record<string, ConfigModel> = {};
  for (const id of backend.modelIds) {
    const m = toModelConfig(backend, id);
    if (m) models[id] = m;
  }

  const options: ConfigProvider["options"] = { baseURL: backend.baseURL };
  if (backend.apiKey) options.apiKey = backend.apiKey;

  return {
    name: descriptor.displayName,
    npm: descriptor.packageName,
    options,
    models,
  };
}

/** Return n only when it is a positive finite number. */
function positive(n: number | undefined): number | undefined {
  return typeof n === "number" && Number.isFinite(n) && n > 0 ? n : undefined;
}

/** Re-export for callers that just want the default-capabilities reference. */
export { DEFAULT_CAPABILITIES };
