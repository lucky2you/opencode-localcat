/**
 * Normalization: convert each backend's discovered metadata into the two
 * provider shapes this plugin can emit.
 *
 *   - V2: a `Provider.Info` + `Model.Info[]` registration contributed through
 *     `ctx.provider.transform` (see `toProviderRegistration` / `toModelInfo`).
 *   - V1: a `provider.<id>` config block injected through the `config` hook
 *     (see `toProviderConfig` / `toModelConfig`).
 *
 * Both paths share the same capability/limit/name helpers so the two APIs
 * always describe a model identically.
 *
 * Capability precedence (highest to lowest):
 *   1. Extended-endpoint metadata (llama.cpp /props, litellm model_info,
 *      workbuddy-wild context_length/max_output_tokens) — produced by discovery.ts.
 *   2. Name-pattern heuristics — capabilities.ts.
 *   3. opencode's own defaults for unknown models (ctx=200K, output=32K, tools on).
 *
 * For limit fields (context/output/input), extended metadata wins outright
 * when present; otherwise we fall through to opencode's default.
 *
 * Reasoning aggregation:
 *   - V2: `compatibility.reasoningField` tells opencode which streamed delta
 *     field carries reasoning tokens and to replay that field on prior
 *     assistant messages (replaces V1's `interleaved`).
 *   - V1: `interleaved: { field }` plus a custom `fetch`
 *     (`options.fetch = withReasoningStreamFix(...)`) that strips empty
 *     placeholder delta fields the bundled `@ai-sdk/openai-compatible` parser
 *     would otherwise mistake for tool calls.
 */

import { Model, Provider } from "@opencode/plugin";
import { getBackend, type BackendDescriptor } from "./backends.js";
import { inferCapabilities, mergeCapabilities } from "./capabilities.js";
import { defaultFetch, withReasoningStreamFix } from "./stream-fetch.js";
import type { DiscoveredProvider, NormalizedCapabilities } from "./types.js";

/** opencode's fallback assumptions for an unknown model. */
const DEFAULT_CONTEXT = 200_000;
const DEFAULT_OUTPUT = 32_000;

/**
 * Runtime package opencode V2 uses to talk to all three backends. They all
 * speak the OpenAI Chat Completions protocol on `/v1/chat/completions`, so the
 * built-in openai-compatible provider package is the correct route.
 * (V1 uses the `@ai-sdk/openai-compatible` npm package instead — see
 * `BackendDescriptor.packageName`.)
 */
const V2_PACKAGE = "@opencode/ai/providers/openai-compatible";

// ---------------------------------------------------------------------------
// Shared helpers (used by both the V1 and V2 output shapes)
// ---------------------------------------------------------------------------

/** Cleaned-up view of a single model: capabilities, limits, name, reasoning. */
interface ResolvedModel {
  descriptor: BackendDescriptor;
  caps: NormalizedCapabilities;
  context: number;
  output: number;
  input?: number;
  name: string;
  reasoningField?: string;
}

/** Resolve everything both output shapes need from a discovered model. */
function resolveModel(backend: DiscoveredProvider, modelID: string): ResolvedModel | null {
  if (!modelID) return null;

  const descriptor = getBackend(backend.id);
  const ext = backend.extended[modelID];
  const heuristic = inferCapabilities(modelID);

  // Capabilities: extended metadata overrides heuristic field-by-field.
  const caps = ext ? mergeCapabilities(heuristic, ext.capabilities) : heuristic;

  // Limits: extended metadata wins outright when > 0; otherwise default.
  const context = Math.floor(positive(ext?.limits.context) ?? DEFAULT_CONTEXT);
  const output = Math.floor(Math.min(positive(ext?.limits.output) ?? DEFAULT_OUTPUT, context));
  const rawInput = positive(ext?.limits.input);
  const input = rawInput !== undefined ? Math.floor(Math.min(rawInput, context)) : undefined;

  // Display name: for workbuddy-wild we keep the full id (with its required
  // `workbuddy/` or `traework/` prefix) so it matches the server's id, e.g.
  // "workbuddy/hy3". For other backends we use the leaf of the id plus the
  // backend's display name in parens, e.g. "hy3 (LiteLLM)".
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

  return { descriptor, caps, context, output, input, name, reasoningField };
}

/** Return n only when it is a positive finite number. */
function positive(n: number | undefined): number | undefined {
  return typeof n === "number" && Number.isFinite(n) && n > 0 ? n : undefined;
}

// ---------------------------------------------------------------------------
// opencode V2 shape: Provider.Info + Model.Info[]
// ---------------------------------------------------------------------------

/** A provider plus its models, ready to hand to `editor.add(...)`. */
export interface ProviderRegistration {
  info: Provider.Info;
  models: Model.Info[];
}

/** Build the V2 Model.Info for a single discovered model (null if id is empty). */
export function toModelInfo(
  backend: DiscoveredProvider,
  providerID: Provider.ID,
  modelID: string,
): Model.Info | null {
  const resolved = resolveModel(backend, modelID);
  if (!resolved) return null;
  const { caps, context, output, input, name, reasoningField } = resolved;

  return {
    ...Model.Info.default(providerID, Model.ID.make(modelID)),
    name,
    capabilities: {
      tools: caps.tools,
      input: caps.input,
      output: caps.output,
    },
    limit: {
      context,
      output,
      ...(input !== undefined ? { input } : {}),
    },
    // Tell opencode where each backend actually streams its reasoning tokens.
    // We set `reasoningField` for EVERY model of a backend that declares one,
    // not only those flagged `reasoning`. Rationale: the capability heuristics
    // under-detect reasoning (e.g. most workbuddy-wild and many litellm models
    // stream `reasoning_content` but aren't on the name-pattern whitelist), so
    // gating on `caps.reasoning` leaves those models un-aggregated. A false
    // positive here is harmless — if the model doesn't emit reasoning, the
    // field is simply absent and opencode ignores it. A false negative, by
    // contrast, is the exact bug we're fixing.
    ...(reasoningField ? { compatibility: { reasoningField } } : {}),
  };
}

/**
 * Convert a full DiscoveredProvider into a V2 provider registration.
 * Callers should only invoke this for online backends with at least one model.
 */
export function toProviderRegistration(backend: DiscoveredProvider): ProviderRegistration {
  const descriptor = getBackend(backend.id);
  const providerID = Provider.ID.make(backend.id);

  const models: Model.Info[] = [];
  for (const id of backend.modelIds) {
    const m = toModelInfo(backend, providerID, id);
    if (m) models.push(m);
  }

  const info: Provider.Info = {
    ...Provider.Info.empty(providerID),
    name: descriptor.displayName,
    // "enabled" keeps the provider selectable exactly like the V1 config hook
    // did, regardless of whether opencode would otherwise auto-activate it.
    activation: "enabled",
    package: V2_PACKAGE,
    settings: { baseURL: backend.baseURL, apiKey: backend.apiKey },
  };

  return { info, models };
}

// ---------------------------------------------------------------------------
// opencode V1 shape: `provider.<id>` config block
// ---------------------------------------------------------------------------

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
    /**
     * Custom fetch installed so streaming responses have empty placeholder
     * delta fields (`tool_calls: []`, `content: ""`, ...) stripped before the
     * OpenAI-compatible parser sees them. opencode forwards this straight to
     * `createOpenAICompatible`, which uses it as the transport.
     */
    fetch?: typeof fetch;
    [key: string]: unknown;
  };
  models: Record<string, ConfigModel>;
}

/** Build the V1 config entry for a single discovered model (null if id is empty). */
export function toModelConfig(
  backend: DiscoveredProvider,
  modelID: string,
): ConfigModel | null {
  const resolved = resolveModel(backend, modelID);
  if (!resolved) return null;
  const { caps, context, output, input, name, reasoningField } = resolved;

  const limit: ConfigModel["limit"] = { context, output };
  if (input !== undefined) limit.input = input;

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
  // opencode forwards provider `options.fetch` to createOpenAICompatible. Use
  // it to strip empty placeholder delta fields that otherwise make the parser
  // split reasoning into one block per token.
  options.fetch = withReasoningStreamFix(defaultFetch());

  return {
    name: descriptor.displayName,
    npm: descriptor.packageName,
    options,
    models,
  };
}
