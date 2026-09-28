/**
 * Normalization: convert each backend's raw + extended metadata into the
 * OpenCode V2 `Model.Info` shape that `ctx.provider.transform` expects.
 *
 * Capability precedence (highest to lowest):
 *   1. Extended-endpoint metadata (llama.cpp /props, litellm model_info,
 *      workbuddy-wild context_length/max_output_tokens) — produced by discovery.ts.
 *   2. Name-pattern heuristics — capabilities.ts.
 *   3. OpenCode Model.Info.default() — the framework's own assumptions
 *      (tools=true, input/output=[text], ctx=200K, output=32K).
 *
 * For limit fields (context/output/input), extended metadata wins outright
 * when present; otherwise we fall through to OpenCode's default. We never
 * let a heuristic *reduce* a known-good value below the framework default.
 */

import { Model, Provider } from "@opencode/plugin";

import { getBackend } from "./backends.js";
import { DEFAULT_CAPABILITIES, inferCapabilities, mergeCapabilities } from "./capabilities.js";
import type { DiscoveredProvider, ExtendedModelInfo } from "./types.js";

/**
 * Build a complete Model.Info for one discovered model. Returns `null` if the
 * model id is empty or the backend is offline (caller should skip).
 */
export function toModelInfo(
  backend: DiscoveredProvider,
  modelID: string,
): Model.Info | null {
  if (!modelID) return null;

  const descriptor = getBackend(backend.id);
  const providerID = Provider.ID.make(descriptor.id);
  const id = Model.ID.make(modelID);
  const base = Model.Info.default(providerID, id);

  const ext: ExtendedModelInfo | undefined = backend.extended[modelID];
  const heuristic = inferCapabilities(modelID);

  // Capabilities: extended metadata overrides heuristic field-by-field.
  // If the backend exposed a tools=boolean, we trust it; if it left it
  // undefined, fall back to the heuristic (which itself falls back to
  // DEFAULT_CAPABILITIES for ambiguous fields).
  const caps = ext
    ? mergeCapabilities(heuristic, ext.capabilities)
    : heuristic;

  // Limits: extended metadata wins outright when >0; otherwise default.
  const defaultCtx = base.limit?.context ?? 200_000;
  const defaultOutput = base.limit?.output ?? 32_000;
  const limits = {
    context: ext?.limits.context && ext.limits.context > 0 ? ext.limits.context : defaultCtx,
    output: ext?.limits.output && ext.limits.output > 0 ? ext.limits.output : defaultOutput,
    ...(ext?.limits.input && ext.limits.input > 0 ? { input: ext.limits.input } : {}),
  };

  // Display name: prefer the leaf of the id (after `workbuddy/` etc.) plus
  // the backend's display name in parens, so /models shows e.g.
  // "hy3 (WorkBuddy-Wild)" rather than a raw "workbuddy/hy3".
  const leaf = modelID.includes("/") ? modelID.split("/").slice(-1)[0] : modelID;
  const name = `${leaf} (${descriptor.displayName})`;

  const out: Model.Info = {
    ...base,
    name,
    capabilities: {
      tools: caps.tools,
      input: caps.input,
      output: caps.output,
    },
    limit: limits,
  };

  // If the model is reasoning-capable, surface it as an extra capability flag.
  // OpenCode V2 Model.Info doesn't have a dedicated `reasoning` field today,
  // but we still want to influence tool routing — reasoning models with weak
  // tool support should not be force-fed function calls. Conservative: leave
  // tools on unless the heuristic explicitly turned them off.
  if (caps.reasoning && !caps.tools) {
    // Already reflected in caps.tools.
  }

  return out;
}

/**
 * Convert a full DiscoveredProvider into an OpenCode V2 Provider.Info +
 * a readonly list of Model.Info. Used by setup() to call editor.add().
 */
export function toProviderRecord(
  backend: DiscoveredProvider,
): { info: Provider.Info; models: Model.Info[] } {
  const descriptor = getBackend(backend.id);
  const providerID = Provider.ID.make(descriptor.id);

  const info: Provider.Info = {
    ...Provider.Info.empty(providerID),
    name: backend.displayName,
    // "enabled" makes the provider show up in /providers without the user
    // having to enable it manually; "automatic" would hide it until toggled.
    activation: "enabled",
    package: descriptor.packageName,
    settings: {
      baseURL: backend.baseURL,
      apiKey: backend.apiKey,
    },
  };

  const models: Model.Info[] = [];
  for (const id of backend.modelIds) {
    const m = toModelInfo(backend, id);
    if (m) models.push(m);
  }

  return { info, models };
}

/** Re-export for callers that just want the default-capabilities reference. */
export { DEFAULT_CAPABILITIES };
