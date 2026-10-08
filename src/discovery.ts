/**
 * Discovery module �?fetches /v1/models from each backend and, where the
 * backend exposes extra metadata (llama.cpp /props, litellm model_info,
 * workbuddy-wild inline context_length/max_output_tokens), reads it and
 * folds it into a NormalizedCapabilities + NormalizedLimits block per model.
 *
 * Every external call is wrapped in try/catch and a per-request timeout.
 * A failed backend produces an offline DiscoveredProvider with online=false
 * rather than throwing �?the plugin's setup() still registers the other two.
 */

import { getBackend, hasWorkBuddyPrefix, resolveConfig } from "./backends.js";
import type {
  BackendConfig,
  BackendID,
  DiscoveredProvider,
  ExtendedModelInfo,
  LiteLLMModelInfo,
  LlamaCppPropsResponse,
  ModelOverride,
  ModelsListResponse,
  NormalizedCapabilities,
  OpenAIModel,
} from "./types.js";

/**
 * Extract per-model reasoning field overrides from user config.
 * Returns a map of model ID → field name, filtering out entries
 * that don't set `reasoningField`.
 */
function buildModelOverrides(
  models?: Record<string, ModelOverride>,
): Record<string, string> | undefined {
  if (!models) return undefined;
  const out: Record<string, string> = {};
  for (const [id, ov] of Object.entries(models)) {
    if (ov?.reasoningField) out[id] = ov.reasoningField;
  }
  return Object.keys(out).length ? out : undefined;
}

// ---------------------------------------------------------------------------
// HTTP helper
// ---------------------------------------------------------------------------

export interface FetchJSONResult<T> {
  status: number;
  body: T;
}

/**
 * Fetch a URL as JSON with a timeout via AbortController.
 *
 * @param url     absolute URL
 * @param init    extra RequestInit (headers, method). Authorization is added
 *                automatically if `bearer` is non-empty.
 * @param bearer  bearer token to send as `Authorization: Bearer <token>`.
 *                Pass empty string to skip the header.
 * @param timeoutMs  per-request timeout. On timeout, throws `TimeoutError`.
 */
export async function fetchJSON<T>(
  url: string,
  init: RequestInit = {},
  bearer: string,
  timeoutMs: number,
): Promise<FetchJSONResult<T>> {
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...(init.headers as Record<string, string> | undefined),
  };
  if (bearer) headers.Authorization = `Bearer ${bearer}`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Math.max(500, timeoutMs));
  try {
    const res = await fetch(url, {
      ...init,
      headers,
      signal: controller.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new HttpError(res.status, url, text);
    }
    const body = (await res.json()) as T;
    return { status: res.status, body };
  } finally {
    clearTimeout(timer);
  }
}

export class HttpError extends Error {
  constructor(
    public readonly status: number,
    public readonly url: string,
    public readonly bodyText: string,
  ) {
    super(`HTTP ${status} from ${url}${bodyText ? `: ${bodyText.slice(0, 200)}` : ""}`);
    this.name = "HttpError";
  }
}

export class TimeoutError extends Error {
  constructor(public readonly url: string) {
    super(`Request timed out: ${url}`);
    this.name = "TimeoutError";
  }
}

// ---------------------------------------------------------------------------
// Public discovery API
// ---------------------------------------------------------------------------

/**
 * Discover all enabled backends. Returns one DiscoveredProvider per backend,
 * regardless of whether it succeeded �?failures become offline records.
 *
 * Backends are fetched sequentially in the order they're declared in BACKENDS
 * (llama-cpp �?litellm �?workbuddy-wild). We could parallelize with
 * Promise.all, but the per-backend calls are independent and small enough
 * that the added complexity isn't worth it; sequential also makes the
 * warning log easier to read when several backends are down.
 */
export async function discoverAll(
  options: {
    backends?: Partial<Record<BackendID, BackendConfig>>;
  },
): Promise<DiscoveredProvider[]> {
  const ids: BackendID[] = ["llama-cpp", "litellm", "workbuddy-wild"];
  const results: DiscoveredProvider[] = [];
  for (const id of ids) {
    const descriptor = getBackend(id);
    const raw = options.backends?.[id];
    const cfg = resolveConfig(descriptor, raw);
    if (!cfg.enabled) {
      // Skip silently �?user explicitly disabled it.
      continue;
    }
    try {
      const discovered = await discoverBackend(id, cfg);
      results.push(discovered);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.warn(`[local-models] ${id} discovery failed: ${msg}`);
      results.push({
        id,
        displayName: descriptor.displayName,
        online: false,
        error: msg,
        baseURL: cfg.baseURL,
        apiKey: cfg.apiKey,
        modelIds: [],
        extended: {},
        reasoningField: cfg.reasoningField,
        modelOverrides: buildModelOverrides(cfg.models),
      });
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Per-backend dispatch
// ---------------------------------------------------------------------------

async function discoverBackend(
  id: BackendID,
  cfg: BackendConfig & { baseURL: string; apiKey: string; timeoutMs: number; reasoningField: string; models?: Record<string, ModelOverride> },
): Promise<DiscoveredProvider> {
  switch (id) {
    case "llama-cpp":
      return discoverLlamaCpp(cfg);
    case "litellm":
      return discoverLiteLLM(cfg);
    case "workbuddy-wild":
      return discoverWorkBuddyWild(cfg);
  }
}

// ---------------------------------------------------------------------------
// llama.cpp
// ---------------------------------------------------------------------------

async function discoverLlamaCpp(
  cfg: BackendConfig & { baseURL: string; apiKey: string; timeoutMs: number; reasoningField: string; models?: Record<string, ModelOverride> },
): Promise<DiscoveredProvider> {
  const descriptor = getBackend("llama-cpp");
  // /v1/models returns OpenAI-shaped list. We strip /v1 suffix because some
  // llama.cpp builds 404 on /v1/v1/models; we want /v1/models exactly.
  const modelsURL = joinURL(cfg.baseURL, "/models");
  const propsURL = joinURL(stripVSuffix(cfg.baseURL), "/props");

  const list = await fetchJSON<ModelsListResponse>(
    modelsURL,
    { headers: cfg.headers },
    cfg.apiKey,
    cfg.timeoutMs,
  );

  const extended: Record<string, ExtendedModelInfo> = {};

  // Best-effort: fetch /props once per model. Each call gets a tighter
  // 1.5s timeout (autoload=0 means we don't actually load the model �?fast).
  // Failures here are non-fatal: we just fall back to name heuristics.
  const probeTimeout = Math.min(1500, cfg.timeoutMs);

  for (const m of list.body.data ?? []) {
    if (!m?.id) continue;
    try {
      const props = await fetchJSON<LlamaCppPropsResponse>(
        `${propsURL}?model=${encodeURIComponent(m.id)}&autoload=0`,
        { headers: cfg.headers },
        cfg.apiKey,
        probeTimeout,
      );
      extended[m.id] = extendLlamaCpp(m, props.body);
    } catch (err) {
      // Probe failure is fine �?fall back to inline meta + name heuristics.
      extended[m.id] = extendLlamaCpp(m, undefined);
      if (!(err instanceof HttpError && err.status === 404)) {
        // Don't log 404s �?expected on older llama.cpp builds without /props.
        console.warn(`[local-models] llama-cpp /props probe failed for ${m.id}: ${(err as Error).message}`);
      }
    }
  }

  return {
    id: "llama-cpp",
    displayName: descriptor.displayName,
    online: true,
    baseURL: cfg.baseURL,
    apiKey: cfg.apiKey,
    modelIds: list.body.data?.map((m) => m.id).filter(Boolean) ?? [],
    extended,
    reasoningField: cfg.reasoningField,
    modelOverrides: buildModelOverrides(cfg.models),
  };
}

/**
 * Build ExtendedModelInfo for a llama.cpp model. We read, in priority order:
 *   1. /props response (per-model): default_generation_settings.n_ctx,
 *      modalities.vision, chat_template (for thinking detection).
 *   2. Inline `meta.n_ctx` from /v1/models (some builds expose it directly).
 *   3. Multi-preset router shape: models[].status.args parsed for -c/--ctx-size.
 *   4. Conservative defaults: ctx=8192, output=min(ctx/2,16384).
 */
function extendLlamaCpp(model: OpenAIModel, props: LlamaCppPropsResponse | undefined): ExtendedModelInfo {
  // ---- Context window ----
  let context = 0;
  // From inline meta (llama.cpp server's /v1/models returns `meta` on each model)
  // llama.cpp's /v1/models may put context info under `meta.n_ctx` (newer
  // server builds) or directly at top level as `n_ctx` (older builds).
  // OpenAIModel has an index signature, so we read via string indexing.
  const inlineMeta = model.meta as Record<string, unknown> | undefined;
  if (inlineMeta && typeof inlineMeta.n_ctx === "number") {
    context = inlineMeta.n_ctx;
  } else if (typeof model.n_ctx === "number") {
    context = model.n_ctx;
  }
  // From /props
  const dgs = props?.default_generation_settings;
  if (context === 0 && typeof dgs?.n_ctx === "number") {
    context = dgs.n_ctx;
  }
  // Multi-preset router: scan args for -c / --ctx-size
  if (context === 0) {
    const routerModel = props?.models?.find((m) => m?.id === model.id || m?.name === model.id);
    const args = routerModel?.status?.args;
    if (args) {
      const ctx = scanNumericFlag(args, ["--ctx-size", "-c"]) ?? routerModel.status?.n_ctx;
      if (ctx) context = ctx;
    }
  }
  if (context <= 0) context = 8192; // safe fallback

  // ---- Output limit ----
  let output = 0;
  const predictArg = dgs ? scanNumericFlagFromSettings(dgs) : undefined;
  if (predictArg) output = predictArg;
  const routerModel = props?.models?.find((m) => m?.id === model.id || m?.name === model.id);
  if (output === 0 && routerModel?.status?.args) {
    const p = scanNumericFlag(routerModel.status.args, ["--predict", "-n", "--n-predict"]);
    if (p) output = p;
  }
  if (output <= 0) output = Math.min(Math.floor(context / 2), 16384);

  // ---- Modalities / capabilities ----
  const caps: NormalizedCapabilities = {
    tools: true,
    input: ["text"],
    output: ["text"],
    reasoning: false,
  };
  if (dgs?.modalities?.vision) {
    if (!caps.input.includes("image")) caps.input.push("image");
  }
  // Multi-preset router exposes architecture.input_modalities
  const modalities = routerModel?.architecture?.input_modalities;
  if (modalities && modalities.includes("image") && !caps.input.includes("image")) {
    caps.input.push("image");
  }
  // Thinking detection from chat_template �?same markers llama.cpp itself keys on.
  const template = dgs?.chat_template;
  if (template) {
    const THINKING_MARKERS = [
      "enable_thinking",
      "reasoning_content",
      "<think>",
      "[THINK]",
      "<|channel|>analysis",
      "<|channel|>thought",
    ];
    if (THINKING_MARKERS.some((m) => template.includes(m))) {
      caps.reasoning = true;
    }
  }

  return {
    capabilities: caps,
    limits: { context, output },
  };
}

// ---------------------------------------------------------------------------
// litellm
// ---------------------------------------------------------------------------

async function discoverLiteLLM(
  cfg: BackendConfig & { baseURL: string; apiKey: string; timeoutMs: number; includeNonChat?: boolean; reasoningField: string; models?: Record<string, ModelOverride> },
): Promise<DiscoveredProvider> {
  const descriptor = getBackend("litellm");
  const url = joinURL(cfg.baseURL, "/models");

  const list = await fetchJSON<ModelsListResponse & { data?: Array<OpenAIModel & { model_info?: LiteLLMModelInfo }> }>(
    url,
    { headers: cfg.headers },
    cfg.apiKey,
    cfg.timeoutMs,
  );

  const extended: Record<string, ExtendedModelInfo> = {};
  const acceptedIds: string[] = [];
  for (const m of list.body.data ?? []) {
    if (!m?.id) continue;
    // model_info may be nested under `model_info` or flattened onto the model.
    const info: LiteLLMModelInfo | undefined =
      (m.model_info as LiteLLMModelInfo | undefined) ??
      pickLiteLLMFields(m);
    // Skip non-chat models unless the user opted in.
    if (info?.mode && info.mode !== "chat" && !cfg.includeNonChat) {
      continue;
    }
    extended[m.id] = extendLiteLLM(info);
    acceptedIds.push(m.id);
  }

  return {
    id: "litellm",
    displayName: descriptor.displayName,
    online: true,
    baseURL: cfg.baseURL,
    apiKey: cfg.apiKey,
    modelIds: acceptedIds,
    extended,
    reasoningField: cfg.reasoningField,
    modelOverrides: buildModelOverrides(cfg.models),
  };
}

/** Pull known litellm model_info fields off the flat model object (some proxy versions flatten). */
function pickLiteLLMFields(m: OpenAIModel): LiteLLMModelInfo | undefined {
  const keys: (keyof LiteLLMModelInfo)[] = [
    "max_input_tokens",
    "max_output_tokens",
    "max_tokens",
    "mode",
    "supports_function_calling",
    "supports_parallel_function_calling",
    "supports_vision",
    "supports_audio_input",
    "supports_audio_output",
    "supports_reasoning",
    "supports_response_schema",
    "supports_system_messages",
    "supports_prompt_caching",
    "supports_web_search",
    "modalities",
  ];
  const out: LiteLLMModelInfo = {};
  let found = false;
  for (const k of keys) {
    const v = (m as Record<string, unknown>)[k as string];
    if (v !== undefined) {
      (out as Record<string, unknown>)[k as string] = v;
      found = true;
    }
  }
  return found ? out : undefined;
}

function extendLiteLLM(info: LiteLLMModelInfo | undefined): ExtendedModelInfo {
  // ---- Limits ----
  const context = info?.max_input_tokens ?? info?.max_tokens ?? 200_000;
  const output = info?.max_output_tokens ?? info?.max_tokens ?? 32_000;

  // ---- Capabilities ----
  const caps: NormalizedCapabilities = {
    tools: info?.supports_function_calling ?? true,
    input: ["text"],
    output: ["text"],
    reasoning: info?.supports_reasoning ?? false,
  };
  // Modalities (newer litellm exposes this)
  if (info?.modalities?.input) {
    caps.input = normalizeModalities(info.modalities.input);
  } else if (info?.supports_vision) {
    if (!caps.input.includes("image")) caps.input.push("image");
  }
  if (info?.supports_audio_input && !caps.input.includes("audio")) {
    caps.input.push("audio");
  }
  if (info?.modalities?.output) {
    caps.output = normalizeModalities(info.modalities.output);
  }
  if (info?.supports_audio_output && !caps.output.includes("audio")) {
    caps.output.push("audio");
  }

  return {
    capabilities: caps,
    limits: {
      context,
      output: Math.min(output, context),
      input: context,
    },
  };
}

function normalizeModalities(arr: string[] | undefined): string[] {
  if (!arr) return ["text"];
  const out: string[] = [];
  for (const m of arr) {
    const lower = m.toLowerCase();
    // litellm uses "speech" for audio; OpenCode uses "audio".
    const mapped = lower === "speech" ? "audio" : lower;
    if (!out.includes(mapped)) out.push(mapped);
  }
  if (out.length === 0) out.push("text");
  return out;
}

// ---------------------------------------------------------------------------
// workbuddy-wild
// ---------------------------------------------------------------------------

async function discoverWorkBuddyWild(
  cfg: BackendConfig & { baseURL: string; apiKey: string; timeoutMs: number; reasoningField: string; models?: Record<string, ModelOverride> },
): Promise<DiscoveredProvider> {
  const descriptor = getBackend("workbuddy-wild");
  const url = joinURL(cfg.baseURL, "/models");

  const list = await fetchJSON<ModelsListResponse>(url, { headers: cfg.headers }, cfg.apiKey, cfg.timeoutMs);

  const extended: Record<string, ExtendedModelInfo> = {};
  const acceptedIds: string[] = [];
  for (const m of list.body.data ?? []) {
    if (!m?.id) continue;
    if (!hasWorkBuddyPrefix(m.id)) {
      // Defensive: server already enforces this, but skip if it ever slips through.
      console.warn(`[local-models] workbuddy-wild model id without prefix, skipping: ${m.id}`);
      continue;
    }
    extended[m.id] = extendWorkBuddyWild(m);
    acceptedIds.push(m.id);
  }

  return {
    id: "workbuddy-wild",
    displayName: descriptor.displayName,
    online: true,
    baseURL: cfg.baseURL,
    apiKey: cfg.apiKey,
    modelIds: acceptedIds,
    extended,
    reasoningField: cfg.reasoningField,
    modelOverrides: buildModelOverrides(cfg.models),
  };
}

function extendWorkBuddyWild(model: OpenAIModel): ExtendedModelInfo {
  const context = typeof model.context_length === "number" && model.context_length > 0
    ? model.context_length
    : 200_000;
  const output = typeof model.max_output_tokens === "number" && model.max_output_tokens > 0
    ? model.max_output_tokens
    : Math.min(32_000, context);

  // workbuddy-wild doesn't expose supports_* flags, so capabilities fall
  // through to name heuristics in normalize.ts.
  return {
    capabilities: {
      tools: true,
      input: ["text"],
      output: ["text"],
      reasoning: false,
    },
    limits: { context, output, input: context },
  };
}

// ---------------------------------------------------------------------------
// URL helpers
// ---------------------------------------------------------------------------

/**
 * Append `path` to `base`. If `base` already ends with `path` (e.g. /v1 + /models
 * = /v1/models), don't double it. Ensures exactly one leading slash on `path`.
 */
function joinURL(base: string, path: string): string {
  const cleanBase = base.replace(/\/+$/, "");
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  if (cleanBase.endsWith(cleanPath)) return cleanBase;
  // Handle the case where base is "http://x/v1" and path is "/models" �?"http://x/v1/models".
  // Also handle "http://x" + "/v1/models".
  return `${cleanBase}${cleanPath}`;
}

/** Strip a trailing /v1 (case-insensitive) from a base URL, used to derive the /props origin. */
function stripVSuffix(base: string): string {
  return base.replace(/\/v1\/?$/i, "");
}

// ---------------------------------------------------------------------------
// CLI-arg scanning (for llama.cpp multi-preset router shape)
// ---------------------------------------------------------------------------

/**
 * Scan an argv array for a flag that takes a numeric value, either as
 * `--flag=VALUE` or `--flag VALUE` or `-x VALUE`. Returns the parsed number
 * or undefined.
 */
function scanNumericFlag(args: string[], flags: string[]): number | undefined {
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (!a) continue;
    // --flag=VALUE form
    const eq = a.indexOf("=");
    if (eq > 0) {
      const name = a.slice(0, eq);
      if (flags.includes(name)) {
        const n = Number(a.slice(eq + 1));
        if (Number.isFinite(n) && n > 0) return n;
      }
      continue;
    }
    // --flag VALUE form
    if (flags.includes(a)) {
      const next = args[i + 1];
      if (next) {
        const n = Number(next);
        if (Number.isFinite(n) && n > 0) return n;
      }
    }
  }
  return undefined;
}

/**
 * Read n_predict from default_generation_settings. The field can be a number
 * or null (older llama.cpp builds). We do NOT use n_batch here �?that's the
 * batch size, not the output cap.
 */
function scanNumericFlagFromSettings(dgs: NonNullable<LlamaCppPropsResponse["default_generation_settings"]>): number | undefined {
  const v = dgs.n_predict;
  if (typeof v === "number" && v > 0) return v;
  return undefined;
}

