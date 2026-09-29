/**
 * Type definitions for the opencode-localcat plugin.
 *
 * These describe the shape of plugin options in opencode.jsonc and the
 * responses we parse from each backend's OpenAI-compatible /v1/models
 * endpoint plus the backend-specific extended metadata endpoints.
 *
 * Everything here is intentionally permissive (string indices, optional
 * fields) because we are crossing three independently-versioned servers
 * (llama.cpp, litellm, workbuddy-wild) and a missing field must never
 * break discovery — it just falls through to the next capability source.
 */

// ---------------------------------------------------------------------------
// Plugin options (opencode.jsonc → plugin tuple `[[spec, options]]`)
// ---------------------------------------------------------------------------

export type BackendID = "llama-cpp" | "litellm" | "workbuddy-wild";

export interface BackendConfig {
  /** Fully-qualified OpenAI-compatible base URL, including /v1 suffix. */
  baseURL: string;
  /** Bearer token. "no-key-required" sentinel for unauthenticated llama.cpp servers. */
  apiKey?: string;
  /** Set false to skip a backend without removing its config block. Default: true. */
  enabled?: boolean;
  /** Per-request HTTP timeout for /v1/models and any extended endpoints. Default: 5000. */
  timeoutMs?: number;
  /** Extra HTTP headers merged onto every discovery request. */
  headers?: Record<string, string>;
  /** (litellm only) Include models whose mode is not "chat". Default: false. */
  includeNonChat?: boolean;
  /**
   * Field in each streamed chat delta that carries reasoning/thinking tokens.
   * Defaults to "reasoning_content". opencode aggregates reasoning into a
   * single block only when this matches the backend's actual streaming field;
   * a mismatch (or leaving it unset for a reasoning model) makes every delta
   * render as its own "Thought" block. Set it if your backend uses a
   * different field (e.g. "reasoning", "reasoning_text", "thinking").
   */
  reasoningField?: string;
}

export interface PluginOptions {
  /** Per-backend configuration. Keys not present are treated as disabled. */
  backends?: Partial<Record<BackendID, BackendConfig>>;
}

// ---------------------------------------------------------------------------
// OpenAI-compatible /v1/models response (common to all three backends)
// ---------------------------------------------------------------------------

export interface OpenAIModel {
  id: string;
  object?: string;
  created?: number;
  owned_by?: string;
  // workbuddy-wild embeds these directly on the model object:
  context_length?: number;
  max_output_tokens?: number;
  // litellm flattens model_info onto the model object on some versions:
  // supports_*, max_input_tokens, max_output_tokens, mode, modalities, ...
  // We index by string to stay tolerant of all the variants.
  [key: string]: unknown;
}

export interface ModelsListResponse {
  object: "list";
  data: OpenAIModel[];
}

// ---------------------------------------------------------------------------
// llama.cpp /props extended endpoint
// ---------------------------------------------------------------------------

export interface LlamaCppDefaultGenerationSettings {
  n_ctx?: number;
  n_batch?: number;
  n_predict?: number | null;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  // Newer llama.cpp servers expose modalities.vision when a mmproj is loaded.
  modalities?: {
    vision?: boolean;
    audio?: boolean;
    text?: boolean;
  };
  // Some builds surface the chat template so clients can detect thinking models.
  chat_template?: string;
}

export interface LlamaCppPropsResponse {
  default_generation_settings?: LlamaCppDefaultGenerationSettings;
  total_slots?: number;
  model_alias?: string;
  // Multi-preset router shape adds a `models` array; each preset has its own status.args.
  models?: Array<{
    id?: string;
    name?: string;
    status?: {
      preset?: string;
      args?: string[]; // we scan for -c/--ctx-size and -n/--predict
      n_ctx?: number;
    };
    architecture?: {
      input_modalities?: string[]; // e.g. ["text","image"]
    };
  }>;
}

// ---------------------------------------------------------------------------
// litellm model_info (per-model metadata block)
// ---------------------------------------------------------------------------

export interface LiteLLMModelInfo {
  max_input_tokens?: number;
  max_output_tokens?: number;
  max_tokens?: number; // legacy alias
  mode?: "chat" | "embedding" | "completion" | "audio_transcription" | "audio_speech" | "image_generation" | "moderation" | "rerank" | "search";
  supports_function_calling?: boolean;
  supports_parallel_function_calling?: boolean;
  supports_vision?: boolean;
  supports_audio_input?: boolean;
  supports_audio_output?: boolean;
  supports_reasoning?: boolean;
  supports_response_schema?: boolean;
  supports_system_messages?: boolean;
  supports_prompt_caching?: boolean;
  supports_web_search?: boolean;
  modalities?: {
    input?: string[]; // e.g. ["text","image","audio"]
    output?: string[]; // e.g. ["text","audio"]
  };
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Normalized extended metadata (produced by discovery, consumed by normalize)
// ---------------------------------------------------------------------------

export interface NormalizedCapabilities {
  tools: boolean;
  /** Modalities the model can accept as input: subset of "text","image","audio". */
  input: string[];
  /** Modalities the model can produce as output: subset of "text","image","audio". */
  output: string[];
  /** Whether the model emits reasoning traces (e.g. R1/o1/Qwen3 thinking). */
  reasoning: boolean;
}

export interface NormalizedLimits {
  /** Maximum total context (input + output) in tokens. */
  context: number;
  /** Maximum output tokens per response. */
  output: number;
  /** Maximum input tokens. Optional; undefined means "no separate cap". */
  input?: number;
}

export interface ExtendedModelInfo {
  capabilities: NormalizedCapabilities;
  limits: NormalizedLimits;
}

// ---------------------------------------------------------------------------
// Output of a per-backend discover() call
// ---------------------------------------------------------------------------

export interface DiscoveredProvider {
  /** Backend identifier (llama-cpp / litellm / workbuddy-wild). */
  id: BackendID;
  /** Display name shown in /providers. */
  displayName: string;
  /** Whether the backend responded and produced at least one model. */
  online: boolean;
  /** Error message if online=false; undefined otherwise. */
  error?: string;
  /** OpenAI-compatible base URL the user configured. */
  baseURL: string;
  /** API key (or "no-key-required" sentinel) the user configured. */
  apiKey?: string;
  /** Discovered model IDs (post-filter, post-normalization, ready for Model.Info). */
  modelIds: string[];
  /** Per-model extended metadata, keyed by model ID. */
  extended: Record<string, ExtendedModelInfo>;
  /** Streamed-delta field carrying reasoning tokens (for opencode aggregation). */
  reasoningField: string;
}

