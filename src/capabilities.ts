/**
 * Name-pattern heuristics for inferring model capabilities when the backend
 * does not expose explicit `supports_*` flags.
 *
 * This is the *fallback* layer — discovery.ts always prefers real metadata
 * from `/props` (llama.cpp), `model_info` (litellm), or `context_length` /
 * `max_output_tokens` (workbuddy-wild). Only when those are absent does this
 * module get called.
 *
 * Patterns are deliberately conservative: false positives are worse than false
 * negatives here, because OpenCode uses `capabilities.tools` to decide whether
 * to send function-call payloads (sending them to a model that can't handle
 * them produces garbage). When in doubt, we leave a flag at its default
 * (tools=true, input=[text], output=[text]) which matches OpenCode's own
 * fallback assumptions for unknown models.
 */

import type { NormalizedCapabilities } from "./types.js";

/** Default capabilities — matches OpenCode V2's Model.Info.default() assumptions. */
export const DEFAULT_CAPABILITIES: NormalizedCapabilities = {
  tools: true,
  input: ["text"],
  output: ["text"],
  reasoning: false,
};

// ---------------------------------------------------------------------------
// Compiled patterns (lowercased). Each entry is a RegExp; we test against
// `modelId.toLowerCase()`. Avoid catastrophic backtracking — these are all
// simple alternations / anchored suffixes.
// ---------------------------------------------------------------------------

const VISION_PATTERNS: RegExp[] = [
  /(^|[-_])vl(?![-_a-z0-9])/i, // ends with -vl, -vl<digit>, etc.
  /-vision\b/i,
  /\bllava\b/i,
  /\bminicpm-v\b/i,
  /\bqwen2-vl\b/i,
  /\bqwen2\.5-vl\b/i,
  /\bgpt-4o\b/i, // OpenAI vision-capable models
  /\bgpt-4-vision\b/i,
  /\bclaude-3(?!.*haiku.*$)/i, // Claude 3 family except haiku (text-only); kept conservative
  /\bgemini-1\.5/i, // Gemini 1.5 Pro/Flash accept images
  /\bgemini-2/i,
  /\binternvl\b/i,
  /\bcogvlm\b/i,
  /\bllama-3\.2-11b\b/i, // llama 3.2 vision variants
  /\bllama-3\.2-90b\b/i,
  /\bllama4\b/i, // llama 4 is multimodal
  /\bllama-4\b/i,
];

/**
 * Known tool-capable model patterns. Currently informational only — the
 * default for `tools` is already `true` (matching OpenCode's assumption), so
 * we don't *raise* `tools` here. This list is kept for a future override
 * path where `tools` would default to false for unknown ids and only these
 * patterns would opt a model back in. Exported in case downstream code wants
 * to inspect it.
 */
export const TOOL_PATTERNS: RegExp[] = [
  /\bfunction\b/i,
  /\btools?\b/i,
  /\bfc[-_]/i,
  /\bgpt-4o\b/i, // OpenAI gpt-4o supports tools
  /\bgpt-4-1106\b/i,
  /\bgpt-4-0125\b/i,
  /\bgpt-5\b/i,
  /\bclaude-3\.5-sonnet\b/i,
  /\bclaude-3-5\b/i,
  /\bclaude-3\.7\b/i,
  /\bclaude-4\b/i,
  /\bgemini-(1\.5|2)/i,
  /\bqwen2-72b\b/i,
  /\bqwen2\.5-72b\b/i,
  /\bdeepseek-v3\b/i,
  /\bdeepseek-v4\b/i,
  /\bglm-4\b/i,
  /\bglm-4\.5\b/i,
];

const REASONING_PATTERNS: RegExp[] = [
  /\b(o1|o3|o4)\b/i, // OpenAI reasoning family
  /\b-r1\b/i,
  /\bdeepseek-r1\b/i,
  /\bqwq\b/i,
  /\bqwen3.*think/i,
  /\breasoning\b/i,
  /\bthinker\b/i,
  // workbuddy-wild README marks hy3 and DeepSeek V4 Flash as reasoning.
  /\bhy3\b/i,
  /\bdeepseek-v4-flash\b/i,
  // oMLX thinking models
  /\bgpt-oss\b/i,
  /\bgemma-?3/i, // Gemma 3 chat template has thinking markers per llama.cpp
  /\bglm-z1\b/i,
  /\bglm-4-.*think/i,
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Infer capabilities from a model ID string only. Returns a fresh object;
 * callers can merge results from real metadata on top.
 *
 * The logic is:
 *  - Start from DEFAULT_CAPABILITIES (tools=true, input/output=[text], no reasoning).
 *  - If any VISION_PATTERN matches, add "image" to input (and keep "text").
 *  - If any TOOL_PATTERN matches, tools=true (already the default, but explicit).
 *    If none match AND the model id is a known small/specialized id (e.g.
 *    embedding, completion-only, code-only), set tools=false.
 *  - If any REASONING_PATTERN matches, reasoning=true.
 *
 * Note: We do NOT aggressively set tools=false on unknown local models,
 * because most modern instruct models served by llama.cpp/litellm can be
 * coaxed into tool use via the OpenAI format even if not officially supported.
 * OpenCode's fallback default is tools=true, and we mirror that here.
 */
export function inferCapabilities(modelId: string): NormalizedCapabilities {
  const caps: NormalizedCapabilities = {
    tools: DEFAULT_CAPABILITIES.tools,
    input: [...DEFAULT_CAPABILITIES.input],
    output: [...DEFAULT_CAPABILITIES.output],
    reasoning: false,
  };

  const id = modelId.toLowerCase();

  if (VISION_PATTERNS.some((re) => re.test(id))) {
    if (!caps.input.includes("image")) caps.input.push("image");
  }

  // Tools: optimistic default (true). Only turn OFF for clearly non-tool models.
  if (looksNonTool(id)) {
    caps.tools = false;
  }
  // Even for "tool-capable" patterns, we don't change caps.tools — it's already
  // true by default. The TOOL_PATTERNS list is reserved for future override logic.

  if (REASONING_PATTERNS.some((re) => re.test(id))) {
    caps.reasoning = true;
  }

  return caps;
}

/**
 * Heuristic: does this model id look like something that should NOT receive
 * tool-call payloads? Returns true for embedding/completion-only/code-only
 * names. Conservative — when unsure, returns false (keep tools on).
 */
function looksNonTool(id: string): boolean {
  return (
    /\bembedding\b/i.test(id) ||
    /\bembed\b/i.test(id) ||
    /-e5-/i.test(id) ||
    /\bbge\b/i.test(id) ||
    /\bmxbai-embed\b/i.test(id) ||
    /\bcompletion-only\b/i.test(id) ||
    /\bmoderation\b/i.test(id) ||
    /\brerank\b/i.test(id) ||
    /\bwhisper\b/i.test(id) ||
    /\btts\b/i.test(id) ||
    /\bdall-?e\b/i.test(id) ||
    /\bstable-diffusion\b/i.test(id)
  );
}

/**
 * Merge two capability sets. `b` (the higher-priority source, e.g. real
 * backend metadata) overrides `a` (the lower-priority source, e.g. name
 * heuristics) field-by-field. Arrays are unioned (de-duped, order preserved).
 *
 * For booleans, `b`'s value wins *only when defined* — undefined in `b`
 * means "fall back to `a`".
 */
export function mergeCapabilities(
  a: NormalizedCapabilities,
  b: Partial<NormalizedCapabilities>,
): NormalizedCapabilities {
  return {
    tools: b.tools ?? a.tools,
    input: unionPreservingOrder(a.input, b.input ?? []),
    output: unionPreservingOrder(a.output, b.output ?? []),
    reasoning: b.reasoning ?? a.reasoning,
  };
}

function unionPreservingOrder(...arrays: string[][]): string[] {
  const out: string[] = [];
  for (const arr of arrays) {
    for (const item of arr) {
      if (!out.includes(item)) out.push(item);
    }
  }
  return out;
}
