/**
 * Stream normalization for OpenAI-compatible backends.
 *
 * Some servers (notably workbuddy-wild's `workbuddy/*` passthrough) include
 * empty placeholder fields on every streaming delta, e.g.:
 *
 *   {"delta":{"content":"","reasoning_content":"The","function_call":null,
 *             "refusal":"","tool_calls":[],"extra_fields":null}}
 *
 * opencode's bundled `@ai-sdk/openai-compatible` parser treats a *present*
 * `tool_calls` key as a tool call (`if (delta.tool_calls != null)`), and an
 * empty array `[]` satisfies that check. The parser then closes the open
 * reasoning block before every subsequent delta, so each 1-2 token reasoning
 * chunk renders as its own "Thought" block instead of one aggregated block.
 *
 * We install a custom `fetch` on the provider (opencode forwards
 * `provider.<id>.options.fetch` to `createOpenAICompatible`). It rewrites
 * streaming SSE chunks to drop these empty placeholders, matching the shape a
 * well-behaved OpenAI server emits. Without them the parser aggregates
 * reasoning correctly.
 */

/** A fetch implementation, matching the WHATWG signature. */
export type FetchFn = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

/** Drop empty placeholder fields from a single streaming `delta` object. */
function stripEmptyDelta(delta: Record<string, unknown>): boolean {
  let changed = false;
  if (delta.content === "") {
    delete delta.content;
    changed = true;
  }
  if (delta.reasoning_content === "") {
    delete delta.reasoning_content;
    changed = true;
  }
  if (delta.reasoning === "") {
    delete delta.reasoning;
    changed = true;
  }
  if (delta.reasoning_text === "") {
    delete delta.reasoning_text;
    changed = true;
  }
  if (Array.isArray(delta.tool_calls) && delta.tool_calls.length === 0) {
    delete delta.tool_calls;
    changed = true;
  }
  if (delta.function_call === null) {
    delete delta.function_call;
    changed = true;
  }
  if (delta.refusal === "" || delta.refusal === null) {
    delete delta.refusal;
    changed = true;
  }
  if (delta.extra_fields === null) {
    delete delta.extra_fields;
    changed = true;
  }
  return changed;
}

/** Rewrite one SSE line, returning the original when nothing changed. */
function rewriteSseLine(line: string): string {
  if (!line.startsWith("data:")) return line;
  const payload = line.slice("data:".length).trimStart();
  if (payload === "[DONE]") return line;
  let chunk: unknown;
  try {
    chunk = JSON.parse(payload);
  } catch {
    return line;
  }
  if (!chunk || typeof chunk !== "object") return line;

  let changed = false;
  const choices = (chunk as { choices?: unknown }).choices;
  if (Array.isArray(choices)) {
    for (const choice of choices) {
      if (!choice || typeof choice !== "object") continue;
      const delta = (choice as { delta?: unknown }).delta;
      if (delta && typeof delta === "object" && !Array.isArray(delta)) {
        if (stripEmptyDelta(delta as Record<string, unknown>)) changed = true;
      }
    }
  }
  return changed ? `data: ${JSON.stringify(chunk)}` : line;
}

/** Wrap an SSE Response body with a transform that prunes empty delta fields. */
export function normalizeSseResponse(response: Response): Response {
  if (!response.body) return response;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffered = "";

  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) {
        if (buffered) controller.enqueue(encoder.encode(rewriteSseLine(buffered)));
        controller.close();
        return;
      }
      buffered += decoder.decode(value, { stream: true });
      const lines = buffered.split("\n");
      buffered = lines.pop() ?? "";
      if (lines.length > 0) {
        let out = "";
        for (const line of lines) out += `${rewriteSseLine(line)}\n`;
        controller.enqueue(encoder.encode(out));
      }
    },
    cancel(reason) {
      return reader.cancel(reason);
    },
  });

  const headers = new Headers(response.headers);
  // Body is re-encoded from the already-decoded stream, so length/encoding
  // headers no longer describe it.
  headers.delete("content-length");
  headers.delete("content-encoding");
  return new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

/**
 * Wrap a fetch so streaming chat-completion responses have their empty
 * placeholder delta fields stripped. Non-SSE responses pass through untouched.
 */
export function withReasoningStreamFix(baseFetch: FetchFn): FetchFn {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const response = await baseFetch(input, init);
    const contentType = response.headers.get("content-type") ?? "";
    if (response.body && contentType.includes("text/event-stream")) {
      return normalizeSseResponse(response);
    }
    return response;
  };
}

/** Default fetch used as the base implementation. */
export function defaultFetch(): FetchFn {
  return globalThis.fetch.bind(globalThis) as FetchFn;
}
