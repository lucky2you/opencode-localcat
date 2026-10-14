import { test } from "node:test";
import assert from "node:assert/strict";
import { ReadableStream } from "node:stream/web";

import { normalizeSseResponse, withReasoningStreamFix } from "../dist/stream-fetch.js";

function sseResponse(deltas, { contentType = "text/event-stream" } = {}) {
  const body = deltas
    .map((d) => `data: ${JSON.stringify({ object: "chat.completion.chunk", choices: [{ index: 0, delta: d }] })}\n\n`)
    .join("") + "data: [DONE]\n\n";
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(body));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { "content-type": contentType } });
}

async function readText(response) {
  return await response.text();
}

function parseDeltas(text) {
  return text
    .split("\n")
    .filter((l) => l.startsWith("data: ") && !l.includes("[DONE]"))
    .map((l) => JSON.parse(l.slice(6)).choices[0].delta);
}

// Mirrors opencode's bundled @ai-sdk/openai-compatible parser: a *present*
// tool_calls key (even an empty array) closes the open reasoning block.
function countReasoningBlocks(deltas) {
  let open = false;
  let blocks = 0;
  for (const C of deltas) {
    const reasoning = C.reasoning_content ?? C.reasoning;
    if (reasoning) {
      if (!open) {
        blocks++;
        open = true;
      }
    }
    if (C.content && open) open = false;
    if (C.tool_calls != null && open) open = false;
  }
  return blocks;
}

test("strips empty placeholder fields from streaming deltas", async () => {
  const raw = [
    { role: "assistant", content: "", reasoning_content: "", function_call: null, refusal: "", tool_calls: [], extra_fields: null },
    { content: "", reasoning_content: "The", function_call: null, refusal: "", tool_calls: [], extra_fields: null },
    { content: "", reasoning_content: " user", function_call: null, refusal: "", tool_calls: [], extra_fields: null },
    { content: "7", reasoning_content: "", function_call: null, refusal: "", tool_calls: [], extra_fields: null },
  ];
  const out = parseDeltas(await readText(normalizeSseResponse(sseResponse(raw))));

  assert.deepEqual(out[0], { role: "assistant" });
  assert.deepEqual(out[1], { reasoning_content: "The" });
  assert.deepEqual(out[2], { reasoning_content: " user" });
  assert.deepEqual(out[3], { content: "7" });
});

test("preserves real tool_calls and non-empty content", async () => {
  const raw = [
    { content: "", reasoning_content: "", tool_calls: [{ index: 0, id: "call_1", function: { name: "f", arguments: "{}" } }] },
    { content: "hello", reasoning_content: "" },
  ];
  const out = parseDeltas(await readText(normalizeSseResponse(sseResponse(raw))));

  assert.equal(out[0].tool_calls.length, 1);
  assert.equal(out[0].tool_calls[0].id, "call_1");
  assert.deepEqual(out[1], { content: "hello" });
});

test("normalizing workbuddy-style stream yields a single reasoning block", async () => {
  const raw = [
    { content: "", reasoning_content: "The", tool_calls: [] },
    { content: "", reasoning_content: " user", tool_calls: [] },
    { content: "", reasoning_content: " asks", tool_calls: [] },
    { content: "7", reasoning_content: "", tool_calls: [] },
  ];
  const fixed = parseDeltas(await readText(normalizeSseResponse(sseResponse(raw))));
  const original = raw;

  assert.equal(countReasoningBlocks(original), 3, "raw stream splits into per-token blocks");
  assert.equal(countReasoningBlocks(fixed), 1, "fixed stream aggregates into one block");
});

test("non-SSE responses are returned untouched", async () => {
  const json = new Response('{"ok":true}', { status: 200, headers: { "content-type": "application/json" } });
  const fetchImpl = withReasoningStreamFix(async () => json);
  const res = await fetchImpl("http://example.test/v1/chat/completions");
  assert.equal(await res.text(), '{"ok":true}');
});
