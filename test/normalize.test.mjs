import { test } from "node:test";
import assert from "node:assert/strict";

import { toProviderConfig, toProviderRegistration } from "../dist/normalize.js";

function backend(overrides = {}) {
  return {
    id: "workbuddy-wild",
    online: true,
    baseURL: "http://127.0.0.1:7863/v1",
    apiKey: "WorkBuddy2API",
    modelIds: ["workbuddy/hy3"],
    extended: {},
    reasoningField: "reasoning_content",
    ...overrides,
  };
}

test("registers a V2 Provider.Info from a discovered backend", () => {
  const reg = toProviderRegistration(backend());

  assert.equal(reg.info.id, "workbuddy-wild");
  assert.equal(reg.info.name, "WorkBuddy-Wild");
  assert.equal(reg.info.activation, "enabled");
  assert.equal(reg.info.package, "@opencode/ai/providers/openai-compatible");
  assert.equal(reg.info.settings.baseURL, "http://127.0.0.1:7863/v1");
  assert.equal(reg.info.settings.apiKey, "WorkBuddy2API");
});

test("uses extended metadata for context/output limits when present", () => {
  const reg = toProviderRegistration(
    backend({
      extended: {
        "workbuddy/hy3": {
          capabilities: { tools: true, input: ["text"], output: ["text"] },
          limits: { context: 131_072, output: 32_768, input: 131_072 },
        },
      },
    }),
  );

  const model = reg.models[0];
  assert.equal(model.limit.context, 131_072);
  assert.equal(model.limit.output, 32_768);
  assert.equal(model.limit.input, 131_072);
});

test("falls back to opencode defaults when metadata is missing", () => {
  const model = toProviderRegistration(backend()).models[0];
  assert.equal(model.limit.context, 200_000);
  assert.equal(model.limit.output, 32_000);
});

test("sets compatibility.reasoningField from the backend default", () => {
  const model = toProviderRegistration(backend()).models[0];
  assert.equal(model.compatibility.reasoningField, "reasoning_content");
});

test("per-model reasoningField override wins over the backend default", () => {
  const reg = toProviderRegistration(
    backend({
      modelIds: ["workbuddy/hy3", "traework/DeepSeek-V4-Flash"],
      modelOverrides: { "traework/DeepSeek-V4-Flash": "reasoning" },
    }),
  );

  const hy3 = reg.models.find((m) => m.id === "workbuddy/hy3");
  const flash = reg.models.find((m) => m.id === "traework/DeepSeek-V4-Flash");
  assert.equal(hy3.compatibility.reasoningField, "reasoning_content");
  assert.equal(flash.compatibility.reasoningField, "reasoning");
});

test("keeps the full workbuddy model id and filters empty ids", () => {
  const reg = toProviderRegistration(
    backend({ modelIds: ["workbuddy/hy3", "", "traework/DeepSeek-V4-Flash"] }),
  );

  assert.equal(reg.models.length, 2);
  assert.equal(reg.models[0].id, "workbuddy/hy3");
  assert.equal(reg.models[0].name, "workbuddy/hy3");
  assert.equal(reg.models[0].modelID, "workbuddy/hy3");
});

test("labels non-workbuddy models with the backend display name", () => {
  const reg = toProviderRegistration(
    backend({
      id: "litellm",
      baseURL: "http://127.0.0.1:4000/v1",
      apiKey: "sk-litellm-proxy",
      modelIds: ["openai/gpt-4o"],
      reasoningField: undefined,
    }),
  );

  assert.equal(reg.models[0].name, "gpt-4o (LiteLLM)");
  // No reasoningField configured -> no compatibility block.
  assert.equal(reg.models[0].compatibility, undefined);
});

// ---------------------------------------------------------------------------
// V1 shape: `provider.<id>` config block
// ---------------------------------------------------------------------------

test("V1 block uses the @ai-sdk/openai-compatible npm package", () => {
  const block = toProviderConfig(backend());

  assert.equal(block.name, "WorkBuddy-Wild");
  assert.equal(block.npm, "@ai-sdk/openai-compatible");
  assert.equal(block.options.baseURL, "http://127.0.0.1:7863/v1");
  assert.equal(block.options.apiKey, "WorkBuddy2API");
  // The V1 path installs the custom SSE-normalizing fetch.
  assert.equal(typeof block.options.fetch, "function");
});

test("V1 model entry carries interleaved + tool_call + limits", () => {
  const block = toProviderConfig(
    backend({
      extended: {
        "workbuddy/hy3": {
          capabilities: { tools: true, input: ["text", "image"], output: ["text"] },
          limits: { context: 131_072, output: 32_768, input: 131_072 },
        },
      },
    }),
  );

  const model = block.models["workbuddy/hy3"];
  assert.equal(model.name, "workbuddy/hy3");
  assert.equal(model.interleaved.field, "reasoning_content");
  assert.equal(model.tool_call, true);
  assert.equal(model.attachment, true);
  assert.equal(model.limit.context, 131_072);
  assert.equal(model.limit.output, 32_768);
  assert.equal(model.limit.input, 131_072);
});

test("V1 per-model reasoningField override wins over the backend default", () => {
  const block = toProviderConfig(
    backend({
      modelIds: ["workbuddy/hy3", "traework/DeepSeek-V4-Flash"],
      modelOverrides: { "traework/DeepSeek-V4-Flash": "reasoning" },
    }),
  );

  assert.equal(block.models["workbuddy/hy3"].interleaved.field, "reasoning_content");
  assert.equal(block.models["traework/DeepSeek-V4-Flash"].interleaved.field, "reasoning");
});

test("V1 filters empty ids and labels non-workbuddy models", () => {
  const block = toProviderConfig(
    backend({
      id: "litellm",
      baseURL: "http://127.0.0.1:4000/v1",
      apiKey: "sk-litellm-proxy",
      modelIds: ["openai/gpt-4o", ""],
      reasoningField: undefined,
    }),
  );

  assert.deepEqual(Object.keys(block.models), ["openai/gpt-4o"]);
  assert.equal(block.models["openai/gpt-4o"].name, "gpt-4o (LiteLLM)");
  assert.equal(block.models["openai/gpt-4o"].interleaved, undefined);
});
