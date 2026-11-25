import { test } from "node:test";
import assert from "node:assert/strict";

import {
  hasWorkBuddyPrefix,
  isWorkBuddyModelId,
  WORKBUDDY_WILD_BARE_ID_ALLOWLIST,
  WORKBUDDY_WILD_SYNTHETIC_OWNER,
} from "../dist/backends.js";

test("hasWorkBuddyPrefix accepts only the workbuddy/ and traework/ prefixes", () => {
  assert.equal(hasWorkBuddyPrefix("workbuddy/hy4flash-free"), true);
  assert.equal(hasWorkBuddyPrefix("traework/glm-5.2"), true);
  assert.equal(hasWorkBuddyPrefix("auto-free"), false);
  assert.equal(hasWorkBuddyPrefix("auto-nofree"), false);
});

test("isWorkBuddyModelId keeps prefixed ids", () => {
  assert.equal(isWorkBuddyModelId("workbuddy/auto"), true);
  assert.equal(isWorkBuddyModelId("workbuddy/hy4flash-free"), true);
  assert.equal(isWorkBuddyModelId("traework/DeepSeek-V4-Flash"), true);
});

test("isWorkBuddyModelId keeps synthetic auto aliases via owned_by", () => {
  // Both `auto-free` and `auto-nofree` are owned_by:"synthetic" and are valid,
  // callable model ids (a 503 account cooldown is not an invalid-model error).
  assert.equal(isWorkBuddyModelId("auto-free", WORKBUDDY_WILD_SYNTHETIC_OWNER), true);
  assert.equal(isWorkBuddyModelId("auto-nofree", WORKBUDDY_WILD_SYNTHETIC_OWNER), true);
});

test("isWorkBuddyModelId keeps known bare aliases even without owned_by", () => {
  // Fallback for responses that omit `owned_by`.
  assert.equal(isWorkBuddyModelId("auto-free"), true);
  assert.equal(isWorkBuddyModelId("auto-nofree"), true);
  assert.ok(WORKBUDDY_WILD_BARE_ID_ALLOWLIST.includes("auto-free"));
  assert.ok(WORKBUDDY_WILD_BARE_ID_ALLOWLIST.includes("auto-nofree"));
});

test("isWorkBuddyModelId still rejects non-synthetic bare ids", () => {
  assert.equal(isWorkBuddyModelId("some-random-model"), false);
  assert.equal(isWorkBuddyModelId("some-random-model", "someone-else"), false);
  assert.equal(isWorkBuddyModelId(""), false);
});
