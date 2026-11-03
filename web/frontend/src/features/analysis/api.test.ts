import { afterEach, describe, expect, test, vi } from "vitest";

import { createRun } from "./api";


afterEach(() => vi.unstubAllGlobals());


describe("analysis API", () => {
  test("turns FastAPI validation details into a readable error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      detail: [{ loc: ["body", "inputs"], msg: "Value error, tensor inputs exceed the limit", type: "value_error" }],
    }), { status: 422, headers: { "Content-Type": "application/json" } })));

    await expect(createRun({
      operator_code: "class Model: pass",
      inputs: [{ name: "x", shape: [1], dtype: "float32" }],
      hardware: "H100_PCIe",
    })).rejects.toThrow("inputs: Value error, tensor inputs exceed the limit");
  });
});
