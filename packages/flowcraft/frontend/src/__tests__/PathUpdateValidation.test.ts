import { describe, expect, it } from "vitest";

import { validateValueByPath } from "@/../spacetime-module/src/utils/type-validator";
import { NodeSchema } from "@/generated/flowcraft/v1/core/node_pb";

describe("PathUpdate Type Validation", () => {
  it("should validate a correct string property", () => {
    expect(() => {
      validateValueByPath(NodeSchema, ["state", "displayName"], "New Name");
    }).not.toThrow();
  });

  it("should validate a correct enum property", () => {
    expect(() => {
      validateValueByPath(NodeSchema, ["state", "activeMode"], 1); // 1 = RenderMode.MODE_MEDIA
    }).not.toThrow();
  });

  it("should throw error if type mismatch (string instead of enum)", () => {
    expect(() => {
      validateValueByPath(NodeSchema, ["state", "activeMode"], "Invalid");
    }).toThrow(/Expected enum/);
  });

  it("should throw error if path is invalid", () => {
    expect(() => {
      validateValueByPath(NodeSchema, ["state", "nonExistentField"], 123);
    }).toThrow(/Field 'nonExistentField' not found/);
  });

  it("should throw error if traversing a scalar as message", () => {
    expect(() => {
      validateValueByPath(NodeSchema, ["state", "displayName", "subField"], "val");
    }).toThrow(/terminal type/);
  });
});
