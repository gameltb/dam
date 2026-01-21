import { describe, expect, it } from "vitest";

import { applyPathToObj } from "@/../spacetime-module/src/utils/path-utils";

describe("PathUpdate Core Logic (Non-Spacetime)", () => {
  it("should update a top-level property", () => {
    const obj = { name: "old" };
    const result = applyPathToObj(obj, ["name"], "new", 0); // 0 = REPLACE
    expect(result.name).toBe("new");
  });

  it("should update a nested property", () => {
    const obj = { data: { profile: { name: "old" } } };
    const result = applyPathToObj(obj, ["data", "profile", "name"], "new", 0);
    expect(result.data.profile.name).toBe("new");
  });

  it("should delete a property", () => {
    const obj = { data: { keep: "stay", temp: "delete-me" } };
    const result = applyPathToObj(obj, ["data", "temp"], null, 2); // 2 = DELETE
    expect(result.data.temp).toBeUndefined();
    expect(result.data.keep).toBe("stay");
  });

  it("should create path if it doesn't exist during replace", () => {
    const obj: any = {};
    const result = applyPathToObj(obj, ["a", "b", "c"], 123, 0);
    expect(result.a.b.c).toBe(123);
  });
});
