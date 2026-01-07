/**
 * @file nodeUtils.test.ts
 * @problem Complex node objects with circular references or functions cannot be safely serialized to Protobuf or Yjs.
 * @requirement Provide a robust dehydrateNode utility that recursively removes functions and undefined values, ensuring serializable node state.
 */
import { describe, expect, it } from "vitest";

import { dehydrateNode } from "../nodeUtils";

describe("nodeUtils - dehydrateNode", () => {
  it("should remove functions from an object", () => {
    const input = {
      data: {
        fn: () => {
          console.log("hello");
        },
        label: "Test",
      },
      id: "1",
      onDrag: () => {
        /* empty */
      },
    };

    const result = dehydrateNode(input);

    expect(result).toEqual({
      data: {
        label: "Test",
      },
      id: "1",
    });
    expect((result as Record<string, unknown>).onDrag).toBeUndefined();
    expect(
      ((result as Record<string, unknown>).data as Record<string, unknown>).fn,
    ).toBeUndefined();
  });

  it("should recursively clean arrays", () => {
    const input = [
      {
        fn: () => {
          /* empty */
        },
        id: "1",
      },
      {
        data: {
          fn: () => {
            /* empty */
          },
        },
        id: "2",
      },
    ];

    const result = dehydrateNode(input);

    expect(result).toEqual([{ id: "1" }, { data: {}, id: "2" }]);
  });

  it("should handle null and primitives", () => {
    expect(dehydrateNode(null)).toBe(null);
    expect(dehydrateNode(123)).toBe(123);
    expect(dehydrateNode("test")).toBe("test");
  });

  it("should remove undefined values", () => {
    const input = { a: 1, b: undefined };
    const result = dehydrateNode(input);
    expect(result).toEqual({ a: 1 });
    expect(result).not.toHaveProperty("b");
  });
});
