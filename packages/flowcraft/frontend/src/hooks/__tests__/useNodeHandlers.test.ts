import { renderHook } from "@testing-library/react";
/**
 * @file useNodeHandlers.test.ts
 * @problem Node logic (style, mode, handles) was tightly coupled, making it hard to test selection and media-specific behaviors.
 * @requirement Ensure that node handlers correctly calculate selection styles, mode-specific min-heights, and aspect ratio locking for media nodes.
 */
import { describe, expect, it } from "vitest";

import { MediaType, RenderMode } from "@/types";
import { type DynamicNodeData } from "@/types";

import { useNodeHandlers } from "../useNodeHandlers";

describe("useNodeHandlers", () => {
  it("should calculate correct styles and minHeight for widget mode", () => {
    const data: DynamicNodeData = {
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [],
      label: "Test Node",
      modes: [RenderMode.MODE_WIDGETS],
      outputPorts: [],
      widgets: [{ id: "w1", label: "W1", type: 1, value: 0 }],
    };

    const { result } = renderHook(() => useNodeHandlers(data, false));

    expect(result.current.isMedia).toBe(false);
    expect(result.current.minHeight).toBeGreaterThan(100); // Header + Widget + Padding
    expect(result.current.containerStyle.borderColor).toBe(
      "var(--node-border)",
    );
  });

  it("should calculate correct styles for selected state", () => {
    const data: DynamicNodeData = {
      label: "Test Node",
      modes: [RenderMode.MODE_WIDGETS],
    };

    const { result } = renderHook(() => useNodeHandlers(data, true));

    expect(result.current.containerStyle.borderColor).toBe(
      "var(--primary-color)",
    );
    expect(result.current.containerStyle.boxShadow).toContain(
      "var(--primary-color)",
    );
  });

  it("should lock aspect ratio for image media", () => {
    const data: DynamicNodeData = {
      activeMode: RenderMode.MODE_MEDIA,
      label: "Image",
      media: {
        type: MediaType.MEDIA_IMAGE,
        url: "test.png",
      },
      modes: [RenderMode.MODE_MEDIA],
    };

    const { result } = renderHook(() => useNodeHandlers(data, false));

    expect(result.current.isMedia).toBe(true);
    expect(result.current.shouldLockAspectRatio).toBe(true);
  });
});
