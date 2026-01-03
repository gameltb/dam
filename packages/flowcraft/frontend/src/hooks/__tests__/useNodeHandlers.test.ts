/**
 * @file useNodeHandlers.test.ts
 * @problem Node logic (style, mode, handles) was tightly coupled, making it hard to test selection and media-specific behaviors.
 * @requirement Ensure that node handlers correctly calculate selection styles, mode-specific min-heights, and aspect ratio locking for media nodes.
 */
import { describe, it, expect } from "vitest";
import { renderHook } from "@testing-library/react";
import { useNodeHandlers } from "../useNodeHandlers";
import { MediaType, RenderMode } from "../../generated/flowcraft/v1/node_pb";
import { type DynamicNodeData } from "../../types";

describe("useNodeHandlers", () => {
  it("should calculate correct styles and minHeight for widget mode", () => {
    const data: DynamicNodeData = {
      label: "Test Node",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      widgets: [{ id: "w1", type: 1, label: "W1", value: 0 }],
      inputPorts: [],
      outputPorts: [],
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
      label: "Image",
      modes: [RenderMode.MODE_MEDIA],
      activeMode: RenderMode.MODE_MEDIA,
      media: {
        type: MediaType.MEDIA_IMAGE,
        url: "test.png",
      },
    };

    const { result } = renderHook(() => useNodeHandlers(data, false));

    expect(result.current.isMedia).toBe(true);
    expect(result.current.shouldLockAspectRatio).toBe(true);
  });
});
