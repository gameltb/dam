import { create } from "@bufbuild/protobuf";
import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { NodeDataSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { MediaType, RenderMode } from "@/types";
import { type DynamicNodeData } from "@/types";

import { useNodeHandlers } from "../useNodeHandlers";

describe("useNodeHandlers", () => {
  it("should calculate correct styles and minHeight for widget mode", () => {
    const data = create(NodeDataSchema, {
      activeMode: RenderMode.MODE_WIDGETS,
      availableModes: [RenderMode.MODE_WIDGETS],
      displayName: "Test Node",
      inputPorts: [],
      outputPorts: [],
      widgets: [{ id: "w1", label: "W1", type: 1 }],
    }) as DynamicNodeData;

    const { result } = renderHook(() => useNodeHandlers(data, false));

    expect(result.current.isMedia).toBe(false);
    expect(result.current.minHeight).toBeGreaterThan(100); // Header + Widget + Padding
    expect(result.current.containerStyle.borderColor).toBe("var(--node-border)");
  });

  it("should calculate correct styles for selected state", () => {
    const data = create(NodeDataSchema, {
      availableModes: [RenderMode.MODE_WIDGETS],
      displayName: "Test Node",
    }) as DynamicNodeData;

    const { result } = renderHook(() => useNodeHandlers(data, true));

    expect(result.current.containerStyle.borderColor).toBe("var(--primary-color)");
    expect(result.current.containerStyle.boxShadow).toContain("var(--primary-color)");
  });

  it("should lock aspect ratio for image media", () => {
    const data = create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      availableModes: [RenderMode.MODE_MEDIA],
      displayName: "Image",
      media: {
        type: MediaType.MEDIA_IMAGE,
        url: "test.png",
      },
    }) as DynamicNodeData;

    const { result } = renderHook(() => useNodeHandlers(data, false));

    expect(result.current.isMedia).toBe(true);
    expect(result.current.shouldLockAspectRatio).toBe(true);
  });
});
