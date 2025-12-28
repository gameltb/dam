import { describe, it, expect } from "vitest";
import { renderHook } from "@testing-library/react";
import { useNodeLayout } from "../useNodeLayout";
import { flowcraft_proto } from "../../generated/flowcraft_proto";
import type { DynamicNodeData } from "../../types";

const RenderMode = flowcraft_proto.v1.RenderMode;
const MediaType = flowcraft_proto.v1.MediaType;

describe("useNodeLayout", () => {
  it("calculates min height for widgets mode correctly", () => {
    const data: DynamicNodeData = {
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [{ id: "1", name: "in", type: "string" }],
      outputPorts: [{ id: "2", name: "out", type: "string" }],
      widgets: [{ id: "w1", type: "text", label: "W1", value: "" }],
    };

    const { result } = renderHook(() => useNodeLayout(data));

    // HEADER(46) + PORTS(1*24) + WIDGETS(1*55) + PADDING(20) = 46 + 24 + 55 + 20 = 145
    expect(result.current.minHeight).toBe(145);
    expect(result.current.isMedia).toBe(false);
  });

  it("calculates min height for media mode (default)", () => {
    const data: DynamicNodeData = {
      activeMode: RenderMode.MODE_MEDIA,
      media: { type: MediaType.MEDIA_IMAGE, url: "test.jpg" },
    };

    const { result } = renderHook(() => useNodeLayout(data));

    expect(result.current.minHeight).toBe(50);
    expect(result.current.isMedia).toBe(true);
  });

  it("calculates min height for audio media", () => {
    const data: DynamicNodeData = {
      activeMode: RenderMode.MODE_MEDIA,
      media: { type: MediaType.MEDIA_AUDIO, url: "test.mp3" },
    };

    const { result } = renderHook(() => useNodeLayout(data));

    expect(result.current.minHeight).toBe(110);
    expect(result.current.isAudio).toBe(true);
  });
});
