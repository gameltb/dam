import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MediaType, PortMainType, PortStyle, RenderMode } from "@/types";
import { type DynamicNodeData } from "@/types";

import { useNodeLayout } from "../useNodeLayout";

describe("useNodeLayout", () => {
  it("should calculate correct minHeight for widget mode based on ports and widgets", () => {
    const data: DynamicNodeData = {
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [
        {
          id: "in1",
          label: "In 1",
          style: PortStyle.CIRCLE,
          type: {
            isGeneric: false,
            itemType: "",
            mainType: PortMainType.STRING,
          },
        },
      ],
      label: "Node",
      modes: [RenderMode.MODE_WIDGETS],
      outputPorts: [
        {
          id: "out1",
          label: "Out 1",
          style: PortStyle.CIRCLE,
          type: {
            isGeneric: false,
            itemType: "",
            mainType: PortMainType.STRING,
          },
        },
      ],
      widgets: [{ id: "w1", label: "W1", type: 1, value: 0 }],
    };

    const { result } = renderHook(() => useNodeLayout(data));

    // HEADER(46) + 1*PORT(24) + 1*WIDGET(55) + PADDING(20) = 145
    expect(result.current.minHeight).toBe(145);
    expect(result.current.isMedia).toBe(false);
  });

  it("should return media renderer config for media mode", () => {
    const data: DynamicNodeData = {
      activeMode: RenderMode.MODE_MEDIA,
      label: "Image",
      media: { type: MediaType.MEDIA_IMAGE, url: "test.jpg" },
      modes: [RenderMode.MODE_MEDIA],
    };

    const { result } = renderHook(() => useNodeLayout(data));

    expect(result.current.isMedia).toBe(true);
    expect(result.current.minHeight).toBe(50);
    expect(result.current.minWidth).toBe(180);
  });

  it("should handle audio-specific layout", () => {
    const data: DynamicNodeData = {
      activeMode: RenderMode.MODE_MEDIA,
      label: "Audio",
      media: { type: MediaType.MEDIA_AUDIO, url: "test.mp3" },
      modes: [RenderMode.MODE_MEDIA],
    };

    const { result } = renderHook(() => useNodeLayout(data));

    expect(result.current.isAudio).toBe(true);
    expect(result.current.minHeight).toBe(110);
    expect(result.current.minWidth).toBe(240);
  });
});
