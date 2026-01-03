import { describe, it, expect } from "vitest";
import { renderHook } from "@testing-library/react";
import { useNodeLayout } from "../useNodeLayout";
import {
  RenderMode,
  MediaType,
  PortTypeSchema,
  PortSchema,
  PortStyle,
} from "../../generated/flowcraft/v1/core/node_pb";
import { PortMainType } from "../../generated/flowcraft/v1/core/base_pb";
import { type DynamicNodeData } from "../../types";
import { create } from "@bufbuild/protobuf";

describe("useNodeLayout", () => {
  it("should calculate correct minHeight for widget mode based on ports and widgets", () => {
    const data: DynamicNodeData = {
      label: "Node",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [
        create(PortSchema, {
          id: "in1",
          label: "In 1",
          type: create(PortTypeSchema, { mainType: PortMainType.STRING }),
          style: PortStyle.CIRCLE,
        }),
      ],
      outputPorts: [
        create(PortSchema, {
          id: "out1",
          label: "Out 1",
          type: create(PortTypeSchema, { mainType: PortMainType.STRING }),
          style: PortStyle.CIRCLE,
        }),
      ],
      widgets: [{ id: "w1", type: 1, label: "W1", value: 0 }],
    };

    const { result } = renderHook(() => useNodeLayout(data));

    // HEADER(46) + 1*PORT(24) + 1*WIDGET(55) + PADDING(20) = 145
    expect(result.current.minHeight).toBe(145);
    expect(result.current.isMedia).toBe(false);
  });

  it("should return media renderer config for media mode", () => {
    const data: DynamicNodeData = {
      label: "Image",
      modes: [RenderMode.MODE_MEDIA],
      activeMode: RenderMode.MODE_MEDIA,
      media: { type: MediaType.MEDIA_IMAGE, url: "test.jpg" },
    };

    const { result } = renderHook(() => useNodeLayout(data));

    expect(result.current.isMedia).toBe(true);
    expect(result.current.minHeight).toBe(50);
    expect(result.current.minWidth).toBe(180);
  });

  it("should handle audio-specific layout", () => {
    const data: DynamicNodeData = {
      label: "Audio",
      modes: [RenderMode.MODE_MEDIA],
      activeMode: RenderMode.MODE_MEDIA,
      media: { type: MediaType.MEDIA_AUDIO, url: "test.mp3" },
    };

    const { result } = renderHook(() => useNodeLayout(data));

    expect(result.current.isAudio).toBe(true);
    expect(result.current.minHeight).toBe(110);
    expect(result.current.minWidth).toBe(240);
  });
});
