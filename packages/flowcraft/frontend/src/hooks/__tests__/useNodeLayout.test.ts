import { create } from "@bufbuild/protobuf";
import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { NodeDataSchema, PortTypeSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { MediaType, PortMainType, PortStyle, RenderMode } from "@/types";
import { type DynamicNodeData } from "@/types";

import { useNodeLayout } from "../useNodeLayout";

describe("useNodeLayout", () => {
  it("should calculate correct minHeight for widget mode based on ports and widgets", () => {
    const data = create(NodeDataSchema, {
      activeMode: RenderMode.MODE_WIDGETS,
      availableModes: [RenderMode.MODE_WIDGETS],
      displayName: "Node",
      inputPorts: [
        {
          id: "in1",
          label: "In 1",
          style: PortStyle.CIRCLE,
          type: create(PortTypeSchema, {
            isGeneric: false,
            itemType: "",
            mainType: PortMainType.STRING,
          }),
        },
      ],
      outputPorts: [
        {
          id: "out1",
          label: "Out 1",
          style: PortStyle.CIRCLE,
          type: create(PortTypeSchema, {
            isGeneric: false,
            itemType: "",
            mainType: PortMainType.STRING,
          }),
        },
      ],
      widgets: [{ id: "w1", label: "W1", type: 1 }],
    }) as DynamicNodeData;

    const { result } = renderHook(() => useNodeLayout(data));

    // HEADER(46) + 1*PORT(24) + 1*WIDGET(55) + PADDING(20) = 145
    expect(result.current.minHeight).toBe(145);
    expect(result.current.isMedia).toBe(false);
  });

  it("should return media renderer config for media mode", () => {
    const data = create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      availableModes: [RenderMode.MODE_MEDIA],
      displayName: "Image",
      media: { type: MediaType.MEDIA_IMAGE, url: "test.jpg" },
    }) as DynamicNodeData;

    const { result } = renderHook(() => useNodeLayout(data));

    expect(result.current.isMedia).toBe(true);
    expect(result.current.minHeight).toBe(50);
    expect(result.current.minWidth).toBe(180);
  });

  it("should handle audio-specific layout", () => {
    const data = create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      availableModes: [RenderMode.MODE_MEDIA],
      displayName: "Audio",
      media: { type: MediaType.MEDIA_AUDIO, url: "test.mp3" },
    }) as DynamicNodeData;

    const { result } = renderHook(() => useNodeLayout(data));

    expect(result.current.isAudio).toBe(true);
    expect(result.current.minHeight).toBe(110);
    expect(result.current.minWidth).toBe(240);
  });
});
