import { describe, it, expect } from "vitest";
import { renderHook } from "@testing-library/react";
import { useNodeLayout } from "../useNodeLayout";
import {
  RenderMode,
  MediaType,
  WidgetType,
  PortStyle,
  PortTypeSchema,
  PortSchema,
} from "../../generated/core/node_pb";
import type { DynamicNodeData } from "../../types";
import { create } from "@bufbuild/protobuf";

describe("useNodeLayout", () => {
  it("calculates min height for widgets mode correctly", () => {
    const data: DynamicNodeData = {
      label: "Test",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [
        create(PortSchema, {
          id: "1",
          label: "in",
          type: create(PortTypeSchema, { mainType: "string" }),
          style: PortStyle.CIRCLE,
        }),
      ],
      outputPorts: [
        create(PortSchema, {
          id: "2",
          label: "out",
          type: create(PortTypeSchema, { mainType: "string" }),
          style: PortStyle.CIRCLE,
        }),
      ],
      widgets: [
        {
          id: "w1",
          type: WidgetType.WIDGET_TEXT,
          label: "W1",
          value: "",
        },
      ],
    };

    const { result } = renderHook(() => useNodeLayout(data));

    // HEADER(46) + PORTS(1*24) + WIDGETS(1*55) + PADDING(20) = 46 + 24 + 55 + 20 = 145
    expect(result.current.minHeight).toBe(145);
    expect(result.current.isMedia).toBe(false);
  });

  it("calculates min height for media mode (default)", () => {
    const data: DynamicNodeData = {
      label: "Test",
      modes: [RenderMode.MODE_MEDIA],
      activeMode: RenderMode.MODE_MEDIA,
      media: { type: MediaType.MEDIA_IMAGE, url: "test.jpg" },
    };

    const { result } = renderHook(() => useNodeLayout(data));

    expect(result.current.minHeight).toBe(50);
    expect(result.current.isMedia).toBe(true);
  });

  it("calculates min height for audio media", () => {
    const data: DynamicNodeData = {
      label: "Test",
      modes: [RenderMode.MODE_MEDIA],
      activeMode: RenderMode.MODE_MEDIA,
      media: { type: MediaType.MEDIA_AUDIO, url: "test.mp3" },
    };

    const { result } = renderHook(() => useNodeLayout(data));

    expect(result.current.minHeight).toBe(110);
    expect(result.current.isAudio).toBe(true);
  });
});
