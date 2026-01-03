import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { PaneContextMenu } from "./PaneContextMenu";
import { create } from "@bufbuild/protobuf";
import {
  NodeTemplateSchema,
  RenderMode,
} from "../../generated/flowcraft/v1/node_pb";
import { toProtoNodeData } from "../../utils/protoAdapter";

describe("PaneContextMenu", () => {
  it("renders basic menu items", () => {
    const onClose = vi.fn();
    const onAddNode = vi.fn();
    const onAutoLayout = vi.fn();

    const templates = [
      create(NodeTemplateSchema, {
        templateId: "test-node",
        displayName: "Test Node",
        menuPath: ["Basic"],
        defaultState: toProtoNodeData({
          label: "Test",
          modes: [RenderMode.MODE_WIDGETS],
          activeMode: RenderMode.MODE_WIDGETS,
          widgets: [],
        }),
      }),
    ];

    render(
      <PaneContextMenu
        x={0}
        y={0}
        onClose={onClose}
        templates={templates}
        onAddNode={onAddNode}
        onAutoLayout={onAutoLayout}
      />,
    );

    // Check if sections are rendered
    expect(screen.getByText("ADD NODE")).toBeDefined();
    expect(screen.getByText("🪄 Auto Layout")).toBeDefined();
  });
});
