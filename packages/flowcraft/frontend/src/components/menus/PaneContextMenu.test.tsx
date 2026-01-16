import { create } from "@bufbuild/protobuf";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { NodeTemplateSchema, RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { appNodeDataToProto } from "@/utils/nodeProtoUtils";

import { PaneContextMenu } from "./PaneContextMenu";

describe("PaneContextMenu", () => {
  it("renders basic menu items", () => {
    const onClose = vi.fn();
    const onAddNode = vi.fn();
    const onAutoLayout = vi.fn();

    const templates = [
      create(NodeTemplateSchema, {
        defaultState: appNodeDataToProto({
          activeMode: RenderMode.MODE_WIDGETS,
          availableModes: [RenderMode.MODE_WIDGETS],
          displayName: "Test",
          widgets: [],
        } as any),
        displayName: "Test Node",
        menuPath: ["Basic"],
        templateId: "test-node",
      }),
    ];

    render(
      <PaneContextMenu
        onAddNode={onAddNode}
        onAutoLayout={onAutoLayout}
        onClose={onClose}
        templates={templates}
        x={0}
        y={0}
      />,
    );

    // Check if sections are rendered
    expect(screen.getByText(/Add Node/i)).toBeDefined();
    expect(screen.getByText("🪄 Auto Layout")).toBeDefined();
  });
});
