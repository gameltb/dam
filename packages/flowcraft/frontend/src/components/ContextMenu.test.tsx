import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import ContextMenu from "./ContextMenu";
import { RenderMode } from "../generated/core/node_pb";

describe("ContextMenu", () => {
  it("renders basic menu items", () => {
    const onClose = vi.fn();
    const onAddNode = vi.fn();
    const onAutoLayout = vi.fn();

    const templates = [
      {
        id: "test-node",
        label: "Test Node",
        path: ["Basic"],
        defaultData: {
          label: "Test",
          modes: [RenderMode.RENDER_MODE_WIDGETS],
          activeMode: RenderMode.RENDER_MODE_WIDGETS,
          widgets: [],
        },
      },
    ];

    render(
      <ContextMenu
        x={0}
        y={0}
        onClose={onClose}
        templates={templates}
        onAddNode={onAddNode}
        onAutoLayout={onAutoLayout}
        isPaneMenu={true}
      />,
    );

    // Check if sections are rendered
    expect(screen.getByText("ADD NODE")).toBeDefined();
    expect(screen.getByText("🪄 Auto Layout")).toBeDefined();
  });
});