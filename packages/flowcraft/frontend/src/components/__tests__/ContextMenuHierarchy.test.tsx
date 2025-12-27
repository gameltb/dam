import { describe, it, expect, vi } from "vitest";
import { render, fireEvent, screen } from "@testing-library/react";
import { ContextMenu } from "../ContextMenu";
import { RenderMode } from "../../types";

describe("ContextMenu - Template Hierarchy", () => {
  const mockTemplates = [
    {
      id: "t1",
      label: "Text Node",
      path: ["Input", "Basic"],
      defaultData: {
        label: "T",
        modes: [RenderMode.MODE_WIDGETS],
        activeMode: RenderMode.MODE_WIDGETS,
        widgets: [],
      },
    },
    {
      id: "t2",
      label: "AND Gate",
      path: ["Logic"],
      defaultData: {
        label: "AND",
        modes: [RenderMode.MODE_WIDGETS],
        activeMode: RenderMode.MODE_WIDGETS,
        widgets: [],
      },
    },
  ];

  it("should group templates by path and show submenus on hover", () => {
    const onAddNode = vi.fn();
    render(
      <ContextMenu
        x={0}
        y={0}
        onClose={() => {
          /* do nothing */
        }}
        onToggleTheme={() => {
          /* do nothing */
        }}
        templates={mockTemplates}
        onAddNode={onAddNode}
        onAutoLayout={() => {
          /* do nothing */
        }}
        isPaneMenu={true}
      />,
    );

    // Initial state: Categories should be visible
    expect(screen.getByText("Input")).toBeInTheDocument();
    expect(screen.getByText("Logic")).toBeInTheDocument();

    // Hover over "Input"
    fireEvent.mouseEnter(screen.getByText("Input"));

    // Nested category "Basic" should appear
    expect(screen.getByText("Basic")).toBeInTheDocument();

    // Hover over "Basic"
    fireEvent.mouseEnter(screen.getByText("Basic"));

    // Leaf node "+ Text Node" should appear
    const leaf = screen.getByText("+ Text Node");
    expect(leaf).toBeInTheDocument();

    // Click leaf node
    fireEvent.click(leaf);
    expect(onAddNode).toHaveBeenCalledWith(mockTemplates[0]);
  });
});
