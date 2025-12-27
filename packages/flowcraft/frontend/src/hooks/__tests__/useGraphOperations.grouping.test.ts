import { describe, it, expect, vi, beforeEach, type Mock } from "vitest";
import { renderHook } from "@testing-library/react";
import { useGraphOperations } from "../useGraphOperations";
import { useFlowStore } from "../../store/flowStore";
import { flowcraft_proto } from "../../generated/flowcraft_proto";

// Mock the store
vi.mock("../../store/flowStore", () => ({
  useFlowStore: vi.fn(),
}));

/**
 * PROBLEM: Users couldn't group selected nodes via context menu.
 * REQUIREMENT: Implement groupSelected to calculate bounding box, create a group node, and reparent selected nodes.
 */
describe("useGraphOperations - Grouping", () => {
  const mockApplyMutations = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should create a group node and reparent selected nodes", () => {
    (useFlowStore as unknown as Mock).mockReturnValue({
      nodes: [
        {
          id: "node-1",
          position: { x: 100, y: 100 },
          measured: { width: 100, height: 50 },
          selected: true,
        },
        {
          id: "node-2",
          position: { x: 300, y: 200 },
          measured: { width: 100, height: 50 },
          selected: true,
        },
        {
          id: "node-3",
          position: { x: 500, y: 500 },
          selected: false, // Not selected, should not be grouped
        },
      ],
      applyMutations: mockApplyMutations,
    });

    const { result } = renderHook(() =>
      useGraphOperations({ clientVersion: 1 }),
    );

    result.current.groupSelected();

    expect(mockApplyMutations).toHaveBeenCalled();

    const mutations: flowcraft_proto.v1.IGraphMutation[] = (
      mockApplyMutations as vi.Mock
    ).mock.calls[0][0];

    // Should have 3 mutations: 1 addNode (group) and 2 updateNode (reparenting)

    expect(mutations.length).toBe(3);

    const addGroupMutation = mutations.find((m) => m.addNode);

    expect(addGroupMutation).toBeDefined();

    const groupNode = addGroupMutation?.addNode?.node;

    expect(groupNode).toBeDefined();

    if (!groupNode) return;

    expect(groupNode.type).toBe("groupNode");

    // Bounding box calculation:

    // node-1: (100, 100) to (200, 150)

    // node-2: (300, 200) to (400, 250)

    // minX=100, minY=100, maxX=400, maxY=250

    // padding=40 -> groupX=60, groupY=60, groupW=300+80=380, groupH=150+80=230

    expect(groupNode.position?.x).toBe(60);

    expect(groupNode.position?.y).toBe(60);

    expect(groupNode.width).toBe(380);

    expect(groupNode.height).toBe(230);

    const reparent1 = mutations.find((m) => m.updateNode?.id === "node-1");

    expect(reparent1?.updateNode?.parentId).toBe(groupNode.id);

    // Relative position: 100 - 60 = 40

    expect(reparent1?.updateNode?.position?.x).toBe(40);

    expect(reparent1?.updateNode?.position?.y).toBe(40);

    const reparent2 = mutations.find((m) => m.updateNode?.id === "node-2");

    expect(reparent2?.updateNode?.parentId).toBe(groupNode.id);

    // Relative position: 300 - 60 = 240, 200 - 60 = 140

    expect(reparent2?.updateNode?.position?.x).toBe(240);

    expect(reparent2?.updateNode?.position?.y).toBe(140);
  });

  it("should not group if less than 2 nodes are selected", () => {
    (useFlowStore as unknown as Mock).mockReturnValue({
      nodes: [
        {
          id: "node-1",
          position: { x: 100, y: 100 },
          selected: true,
        },
      ],
      applyMutations: mockApplyMutations,
    });

    const { result } = renderHook(() =>
      useGraphOperations({ clientVersion: 1 }),
    );

    result.current.groupSelected();

    expect(mockApplyMutations).not.toHaveBeenCalled();
  });
});
