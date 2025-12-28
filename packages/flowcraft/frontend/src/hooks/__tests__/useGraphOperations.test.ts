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
 * PROBLEM: Nodes overlapped during auto-layout.
 * CAUSE: Layout algorithm used 0 dimensions for unmeasured nodes and didn't persist dimensions in mutations.
 * REQUIREMENT: Provide fallback dimensions (300x200) and persist used dimensions in the updateNode mutation.
 */
describe("useGraphOperations - Auto Layout", () => {
  const mockApplyMutations = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useFlowStore as Mock).mockReturnValue({
      nodes: [
        {
          id: "1",
          position: { x: 0, y: 0 },
          measured: { width: 200, height: 100 },
        },
        {
          id: "2",
          position: { x: 0, y: 0 },
          measured: { width: 200, height: 100 },
        },
      ],
      edges: [{ id: "e1-2", source: "1", target: "2" }],
      applyMutations: mockApplyMutations,
      setNodes: vi.fn(),
      setClipboard: vi.fn(),
    });
  });

  it("should include dimensions in updateNode mutations during auto-layout", () => {
    const { result } = renderHook(() =>
      useGraphOperations({ clientVersion: 1 }),
    );

    result.current.autoLayout();

    expect(mockApplyMutations).toHaveBeenCalled();
    const mutations = mockApplyMutations.mock
      .calls[0][0] as flowcraft_proto.v1.IGraphMutation[];

    const firstUpdate = mutations.find(
      (m: flowcraft_proto.v1.IGraphMutation) => m.updateNode?.id === "1",
    );
    expect(firstUpdate?.updateNode).toHaveProperty("width", 200);
    expect(firstUpdate?.updateNode).toHaveProperty("height", 100);
  });

  it("should use fallback dimensions if measured is missing", () => {
    (useFlowStore as Mock).mockReturnValue({
      nodes: [{ id: "1", position: { x: 0, y: 0 } }],
      edges: [],
      applyMutations: mockApplyMutations,
      setNodes: vi.fn(),
      setClipboard: vi.fn(),
    });

    const { result } = renderHook(() =>
      useGraphOperations({ clientVersion: 1 }),
    );
    result.current.autoLayout();

    const mutations = mockApplyMutations.mock
      .calls[0][0] as flowcraft_proto.v1.IGraphMutation[];
    const update = mutations[0]?.updateNode;
    expect(update?.width).toBe(300);
    expect(update?.height).toBe(200);
  });
});
