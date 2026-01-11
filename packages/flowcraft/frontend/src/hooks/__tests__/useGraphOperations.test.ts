import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, type Mock, vi } from "vitest";

import { type GraphMutation } from "@/generated/flowcraft/v1/core/service_pb";
import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";

import { useGraphOperations } from "../useGraphOperations";

// Mock the stores
vi.mock("@/store/flowStore", () => ({
  useFlowStore: vi.fn(),
}));

vi.mock("@/store/uiStore", () => ({
  useUiStore: vi.fn(),
}));

/**
 * PROBLEM: Nodes overlapped during auto-layout.
 * CAUSE: Layout algorithm used 0 dimensions for unmeasured nodes and didn't persist dimensions in mutations.
 * REQUIREMENT: Provide fallback dimensions (300x200) and persist used dimensions in the updateNode mutation.
 */
describe("useGraphOperations - Auto Layout", () => {
  const mockApplyMutations = vi.fn();
  const mockSetClipboard = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useFlowStore as unknown as Mock).mockReturnValue({
      applyMutations: mockApplyMutations,
      edges: [{ id: "e1-2", source: "1", target: "2" }],
      nodes: [
        {
          id: "1",
          measured: { height: 100, width: 200 },
          position: { x: 0, y: 0 },
        },
        {
          id: "2",
          measured: { height: 100, width: 200 },
          position: { x: 0, y: 0 },
        },
      ],
    });

    (useUiStore as unknown as Mock).mockReturnValue({
      clipboard: null,
      setClipboard: mockSetClipboard,
    });
    // For direct access to getState() in useGraphOperations
    (useUiStore as unknown as { getState: () => unknown }).getState = () => ({
      clipboard: null,
      setClipboard: mockSetClipboard,
    });
  });

  it("should include dimensions in updateNode mutations during auto-layout", () => {
    const { result } = renderHook(() => useGraphOperations());

    result.current.autoLayout();

    expect(mockApplyMutations).toHaveBeenCalled();
    const calls = mockApplyMutations.mock.calls;
     
    const mutations = (calls[0]?.[0] as GraphMutation[]) ?? [];

    const firstUpdate = mutations.find(
      (m: GraphMutation) =>
        m.operation.case === "updateNode" && m.operation.value.id === "1",
    );
    const opVal =
      firstUpdate?.operation.case === "updateNode"
        ? firstUpdate.operation.value
        : null;
    expect(opVal?.presentation).toHaveProperty("width", 200);
    expect(opVal?.presentation).toHaveProperty("height", 100);
  });

  it("should use fallback dimensions if measured is missing", () => {
    (useFlowStore as unknown as Mock).mockReturnValue({
      applyMutations: mockApplyMutations,
      edges: [],
      nodes: [{ id: "1", position: { x: 0, y: 0 } }],
    });

    const { result } = renderHook(() => useGraphOperations());
    result.current.autoLayout();

    const calls = mockApplyMutations.mock.calls;
     
    const mutations = (calls[0]?.[0] as GraphMutation[]) ?? [];
    const op = mutations[0]?.operation;
    if (op?.case === "updateNode") {
      expect(op.value.presentation?.width).toBe(300);
      expect(op.value.presentation?.height).toBe(200);
    } else {
      throw new Error("Expected updateNode mutation");
    }
  });
});
