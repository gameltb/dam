import { describe, it, expect, vi, beforeEach, type Mock } from "vitest";
import { renderHook } from "@testing-library/react";
import { useGraphOperations } from "../useGraphOperations";
import { useFlowStore } from "../../store/flowStore";
import { useUiStore } from "../../store/uiStore";
import { type GraphMutation } from "../../generated/flowcraft/v1/service_pb";

// Mock the stores
vi.mock("../../store/flowStore", () => ({
  useFlowStore: vi.fn(),
}));

vi.mock("../../store/uiStore", () => ({
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
    });

    (useUiStore as unknown as Mock).mockReturnValue({
      setClipboard: mockSetClipboard,
      clipboard: null,
    });
    // For direct access to getState() in useGraphOperations
    (useUiStore as unknown as { getState: () => unknown }).getState = () => ({
      setClipboard: mockSetClipboard,
      clipboard: null,
    });
  });

  it("should include dimensions in updateNode mutations during auto-layout", () => {
    const { result } = renderHook(() =>
      useGraphOperations({ clientVersion: 1 }),
    );

    result.current.autoLayout();

    expect(mockApplyMutations).toHaveBeenCalled();
    const calls = mockApplyMutations.mock.calls;
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
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
      nodes: [{ id: "1", position: { x: 0, y: 0 } }],
      edges: [],
      applyMutations: mockApplyMutations,
    });

    const { result } = renderHook(() =>
      useGraphOperations({ clientVersion: 1 }),
    );
    result.current.autoLayout();

    const calls = mockApplyMutations.mock.calls;
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
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
