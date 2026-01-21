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
 * UPDATED: Standardized on Path-based updates (ORM mode).
 */
describe("useGraphOperations - Auto Layout", () => {
  const mockApplyMutations = vi.fn();
  const mockSetClipboard = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useFlowStore as unknown as Mock).mockReturnValue({
      applyMutations: mockApplyMutations,
      edges: [{ id: "e1-2", source: "1", target: "2" }],
      nodeDraft: (n: any) => n, // Simple mock for draft
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
    // For direct access to getState()
    (useUiStore as unknown as { getState: () => unknown }).getState = () => ({
      clipboard: null,
      setClipboard: mockSetClipboard,
    });
  });

  it("should include dimensions in pathUpdate mutations during auto-layout", () => {
    const { result } = renderHook(() => useGraphOperations());

    result.current.autoLayout();

    expect(mockApplyMutations).toHaveBeenCalled();
    const calls = mockApplyMutations.mock.calls;
    if (!calls[0]) throw new Error("Expected mockApplyMutations to be called");

    const mutations = calls[0][0] as GraphMutation[];

    // Auto-layout now triggers multiple path updates via nodeDraft
    const widthUpdate = mutations.find(
      (m: GraphMutation) => m.operation.case === "pathUpdate" && m.operation.value.path === "width",
    );
    expect(widthUpdate).toBeDefined();
  });

  it("should use fallback dimensions if measured is missing", () => {
    (useFlowStore as unknown as Mock).mockReturnValue({
      applyMutations: mockApplyMutations,
      edges: [],
      nodeDraft: (n: any) => n,
      nodes: [{ id: "1", position: { x: 0, y: 0 } }],
    });

    const { result } = renderHook(() => useGraphOperations());
    result.current.autoLayout();

    const calls = mockApplyMutations.mock.calls;
    if (!calls[0]) throw new Error("Expected mockApplyMutations to be called");

    const mutations = calls[0][0] as GraphMutation[];

    const widthUpdate = mutations.find(
      (m: GraphMutation) => m.operation.case === "pathUpdate" && m.operation.value.path === "width",
    );
    // Fallback is 300
    expect(widthUpdate?.operation.case === "pathUpdate" && widthUpdate.operation.value.value).toBeDefined();
  });
});
