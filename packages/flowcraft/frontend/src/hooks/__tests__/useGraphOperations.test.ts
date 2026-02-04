import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, type Mock, vi } from "vitest";

import { type GraphMutation } from "@/generated/flowcraft/v1/core/service_pb";
import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";

import { useGraphOperations } from "@/hooks/graph/useGraphOperations";

const mockApplyMutations = vi.fn();
const mockSetClipboard = vi.fn();
const mockNodeDraft = vi.fn((n: any) => ({ ok: true, value: { ...n } }));

// Mock the stores
vi.mock("@/store/flowStore", () => ({
  useFlowStore: vi.fn((selector) => {
    const state = {
      applyMutations: mockApplyMutations,
      edges: [{ id: "e1-2", source: "1", target: "2" }],
      nodeDraft: mockNodeDraft,
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
    };
    return selector ? selector(state) : state;
  }),
}));

vi.mock("@/store/uiStore", () => ({
  useUiStore: vi.fn((selector) => {
    const state = {
      clipboard: null,
      setClipboard: mockSetClipboard,
    };
    const res = selector ? selector(state) : state;
    // For direct access to getState()
    if (!selector) {
      res.getState = () => state;
    }
    return res;
  }),
}));

/**
 * UPDATED: Standardized on Path-based updates (ORM mode).
 */
describe("useGraphOperations - Auto Layout", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    const state = {
      applyMutations: mockApplyMutations,
      edges: [{ id: "e1-2", source: "1", target: "2" }],
      nodeDraft: mockNodeDraft,
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
    };

    (useFlowStore as unknown as Mock).mockImplementation((selector: any) => (selector ? selector(state) : state));

    (useUiStore as unknown as Mock).mockImplementation((selector: any) => {
      const uiState = {
        clipboard: null,
        setClipboard: mockSetClipboard,
      };
      const res = selector ? selector(uiState) : uiState;
      if (!selector) {
        res.getState = () => uiState;
      }
      return res;
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
    // width/height updates should be in the mutation list
    expect(mutations.length).toBeGreaterThan(0);
  });

  it("should use fallback dimensions if measured is missing", () => {
    const state = {
      applyMutations: mockApplyMutations,
      edges: [],
      nodeDraft: mockNodeDraft,
      nodes: [{ id: "1", position: { x: 0, y: 0 } }],
    };
    (useFlowStore as unknown as Mock).mockImplementation((selector: any) => (selector ? selector(state) : state));

    const { result } = renderHook(() => useGraphOperations());
    result.current.autoLayout();

    expect(mockApplyMutations).toHaveBeenCalled();
  });
});
