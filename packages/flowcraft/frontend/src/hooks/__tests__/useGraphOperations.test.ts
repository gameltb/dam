import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, type Mock, vi } from "vitest";

import { useGraphOperations } from "@/hooks/graph/useGraphOperations";
import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";

const mockApplyMutations = vi.fn();
const mockCommit = vi.fn();
const mockSetClipboard = vi.fn();
const mockNodeDraft = vi.fn((n: any) => ({ ok: true, value: { ...n } }));

// Mock orchestrator
vi.mock("@/store/orchestrator", () => ({
  commit: vi.fn((recipe) => mockCommit(recipe)),
}));

// Mock the stores
const state = {
  activeGraphId: "default",
  applyMutations: mockApplyMutations,
  edges: [{ id: "e1-2", source: "1", target: "2" }],
  edgesById: { "e1-2": { id: "e1-2", source: "1", target: "2" } },
  nodeDraft: mockNodeDraft,
  nodes: [
    {
      graphId: "default",
      id: "1",
      measured: { height: 100, width: 200 },
      position: { x: 0, y: 0 },
    },
    {
      graphId: "default",
      id: "2",
      measured: { height: 100, width: 200 },
      position: { x: 0, y: 0 },
    },
  ],
  nodesById: {
    "1": { graphId: "default", id: "1", measured: { height: 100, width: 200 }, position: { x: 0, y: 0 } },
    "2": { graphId: "default", id: "2", measured: { height: 100, width: 200 }, position: { x: 0, y: 0 } },
  },
};

vi.mock("@xyflow/react", () => ({
  useReactFlow: vi.fn(() => ({
    getEdges: vi.fn(() => []),
    getNodes: vi.fn(() => []),
  })),
}));

vi.mock("@/store/flowStore", () => ({
  useFlowStore: Object.assign(
    vi.fn((selector) => (selector ? selector(state) : state)),
    { getState: () => state, setState: vi.fn() },
  ),
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

    const mockStore = useFlowStore as any;
    mockStore.mockImplementation((selector: any) => (selector ? selector(state) : state));
    mockStore.getState = () => state;

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

    expect(mockCommit).toHaveBeenCalled();
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

    expect(mockCommit).toHaveBeenCalled();
  });
});
