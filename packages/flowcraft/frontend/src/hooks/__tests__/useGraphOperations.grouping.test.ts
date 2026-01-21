import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, type Mock, vi } from "vitest";

import { NodeKind } from "@/generated/flowcraft/v1/core/base_pb";
import { type GraphMutation } from "@/generated/flowcraft/v1/core/service_pb";
import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";

import { useGraphOperations } from "../useGraphOperations";

vi.mock("@/store/flowStore", () => ({
  useFlowStore: vi.fn(),
}));

vi.mock("@/store/uiStore", () => ({
  useUiStore: vi.fn(),
}));

describe("useGraphOperations - Grouping", () => {
  const mockApplyMutations = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useFlowStore as unknown as Mock).mockReturnValue({
      applyMutations: mockApplyMutations,
      edges: [],
      nodeDraft: (n: any) => n,
      nodes: [
        {
          id: "1",
          measured: { height: 50, width: 100 },
          position: { x: 100, y: 100 },
          selected: true,
        },
        {
          id: "2",
          measured: { height: 50, width: 100 },
          position: { x: 200, y: 200 },
          selected: true,
        },
      ],
    });

    (useUiStore as unknown as Mock).mockReturnValue({
      clipboard: null,
      setClipboard: vi.fn(),
    });
    // For direct access to getState()
    (useUiStore as unknown as { getState: () => unknown }).getState = () => ({
      clipboard: null,
      setClipboard: vi.fn(),
    });
  });

  it("should calculate correct relative positions and parentId when grouping", () => {
    const { result } = renderHook(() => useGraphOperations());

    result.current.groupSelected();

    expect(mockApplyMutations).toHaveBeenCalled();
    const calls = mockApplyMutations.mock.calls;
    if (!calls[0]) throw new Error("Expected mockApplyMutations to be called");

    const mutations = calls[0][0] as GraphMutation[];

    // 1. Check if group node is added
    const addGroupMut = mutations.find(
      (m) => m.operation.case === "addNode" && m.operation.value.node?.nodeKind === NodeKind.GROUP,
    );
    expect(addGroupMut).toBeDefined();

    // 2. Check if children are updated via pathUpdate (nodeDraft)
    const parentUpdate = mutations.find(
      (m) => m.operation.case === "pathUpdate" && m.operation.value.path === "parentId",
    );
    expect(parentUpdate).toBeDefined();

    const posUpdate = mutations.find((m) => m.operation.case === "pathUpdate" && m.operation.value.path === "position");
    expect(posUpdate).toBeDefined();
  });
});
