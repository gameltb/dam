import { describe, it, expect, vi, beforeEach, type Mock } from "vitest";
import { renderHook } from "@testing-library/react";
import { useGraphOperations } from "../useGraphOperations";
import { useFlowStore } from "../../store/flowStore";
import { useUiStore } from "../../store/uiStore";
import { type GraphMutation } from "../../generated/flowcraft/v1/core/service_pb";
import { NodeKind } from "../../generated/flowcraft/v1/core/base_pb";

vi.mock("../../store/flowStore", () => ({
  useFlowStore: vi.fn(),
}));

vi.mock("../../store/uiStore", () => ({
  useUiStore: vi.fn(),
}));

describe("useGraphOperations - Grouping", () => {
  const mockApplyMutations = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useFlowStore as unknown as Mock).mockReturnValue({
      nodes: [
        {
          id: "1",
          position: { x: 100, y: 100 },
          selected: true,
          measured: { width: 100, height: 50 },
        },
        {
          id: "2",
          position: { x: 200, y: 200 },
          selected: true,
          measured: { width: 100, height: 50 },
        },
      ],
      edges: [],
      applyMutations: mockApplyMutations,
    });

    (useUiStore as unknown as Mock).mockReturnValue({
      clipboard: null,
      setClipboard: vi.fn(),
    });
    // For direct access to getState() in useClipboard (if used)
    (useUiStore as unknown as { getState: () => unknown }).getState = () => ({
      clipboard: null,
      setClipboard: vi.fn(),
    });
  });

  it("should calculate correct relative positions and parentId when grouping", () => {
    const { result } = renderHook(() =>
      useGraphOperations({ clientVersion: 1 }),
    );

    result.current.groupSelected();

    expect(mockApplyMutations).toHaveBeenCalled();
    const calls = mockApplyMutations.mock.calls;
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
    const mutations = (calls[0]?.[0] as GraphMutation[]) ?? [];

    // 1. Check if group node is added

    const addGroupMut = mutations.find(
      (m) =>
        m.operation.case === "addNode" &&
        m.operation.value.node?.nodeKind === NodeKind.GROUP,
    );
    expect(addGroupMut).toBeDefined();

    // Bounding Box: minX=100, minY=100, maxX=300, maxY=250 (padding=40)
    // groupX = 100 - 40 = 60, groupY = 100 - 40 = 60
    const groupPos =
      addGroupMut?.operation.case === "addNode"
        ? addGroupMut.operation.value.node?.presentation?.position
        : null;
    expect(groupPos).toMatchObject({ x: 60, y: 60 });

    // 2. Check if children are updated with parentId and relative positions
    const updateNode1 = mutations.find(
      (m) => m.operation.case === "updateNode" && m.operation.value.id === "1",
    );
    if (updateNode1?.operation.case === "updateNode") {
      expect(updateNode1.operation.value.presentation?.parentId).toBeDefined();
      // relativeX = 100 - 60 = 40, relativeY = 100 - 60 = 40
      expect(updateNode1.operation.value.presentation?.position).toMatchObject({
        x: 40,
        y: 40,
      });
    } else {
      throw new Error("Expected updateNode mutation");
    }

    const updateNode2 = mutations.find(
      (m) => m.operation.case === "updateNode" && m.operation.value.id === "2",
    );
    if (updateNode2?.operation.case === "updateNode") {
      // relativeX = 200 - 60 = 140, relativeY = 200 - 60 = 140
      expect(updateNode2.operation.value.presentation?.position).toMatchObject({
        x: 140,
        y: 140,
      });
    } else {
      throw new Error("Expected updateNode mutation");
    }
  });
});
