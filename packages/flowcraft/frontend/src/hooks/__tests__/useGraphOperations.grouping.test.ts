/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/no-unsafe-member-access */
import { describe, it, expect, vi, beforeEach, type Mock } from "vitest";
import { renderHook } from "@testing-library/react";
import { useGraphOperations } from "../useGraphOperations";
import { useFlowStore } from "../../store/flowStore";
import { useUiStore } from "../../store/uiStore";

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

    (useUiStore as unknown as { getState: () => any }).getState = () => ({
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
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const mutations = mockApplyMutations.mock.calls[0][0] as any[];

    // 1. Check if group node is added

    const addGroupMut = mutations.find(
      (m: any) => m.addNode?.node?.type === "groupNode",
    );
    expect(addGroupMut).toBeDefined();

    // Bounding Box: minX=100, minY=100, maxX=300, maxY=250 (padding=40)
    // groupX = 100 - 40 = 60, groupY = 100 - 40 = 60
    const groupPos = addGroupMut?.addNode?.node?.position;
    expect(groupPos).toEqual({ x: 60, y: 60 });

    // 2. Check if children are updated with parentId and relative positions
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const updateNode1 = mutations.find((m: any) => m.updateNode?.id === "1");
    expect(updateNode1?.updateNode?.parentId).toBeDefined();
    // relativeX = 100 - 60 = 40, relativeY = 100 - 60 = 40
    expect(updateNode1?.updateNode?.position).toEqual({ x: 40, y: 40 });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const updateNode2 = mutations.find((m: any) => m.updateNode?.id === "2");
    // relativeX = 200 - 60 = 140, relativeY = 200 - 60 = 140
    expect(updateNode2?.updateNode?.position).toEqual({ x: 140, y: 140 });
  });
});
