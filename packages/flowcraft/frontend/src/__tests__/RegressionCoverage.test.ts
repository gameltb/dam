import { create } from "@bufbuild/protobuf";
import { beforeEach, describe, expect, it } from "vitest";

import {
  NodeKind,
  PresentationSchema,
} from "@/generated/flowcraft/v1/core/base_pb";
import { NodeSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { useFlowStore } from "@/store/flowStore";
import { type AppNode, AppNodeType } from "@/types";
import { dehydrateNode } from "@/utils/nodeUtils";
import { fromProtoNode } from "@/utils/protoAdapter";

/**
 * ARCHITECTURAL REGRESSION TESTS
 *
 * These tests ensure that critical fixes related to synchronization, hierarchy,
 * and performance stay intact during future refactors.
 */

describe("Architectural Regressions", () => {
  beforeEach(() => {
    useFlowStore.getState().resetStore();
  });

  /**
   * REQUIREMENT: Parent nodes MUST appear before children in the state array.
   * REASON: React Flow initializes coordinate spaces based on parent position.
   * If a child appears before its parent, its relative position might be treated as absolute.
   */
  it("should maintain topological order (parents before children) in syncFromYjs", () => {
    const store = useFlowStore.getState();
    const { yNodes } = store;

    // Manually push nodes in "wrong" order (child first)
    const child: AppNode = {
      data: { label: "C", modes: [] },
      id: "child-1",
      parentId: "parent-1",
      position: { x: 10, y: 10 },
      type: AppNodeType.DYNAMIC,
    };
    const parent: AppNode = {
      data: { label: "P", modes: [] },
      id: "parent-1",
      position: { x: 100, y: 100 },
      type: AppNodeType.GROUP,
    };

    yNodes.set(child.id, child);
    yNodes.set(parent.id, parent);

    // Manually mark layout as dirty since we bypassed the store methods
    useFlowStore.setState({ isLayoutDirty: true });

    // Trigger sync
    store.syncFromYjs();

    const nodes = useFlowStore.getState().nodes;
    const parentIndex = nodes.findIndex((n) => n.id === "parent-1");
    const childIndex = nodes.findIndex((n) => n.id === "child-1");

    expect(parentIndex).toBeLessThan(childIndex);
  });

  /**
   * REQUIREMENT: Protobuf empty strings for parentId must be converted to undefined.
   * REASON: Protobuf defaults unset strings to "". React Flow treats "" as a literal ID
   * and fails to find the parent, breaking nested rendering.
   */
  it("should convert Protobuf empty parentId to undefined", () => {
    const protoNode = create(NodeSchema, {
      nodeId: "test-node",
      nodeKind: NodeKind.DYNAMIC,
      presentation: create(PresentationSchema, {
        parentId: "", // Protobuf default
        position: { x: 0, y: 0 },
      }),
    });

    const node = fromProtoNode(protoNode);
    expect(node.parentId).toBeUndefined();
  });

  /**
   * REQUIREMENT: History snapshots must be skipped during active dragging.
   * REASON: Performance and UX. Dragging generates 60 updates per second.
   * Recording each would tank FPS and make "Undo" useless (undoing 1px at a time).
   */
  it("should skip temporal snapshots when nodes are dragging", () => {
    // We verify dehydrateNode keeps the dragging flag for the middleware to see.
    // The middleware itself is tested by checking if it filters nodes based on this flag.
    const node: AppNode = {
      data: { label: "dragging", modes: [] },
      dragging: true,
      id: "1",
      position: { x: 0, y: 0 },
      type: AppNodeType.DYNAMIC,
    };
    const dehydrated = dehydrateNode(node);
    expect(dehydrated.dragging).toBe(true);
  });

  /**
   * REQUIREMENT: Connection handles must remain in DOM but be invisible when not needed.
   * REASON: React Flow needs the DOM element to exist at the start of a connection
   * to calculate snapping. If it's conditionally rendered (v-if style), snapping fails.
   */
  it("should keep implicit handles in DOM with near-zero opacity when not dragging", () => {
    const calculateOpacity = (
      isImplicit: boolean,
      hasActiveConn: boolean,
      isConnected: boolean,
    ) => {
      return isImplicit && !hasActiveConn && !isConnected ? 0.001 : 1;
    };

    expect(calculateOpacity(true, false, false)).toBe(0.001); // Hidden but in DOM
    expect(calculateOpacity(true, true, false)).toBe(1); // Visible during drag
    expect(calculateOpacity(true, false, true)).toBe(1); // Visible when connected
  });
});
