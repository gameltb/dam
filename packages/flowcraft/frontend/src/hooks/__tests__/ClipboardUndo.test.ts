import { describe, it, expect, beforeEach } from "vitest";
import { useFlowStore } from "../../store/flowStore";
import { useUiStore } from "../../store/uiStore";
import { create } from "@bufbuild/protobuf";
import {
  GraphMutationSchema,
  AddSubGraphSchema,
  AddNodeSchema,
} from "../../generated/flowcraft/v1/service_pb";
import { NodeSchema } from "../../generated/flowcraft/v1/node_pb";
import {
  PresentationSchema,
  NodeKind,
} from "../../generated/flowcraft/v1/base_pb";

describe("Direct Store Clipboard Logic", () => {
  beforeEach(() => {
    useFlowStore.getState().resetStore();
    useUiStore.getState().setClipboard(null);
  });

  it("should add subgraph and sync correctly", () => {
    const store = useFlowStore.getState();

    // 1. Manually apply mutation (bypassing Hook)
    store.applyMutations([
      create(GraphMutationSchema, {
        operation: {
          case: "addSubgraph",
          value: create(AddSubGraphSchema, {
            nodes: [
              create(NodeSchema, {
                nodeId: "test-node",
                nodeKind: NodeKind.DYNAMIC,
                presentation: create(PresentationSchema, {
                  position: { x: 0, y: 0 },
                  width: 0,
                  height: 0,
                  parentId: "",
                }),
                isSelected: false,
              }),
            ],
            edges: [],
          }),
        },
      }),
    ]);

    // 2. Immediate checks
    expect(store.yNodes.size).toBe(1);

    const updatedStore = useFlowStore.getState();
    expect(updatedStore.nodes.length).toBe(1);
    expect(updatedStore.nodes[0]?.id).toBe("test-node");
  });

  it("should undo addSubgraph correctly", () => {
    const store = useFlowStore.getState();
    const temporal = useFlowStore.temporal.getState();

    // 1. Add node
    store.applyMutations([
      create(GraphMutationSchema, {
        operation: {
          case: "addSubgraph",
          value: create(AddSubGraphSchema, {
            nodes: [
              create(NodeSchema, {
                nodeId: "undo-node",
                nodeKind: NodeKind.DYNAMIC,
                presentation: create(PresentationSchema, {
                  position: { x: 0, y: 0 },
                  width: 0,
                  height: 0,
                  parentId: "",
                }),
                isSelected: false,
              }),
            ],
            edges: [],
          }),
        },
      }),
    ]);

    expect(useFlowStore.getState().nodes.length).toBe(1);
    expect(useFlowStore.getState().yNodes.size).toBe(1);

    // 2. Undo
    temporal.undo();

    const afterUndoStore = useFlowStore.getState();
    expect(afterUndoStore.nodes.length).toBe(0);
    expect(afterUndoStore.yNodes.size).toBe(0);
  });

  it("should copy and paste correctly", () => {
    const store = useFlowStore.getState();

    // 1. Setup a selected node
    store.applyMutations([
      create(GraphMutationSchema, {
        operation: {
          case: "addNode",
          value: create(AddNodeSchema, {
            node: create(NodeSchema, {
              nodeId: "node-1",
              nodeKind: NodeKind.DYNAMIC,
              presentation: create(PresentationSchema, {
                position: { x: 100, y: 100 },
                width: 0,
                height: 0,
                parentId: "",
              }),
              isSelected: false,
            }),
          }),
        },
      }),
    ]);

    // Manually trigger selection change to sync it
    store.onNodesChange([{ id: "node-1", type: "select", selected: true }]);

    expect(useFlowStore.getState().nodes[0]?.selected).toBe(true);

    // 2. Perform Copy (simulating Hook logic)
    const selectedNodes = useFlowStore
      .getState()
      .nodes.filter((n) => n.selected);
    expect(selectedNodes.length).toBe(1);
    useUiStore.getState().setClipboard({ nodes: selectedNodes, edges: [] });

    // 3. Perform Paste
    const clipboard = useUiStore.getState().clipboard;
    expect(clipboard).not.toBeNull();

    const newNodeId = "node-2"; // In real hook it's a uuid
    store.applyMutations([
      create(GraphMutationSchema, {
        operation: {
          case: "addSubgraph",
          value: create(AddSubGraphSchema, {
            nodes: [
              create(NodeSchema, {
                nodeId: newNodeId,
                nodeKind: NodeKind.DYNAMIC,
                presentation: create(PresentationSchema, {
                  position: { x: 140, y: 140 },
                  width: 0,
                  height: 0,
                  parentId: "",
                }),
                isSelected: false,
              }),
            ],
            edges: [],
          }),
        },
      }),
    ]);

    const finalStore = useFlowStore.getState();
    expect(finalStore.nodes.length).toBe(2);
    expect(finalStore.yNodes.size).toBe(2);
    expect(finalStore.nodes.find((n) => n.id === "node-2")).toBeDefined();
  });
});
