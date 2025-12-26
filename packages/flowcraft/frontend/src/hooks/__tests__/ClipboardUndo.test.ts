import { describe, it, expect, beforeEach } from "vitest";
import { useFlowStore } from "../../store/flowStore";

describe("Direct Store Clipboard Logic", () => {
  beforeEach(() => {
    useFlowStore.getState().resetStore();
  });

  it("should add subgraph and sync correctly", () => {
    const store = useFlowStore.getState();

    // 1. Manually apply mutation (bypassing Hook)
    store.applyMutations([
      {
        addSubgraph: {
          nodes: [{ id: "test-node", type: "dynamic" }],
          edges: [],
        },
      },
    ]);

    // 2. Immediate checks
    expect(store.yNodes.size).toBe(1);

    const updatedStore = useFlowStore.getState();
    expect(updatedStore.nodes.length).toBe(1);
    expect(updatedStore.nodes[0].id).toBe("test-node");
  });

  it("should undo addSubgraph correctly", () => {
    const store = useFlowStore.getState();
    const temporal = useFlowStore.temporal.getState();

    // 1. Add node
    store.applyMutations([
      {
        addSubgraph: {
          nodes: [{ id: "undo-node", type: "dynamic" }],
          edges: [],
        },
      },
    ]);

    expect(useFlowStore.getState().nodes.length).toBe(1);
    expect(useFlowStore.getState().yNodes.size).toBe(1);

    // 2. Undo
    temporal.undo();

    const afterUndoStore = useFlowStore.getState();
    expect(afterUndoStore.nodes.length).toBe(0);
    // This is where it likely fails: yNodes is not managed by Zundo
    expect(afterUndoStore.yNodes.size).toBe(0);
  });

  it("should copy and paste correctly", () => {
    const store = useFlowStore.getState();

    // 1. Setup a selected node
    store.applyMutations([
      {
        addNode: {
          node: { id: "node-1", type: "dynamic", position: { x: 100, y: 100 } },
        },
      },
    ]);

    // Manually trigger selection change to sync it
    store.onNodesChange([{ id: "node-1", type: "select", selected: true }]);

    expect(useFlowStore.getState().nodes[0].selected).toBe(true);

    // 2. Perform Copy (simulating Hook logic)
    const selectedNodes = useFlowStore
      .getState()
      .nodes.filter((n) => n.selected);
    expect(selectedNodes.length).toBe(1);
    store.setClipboard({ nodes: selectedNodes, edges: [] });

    // 3. Perform Paste
    const clipboard = useFlowStore.getState().clipboard;
    expect(clipboard).not.toBeNull();

    const newNodeId = "node-2"; // In real hook it's a uuid
    store.applyMutations([
      {
        addSubgraph: {
          nodes: [
            {
              ...clipboard!.nodes[0],
              id: newNodeId,
              position: { x: 140, y: 140 },
            },
          ],
          edges: [],
        },
      },
    ]);

    const finalStore = useFlowStore.getState();
    expect(finalStore.nodes.length).toBe(2);
    expect(finalStore.yNodes.size).toBe(2);
    expect(finalStore.nodes.find((n) => n.id === "node-2")).toBeDefined();
  });
});
