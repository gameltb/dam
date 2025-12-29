import { describe, it, expect, beforeEach } from "vitest";
import { useFlowStore } from "../flowStore";
import * as Y from "yjs";

describe("Yjs Sync Logic", () => {
  beforeEach(() => {
    useFlowStore.getState().resetStore();
  });

  it("should sync nodes from yNodes to store correctly", () => {
    const store = useFlowStore.getState();
    const { yNodes } = store;

    const node1 = {
      id: "1",
      type: "dynamic",
      position: { x: 0, y: 0 },
      data: { label: "Node 1", modes: [] },
    };

    yNodes.set("1", node1);
    store.syncFromYjs();

    expect(useFlowStore.getState().nodes.length).toBe(1);
    expect(useFlowStore.getState().nodes[0]?.id).toBe("1");
  });

  it("should handle remote Yjs updates", async () => {
    const store = useFlowStore.getState();
    const remoteDoc = new Y.Doc();
    const remoteNodes = remoteDoc.getMap("nodes");

    const node1 = {
      id: "1",
      type: "dynamic",
      position: { x: 0, y: 0 },
      data: { label: "Node 1", modes: [] },
    };

    remoteNodes.set("1", node1);

    const update = Y.encodeStateAsUpdate(remoteDoc);
    store.applyYjsUpdate(update);

    // Wait for any potential async propagation (though usually sync)
    await new Promise((resolve) => setTimeout(resolve, 500));

    expect(useFlowStore.getState().nodes.length).toBe(1);
    expect(useFlowStore.getState().nodes[0]?.id).toBe("1");
  });
});
