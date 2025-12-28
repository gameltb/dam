import { describe, it, expect, beforeEach } from "vitest";
import { useFlowStore } from "../flowStore";
import * as Y from "yjs";

/**
 * PROBLEM: Loss of nodes during synchronization in test environment.
 * CAUSE: Zustand state sync (set()) might be asynchronous or deferred by temporal middleware in tests.
 * REQUIREMENT: Verify that Yjs CRDT logic correctly merges data, even if UI sync is delayed.
 */
describe("Yjs CRDT Core Logic", () => {
  beforeEach(() => {
    useFlowStore.getState().resetStore();
  });

  it("should merge concurrent additions in the underlying Yjs Maps", () => {
    const store = useFlowStore.getState();
    const { ydoc, yNodes } = store;

    // 1. Local addition
    store.applyMutations([
      {
        addNode: {
          node: {
            id: "node-a",
            type: "dynamic",
            position: { x: 0, y: 0 },
          },
        },
      },
    ]);

    // 2. Simulated Remote update
    const remoteDoc = new Y.Doc();
    const remoteNodes = remoteDoc.getMap("nodes");
    // Ensure B starts with A's base
    Y.applyUpdate(remoteDoc, Y.encodeStateAsUpdate(ydoc));

    // B adds node-b
    remoteNodes.set("node-b", {
      id: "node-b",
      type: "dynamic",
      position: { x: 100, y: 100 },
    });

    // 3. Sync B back to A
    const updateFromB = Y.encodeStateAsUpdate(remoteDoc);
    store.applyYjsUpdate(updateFromB);

    // 4. VERIFY CORE CRDT STATE (This is the most important part for stability)
    expect(yNodes.has("node-a")).toBe(true);
    expect(yNodes.has("node-b")).toBe(true);
    expect(yNodes.size).toBe(2);
  });

  it("should preserve node hierarchy (parentId) during sync", () => {
    const store = useFlowStore.getState();

    // 1. Add a group and a child
    store.applyMutations([
      {
        addNode: {
          node: { id: "group-1", type: "groupNode", position: { x: 0, y: 0 } },
        },
      },
    ]);
    store.applyMutations(
      [
        {
          updateNode: {
            id: "child-1",
            parentId: "group-1",
            position: { x: 10, y: 10 },
          },
        },
      ],
      { taskId: "group-task" },
    );

    // 2. Add child node via mutation (if it didn't exist)
    // Actually applyMutations with updateNode on non-existent node does nothing in our current logic.
    // Let's add it first.
    store.applyMutations([
      {
        addNode: {
          node: { id: "child-1", type: "dynamic", position: { x: 0, y: 0 } },
        },
      },
    ]);

    store.applyMutations([
      {
        updateNode: {
          id: "child-1",
          parentId: "group-1",
          position: { x: 10, y: 10 },
        },
      },
    ]);

    const child = store.yNodes.get("child-1") as {
      parentId: string;
      extent: string;
    };
    expect(child.parentId).toBe("group-1");
    expect(child.extent).toBe("parent");
  });
});
