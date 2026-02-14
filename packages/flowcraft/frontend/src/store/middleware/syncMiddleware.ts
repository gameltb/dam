import { type Edge } from "@xyflow/react";
import { type Patch } from "immer";

import { type AppNode, type DynamicNodeData, type MutationContext } from "@/types";
import { log } from "@/utils/logger";
import { type PbConnection } from "@/utils/pb-client";

import { useFlowStore } from "../flowStore";
import { type RFState } from "../types";
import { type GraphMiddleware, MutationDirection } from "./types";

/**
 * SyncDescriptor
 * Defines how a specific slice of the store maps to a remote Reducer.
 */
interface SyncDescriptor {
  execute: (id: string, patch: Patch, store: RFState, conn: PbConnection, context: MutationContext) => void;
  matches: (path: (number | string)[]) => boolean;
}

/**
 * SYNC_REGISTRY
 * Optimized declarative rules for bidirectional synchronization.
 */
const SYNC_REGISTRY: SyncDescriptor[] = [
  // 1. Viewport Layer (Immediate)
  {
    execute: (_, __, store, conn) => {
      const { activeScopeId, viewport } = store;
      const scopeId = activeScopeId || "root";
      conn.pbreducers.updateViewport({
        id: scopeId,
        viewport: {
          x: viewport.x,
          y: viewport.y,
          zoom: viewport.zoom,
        },
      });
    },
    matches: (path) => path[0] === "viewport",
  },
  // 2. Node Transform Layer (DEFERRED until interaction end)
  {
    execute: (id, _, store, conn, context) => {
      // CRITICAL: Only sync to DB if the interaction has finished
      if (!context.isInteractionEnd) return;

      const node = store.nodesById[id];
      if (!node) throw new Error(`[Sync] Node ${id} not found for transform`);

      log.sync("OUT", `Committing transform for node ${id}`);
      conn.reducers.setNodePosition({ nodeId: id, x: node.position.x, y: node.position.y });
      conn.reducers.setNodeSize({ height: node.height, nodeId: id, width: node.width });
    },
    matches: (path) =>
      path[0] === "nodesById" &&
      (["height", "position", "width"].includes(path[2] as string) ||
        (path[2] === "presentation" && ["height", "position", "width"].includes(path[3] as string))),
  },
  // 3. Node Hierarchy Layer (Immediate for direct calls, Deferred for drag-drops)
  {
    execute: (id, _, store, conn, context) => {
      // If it's part of a node change event, wait for the drop
      if (context.description?.includes("Resizing") || context.description?.includes("Dragging")) {
        if (!context.isInteractionEnd) return;
      }

      const node = store.nodesById[id];
      if (!node) throw new Error(`[Sync] Node ${id} not found for hierarchy`);
      conn.reducers.setNodeParent({ nodeId: id, parentId: node.parentId ?? "" });
    },
    matches: (path) =>
      path[0] === "nodesById" && (path[2] === "parentId" || (path[2] === "presentation" && path[3] === "parentId")),
  },
  // 4. Node Domain Data Layer (Immediate)
  {
    execute: (id, _, store, conn) => {
      const node = store.nodesById[id];
      if (!node) throw new Error(`[Sync] Node ${id} not found for data`);
      conn.pbreducers.setNodeDataPb({ nodeId: id, state: node.data });
    },
    matches: (path) => path[0] === "nodesById" && path[2] === "data",
  },
  // 5. Node Structure Layer (Immediate)
  {
    execute: (id, patch, _, conn) => {
      if (patch.op === "add") {
        const node = patch.value as AppNode;
        const data = node.data as DynamicNodeData;
        const templateId = data.templateId;

        if (!node) throw new Error("[Sync] Missing node value");
        if (!templateId)
          throw new Error(`[Sync] Missing templateId for new node ${id}. Aborting to prevent DB corruption.`);

        conn.pbreducers.createNodePb({
          node: {
            nodeId: id,
            nodeKind: 1,
            presentation: {
              height: node.height,
              isHidden: false,
              isInitialized: true,
              isLocked: false,
              isSelected: false,
              parentId: node.parentId ?? "",
              position: { x: node.position.x, y: node.position.y },
              scopeId: node.scopeId || "root",
              width: node.width,
              zIndex: 0,
            },
            state: node.data,
            templateId: templateId,
          },
        });
      } else if (patch.op === "remove") {
        conn.reducers.removeNode({ id });
      }
    },
    matches: (path) => path[0] === "nodesById" && path.length === 2,
  },
  // 6. Graph Topology Layer (Immediate)
  {
    execute: (id, patch, _, conn) => {
      if (patch.op === "add" && patch.value && typeof patch.value === "object" && "source" in patch.value) {
        conn.pbreducers.addEdgePb({ edge: patch.value as Edge });
      } else if (patch.op === "remove") {
        conn.reducers.removeEdge({ id });
      }
    },
    matches: (path) => path[0] === "edgesById",
  },
];

export const syncMiddleware: GraphMiddleware = (event, next) => {
  const { context, direction, patches } = event;

  if (direction !== MutationDirection.OUTGOING) {
    next(event);
    return;
  }

  const store = useFlowStore.getState();
  const conn = store.spacetimeConn;

  if (!conn) {
    next(event);
    return;
  }

  const executed = new Set<string>();

  patches.forEach((patch) => {
    const id = patch.path[1] as string;

    for (const descriptor of SYNC_REGISTRY) {
      if (descriptor.matches(patch.path)) {
        const key = `${SYNC_REGISTRY.indexOf(descriptor)}-${id}`;

        if (!executed.has(key)) {
          // LOG: Outgoing sync
          const pathStr = patch.path.join(".");
          log.sync("OUT", `Syncing ${pathStr}`, {
            context,
            entityId: id,
            op: patch.op,
            path: patch.path,
            value: patch.value,
          });

          descriptor.execute(id, patch, store, conn, context);

          executed.add(key);
        }

        break;
      }
    }
  });

  next(event);
};
