import { create as createProto, fromJsonString, toJsonString } from "@bufbuild/protobuf";
import { type Infer } from "spacetimedb";

import { ChatMessagePartSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { PositionSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { type NodeDataRow, type NodeTransformsRow, type NodesRow } from "@/generated/spacetime";
import { useFlowStore } from "@/store/flowStore";
import { type DynamicNodeData, Scope, isChatNode } from "@/types";
import { GraphMapper } from "@/utils/graphMapper";
import { type SyncedLens } from "@/utils/lens-types";
import { partsToText } from "@/components/media/chat/utils";
import { type PbConnection } from "@/utils/pb-client";
import { log } from "@/utils/logger";

/**
 * NodeMaterializer (V2.1 - Correct Listener Management)
 */
export const nodeMaterializer = {
  name: "node",

  setup: (conn: PbConnection, activeScopeId: string | null) => {
    // 1. Define Listeners
    const onInsert = (_ctx: any, row: Infer<typeof NodesRow>) => {
      log.sync("IN", `Node Insert: ${row.nodeId}`, { row });
      useFlowStore.setState((s) => ({
        nodesById: { ...s.nodesById, [row.nodeId]: GraphMapper.createSkeleton(row) },
      }));
    };

    const onUpdateTransform = (_ctx: any, _oldRow: any, row: Infer<typeof NodeTransformsRow>) => {
      log.sync("IN", `Transform Update: ${row.nodeId}`, { row });
      useFlowStore.setState((s) => {
        const n = s.nodesById[row.nodeId];
        if (!n || n.dragging) return s;
        return { nodesById: { ...s.nodesById, [row.nodeId]: GraphMapper.applyTransform(n, row) } };
      });
    };

    const onInsertTransform = (_ctx: any, row: Infer<typeof NodeTransformsRow>) => {
      // Re-use update logic for insert, as it just applies the transform
      onUpdateTransform(_ctx, null, row);
    };

    const onUpdateData = (_ctx: any, _oldRow: any, row: Infer<typeof NodeDataRow>) => {
      log.sync("IN", `Data Update: ${row.nodeId}`, { row });
      useFlowStore.setState((s) => {
        const n = s.nodesById[row.nodeId];
        if (!n) return s;
        return { nodesById: { ...s.nodesById, [row.nodeId]: GraphMapper.applyData(n, row) } };
      });
    };

    const onInsertData = (_ctx: any, row: Infer<typeof NodeDataRow>) => {
      // Re-use update logic for insert
      onUpdateData(_ctx, null, row);
    };

    // 2. Register Listeners
    conn.db.nodes.onInsert(onInsert);
    conn.db.nodeTransforms.onUpdate(onUpdateTransform);
    conn.db.nodeTransforms.onInsert(onInsertTransform);
    conn.db.nodeData.onUpdate(onUpdateData);
    conn.db.nodeData.onInsert(onInsertData);

    // 3. Selective Subscriptions
    const scopeFilter = activeScopeId ? `'${activeScopeId}'` : `'${Scope.ROOT}'`;
    const sub = conn
      .subscriptionBuilder()
      .onApplied(() => {
        log.sync("IN", "Subscription Applied");
        useFlowStore.getState().refreshView();
      })
      .subscribe([
        `SELECT * FROM nodes WHERE nodeId IN (SELECT nodeId FROM node_metadata WHERE scopeId = ${scopeFilter})`,
        `SELECT * FROM node_metadata WHERE scopeId = ${scopeFilter}`,
        `SELECT * FROM node_transforms WHERE nodeId IN (SELECT nodeId FROM node_metadata WHERE scopeId = ${scopeFilter})`,
        `SELECT * FROM node_data WHERE nodeId IN (SELECT nodeId FROM node_metadata WHERE scopeId = ${scopeFilter})`,
        `SELECT * FROM edges`,
      ]);

    return () => {
      // 4. Correct Cleanup using the same callback references
      conn.db.nodes.removeOnInsert(onInsert);
      conn.db.nodeTransforms.removeOnUpdate(onUpdateTransform);
      conn.db.nodeTransforms.removeOnInsert(onInsertTransform);
      conn.db.nodeData.removeOnUpdate(onUpdateData);
      conn.db.nodeData.removeOnInsert(onInsertData);
      sub.unsubscribe();
    };
  },
};

/**
 * Node Lenses
 */
export const NodeLenses = {
  chatHead: (nodeId: string): SyncedLens<string> => ({
    category: "node",
    description: `Update chat head for ${nodeId}`,
    get: (s) => {
      const n = s.nodesById[nodeId];
      return n && isChatNode(n) ? n.data.extension.value.conversationHeadId : "";
    },
    id: nodeId,
    set: (d, messageId) => {
      const node = d.nodesById[nodeId];
      if (node && isChatNode(node)) {
        node.data.extension.value.conversationHeadId = messageId;
        node.data.extension.value.isHistoryCleared = false;
      }
    },
  }),

  chatTreeId: (nodeId: string): SyncedLens<string> => ({
    category: "node",
    description: `Read chat tree id for ${nodeId}`,
    get: (s) => {
      const n = s.nodesById[nodeId];
      if (n && isChatNode(n)) return n.data.extension.value.treeId || nodeId;
      return nodeId;
    },
    id: nodeId,
    set: () => {},
  }),

  layout: (
    nodeId: string,
  ): SyncedLens<{ height?: number; width?: number; x?: number; y?: number }> => ({
    category: "node",
    description: `Update transform for node ${nodeId}`,
    get: (s) => {
      const n = s.nodesById[nodeId];
      return { height: n?.height, width: n?.width, x: n?.position.x, y: n?.position.y };
    },
    id: nodeId,
    set: (d, layout) => {
      const node = d.nodesById[nodeId];
      if (!node) return;
      if (layout.x !== undefined) node.position.x = layout.x;
      if (layout.y !== undefined) node.position.y = layout.y;
      if (layout.width !== undefined) node.width = layout.width;
      if (layout.height !== undefined) node.height = layout.height;

      if (node.presentation) {
        if (node.presentation.position) {
          if (layout.x !== undefined) node.presentation.position.x = layout.x;
          if (layout.y !== undefined) node.presentation.position.y = layout.y;
        } else if (layout.x !== undefined || layout.y !== undefined) {
          node.presentation.position = createProto(PositionSchema, {
            x: layout.x ?? node.position.x,
            y: layout.y ?? node.position.y,
          });
        }
        if (layout.width !== undefined) node.presentation.width = layout.width;
        if (layout.height !== undefined) node.presentation.height = layout.height;
      }
    },
  }),

  messageContent: (nodeId: string): SyncedLens<string> => ({
    category: "node",
    description: `Update node ${nodeId} message content`,
    get: (s) => {
      const n = s.nodesById[nodeId];
      try {
        const metadata = n?.data?.metadata;
        const json = metadata ? metadata["parts_json"] : undefined;
        if (!json) return "";
        const rawParts = JSON.parse(json);
        const parts = rawParts.map((p: any) =>
          fromJsonString(ChatMessagePartSchema, JSON.stringify(p)),
        );
        return partsToText(parts).trim();
      } catch {
        return "";
      }
    },
    id: nodeId,
    set: (d, newText) => {
      const node = d.nodesById[nodeId];
      if (node && node.data) {
        if (!node.data.metadata) node.data.metadata = {};
        const part = createProto(ChatMessagePartSchema, {
          part: { case: "text", value: newText },
        });
        node.data.metadata["parts_json"] = JSON.stringify([
          JSON.parse(toJsonString(ChatMessagePartSchema, part)),
        ]);
      }
    },
  }),

  prop: <K extends keyof DynamicNodeData>(
    nodeId: string,
    key: K,
  ): SyncedLens<DynamicNodeData[K]> => ({
    category: "node",
    description: `Update node ${nodeId}.${String(key)}`,
    get: (s) => (s.nodesById[nodeId]?.data as DynamicNodeData)?.[key],
    id: nodeId,
    set: (d, val) => {
      const node = d.nodesById[nodeId];
      // Safer check using type assertion for indexing into known structure
      if (node && node.data) {
        (node.data as DynamicNodeData)[key] = val;
      }
    },
  }),
};