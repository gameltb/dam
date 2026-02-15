import { create as createProto, fromJsonString, toJsonString } from "@bufbuild/protobuf";
import { type Infer } from "spacetimedb";

import { partsToText } from "@/components/media/chat/utils";
import { ChatMessagePartSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { PositionSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { type NodeDataRow, type NodesRow, type NodeTransformsRow } from "@/generated/spacetime";
import { useFlowStore } from "@/store/flowStore";
import { type DynamicNodeData, isChatNode } from "@/types";
import { GraphMapper } from "@/utils/graphMapper";
import { type SyncedLens } from "@/utils/lens-types";
import { log } from "@/utils/logger";
import { type PbConnection } from "@/utils/pb-client";
import { applySyncInsert, applySyncUpdate } from "@/utils/syncUtils";

export const nodeMaterializer = {
  name: "node",

  setup: (conn: PbConnection, _activeScopeId: null | string) => {
    const onInsert = (_ctx: unknown, row: Infer<typeof NodesRow>) => {
      log.sync("IN", `Node Insert: ${row.nodeId}`, { row });
      applySyncInsert(row.nodeId, GraphMapper.createSkeleton(row));
    };

    const onUpdateTransform = (
      _ctx: unknown,
      _oldRow: Infer<typeof NodeTransformsRow> | null,
      row: Infer<typeof NodeTransformsRow>,
    ) => {
      // SILENT: Transforms are high-frequency
      applySyncUpdate(row.nodeId, (n) => GraphMapper.applyTransform(n, row));
    };

    const onInsertTransform = (_ctx: unknown, row: Infer<typeof NodeTransformsRow>) => {
      onUpdateTransform(_ctx, null, row);
    };

    const onUpdateData = (_ctx: unknown, _oldRow: Infer<typeof NodeDataRow> | null, row: Infer<typeof NodeDataRow>) => {
      log.sync("IN", `Data Update: ${row.nodeId}`, { row });
      applySyncUpdate(row.nodeId, (n) => GraphMapper.applyData(n, row), false);
    };

    const onInsertData = (_ctx: unknown, row: Infer<typeof NodeDataRow>) => {
      onUpdateData(_ctx, null, row);
    };
    conn.db.nodes.onInsert(onInsert);
    conn.db.nodeTransforms.onUpdate(onUpdateTransform);
    conn.db.nodeTransforms.onInsert(onInsertTransform);
    conn.db.nodeData.onUpdate(onUpdateData);
    conn.db.nodeData.onInsert(onInsertData);

    // 3. Simplified Subscriptions (SpacetimeDB does not support subqueries in subscriptions)
    const sub = conn
      .subscriptionBuilder()
      .onApplied(() => {
        log.sync("IN", "Subscription Applied - Running Full Sync");
        useFlowStore.getState().syncWithDatabase();
      })
      .onError((ctx) => {
        log.error("Sync", `Node Subscription Error: ${ctx.event}`);
      })
      .subscribe([
        "SELECT * FROM nodes",
        "SELECT * FROM node_metadata",
        "SELECT * FROM node_transforms",
        "SELECT * FROM node_data",
        "SELECT * FROM edges",
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
      if (!n || !isChatNode(n)) return "";
      return n.data.extension.value.conversationHeadId;
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
      if (!n || !isChatNode(n)) return nodeId;
      return n.data.extension.value.treeId || nodeId;
    },
    id: nodeId,
    set: () => {},
  }),

  layout: (nodeId: string): SyncedLens<{ height?: number; width?: number; x?: number; y?: number }> => ({
    category: "node",
    description: `Update transform for node ${nodeId}`,
    get: (s) => {
      const n = s.nodesById[nodeId];
      if (!n?.position) return {};
      return {
        height: n.height,
        width: n.width,
        x: n.position.x,
        y: n.position.y,
      };
    },
    id: nodeId,
    set: (d, layout) => {
      const node = d.nodesById[nodeId];
      if (!node) return;

      if (layout.x !== undefined || layout.y !== undefined) {
        if (!node.position) node.position = { x: 0, y: 0 };
        if (layout.x !== undefined) node.position.x = layout.x;
        if (layout.y !== undefined) node.position.y = layout.y;
      }

      if (layout.width !== undefined) node.width = layout.width;
      if (layout.height !== undefined) node.height = layout.height;

      if (node.presentation) {
        if (!node.presentation.position) {
          node.presentation.position = createProto(PositionSchema, {
            x: node.position.x,
            y: node.position.y,
          });
        }
        if (layout.x !== undefined) node.presentation.position.x = layout.x;
        if (layout.y !== undefined) node.presentation.position.y = layout.y;
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
      if (!n) return "";
      try {
        const metadata = n.data.metadata;
        const json = metadata ? metadata.parts_json : undefined;
        if (!json) return "";
        const rawParts = JSON.parse(json) as unknown[];
        const parts = rawParts.map((p) => fromJsonString(ChatMessagePartSchema, JSON.stringify(p)));
        return partsToText(parts).trim();
      } catch {
        return "";
      }
    },
    id: nodeId,
    set: (d, newText) => {
      const node = d.nodesById[nodeId];
      if (node?.data) {
        if (!node.data.metadata) node.data.metadata = {};
        const part = createProto(ChatMessagePartSchema, {
          part: { case: "text", value: newText },
        });
        node.data.metadata.parts_json = JSON.stringify([JSON.parse(toJsonString(ChatMessagePartSchema, part))]);
      }
    },
  }),

  prop: <K extends keyof DynamicNodeData>(nodeId: string, key: K): SyncedLens<DynamicNodeData[K]> => ({
    category: "node",
    description: `Update node ${nodeId}.${String(key)}`,
    get: (s) => {
      const n = s.nodesById[nodeId];
      if (!n) return undefined as any;
      return (n.data as DynamicNodeData)[key];
    },
    id: nodeId,
    set: (d, val) => {
      const node = d.nodesById[nodeId];
      // Safer check using type assertion for indexing into known structure
      if (node?.data) {
        (node.data as DynamicNodeData)[key] = val;
      }
    },
  }),
};
