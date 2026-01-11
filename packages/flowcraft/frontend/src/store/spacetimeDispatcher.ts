import { toJson } from "@bufbuild/protobuf";

import { NodeDataSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { type GraphMutation } from "@/generated/flowcraft/v1/core/service_pb";
import { type DbConnection } from "@/generated/spacetime";

/**
 * Helper types to extract oneof cases and values from Protobuf ES messages.
 */
type MutationOp = GraphMutation["operation"];
type MutationCase = NonNullable<MutationOp["case"]>;
type MutationValue<K extends MutationCase> = Extract<
  MutationOp,
  { case: K }
>["value"];

/**
 * Exhaustive handler map for all graph mutations.
 * If a case is missing, TypeScript will throw an error.
 */
const HANDLERS: {
  [K in MutationCase]: (conn: DbConnection, val: MutationValue<K>) => void;
} = {
  addEdge: (conn, val) => {
    conn.reducers.addEdge({
      id: val.edge?.edgeId ?? "",
      sourceHandle: val.edge?.sourceHandle ?? "",
      sourceId: val.edge?.sourceNodeId ?? "",
      targetHandle: val.edge?.targetHandle ?? "",
      targetId: val.edge?.targetNodeId ?? "",
    });
  },
  addNode: (conn, val) => {
    const node = val.node;
    if (!node) return;
    conn.reducers.createNode({
      dataJson: JSON.stringify(toJson(NodeDataSchema, node.state!)),
      height: node.presentation?.height ?? 0,
      id: node.nodeId,
      isSelected: node.isSelected,
      kind: node.nodeKind,
      parentId: node.presentation?.parentId ?? "",
      posX: node.presentation?.position?.x ?? 0,
      posY: node.presentation?.position?.y ?? 0,
      templateId: node.templateId,
      width: node.presentation?.width ?? 0,
    });
  },
  clearGraph: () => {
    // Optionally implement global clear via a new reducer
  },
  removeEdge: (conn, val) => {
    conn.reducers.removeEdge({ id: val.id });
  },
  removeNode: (conn, val) => {
    conn.reducers.removeNode({ id: val.id });
  },
  pathUpdate: () => {
    // Implement path specific sync if needed in the future
  },
  updateNode: (conn, val) => {
    const p = val.presentation;
    if (p) {
      conn.reducers.updateNodeLayout({
        height: p.height,
        id: val.id,
        width: p.width,
        x: p.position?.x ?? 0,
        y: p.position?.y ?? 0,
      });
    }
    if (val.data) {
      conn.reducers.updateNodeData({
        dataJson: JSON.stringify(toJson(NodeDataSchema, val.data)),
        id: val.id,
      });
    }
  },
};

/**
 * Dispatches a generic GraphMutation to the appropriate SpacetimeDB reducer.
 */
export const dispatchToSpacetime = (
  conn: DbConnection,
  mutation: GraphMutation,
) => {
  const { case: opCase, value } = mutation.operation;
  if (!opCase) return;

  const handler = HANDLERS[opCase] as (conn: DbConnection, v: any) => void;
  if (handler) {
    handler(conn, value);
  }
};