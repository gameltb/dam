import { create } from "@bufbuild/protobuf";

import { NodeSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { type GraphMutation } from "@/generated/flowcraft/v1/core/service_pb";
import { type PbConnection } from "@/utils/pb-client";

type MutationCase = NonNullable<MutationOp["case"]>;
type MutationOp = GraphMutation["operation"];
type MutationValue<K extends MutationCase> = Extract<MutationOp, { case: K }>["value"];

/**
 * 直接映射 Mutation 到 PbConnection.reducers
 * 使用 camelCase 命名以对齐官方 SDK 规范
 */
const HANDLERS: {
  [K in MutationCase]: (conn: PbConnection, val: MutationValue<K>) => void;
} = {
  addEdge: (conn, val) => {
    if (val.edge) conn.pbreducers.addEdgePb({ edge: val.edge });
  },
  addNode: (conn, val) => {
    if (val.node) conn.pbreducers.createNodePb({ node: val.node });
  },
  clearGraph: () => {
    // Implement if a clear_graph reducer exists
  },
  pathUpdate: () => {
    // Handle path updates if needed
  },
  removeEdge: (conn, val) => {
    if (val.id) conn.reducers.removeEdge({ id: val.id });
  },
  removeNode: (conn, val) => {
    if (val.id) conn.reducers.removeNode({ id: val.id });
  },
  updateNode: (conn, val) => {
    if (val.id) {
      conn.pbreducers.updateNodePb({
        node: create(NodeSchema, {
          nodeId: val.id,
          presentation: val.presentation,
          state: val.data,
        }),
      });
    }
  },
};

export const dispatchToSpacetime = (conn: PbConnection, mutation: GraphMutation) => {
  const { case: opCase, value } = mutation.operation;
  if (!opCase) return;

  const handler = HANDLERS[opCase] as (conn: PbConnection, v: unknown) => void;
  handler(conn, value);
};
