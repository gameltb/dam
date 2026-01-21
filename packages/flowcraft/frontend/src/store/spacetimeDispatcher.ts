import {
  AddEdgeRequestSchema,
  AddNodeRequestSchema,
  ClearGraphRequestSchema,
  PathUpdateRequestSchema,
  RemoveEdgeRequestSchema,
  RemoveNodeRequestSchema,
  ReparentNodeRequestSchema,
} from "@/generated/flowcraft/v1/core/service_pb";
import { type PbConnection } from "@/utils/pb-client";

export const dispatchToSpacetime = (conn: PbConnection, input: any) => {
  switch (input.$typeName) {
    case AddEdgeRequestSchema.typeName:
      if (input.edge) conn.pbreducers.addEdgePb({ edge: input.edge });
      break;

    case AddNodeRequestSchema.typeName:
      if (input.node) conn.pbreducers.createNodePb({ node: input.node });
      break;

    case ClearGraphRequestSchema.typeName:
      if ((conn.reducers as any).clearGraph) {
        (conn.reducers as any).clearGraph();
      }
      break;

    case PathUpdateRequestSchema.typeName:
      // 现在后端已支持增量路径更新 Reducer
      conn.pbreducers.pathUpdatePb({ req: input });
      break;

    case RemoveEdgeRequestSchema.typeName:
      conn.reducers.removeEdge({ id: input.id });
      break;

    case RemoveNodeRequestSchema.typeName:
      conn.reducers.removeNode({ id: input.id });
      break;

    case ReparentNodeRequestSchema.typeName:
      conn.pbreducers.reparentNodePb({ req: input });
      break;
  }
};
