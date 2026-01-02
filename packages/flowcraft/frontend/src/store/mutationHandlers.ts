import { type Edge } from "@xyflow/react";
import { type AppNode } from "../types";
import {
  type GraphMutation,
  GraphMutationSchema,
} from "../generated/core/service_pb";
import { create as createProto } from "@bufbuild/protobuf";
import * as Y from "yjs";
import { dehydrateNode } from "../utils/nodeUtils";
import { fromProtoNode } from "../utils/protoAdapter";

export const handleGraphMutation = (
  mutInput: GraphMutation,
  yNodes: Y.Map<unknown>,
  yEdges: Y.Map<unknown>,
) => {
  const mut = createProto(GraphMutationSchema, mutInput);
  const op = mut.operation;

  if (!op.case) return;

  switch (op.case) {
    case "addNode":
      if (op.value.node) {
        const node = fromProtoNode(op.value.node);
        if (node.id) {
          yNodes.set(node.id, dehydrateNode(node));
        }
      }
      break;

    case "updateNode": {
      const val = op.value;
      const id = val.id;
      if (!id) break;

      const existing = yNodes.get(id) as AppNode | undefined;
      if (existing) {
        const updated = { ...existing } as AppNode;

        if (val.position) {
          updated.position = {
            x: val.position.x || updated.position.x,
            y: val.position.y || updated.position.y,
          };
        }

        if (val.width !== 0 || val.height !== 0) {
          const newWidth =
            val.width || (updated.measured ? updated.measured.width : 0);
          const newHeight =
            val.height || (updated.measured ? updated.measured.height : 0);

          updated.measured = {
            width: newWidth,
            height: newHeight,
          };
          updated.style = {
            ...updated.style,
            width: updated.measured.width,
            height: updated.measured.height,
          };
        }

        if (val.data) {
          updated.data = {
            ...updated.data,
            ...(val.data as Record<string, unknown>),
          };
        }

        const pId = val.parentId;
        updated.parentId = (pId === "" ? undefined : pId) ?? undefined;
        // Usually when parentId is set, we want the node to be contained within parent
        updated.extent = updated.parentId ? "parent" : undefined;
        // Preserve extent if it was already set and not explicitly removed
        if (updated.parentId && !updated.extent) {
          updated.extent = "parent";
        }

        yNodes.set(id, dehydrateNode(updated));
      }
      break;
    }

    case "removeNode":
      if (op.value.id) {
        yNodes.delete(op.value.id);
      }
      break;

    case "addEdge":
      if (op.value.edge) {
        const edge = op.value.edge as Edge;
        yEdges.set(edge.id, edge);
      }
      break;

    case "removeEdge":
      if (op.value.id) {
        yEdges.delete(op.value.id);
      }
      break;

    case "addSubgraph":
      op.value.nodes.forEach((n) => {
        const node = fromProtoNode(n);
        if (node.id) yNodes.set(node.id, dehydrateNode(node));
      });

      op.value.edges.forEach((e) => {
        const edge = e as Edge;
        if (edge.id) yEdges.set(edge.id, edge);
      });
      break;

    case "clearGraph":
      yNodes.clear();
      yEdges.clear();
      break;
  }
};
