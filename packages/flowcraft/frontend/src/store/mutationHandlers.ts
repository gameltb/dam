import { type Edge } from "@xyflow/react";
import { type AppNode } from "../types";
import {
  type GraphMutation,
  GraphMutationSchema,
  PathUpdate_UpdateType,
} from "../generated/flowcraft/v1/core/service_pb";
import { create as createProto } from "@bufbuild/protobuf";
import * as Y from "yjs";
import { dehydrateNode } from "../utils/nodeUtils";
import { fromProtoNode, fromProtoNodeData } from "../utils/protoAdapter";

/**
 * 极简的路径设置工具，支持 a.b.c 格式
 */
function setByPath(obj: any, path: string, value: any, merge = false) {
  const parts = path.split(".");
  let current = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i]!;
    if (!(part in current)) {
      current[part] = {};
    }
    current = current[part];
  }
  const lastPart = parts[parts.length - 1]!;
  if (merge && typeof value === "object" && value !== null) {
    current[lastPart] = {
      ...(current[lastPart] as object),
      ...(value as object),
    };
  } else {
    current[lastPart] = value;
  }
}

export const handleGraphMutation = (
  mutInput: GraphMutation,
  yNodes: Y.Map<unknown>,
  yEdges: Y.Map<unknown>,
) => {
  const mut = createProto(GraphMutationSchema, mutInput);
  const op = mut.operation;

  if (!op.case) return;

  switch (op.case) {
    case "pathUpdate": {
      const { targetId, path, valueJson, type } = op.value;
      const existing = yNodes.get(targetId) as any;
      if (existing) {
        try {
          const val = JSON.parse(valueJson);
          const updated = JSON.parse(JSON.stringify(existing)); // 深拷贝以确保纯净
          setByPath(updated, path, val, type === PathUpdate_UpdateType.MERGE);
          yNodes.set(targetId, updated);
        } catch (e) {
          console.error("[Mutation] Failed to apply path update:", e);
        }
      }
      break;
    }
    case "addNode":
      if (op.value.node) {
        const node = fromProtoNode(op.value.node);
        if (node.id) {
          const existing = yNodes.get(node.id) as AppNode | undefined;
          if (existing && existing.type === node.type) {
            // Merge data if types match
            const updated = {
              ...existing,
              ...node,
              data: {
                ...(existing.data as Record<string, unknown>),
                ...(node.data as Record<string, unknown>),
              },
            } as unknown as AppNode;
            yNodes.set(node.id, dehydrateNode(updated));
          } else {
            yNodes.set(node.id, dehydrateNode(node));
          }
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

        if (val.presentation) {
          const pres = val.presentation;
          if (pres.position) {
            updated.position = {
              x: pres.position.x,
              y: pres.position.y,
            };
          }

          if (pres.width !== 0 || pres.height !== 0) {
            const newWidth = pres.width || (updated.measured?.width ?? 0);
            const newHeight = pres.height || (updated.measured?.height ?? 0);

            updated.measured = {
              width: newWidth,
              height: newHeight,
            };
            updated.style = {
              ...updated.style,
              width: newWidth,
              height: newHeight,
            };
          }

          const pId = pres.parentId;
          updated.parentId = pId === "" ? undefined : pId;
          updated.extent = updated.parentId ? "parent" : undefined;
        }

        if (val.data) {
          const appData = fromProtoNodeData(val.data);
          updated.data = {
            ...updated.data,
            ...appData,
          };
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
        const edge = op.value.edge;
        const rfEdge: Edge = {
          id: edge.edgeId,
          source: edge.sourceNodeId,
          target: edge.targetNodeId,
          sourceHandle: edge.sourceHandle || undefined,
          targetHandle: edge.targetHandle || undefined,
          data: edge.metadata,
        };
        yEdges.set(rfEdge.id, rfEdge);
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
        const rfEdge: Edge = {
          id: e.edgeId,
          source: e.sourceNodeId,
          target: e.targetNodeId,
          sourceHandle: e.sourceHandle || undefined,
          targetHandle: e.targetHandle || undefined,
          data: e.metadata,
        };
        if (rfEdge.id) yEdges.set(rfEdge.id, rfEdge);
      });
      break;

    case "clearGraph":
      yNodes.clear();
      yEdges.clear();
      break;
  }
};
