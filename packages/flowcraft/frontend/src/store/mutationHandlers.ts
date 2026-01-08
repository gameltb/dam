import { create as createProto, toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { type Edge } from "@xyflow/react";
import * as Y from "yjs";

import {
  type GraphMutation,
  GraphMutationSchema,
  PathUpdate_UpdateType,
} from "@/generated/flowcraft/v1/core/service_pb";
import { type AppNode } from "@/types";
import { dehydrateNode } from "@/utils/nodeUtils";
import { fromProtoNode, fromProtoNodeData } from "@/utils/protoAdapter";

/**
 * 极简的路径设置工具，支持 a.b.c 格式
 */
function setByPath(
  obj: Record<string, unknown>,
  path: string,
  value: unknown,
  merge = false,
) {
  const parts = path.split(".");
  let current = obj;

  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    if (part === undefined) break;
    if (
      !(part in current) ||
      typeof current[part] !== "object" ||
      current[part] === null
    ) {
      current[part] = {};
    }
    current = current[part] as Record<string, unknown>;
  }

  const lastPart = parts[parts.length - 1];
  if (lastPart === undefined) return;
  if (merge && typeof value === "object" && value !== null) {
    const existingValue = current[lastPart];
    current[lastPart] = {
      ...(typeof existingValue === "object" && existingValue !== null
        ? existingValue
        : {}),
      ...value,
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
    case "addEdge":
      if (op.value.edge) {
        const edge = op.value.edge;
        const rfEdge: Edge = {
          data: edge.metadata,
          id: edge.edgeId,
          source: edge.sourceNodeId,
          sourceHandle: edge.sourceHandle || undefined,
          target: edge.targetNodeId,
          targetHandle: edge.targetHandle || undefined,
        };
        yEdges.set(rfEdge.id, rfEdge);
      }
      break;
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

    case "clearGraph":
      yNodes.clear();
      yEdges.clear();
      break;

    case "pathUpdate": {
      const { path, targetId, type, value } = op.value;
      const existing = yNodes.get(targetId);
      if (existing && value) {
        try {
          const val = toJson(ValueSchema, value);
          const updated = JSON.parse(JSON.stringify(existing)) as Record<
            string,
            unknown
          >;
          setByPath(updated, path, val, type === PathUpdate_UpdateType.MERGE);
          yNodes.set(targetId, updated);
        } catch (e) {
          console.error("[Mutation] Failed to apply path update:", e);
        }
      }
      break;
    }

    case "removeEdge":
      if (op.value.id) {
        yEdges.delete(op.value.id);
      }
      break;

    case "removeNode":
      if (op.value.id) {
        yNodes.delete(op.value.id);
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
              height: newHeight,
              width: newWidth,
            };
            updated.style = {
              ...updated.style,
              height: newHeight,
              width: newWidth,
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
  }
};
