import { toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";

import {
  type GraphMutation,
  PathUpdate_UpdateType,
} from "../generated/flowcraft/v1/core/service_pb";
import { type AppNode, type DynamicNodeData } from "../types";
import { fromProtoNode, fromProtoNodeData } from "../utils/protoAdapter";
import { NodeRegistry } from "./registry";

/**
 * 执行单条突变并应用到 serverGraph
 */
export function executeMutation(
  mut: GraphMutation,
  graph: { edges: unknown[]; nodes: AppNode[] },
) {
  const op = mut.operation;
  if (!op.case) return;

  switch (op.case) {
    case "addEdge": {
      const e = op.value.edge;
      if (typeof e === "undefined") break;
      graph.edges.push({
        data: e.metadata as Record<string, unknown>,
        id: e.edgeId,
        source: e.sourceNodeId,
        sourceHandle: e.sourceHandle || undefined,
        target: e.targetNodeId,
        targetHandle: e.targetHandle || undefined,
      });
      break;
    }

    case "addNode":
      if (op.value.node) {
        const node = fromProtoNode(op.value.node);
        const templateId = (node.data as DynamicNodeData).typeId;
        const template = NodeRegistry.getDefinition(templateId ?? "")?.template;

        if (template && template.defaultState) {
          const defaultData = fromProtoNodeData(template.defaultState);
          const nodeData = node.data as DynamicNodeData;
          node.data = {
            ...defaultData,
            ...nodeData,
            widgetsValues: {
              ...(defaultData.widgetsValues ?? {}),
              ...(nodeData.widgetsValues ?? {}),
            },
          };
        }
        graph.nodes.push(node);
      }
      break;

    case "clearGraph":
      graph.nodes = [];
      graph.edges = [];
      break;

    case "pathUpdate": {
      const { path, targetId, type, value } = op.value;
      const node = graph.nodes.find((n) => n.id === targetId);
      if (node && value) {
        try {
          const val = toJson(ValueSchema, value);
          setByPath(
            node as unknown as Record<string, unknown>,
            path,
            val,
            type === PathUpdate_UpdateType.MERGE,
          );
        } catch (e) {
          console.error("[MutationExecutor] Path update error:", e);
        }
      }
      break;
    }

    case "removeEdge":
      graph.edges = graph.edges.filter(
        (e) => (e as { id: string }).id !== op.value.id,
      );
      break;

    case "removeNode":
      graph.nodes = graph.nodes.filter((n) => n.id !== op.value.id);
      break;

    case "updateNode": {
      const val = op.value;
      const node = graph.nodes.find((n) => n.id === val.id);
      if (node) {
        if (val.presentation) {
          const pres = val.presentation;
          if (pres.position)
            node.position = { x: pres.position.x, y: pres.position.y };
          if (pres.width || pres.height) {
            node.measured = {
              height: (pres.height || node.measured?.height) ?? 0,
              width: (pres.width || node.measured?.width) ?? 0,
            };
          }
          node.parentId = pres.parentId === "" ? undefined : pres.parentId;
        }
        if (val.data) {
          const appData = fromProtoNodeData(val.data);
          node.data = { ...node.data, ...appData };
        }
      }
      break;
    }
  }
}

/**
 * 路径设置工具 (内部使用)
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
