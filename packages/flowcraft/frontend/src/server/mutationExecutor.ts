import { type AppNode, type DynamicNodeData } from "../types";
import { fromProtoNode, fromProtoNodeData } from "../utils/protoAdapter";
import {
  type GraphMutation,
  PathUpdate_UpdateType,
} from "../generated/flowcraft/v1/core/service_pb";
import { NodeRegistry } from "./registry";

/**
 * 路径设置工具 (内部使用)
 */
function setByPath(obj: any, path: string, value: any, merge = false) {
  const parts = path.split(".");
  let current = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i]!;
    if (!(part in current)) current[part] = {};
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

/**
 * 执行单条突变并应用到 serverGraph
 */
export function executeMutation(
  mut: GraphMutation,
  graph: { nodes: AppNode[]; edges: any[] },
) {
  const op = mut.operation;
  if (!op.case) return;

  switch (op.case) {
    case "addNode":
      if (op.value.node) {
        const node = fromProtoNode(op.value.node);
        const templateId = (node.data as DynamicNodeData).typeId;
        const template = NodeRegistry.getDefinition(templateId || "")?.template;

        if (template && template.defaultState) {
          const defaultData = fromProtoNodeData(template.defaultState);
          const nodeData = node.data as DynamicNodeData;
          node.data = {
            ...defaultData,
            ...nodeData,
            widgetsValues: {
              ...(defaultData.widgetsValues || {}),
              ...(nodeData.widgetsValues || {}),
            },
          };
        }
        graph.nodes.push(node);
      }
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
              width: pres.width || node.measured?.width || 0,
              height: pres.height || node.measured?.height || 0,
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

    case "removeNode":
      graph.nodes = graph.nodes.filter((n) => n.id !== op.value.id);
      break;

    case "addEdge":
      if (op.value.edge) {
        const e = op.value.edge;
        graph.edges.push({
          id: e.edgeId,
          source: e.sourceNodeId,
          target: e.targetNodeId,
          sourceHandle: e.sourceHandle || undefined,
          targetHandle: e.targetHandle || undefined,
          data: (e.metadata as Record<string, unknown>) || {},
        });
      }
      break;

    case "removeEdge":
      graph.edges = graph.edges.filter((e) => e.id !== op.value.id);
      break;

    case "pathUpdate": {
      const { targetId, path, valueJson, type } = op.value;
      const node = graph.nodes.find((n) => n.id === targetId);
      if (node) {
        try {
          const val = JSON.parse(valueJson);
          setByPath(node, path, val, type === PathUpdate_UpdateType.MERGE);
        } catch (e) {
          console.error("[MutationExecutor] Path update error:", e);
        }
      }
      break;
    }

    case "clearGraph":
      graph.nodes = [];
      graph.edges = [];
      break;
  }
}
