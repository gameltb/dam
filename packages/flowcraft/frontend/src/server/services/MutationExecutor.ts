import { toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";

import { type GraphMutation } from "@/generated/flowcraft/v1/core/service_pb";
import { PathUpdateRequest_UpdateType } from "@/generated/flowcraft/v1/core/service_pb";
import { type AppNode, type Edge } from "@/types";

export function executeMutation(mut: GraphMutation, graph: { edges: Edge[]; nodes: AppNode[] }) {
  const op = mut.operation;
  if (!op.case) return;

  switch (op.case) {
    case "addSubGraph": {
      // Logic for adding subgraph
      break;
    }
    // ... other cases ...
    case "pathUpdate": {
      const input = op.value;
      const node = graph.nodes.find((n) => n.id === input.targetId);
      if (node) {
        const jsValue = input.value ? (toJson(ValueSchema, input.value) as any) : null;
        resolveAndApplyPath(node, input.path, jsValue, input.type);
      }
      break;
    }
    case "removeEdge": {
      graph.edges = graph.edges.filter((e) => e.id !== op.value.id);
      break;
    }
    case "removeNode": {
      graph.nodes = graph.nodes.filter((n) => n.id !== op.value.id);
      graph.edges = graph.edges.filter((e) => e.source !== op.value.id && e.target !== op.value.id);
      break;
    }
    case "reparentNode": {
      const node = graph.nodes.find((n) => n.id === op.value.nodeId);
      if (node) {
        node.parentId = op.value.newParentId || undefined;
        if (op.value.newPosition) {
          node.position = { x: op.value.newPosition.x, y: op.value.newPosition.y };
        }
      }
      break;
    }
  }
}

/**
 * 递归解析路径并应用修改。
 * 支持 snake_case 和 camelCase。
 */
function resolveAndApplyPath(target: any, path: string, value: any, type: PathUpdateRequest_UpdateType) {
  const parts = path.split(".");
  let current = target;

  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i]!;
    // 兼容层：查找对应的 JS 属性名
    const key =
      Object.keys(current).find((k) => k === part || k.replace(/([A-Z])/g, "_$1").toLowerCase() === part) || part;

    if (!current[key]) current[key] = {};
    current = current[key];
  }

  const lastPart = parts[parts.length - 1]!;
  const lastKey =
    Object.keys(current).find((k) => k === lastPart || k.replace(/([A-Z])/g, "_$1").toLowerCase() === lastPart) ||
    lastPart;

  if (type === PathUpdateRequest_UpdateType.DELETE) {
    delete current[lastKey];
  } else {
    current[lastKey] = value;
  }
}
