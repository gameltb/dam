import { type GraphMutation } from "@/generated/flowcraft/v1/core/service_pb";
import { type AppNode, AppNodeType, type Edge } from "@/types";
import { fromProtoNode, fromProtoNodeData } from "@/utils/protoAdapter";

export function executeMutation(
  mut: GraphMutation,
  graph: { edges: Edge[]; nodes: AppNode[] },
) {
  const op = mut.operation;
  if (!op.case) return;

  switch (op.case) {
    case "addEdge": {
      const edge = op.value.edge;
      if (edge) {
        graph.edges.push({
          data: edge.metadata,
          id: edge.edgeId,
          source: edge.sourceNodeId,
          sourceHandle: edge.sourceHandle,
          target: edge.targetNodeId,
          targetHandle: edge.targetHandle,
        });
      }
      break;
    }
    case "addNode": {
      if (op.value.node) {
        const newNode = fromProtoNode(op.value.node);
        graph.nodes.push(newNode);
      }
      break;
    }
    case "clearGraph": {
      graph.nodes = [];
      graph.edges = [];
      break;
    }
    case "pathUpdate": {
      const node = graph.nodes.find((n) => n.id === op.value.targetId);
      if (node && node.type === AppNodeType.DYNAMIC) {
        const path = op.value.path.split(".");
        // We use a recursive approach or a safe iterative approach with unknown/record
        let current: Record<string, unknown> = node as unknown as Record<
          string,
          unknown
        >;

        for (let i = 0; i < path.length - 1; i++) {
          const part = path[i];
          if (part === undefined) continue;

          if (
            typeof current[part] !== "object" ||
            current[part] === null ||
            Array.isArray(current[part])
          ) {
            current[part] = {};
          }
          current = current[part] as Record<string, unknown>;
        }

        const lastPart = path[path.length - 1];
        if (lastPart !== undefined) {
          current[lastPart] = op.value.value;
        }
      }
      break;
    }
    case "removeEdge": {
      graph.edges = graph.edges.filter((e) => e.id !== op.value.id);
      break;
    }
    case "removeNode": {
      graph.nodes = graph.nodes.filter((n) => n.id !== op.value.id);
      graph.edges = graph.edges.filter(
        (e) => e.source !== op.value.id && e.target !== op.value.id,
      );
      break;
    }
    case "updateNode": {
      const node = graph.nodes.find((n) => n.id === op.value.id);
      if (node) {
        if (op.value.data) {
          node.data = {
            ...node.data,
            ...fromProtoNodeData(op.value.data),
          };
        }
        if (op.value.presentation) {
          const pres = op.value.presentation;
          if (pres.position) {
            node.position = { x: pres.position.x, y: pres.position.y };
          }
          if (pres.width && pres.height) {
            node.measured = { height: pres.height, width: pres.width };
          }
          if (pres.parentId) {
            node.parentId = pres.parentId;
          } else {
            node.parentId = undefined;
          }
        }
      }
      break;
    }
  }
}
