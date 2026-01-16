import { type GraphMutation } from "@/generated/flowcraft/v1/core/service_pb";
import { type AppNode, type Edge } from "@/types";
import { fromProtoNode } from "@/utils/protoAdapter";

export function executeMutation(mut: GraphMutation, graph: { edges: Edge[]; nodes: AppNode[] }) {
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
      // Server-side path update logic simplified
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
    case "updateNode": {
      const node = graph.nodes.find((n) => n.id === op.value.id);
      if (node) {
        if (op.value.data) {
          node.data = {
            ...node.data,
            ...op.value.data,
          } as any;
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
