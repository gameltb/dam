import { create } from "@bufbuild/protobuf";
import { type Edge } from "@xyflow/react";

import { NodeDataSchema, type Node as ProtoNode } from "@/generated/flowcraft/v1/core/node_pb";
import { type GraphSnapshot } from "@/generated/flowcraft/v1/core/service_pb";
import { type AppNode, AppNodeType, type DynamicNodeData } from "@/types";

import { appEdgeToProto, appNodeToProto, protoToAppEdge } from "./nodeProtoUtils";
import { KIND_TO_NODE_TYPE } from "./nodeUtils";

export { protoToAppEdge as fromProtoEdge, appEdgeToProto as toProtoEdge, appNodeToProto as toProtoNode };

export function fromProtoGraph(protoGraph: GraphSnapshot): {
  edges: Edge[];
  nodes: AppNode[];
} {
  const nodes: AppNode[] = protoGraph.nodes.map((n) => fromProtoNode(n));
  const edges: Edge[] = protoGraph.edges.map(protoToAppEdge);

  return { edges, nodes };
}

export function fromProtoNode(n: ProtoNode): AppNode {
  const reactFlowType = KIND_TO_NODE_TYPE[n.nodeKind] ?? AppNodeType.DYNAMIC;

  const appData = (n.state ?? create(NodeDataSchema, {})) as DynamicNodeData;
  appData.templateId = n.templateId;

  const pres = n.presentation;
  let parentId: string | undefined = pres?.parentId;
  if (parentId === "") parentId = undefined;

  const node: AppNode = {
    data: appData,
    extent: parentId ? "parent" : undefined,
    id: n.nodeId,
    parentId,
    position: { x: pres?.position?.x ?? 0, y: pres?.position?.y ?? 0 },
    selected: n.isSelected,
    type: reactFlowType,
  };

  if (pres && pres.width && pres.height) {
    node.measured = { height: pres.height, width: pres.width };
    // @ts-ignore - React Flow style property handling
    node.style = { ...node.style, height: pres.height, width: pres.width };
  }

  return node;
}
