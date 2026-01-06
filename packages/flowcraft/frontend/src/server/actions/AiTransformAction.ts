import { create } from "@bufbuild/protobuf";
import {
  ActionTemplateSchema,
  ActionExecutionStrategy,
} from "../../generated/flowcraft/v1/core/action_pb";
import { NodeRegistry, type ActionHandlerContext } from "../registry";
import { serverGraph, incrementVersion } from "../db";
import { NodeSchema } from "../../generated/flowcraft/v1/core/node_pb";
import { MutationListSchema } from "../../generated/flowcraft/v1/core/service_pb";
import { v4 as uuidv4 } from "uuid";
import { fromProtoNode, toProtoNode } from "../../utils/protoAdapter";
import { addChatMessage } from "../db";

async function handleAiTransform(ctx: ActionHandlerContext) {
  const { sourceNodeId, contextNodeIds = [], params, emitMutation } = ctx;
  const instruction =
    (params.instruction as string) || "Context aware transform";

  const targetNodes = serverGraph.nodes.filter(
    (n) => contextNodeIds.includes(n.id) || n.id === sourceNodeId,
  );

  const contextText = targetNodes
    .map((n) => `[Node ${n.id}]: ${n.data.label || n.data.displayName}`)
    .join("\n");

  const maxY = Math.max(
    ...targetNodes.map((n) => n.position.y + (n.measured?.height ?? 200)),
    0,
  );
  const avgX =
    targetNodes.reduce((acc, n) => acc + n.position.x, 0) /
    (targetNodes.length || 1);

  const newNodeId = `chat-${uuidv4().slice(0, 8)}`;

  const userMsgId = uuidv4();
  addChatMessage({
    id: userMsgId,
    parentId: null,
    role: "user",
    content: `Context:\n${contextText}\n\nInstruction: ${instruction}`,
    nodeId: newNodeId,
  });

  const newNode = fromProtoNode(
    create(NodeSchema, {
      nodeId: newNodeId,
      nodeKind: 1,
      templateId: "tpl-ai-chat",
      presentation: {
        position: { x: avgX, y: maxY + 50 },
        width: 400,
        height: 500,
        isInitialized: true,
      },
      state: {
        displayName: "AI Transform",
        metadata: {
          conversation_head_id: userMsgId,
        },
      },
    }),
  );

  serverGraph.nodes.push(newNode);
  incrementVersion();

  emitMutation(
    create(MutationListSchema, {
      mutations: [
        {
          operation: { case: "addNode", value: { node: toProtoNode(newNode) } },
        },
      ],
      source: 2,
    }),
  );
}

NodeRegistry.registerGlobalAction(
  create(ActionTemplateSchema, {
    id: "ai-transform",
    label: "AI Generate (Context Aware)",
    path: ["AI Tools"],
    strategy: ActionExecutionStrategy.EXECUTION_BACKGROUND,
  }),
  handleAiTransform,
);
