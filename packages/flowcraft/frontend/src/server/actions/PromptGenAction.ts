import { create } from "@bufbuild/protobuf";
import { v4 as uuidv4 } from "uuid";

import {
  ActionExecutionStrategy,
  ActionTemplateSchema,
} from "../../generated/flowcraft/v1/core/action_pb";
import { NodeSchema } from "../../generated/flowcraft/v1/core/node_pb";
import { MutationListSchema } from "../../generated/flowcraft/v1/core/service_pb";
import { fromProtoNode, toProtoNode } from "../../utils/protoAdapter";
import { incrementVersion, serverGraph } from "../db";
import { addChatMessage } from "../db";
import { type ActionHandlerContext, NodeRegistry } from "../registry";

function handlePromptGen(ctx: ActionHandlerContext): Promise<void> {
  const { contextNodeIds = [], emitMutation, params, sourceNodeId } = ctx;
  const prompt =
    (params.case === "promptGen" ? params.value.prompt : "") ||
    "Generate from prompt";

  const targetNodes = serverGraph.nodes.filter(
    (n) => contextNodeIds.includes(n.id) || n.id === sourceNodeId,
  );

  const contextText = targetNodes
    .map((n) => `[Node ${n.id}]: ${n.data.label ?? "Untitled"}`)
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
    content: contextText
      ? `Context:\n${contextText}\n\nPrompt: ${prompt}`
      : prompt,
    id: userMsgId,
    nodeId: newNodeId,
    parentId: null,
    role: "user",
  });

  const newNode = fromProtoNode(
    create(NodeSchema, {
      nodeId: newNodeId,

      nodeKind: 1,

      presentation: {
        height: 500,

        isInitialized: true,

        position: { x: avgX, y: maxY + 50 },

        width: 400,
      },

      state: {
        displayName: "Prompt Result",

        extension: {
          case: "chat",

          value: {
            conversationHeadId: userMsgId,
          },
        },
      },

      templateId: "flowcraft.node.ai.conversational",
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

  return Promise.resolve();
}

NodeRegistry.registerGlobalAction(
  create(ActionTemplateSchema, {
    id: "flowcraft.action.graph.prompt_to_chat",

    label: "Generate from Prompt",

    path: ["AI Tools"],

    strategy: ActionExecutionStrategy.EXECUTION_BACKGROUND,
  }),

  handlePromptGen,
);
