import { create } from "@bufbuild/protobuf";
import { v4 as uuidv4 } from "uuid";

import {
  NodeDataSchema,
  NodeTemplateSchema,
  RenderMode,
} from "@/generated/flowcraft/v1/core/node_pb";

import { NodeRegistry } from "../services/NodeRegistry";
import { ChatNodeInstance } from "./ChatNodeInstance";

NodeRegistry.register({
  createInstance: (nodeId: string) => new ChatNodeInstance(uuidv4(), nodeId),
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_CHAT,
      availableModes: [RenderMode.MODE_CHAT, RenderMode.MODE_WIDGETS],
      displayName: "AI Assistant",
      extension: {
        case: "chat",
        value: {
          conversationHeadId: "",
          isHistoryCleared: false,
          treeId: "",
        },
      },
    }),
    displayName: "AI Conversational Assistant",
    menuPath: ["AI"],
    templateId: "flowcraft.node.ai.conversational",
  }),
});
