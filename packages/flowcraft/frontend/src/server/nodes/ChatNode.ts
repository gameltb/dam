import { create } from "@bufbuild/protobuf";
import {
  NodeTemplateSchema,
  NodeDataSchema,
  RenderMode,
} from "../../generated/flowcraft/v1/core/node_pb";
import { NodeRegistry } from "../registry";

NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    templateId: "tpl-ai-chat",
    displayName: "AI Chat",
    menuPath: ["AI"],
    defaultState: create(NodeDataSchema, {
      displayName: "AI Assistant",
      availableModes: [RenderMode.MODE_CHAT, RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_CHAT,
      metadata: {
        chat_history: JSON.stringify([
          {
            id: "1",
            role: "assistant",
            content: "Hello! How can I help you today?",
          },
        ]),
      },
    }),
  }),
});
