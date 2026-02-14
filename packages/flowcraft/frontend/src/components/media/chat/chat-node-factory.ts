import { create, toJsonString } from "@bufbuild/protobuf";
import { ChatMessagePartSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { NodeDataSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { AppNodeType, RenderMode } from "@/types";

/**
 * Creates the initial state for a new Chat Message node.
 */
export function createChatMessageNode(id: string, parentId: string, content: string, scopeId: string, position: { x: number, y: number }) {
  return {
    data: {
      $typeName: NodeDataSchema.typeName,
      activeMode: RenderMode.MODE_MARKDOWN,
      availableModes: [RenderMode.MODE_MARKDOWN],
      displayName: "User Message",
      extension: { case: undefined, value: undefined },
      inputPorts: [],
      metadata: {
        parts_json: JSON.stringify([
          JSON.parse(
            toJsonString(
              ChatMessagePartSchema,
              create(ChatMessagePartSchema, {
                part: { case: "text", value: content.trim() },
              }),
            ),
          ),
        ]),
        role: "user",
        timestamp: Date.now().toString(),
      },
      outputPorts: [],
      schemaVersion: 1,
      taskId: "",
      widgets: [],
    },
    height: 150,
    id,
    parentId,
    position: { x: position.x, y: position.y + 300 },
    scopeId,
    type: AppNodeType.CHAT_MESSAGE,
    width: 300,
  };
}
