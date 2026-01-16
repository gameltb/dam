/** AUTO-GENERATED - DO NOT EDIT **/ 
/* eslint-disable */
import { ChatSyncMessageSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { EdgeSchema, NodeSchema, NodeTemplateSchema, TaskUpdateSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { ChatMessageSchema, InferenceConfigDiscoveryResponseSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { ViewportSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { ActionExecutionRequestSchema } from "@/generated/flowcraft/v1/core/action_pb";
import { NodeSignalSchema } from "@/generated/flowcraft/v1/core/signals_pb";


/**
 * PB 覆盖清单
 */
export const PB_REDUCERS_MAP = {
  "addChatMessage": {
      message: { schema: ChatSyncMessageSchema }
  },
  "registerTemplate": {
      template: { schema: NodeTemplateSchema }
  },
  "updateInferenceConfig": {
      config: { schema: InferenceConfigDiscoveryResponseSchema }
  },
  "addEdgePb": {
      edge: { schema: EdgeSchema }
  },
  "createNodePb": {
      node: { schema: NodeSchema }
  },
  "updateNodePb": {
      node: { schema: NodeSchema }
  },
  "updateViewport": {
      viewport: { schema: ViewportSchema }
  },
  "executeAction": {
      request: { schema: ActionExecutionRequestSchema }
  },
  "sendNodeSignal": {
      signal: { schema: NodeSignalSchema }
  },
  "updateTaskStatus": {
      update: { schema: TaskUpdateSchema }
  } 
} as const;

/**
 * 表与 Protobuf Schema 的映射
 */
export const TABLE_TO_PROTO: Record<string, { schema: any, field: string }> = {
  "nodes": { schema: NodeSchema, field: "state" },
  "edges": { schema: EdgeSchema, field: "state" },
  "viewportState": { schema: ViewportSchema, field: "state" },
  "chatMessages": { schema: ChatMessageSchema, field: "state" },
  "nodeTemplates": { schema: NodeTemplateSchema, field: "state" },
  "inferenceConfig": { schema: InferenceConfigDiscoveryResponseSchema, field: "state" },
  "tasks": { schema: ActionExecutionRequestSchema, field: "request" } 
};
