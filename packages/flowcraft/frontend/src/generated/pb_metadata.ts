/** AUTO-GENERATED - DO NOT EDIT **/ 
/* eslint-disable */
import { type GenMessage } from "@bufbuild/protobuf/codegenv2";
import { ChatSyncMessageSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { EdgeSchema, NodeSchema, NodeTemplateSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { ChatMessageSchema, InferenceConfigDiscoveryResponseSchema, PathUpdateRequestSchema, ReparentNodeRequestSchema, ResetNodeRequestSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { TaskAuditLogSchema, TaskDefinitionSchema, TaskUpdateSchema, WorkerInfoSchema } from "@/generated/flowcraft/v1/core/kernel_pb";
import { ViewportSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { ActionExecutionRequestSchema } from "@/generated/flowcraft/v1/core/action_pb";
import { NodeSignalSchema } from "@/generated/flowcraft/v1/core/signals_pb";

import { type DbConnection } from "./spacetime";

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
  "submitTask": {
      task: { schema: TaskDefinitionSchema } 
  },
  "registerWorker": {
      info: { schema: WorkerInfoSchema } 
  },
  "updateTaskProgress": {
      update: { schema: TaskUpdateSchema } 
  },
  "logTaskEvent": {
      log: { schema: TaskAuditLogSchema } 
  },
  "pathUpdatePb": {
      req: { schema: PathUpdateRequestSchema } 
  },
  "reparentNodePb": {
      req: { schema: ReparentNodeRequestSchema } 
  },
  "addEdgePb": {
      edge: { schema: EdgeSchema } 
  },
  "createNodePb": {
      node: { schema: NodeSchema } 
  },
  "updateViewport": {
      viewport: { schema: ViewportSchema } 
  },
  "resetNode": {
      req: { schema: ResetNodeRequestSchema } 
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
export const TABLE_TO_PROTO: Record<string, { schema: GenMessage<any>, field: string }> = {
  "nodes": { schema: NodeSchema, field: "state" },
  "edges": { schema: EdgeSchema, field: "state" },
  "viewportState": { schema: ViewportSchema, field: "state" },
  "chatMessages": { schema: ChatMessageSchema, field: "state" },
  "nodeTemplates": { schema: NodeTemplateSchema, field: "state" },
  "inferenceConfig": { schema: InferenceConfigDiscoveryResponseSchema, field: "state" } 
} as const;

/**
 * 编译时类型安全断言：确保所有映射的 Reducer 在 SDK 中都存在
 */
type AssertReducersExist = keyof typeof PB_REDUCERS_MAP extends keyof DbConnection["reducers"]
  ? true
  : never;
export const _ASSERT_REDUCERS_SAFE: AssertReducersExist = true;
