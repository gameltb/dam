/** AUTO-GENERATED - DO NOT EDIT **/
/* eslint-disable */
import { NodeSchema, EdgeSchema, TaskUpdateSchema, NodeTemplateSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { ViewportSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { InferenceConfigDiscoveryResponseSchema } from "@/generated/flowcraft/v1/core/service_pb";


export const TABLE_TO_PROTO: Record<string, { schema: any, field: string }> = {
  "nodes": { schema: NodeSchema, field: "state" },
  "edges": { schema: EdgeSchema, field: "state" },
  "viewport_state": { schema: ViewportSchema, field: "state" },
  "tasks": { schema: TaskUpdateSchema, field: "result" },
  "node_templates": { schema: NodeTemplateSchema, field: "state" },
  "inference_config": { schema: InferenceConfigDiscoveryResponseSchema, field: "state" }
};
