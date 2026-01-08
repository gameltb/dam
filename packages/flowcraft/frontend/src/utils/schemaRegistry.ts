import type { RJSFSchema } from "@rjsf/utils";

import ImageEnhanceParamsSchema from "@/generated/schemas/flowcraft_proto.v1.ImageEnhanceParams.schema.json";
import NodeTransformParamsSchema from "@/generated/schemas/flowcraft_proto.v1.NodeTransformParams.schema.json";
import PromptGenParamsSchema from "@/generated/schemas/flowcraft_proto.v1.PromptGenParams.schema.json";

/**
 * 静态 Schema 注册表
 * 将 templateId 映射到生成的 JSON Schema
 */
const STATIC_SCHEMAS: Record<string, unknown> = {
  "flowcraft.action.graph.context_enhance": ImageEnhanceParamsSchema,
  "flowcraft.action.graph.context_transform": NodeTransformParamsSchema,
  // 动作参数 Schema
  "flowcraft.action.graph.prompt_to_chat": PromptGenParamsSchema,

  // 节点 Widget Schema
  "flowcraft.node.utility.stream_viewer": {
    properties: {
      agent_name: { default: "Assistant", title: "Agent Name", type: "string" },
      logs: {
        title: "Execution Logs",
        type: "string",
        uiWidget: "streamingText",
      },
      run: {
        title: "Run Command",
        type: "boolean",
        uiWidget: "signalButton",
      },
    },
    type: "object",
  },
};

/**
 * 保留的动态模板 ID
 * 当使用这些 ID 时，系统将从后端负载中读取 Schema 字段
 */
export const DYNAMIC_TEMPLATE_IDS = ["dynamic-node", "dynamic-action"];

/**
 * 获取指定模板的 Schema
 */
export function getSchemaForTemplate(
  templateId: string,
  dynamicSchema?: Record<string, unknown>,
): null | RJSFSchema {
  // 1. 优先查找本地静态注册表
  if (STATIC_SCHEMAS[templateId]) {
    return STATIC_SCHEMAS[templateId] as RJSFSchema;
  }

  // 2. 使用动态 Schema (如果存在)
  if (dynamicSchema) {
    return dynamicSchema as RJSFSchema;
  }

  return null;
}
