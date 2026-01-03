import PromptGenParamsSchema from "../generated/schemas/flowcraft.v1.params.PromptGenParams.schema.json";
import ImageEnhanceParamsSchema from "../generated/schemas/flowcraft.v1.params.ImageEnhanceParams.schema.json";
import NodeTransformParamsSchema from "../generated/schemas/flowcraft.v1.params.NodeTransformParams.schema.json";
import type { RJSFSchema } from "@rjsf/utils";

/**
 * 静态 Schema 注册表
 * 将 templateId 映射到生成的 JSON Schema
 */
const STATIC_SCHEMAS: Record<string, any> = {
  // 动作参数 Schema
  "prompt-gen": PromptGenParamsSchema,
  "ai-enhance": ImageEnhanceParamsSchema,
  "ai-transform": NodeTransformParamsSchema,

  // 节点 Widget Schema
  "tpl-stream-node": {
    type: "object",
    properties: {
      agent_name: { type: "string", title: "Agent Name", default: "Assistant" },
      logs: {
        type: "string",
        title: "Execution Logs",
        uiWidget: "streamingText",
      },
      run: {
        type: "boolean",
        title: "Run Command",
        uiWidget: "signalButton",
      },
    },
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
  dynamicSchema?: string,
): RJSFSchema | null {
  // 1. 优先查找本地静态注册表
  if (STATIC_SCHEMAS[templateId]) {
    return STATIC_SCHEMAS[templateId] as RJSFSchema;
  }

  // 2. 如果是特定的动态 ID，则解析传入的动态 Schema
  if (DYNAMIC_TEMPLATE_IDS.includes(templateId) && dynamicSchema) {
    try {
      return JSON.parse(dynamicSchema) as RJSFSchema;
    } catch (e) {
      console.error(`Failed to parse dynamic schema for ${templateId}`, e);
    }
  }

  return null;
}
