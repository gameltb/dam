import { type NodeTemplate } from "../generated/flowcraft/v1/core/node_pb";
import { type ActionTemplate } from "../generated/flowcraft/v1/core/action_pb";
import { type AppNode } from "../types";

export interface NodeExecutionContext {
  node: AppNode;
  params: Record<string, any>;
  taskId: string;
  emitTaskUpdate: (update: any) => void;
  emitMutation: (mutation: any) => void;
  emitStreamChunk: (chunk: any) => void;
}

export interface NodeDefinition {
  template: NodeTemplate;
  actions?: ActionTemplate[];
  // 执行逻辑：当该类型的节点收到执行请求时触发
  execute?: (ctx: NodeExecutionContext) => Promise<void>;
}

class NodeRegistryImpl {
  private definitions = new Map<string, NodeDefinition>();

  register(def: NodeDefinition) {
    this.definitions.set(def.template.templateId, def);
  }

  getTemplates(): NodeTemplate[] {
    return Array.from(this.definitions.values()).map((d) => d.template);
  }

  getActionsForNode(templateId: string): ActionTemplate[] {
    return this.definitions.get(templateId)?.actions || [];
  }

  getExecutor(templateId: string) {
    return this.definitions.get(templateId)?.execute;
  }

  getDefinition(templateId: string) {
    return this.definitions.get(templateId);
  }
}

export const NodeRegistry = new NodeRegistryImpl();
