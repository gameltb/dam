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

export interface ActionHandlerContext {
  actionId: string;
  sourceNodeId?: string;
  contextNodeIds?: string[];
  selectedNodeIds: string[];
  params: Record<string, any>;
  taskId: string;
  emitTaskUpdate: (update: any) => void;
  emitMutation: (mutation: any) => void;
  emitStreamChunk: (chunk: any) => void;
}

export type ActionHandler = (ctx: ActionHandlerContext) => Promise<void>;

class NodeRegistryImpl {
  private definitions = new Map<string, NodeDefinition>();
  private globalActions = new Map<string, ActionTemplate>();
  private actionHandlers = new Map<string, ActionHandler>();

  /**
   * 注册一个节点定义 (显式注册函数，代替装饰器以获得更好的类型支持)
   */
  register(def: NodeDefinition) {
    this.definitions.set(def.template.templateId, def);
    return this;
  }

  /**
   * 注册一个全局动作
   */
  registerGlobalAction(template: ActionTemplate, handler?: ActionHandler) {
    this.globalActions.set(template.id, template);
    if (handler) {
      this.actionHandlers.set(template.id, handler);
    }
    return this;
  }

  getTemplates(): NodeTemplate[] {
    return Array.from(this.definitions.values()).map((d) => d.template);
  }

  getActionsForNode(templateId: string): ActionTemplate[] {
    return this.definitions.get(templateId)?.actions || [];
  }

  getGlobalActions(): ActionTemplate[] {
    return Array.from(this.globalActions.values());
  }

  getExecutor(templateId: string) {
    return this.definitions.get(templateId)?.execute;
  }

  getActionHandler(actionId: string) {
    return this.actionHandlers.get(actionId);
  }

  getDefinition(templateId: string) {
    return this.definitions.get(templateId);
  }
}

export const NodeRegistry = new NodeRegistryImpl();
