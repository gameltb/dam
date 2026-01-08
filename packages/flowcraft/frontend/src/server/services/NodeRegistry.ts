import {
  type ActionExecutionRequest,
  type ActionTemplate,
} from "@/generated/flowcraft/v1/core/action_pb";
import {
  type NodeTemplate,
  type TaskUpdate,
} from "@/generated/flowcraft/v1/core/node_pb";
import {
  type MutationList,
  type NodeEvent,
} from "@/generated/flowcraft/v1/core/service_pb";
import { type NodeSignal } from "@/generated/flowcraft/v1/core/signals_pb";
import { type AppNode } from "@/types";

export type ActionHandler = (ctx: ActionHandlerContext) => Promise<void>;

export interface ActionHandlerContext {
  actionId: string;
  contextNodeIds?: string[];
  emitMutation: (mutation: MutationList) => void;
  emitNodeEvent: (event: NodeEvent) => void;
  emitTaskUpdate: (update: TaskUpdate) => void;
  node?: AppNode;
  params: ActionExecutionRequest["params"];
  selectedNodeIds: string[];
  sourceNodeId?: string;
  taskId: string;
}

import { NodeInstance } from "./NodeInstance";
// ... (imports)
export interface NodeDefinition {
  actions?: ActionTemplate[];
  /**
   * Optional: Create a persistent instance for this node.
   */
  createInstance?: (nodeId: string) => NodeInstance;
  execute?: (ctx: NodeExecutionContext) => Promise<void>;
  template: NodeTemplate;
}

export interface NodeExecutionContext {
  actionId?: string;
  emitMutation: (mutation: MutationList) => void;
  emitNodeEvent: (event: NodeEvent) => void;
  emitTaskUpdate: (update: TaskUpdate) => void;
  emitWidgetStream: (widgetId: string, chunk: string, isDone?: boolean) => void;
  node: AppNode;
  params: ActionExecutionRequest["params"] | NodeSignal["payload"];
  taskId: string;
}

class NodeRegistryImpl {
  private actionHandlers = new Map<string, ActionHandler>();
  private definitions = new Map<string, NodeDefinition>();
  private globalActions = new Map<string, ActionTemplate>();

  getActionHandler(actionId: string) {
    return this.actionHandlers.get(actionId);
  }

  getActionsForNode(templateId: string): ActionTemplate[] {
    return this.definitions.get(templateId)?.actions ?? [];
  }

  getDefinition(templateId: string) {
    return this.definitions.get(templateId);
  }

  getExecutor(templateId: string) {
    return this.definitions.get(templateId)?.execute;
  }

  getGlobalActions(): ActionTemplate[] {
    return Array.from(this.globalActions.values());
  }

  getTemplates(): NodeTemplate[] {
    return Array.from(this.definitions.values()).map((d) => d.template);
  }

  /**
   * 注册一个节点定义
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
}

export const NodeRegistry = new NodeRegistryImpl();
