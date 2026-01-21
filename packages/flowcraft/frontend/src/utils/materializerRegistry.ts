import { type XYPosition } from "@xyflow/react";
import dagre from "dagre";

import { type AppNode } from "@/types";

/**
 * 物化配置接口
 */
export interface MaterializerConfig<T = any> {
  getItemId: (item: T) => string;
  getItemParentId?: (item: T) => null | string;
  getItems: (activeScopeId: string) => T[];
  layout: (
    items: T[],
    existingNodes: AppNode[],
    activeScopeId: string,
  ) => {
    data: any;
    id: string;
    position: XYPosition;
    templateId: string;
  }[];
  scopeType: string;
}

class MaterializerRegistry {
  private configs = new Map<string, MaterializerConfig>();
  getAll() {
    return Array.from(this.configs.values());
  }
  register(config: MaterializerConfig) {
    this.configs.set(config.scopeType, config);
  }
}

export const materializerRegistry = new MaterializerRegistry();

/**
 * 默认排布逻辑 (Dagre 自动布局)
 */
export const dagreLayout = (
  newItems: any[],
  existingNodes: AppNode[],
  config: {
    getItemId: (item: any) => string;
    getItemParentId: (item: any) => null | string;
    getTemplateId: (item: any) => string;
    mapData: (item: any) => any;
    nodeHeight: number;
    nodeWidth: number;
    rankdir?: "LR" | "TB";
  },
) => {
  const g = new dagre.graphlib.Graph();
  g.setGraph({
    nodesep: 100,
    rankdir: config.rankdir || "LR",
    ranksep: 200,
  });
  g.setDefaultEdgeLabel(() => ({}));

  // 1. 添加所有节点（包括现有节点作为锚点）
  existingNodes.forEach((n) => {
    g.setNode(n.id, { height: n.measured?.height || config.nodeHeight, width: n.measured?.width || config.nodeWidth });
  });

  newItems.forEach((item) => {
    g.setNode(config.getItemId(item), { height: config.nodeHeight, width: config.nodeWidth });
  });

  // 2. 构建边关系（仅用于计算位置）
  // 假设新项之间或与旧项之间存在父子引用关系
  [...newItems, ...existingNodes].forEach((nodeOrItem) => {
    const id = "id" in nodeOrItem ? nodeOrItem.id : config.getItemId(nodeOrItem);
    const pId = "parentId" in nodeOrItem ? nodeOrItem.parentId : config.getItemParentId(nodeOrItem);
    if (pId && g.hasNode(pId)) {
      g.setEdge(pId, id);
    }
  });

  dagre.layout(g);

  // 3. 返回新节点的计算位置
  return newItems.map((item) => {
    const id = config.getItemId(item);
    const node = g.node(id);
    return {
      data: config.mapData(item),
      id,
      position: {
        x: node.x - config.nodeWidth / 2,
        y: node.y - config.nodeHeight / 2,
      },
      templateId: config.getTemplateId(item),
    };
  });
};
