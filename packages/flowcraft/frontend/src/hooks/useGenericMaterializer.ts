import { create as createProto } from "@bufbuild/protobuf";
import { useEffect } from "react";
import { useShallow } from "zustand/react/shallow";

import { NodeKind, PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { NodeSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { AddNodeRequestSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";
import { materializerRegistry } from "@/utils/materializerRegistry";
import { appNodeDataToProto } from "@/utils/nodeProtoUtils";

/**
 * 通用 JIT 物化器
 * 自动识别当前作用域类型，并补全缺失的物理节点
 */
export const useGenericMaterializer = () => {
  const activeScopeId = useUiStore((s) => s.activeScopeId);
  const { allNodes, applyMutations } = useFlowStore(
    useShallow((s) => ({
      allNodes: s.allNodes,
      applyMutations: s.applyMutations,
    })),
  );

  useEffect(() => {
    if (!activeScopeId) return;

    // 1. 寻找匹配的物化配置
    // 这里我们可以根据 activeScopeId 对应的节点类型来查找
    const activeNode = allNodes.find((n) => n.id === activeScopeId);
    if (!activeNode) return;

    const configs = materializerRegistry.getAll();
    const config = configs.find((c) => activeNode.data.templateId?.includes(c.scopeType));

    if (!config) return;

    // 2. 获取逻辑项并识别缺失项
    const allItems = config.getItems(activeScopeId);
    const existingNodesInScope = allNodes.filter((n) => n.parentId === activeScopeId);

    const missingItems = allItems.filter(
      (item) => !existingNodesInScope.some((en) => en.id === config.getItemId(item)),
    );

    if (missingItems.length > 0) {
      console.log(`[Materializer] Scope: ${activeScopeId}, Adding ${missingItems.length} new nodes.`);

      // 3. 执行布局逻辑 (传入现有节点以保持其固定)
      const newNodeConfigs = config.layout(missingItems, existingNodesInScope, activeScopeId);

      const mutations = newNodeConfigs.map((nc) => {
        const node = createProto(NodeSchema, {
          nodeId: nc.id,
          nodeKind: NodeKind.DYNAMIC,
          presentation: createProto(PresentationSchema, {
            isInitialized: true,
            parentId: activeScopeId,
            position: nc.position,
          }),
          state: appNodeDataToProto(nc.data),
          templateId: nc.templateId,
        });

        return createProto(AddNodeRequestSchema, { node });
      });

      applyMutations(mutations, { description: `JIT Materialize: ${config.scopeType}` });
    }
  }, [activeScopeId, allNodes, applyMutations]);
};
