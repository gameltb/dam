import { create as createProto } from "@bufbuild/protobuf";
import { useCallback } from "react";
import { toast } from "react-hot-toast";

import type { ActionTemplate } from "@/generated/flowcraft/v1/core/action_pb";
import type { NodeTemplate } from "@/types";

import { PositionSchema, PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { NodeDataSchema, RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { type ContextMenuData } from "@/hooks/nodes/useNodeEventListener";
import { useFlowStore } from "@/store/flowStore";
import { useNavigationStore } from "@/store/ui/navigationStore";
import { AppNodeType, Scope } from "@/types";

export const useAppActions = (
  setPendingAction: (a: null | { actionId: string; nodeId: string }) => void,
  contextMenu: ContextMenuData | null,
  closeContextMenu: () => void,
) => {
  const { addNode } = useFlowStore();
  const activeScopeId = useNavigationStore((s) => s.activeScopeId);

  const handleAddNode = useCallback(
    (t: NodeTemplate) => {
      const nodeId = crypto.randomUUID();
      const { activeGraphId } = useFlowStore.getState();

      const defaultData = t.defaultState ? { ...t.defaultState } : {};

      const nodeData = {
        ...createProto(NodeDataSchema, {
          activeMode: RenderMode.MODE_MEDIA,
          availableModes: [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS],
          displayName: t.displayName || `New ${t.templateId}`,
          ...defaultData,
        }),
        templateId: t.templateId, // MUST set this for syncMiddleware
      };

      const position = { x: 100, y: 100 };

      addNode({
        data: nodeData as any,
        graphId: activeGraphId || "default",
        height: t.defaultHeight || 200,
        id: nodeId,
        parentId: activeScopeId || undefined,
        position,
        presentation: createProto(PresentationSchema, {
          height: t.defaultHeight || 200,
          parentId: activeScopeId ?? "",
          position: createProto(PositionSchema, position),
          scopeId: activeScopeId || Scope.ROOT,
          width: t.defaultWidth || 300,
        }),
        scopeId: activeScopeId || Scope.ROOT,
        type: AppNodeType.DYNAMIC,
        width: t.defaultWidth || 300,
      });

      closeContextMenu();
    },
    [activeScopeId, closeContextMenu, addNode],
  );

  const handleExecuteAction = useCallback(
    (_action: ActionTemplate, _params?: Record<string, unknown>) => {
      if (contextMenu?.nodeId) {
        setPendingAction({ actionId: _action.id, nodeId: contextMenu.nodeId });
        closeContextMenu();
      }
    },
    [setPendingAction, closeContextMenu, contextMenu],
  );

  const exportBranch = useCallback(() => {
    if (!contextMenu?.nodeId) return;
    toast.success("Branch export started (legacy logic to be updated)");
    closeContextMenu();
  }, [contextMenu, closeContextMenu]);

  return { exportBranch, handleAddNode, handleExecuteAction };
};
