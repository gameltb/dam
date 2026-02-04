import { create as createProto } from "@bufbuild/protobuf";
import { useCallback } from "react";
import { toast } from "react-hot-toast";

import type { ActionTemplate } from "@/generated/flowcraft/v1/core/action_pb";
import type { NodeTemplate } from "@/types";

import { PositionSchema, PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { NodeDataSchema, RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { useFlowStore } from "@/store/flowStore";
import { useNavigationStore } from "@/store/ui/navigationStore";
import { AppNodeType, Scope } from "@/types";

export const useAppActions = (
  setPendingAction: (a: any | null) => void,
  contextMenu: any,
  closeContextMenu: () => void,
) => {
  const { addNode } = useFlowStore();
  const activeScopeId = useNavigationStore((s) => s.activeScopeId);

  const handleAddNode = useCallback(
    (t: NodeTemplate) => {
      const nodeId = crypto.randomUUID();

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
        height: t.defaultHeight || 200,
        id: nodeId,
        parentId: activeScopeId || undefined,
        scopeId: activeScopeId || Scope.ROOT,
        position,
        presentation: createProto(PresentationSchema, {
          height: t.defaultHeight || 200,
          parentId: activeScopeId || "",
          scopeId: activeScopeId || Scope.ROOT,
          position: createProto(PositionSchema, position),
          width: t.defaultWidth || 300,
        }),
        type: AppNodeType.DYNAMIC,
        width: t.defaultWidth || 300,
      });

      closeContextMenu();
    },
    [activeScopeId, closeContextMenu, addNode],
  );

  const handleExecuteAction = useCallback(
    (action: ActionTemplate, _params?: any) => {
      if (contextMenu?.node?.id) {
        setPendingAction({ actionId: action.id, nodeId: contextMenu.node.id });
      }
      closeContextMenu();
    },
    [setPendingAction, closeContextMenu, contextMenu],
  );

  const exportBranch = useCallback(() => {
    if (!contextMenu?.node) return;
    toast.success("Branch export started (legacy logic to be updated)");
    closeContextMenu();
  }, [contextMenu, closeContextMenu]);

  return { exportBranch, handleAddNode, handleExecuteAction };
};
