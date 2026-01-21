import { create } from "@bufbuild/protobuf";
import { type JsonObject } from "@bufbuild/protobuf";
import { useReactFlow } from "@xyflow/react";
import { useCallback } from "react";
import { useShallow } from "zustand/react/shallow";

import { ActionExecutionRequestSchema, type ActionTemplate } from "@/generated/flowcraft/v1/core/action_pb";
import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import {
  AppNodeType,
  MutationSource,
  type NodeId,
  type NodeTemplate,
  type TaskId,
  TaskStatus,
  type TemplateId,
} from "@/types";
import { getMediaTypeFromMime } from "@/utils/nodeUtils";

import { useNodeOperations } from "./useNodeOperations";

export const useAppActions = (
  setPendingAction: (action: ActionTemplate | null) => void,
  contextMenu: null | { nodeId?: NodeId; x: number; y: number },
  closeContextMenuAndClear: () => void,
) => {
  const { screenToFlowPosition } = useReactFlow();
  const { applyMutations, nodes } = useFlowStore(
    useShallow((s) => ({
      applyMutations: s.applyMutations,
      nodes: s.nodes,
    })),
  );
  const { addNode } = useNodeOperations(applyMutations);

  const handleExecuteAction = useCallback(
    (action: ActionTemplate, params: Record<string, unknown> = {}) => {
      if (action.paramsSchema && Object.keys(params).length === 0) {
        setPendingAction(action);
        closeContextMenuAndClear();
        return;
      }
      const effectiveNodeId = contextMenu?.nodeId ?? nodes.find((n: any) => n.selected)?.id ?? "";
      if (!effectiveNodeId && nodes.filter((n: any) => n.selected).length === 0) return;

      const taskId = crypto.randomUUID() as TaskId;
      useTaskStore.getState().registerTask({
        label: action.label,
        source: MutationSource.SOURCE_REMOTE_TASK,
        taskId,
      });

      const { spacetimeConn } = useFlowStore.getState();
      if (spacetimeConn) {
        try {
          const request = create(ActionExecutionRequestSchema, {
            actionId: action.id,
            contextNodeIds: nodes.filter((n: any) => n.selected).map((n: any) => n.id),
            params: {
              case: "paramsStruct",
              value: params as JsonObject,
            },
            sourceNodeId: effectiveNodeId,
          });

          spacetimeConn.pbreducers.executeAction({
            id: taskId,
            request,
          });
        } catch (err) {
          console.error("[SpacetimeDB] Failed to execute action:", err);
          useTaskStore.getState().updateTask(taskId, {
            message: String(err),
            status: TaskStatus.FAILED,
          });
        }
      }

      setPendingAction(null);
      closeContextMenuAndClear();
    },
    [closeContextMenuAndClear, contextMenu, nodes, setPendingAction],
  );

  const handleAddNode = useCallback(
    (template: NodeTemplate) => {
      if (!contextMenu) return;
      const pos = screenToFlowPosition({ x: contextMenu.x, y: contextMenu.y });

      addNode(AppNodeType.DYNAMIC, pos, {
        ...template.defaultState,
        displayName: template.displayName,
        templateId: template.templateId,
      });
    },
    [addNode, contextMenu, screenToFlowPosition],
  );

  const handleDrop = useCallback(
    (event: React.DragEvent, templates: NodeTemplate[]) => {
      event.preventDefault();
      const dropLogic = async () => {
        const files = event.dataTransfer.files;
        if (files.length === 0) return;
        const position = screenToFlowPosition({
          x: event.clientX,
          y: event.clientY,
        });

        for (const file of Array.from(files)) {
          const formData = new FormData();
          formData.append("file", file);
          const res = await fetch("/api/upload", {
            body: formData,
            method: "POST",
          });
          const asset = (await res.json()) as {
            mimeType: string;
            name: string;
            url: string;
          };

          let tpl = "flowcraft.node.media.document" as TemplateId;
          const mediaType = getMediaTypeFromMime(asset.mimeType);
          if (mediaType === MediaType.MEDIA_IMAGE || mediaType === MediaType.MEDIA_VIDEO) {
            tpl = "flowcraft.node.media.visual" as TemplateId;
          }

          const template = templates.find((t) => t.templateId === tpl);

          const extension: any =
            tpl === "flowcraft.node.media.visual"
              ? { case: "visual", value: { content: "", type: mediaType, url: asset.url } }
              : { case: "document", value: { content: "", type: mediaType, url: asset.url } };

          addNode(AppNodeType.DYNAMIC, position, {
            ...template?.defaultState,
            displayName: asset.name || (template?.displayName ?? "New Asset"),
            extension,
            templateId: tpl,
          });
        }
      };
      void dropLogic().catch((e: unknown) => {
        console.error(e);
      });
    },
    [addNode, screenToFlowPosition],
  );

  return { handleAddNode, handleDrop, handleExecuteAction };
};
