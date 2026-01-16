import { create } from "@bufbuild/protobuf";
import { type JsonObject } from "@bufbuild/protobuf";
import { useReactFlow } from "@xyflow/react";
import { useCallback } from "react";

import { ActionExecutionRequestSchema, type ActionTemplate } from "@/generated/flowcraft/v1/core/action_pb";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { type AppNode, MutationSource, type NodeTemplate, TaskStatus, type NodeId, type TaskId, type TemplateId } from "@/types";

import { useNodeOperations } from "./useNodeOperations";

export const useAppActions = (
  setPendingAction: (action: ActionTemplate | null) => void,
  contextMenu: null | { nodeId?: NodeId; x: number; y: number },
  closeContextMenuAndClear: () => void,
) => {
  const { screenToFlowPosition } = useReactFlow();
  const { applyMutations, nodes } = useFlowStore();
  const { addNode } = useNodeOperations(applyMutations);

  const handleExecuteAction = useCallback(
    (action: ActionTemplate, params: Record<string, unknown> = {}) => {
      if (action.paramsSchema && Object.keys(params).length === 0) {
        setPendingAction(action);
        closeContextMenuAndClear();
        return;
      }
      const effectiveNodeId = contextMenu?.nodeId ?? nodes.find((n: AppNode) => n.selected)?.id ?? "";
      if (!effectiveNodeId && nodes.filter((n: AppNode) => n.selected).length === 0) return;

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
            contextNodeIds: nodes.filter((n: AppNode) => n.selected).map((n: AppNode) => n.id),
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
            status: TaskStatus.TASK_FAILED,
          });
        }
      } else {
        console.warn("[App] Cannot execute action: No SpacetimeDB connection");
      }

      setPendingAction(null);
      closeContextMenuAndClear();
    },
    [closeContextMenuAndClear, contextMenu, nodes, setPendingAction, applyMutations],
  );

  const handleAddNode = useCallback(
    (template: NodeTemplate) => {
      if (!contextMenu) return;
      const pos = screenToFlowPosition({ x: contextMenu.x, y: contextMenu.y });
      addNode(
        template.templateId as TemplateId,
        pos,
        {
          ...template.defaultState,
          displayName: template.displayName,
          templateId: template.templateId as TemplateId,
        },
        template.defaultWidth,
        template.defaultHeight,
      );
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
          if (asset.mimeType.startsWith("image/")) tpl = "flowcraft.node.media.visual" as TemplateId;
          const template = templates.find((t) => t.templateId === tpl);
          addNode(tpl, position, {
            ...template?.defaultState,
            displayName: asset.name ? asset.name : (template?.displayName ?? "New Asset"),
            templateId: tpl,
            widgetsValues: { mimeType: asset.mimeType, url: asset.url },
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
