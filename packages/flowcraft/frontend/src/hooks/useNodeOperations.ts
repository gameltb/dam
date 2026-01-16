import type { XYPosition } from "@xyflow/react";

import { create } from "@bufbuild/protobuf";
import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";

import { type GraphMutation, GraphMutationSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { type MutationContext } from "@/store/types";
import { type AppNode, AppNodeType, type NodeId, type DynamicNodeData } from "@/types";
import { appNodeToProto } from "@/utils/nodeProtoUtils";
import { mapToRenderMode } from "@/utils/nodeUtils";

export const useNodeOperations = (applyMutations: (mutations: GraphMutation[], context?: MutationContext) => void) => {
  const addNode = useCallback(
    (
      _templateId: string,
      position: XYPosition,
      initialData?: Partial<DynamicNodeData>,
      initialWidth = 300,
      initialHeight = 200,
    ) => {
      const cleanedInitialData = { ...initialData };
      if (cleanedInitialData.activeMode !== undefined) {
        cleanedInitialData.activeMode = mapToRenderMode(cleanedInitialData.activeMode);
      }
      if (cleanedInitialData.availableModes !== undefined && Array.isArray(cleanedInitialData.availableModes)) {
        cleanedInitialData.availableModes = cleanedInitialData.availableModes.map(mapToRenderMode);
      }

      const newNode: AppNode = {
        data: {
          ...cleanedInitialData,
          displayName: initialData?.displayName ?? "New Node",
        } as DynamicNodeData,
        id: uuidv4() as NodeId,
        measured: { height: initialHeight, width: initialWidth },
        position,
        type: AppNodeType.DYNAMIC,
      };

      // Ensure style is also set for immediate local feedback if needed
      (newNode as any).style = { height: initialHeight, width: initialWidth };

      applyMutations([
        create(GraphMutationSchema, {
          operation: {
            case: "addNode",
            value: { node: appNodeToProto(newNode) },
          },
        }),
      ]);
    },
    [applyMutations],
  );

  const deleteNode = useCallback(
    (nodeId: string) => {
      applyMutations([
        create(GraphMutationSchema, {
          operation: {
            case: "removeNode",
            value: { id: nodeId },
          },
        }),
      ]);
    },
    [applyMutations],
  );

  const deleteEdge = useCallback(
    (edgeId: string) => {
      applyMutations([
        create(GraphMutationSchema, {
          operation: {
            case: "removeEdge",
            value: { id: edgeId },
          },
        }),
      ]);
    },
    [applyMutations],
  );

  return { addNode, deleteEdge, deleteNode };
};
