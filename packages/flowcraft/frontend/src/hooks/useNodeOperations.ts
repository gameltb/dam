import type { XYPosition } from "@xyflow/react";

import { create } from "@bufbuild/protobuf";
import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";

import {
  type GraphMutation,
  GraphMutationSchema,
} from "../generated/flowcraft/v1/core/service_pb";
import { type MutationContext } from "../store/flowStore";
import { type AppNode, AppNodeType, type DynamicNodeData } from "../types";
import { toProtoNode } from "../utils/protoAdapter";

export const useNodeOperations = (
  applyMutations: (
    mutations: GraphMutation[],
    context?: MutationContext,
  ) => void,
) => {
  const addNode = useCallback(
    (
      templateId: string,
      position: XYPosition,
      initialData?: Partial<DynamicNodeData>,
      initialWidth = 300,
      initialHeight = 200,
    ) => {
      const newNode: AppNode = {
        data: {
          label: "Loading...",
          modes: [],
          typeId: templateId,
          ...initialData,
        } as DynamicNodeData,
        id: uuidv4(),
        measured: { height: initialHeight, width: initialWidth },
        position,
        style: { height: initialHeight, width: initialWidth },
        type: AppNodeType.DYNAMIC,
      };

      applyMutations([
        create(GraphMutationSchema, {
          operation: {
            case: "addNode",
            value: { node: toProtoNode(newNode) },
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
