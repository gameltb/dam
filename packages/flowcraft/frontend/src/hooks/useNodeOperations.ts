import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import type { AppNode, DynamicNodeData } from "../types";
import type { XYPosition } from "@xyflow/react";
import {
  GraphMutationSchema,
  type GraphMutation,
} from "../generated/flowcraft/v1/core/service_pb";
import { create } from "@bufbuild/protobuf";
import { type MutationContext } from "../store/flowStore";
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
        id: uuidv4(),
        type: "dynamic",
        position,
        measured: { width: initialWidth, height: initialHeight },
        style: { width: initialWidth, height: initialHeight },
        data: {
          label: "Loading...",
          typeId: templateId,
          ...initialData,
        } as any,
      } as AppNode;

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

  return { addNode, deleteNode, deleteEdge };
};
