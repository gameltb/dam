import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import type { AppNode, DynamicNodeData } from "../types";
import type { XYPosition } from "@xyflow/react";
import {
  GraphMutationSchema,
  type GraphMutation,
} from "../generated/core/service_pb";
import { type Node as ProtoNode } from "../generated/core/node_pb";
import { create } from "@bufbuild/protobuf";
import { type MutationContext } from "../store/flowStore";

export const useNodeOperations = (
  applyMutations: (
    mutations: GraphMutation[],
    context?: MutationContext,
  ) => void,
) => {
  const addNode = useCallback(
    (
      type: string,
      data: Partial<AppNode["data"]>,
      position: XYPosition,
      typeId?: string,
      initialWidth = 300,
      initialHeight = 200,
    ) => {
      const dynamicData = data as DynamicNodeData | undefined;

      const newNode: AppNode = {
        id: uuidv4(),
        type,
        position,
        measured: { width: initialWidth, height: initialHeight },
        style: { width: initialWidth, height: initialHeight },
        data: {
          label: "New Node",
          modes: [],
          ...data,
          typeId: typeId ?? dynamicData?.typeId,
        },
      } as AppNode;

      applyMutations([
        create(GraphMutationSchema, {
          operation: {
            case: "addNode",
            value: { node: newNode as unknown as ProtoNode },
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
