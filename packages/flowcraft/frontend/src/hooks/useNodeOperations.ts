import { create as createProto } from "@bufbuild/protobuf";
import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";

import { PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import {
  AddNodeRequestSchema,
  RemoveEdgeRequestSchema,
  RemoveNodeRequestSchema,
} from "@/generated/flowcraft/v1/core/service_pb";
import { type MutationContext, type MutationInput } from "@/store/types";
import { type AppNode, AppNodeType } from "@/types";
import { appNodeToProto } from "@/utils/nodeProtoUtils";

export function useNodeOperations(applyMutations: (mutations: MutationInput[], context?: MutationContext) => void) {
  const addNode = useCallback(
    (type: AppNodeType, position: { x: number; y: number }, data: any = {}) => {
      const id = uuidv4();
      const newNode: AppNode = {
        data,
        height: 400,
        id,
        position,
        presentation: createProto(PresentationSchema, {
          height: 400,
          isInitialized: true,
          position: { x: position.x, y: position.y },
          width: 500,
        }),
        type,
        width: 500,
      };

      // 必须使用 createProto 确保它是一个真正的 Message 对象
      applyMutations([createProto(AddNodeRequestSchema, { node: appNodeToProto(newNode) })]);
    },
    [applyMutations],
  );

  const deleteNode = useCallback(
    (id: string) => {
      applyMutations([createProto(RemoveNodeRequestSchema, { id })]);
    },
    [applyMutations],
  );

  const deleteEdge = useCallback(
    (id: string) => {
      applyMutations([createProto(RemoveEdgeRequestSchema, { id })]);
    },
    [applyMutations],
  );

  return { addNode, deleteEdge, deleteNode };
}
