import { create as createProto } from "@bufbuild/protobuf";

import { PositionSchema, PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { type AppNode, AppNodeType, type Edge } from "@/types";

export const Generators = {
  createPlaceholderNode: (id: string, graphId: string, scopeId = "root"): AppNode => {
    return {
      data: {
        availableModes: [],
        displayName: "Generating…",
      } as any,
      graphId,
      height: 200,
      id,
      position: { x: 0, y: 0 },
      presentation: createProto(PresentationSchema, {
        height: 200,
        position: createProto(PositionSchema, { x: 0, y: 0 }),
        scopeId,
        width: 300,
      }),
      scopeId,
      type: AppNodeType.DYNAMIC,
      width: 300,
    };
  },

  generateGallery: (): { edges: Edge[]; nodes: AppNode[] } => {
    console.log("[Generators] Generating gallery showcase...");
    return {
      edges: [],
      nodes: [],
    };
  },
};

export const generateGallery = Generators.generateGallery;
