import { create as createProto } from "@bufbuild/protobuf";
import { PositionSchema, PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { AppNodeType, type AppNode, type Edge } from "@/types";

export const Generators = {
  createPlaceholderNode: (id: string, scopeId: string = "root"): AppNode => {
    return {
      data: {
        displayName: "Generating…",
        availableModes: [],
      } as any,
      height: 200,
      id,
      position: { x: 0, y: 0 },
      scopeId,
      presentation: createProto(PresentationSchema, {
        height: 200,
        position: createProto(PositionSchema, { x: 0, y: 0 }),
        width: 300,
        scopeId,
      }),
      type: AppNodeType.DYNAMIC,
      width: 300,
    };
  },

  generateGallery: (): { nodes: AppNode[], edges: Edge[] } => {
    console.log("[Generators] Generating gallery showcase...");
    return {
      nodes: [],
      edges: [],
    };
  }
};

export const generateGallery = Generators.generateGallery;
