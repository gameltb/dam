import { create as createProto } from "@bufbuild/protobuf";

import { PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { type AppNode, AppNodeType, type Edge, RenderMode } from "@/types";

export function generateGallery(): { edges: Edge[]; nodes: AppNode[] } {
  const nodes: AppNode[] = [
    {
      data: {
        activeMode: RenderMode.MODE_MARKDOWN,
        availableModes: [RenderMode.MODE_MARKDOWN],
        displayName: "Welcome to Flowcraft",
        media: {
          content: "# Welcome\n\nThis is a sample graph showing off the various node types.",
          type: 4, // MARKDOWN
        },
        templateId: "flowcraft.node.media.document",
      } as any,
      id: "welcome-node",
      position: { x: 100, y: 100 },
      presentation: createProto(PresentationSchema, {
        height: 400,
        isInitialized: true,
        position: { x: 100, y: 100 },
        width: 500,
      }),
      type: AppNodeType.DYNAMIC,
    },
  ];

  return { edges: [], nodes };
}
