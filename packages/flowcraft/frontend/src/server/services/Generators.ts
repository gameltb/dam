import { type AppNode, AppNodeType, type Edge, RenderMode } from "@/types";

export function generateGallery(): { edges: Edge[]; nodes: AppNode[] } {
  const nodes: AppNode[] = [
    {
      data: {
        activeMode: RenderMode.MODE_MARKDOWN,
        label: "Welcome to Flowcraft",
        media: {
          content:
            "# Welcome\n\nThis is a sample graph showing off the various node types.",
          type: 4, // MARKDOWN
        },
        modes: [RenderMode.MODE_MARKDOWN],
        typeId: "flowcraft.node.media.document",
      },
      id: "welcome-node",
      position: { x: 100, y: 100 },
      type: AppNodeType.DYNAMIC,
    },
  ];

  return { edges: [], nodes };
}
