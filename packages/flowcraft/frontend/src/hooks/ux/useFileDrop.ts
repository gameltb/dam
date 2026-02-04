import { create } from "@bufbuild/protobuf";
import { useReactFlow } from "@xyflow/react";
import { useCallback } from "react";
import { toast } from "react-hot-toast";

import { NodeDataSchema, RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { VisualNodeStateSchema } from "@/generated/flowcraft/v1/nodes/media_node_pb";
import { useFlowStore } from "@/store/flowStore";

/**
 * Hook to handle file drag-and-drop onto the canvas.
 * Automatically uploads files and creates corresponding media nodes.
 */
export const useFileDrop = () => {
  const { screenToFlowPosition } = useReactFlow();
  const spacetimeConn = useFlowStore((s) => s.spacetimeConn);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();

      const files = Array.from(e.dataTransfer.files);
      if (files.length === 0) return;

      const position = screenToFlowPosition({
        x: e.clientX,
        y: e.clientY,
      });

      const uploadPromises = files.map(async (file) => {
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch("/api/upload", { body: formData, method: "POST" });
        if (!res.ok) throw new Error(`Upload failed for ${file.name}`);
        return {
          mimeType: file.type,
          name: file.name,
          url: (await res.json()).url,
        };
      });

      try {
        const results = await Promise.all(uploadPromises);
        if (!spacetimeConn) return;

        // Logic:
        // If multiple files, create ONE node with a gallery (primary = first file)
        // If single file, create ONE node.

        const primary = results[0]!;
        const galleryUrls = results.length > 1 ? results.map((r) => r.url) : [];

        const nodeId = crypto.randomUUID();

        // Create specialized Visual Node state
        const visualState = create(VisualNodeStateSchema, {
          mimeType: primary.mimeType,
          url: primary.url,
        });

        const nodeData = create(NodeDataSchema, {
          activeMode: RenderMode.MODE_MEDIA,
          availableModes: [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS],
          displayName: primary.name,
          extension: {
            case: "visual",
            value: visualState,
          },
          media: {
            aspectRatio: 1.33,
            content: "",
            galleryUrls: galleryUrls,
            type: primary.mimeType.startsWith("video/") ? 2 : 1, // MEDIA_VIDEO or MEDIA_IMAGE
            url: primary.url,
          },
        });

        // Call reducer to create node in SpacetimeDB
        spacetimeConn.pbreducers.createNodePb({
          node: {
            nodeId,
            nodeKind: 1, // DYNAMIC
            presentation: {
              height: 225,
              isInitialized: true,
              position: { x: position.x, y: position.y },
              width: 300,
            },
            state: nodeData,
            templateId: VisualNodeStateSchema.typeName,
          },
        } as any);

        toast.success(`Created node with ${files.length} items.`);
      } catch (err) {
        console.error(err);
        toast.error("Failed to process dropped files.");
      }
    },
    [screenToFlowPosition, spacetimeConn],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  return { handleDragOver, handleDrop };
};
