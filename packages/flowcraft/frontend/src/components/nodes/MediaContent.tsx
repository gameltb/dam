import React, { memo } from "react";
import { MediaType } from "../../generated/flowcraft/v1/core/node_pb";
import {
  GraphMutationSchema,
} from "../../generated/flowcraft/v1/core/service_pb";
import { PresentationSchema } from "../../generated/flowcraft/v1/core/base_pb";
import { create as createProto } from "@bufbuild/protobuf";
import type { DynamicNodeData } from "../../types";
import { GalleryWrapper } from "../media/GalleryWrapper";
import { PortHandle } from "../base/PortHandle";
import { useNodeHandlers } from "../../hooks/useNodeHandlers";
import { useFlowStore } from "../../store/flowStore";

import { getPortColor, getPortShape } from "../../utils/themeUtils";

import { MEDIA_RENDERERS } from "../media/mediaRenderRegistry";

interface MediaContentProps {
  id: string;
  data: DynamicNodeData;
  onOverflowChange?: (o: "visible" | "hidden") => void;
  width?: number;
  height?: number;
}

export const MediaContent: React.FC<MediaContentProps> = memo(
  ({ id, data, onOverflowChange, width, height }) => {
    const { onChange, onGalleryItemContext } = useNodeHandlers(data);
    const dispatchNodeEvent = useFlowStore((state) => state.dispatchNodeEvent);

    const getEffectiveMedia = () => {
      if (data.media?.url) return data.media;

      // Fallback to widgetsValues (e.g. from drag & drop)
      const url = data.widgetsValues?.url as string | undefined;
      const mimeType = data.widgetsValues?.mimeType as string | undefined;
      const content = data.widgetsValues?.content as string | undefined;

      if (!url && !content) return null;

      let type = MediaType.MEDIA_UNSPECIFIED;
      if (mimeType?.startsWith("image/")) type = MediaType.MEDIA_IMAGE;
      else if (mimeType?.startsWith("video/")) type = MediaType.MEDIA_VIDEO;
      else if (mimeType?.startsWith("audio/")) type = MediaType.MEDIA_AUDIO;
      else if (mimeType === "text/markdown") type = MediaType.MEDIA_MARKDOWN;

      return {
        url: url || "",
        type,
        content: content || "",
        aspectRatio: data.media?.aspectRatio ?? 0,
        galleryUrls: [],
      };
    };

    const media = getEffectiveMedia();
    if (!media) return null;

    const nodeWidth = width ?? 240;
    const nodeHeight = height ?? 180;

    const handleOpenPreview = (index = 0) => {
      dispatchNodeEvent("open-preview", { nodeId: id, index });
    };

    const handleDimensionsLoad = (ratio: number) => {
      // 1. Update internal aspect ratio metadata
      if (Math.abs((media.aspectRatio ?? 0) - ratio) > 0.01) {
        onChange(id, {
          media: { ...media, aspectRatio: ratio },
        });

        // 2. Adjust node physical dimensions to match the ratio
        // We keep the width and adjust the height
        const currentWidth = width ?? 240;
        const targetHeight = Math.round(currentWidth / ratio);

        // Only update if the difference is significant to avoid infinite loops
        if (Math.abs((height ?? 0) - targetHeight) > 5) {
          const { applyMutations } = useFlowStore.getState();
          applyMutations([
            createProto(GraphMutationSchema, {
              operation: {
                case: "updateNode",
                value: {
                  id: id,
                  presentation: createProto(PresentationSchema, {
                    width: currentWidth,
                    height: targetHeight,
                    isInitialized: true,
                  }),
                },
              },
            }),
          ]);
        }
      }
    };

    const renderContent = (
      url: string,
      type: MediaType,
      index = 0,
      content?: string,
    ) => {
      const Renderer = MEDIA_RENDERERS[type];
      if (!Renderer) {
        return (
          <div style={{ padding: "20px", textAlign: "center" }}>
            Unsupported media type: {type} for {url}
          </div>
        );
      }

      return (
        <Renderer
          url={url}
          content={content}
          onEdit={(newContent) => {
            onChange(id, {
              widgetsValues: { ...data.widgetsValues, content: newContent },
              media: { ...media, content: newContent },
            });
          }}
          index={index}
          onDimensionsLoad={handleDimensionsLoad}
          onOpenPreview={handleOpenPreview}
        />
      );
    };

    const gallery = media.galleryUrls ?? [];
    const inputs = data.inputPorts ?? [];
    const outputs = data.outputPorts ?? [];

    // --- Layer 1: Core Media Content (Clipped for rounded corners) ---
    const mediaLayer = (
      <div
        className="nopan"
        style={{
          width: "100%",
          height: "100%",
          borderRadius: "inherit",
          overflow: "hidden",
          position: "relative",
          pointerEvents: "auto",
        }}
      >
        {renderContent(media.url || "", media.type, 0, media.content)}
      </div>
    );

    // --- Layer 2: Interaction & Overlay Layer (Visible overflow) ---
    // This layer hosts ports and gallery expansions which must extend beyond node borders.
    const overlayLayer = (
      <div
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none", // Click-through by default
          overflow: "visible",
          zIndex: 100,
        }}
      >
        {/* Gallery Logic (Floating over content) */}
        {gallery.length > 0 && (
          <div style={{ pointerEvents: "none", width: "100%", height: "100%" }}>
            <GalleryWrapper
              id={id}
              nodeWidth={nodeWidth}
              nodeHeight={nodeHeight}
              mainContent={
                <div
                  style={{
                    width: "100%",
                    height: "100%",
                    pointerEvents: "none",
                  }}
                />
              } // Invisible ghost to drive layout
              gallery={gallery}
              mediaType={media.type}
              renderItem={(url) => {
                return (
                  <div
                    style={{
                      width: "100%",
                      height: "100%",
                      borderRadius: "4px",
                      overflow: "hidden",
                    }}
                  >
                    {renderContent(url, media.type, gallery.indexOf(url) + 1)}
                  </div>
                );
              }}
              onGalleryItemContext={(nodeId, url, mType, x, y) => {
                onGalleryItemContext(nodeId, url, mType, x, y);
              }}
              onExpand={
                (expanded) =>
                  onOverflowChange?.(expanded ? "visible" : "visible") // Force parent to stay visible
              }
            />
          </div>
        )}

        {/* Triangle Ports Layer */}
        <div style={{ pointerEvents: "none" }}>
          {outputs.map((port, idx) => (
            <div
              key={port.id || idx}
              style={{
                position: "absolute",
                right: 0,
                top: "50%",
                transform: "translateY(-50%)",
                pointerEvents: "auto",
              }}
            >
              <PortHandle
                nodeId={id}
                portId={port.id}
                type="source"
                style={getPortShape(port.type)}
                color={getPortColor(port.type)}
                isPresentation={true}
              />
            </div>
          ))}
          {inputs.map((port, idx) => (
            <div
              key={port.id || idx}
              style={{
                position: "absolute",
                left: 0,
                top: "50%",
                transform: "translateY(-50%)",
                pointerEvents: "auto",
              }}
            >
              <PortHandle
                nodeId={id}
                portId={port.id}
                type="target"
                style={getPortShape(port.type)}
                color={getPortColor(port.type)}
                isPresentation={true}
              />
            </div>
          ))}
        </div>
      </div>
    );

    return (
      <div
        onDoubleClick={(e) => {
          e.stopPropagation();
          handleOpenPreview(0);
        }}
        style={{
          position: "relative",
          width: "100%",
          height: "100%",
          borderRadius: "inherit",
          // The base container must remain visible to show the overlay
          overflow: "visible",
          pointerEvents: "auto",
        }}
      >
        {/* Top Drag Handle (Invisible Overlay) */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: "30px", // Slightly larger hit area
            zIndex: 50,
            pointerEvents: "auto",
            cursor: "grab",
            borderRadius: "8px 8px 0 0",
          }}
        />
        {mediaLayer}
        {overlayLayer}
      </div>
    );
  },
);
