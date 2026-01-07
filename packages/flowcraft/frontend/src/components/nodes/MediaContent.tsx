import { create as createProto } from "@bufbuild/protobuf";
import React, { memo } from "react";

import { PresentationSchema } from "../../generated/flowcraft/v1/core/base_pb";
import { MediaType } from "../../generated/flowcraft/v1/core/node_pb";
import { GraphMutationSchema } from "../../generated/flowcraft/v1/core/service_pb";
import { useNodeHandlers } from "../../hooks/useNodeHandlers";
import { useFlowStore } from "../../store/flowStore";
import { type DynamicNodeData, FlowEvent } from "../../types";
import { getMediaTypeFromMime } from "../../utils/nodeUtils";
import { getPortColor, getPortShape } from "../../utils/themeUtils";
import { PortHandle } from "../base/PortHandle";
import { GalleryWrapper } from "../media/GalleryWrapper";
import { MEDIA_RENDERERS } from "../media/mediaRenderRegistry";

interface MediaContentProps {
  data: DynamicNodeData;
  height?: number;
  id: string;
  onOverflowChange?: (o: "hidden" | "visible") => void;
  width?: number;
}

export const MediaContent: React.FC<MediaContentProps> = memo(
  ({ data, height, id, onOverflowChange, width }) => {
    const { onChange, onGalleryItemContext } = useNodeHandlers(data);
    const dispatchNodeEvent = useFlowStore((state) => state.dispatchNodeEvent);

    const getEffectiveMedia = () => {
      if (data.media?.url) return data.media;

      // Fallback to widgetsValues (e.g. from drag & drop)
      const url = data.widgetsValues?.url as string | undefined;
      const mimeType = data.widgetsValues?.mimeType as string | undefined;
      const content = data.widgetsValues?.content as string | undefined;

      if (!url && !content) return null;

      return {
        aspectRatio: data.media?.aspectRatio ?? 0,
        content: content ?? "",
        galleryUrls: [],
        type: getMediaTypeFromMime(mimeType),
        url: url ?? "",
      };
    };

    const media = getEffectiveMedia();
    if (!media) return null;

    const nodeWidth = width ?? 240;
    const nodeHeight = height ?? 180;

    const handleOpenPreview = (index = 0) => {
      dispatchNodeEvent(FlowEvent.OPEN_PREVIEW, { index, nodeId: id });
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
                    height: targetHeight,
                    isInitialized: true,
                    width: currentWidth,
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
          content={content}
          index={index}
          onDimensionsLoad={handleDimensionsLoad}
          onEdit={(newContent) => {
            onChange(id, {
              media: { ...media, content: newContent },
              widgetsValues: { ...data.widgetsValues, content: newContent },
            });
          }}
          onOpenPreview={handleOpenPreview}
          url={url}
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
          borderRadius: "inherit",
          height: "100%",
          overflow: "hidden",
          pointerEvents: "auto",
          position: "relative",
          width: "100%",
        }}
      >
        {renderContent(media.url ?? "", media.type, 0, media.content)}
      </div>
    );

    // --- Layer 2: Interaction & Overlay Layer (Visible overflow) ---
    // This layer hosts ports and gallery expansions which must extend beyond node borders.
    const overlayLayer = (
      <div
        style={{
          inset: 0,
          overflow: "visible",
          pointerEvents: "none", // Click-through by default
          position: "absolute",
          zIndex: 100,
        }}
      >
        {/* Gallery Logic (Floating over content) */}
        {gallery.length > 0 && (
          <div style={{ height: "100%", pointerEvents: "none", width: "100%" }}>
            <GalleryWrapper
              gallery={gallery}
              id={id}
              mainContent={
                <div
                  style={{
                    height: "100%",
                    pointerEvents: "none",
                    width: "100%",
                  }}
                />
              } // Invisible ghost to drive layout
              mediaType={media.type}
              nodeHeight={nodeHeight}
              nodeWidth={nodeWidth}
              onExpand={
                (expanded) =>
                  onOverflowChange?.(expanded ? "visible" : "visible") // Force parent to stay visible
              }
              onGalleryItemContext={(nodeId, url, mType, x, y) => {
                onGalleryItemContext(nodeId, url, mType, x, y);
              }}
              renderItem={(url) => {
                return (
                  <div
                    style={{
                      borderRadius: "4px",
                      height: "100%",
                      overflow: "hidden",
                      width: "100%",
                    }}
                  >
                    {renderContent(url, media.type, gallery.indexOf(url) + 1)}
                  </div>
                );
              }}
            />
          </div>
        )}

        {/* Triangle Ports Layer */}
        <div style={{ pointerEvents: "none" }}>
          {outputs.map((port, idx) => (
            <div
              key={port.id || idx}
              style={{
                pointerEvents: "auto",
                position: "absolute",
                right: 0,
                top: "50%",
                transform: "translateY(-50%)",
              }}
            >
              <PortHandle
                color={getPortColor(port.type)}
                isPresentation={true}
                nodeId={id}
                portId={port.id}
                style={getPortShape(port.type)}
                type="source"
              />
            </div>
          ))}
          {inputs.map((port, idx) => (
            <div
              key={port.id || idx}
              style={{
                left: 0,
                pointerEvents: "auto",
                position: "absolute",
                top: "50%",
                transform: "translateY(-50%)",
              }}
            >
              <PortHandle
                color={getPortColor(port.type)}
                isPresentation={true}
                nodeId={id}
                portId={port.id}
                style={getPortShape(port.type)}
                type="target"
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
          borderRadius: "inherit",
          height: "100%",
          // The base container must remain visible to show the overlay
          overflow: "visible",
          pointerEvents: "auto",
          position: "relative",
          width: "100%",
        }}
      >
        {/* Top Drag Handle (Invisible Overlay) */}
        <div
          style={{
            borderRadius: "8px 8px 0 0",
            cursor: "grab",
            height: "30px", // Slightly larger hit area
            left: 0,
            pointerEvents: "auto",
            position: "absolute",
            right: 0,
            top: 0,
            zIndex: 50,
          }}
        />
        {mediaLayer}
        {overlayLayer}
      </div>
    );
  },
);
