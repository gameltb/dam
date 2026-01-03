import React, { memo } from "react";
import { MediaType } from "../../generated/flowcraft/v1/node_pb";
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

    if (!data.media) return null;

    const nodeWidth = width ?? 240;
    const nodeHeight = height ?? 180;

    const handleOpenPreview = (index = 0) => {
      dispatchNodeEvent("open-preview", { nodeId: id, index });
    };

    const handleDimensionsLoad = (ratio: number) => {
      if (
        data.media &&
        Math.abs((data.media.aspectRatio ?? 0) - ratio) > 0.01
      ) {
        onChange(id, {
          media: { ...data.media, aspectRatio: ratio },
        });
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
            Unsupported media: {url}
          </div>
        );
      }

      return (
        <Renderer
          url={url}
          content={content}
          onEdit={(newContent) => {
            if (data.media) {
              onChange(id, {
                media: { ...data.media, content: newContent },
              });
            }
          }}
          index={index}
          onDimensionsLoad={handleDimensionsLoad}
          onOpenPreview={handleOpenPreview}
        />
      );
    };

    const gallery = data.media.galleryUrls ?? [];
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
        {renderContent(
          data.media.url ?? "",
          data.media.type,
          0,
          data.media.content,
        )}
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
              mediaType={data.media.type}
              renderItem={(url) => {
                if (!data.media) return null;
                return (
                  <div
                    style={{
                      width: "100%",
                      height: "100%",
                      borderRadius: "4px",
                      overflow: "hidden",
                    }}
                  >
                    {renderContent(
                      url,
                      data.media.type,
                      gallery.indexOf(url) + 1,
                    )}
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
