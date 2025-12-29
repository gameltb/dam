import React, { memo } from "react";
import { MediaType } from "../../generated/core/node_pb";
import type { DynamicNodeData } from "../../types";
import { ImageRenderer } from "../media/ImageRenderer";
import { VideoRenderer } from "../media/VideoRenderer";
import { AudioRenderer } from "../media/AudioRenderer";
import { MarkdownRenderer } from "../media/MarkdownRenderer";
import { GalleryWrapper } from "../media/GalleryWrapper";
import { PortHandle } from "../base/PortHandle";
import { useNodeHandlers } from "../../hooks/useNodeHandlers";
import { useFlowStore } from "../../store/flowStore";

import { getPortColor, getPortShape } from "../../utils/themeUtils";

interface MediaContentProps {
  id: string;
  data: DynamicNodeData;
  onOverflowChange?: (o: "visible" | "hidden") => void;
  width?: number;
  height?: number;
}

export const MediaContent: React.FC<MediaContentProps> = memo(
  ({ id, data, onOverflowChange, width, height }) => {
    const { onChange, onGalleryItemContext } = useNodeHandlers();
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

    const renderContent = (url: string, type: MediaType, index = 0) => {
      switch (type) {
        case MediaType.MEDIA_IMAGE:
          return (
            <div
              onDoubleClick={() => {
                handleOpenPreview(index);
              }}
              style={{ width: "100%", height: "100%" }}
            >
              <ImageRenderer
                url={url}
                onDimensionsLoad={handleDimensionsLoad}
              />
            </div>
          );
        case MediaType.MEDIA_VIDEO:
          return (
            <div
              onDoubleClick={() => {
                handleOpenPreview(index);
              }}
              style={{ width: "100%", height: "100%" }}
            >
              <VideoRenderer
                url={url}
                autoPlay
                onDimensionsLoad={handleDimensionsLoad}
              />
            </div>
          );
        case MediaType.MEDIA_AUDIO:
          return (
            <div
              onDoubleClick={() => {
                handleOpenPreview(index);
              }}
              style={{ width: "100%", height: "100%" }}
            >
              <AudioRenderer url={url} />
            </div>
          );
        default:
          return (
            <div style={{ padding: "20px", textAlign: "center" }}>
              Unsupported media: {url}
            </div>
          );
      }
    };

    const gallery = data.media.galleryUrls ?? [];
    const inputs = data.inputPorts ?? [];
    const outputs = data.outputPorts ?? [];

    // --- Layer 1: Core Media Content (Clipped for rounded corners) ---
    const mediaLayer = (
      <div
        style={{
          width: "100%",
          height: "100%",
          borderRadius: "inherit",
          overflow: "hidden",
          position: "relative",
        }}
      >
        {data.media.type === MediaType.MEDIA_MARKDOWN ? (
          <MarkdownRenderer
            content={data.media.content ?? ""}
            onEdit={(newContent) => {
              if (data.media) {
                onChange(id, {
                  media: { ...data.media, content: newContent },
                });
              }
            }}
          />
        ) : (
          renderContent(data.media.url ?? "", data.media.type, 0)
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
          zIndex: 10,
        }}
      >
        {/* Gallery Logic (Floating over content) */}
        {data.media.type !== MediaType.MEDIA_MARKDOWN && (
          <div style={{ pointerEvents: "auto", width: "100%", height: "100%" }}>
            <GalleryWrapper
              id={id}
              nodeWidth={nodeWidth}
              nodeHeight={nodeHeight}
              mainContent={<div style={{ width: "100%", height: "100%" }} />} // Invisible ghost to drive layout
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
        <div style={{ pointerEvents: "auto" }}>
          {outputs.map((port, idx) => (
            <div
              key={port.id || idx}
              style={{
                position: "absolute",
                right: 0,
                top: "50%",
                transform: "translateY(-50%)",
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
        style={{
          position: "relative",
          width: "100%",
          height: "100%",
          borderRadius: "inherit",
          // The base container must remain visible to show the overlay
          overflow: "visible",
        }}
      >
        {mediaLayer}
        {overlayLayer}
      </div>
    );
  },
);
