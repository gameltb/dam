import React, { memo } from "react";
import { flowcraft_proto } from "../../generated/flowcraft_proto";
import type { DynamicNodeData } from "../../types";
import { ImageRenderer } from "../media/ImageRenderer";
import { VideoRenderer } from "../media/VideoRenderer";
import { AudioRenderer } from "../media/AudioRenderer";
import { MarkdownRenderer } from "../media/MarkdownRenderer";
import { GalleryWrapper } from "../media/GalleryWrapper";
import { useFlowStore } from "../../store/flowStore";

const MediaType = flowcraft_proto.v1.MediaType;

interface MediaContentProps {
  id: string;
  data: DynamicNodeData;
  onOverflowChange?: (o: "visible" | "hidden") => void;
  width?: number;
  height?: number;
}

export const MediaContent: React.FC<MediaContentProps> = memo(
  ({ id, data, onOverflowChange, width, height }) => {
    const dispatchNodeEvent = useFlowStore((state) => state.dispatchNodeEvent);

    if (!data.media) return null;

    const nodeWidth = width ?? 240;
    const nodeHeight = height ?? 180;

    const handleOpenPreview = (index = 0) => {
      dispatchNodeEvent("open-preview", { nodeId: id, index });
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
              <ImageRenderer url={url} />
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
              <VideoRenderer url={url} autoPlay />
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

    if (data.media.type === MediaType.MEDIA_MARKDOWN) {
      return (
        <MarkdownRenderer
          content={data.media.content ?? ""}
          onEdit={(newContent) => {
            if (data.media) {
              data.onChange(id, {
                media: { ...data.media, content: newContent },
              });
            }
          }}
        />
      );
    }

    const gallery = data.media.galleryUrls ?? [];
    return (
      <GalleryWrapper
        id={id}
        nodeWidth={nodeWidth}
        nodeHeight={nodeHeight}
        mainContent={renderContent(data.media.url ?? "", data.media.type, 0)}
        gallery={gallery}
        mediaType={data.media.type}
        renderItem={(url) => {
          if (!data.media) return null;
          return renderContent(url, data.media.type, gallery.indexOf(url) + 1);
        }}
        onGalleryItemContext={(nodeId, url, mediaType, x, y) => {
          data.onGalleryItemContext?.(nodeId, url, mediaType, x, y);
        }}
        onExpand={(expanded) =>
          onOverflowChange?.(expanded ? "visible" : "hidden")
        }
      />
    );
  },
);
