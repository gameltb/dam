import React, { memo } from "react";
import { useShallow } from "zustand/react/shallow";

import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { useNodeHandlers } from "@/hooks/useNodeHandlers";
import { useFlowStore } from "@/store/flowStore";
import { type DynamicNodeData, FlowEvent } from "@/types";
import { getPortColor, getPortShape } from "@/utils/themeUtils";

import { PortHandle } from "../base/PortHandle";
import { GalleryWrapper } from "../media/GalleryWrapper";
import { MEDIA_CONFIGS } from "../media/mediaConfigs";
import { MEDIA_RENDERERS } from "../media/mediaRenderRegistry";

interface MediaContentProps {
  data: DynamicNodeData;
  height?: number;
  id: string;
  onOverflowChange?: (o: any) => void;
  width?: number;
}

const MediaContentComponent: React.FC<MediaContentProps> = memo(({ data, height, id, onOverflowChange, width }) => {
  const { onChange, onGalleryItemContext } = useNodeHandlers(data);
  const { allNodes, dispatchNodeEvent, nodeDraft } = useFlowStore(
    useShallow((s) => ({
      allNodes: s.allNodes,
      dispatchNodeEvent: s.dispatchNodeEvent,
      nodeDraft: s.nodeDraft,
    })),
  );

  const getNormalizedMedia = () => {
    let type = MediaType.MEDIA_UNSPECIFIED;
    let url = "";
    let content = "";
    let aspectRatio = 1.33;
    let galleryUrls: string[] = [];

    if (data.extension?.case === "visual") {
      const v = data.extension.value;
      type = v.mimeType.startsWith("video/") ? MediaType.MEDIA_VIDEO : MediaType.MEDIA_IMAGE;
      url = v.url;
      aspectRatio = 1.33;
    } else if (data.extension?.case === "document") {
      const d = data.extension.value;
      type = MediaType.MEDIA_MARKDOWN;
      content = d.content;
    } else if (data.extension?.case === "acoustic") {
      const a = data.extension.value;
      type = MediaType.MEDIA_AUDIO;
      url = a.url;
    } else if (data.media) {
      type = data.media.type;
      url = data.media.url;
      content = data.media.content;
      aspectRatio = data.media.aspectRatio;
      galleryUrls = data.media.galleryUrls;
    }

    if (type === MediaType.MEDIA_UNSPECIFIED && !url && !content) return null;
    return { aspectRatio, content, galleryUrls, type, url };
  };

  const media = getNormalizedMedia();
  if (!media) {
    return (
      <div className="flex items-center justify-center h-full w-full bg-muted/20 text-[10px] text-muted-foreground uppercase font-bold">
        No Media Data
      </div>
    );
  }

  const nodeWidth = width ?? 240;
  const nodeHeight = height ?? 180;

  const handleOpenPreview = (index = 0) => {
    dispatchNodeEvent(FlowEvent.OPEN_PREVIEW, { index, nodeId: id });
  };

  const handleDimensionsLoad = (ratio: number) => {
    if (Math.abs((media.aspectRatio ?? 0) - ratio) > 0.01) {
      onChange(id, {
        media: { ...media, aspectRatio: ratio } as any,
      });

      const currentWidth = width ?? 240;
      const targetHeight = Math.round(currentWidth / ratio);

      if (Math.abs((height ?? 0) - targetHeight) > 5) {
        const node = allNodes.find((n) => n.id === id);
        if (node) {
          const res = nodeDraft(node);
          if (res.ok) {
            const draft = res.value;
            draft.height = targetHeight;
            draft.width = currentWidth;
          }
        }
      }
    }
  };

  const renderContent = (url: string, type: MediaType, index = 0, content?: string) => {
    const mediaType = type;

    if (mediaType === MediaType.MEDIA_UNSPECIFIED) {
      throw new Error(`[MediaContent] Unspecified media type for node ${id}`);
    }

    const Renderer = MEDIA_RENDERERS[mediaType];
    if (!Renderer) {
      throw new Error(`[MediaContent] No renderer for MediaType: ${mediaType}`);
    }

    return (
      <Renderer
        content={content}
        index={index}
        onDimensionsLoad={handleDimensionsLoad}
        onEdit={(newContent: string) => {
          if (data.extension?.case === "document") {
            onChange(id, {
              extension: {
                case: "document",
                value: { ...data.extension.value, content: newContent },
              },
            });
          }
        }}
        onOpenPreview={handleOpenPreview}
        url={url}
      />
    );
  };

  return (
    <div
      className="relative h-full w-full overflow-visible"
      onDoubleClick={(e) => {
        e.stopPropagation();
        handleOpenPreview(0);
      }}
    >
      <div className="nopan relative h-full w-full overflow-hidden rounded-[inherit] pointer-events-auto">
        {renderContent(media.url, media.type, 0, media.content)}
      </div>

      <div className="absolute inset-0 overflow-visible pointer-events-none z-[100]">
        {media.galleryUrls.length > 0 && (
          <GalleryWrapper
            gallery={media.galleryUrls}
            id={id}
            mainContent={<div className="h-full w-full pointer-events-none" />}
            mediaType={media.type}
            nodeHeight={nodeHeight}
            nodeWidth={nodeWidth}
            onExpand={() => onOverflowChange?.("visible")}
            onGalleryItemContext={onGalleryItemContext}
            renderItem={(url) => (
              <div className="h-full w-full overflow-hidden rounded-sm">
                {renderContent(url, media.type, media.galleryUrls.indexOf(url) + 1)}
              </div>
            )}
          />
        )}

        <div className="pointer-events-none">
          {data.outputPorts?.map((port: any, idx: number) => (
            <div className="absolute right-0 top-1/2 -translate-y-1/2 pointer-events-auto" key={port.id || idx}>
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
          {data.inputPorts?.map((port: any, idx: number) => (
            <div className="absolute left-0 top-1/2 -translate-y-1/2 pointer-events-auto" key={port.id || idx}>
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
    </div>
  );
});

export const MediaContent = Object.assign(MediaContentComponent, {
  defaultSize: { height: 400, width: 500 },

  getMinSize: (type: MediaType) => {
    const config = MEDIA_CONFIGS[type];

    return config ? { height: config.minHeight, width: config.minWidth } : { height: 150, width: 200 };
  },
});
