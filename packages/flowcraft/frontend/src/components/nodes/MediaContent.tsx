import { create } from "@bufbuild/protobuf";
import { Layers, MessageSquareText } from "lucide-react";
import React, { memo } from "react";
import { useShallow } from "zustand/react/shallow";

import { MediaContentSchema, MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { type Port } from "@/generated/flowcraft/v1/core/node_pb";
import { useNodeHandlers } from "@/hooks/nodes/useNodeHandlers";
import { useFlowStore } from "@/store/flowStore";
import { editNode } from "@/store/orchestrator";
import { type DynamicNodeData, FlowEvent, OverflowMode } from "@/types";
import { mapToMediaType } from "@/utils/nodeUtils";
import { getPortColor, getPortShape } from "@/utils/themeUtils";

import { MEDIA_CONFIGS } from "../media/mediaConfigs";
import { MEDIA_RENDERERS } from "../media/mediaRenderRegistry";
import { GalleryOverlay } from "./parts/GalleryOverlay";
import { PortHandle } from "./PortHandle";

interface MediaContentProps {
  data: DynamicNodeData;
  height?: number;
  id: string;
  onGalleryItemContext?: (nodeId: string, url: string, mediaType: MediaType, x: number, y: number) => void;
  onOverflowChange?: (o: OverflowMode) => void;
  width?: number;
}

const MediaContentComponent: React.FC<MediaContentProps> = memo(({ data, height, id, onGalleryItemContext, width }) => {
  const { onChange } = useNodeHandlers(data);
  const { dispatchNodeEvent } = useFlowStore(
    useShallow((s) => ({
      dispatchNodeEvent: s.dispatchNodeEvent,
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
      type = mapToMediaType(v.mimeType.startsWith("video/") ? MediaType.MEDIA_VIDEO : MediaType.MEDIA_IMAGE);
      url = v.url;
      aspectRatio = 1.33;
      // Use type guard or optional property check instead of any
      if ("galleryUrls" in v && Array.isArray(v.galleryUrls)) {
        galleryUrls = v.galleryUrls as string[];
      }
    } else if (data.extension?.case === "document") {
      const d = data.extension.value;
      type = MediaType.MEDIA_MARKDOWN;
      content = d.content;
    } else if (data.extension?.case === "acoustic") {
      const a = data.extension.value;
      type = MediaType.MEDIA_AUDIO;
      url = a.url;
    } else if (data.media) {
      type = mapToMediaType(data.media.type);
      url = data.media.url;
      content = data.media.content;
      aspectRatio = data.media.aspectRatio;
      galleryUrls = data.media.galleryUrls;
    }

    if (data.media?.galleryUrls && galleryUrls.length === 0) {
      galleryUrls = data.media.galleryUrls;
    }

    if (type === MediaType.MEDIA_UNSPECIFIED && !url && !content) return null;
    return { aspectRatio, content, galleryUrls, type, url };
  };

  const media = getNormalizedMedia();
  if (!media) {
    const isSubgraph = data.extension?.case === "subgraph";
    const subgraphData = data.extension?.case === "subgraph" ? data.extension.value : undefined;

    if (isSubgraph && subgraphData) {
      return (
        <div className="flex flex-col items-center justify-center h-full w-full bg-primary/5 text-primary border-2 border-primary/20 rounded-md gap-3">
          <div className="p-4 bg-primary/10 rounded-full">
            <Layers className="text-primary opacity-80" size={32} />
          </div>
          <div className="flex flex-col items-center">
            <span className="text-xs font-black uppercase tracking-widest opacity-60">Subgraph / Session</span>
            <span className="text-[10px] font-mono opacity-40">
              {String(subgraphData?.subgraphId || id).slice(0, 8)}
            </span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1 bg-primary/10 rounded text-[9px] font-bold opacity-70">
            <MessageSquareText size={10} />
            CONVERSATION
          </div>
        </div>
      );
    }
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
        media: create(MediaContentSchema, { ...media, aspectRatio: ratio }),
      });

      const currentWidth = width ?? 240;
      const targetHeight = Math.round(currentWidth / ratio);

      if (Math.abs((height ?? 0) - targetHeight) > 5) {
        editNode(id, (draft) => {
          draft.height = targetHeight;
          draft.width = currentWidth;
        });
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
      throw new Error(`[MediaContent] No renderer for MediaType: ${mediaType.toString()}`);
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
      <div className="relative h-full w-full overflow-hidden rounded-[inherit] pointer-events-auto">
        {renderContent(media.url, media.type, 0, media.content)}
      </div>

      <div className="absolute inset-0 overflow-visible pointer-events-none z-[100]">
        {media.galleryUrls.length > 0 && (
          <GalleryOverlay
            gallery={media.galleryUrls}
            id={id}
            mediaType={media.type}
            nodeHeight={nodeHeight}
            nodeWidth={nodeWidth}
            onGalleryItemContext={onGalleryItemContext}
            renderItem={(url) => (
              <div className="h-full w-full overflow-hidden rounded-sm">
                {renderContent(url, media.type, media.galleryUrls.indexOf(url) + 1)}
              </div>
            )}
          />
        )}

        <div className="pointer-events-none">
          {data.outputPorts?.map((port: Port, idx: number) => (
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
          {data.inputPorts?.map((port: Port, idx: number) => (
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
