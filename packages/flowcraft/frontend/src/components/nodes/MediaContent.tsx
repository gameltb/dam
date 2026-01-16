import { create as createProto } from "@bufbuild/protobuf";
import React, { memo } from "react";

import {
  MediaType,
  type MediaContent as ProtoMediaContent,
  PresentationSchema,
} from "@/generated/flowcraft/v1/core/base_pb";
import { GraphMutationSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { useNodeHandlers } from "@/hooks/useNodeHandlers";
import { useFlowStore } from "@/store/flowStore";
import { type DynamicNodeData, FlowEvent } from "@/types";
import { getMediaTypeFromMime } from "@/utils/nodeUtils";
import { getPortColor, getPortShape } from "@/utils/themeUtils";

import { PortHandle } from "../base/PortHandle";
import { GalleryWrapper } from "../media/GalleryWrapper";
import { MEDIA_RENDERERS } from "../media/mediaRenderRegistry";

interface MediaContentProps {
  data: DynamicNodeData;
  height?: number;
  id: string;
  onOverflowChange?: (o: any) => void;
  width?: number;
}

export const MediaContent: React.FC<MediaContentProps> = memo(({ data, height, id, onOverflowChange, width }) => {
  const { onChange, onGalleryItemContext } = useNodeHandlers(data);
  const dispatchNodeEvent = useFlowStore((state) => state.dispatchNodeEvent);

  const getEffectiveMedia = () => {
    if (data.media?.url) return data.media;

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
    if (Math.abs((media.aspectRatio ?? 0) - ratio) > 0.01) {
      onChange(id, {
        media: { ...media, aspectRatio: ratio } as ProtoMediaContent,
      });

      const currentWidth = width ?? 240;
      const targetHeight = Math.round(currentWidth / ratio);

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

  const renderContent = (url: string, type: MediaType, index = 0, content?: string) => {
    const Renderer = MEDIA_RENDERERS[type];
    if (!Renderer) return null;

    return (
      <Renderer
        content={content}
        index={index}
        onDimensionsLoad={handleDimensionsLoad}
        onEdit={(newContent: string) => {
          onChange(id, {
            media: { ...media, content: newContent } as ProtoMediaContent,
            widgetsValues: { ...data.widgetsValues, content: newContent },
          });
        }}
        onOpenPreview={handleOpenPreview}
        url={url}
      />
    );
  };

  const gallery = (media.galleryUrls as string[]) ?? [];
  const inputs = data.inputPorts ?? [];
  const outputs = data.outputPorts ?? [];

  return (
    <div
      className="relative h-full w-full overflow-visible"
      onDoubleClick={(e) => {
        e.stopPropagation();
        handleOpenPreview(0);
      }}
    >
      <div className="nopan relative h-full w-full overflow-hidden rounded-[inherit] pointer-events-auto">
        {renderContent(media.url ?? "", media.type, 0, media.content)}
      </div>

      <div className="absolute inset-0 overflow-visible pointer-events-none z-[100]">
        {gallery.length > 0 && (
          <GalleryWrapper
            gallery={gallery}
            id={id}
            mainContent={<div className="h-full w-full pointer-events-none" />}
            mediaType={media.type}
            nodeHeight={nodeHeight}
            nodeWidth={nodeWidth}
            onExpand={() => onOverflowChange?.("visible")}
            onGalleryItemContext={onGalleryItemContext}
            renderItem={(url) => (
              <div className="h-full w-full overflow-hidden rounded-sm">
                {renderContent(url, media.type, gallery.indexOf(url) + 1)}
              </div>
            )}
          />
        )}

        <div className="pointer-events-none">
          {outputs.map((port, idx) => (
            <div className="absolute right-0 top-1/2 -translate-y-1/2 pointer-events-auto" key={port.id || idx}>
              <PortHandle
                color={getPortColor(port.type as any)}
                isPresentation={true}
                nodeId={id}
                portId={port.id}
                style={getPortShape(port.type as any)}
                type="source"
              />
            </div>
          ))}
          {inputs.map((port, idx) => (
            <div className="absolute left-0 top-1/2 -translate-y-1/2 pointer-events-auto" key={port.id || idx}>
              <PortHandle
                color={getPortColor(port.type as any)}
                isPresentation={true}
                nodeId={id}
                portId={port.id}
                style={getPortShape(port.type as any)}
                type="target"
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
});
