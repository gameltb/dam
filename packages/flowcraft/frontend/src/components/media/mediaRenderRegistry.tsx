import React from "react";
import { MediaType } from "../../generated/flowcraft/v1/core/node_pb";
import { ImageRenderer } from "./ImageRenderer";
import { VideoRenderer } from "./VideoRenderer";
import { AudioRenderer } from "./AudioRenderer";
import { MarkdownRenderer } from "./MarkdownRenderer";

export interface MediaRenderProps {
  url: string;
  content?: string;
  onEdit?: (newContent: string) => void;
  onDimensionsLoad?: (ratio: number) => void;
  onOpenPreview?: (index: number) => void;
  index: number;
}

export const MEDIA_RENDERERS: Record<
  number,
  React.ComponentType<MediaRenderProps>
> = {
  [MediaType.MEDIA_IMAGE]: ({ url, onDimensionsLoad }) => (
    <div style={{ width: "100%", height: "100%" }}>
      <ImageRenderer url={url} onDimensionsLoad={onDimensionsLoad} />
    </div>
  ),
  [MediaType.MEDIA_VIDEO]: ({ url, onDimensionsLoad }) => (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: "transparent",
      }}
    >
      <VideoRenderer url={url} autoPlay onDimensionsLoad={onDimensionsLoad} />
    </div>
  ),
  [MediaType.MEDIA_AUDIO]: ({ url, onOpenPreview, index }) => (
    <div
      onDoubleClick={() => {
        onOpenPreview?.(index);
      }}
      style={{ width: "100%", height: "100%" }}
    >
      <AudioRenderer url={url} />
    </div>
  ),
  [MediaType.MEDIA_MARKDOWN]: ({ content, onEdit }) => (
    <MarkdownRenderer content={content ?? ""} onEdit={onEdit} />
  ),
};
