import React from "react";

import { MediaType } from "../../generated/flowcraft/v1/core/node_pb";
import { AudioRenderer } from "./AudioRenderer";
import { ImageRenderer } from "./ImageRenderer";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { VideoRenderer } from "./VideoRenderer";

export interface MediaRenderProps {
  content?: string;
  index: number;
  onDimensionsLoad?: (ratio: number) => void;
  onEdit?: (newContent: string) => void;
  onOpenPreview?: (index: number) => void;
  url: string;
}

export const MEDIA_RENDERERS: Record<
  number,
  React.ComponentType<MediaRenderProps>
> = {
  [MediaType.MEDIA_AUDIO]: ({ index, onOpenPreview, url }) => (
    <div
      onDoubleClick={() => {
        onOpenPreview?.(index);
      }}
      style={{ height: "100%", width: "100%" }}
    >
      <AudioRenderer url={url} />
    </div>
  ),
  [MediaType.MEDIA_IMAGE]: ({ onDimensionsLoad, url }) => (
    <div style={{ height: "100%", width: "100%" }}>
      <ImageRenderer onDimensionsLoad={onDimensionsLoad} url={url} />
    </div>
  ),
  [MediaType.MEDIA_MARKDOWN]: ({ content, onEdit }) => (
    <MarkdownRenderer content={content ?? ""} onEdit={onEdit} />
  ),
  [MediaType.MEDIA_VIDEO]: ({ onDimensionsLoad, url }) => (
    <div
      style={{
        background: "transparent",
        height: "100%",
        width: "100%",
      }}
    >
      <VideoRenderer autoPlay onDimensionsLoad={onDimensionsLoad} url={url} />
    </div>
  ),
};
