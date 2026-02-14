import { useMemo } from "react";

import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { type DynamicNodeData } from "@/types";

export function useNodeMedia(data: DynamicNodeData | undefined) {
  return useMemo(() => {
    const isMedia = data?.activeMode === RenderMode.MODE_MEDIA;

    const getMediaType = (): MediaType | undefined => {
      if (data?.media?.type !== undefined) return data.media.type;
      const ext = data?.extension;
      if (ext?.case === "visual") return MediaType.MEDIA_IMAGE;
      if (ext?.case === "acoustic") return MediaType.MEDIA_AUDIO;
      if (ext?.case === "document") return MediaType.MEDIA_MARKDOWN;
      return undefined;
    };

    const mediaType = getMediaType();
    const isAudio = isMedia && mediaType === MediaType.MEDIA_AUDIO;
    const isImage = isMedia && mediaType === MediaType.MEDIA_IMAGE;
    const isVideo = isMedia && mediaType === MediaType.MEDIA_VIDEO;

    const shouldLockAspectRatio = isMedia && (isImage || isVideo);

    return {
      isAudio,
      isImage,
      isMedia,
      isVideo,
      mediaType,
      shouldLockAspectRatio,
    };
  }, [data]);
}
