import { MediaType } from "../../generated/flowcraft/v1/core/node_pb";

export const AUDIO_RENDERER_CONFIG = {
  minHeight: 110,
  minWidth: 240,
};

export const IMAGE_RENDERER_CONFIG = {
  minHeight: 50,
  minWidth: 180,
};

export const VIDEO_RENDERER_CONFIG = {
  minHeight: 50,
  minWidth: 180,
};

export const MARKDOWN_RENDERER_CONFIG = {
  minHeight: 120,
  minWidth: 240,
};

export const MEDIA_CONFIGS: Record<
  number,
  { minHeight: number; minWidth: number }
> = {
  [MediaType.MEDIA_AUDIO]: AUDIO_RENDERER_CONFIG,
  [MediaType.MEDIA_IMAGE]: IMAGE_RENDERER_CONFIG,
  [MediaType.MEDIA_VIDEO]: VIDEO_RENDERER_CONFIG,
  [MediaType.MEDIA_MARKDOWN]: MARKDOWN_RENDERER_CONFIG,
};
