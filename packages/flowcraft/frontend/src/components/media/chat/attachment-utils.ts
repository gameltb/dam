import { create } from "@bufbuild/protobuf";
import { type FileUIPart } from "ai";

import { ChatMessagePartSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { uploadFile } from "@/utils/assetUtils";

/**
 * Maps UI attachments to Protobuf Message Parts.
 */
export function mapAttachmentsToParts(attachments: FileUIPart[]) {
  return attachments.map((att) =>
    create(ChatMessagePartSchema, {
      part: {
        case: "media",
        value: {
          aspectRatio: 0,
          content: "",
          galleryUrls: [],
          type: att.mediaType.startsWith("image") ? MediaType.MEDIA_IMAGE : MediaType.MEDIA_UNSPECIFIED,
          url: att.url,
        },
      },
    }),
  );
}

/**
 * Uploads blob attachments to the asset server and returns a list of final attachments.
 */
export async function processAttachments(files: FileUIPart[]): Promise<FileUIPart[]> {
  const finalAttachments: FileUIPart[] = [];
  for (const file of files) {
    if (file.url.startsWith("blob:")) {
      try {
        const response = await fetch(file.url);
        const blob = await response.blob();
        const url = await uploadFile(
          new File([blob], file.filename ?? "img.png", {
            type: file.mediaType,
          }),
        );
        if (url) finalAttachments.push({ ...file, url });
      } catch (err) {
        console.error("[ChatActions] Failed to upload attachment:", file.filename, err);
      }
    } else {
      finalAttachments.push(file);
    }
  }
  return finalAttachments;
}
