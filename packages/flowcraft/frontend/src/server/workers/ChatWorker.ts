import { type PbClient } from "../../utils/pb-client";
import { type DbConnection } from "@/generated/spacetime";
import { BaseWorker } from "@/shared/BaseWorker";
import { type ChatGeneratePayload, TaskQueue } from "@/shared/task-protocol";

export class ChatWorker extends BaseWorker<TaskQueue.CHAT_GENERATE> {
  constructor(conn: DbConnection, pbClient: PbClient) {
    super(conn, TaskQueue.CHAT_GENERATE, pbClient);
  }

  protected async perform(
    payload: ChatGeneratePayload,
    onProgress: (p: number, msg?: string) => void,
  ): Promise<unknown> {
    console.log("[ChatWorker] Processing generation:", payload);
    onProgress(10, "Initializing...");

    await new Promise((resolve) => setTimeout(resolve, 1000));
    onProgress(50, "Generating...");

    return { success: true };
  }
}
