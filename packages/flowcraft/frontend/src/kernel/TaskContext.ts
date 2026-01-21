import { type Message } from "@bufbuild/protobuf";

/**
 * Unified execution context for both Frontend and Worker.
 */
export interface TaskContext {
  // Result reporting
  complete: (result: Message<any> | Record<string, any>) => Promise<void>;
  // Configuration captured at submission
  config: Record<string, any>;

  fail: (error: string) => Promise<void>;
  // Lifecycle control
  isCancelled: () => boolean;

  // Logging
  log: (message: string, level?: "debug" | "error" | "info" | "warn") => Promise<void>;

  nodeId: string;

  params: Record<string, any>;

  taskId: string;
  // Progress reporting
  updateProgress: (percentage: number, message?: string) => Promise<void>;
}

export abstract class BaseTaskLogic<P = any, R = any> {
  abstract run(ctx: TaskContext, params: P): Promise<R>;
}
