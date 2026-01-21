import { type MutationSource } from "@/types";

import { type MutationInput } from "../types";

export enum MutationDirection {
  INCOMING = "incoming",
  OUTGOING = "outgoing",
}

export type GraphMiddleware = (event: GraphMutationEvent, next: MiddlewareNext) => void;

export interface GraphMutationEvent {
  context: MutationContext;
  direction: MutationDirection;
  mutations: MutationInput[];
}

export type MiddlewareNext = (event: GraphMutationEvent) => void;

export interface MutationContext {
  description?: string;
  source?: MutationSource;
  taskId?: string;
}
