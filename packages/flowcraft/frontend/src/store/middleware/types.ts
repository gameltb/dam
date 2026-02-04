import { type Patch } from "immer";

import { type MutationContext } from "@/types";

export enum MutationDirection {
  INCOMING = "incoming",
  OUTGOING = "outgoing",
}

export type GraphMiddleware = (event: GraphMutationEvent, next: MiddlewareNext) => void;

export interface GraphMutationEvent {
  context: MutationContext;
  direction: MutationDirection;
  inversePatches?: Patch[];
  /**
   * Uses Immer Patches to describe state changes.
   * Advantages: Fully automated, no string hardcoding, natural Undo support.
   */
  patches: Patch[];
}

export type MiddlewareNext = (event: GraphMutationEvent) => void;
