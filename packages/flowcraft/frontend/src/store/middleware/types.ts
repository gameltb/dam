import { type GraphMutation } from "@/generated/flowcraft/v1/core/service_pb";
import { type MutationContext } from "@/store/types";

export enum MutationDirection {
  INCOMING = "INCOMING", // From Server
  INTERNAL = "INTERNAL", // From Undo/Redo/Internal Logic
  OUTGOING = "OUTGOING", // From UI/Local Action
}

export type GraphMiddleware = (event: GraphMutationEvent, next: MiddlewareNext) => void;

export interface GraphMutationEvent {
  context: MutationContext;
  direction: MutationDirection;
  mutations: GraphMutation[];
}

export type MiddlewareNext = (event: GraphMutationEvent) => void;
