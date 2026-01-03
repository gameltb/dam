import { type GraphMutation } from "../../generated/flowcraft/v1/service_pb";
import { type MutationContext } from "../flowStore";

export enum MutationDirection {
  INCOMING = "INCOMING", // From Server
  OUTGOING = "OUTGOING", // From UI/Local Action
  INTERNAL = "INTERNAL", // From Undo/Redo/Internal Logic
}

export interface GraphMutationEvent {
  mutations: GraphMutation[];
  context: MutationContext;
  direction: MutationDirection;
}

export type MiddlewareNext = (event: GraphMutationEvent) => void;

export type GraphMiddleware = (
  event: GraphMutationEvent,
  next: MiddlewareNext,
) => void;
