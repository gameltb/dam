import { EventEmitter } from "events";

import { type AppNode, type Edge } from "@/types";

import { generateGallery } from "./Generators";

export let serverVersion = 0;

export const incrementVersion = () => {
  serverVersion++;
};

export const serverGraph: {
  edges: Edge[];
  nodes: AppNode[];
} = {
  edges: [],
  nodes: [],
};

export const eventBus = new EventEmitter();

// In-memory mutation log (previously SQLite)
const mutationLog: any[] = [];

export function getMutations(fromSeq: number, toSeq?: number) {
  return mutationLog.filter(
    (m) => m.seq >= fromSeq && (!toSeq || m.seq <= toSeq),
  );
}

export function loadFromDisk() {
  // SQLite loading removed.
  // Initializing with showcase gallery if empty.
  if (serverGraph.nodes.length === 0) {
    console.log("[Persistence] Initializing in-memory gallery showcase.");
    const gallery = generateGallery();
    serverGraph.nodes = gallery.nodes;
    serverGraph.edges = gallery.edges;
    serverVersion = 1;
  }
}

export function logMutation(
  type: string,
  payloadBinary: Uint8Array,
  source: number,
  description?: string,
) {
  const seq = mutationLog.length;
  const entry = {
    description: description ?? null,
    payload: Buffer.from(payloadBinary),
    seq,
    source,
    timestamp: Date.now(),
    type,
    user_id: null,
  };
  mutationLog.push(entry);
  return seq;
}

export function syncToDB() {
  // SQLite sync removed. SpacetimeDB is the source of truth.
}

export {
  addChatMessage,
  branchAndEditMessage,
  clearChatHistory,
  duplicateBranch,
  getChatHistory,
  getMessage,
} from "./ChatService";
