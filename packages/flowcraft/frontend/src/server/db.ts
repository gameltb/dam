import { EventEmitter } from "events";
import { type AppNode, type Edge } from "../types";
import fs from "fs";
import path from "path";
import { generateGallery } from "./generators";

const STORAGE_DIR = process.env.FLOWCRAFT_STORAGE_DIR || path.join(process.cwd(), "storage");
if (!fs.existsSync(STORAGE_DIR)) {
  fs.mkdirSync(STORAGE_DIR, { recursive: true });
}
const STORAGE_FILE = path.join(STORAGE_DIR, "storage.json");

export let serverVersion = 0;

const setServerVersion = (v: number) => {
  serverVersion = v;
};

export const incrementVersion = () => {
  serverVersion++;
  saveToDisk();
};

export const serverGraph: {
  nodes: AppNode[];
  edges: Edge[];
} = {
  nodes: [],
  edges: [],
};

export const eventBus = new EventEmitter();

function saveToDisk() {
  try {
    const data = JSON.stringify(
      {
        nodes: serverGraph.nodes,
        edges: serverGraph.edges,
        version: serverVersion,
      },
      null,
      2,
    );
    fs.writeFileSync(STORAGE_FILE, data);
  } catch (err) {
    console.error("Failed to save to disk:", err);
  }
}

export function loadFromDisk() {
  try {
    if (fs.existsSync(STORAGE_FILE)) {
      const data = fs.readFileSync(STORAGE_FILE, "utf-8");
      const parsed = JSON.parse(data);
      serverGraph.nodes = parsed.nodes || [];
      serverGraph.edges = parsed.edges || [];
      serverVersion = parsed.version || 0;
      console.log(`[DB] Loaded ${serverGraph.nodes.length} nodes from disk.`);
    }

    if (serverGraph.nodes.length === 0) {
      console.log("[DB] Initializing empty graph with gallery showcase.");
      const gallery = generateGallery();
      serverGraph.nodes = gallery.nodes;
      serverGraph.edges = gallery.edges;
      serverVersion = 1;
      saveToDisk();
    }
  } catch (err) {
    console.error("Failed to load from disk:", err);
  }
}
