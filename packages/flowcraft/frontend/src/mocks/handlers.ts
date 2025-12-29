import { http, HttpResponse } from "msw";
import { nodeTemplates } from "./templates";
import { generateGallery } from "./generators";
import { setServerNodes, setServerEdges } from "./db";

// --- Initialize State ---
const gallery = generateGallery();
setServerNodes(gallery.nodes);
setServerEdges(gallery.edges);

// --- MSW Route Definitions ---

export const handlers = [
  // 1. Templates API (REST) - Keep this if needed, though currently unused by app logic
  http.get("/api/node-templates", () => HttpResponse.json(nodeTemplates)),
];
