import { http, HttpResponse } from "msw";
import { nodeTemplates, actionTemplates } from "./templates";
import { generateGallery } from "./generators";
import { setServerNodes, setServerEdges } from "./db";
import { handleWSMessage } from "./wsLogic";
import { flowcraft_proto } from "../generated/flowcraft_proto";

// --- Initialize State ---
const gallery = generateGallery();
setServerNodes(gallery.nodes);
setServerEdges(gallery.edges);

// --- MSW Route Definitions ---

export const handlers = [
  // 1. Templates API (REST)
  http.get("/api/node-templates", () => HttpResponse.json(nodeTemplates)),

  // 2. Actions Discovery (REST)
  http.post("/api/actions/discover", () =>
    HttpResponse.json({ actions: actionTemplates }),
  ),

  // 3. Unified WebSocket Protocol (Stream over HTTP)
  http.post("/api/ws", async ({ request }) => {
    const clientMsg = (await request.json()) as flowcraft_proto.v1.IFlowMessage;

    const stream = new ReadableStream({
      async start(controller) {
        await handleWSMessage(clientMsg, controller);
        controller.close();
      },
    });

    return new HttpResponse(stream, {
      headers: {
        "Content-Type": "text/plain",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }),
];
