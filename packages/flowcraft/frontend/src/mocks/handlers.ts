import { http, HttpResponse } from "msw";
import { v4 as uuidv4 } from "uuid";
import type { AppNode, NodeTemplate } from "../types";
import type { Edge, Viewport } from "@xyflow/react";

// --- Node Templates (Server-Defined creation options) ---

const nodeTemplates: NodeTemplate[] = [
  {
    id: "tpl-text-input",
    label: "Text Input",
    path: ["Basic", "Inputs"],
    defaultData: {
      label: "New Text Input",
      modes: ["widgets"],
      activeMode: "widgets",
      widgets: [{ id: "txt1", type: "text", label: "Value", value: "" }],
      inputType: "any",
      outputType: "text",
    },
  },
  {
    id: "tpl-image-viewer",
    label: "Image Viewer",
    path: ["Basic", "Display"],
    defaultData: {
      label: "Image Viewer",
      modes: ["media", "widgets"],
      activeMode: "media",
      media: { type: "image", url: "https://picsum.photos/400/300" },
      widgets: [
        {
          id: "img-url",
          type: "text",
          label: "URL",
          value: "https://picsum.photos/400/300",
        },
      ],
      inputType: "any",
      outputType: "image",
    },
  },
  {
    id: "tpl-complex-form",
    label: "Configurator",
    path: ["Advanced"],
    defaultData: {
      label: "Settings",
      modes: ["widgets"],
      activeMode: "widgets",
      widgets: [
        {
          id: "cfg-1",
          type: "select",
          label: "Type",
          value: "a",
          options: [
            { label: "A", value: "a" },
            { label: "B", value: "b" },
          ],
        },
        { id: "cfg-2", type: "checkbox", label: "Enable Feature", value: true },
        { id: "cfg-3", type: "slider", label: "Intensity", value: 50 },
      ],
      inputType: "any",
      outputType: "any",
    },
  },
  {
    id: "tpl-markdown",
    label: "Markdown Note",
    path: ["Basic", "Notes"],
    defaultData: {
      label: "Notes",
      modes: ["media"],
      activeMode: "media",
      media: {
        type: "markdown",
        content: "# New Note\nDouble click to edit me!\n- Point 1\n- Point 2",
      },
      inputType: "any",
      outputType: "any",
    },
  },
];

// --- In-Memory Database ---

let serverVersion = 0;
let serverGraph: { nodes: AppNode[]; edges: Edge[]; viewport?: Viewport } = {
  nodes: [
    // 1. Media-only node (Locked ratio) with Gallery
    {
      id: "node-media",
      type: "dynamic",
      position: { x: 100, y: 100 },
      style: { width: 240, height: 180 },
      data: {
        label: "Locked Media",
        modes: ["media"],
        activeMode: "media",
        media: {
          type: "image",
          url: "https://picsum.photos/id/10/400/300",
          aspectRatio: 4 / 3,
          gallery: [
            "https://picsum.photos/id/11/400/300",
            "https://picsum.photos/id/12/400/300",
            "https://picsum.photos/id/13/400/300",
            "https://picsum.photos/id/14/400/300",
          ],
        },
        inputType: "any",
        outputType: "image",
        onChange: () => {},
      },
    },
    // 2. Video Gallery Node
    {
      id: "node-video-gallery",
      type: "dynamic",
      position: { x: 100, y: 350 },
      style: { width: 320, height: 180 },
      data: {
        label: "Video Gallery",
        modes: ["media"],
        activeMode: "media",
        media: {
          type: "video",
          url: "https://www.w3schools.com/html/mov_bbb.mp4",
          gallery: ["https://www.w3schools.com/html/movie.mp4"],
        },
        inputType: "any",
        outputType: "video",
        onChange: () => {},
      },
    },
    // 3. Markdown Node
    {
      id: "node-markdown",
      type: "dynamic",
      position: { x: 450, y: 350 },
      style: { width: 300, height: 300 },
      data: {
        label: "Project Specs",
        modes: ["media"],
        activeMode: "media",
        media: {
          type: "markdown",
          content:
            "# Flowcraft Specs\n## Features\n- Dynamic Nodes\n- Markdown Support\n- Video Gallery\n\nDouble-click here to edit this documentation!",
        },
        inputType: "any",
        outputType: "any",
        onChange: () => {},
      },
    },
    // 4. Widgets-only node (Free resize, with min constraints)
    {
      id: "node-widgets",
      type: "dynamic",
      position: { x: 400, y: 100 },
      data: {
        label: "Widgets (Min Size)",
        modes: ["widgets"],
        activeMode: "widgets",
        widgets: [
          { id: "w1", type: "text", label: "Title", value: "Legible Text" },
          { id: "w2", type: "checkbox", label: "Enabled", value: true },
          { id: "w3", type: "slider", label: "Value", value: 50 },
        ],
        inputType: "any",
        outputType: "any",
        onChange: () => {},
      },
    },
    // 3. Switchable node
    {
      id: "node-switch",
      type: "dynamic",
      position: { x: 100, y: 400 },
      style: { width: 300, height: 200 },
      data: {
        label: "Switchable",
        modes: ["media", "widgets"],
        activeMode: "widgets",
        media: {
          type: "image",
          url: "https://picsum.photos/600/400",
          aspectRatio: 1.5,
        },
        widgets: [
          { id: "sw1", type: "button", label: "Open Editor", value: null },
        ],
        inputType: "any",
        outputType: "any",
        onChange: () => {},
      },
    },
  ],
  edges: [],
  viewport: { x: 0, y: 0, zoom: 1 },
};

// --- Handlers ---

export const handlers = [
  // 1. Get Graph
  http.get("/api/graph", () => {
    return HttpResponse.json({
      type: "sync_graph",
      payload: {
        version: serverVersion,
        graph: serverGraph,
      },
    });
  }),

  // 2. Get Node Templates
  http.get("/api/node-templates", () => {
    return HttpResponse.json(nodeTemplates);
  }),

  // 3. Execute Action
  http.post("/api/action", async ({ request }) => {
    const body = (await request.json()) as {
      actionId: string;
      nodeId: string;
    };
    const { actionId, nodeId } = body;

    if (actionId === "generate-children") {
      const parentNode = serverGraph.nodes.find((n) => n.id === nodeId);

      if (parentNode) {
        const newNodes: AppNode[] = [
          {
            id: uuidv4(),
            type: "dynamic",
            position: { x: 0, y: 0 },
            data: {
              label: "Child 1",
              modes: ["widgets"],
              activeMode: "widgets",
              widgets: [
                {
                  id: uuidv4(),
                  type: "text",
                  label: "Auto Text",
                  value: "Generated Content",
                },
              ],
              inputType: "any",
              outputType: "any",
              onChange: () => {},
            },
          },
          {
            id: uuidv4(),
            type: "dynamic",
            position: { x: 0, y: 0 },
            data: {
              label: "Child 2",
              modes: ["widgets"],
              activeMode: "widgets",
              widgets: [
                {
                  id: uuidv4(),
                  type: "text",
                  label: "Auto Text",
                  value: "Generated Content",
                },
              ],
              inputType: "any",
              outputType: "any",
              onChange: () => {},
            },
          },
        ];

        const newEdges: Edge[] = [
          {
            id: uuidv4(),
            source: nodeId,
            target: newNodes[0].id,
            type: "system",
            sourceHandle: "system-source",
            targetHandle: "system-target",
          },
          {
            id: uuidv4(),
            source: nodeId,
            target: newNodes[1].id,
            type: "system",
            sourceHandle: "system-source",
            targetHandle: "system-target",
          },
        ];

        serverGraph.nodes = [...serverGraph.nodes, ...newNodes];
        serverGraph.edges = [...serverGraph.edges, ...newEdges];

        return HttpResponse.json({
          type: "apply_changes",
          payload: {
            add: newNodes,
            addEdges: newEdges,
            update: [],
          },
        });
      }
    }

    return HttpResponse.json({ type: "error", message: "Action failed" });
  }),

  // 4. Sync/Update Graph
  http.post("/api/graph", async ({ request }) => {
    const body = (await request.json()) as {
      version: number;
      graph: { nodes: AppNode[]; edges: Edge[]; viewport?: Viewport };
    };
    const { version, graph } = body;

    serverGraph = graph;
    serverVersion = version + 1;

    return HttpResponse.json({
      type: "sync_graph",
      payload: {
        version: serverVersion,
        graph: serverGraph,
      },
    });
  }),
];
