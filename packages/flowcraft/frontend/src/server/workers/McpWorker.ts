import { create as createProto } from "@bufbuild/protobuf";
import { fromBinary } from "@bufbuild/protobuf";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ErrorCode, ListToolsRequestSchema, McpError } from "@modelcontextprotocol/sdk/types.js";

import { NodeDataSchema, NodeSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { type PbConnection } from "@/utils/pb-client";

import logger from "../utils/logger";

interface AddNodeArgs {
  displayName: string;
  templateId: string;
  x?: number;
  y?: number;
}

interface UpdateNodePropertyArgs {
  nodeId: string;
  path: string;
  value: unknown;
}

export class McpWorker {
  private conn: PbConnection;
  private server: Server;

  constructor(conn: PbConnection) {
    this.conn = conn;
    this.server = new Server(
      {
        name: "flowcraft-mcp-worker",
        version: "1.0.0",
      },
      {
        capabilities: {
          tools: {},
        },
      },
    );

    this.setupTools();

    this.server.onerror = (error) => logger.error("[MCP Error]", error);
    process.on("SIGINT", async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  async start() {
    // Note: MCP over stdio is usually for direct CLI integration.
    // If we want third-party access while the server is running,
    // we might need an SSE transport or a dedicated stdio-to-websocket bridge.
    // For now, we implement stdio as it's the standard for local MCP tools.
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    logger.info("MCP Worker started over Stdio");
  }

  private setupTools() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          description: "List all nodes in the current graph",
          inputSchema: {
            properties: {},
            type: "object",
          },
          name: "list_nodes",
        },
        {
          description: "Add a new node to the graph",
          inputSchema: {
            properties: {
              displayName: { description: "Display name for the node", type: "string" },
              templateId: { description: "Template ID for the node", type: "string" },
              x: { type: "number" },
              y: { type: "number" },
            },
            required: ["templateId", "displayName"],
            type: "object",
          },
          name: "add_node",
        },
        {
          description: "Update a property of a node using JSON path",
          inputSchema: {
            properties: {
              nodeId: { type: "string" },
              path: { description: "Dot-separated path (e.g. state.displayName)", type: "string" },
              value: { description: "New value for the property", type: "any" },
            },
            required: ["nodeId", "path", "value"],
            type: "object",
          },
          name: "update_node_property",
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      switch (request.params.name) {
        case "add_node": {
          const { displayName, templateId, x = 0, y = 0 } = request.params.arguments as unknown as AddNodeArgs;
          const nodeId = crypto.randomUUID();

          const node = createProto(NodeSchema, {
            nodeId,
            presentation: {
              isInitialized: true,
              position: { x, y },
            },
            state: createProto(NodeDataSchema, {
              displayName,
            }),
            templateId,
          });

          this.conn.pbreducers.createNodePb({ node });

          return {
            content: [{ text: `Node created with ID: ${nodeId}`, type: "text" }],
          };
        }

        case "list_nodes": {
          const nodes = Array.from(this.conn.db.nodes.iter());
          return {
            content: [
              {
                text: JSON.stringify(nodes, null, 2),
                type: "text",
              },
            ],
          };
        }

        case "update_node_property": {
          const { nodeId, path, value } = request.params.arguments as unknown as UpdateNodePropertyArgs;

          if (path === "presentation.position.x" || path === "presentation.position.y") {
            const transform = this.conn.db.nodeTransforms.nodeId.find(nodeId);
            if (transform) {
              const x = path === "presentation.position.x" ? Number(value) : transform.x;
              const y = path === "presentation.position.y" ? Number(value) : transform.y;
              this.conn.reducers.setNodePosition({ nodeId, x, y });
            }
          } else if (path === "state.displayName") {
            const existing = this.conn.db.nodeData.nodeId.find(nodeId);
            if (existing) {
              const state = fromBinary(NodeDataSchema, existing.state);
              state.displayName = String(value);
              this.conn.pbreducers.setNodeDataPb({ nodeId, state });
            }
          }

          return {
            content: [{ text: `Update request processed for node ${nodeId}`, type: "text" }],
          };
        }

        default:
          throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${request.params.name}`);
      }
    });
  }
}
