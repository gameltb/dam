import { create } from "@bufbuild/protobuf";
import {
  ActionTemplateSchema,
  ActionExecutionStrategy,
} from "../../generated/flowcraft/v1/core/action_pb";
import { NodeRegistry, type ActionHandlerContext } from "../registry";
import { eventBus, serverGraph, incrementVersion, serverVersion } from "../db";
import {
  TaskUpdateSchema,
  NodeSchema,
} from "../../generated/flowcraft/v1/core/node_pb";
import {
  MutationListSchema,
  StreamChunkSchema,
} from "../../generated/flowcraft/v1/core/service_pb";
import { v4 as uuidv4 } from "uuid";
import { fromProtoNode, toProtoNode } from "../../utils/protoAdapter";
import { OpenAI } from "openai";

// Reuse OpenAI setup logic or import it
const AI_CONFIG = {
  apiKey: process.env.OPENAI_API_KEY ?? "your-key-here",
  baseURL: process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1",
  model: process.env.OPENAI_MODEL ?? "gpt-3.5-turbo",
};

// Lazy initialization of OpenAI client
let openaiInstance: OpenAI | null = null;
function getOpenAI() {
  if (!openaiInstance) {
    openaiInstance = new OpenAI({
      apiKey: AI_CONFIG.apiKey,
      baseURL: AI_CONFIG.baseURL,
      dangerouslyAllowBrowser: true, // Needed for vitest/jsdom environment
    });
  }
  return openaiInstance;
}

async function runAiGeneration(req: ActionHandlerContext, instruction: string) {
  const taskId = uuidv4();
  const { sourceNodeId, contextNodeIds = [] } = req;
  const openai = getOpenAI();

  const targetNodes = serverGraph.nodes.filter(
    (n) => contextNodeIds.includes(n.id) || n.id === sourceNodeId,
  );
  const contextText = targetNodes
    .map((n) => `[Node ${n.id}]: ${n.data.label}`)
    .join("\n");

  const maxY = Math.max(
    ...targetNodes.map((n) => n.position.y + (n.measured?.height ?? 200)),
    0,
  );
  const avgX =
    targetNodes.reduce((acc, n) => acc + n.position.x, 0) /
    (targetNodes.length || 1);

  eventBus.emit(
    "taskUpdate",
    create(TaskUpdateSchema, {
      taskId,
      status: 1,
      progress: 10,
      message: "Requesting AI...",
    }),
  );

  const newNodeId = `ai-${uuidv4().slice(0, 8)}`;
  const newNode = fromProtoNode(
    create(NodeSchema, {
      nodeId: newNodeId,
      nodeKind: 1,
      templateId: "tpl-stream-node",
      presentation: {
        position: { x: avgX, y: maxY + 50 },
        width: 400,
        height: 300,
        isInitialized: true,
      },
      state: {
        displayName: "AI Result",
        widgetsValuesJson: JSON.stringify({ agent_name: "OpenAI", logs: "" }),
      },
    }),
  );

  serverGraph.nodes.push(newNode);
  incrementVersion();
  eventBus.emit(
    "mutations",
    create(MutationListSchema, {
      mutations: [
        {
          operation: { case: "addNode", value: { node: toProtoNode(newNode) } },
        },
      ],
      sequenceNumber: BigInt(serverVersion),
      source: 2,
    }),
  );

  try {
    const stream = await openai.chat.completions.create({
      model: AI_CONFIG.model,
      messages: [
        { role: "system", content: "You are a Flowcraft assistant." },
        {
          role: "user",
          content: `Context:\n${contextText}\n\nInstruction: ${instruction}`,
        },
      ],
      stream: true,
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content ?? "";
      if (content) {
        eventBus.emit(
          "streamChunk",
          create(StreamChunkSchema, {
            nodeId: newNodeId,
            widgetId: "logs",
            chunkData: content,
            isDone: false,
          }),
        );
      }
    }
    eventBus.emit(
      "taskUpdate",
      create(TaskUpdateSchema, {
        taskId,
        status: 2,
        progress: 100,
        message: "AI complete",
      }),
    );
  } catch (err) {
    eventBus.emit(
      "taskUpdate",
      create(TaskUpdateSchema, {
        taskId,
        status: 3,
        message: `Error: ${err instanceof Error ? err.message : String(err)}`,
      }),
    );
  }
}

// 注册动作
NodeRegistry.registerGlobalAction(
  create(ActionTemplateSchema, {
    id: "ai-enhance",
    label: "Enhance",
    path: ["AI Tools"],
    strategy: ActionExecutionStrategy.EXECUTION_BACKGROUND,
  }),
  (ctx) => runAiGeneration(ctx, "Enhance the content"),
);

NodeRegistry.registerGlobalAction(
  create(ActionTemplateSchema, {
    id: "prompt-gen",
    label: "Generate from Prompt",
    path: ["AI Tools"],
    strategy: ActionExecutionStrategy.EXECUTION_BACKGROUND,
  }),
  (ctx) => runAiGeneration(ctx, ctx.params.prompt ?? "Generate from prompt"),
);

NodeRegistry.registerGlobalAction(
  create(ActionTemplateSchema, {
    id: "ai-transform",
    label: "AI Generate (Context Aware)",
    path: ["AI Tools"],
    strategy: ActionExecutionStrategy.EXECUTION_BACKGROUND,
  }),
  (ctx) =>
    runAiGeneration(ctx, ctx.params.instruction ?? "Context aware transform"),
);
