import { NodeRegistry } from "../registry";
import { ActionTemplateSchema } from "../../generated/flowcraft/v1/core/action_pb";
import { StreamChunkSchema } from "../../generated/flowcraft/v1/core/service_pb";
import { create } from "@bufbuild/protobuf";
import OpenAI from "openai";

const MODEL_NAME = process.env.OPENAI_MODEL || "gpt-3.5-turbo";

// Configure OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || "dummy-key",
  baseURL: process.env.OPENAI_BASE_URL,
});

NodeRegistry.registerGlobalAction(
  create(ActionTemplateSchema, {
    id: "chat:generate",
    label: "Generate Chat",
    strategy: 0,
  }),
  async (ctx) => {
    const { params, emitStreamChunk, sourceNodeId } = ctx;
    const { messages } = params;

    if (!messages || !Array.isArray(messages)) {
      console.error("Invalid messages format");
      return;
    }

    try {
      const stream = await openai.chat.completions.create({
        model: MODEL_NAME,
        messages: messages.map((m: any) => ({
          role: m.role,
          content: m.content,
        })),
        stream: true,
      });

      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || "";
        if (content) {
          emitStreamChunk(
            create(StreamChunkSchema, {
              nodeId: sourceNodeId,
              chunkData: content,
              isDone: false,
            }),
          );
        }
      }

      // Final done message
      emitStreamChunk(
        create(StreamChunkSchema, {
          nodeId: sourceNodeId,
          chunkData: "",
          isDone: true,
        }),
      );
    } catch (err: any) {
      console.error("Chat generation failed:", err);
      emitStreamChunk(
        create(StreamChunkSchema, {
          nodeId: sourceNodeId,
          chunkData: `Error: ${err.message}`,
          isDone: true,
        }),
      );
    }
  },
);
