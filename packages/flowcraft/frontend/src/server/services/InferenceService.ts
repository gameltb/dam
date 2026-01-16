import OpenAI from "openai";

import { INFERENCE_CONFIG, type InferenceEndpoint } from "../config/inference";
import logger from "../utils/logger";

class InferenceService {
  private clients = new Map<string, OpenAI>();

  async chatCompletion(params: {
    endpointId?: string;
    messages: OpenAI.Chat.ChatCompletionMessageParam[];
    model?: string;
    stream?: boolean;
  }) {
    logger.info(`chatCompletion called for model: ${String(params.model)}, endpoint: ${String(params.endpointId)}`);
    const endpointId = params.endpointId ?? INFERENCE_CONFIG.defaultEndpointId;
    const model = params.model ?? INFERENCE_CONFIG.defaultModel;

    try {
      const client = this.getClient(endpointId);
      logger.info(`Client obtained for endpoint ${endpointId}. Starting request...`);

      const response = await client.chat.completions.create({
        messages: params.messages,
        model,
        stream: params.stream ?? false,
      });
      logger.info(`Request successful. Returning response/stream.`);
      return response;
    } catch (error) {
      logger.error(`OpenAI API Request Failed:`, error);
      throw error;
    }
  }

  getConfig() {
    return INFERENCE_CONFIG;
  }

  getEndpoints(): InferenceEndpoint[] {
    return INFERENCE_CONFIG.endpoints;
  }

  private getClient(endpointId: string): OpenAI {
    const existing = this.clients.get(endpointId);
    if (existing) return existing;

    const endpoint = INFERENCE_CONFIG.endpoints.find((e) => e.id === endpointId);
    if (!endpoint) {
      throw new Error(`Endpoint ${endpointId} not found`);
    }

    const client = new OpenAI({
      apiKey: endpoint.apiKey,
      baseURL: endpoint.baseURL,
    });
    this.clients.set(endpointId, client);
    return client;
  }
}

export const inferenceService = new InferenceService();
