import OpenAI from "openai";

import { INFERENCE_CONFIG, type InferenceEndpoint } from "../config/inference";

class InferenceService {
  private clients = new Map<string, OpenAI>();

  async chatCompletion(params: {
    endpointId?: string;
    messages: OpenAI.Chat.ChatCompletionMessageParam[];
    model?: string;
    stream?: boolean;
  }) {
    const endpointId = params.endpointId ?? INFERENCE_CONFIG.defaultEndpointId;
    const model = params.model ?? INFERENCE_CONFIG.defaultModel;
    const client = this.getClient(endpointId);

    return client.chat.completions.create({
      messages: params.messages,
      model,
      stream: params.stream ?? false,
    });
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

    const endpoint = INFERENCE_CONFIG.endpoints.find(
      (e) => e.id === endpointId,
    );
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
