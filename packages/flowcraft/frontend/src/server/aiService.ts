import OpenAI from "openai";
import { AI_CONFIG, type AiEndpoint } from "./aiConfig";

class AiService {
  private clients = new Map<string, OpenAI>();

  private getClient(endpointId: string): OpenAI {
    const existing = this.clients.get(endpointId);
    if (existing) return existing;

    const endpoint = AI_CONFIG.endpoints.find((e) => e.id === endpointId);
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

  async chatCompletion(params: {
    messages: any[];
    model?: string;
    endpointId?: string;
    stream?: boolean;
  }) {
    const endpointId = params.endpointId || AI_CONFIG.defaultEndpointId;
    const model = params.model || AI_CONFIG.defaultModel;
    const client = this.getClient(endpointId);

    return client.chat.completions.create({
      model,
      messages: params.messages,
      stream: params.stream ?? false,
    });
  }

  getEndpoints(): AiEndpoint[] {
    return AI_CONFIG.endpoints;
  }
}

export const aiService = new AiService();
