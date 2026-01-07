import "dotenv/config";

export interface AiEndpoint {
  apiKey: string;
  baseURL: string;
  id: string;
  models: string[];
  name: string;
}

export interface AiServiceConfig {
  defaultEndpointId: string;
  defaultModel: string;
  endpoints: AiEndpoint[];
}

export const AI_CONFIG: AiServiceConfig = {
  defaultEndpointId: "openai",
  defaultModel: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
  endpoints: [
    {
      apiKey: process.env.OPENAI_API_KEY ?? "dummy-key",
      baseURL: process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1",
      id: "openai",
      models: ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
      name: "OpenAI",
    },
    {
      apiKey: process.env.ANTHROPIC_API_KEY ?? "",
      baseURL: process.env.ANTHROPIC_BASE_URL ?? "https://api.anthropic.com/v1", // Usually needs a proxy if using OpenAI client
      id: "anthropic",
      models: ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
      name: "Anthropic (Proxy)",
    },
    // Add more endpoints as needed
  ],
};
