import "dotenv/config";

export interface AiEndpoint {
  id: string;
  name: string;
  baseURL: string;
  apiKey: string;
  models: string[];
}

export interface AiServiceConfig {
  endpoints: AiEndpoint[];
  defaultEndpointId: string;
  defaultModel: string;
}

export const AI_CONFIG: AiServiceConfig = {
  endpoints: [
    {
      id: "openai",
      name: "OpenAI",
      baseURL: process.env.OPENAI_BASE_URL || "https://api.openai.com/v1",
      apiKey: process.env.OPENAI_API_KEY || "dummy-key",
      models: ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    },
    {
      id: "anthropic",
      name: "Anthropic (Proxy)",
      baseURL: process.env.ANTHROPIC_BASE_URL || "https://api.anthropic.com/v1", // Usually needs a proxy if using OpenAI client
      apiKey: process.env.ANTHROPIC_API_KEY || "",
      models: ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
    },
    // Add more endpoints as needed
  ],
  defaultEndpointId: "openai",
  defaultModel: process.env.OPENAI_MODEL || "gpt-4o-mini",
};
