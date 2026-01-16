import "dotenv/config";
import fs from "fs";
import path from "path";

export interface InferenceConfig {
  defaultEndpointId: string;
  defaultModel: string;
  endpoints: InferenceEndpoint[];
}

export interface InferenceEndpoint {
  apiKey: string;
  baseURL: string;
  id: string;
  models: string[];
  name: string;
}

const CONFIG_FILE = path.join(process.cwd(), "inference.json");

function loadConfig(): InferenceConfig {
  let config: Partial<InferenceConfig> = {};

  if (fs.existsSync(CONFIG_FILE)) {
    try {
      const fileContent = fs.readFileSync(CONFIG_FILE, "utf-8");
      config = JSON.parse(fileContent) as Partial<InferenceConfig>;
    } catch (err) {
      console.error(`[Config] Failed to parse ${CONFIG_FILE}:`, err);
    }
  }

  return {
    defaultEndpointId: config.defaultEndpointId ?? "openai",
    defaultModel: config.defaultModel ?? process.env.OPENAI_MODEL ?? "gpt-4o-mini",
    endpoints: config.endpoints ?? [
      {
        apiKey: process.env.OPENAI_API_KEY ?? "dummy-key",
        baseURL: process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1",
        id: "openai",
        models: ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        name: "OpenAI",
      },
      {
        apiKey: process.env.ANTHROPIC_API_KEY ?? "",
        baseURL: process.env.ANTHROPIC_BASE_URL ?? "https://api.anthropic.com/v1",
        id: "anthropic",
        models: ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
        name: "Anthropic (Proxy)",
      },
    ],
  };
}

export const INFERENCE_CONFIG = loadConfig();
