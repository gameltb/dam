import * as fs from "fs";
import { parse } from "yaml";

export interface Config {
  pb_client: {
    output_path: string;
    reducers_dir: string;
  };
  stdb_schema: {
    output_path: string;
  };
}

/**
 * 加载并验证配置文件
 */
export function loadConfig(): Config {
  const configPath = "flowcraft.config.yaml";
  if (!fs.existsSync(configPath)) {
    throw new Error(`❌ Configuration file missing: ${configPath}`);
  }

  const file = fs.readFileSync(configPath, "utf8");
  const config = parse(file) as Config;

  // 严格验证每一个必需字段
  validate(config, "stdb_schema.output_path");
  validate(config, "pb_client.reducers_dir");
  validate(config, "pb_client.output_path");

  return config;
}

function validate(config: Config, path: string) {
  const parts = path.split(".");
  let current: any = config;
  for (const part of parts) {
    if (!current || (current as Record<string, unknown>)[part] === undefined) {
      throw new Error(`❌ Missing required config field: ${path} in flowcraft.config.yaml`);
    }
    current = (current as Record<string, unknown>)[part];
  }
}
