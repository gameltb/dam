import * as fs from "fs";
import * as path from "path";
import * as yaml from "yaml";

export interface FlowcraftConfig {
  pb_client: {
    output_path: string;
    reducers_dir: string;
  };
}

export function loadConfig(): FlowcraftConfig {
  const configPath = path.resolve("flowcraft.config.yaml");
  const file = fs.readFileSync(configPath, "utf8");
  return yaml.parse(file) as FlowcraftConfig;
}
