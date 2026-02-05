import "dotenv/config";
import path from "path";

export interface ServerConfig {
  assetsDir: string;
  host: string;
  port: number;
  storageDir: string;
}

const storageDir = process.env.FLOWCRAFT_STORAGE_DIR ?? path.join(process.cwd(), "storage");

export const SERVER_CONFIG: ServerConfig = {
  assetsDir: path.join(storageDir, "assets"),
  host: process.env.HOST ?? "0.0.0.0",
  port: parseInt(process.env.PORT ?? "3001", 10),
  storageDir,
};
