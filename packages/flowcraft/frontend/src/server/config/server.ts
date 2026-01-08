import "dotenv/config";
import path from "path";

export interface ServerConfig {
  assetsDir: string;
  dbFile: string;
  host: string;
  port: number;
  storageDir: string;
}

const storageDir =
  process.env.FLOWCRAFT_STORAGE_DIR ?? path.join(process.cwd(), "storage");

export const SERVER_CONFIG: ServerConfig = {
  assetsDir: path.join(storageDir, "assets"),
  dbFile: path.join(storageDir, "flowcraft.db"),
  host: process.env.HOST ?? "0.0.0.0",
  port: parseInt(process.env.PORT ?? "3000", 10),
  storageDir,
};
