import fs from "fs";
import path from "path";
import { v4 as uuidv4 } from "uuid";

import { SERVER_CONFIG } from "../config/server";

export const AssetService = {
  saveAsset(params: { buffer: Buffer; mimeType: string; name: string }) {
    if (!fs.existsSync(SERVER_CONFIG.assetsDir)) {
      fs.mkdirSync(SERVER_CONFIG.assetsDir, { recursive: true });
    }

    const id = uuidv4();
    const fileName = `${id}-${params.name}`;
    const filePath = path.join(SERVER_CONFIG.assetsDir, fileName);

    fs.writeFileSync(filePath, params.buffer);

    return {
      id,
      mimeType: params.mimeType,
      name: params.name,
      url: `/uploads/${fileName}`,
    };
  },
};
