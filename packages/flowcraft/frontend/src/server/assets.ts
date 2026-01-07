import fs from "fs";
import path from "path";
import { v4 as uuidv4 } from "uuid";

// 资产元数据接口
export interface Asset {
  id: string;
  mimeType: string;
  name: string;
  path: string;
  url: string;
}

class AssetService {
  private assets = new Map<string, Asset>();
  private uploadDir: string;

  constructor() {
    const storageDir =
      process.env.FLOWCRAFT_STORAGE_DIR ?? path.join(process.cwd(), "storage");
    this.uploadDir = path.join(storageDir, "assets");
    if (!fs.existsSync(this.uploadDir)) {
      fs.mkdirSync(this.uploadDir, { recursive: true });
    }
  }

  getAsset(id: string): Asset | undefined {
    return this.assets.get(id);
  }

  saveAsset(file: { buffer: Buffer; mimeType: string; name: string }): Asset {
    const id = uuidv4();
    const fileName = `${id}-${file.name}`;
    const filePath = path.join(this.uploadDir, fileName);

    fs.writeFileSync(filePath, file.buffer);

    const asset: Asset = {
      id,
      mimeType: file.mimeType,
      name: file.name,
      path: filePath,
      url: `/uploads/${fileName}`,
    };

    this.assets.set(id, asset);
    return asset;
  }
}

export const Assets = new AssetService();
