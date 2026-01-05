import { v4 as uuidv4 } from "uuid";
import path from "path";
import fs from "fs";

// 资产元数据接口
export interface Asset {
  id: string;
  name: string;
  mimeType: string;
  url: string;
  path: string;
}

class AssetService {
  private assets = new Map<string, Asset>();
  private uploadDir: string;

  constructor() {
    const storageDir =
      process.env.FLOWCRAFT_STORAGE_DIR || path.join(process.cwd(), "storage");
    this.uploadDir = path.join(storageDir, "assets");
    if (!fs.existsSync(this.uploadDir)) {
      fs.mkdirSync(this.uploadDir, { recursive: true });
    }
  }

  async saveAsset(file: {
    name: string;
    mimeType: string;
    buffer: Buffer;
  }): Promise<Asset> {
    const id = uuidv4();
    const fileName = `${id}-${file.name}`;
    const filePath = path.join(this.uploadDir, fileName);

    fs.writeFileSync(filePath, file.buffer);

    const asset: Asset = {
      id,
      name: file.name,
      mimeType: file.mimeType,
      url: `/uploads/${fileName}`,
      path: filePath,
    };

    this.assets.set(id, asset);
    return asset;
  }

  getAsset(id: string): Asset | undefined {
    return this.assets.get(id);
  }
}

export const Assets = new AssetService();
