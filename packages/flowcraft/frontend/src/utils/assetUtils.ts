export const uploadFile = async (file: File): Promise<null | string> => {
  const formData = new FormData();
  formData.append("file", file);
  try {
    const response = await fetch("/api/upload", {
      body: formData,
      method: "POST",
    });
    const asset = (await response.json()) as { url: string };
    return asset.url;
  } catch (err) {
    console.error("Upload failed:", err);
    return null;
  }
};
