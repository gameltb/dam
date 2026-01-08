import { ReactFlowProvider } from "@xyflow/react";
import { StrictMode } from "react";

import "./index.css";
import { createRoot } from "react-dom/client";

import App from "./App.tsx";
import { useFlowStore } from "@/store/flowStore";
import { initStoreOrchestrator } from "@/store/orchestrator";
import ThemeProvider from "./ThemeProvider.tsx";

initStoreOrchestrator();

declare global {
  interface Window {
    useFlowStore: typeof useFlowStore;
  }
}

if (process.env.NODE_ENV === "development") {
  window.useFlowStore = useFlowStore;
}

// Disabled MSW mocking in favor of real Node.js backend
async function enableMocking() {
  return Promise.resolve();
}

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Failed to find root element");

void enableMocking().then(() => {
  createRoot(rootElement).render(
    <StrictMode>
      <ThemeProvider>
        <ReactFlowProvider>
          <App />
        </ReactFlowProvider>
      </ThemeProvider>
    </StrictMode>,
  );
});
