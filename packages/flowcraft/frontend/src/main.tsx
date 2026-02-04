import { ReactFlowProvider } from "@xyflow/react";
import { enablePatches } from "immer";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./index.css";
import { SpacetimeConnector } from "@/components/SpacetimeConnector";
import { useFlowStore } from "@/store/flowStore";
import { initStoreOrchestrator } from "@/store/orchestrator";
import { initGlobal } from "@/utils/initGlobal";

import App from "./App.tsx";
import { ThemeProvider } from "./ThemeProvider.tsx";

initGlobal();
initStoreOrchestrator();
enablePatches();

declare global {
  interface Window {
    useFlowStore: typeof useFlowStore;
  }
}

if (process.env.NODE_ENV === "development") {
  window.useFlowStore = useFlowStore;
}

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Failed to find root element");

createRoot(rootElement).render(
  <StrictMode>
    <ThemeProvider>
      <ReactFlowProvider>
        <SpacetimeConnector>
          <App />
        </SpacetimeConnector>
      </ReactFlowProvider>
    </ThemeProvider>
  </StrictMode>,
);