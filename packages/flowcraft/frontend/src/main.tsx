import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import ThemeProvider from "./ThemeProvider.tsx";
import { ReactFlowProvider } from "@xyflow/react";
import { initStoreOrchestrator } from "./store/orchestrator";
import { useFlowStore } from "./store/flowStore";

initStoreOrchestrator();

if (typeof window !== "undefined") {
  (window as any).useFlowStore = useFlowStore;
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
