import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import ThemeProvider from "./ThemeProvider.tsx";
import { initOrchestrator } from "./store/orchestrator";

// Initialize cross-store side-effects
initOrchestrator();

async function enableMocking() {
  if (import.meta.env.DEV) {
    const { worker } = await import("./mocks/browser");
    return worker.start({
      onUnhandledRequest: "bypass",
    });
  }
  return Promise.resolve();
}

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Failed to find root element");

void enableMocking().then(() => {
  createRoot(rootElement).render(
    <StrictMode>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </StrictMode>,
  );
});
