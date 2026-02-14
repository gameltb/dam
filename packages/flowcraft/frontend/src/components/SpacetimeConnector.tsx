import { AlertCircle, Settings2 } from "lucide-react";
import React, { useMemo, useState } from "react";
import { SpacetimeDBProvider } from "spacetimedb/react";

import { DbConnection } from "@/generated/spacetime";
import { useSettingsStore } from "@/store/ui/settingsStore";
import { useUiStore } from "@/store/uiStore";

import { Button } from "./ui/button";

interface SpacetimeConnectorProps {
  children: React.ReactNode;
}

/**
 * Normalizes Server URL
 * 1. Handle relative paths (e.g., /spacetime/) -> Expand to full ws://host/spacetime/
 * 2. Ensure it ends with / (required by SpacetimeDB client)
 * 3. Automatically select ws/wss based on the current page protocol
 */
function normalizeUri(uri: string): string {
  if (!uri) return "";
  let target = uri.trim();

  // 1. Handle relative paths
  if (target.startsWith("/")) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    target = `${protocol}//${host}${target}`;
  }

  // 2. Ensure it ends with a slash
  if (!target.endsWith("/")) {
    target = `${target}/`;
  }

  // 3. Handle pure domain names without protocol (e.g., localhost:3000)
  if (!target.includes("://")) {
    target = `ws://${target}`;
  }

  return target;
}

/**
 * SpacetimeConnector (Resilient Version)
 */
export const SpacetimeConnector: React.FC<SpacetimeConnectorProps> = ({ children }) => {
  const serverAddress = useSettingsStore((s) => s.serverAddress);
  const setSettingsOpen = useUiStore((s) => s.setSettingsOpen);
  const [initError, setInitError] = useState<null | string>(null);

  const normalizedUri = useMemo(() => normalizeUri(serverAddress), [serverAddress]);

  const builder = useMemo(() => {
    try {
      setInitError(null);
      if (!normalizedUri || normalizedUri.length < 5) {
        throw new Error("Invalid or empty Server Address");
      }

      console.log(`[Spacetime] Attempting connection to: ${normalizedUri}`);
      return DbConnection.builder().withUri(normalizedUri).withModuleName("flowcraft");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to initialize SpacetimeDB builder";
      setInitError(msg);
      console.error(`[Spacetime] Init Error: ${msg}`);
      return null;
    }
  }, [normalizedUri]);

  return (
    <>
      {initError || !builder ? (
        <div className="fixed inset-0 flex items-center justify-center bg-background z-[9999]">
          <div className="max-w-md w-full p-8 border border-destructive/20 bg-destructive/5 rounded-2xl flex flex-col items-center text-center gap-6 shadow-2xl backdrop-blur-md animate-in zoom-in-95 duration-300">
            <div className="p-4 bg-destructive/10 rounded-full text-destructive">
              <AlertCircle size={40} />
            </div>
            <div className="space-y-2">
              <h2 className="text-xl font-bold">Connection Error</h2>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Could not connect to SpacetimeDB. Please verify your server address:
                <code className="block mt-2 p-2 bg-black/20 rounded font-mono text-xs text-foreground break-all">
                  {serverAddress || "(empty)"}
                </code>
              </p>
              {initError && <p className="text-xs text-destructive font-mono mt-2 italic">Reason: {initError}</p>}
            </div>
            <Button
              className="w-full gap-2 py-6 text-base shadow-lg"
              onClick={() => {
                setSettingsOpen(true);
              }}
            >
              <Settings2 size={18} />
              Open Settings to Fix
            </Button>
            <div className="flex flex-col gap-1 items-center">
              <p className="text-[10px] uppercase tracking-widest opacity-30 font-bold">Recovery Environment</p>
              <button
                className="text-[10px] text-primary hover:underline font-bold"
                onClick={() => {
                  window.location.reload();
                }}
              >
                Retry Connection
              </button>
            </div>
          </div>
        </div>
      ) : (
        <SpacetimeDBProvider connectionBuilder={builder}>{children}</SpacetimeDBProvider>
      )}

      {/* 
        CRITICAL: If we are in error mode, we still need to render the settings modal container 
        so that it can be opened. Since SettingsModal is inside App (children), 
        we render children in a hidden port if error occurs, OR we ensure it's outside.
        Actually, the most robust way is to render children but wrapped in a 'disabled' context.
      */}
      {(initError !== null || !builder) && (
        <div className="pointer-events-none opacity-0 absolute inset-0 -z-10 overflow-hidden">{children}</div>
      )}
    </>
  );
};
