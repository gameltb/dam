import { useMemo } from "react";
import { SpacetimeDBProvider } from "spacetimedb/react";

import { DbConnection } from "@/generated/spacetime";
import { useUiStore } from "@/store/uiStore";

export const SpacetimeConnector = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const { serverAddress } = useUiStore((state) => state.settings);

  const builder = useMemo(() => {
    let uri = "";
    if (serverAddress === "/") {
      uri = `ws://${window.location.host}/spacetime/`;
    } else if (serverAddress.startsWith("http")) {
      const url = new URL(serverAddress);
      const protocol = url.protocol === "https:" ? "wss:" : "ws:";
      uri = `${protocol}//${url.host}/spacetime/`;
    } else {
      uri = `ws://${serverAddress}/spacetime/`;
    }

    console.log("[SpacetimeConnector] Connecting to:", uri);

    return DbConnection.builder()
      .withUri(uri)
      .withModuleName("flowcraft")
      .withToken("");
  }, [serverAddress]);

  return (
    <SpacetimeDBProvider connectionBuilder={builder}>
      {children}
    </SpacetimeDBProvider>
  );
};
