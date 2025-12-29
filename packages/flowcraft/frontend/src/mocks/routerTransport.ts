import { createRouterTransport } from "@connectrpc/connect";
import { flowServiceImpl } from "./flowServiceImpl";

export const routerTransport = createRouterTransport(flowServiceImpl);
