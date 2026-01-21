import { create, fromJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";

import { PathUpdateRequest_UpdateType, PathUpdateRequestSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { type NodeSignal } from "@/generated/flowcraft/v1/core/signals_pb";

import { getSpacetimeConn } from "../spacetimeClient";
import logger from "../utils/logger";
import { InstanceHost } from "./InstanceHost";
import { NodeInstance } from "./NodeInstance";

/**
 * Executes a specific action on a node.
 */
export function runAction(actionId: string, nodeId: string, params: any) {
  logger.info(`Running action ${actionId} on node ${nodeId}`);
  // In our mock, runAction is handled by instances or by service directly.
  // For now, if instance exists, we notify it.
  const instances = InstanceHost.getInstance().getInstancesForNode(nodeId);
  instances.forEach((instance) => {
    // Check if it's a NodeInstance before calling signal-like methods
    if (instance instanceof NodeInstance) {
      void instance.handleSignal({ case: actionId, value: params });
    }
  });
}

/**
 * Dispatches a signal to a node instance.
 */
export function runNodeSignal(nodeId: string, payload: NodeSignal["payload"]) {
  const instances = InstanceHost.getInstance().getInstancesForNode(nodeId);
  instances.forEach((instance) => {
    if (instance instanceof NodeInstance) {
      void instance.handleSignal(payload);
    }
  });
}

/**
 * Helper for worker to update a node's property via the new Path-based system.
 */
export function updateNodeProperty(nodeId: string, path: string, value: any) {
  const conn = getSpacetimeConn();
  if (conn) {
    conn.pbreducers.pathUpdatePb({
      req: create(PathUpdateRequestSchema, {
        path: path,
        targetId: nodeId,
        type: PathUpdateRequest_UpdateType.REPLACE,
        value: fromJson(ValueSchema, value),
      }),
    });
  }
}

/**
 * Legacy bridge for updating widget values from instances.
 */
export function updateWidgetValue(nodeId: string, widgetId: string, value: any) {
  updateNodeProperty(nodeId, `data.widgetsValues.${widgetId}`, value);
}
