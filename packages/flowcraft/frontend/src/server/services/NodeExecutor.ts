import { fromBinary } from "@bufbuild/protobuf";

import { NodeDataSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { type NodeSignal } from "@/generated/flowcraft/v1/core/signals_pb";

import { getSpacetimeConn } from "../spacetimeClient";
import logger from "../utils/logger";
import { InstanceHost } from "./InstanceHost";
import { NodeInstance } from "./NodeInstance";

/**
 * Executes a specific action on a node.
 */
export function runAction(actionId: string, nodeId: string, params: unknown) {
  logger.info(`Running action ${actionId} on node ${nodeId}`);
  const instances = InstanceHost.getInstance().getInstancesForNode(nodeId);
  instances.forEach((instance) => {
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
 * Helper for worker to update a node's property via componentized system.
 */
export function updateNodeProperty(nodeId: string, path: string, value: any) {
  const conn = getSpacetimeConn();
  if (!conn) return;

  if (path === "presentation.position.x" || path === "presentation.position.y") {
    const transform = conn.db.nodeTransforms.nodeId.find(nodeId);
    if (transform) {
      const x = path === "presentation.position.x" ? Number(value) : transform.x;
      const y = path === "presentation.position.y" ? Number(value) : transform.y;
      conn.reducers.setNodePosition({ nodeId, x, y });
    }
  } else if (path === "state.displayName") {
    const existing = conn.db.nodeData.nodeId.find(nodeId);
    if (existing) {
      const state = fromBinary(NodeDataSchema, existing.state);
      state.displayName = String(value);
      conn.pbreducers.setNodeDataPb({ nodeId, state });
    }
  }
}

/**
 * Update widget values from instances.
 */
export function updateWidgetValue(nodeId: string, widgetId: string, value: any) {
  const conn = getSpacetimeConn();
  if (conn) {
    conn.reducers.updateWidgetValue({
      id: `${nodeId}-${widgetId}`,
      nodeId,
      value: JSON.stringify(value),
      widgetId,
    });
  }
}
