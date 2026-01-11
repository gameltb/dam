import { schema } from "spacetimedb/server";

import { chatReducers } from "./reducers/chat";
import { nodeReducers } from "./reducers/node";
import { taskReducers } from "./reducers/task";
import { edges, nodes, viewportState } from "./tables/base";
import { chatContents, chatMessages, chatStreams } from "./tables/chat";
import {
  clientTaskAssignments,
  nodeSignals,
  operationLogs,
  tasks,
  widgetValues,
} from "./tables/task";

/**
 * Flowcraft SpacetimeDB Schema
 */
const spacetimedb = schema(
  nodes,
  edges,
  viewportState,
  chatMessages,
  chatContents,
  tasks,
  nodeSignals,
  widgetValues,
  chatStreams,
  operationLogs,
  clientTaskAssignments,
);

// Explicit Static Registrations for the Code Generator
spacetimedb.reducer(
  "create_node",
  nodeReducers.create_node.args,
  nodeReducers.create_node.handler,
);
spacetimedb.reducer(
  "update_node_data",
  nodeReducers.update_node_data.args,
  nodeReducers.update_node_data.handler,
);
spacetimedb.reducer(
  "move_node",
  nodeReducers.move_node.args,
  nodeReducers.move_node.handler,
);
spacetimedb.reducer(
  "remove_node",
  nodeReducers.remove_node.args,
  nodeReducers.remove_node.handler,
);
spacetimedb.reducer(
  "update_node_layout",
  nodeReducers.update_node_layout.args,
  nodeReducers.update_node_layout.handler,
);
spacetimedb.reducer(
  "update_widget_value",
  nodeReducers.update_widget_value.args,
  nodeReducers.update_widget_value.handler,
);
spacetimedb.reducer(
  "add_edge",
  nodeReducers.add_edge.args,
  nodeReducers.add_edge.handler,
);
spacetimedb.reducer(
  "remove_edge",
  nodeReducers.remove_edge.args,
  nodeReducers.remove_edge.handler,
);
spacetimedb.reducer(
  "update_viewport",
  nodeReducers.update_viewport.args,
  nodeReducers.update_viewport.handler,
);

spacetimedb.reducer(
  "add_chat_message",
  chatReducers.add_chat_message.args,
  chatReducers.add_chat_message.handler,
);
spacetimedb.reducer(
  "clear_chat_history",
  chatReducers.clear_chat_history.args,
  chatReducers.clear_chat_history.handler,
);
spacetimedb.reducer(
  "update_chat_stream",
  chatReducers.update_chat_stream.args,
  chatReducers.update_chat_stream.handler,
);

spacetimedb.reducer(
  "execute_action",
  taskReducers.execute_action.args,
  taskReducers.execute_action.handler,
);
spacetimedb.reducer(
  "update_task_status",
  taskReducers.update_task_status.args,
  taskReducers.update_task_status.handler,
);
spacetimedb.reducer(
  "send_node_signal",
  taskReducers.send_node_signal.args,
  taskReducers.send_node_signal.handler,
);
spacetimedb.reducer(
  "assign_current_task",
  taskReducers.assign_current_task.args,
  taskReducers.assign_current_task.handler,
);

spacetimedb.clientDisconnected((ctx) => {
  const identity = ctx.sender.toHexString();
  const existing = ctx.db.clientTaskAssignments.clientIdentity.find(identity);
  if (existing) {
    ctx.db.clientTaskAssignments.clientIdentity.delete(identity);
  }
});

export default spacetimedb;