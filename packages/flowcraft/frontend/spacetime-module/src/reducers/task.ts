import { t } from "spacetimedb/server";

export const taskReducers = {
  execute_action: {
    args: {
      actionId: t.string(),
      id: t.string(),
      nodeId: t.string(),
      paramsJson: t.string(),
    },
    handler: (ctx: any, args: any) => {
      ctx.db.tasks.insert({
        ...args,
        resultJson: "",
        status: "pending",
        timestamp: ctx.timestamp.toMillis(),
      });
    },
  },

  update_task_status: {
    args: {
      id: t.string(),
      resultJson: t.string(),
      status: t.string(),
    },
    handler: (ctx: any, { id, resultJson, status }: any) => {
      const task = ctx.db.tasks.id.find(id);
      if (task) {
        ctx.db.tasks.id.update({ ...task, resultJson, status });
      }
    },
  },

  send_node_signal: {
    args: {
      id: t.string(),
      nodeId: t.string(),
      payloadJson: t.string(),
      signalCase: t.string(),
    },
    handler: (ctx: any, args: any) => {
      ctx.db.nodeSignals.insert({
        ...args,
        timestamp: ctx.timestamp.toMillis(),
      });
    },
  },

  assign_current_task: {
    args: {
      taskId: t.string(),
    },
    handler: (ctx: any, { taskId }: any) => {
      const identity = ctx.sender.toHexString();
      const existing = ctx.db.clientTaskAssignments.clientIdentity.find(identity);
      if (existing) {
        ctx.db.clientTaskAssignments.clientIdentity.update({ clientIdentity: identity, taskId });
      } else {
        ctx.db.clientTaskAssignments.insert({ clientIdentity: identity, taskId });
      }
    },
  },
};
