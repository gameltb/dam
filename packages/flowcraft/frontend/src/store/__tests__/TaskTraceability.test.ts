import { describe, it, beforeAll } from "vitest";
import { enablePatches } from "immer";

import { commit } from "../orchestrator";

beforeAll(() => {
  enablePatches();
});

describe("Task Traceability", () => {
  it("should trace mutations back to tasks", () => {
    commit(
      (_draft) => {
        // Test mutation
      },
      { description: "Test mutation", taskId: "test-task" },
    );
  });

  it("should handle user mutations without tasks", () => {
    commit((_draft) => {}, { description: "User cleared canvas" });
  });
});
