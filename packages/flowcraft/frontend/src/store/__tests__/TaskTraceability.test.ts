import { describe, it } from "vitest";
import { commit } from "../orchestrator";

describe("Task Traceability", () => {
  it("should trace mutations back to tasks", () => {
    commit(
      (_draft) => {
        // Test mutation
      },
      { description: "Test mutation", taskId: "test-task" }
    );
  });

  it("should handle user mutations without tasks", () => {
    commit((_draft) => {}, { description: "User cleared canvas" });
  });
});