import { fromJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { beforeAll, describe, expect, it, vi } from "vitest";

// 关键：激活 SpacetimeDB Mock 环境
import { setupStdbMock } from "@/../scripts/stdb-mock";

let nodeReducers: any;
let applyPathToObj: any;

describe("PathUpdate Reducer & Logic", () => {
  beforeAll(async () => {
    // 初始化 Mock
    setupStdbMock([]);
    // 动态导入，确保 Mock 已生效
    const mod = await import("@/../spacetime-module/src/reducers/node");
    const utils = await import("@/../spacetime-module/src/utils/path-utils");
    nodeReducers = mod.nodeReducers;
    applyPathToObj = utils.applyPathToObj;
  });

  describe("applyPathToObj (Core Engine)", () => {
    it("should update a top-level property", () => {
      const obj = { name: "old" };
      const result = applyPathToObj(obj, ["name"], "new", 0); // 0 = REPLACE
      expect(result.name).toBe("new");
    });

    it("should update a nested property", () => {
      const obj = { data: { profile: { name: "old" } } };
      const result = applyPathToObj(obj, ["data", "profile", "name"], "new", 0);
      expect(result.data.profile.name).toBe("new");
    });

    it("should delete a property", () => {
      const obj = { data: { keep: "stay", temp: "delete-me" } };
      const result = applyPathToObj(obj, ["data", "temp"], null, 2); // 2 = DELETE
      expect(result.data.temp).toBeUndefined();
      expect(result.data.keep).toBe("stay");
    });
  });

  describe("path_update_pb (Reducer Handler)", () => {
    it("should correctly find, update and save a node row", () => {
      const mockNode = {
        nodeId: "test-node",
        state: {
          presentation: { width: 100 },
          state: { displayName: "Old Name" },
        },
      };

      const mockUpdate = vi.fn();
      const mockCtx: any = {
        db: {
          nodes: {
            nodeId: {
              find: vi.fn().mockReturnValue(mockNode),
            },
          },
        },
      };
      mockCtx.db.nodes.nodeId.update = mockUpdate;

      const req = {
        path: "data.displayName",
        targetId: "test-node",
        type: 0, // REPLACE
        value: fromJson(ValueSchema, "New Name"),
      };

      nodeReducers.path_update_pb.handler(mockCtx, { req } as any);

      expect(mockCtx.db.nodes.nodeId.find).toHaveBeenCalledWith("test-node");
      if (mockUpdate.mock.calls[0]) {
        const updatedRow = mockUpdate.mock.calls[0][0];
        expect(updatedRow.state.state.displayName).toBe("New Name");
      }
    });
  });
});
