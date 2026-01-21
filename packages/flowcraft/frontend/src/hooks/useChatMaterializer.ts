import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { dagreLayout, materializerRegistry } from "@/utils/materializerRegistry";

/**
 * 初始化 Chat 物化配置
 */
export const initChatMaterializer = () => {
  materializerRegistry.register({
    getItemId: (item) => item.id,

    getItems: (activeScopeId) => {
      const allMessages = (window as any)._stdb_chat_messages || [];
      return allMessages.filter((m: any) => m.state.treeId === activeScopeId);
    },

    layout: (newItems, existingNodes, _activeScopeId) => {
      return dagreLayout(newItems, existingNodes, {
        getItemId: (m) => m.id,
        getItemParentId: (m) => m.state.parentId || null,
        getTemplateId: () => "flowcraft.node.chat.message",
        mapData: (m) => {
          const msgState = m.state;
          const content = msgState.parts && msgState.parts.length > 0 ? msgState.parts[0].part.value || "" : "";

          return {
            // 确保显式传递非零枚举值
            activeMode: Number(RenderMode.MODE_MEDIA),
            availableModes: [Number(RenderMode.MODE_MEDIA)],
            displayName: msgState.role.toUpperCase(),
            extension: {
              case: "document",
              value: {
                content: content,
                type: Number(MediaType.MEDIA_MARKDOWN),
              },
            },
            media: {
              aspectRatio: 1.33,
              content: content,
              galleryUrls: [],
              type: Number(MediaType.MEDIA_MARKDOWN),
              url: "",
            },
          };
        },
        nodeHeight: 300,
        nodeWidth: 400,
        rankdir: "LR",
      });
    },

    scopeType: "chat",
  });
};
