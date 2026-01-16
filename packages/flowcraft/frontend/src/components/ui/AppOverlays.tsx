import React from "react";
import { MediaPreview } from "../media/MediaPreview";
import { EditorPlaceholder } from "../media/EditorPlaceholder";
import { TaskHistoryDrawer } from "../TaskHistoryDrawer";
import { SideToolbar } from "../SideToolbar";
import { SettingsModal } from "../SettingsModal";
import { ActionParamsModal } from "../ActionParamsModal";
import { type AppNode } from "@/types";
import { SocketStatus } from "@/utils/SocketClient";
import { type ActionTemplate } from "@/generated/flowcraft/v1/core/action_pb";
import { type PreviewData } from "@/hooks/useNodeEventListener";

interface Props {
  nodes: AppNode[];
  previewData: PreviewData | null;
  setPreviewData: (d: PreviewData | null) => void;
  activeEditorId: string | null;
  setActiveEditorId: (id: string | null) => void;
  connectionStatus: SocketStatus;
  pendingAction: ActionTemplate | null;
  setPendingAction: (a: ActionTemplate | null) => void;
  onExecuteAction: (action: ActionTemplate, params?: Record<string, unknown>) => void;
}

export const AppOverlays: React.FC<Props> = ({
  nodes,
  previewData,
  setPreviewData,
  activeEditorId,
  setActiveEditorId,
  connectionStatus,
  pendingAction,
  setPendingAction,
  onExecuteAction,
}) => {
  return (
    <>
      {previewData &&
        nodes
          .filter((n) => n.id === previewData.nodeId)
          .map((node) => (
            <MediaPreview
              key={node.id}
              node={node}
              initialIndex={previewData.index}
              onClose={() => setPreviewData(null)}
            />
          ))}
      {activeEditorId &&
        nodes
          .filter((n) => n.id === activeEditorId)
          .map((node) => <EditorPlaceholder key={node.id} node={node} onClose={() => setActiveEditorId(null)} />)}
      <TaskHistoryDrawer />
      <SideToolbar connectionStatus={connectionStatus} />
      <SettingsModal />
      {pendingAction && (
        <ActionParamsModal
          action={pendingAction}
          onConfirm={(p) => onExecuteAction(pendingAction, p)}
          onCancel={() => setPendingAction(null)}
        />
      )}
    </>
  );
};
