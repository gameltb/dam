import { fromJson, type JsonObject, toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { memo } from "react";

import { PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import { PortStyle, type Widget } from "@/generated/flowcraft/v1/core/node_pb";
import { useNodeHandlers } from "@/hooks/useNodeHandlers";
import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";
import { type DynamicNodeData } from "@/types";
import { getSchemaForTemplate } from "@/utils/schemaRegistry";
import { getPortColor } from "@/utils/themeUtils";

import { PortHandle } from "../base/PortHandle";
import { ChatBot } from "../media/ChatBot";
import { FlowcraftRJSF } from "../widgets/FlowcraftRJSF";
import { WIDGET_COMPONENTS } from "../widgets/widgetConfigs";
import { WidgetWrapper } from "../widgets/WidgetWrapper";
import { NodeLabel } from "./NodeLabel";
import { PortLabelRow } from "./PortLabelRow";

interface WidgetRendererProps {
  nodeId: string;
  onClick: () => void;
  onValueChange: (val: unknown) => void;
  widget: Widget;
}

const WidgetRenderer: React.FC<WidgetRendererProps> = memo(({ onClick, onValueChange, widget }) => {
  const Component = WIDGET_COMPONENTS[widget.type];
  if (!Component) return null;

  const jsValue = widget.value ? toJson(ValueSchema, widget.value) : undefined;

  return (
    <Component
      config={widget.config as any}
      label={widget.label}
      nodeId=""
      onChange={onValueChange}
      onClick={onClick}
      options={widget.options}
      value={jsValue}
    />
  );
});

import { useShallow } from "zustand/react/shallow";

const WidgetContentComponent: React.FC<{
  data: DynamicNodeData;
  id: string;
  onToggleMode: () => void;
  selected?: boolean;
}> = memo(({ data, id, onToggleMode, selected }) => {
  const { onChange, onWidgetClick } = useNodeHandlers(data, selected);
  const { activeChatNodeId, chatViewMode } = useUiStore(
    useShallow((s) => ({
      activeChatNodeId: s.activeChatNodeId,
      chatViewMode: s.chatViewMode,
    })),
  );
  const { allNodes, nodeDraft } = useFlowStore(
    useShallow((s) => ({
      allNodes: s.allNodes,
      nodeDraft: s.nodeDraft,
    })),
  );

  const isChatNode = data.templateId?.toLowerCase().includes("chat");
  const isSidebarMode = activeChatNodeId === id && chatViewMode === "sidebar";
  const isActiveExternally = isSidebarMode || (activeChatNodeId === id && chatViewMode === "fullscreen");

  const isSwitchable = data.availableModes.length > 1;
  const inputs = data.inputPorts ?? [];
  const outputs = data.outputPorts ?? [];

  return (
    <div className="flex flex-col min-w-[150px] flex-1">
      <NodeLabel
        id={id}
        label={data.displayName}
        onChange={(nodeId, val) => {
          onChange(nodeId, { displayName: val });
        }}
        selected={selected}
      />

      <div className="flex flex-col">
        {Array.from({ length: Math.max(inputs.length, outputs.length) }).map((_, i) => (
          <PortLabelRow inputPort={inputs[i]} key={i} nodeId={id} outputPort={outputs[i]} />
        ))}
      </div>

      <div className="flex flex-col gap-2 px-3 py-2 relative">
        {(() => {
          const schema = getSchemaForTemplate(data.templateId ?? "");
          if (!schema) return null;
          return (
            <div className="nodrag nopan">
              <FlowcraftRJSF
                formData={data.widgetsValues ?? {}}
                nodeId={id}
                onChange={(val) => {
                  const node = allNodes.find((n) => n.id === id);
                  if (node) {
                    const res = nodeDraft(node);
                    if (res.ok) {
                      (res.value.data as any).widgetsValues = val as JsonObject;
                    }
                  }
                }}
                schema={schema}
              />
            </div>
          );
        })()}

        {isChatNode && (
          <div className="flex flex-col gap-2 pt-2 border-t border-node-border mt-2">
            {!isActiveExternally ? (
              <div className="h-[300px] border border-node-border rounded-md overflow-hidden bg-background/50">
                <ChatBot nodeId={id} />
              </div>
            ) : (
              <div className="py-8 flex flex-col items-center justify-center border border-dashed border-border rounded-md bg-muted/10 text-[10px]">
                Active in remote mode
              </div>
            )}
          </div>
        )}

        {data.widgets?.map((w: any) => (
          <WidgetWrapper
            inputPortId={w.inputPortId}
            isSwitchable={isSwitchable}
            key={w.id}
            nodeId={id}
            onToggleMode={onToggleMode}
          >
            <div className="relative w-full">
              {w.inputPortId && (
                <PortHandle
                  color={getPortColor({ mainType: PortMainType.STRING } as any)}
                  isImplicit={true}
                  nodeId={id}
                  portId={w.inputPortId}
                  sideOffset={17}
                  style={PortStyle.CIRCLE}
                  type="target"
                />
              )}
              <WidgetRenderer
                nodeId={id}
                onClick={() => {
                  onWidgetClick?.(id, w.id);
                }}
                onValueChange={(val: any) => {
                  const node = allNodes.find((n) => n.id === id);
                  if (!node) return;

                  const res = nodeDraft(node);
                  if (res.ok) {
                    const draft = res.value;
                    const idx = (draft.data.widgets as any[]).findIndex((item: any) => item.id === w.id);
                    if (idx !== -1) {
                      (draft.data.widgets as any[])[idx].value = fromJson(ValueSchema, val);
                    }
                  }
                }}
                widget={w}
              />
            </div>
          </WidgetWrapper>
        ))}
      </div>
    </div>
  );
});

export const WidgetContent = Object.assign(WidgetContentComponent, {
  minSize: { height: 150, width: 200 },
});
