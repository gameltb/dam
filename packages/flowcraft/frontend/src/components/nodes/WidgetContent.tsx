import { type JsonObject } from "@bufbuild/protobuf";
import { Maximize2, MessageSquare, PanelRight } from "lucide-react";
import React, { memo } from "react";

import { PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import { PortStyle } from "@/generated/flowcraft/v1/core/node_pb";
import { useFlowSocket } from "@/hooks/useFlowSocket";
import { useNodeHandlers } from "@/hooks/useNodeHandlers";
import { cn } from "@/lib/utils";
import { useUiStore } from "@/store/uiStore";
import { ChatViewMode, type DynamicNodeData, type WidgetDef } from "@/types";
import { getSchemaForTemplate } from "@/utils/schemaRegistry";
import { getPortColor } from "@/utils/themeUtils";

import { PortHandle } from "../base/PortHandle";
import { ChatBot } from "../media/ChatBot";
import { Button } from "../ui/button";
import { FlowcraftRJSF } from "../widgets/FlowcraftRJSF";
import { WIDGET_COMPONENTS } from "../widgets/widgetConfigs";
import { WidgetWrapper } from "../widgets/WidgetWrapper";
import { NodeLabel } from "./NodeLabel";
import { PortLabelRow } from "./PortLabelRow";

interface WidgetRendererProps {
  nodeId: string;
  onClick: () => void;
  onValueChange: (val: unknown) => void;
  widget: WidgetDef;
}

const WidgetRenderer: React.FC<WidgetRendererProps> = memo(
  ({ nodeId, onClick, onValueChange, widget }) => {
    const { updateWidget } = useFlowSocket({ disablePolling: true });

    const handleValueChange = (val: unknown) => {
      onValueChange(val);
      updateWidget(nodeId, widget.id, val);
    };

    const Component = WIDGET_COMPONENTS[widget.type];
    if (!Component) return null;

    return (
      <Component
        config={widget.config}
        label={widget.label}
        nodeId={nodeId}
        onChange={handleValueChange}
        onClick={onClick}
        options={widget.options}
        value={widget.value}
      />
    );
  },
);

interface WidgetContentProps {
  data: DynamicNodeData;
  id: string;
  onToggleMode: () => void;
  selected?: boolean;
}

export const WidgetContent: React.FC<WidgetContentProps> = memo(
  ({ data, id, onToggleMode, selected }) => {
    const { onChange, onWidgetClick } = useNodeHandlers(data, selected);
    const { activeChatNodeId, chatViewMode, setActiveChat } = useUiStore();

    const isChatNode = data.typeId?.toLowerCase().includes("chat");
    const isSidebarMode =
      activeChatNodeId === id && chatViewMode === ChatViewMode.SIDEBAR;
    const isFullscreenMode =
      activeChatNodeId === id && chatViewMode === ChatViewMode.FULLSCREEN;
    const isActiveExternally = isSidebarMode || isFullscreenMode;

    const isSwitchable = data.modes.length > 1;

    const inputs = data.inputPorts ?? [];
    const outputs = data.outputPorts ?? [];
    const rowCount = Math.max(inputs.length, outputs.length);
    const portRows = [];

    for (let i = 0; i < rowCount; i++) {
      portRows.push({ input: inputs[i], output: outputs[i] });
    }

    return (
      <div className="flex flex-col min-w-[150px] flex-1">
        <NodeLabel
          id={id}
          label={data.label}
          onChange={(nodeId, label) => {
            onChange(nodeId, { label });
          }}
          selected={selected}
        />

        <div className="flex flex-col">
          {portRows.map((row, idx) => (
            <PortLabelRow
              inputPort={row.input}
              key={idx}
              nodeId={id}
              outputPort={row.output}
            />
          ))}
        </div>

        <div className="flex flex-col gap-2 px-3 py-2 relative">
          {/* Schema-Driven Widgets (RJSF) */}
          {(() => {
            const schema = getSchemaForTemplate(
              data.typeId ?? "",
              data.widgetsSchema as JsonObject,
            );
            if (!schema) return null;

            return (
              <div className="nodrag nopan">
                <FlowcraftRJSF
                  formData={data.widgetsValues ?? {}}
                  nodeId={id}
                  onChange={(newValues) => {
                    onChange(id, { widgetsValues: newValues });
                  }}
                  schema={schema}
                />
              </div>
            );
          })()}

          {/* AI Chat Controls and Inline Chat */}
          {isChatNode && (
            <div className="flex flex-col gap-2 pt-2 border-t border-node-border mt-2">
              <div className="flex justify-between items-center px-1">
                <span className="text-[10px] font-bold uppercase text-muted-foreground flex items-center gap-1">
                  <MessageSquare size={10} /> AI Session
                </span>
                <div className="flex gap-1">
                  <Button
                    className="h-6 w-6"
                    onClick={() => {
                      setActiveChat(id, ChatViewMode.FULLSCREEN);
                    }}
                    size="icon"
                    title="Open Fullscreen"
                    variant="ghost"
                  >
                    <Maximize2 size={12} />
                  </Button>
                  <Button
                    className={cn(
                      "h-6 w-6",
                      isSidebarMode && "text-primary-color bg-primary-color/10",
                    )}
                    onClick={() => {
                      setActiveChat(
                        isActiveExternally ? null : id,
                        ChatViewMode.SIDEBAR,
                      );
                    }}
                    size="icon"
                    title={
                      isSidebarMode ? "Dock back to node" : "Open in sidebar"
                    }
                    variant="ghost"
                  >
                    <PanelRight size={14} />
                  </Button>
                </div>
              </div>

              {!isActiveExternally ? (
                <div className="h-[300px] border border-node-border rounded-md overflow-hidden bg-background/50">
                  <ChatBot nodeId={id} />
                </div>
              ) : (
                <div className="py-8 flex flex-col items-center justify-center border border-dashed border-border rounded-md bg-muted/10">
                  <p className="text-[10px] text-muted-foreground">
                    Active in {chatViewMode} mode
                  </p>
                  <Button
                    className="h-auto p-0 text-[10px]"
                    onClick={() => {
                      setActiveChat(id, ChatViewMode.INLINE);
                    }}
                    variant="link"
                  >
                    Restore to node
                  </Button>
                </div>
              )}
            </div>
          )}

          {/* Traditional Hardcoded Widgets */}
          {data.widgets?.map((w) => (
            <div className="relative w-full" key={w.id}>
              <WidgetWrapper
                inputPortId={w.inputPortId}
                isSwitchable={isSwitchable}
                nodeId={id}
                onToggleMode={onToggleMode}
              >
                <div className="relative w-full">
                  {w.inputPortId && (
                    <PortHandle
                      color={getPortColor({
                        isGeneric: false,
                        itemType: "",
                        mainType: PortMainType.STRING,
                      })}
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
                      onWidgetClick(id, w.id);
                    }}
                    onValueChange={(val) => {
                      const updatedWidgets = (data.widgets ?? []).map((item) =>
                        item.id === w.id ? { ...item, value: val } : item,
                      );

                      onChange(id, { widgets: updatedWidgets });
                    }}
                    widget={w}
                  />
                </div>
              </WidgetWrapper>
            </div>
          ))}
        </div>
      </div>
    );
  },
);
