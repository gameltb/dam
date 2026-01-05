import React, { memo } from "react";
import { PortStyle } from "../../generated/flowcraft/v1/core/node_pb";
import type { WidgetDef, DynamicNodeData } from "../../types";
import { WidgetWrapper } from "../widgets/WidgetWrapper";
import { PortHandle } from "../base/PortHandle";
import { useFlowSocket } from "../../hooks/useFlowSocket";
import { NodeLabel } from "./NodeLabel";
import { PortLabelRow } from "./PortLabelRow";
import { useNodeHandlers } from "../../hooks/useNodeHandlers";
import { FlowcraftRJSF } from "../widgets/FlowcraftRJSF";
import { getSchemaForTemplate } from "../../utils/schemaRegistry";
import { getPortColor } from "../../utils/themeUtils";
import { WIDGET_COMPONENTS } from "../widgets/widgetConfigs";
import { ChatBot } from "../media/ChatBot";
import { useUiStore } from "../../store/uiStore";
import { PanelRight, MessageSquare, Maximize2 } from "lucide-react";
import { Button } from "../ui/button";
import { cn } from "../../lib/utils";

interface WidgetRendererProps {
  nodeId: string;
  widget: WidgetDef;
  onValueChange: (val: unknown) => void;
  onClick: () => void;
}

const WidgetRenderer: React.FC<WidgetRendererProps> = memo(
  ({ nodeId, widget, onValueChange, onClick }) => {
    const { updateWidget } = useFlowSocket({ disablePolling: true });

    const handleValueChange = (val: unknown) => {
      onValueChange(val);
      updateWidget(nodeId, widget.id, val);
    };

    const Component = WIDGET_COMPONENTS[widget.type];
    if (!Component) return null;

    return (
      <Component
        nodeId={nodeId}
        value={widget.value}
        label={widget.label}
        options={widget.options}
        config={widget.config}
        onChange={handleValueChange}
        onClick={onClick}
      />
    );
  },
);

interface WidgetContentProps {
  id: string;
  data: DynamicNodeData;
  selected?: boolean;
  onToggleMode: () => void;
}

export const WidgetContent: React.FC<WidgetContentProps> = memo(
  ({ id, data, selected, onToggleMode }) => {
    const { onChange, onWidgetClick } = useNodeHandlers(data, selected);
    const { activeChatNodeId, chatViewMode, setActiveChat } = useUiStore();
    
    const isChatNode = data.typeId?.toLowerCase().includes("chat");
    const isSidebarMode = activeChatNodeId === id && chatViewMode === "sidebar";
    const isFullscreenMode = activeChatNodeId === id && chatViewMode === "fullscreen";
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
          selected={selected}
          onChange={(nodeId, label) => {
            onChange(nodeId, { label });
          }}
        />

        <div className="flex flex-col">
          {portRows.map((row, idx) => (
            <PortLabelRow
              key={idx}
              nodeId={id}
              inputPort={row.input}
              outputPort={row.output}
            />
          ))}
        </div>

        <div className="flex flex-col gap-2 px-3 py-2 relative">
          {/* Schema-Driven Widgets (RJSF) */}
          {(() => {
            const schema = getSchemaForTemplate(
              data.typeId || "",
              data.widgetsSchemaJson,
            );
            if (!schema) return null;

            return (
              <div className="nodrag nopan">
                <FlowcraftRJSF
                  nodeId={id}
                  schema={schema}
                  formData={data.widgetsValues || {}}
                  onChange={(newValues) => {
                    onChange(id, { widgetsValues: newValues });
                  }}
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
                    variant="ghost" 
                    size="icon" 
                    className="h-6 w-6"
                    onClick={() => setActiveChat(id, "fullscreen")}
                    title="Open Fullscreen"
                  >
                    <Maximize2 size={12} />
                  </Button>
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className={cn("h-6 w-6", isSidebarMode && "text-primary-color bg-primary-color/10")}
                    onClick={() => setActiveChat(isActiveExternally ? null : id, "sidebar")}
                    title={isSidebarMode ? "Dock back to node" : "Open in sidebar"}
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
                   <p className="text-[10px] text-muted-foreground">Active in {chatViewMode} mode</p>
                   <Button variant="link" className="h-auto p-0 text-[10px]" onClick={() => setActiveChat(id, "inline")}>
                     Restore to node
                   </Button>
                </div>
              )}
            </div>
          )}

          {/* Traditional Hardcoded Widgets */}
          {data.widgets?.map((w) => (
            <div key={w.id} className="relative w-full">
              <WidgetWrapper
                isSwitchable={isSwitchable}
                onToggleMode={onToggleMode}
                inputPortId={w.inputPortId}
                nodeId={id}
              >
                <div className="relative w-full">
                  {w.inputPortId && (
                    <PortHandle
                      nodeId={id}
                      portId={w.inputPortId}
                      type="target"
                      sideOffset={17}
                      style={PortStyle.CIRCLE}
                      color={getPortColor({
                        mainType: "string",
                        itemType: "",
                        isGeneric: false,
                      })}
                      isImplicit={true}
                    />
                  )}
                  <WidgetRenderer
                    nodeId={id}
                    widget={w}
                    onValueChange={(val) => {
                      const updatedWidgets = (data.widgets ?? []).map((item) =>
                        item.id === w.id ? { ...item, value: val } : item,
                      );

                      onChange(id, { widgets: updatedWidgets });
                    }}
                    onClick={() => {
                      onWidgetClick(id, w.id);
                    }}
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
