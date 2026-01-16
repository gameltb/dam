import { type JsonObject, create, fromJson, toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { memo } from "react";

import { PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import { PortStyle, WidgetSchema, type Widget } from "@/generated/flowcraft/v1/core/node_pb";
import { useFlowSocket } from "@/hooks/useFlowSocket";
import { useNodeHandlers } from "@/hooks/useNodeHandlers";
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

const WidgetRenderer: React.FC<WidgetRendererProps> = memo(({ nodeId, onClick, onValueChange, widget }) => {
  const { updateWidget } = useFlowSocket({ disablePolling: true });

  const handleValueChange = (val: unknown) => {
    onValueChange(val);
    updateWidget(nodeId, widget.id, String(val));
  };

  const Component = WIDGET_COMPONENTS[widget.type];
  if (!Component) return null;

  // Convert PB Value to JS value for the component
  const jsValue = widget.value ? toJson(ValueSchema, widget.value) : undefined;

  return (
    <Component
      config={widget.config as any}
      label={widget.label}
      nodeId={nodeId}
      onChange={handleValueChange}
      onClick={onClick}
      options={widget.options}
      value={jsValue}
    />
  );
});

export const WidgetContent: React.FC<{
  data: DynamicNodeData;
  id: string;
  onToggleMode: () => void;
  selected?: boolean;
}> = memo(({ data, id, onToggleMode, selected }) => {
  const { onChange, onWidgetClick } = useNodeHandlers(data, selected);
  const { activeChatNodeId, chatViewMode } = useUiStore();

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
          const schema = getSchemaForTemplate(data.templateId ?? "", data.widgetsSchema);
          if (!schema) return null;
          return (
            <div className="nodrag nopan">
              <FlowcraftRJSF
                formData={data.widgetsValues ?? {}}
                nodeId={id}
                onChange={(val) => {
                  onChange(id, { widgetsValues: val as JsonObject });
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

        {data.widgets?.map((w) => (
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
                  onWidgetClick(id, w.id);
                }}
                onValueChange={(val) => {
                  const updated = data.widgets.map((item) =>
                    item.id === w.id
                      ? create(WidgetSchema, {
                          ...item,
                          value: fromJson(ValueSchema, val as any),
                        })
                      : item,
                  );
                  onChange(id, { widgets: updated });
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
