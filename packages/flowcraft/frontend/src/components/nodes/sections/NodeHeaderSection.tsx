import { LayoutGrid, MessageSquare, MonitorPlay, Settings2 } from "lucide-react";
import { memo, useCallback } from "react";

import { RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { useNodeHandlers } from "@/hooks/nodes/useNodeHandlers";
import { useNodeProperty } from "@/hooks/core/useNodeProperty";
import { cn } from "@/lib/utils";
import { type DynamicNodeData } from "@/types";

import { NodeLabel } from "../NodeLabel";
import { PortLabelRow } from "../PortLabelRow";

interface NodeHeaderSectionProps {
  data: DynamicNodeData;
  id: string;
  selected?: boolean;
}

const MODE_ICONS: Record<number, any> = {
  [RenderMode.MODE_CHAT]: MessageSquare,
  [RenderMode.MODE_MARKDOWN]: LayoutGrid,
  [RenderMode.MODE_MEDIA]: MonitorPlay,
  [RenderMode.MODE_WIDGETS]: Settings2,
};

export const NodeHeaderSection = memo(({ data, id, selected }: NodeHeaderSectionProps) => {
  useNodeHandlers(data, selected);

  // Two-way binding for the active render mode
  const [activeMode, setActiveMode] = useNodeProperty("activeMode");

  const inputs = data.inputPorts ?? [];
  const outputs = data.outputPorts ?? [];
  const availableModes = data.availableModes ?? [];

  const handleModeChange = useCallback(
    (mode: RenderMode) => {
      setActiveMode(mode);
    },
    [setActiveMode],
  );

  return (
    <div className="flex flex-col border-b border-node-border bg-muted/20">
      <div className="flex items-center justify-between pr-2">
        <NodeLabel selected={selected} />

        {/* Render Mode Switcher */}
        {availableModes.length > 1 && (
          <div className="flex items-center gap-1 bg-background/50 rounded-md p-0.5 border border-border/50">
            {availableModes.map((mode) => {
              const Icon = MODE_ICONS[mode] || Settings2;
              const isActive = activeMode === mode;
              return (
                <button
                  className={cn(
                    "p-1 rounded transition-all hover:bg-primary/10",
                    isActive ? "text-primary bg-primary/10" : "text-muted-foreground opacity-50 hover:opacity-100",
                  )}
                  key={mode}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleModeChange(mode);
                  }}
                  title={`Switch to ${RenderMode[mode]}`}
                >
                  <Icon size={12} />
                </button>
              );
            })}
          </div>
        )}
      </div>

      <div className="flex flex-col">
        {/* Render port labels based on max count of inputs or outputs */}
        {Array.from({ length: Math.max(inputs.length, outputs.length) }).map((_, i) => (
          <PortLabelRow inputPort={inputs[i]} key={i} nodeId={id} outputPort={outputs[i]} />
        ))}
      </div>
    </div>
  );
});

NodeHeaderSection.displayName = "NodeHeaderSection";
