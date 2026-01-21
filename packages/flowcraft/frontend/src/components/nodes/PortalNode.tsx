import { Handle, Position } from "@xyflow/react";
import { MoveUpRight } from "lucide-react";
import { memo } from "react";

export const PortalNode = memo(({ data }: any) => {
  return (
    <div className="px-3 py-1.5 rounded-full bg-primary/10 border border-primary/30 backdrop-blur-md shadow-sm flex items-center gap-2 group hover:bg-primary/20 transition-all">
      <div className="w-5 h-5 rounded-full bg-primary/20 flex items-center justify-center">
        <MoveUpRight className="text-primary" size={10} />
      </div>
      <span className="text-[10px] font-bold text-primary/80 uppercase tracking-wider">{data.label}</span>

      {/* 这是一个全能端点，既可以作为源也可以作为目标 */}
      <Handle
        className="w-2 h-2 !bg-primary border-none opacity-0 group-hover:opacity-100 transition-opacity"
        position={Position.Left}
        type="target"
      />
      <Handle
        className="w-2 h-2 !bg-primary border-none opacity-0 group-hover:opacity-100 transition-opacity"
        position={Position.Right}
        type="source"
      />
    </div>
  );
});
