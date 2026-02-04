import { AlertCircle, Loader2, RefreshCcw } from "lucide-react";
import { memo } from "react";

interface NodeStatusOverlayProps {
  error?: null | string;
  isBusy: boolean;
  isError: boolean;
  message?: null | string;
  onReset?: () => void;
  progress?: number;
}

export const NodeStatusOverlay = memo(
  ({ error, isBusy, isError, message, onReset, progress }: NodeStatusOverlayProps) => {
    if (!isBusy && !isError) return null;

    return (
      <div className="absolute inset-0 z-[100] flex flex-col items-center justify-center rounded-[inherit] bg-background/60 backdrop-blur-[2px] p-4 text-center animate-in fade-in duration-200">
        {isBusy ? (
          <>
            <Loader2 className="w-6 h-6 text-primary animate-spin mb-2" />
            <div className="text-[10px] font-bold text-primary uppercase tracking-wider mb-1">
              {message || "Processing…"}
            </div>
            <div className="w-2/3 h-1 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          </>
        ) : (
          <>
            <AlertCircle className="w-6 h-6 text-destructive mb-2" />
            <div className="text-[10px] font-bold text-destructive uppercase mb-1">Execution Failed</div>
            <div className="text-[9px] text-muted-foreground line-clamp-2 px-2 italic">
              {error || "Unknown runtime error"}
            </div>
            {onReset && (
              <button
                className="mt-3 flex items-center gap-1.5 px-3 py-1.5 bg-destructive/10 hover:bg-destructive/20 text-destructive rounded-full text-[10px] font-medium transition-colors"
                onClick={(e) => {
                  e.stopPropagation();
                  onReset();
                }}
              >
                <RefreshCcw className="w-3 h-3" />
                Force Reset
              </button>
            )}
          </>
        )}
      </div>
    );
  },
);

NodeStatusOverlay.displayName = "NodeStatusOverlay";
