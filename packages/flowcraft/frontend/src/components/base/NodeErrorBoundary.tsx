import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  nodeId?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class NodeErrorBoundary extends Component<Props, State> {
  public override state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public override componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error(
      `Error in Node [${this.props.nodeId || "unknown"}]:`,
      error,
      errorInfo,
    );
  }

  public override render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="flex flex-col items-center justify-center h-full w-full p-4 text-center bg-red-900/10 border border-red-500/50 rounded-lg overflow-hidden">
          <div className="text-red-500 mb-2">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <div className="text-xs font-bold text-red-400 mb-1">Node Error</div>
          <div className="text-[10px] text-red-300/70 truncate w-full">
            {this.state.error?.message || "Something went wrong"}
          </div>
          <button
            className="mt-2 text-[10px] bg-red-500/20 hover:bg-red-500/30 text-red-300 px-2 py-1 rounded transition-colors"
            onClick={() => {
              this.setState({ hasError: false, error: null });
            }}
          >
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
