import { GitGraph } from "lucide-react";
import React, { memo } from "react";

import { useNodeScope } from "@/hooks/useNodeScope";
import { useUiStore } from "@/store/uiStore";

import { Button } from "../ui/button";

export interface ScopeLensProps {
  data: any;
  id: string;
}

/**
 * 镜头模式 HOC
 * 自动为节点注入作用域控制能力
 */
export function withScopeLens<T extends ScopeLensProps>(WrappedComponent: React.ComponentType<T>) {
  return memo((props: T) => {
    const { data, id } = props;
    const activeScopeId = useUiStore((s) => s.activeScopeId);
    const { enter, scopeId } = useNodeScope(id, data.managedScopeId);

    const isActiveScope = activeScopeId === scopeId;

    // LOD: 进入状态显示简约背景
    if (isActiveScope) {
      return (
        <div className="w-full h-full border-2 border-dashed border-primary/20 rounded-xl flex items-center justify-center bg-primary/5 group">
          <div className="text-primary/10 group-hover:text-primary/30 transition-colors flex flex-col items-center">
            <GitGraph size={48} strokeWidth={1} />
            <span className="text-xs font-black uppercase tracking-[0.2em] mt-2">
              Viewing Scope: {data.displayName || id}
            </span>
          </div>
        </div>
      );
    }

    // 正常状态：渲染原始组件并注入控制条
    return (
      <div className="relative w-full h-full group/lens">
        <WrappedComponent {...props} />

        {/* 悬浮控制条 */}
        <div className="absolute top-2 right-2 opacity-0 group-hover/lens:opacity-100 transition-opacity z-50">
          <Button
            className="h-7 w-7 rounded-full shadow-lg border border-primary/20 bg-background/80 backdrop-blur"
            onClick={(e) => {
              e.stopPropagation();
              enter();
            }}
            size="icon"
            title="Enter Scope"
            variant="secondary"
          >
            <GitGraph className="text-primary" size={14} />
          </Button>
        </div>
      </div>
    );
  });
}
