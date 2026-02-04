import React, { useCallback, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

interface MarkdownEditorProps {
  autoFocus?: boolean;
  className?: string;
  onBlur?: () => void;
  onChange: (value: string) => void;
  onKeyDown?: (e: React.KeyboardEvent) => void;
  onSave?: () => void;
  onCancel?: () => void;
  value: string;
}

/**
 * A standalone Markdown Editor component with auto-growing textarea
 * and keyboard shortcut support.
 */
export const MarkdownEditor: React.FC<MarkdownEditorProps> = ({
  autoFocus = true,
  className,
  onBlur,
  onChange,
  onKeyDown,
  onSave,
  onCancel,
  value,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "inherit";
      el.style.height = `${el.scrollHeight}px`;
    }
  }, []);

  useEffect(() => {
    adjustHeight();
  }, [value, adjustHeight]);

  useEffect(() => {
    if (autoFocus && textareaRef.current) {
      textareaRef.current.focus();
      // Move cursor to end
      textareaRef.current.setSelectionRange(value.length, value.length);
    }
  }, [autoFocus, value.length]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (onKeyDown) onKeyDown(e);

    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      onSave?.();
    } else if (e.key === "Escape") {
      e.preventDefault();
      onCancel?.();
    }
  };

  return (
    <textarea
      className={cn(
        "w-full bg-background border border-primary/30 rounded p-2 outline-none resize-none nodrag nowheel font-mono text-sm leading-relaxed transition-colors focus:border-primary/60",
        className,
      )}
      onBlur={onBlur}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={handleKeyDown}
      ref={textareaRef}
      value={value}
    />
  );
};
