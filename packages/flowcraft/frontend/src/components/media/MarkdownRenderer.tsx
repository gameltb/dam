import React, { useCallback, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";

import { cn } from "@/lib/utils";

import { MarkdownEditor } from "../ui/MarkdownEditor";

interface MarkdownRendererProps {
  className?: string;
  content: string;
  isEditing?: boolean;
  onEdit?: (newContent: string) => void;
  onToggleEditing?: (editing: boolean) => void;
  readOnly?: boolean;
}

/**
 * A component that renders Markdown content and can switch to an editing mode.
 */
export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  className,
  content,
  isEditing: externalIsEditing,
  onEdit,
  onToggleEditing,
  readOnly = false,
}) => {
  const [internalIsEditing, setInternalIsEditing] = useState(false);
  const isEditing = externalIsEditing !== undefined ? externalIsEditing : internalIsEditing;

  const [editValue, setEditValue] = useState(content);

  useEffect(() => {
    setEditValue(content);
  }, [content]);

  const startEditing = useCallback(() => {
    if (readOnly) return;
    if (onToggleEditing) {
      onToggleEditing(true);
    } else {
      setInternalIsEditing(true);
    }
  }, [readOnly, onToggleEditing]);

  const saveEdit = useCallback(() => {
    if (onEdit && editValue !== content) {
      onEdit(editValue);
    }
    if (onToggleEditing) {
      onToggleEditing(false);
    } else {
      setInternalIsEditing(false);
    }
  }, [editValue, content, onEdit, onToggleEditing]);

  const cancelEdit = useCallback(() => {
    setEditValue(content);
    if (onToggleEditing) {
      onToggleEditing(false);
    } else {
      setInternalIsEditing(false);
    }
  }, [content, onToggleEditing]);

  const isEmpty = !content && !isEditing;

  return (
    <div
      className={cn("w-full h-full min-h-[1.5em]", className)}
      onDoubleClick={(e) => {
        if (!readOnly && !isEditing) {
          e.stopPropagation();
          startEditing();
        }
      }}
    >
      {isEditing ? (
        <MarkdownEditor
          onBlur={saveEdit}
          onCancel={cancelEdit}
          onChange={setEditValue}
          onSave={saveEdit}
          value={editValue}
        />
      ) : isEmpty ? (
        <div className="italic opacity-30 select-none py-2 px-2">Empty Markdown</div>
      ) : (
        <div className="prose prose-invert prose-sm max-w-none break-words px-2 py-1">
          <ReactMarkdown>{content}</ReactMarkdown>
        </div>
      )}
    </div>
  );
};
