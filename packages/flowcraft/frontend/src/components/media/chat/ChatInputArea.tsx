import { CheckIcon, GlobeIcon } from "lucide-react";
import React, { useState } from "react";

import {
  ModelSelector,
  ModelSelectorContent,
  ModelSelectorGroup,
  ModelSelectorItem,
  ModelSelectorList,
  ModelSelectorName,
  ModelSelectorTrigger,
} from "../../ai-elements/model-selector";
import {
  PromptInput,
  PromptInputActionAddAttachments,
  PromptInputActionMenu,
  PromptInputActionMenuContent,
  PromptInputActionMenuTrigger,
  PromptInputAttachment,
  PromptInputAttachments,
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  type PromptInputMessage,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from "../../ai-elements/prompt-input";
import { Suggestion, Suggestions } from "../../ai-elements/suggestion";
import { type ChatStatus, type ContextNode } from "./types";

const MODELS = [
  { chefSlug: "openai", id: "gpt-4o", name: "GPT-4o", providers: ["openai"] },
  {
    chefSlug: "openai",
    id: "gpt-4o-mini",
    name: "GPT-4o Mini",
    providers: ["openai"],
  },
  {
    chefSlug: "anthropic",
    id: "claude-3-5-sonnet",
    name: "Claude 3.5 Sonnet",
    providers: ["anthropic"],
  },
];

const SUGGESTIONS = [
  "Explain how this graph works",
  "Summarize current results",
  "Optimize this workflow",
];

interface Props {
  droppedNodes: ContextNode[];
  onModelChange: (m: string) => void;
  onSubmit: (msg: PromptInputMessage, model: string, search: boolean) => void;
  onWebSearchChange: (v: boolean) => void;
  selectedModel: string;
  setDroppedNodes: (n: ContextNode[]) => void;
  status: ChatStatus;
  useWebSearch: boolean;
}

export const ChatInputArea: React.FC<Props> = ({
  droppedNodes,
  onModelChange,
  onSubmit,
  onWebSearchChange,
  selectedModel,
  setDroppedNodes,
  status,
  useWebSearch,
}) => {
  const [inputText, setInputText] = useState("");
  const [modelSelectorOpen, setModelSelectorOpen] = useState(false);

  const handleSubmit = (msg: PromptInputMessage) => {
    onSubmit(msg, selectedModel, useWebSearch);
    setInputText("");
  };

  return (
    <div className="shrink-0 bg-muted/5 border-t border-node-border">
      <div className="grid gap-2 pt-2">
        {droppedNodes.length > 0 && (
          <div className="flex flex-wrap gap-2 px-2">
            {droppedNodes.map((n) => (
              <div
                className="group flex items-center gap-1.5 bg-primary/10 text-primary px-2 py-1 rounded-md text-xs border border-primary/20"
                key={n.id}
              >
                <span>{n.label}</span>
                <button
                  className="hover:text-destructive"
                  onClick={() => {
                    setDroppedNodes(droppedNodes.filter((i) => i.id !== n.id));
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <Suggestions className="px-2">
          {SUGGESTIONS.map((s) => (
            <Suggestion
              key={s}
              onClick={() => {
                onSubmit({ files: [], text: s }, selectedModel, useWebSearch);
              }}
              suggestion={s}
            />
          ))}
        </Suggestions>

        <div className="px-2 pb-2">
          <PromptInput onSubmit={handleSubmit}>
            <PromptInputAttachments>
              {(f) => <PromptInputAttachment data={f} key={f.id} />}
            </PromptInputAttachments>
            <PromptInputBody>
              <PromptInputTextarea
                className="min-h-[44px]"
                onChange={(e) => {
                  setInputText(e.target.value);
                }}
                placeholder="Ask anything..."
                value={inputText}
              />
            </PromptInputBody>
            <PromptInputFooter>
              <PromptInputTools>
                <PromptInputActionMenu>
                  <PromptInputActionMenuTrigger />
                  <PromptInputActionMenuContent>
                    <PromptInputActionAddAttachments label="Upload" />
                  </PromptInputActionMenuContent>
                </PromptInputActionMenu>
                <PromptInputButton
                  onClick={() => {
                    onWebSearchChange(!useWebSearch);
                  }}
                  variant={useWebSearch ? "default" : "ghost"}
                >
                  <GlobeIcon size={14} />
                </PromptInputButton>
                <ModelSelector
                  onOpenChange={setModelSelectorOpen}
                  open={modelSelectorOpen}
                >
                  <ModelSelectorTrigger asChild>
                    <PromptInputButton>
                      <span>
                        {MODELS.find((m) => m.id === selectedModel)?.name}
                      </span>
                    </PromptInputButton>
                  </ModelSelectorTrigger>
                  <ModelSelectorContent>
                    <ModelSelectorList>
                      <ModelSelectorGroup heading="Models">
                        {MODELS.map((m) => (
                          <ModelSelectorItem
                            key={m.id}
                            onSelect={() => {
                              onModelChange(m.id);
                              setModelSelectorOpen(false);
                            }}
                            value={m.id}
                          >
                            <ModelSelectorName>{m.name}</ModelSelectorName>
                            {selectedModel === m.id && (
                              <CheckIcon className="ml-auto size-3" />
                            )}
                          </ModelSelectorItem>
                        ))}
                      </ModelSelectorGroup>
                    </ModelSelectorList>
                  </ModelSelectorContent>
                </ModelSelector>
              </PromptInputTools>
              <PromptInputSubmit
                status={
                  status === "streaming"
                    ? "streaming"
                    : status === "submitted"
                      ? "submitted"
                      : "ready"
                }
              />
            </PromptInputFooter>
          </PromptInput>
        </div>
      </div>
    </div>
  );
};
