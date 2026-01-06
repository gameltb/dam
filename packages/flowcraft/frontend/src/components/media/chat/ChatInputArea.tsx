import React, { useState } from "react";
import { GlobeIcon, CheckIcon } from "lucide-react";
import {
  PromptInput,
  PromptInputTextarea,
  PromptInputSubmit,
  PromptInputTools,
  PromptInputBody,
  PromptInputFooter,
  type PromptInputMessage,
  PromptInputButton,
  PromptInputAttachments,
  PromptInputAttachment,
  PromptInputActionAddAttachments,
  PromptInputActionMenu,
  PromptInputActionMenuTrigger,
  PromptInputActionMenuContent,
} from "../../ai-elements/prompt-input";
import {
  ModelSelector,
  ModelSelectorContent,
  ModelSelectorGroup,
  ModelSelectorItem,
  ModelSelectorList,
  ModelSelectorName,
  ModelSelectorTrigger,
} from "../../ai-elements/model-selector";
import { Suggestion, Suggestions } from "../../ai-elements/suggestion";
import { type ChatStatus, type ContextNode } from "./types";

const MODELS = [
  { id: "gpt-4o", name: "GPT-4o", chefSlug: "openai", providers: ["openai"] },
  {
    id: "gpt-4o-mini",
    name: "GPT-4o Mini",
    chefSlug: "openai",
    providers: ["openai"],
  },
  {
    id: "claude-3-5-sonnet",
    name: "Claude 3.5 Sonnet",
    chefSlug: "anthropic",
    providers: ["anthropic"],
  },
];

const SUGGESTIONS = [
  "Explain how this graph works",
  "Summarize current results",
  "Optimize this workflow",
];

interface Props {
  status: ChatStatus;
  onSubmit: (msg: PromptInputMessage, model: string, search: boolean) => void;
  droppedNodes: ContextNode[];
  setDroppedNodes: (n: ContextNode[]) => void;
  selectedModel: string;
  onModelChange: (m: string) => void;
  useWebSearch: boolean;
  onWebSearchChange: (v: boolean) => void;
}

export const ChatInputArea: React.FC<Props> = ({
  status,
  onSubmit,
  droppedNodes,
  setDroppedNodes,
  selectedModel,
  onModelChange,
  useWebSearch,
  onWebSearchChange,
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
                key={n.id}
                className="group flex items-center gap-1.5 bg-primary/10 text-primary px-2 py-1 rounded-md text-xs border border-primary/20"
              >
                <span>{n.label}</span>
                <button
                  onClick={() => {
                    setDroppedNodes(droppedNodes.filter((i) => i.id !== n.id));
                  }}
                  className="hover:text-destructive"
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
                onSubmit({ text: s, files: [] }, selectedModel, useWebSearch);
              }}
              suggestion={s}
            />
          ))}
        </Suggestions>

        <div className="px-2 pb-2">
          <PromptInput onSubmit={handleSubmit}>
            <PromptInputAttachments>
              {(f) => <PromptInputAttachment key={f.id} data={f} />}
            </PromptInputAttachments>
            <PromptInputBody>
              <PromptInputTextarea
                placeholder="Ask anything..."
                onChange={(e) => {
                  setInputText(e.target.value);
                }}
                value={inputText}
                className="min-h-[44px]"
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
                  open={modelSelectorOpen}
                  onOpenChange={setModelSelectorOpen}
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
                            value={m.id}
                            onSelect={() => {
                              onModelChange(m.id);
                              setModelSelectorOpen(false);
                            }}
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
