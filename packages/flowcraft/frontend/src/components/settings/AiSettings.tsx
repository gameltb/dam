import { Check, Edit2, Plus, Trash2 } from "lucide-react";
import React, { useState } from "react";

import { useUiStore } from "@/store/uiStore";
import { type LocalLLMClientConfig } from "@/types";

import { Badge } from "../ui/badge";
import { Button } from "../ui/button";

export const AiSettings: React.FC = () => {
  const {
    addLocalClient,
    removeLocalClient,
    setActiveLocalClient,
    settings,
    updateLocalClient,
  } = useUiStore();

  const [isAdding, setIsAdding] = useState(false);
  const [editingId, setEditingId] = useState<null | string>(null);

  const [formData, setFormData] = useState<Omit<LocalLLMClientConfig, "id">>({
    apiKey: "",
    baseUrl: "http://localhost:1234/v1",
    model: "",
    name: "",
  });

  const handleSave = () => {
    if (editingId) {
      updateLocalClient(editingId, formData);
      setEditingId(null);
    } else {
      addLocalClient(formData);
      setIsAdding(false);
    }
    setFormData({
      apiKey: "",
      baseUrl: "http://localhost:1234/v1",
      model: "",
      name: "",
    });
  };

  const startEdit = (client: LocalLLMClientConfig) => {
    setEditingId(client.id);
    setFormData({
      apiKey: client.apiKey,
      baseUrl: client.baseUrl,
      model: client.model,
      name: client.name,
    });
    setIsAdding(false);
  };

  const cancelEdit = () => {
    setEditingId(null);
    setIsAdding(false);
    setFormData({
      apiKey: "",
      baseUrl: "http://localhost:1234/v1",
      model: "",
      name: "",
    });
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold uppercase tracking-wider text-text-color">
          Local LLM Clients
        </h3>
        {!isAdding && !editingId && (
          <Button
            className="h-8 gap-1 text-xs"
            onClick={() => {
              setIsAdding(true);
            }}
            size="sm"
            variant="outline"
          >
            <Plus size={14} /> Add Client
          </Button>
        )}
      </div>

      {/* Client List */}
      <div className="flex flex-col gap-2">
        {settings.localClients.map((client) => {
          const isActive = settings.activeLocalClientId === client.id;
          const isEditing = editingId === client.id;

          if (isEditing) return null;

          return (
            <div
              className={`flex items-center justify-between p-3 rounded-lg border transition-colors ${
                isActive
                  ? "bg-primary/10 border-primary/50"
                  : "bg-muted/10 border-node-border hover:border-primary/30"
              }`}
              key={client.id}
            >
              <div
                className="flex-1 cursor-pointer"
                onClick={() => {
                  setActiveLocalClient(client.id);
                }}
              >
                <div className="flex items-center gap-2">
                  {isActive && <Check className="text-primary" size={14} />}
                  <span className="text-sm font-semibold text-text-color">
                    {client.name}
                  </span>
                  <Badge className="text-[10px]" variant="outline">
                    {client.model}
                  </Badge>
                </div>
                <div className="text-[10px] text-muted-foreground mt-1 truncate max-w-[300px]">
                  {client.baseUrl}
                </div>
              </div>

              <div className="flex items-center gap-1">
                <Button
                  onClick={(e) => {
                    e.stopPropagation();
                    startEdit(client);
                  }}
                  size="icon-sm"
                  variant="ghost"
                >
                  <Edit2 size={12} />
                </Button>
                <Button
                  className="hover:text-destructive"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeLocalClient(client.id);
                  }}
                  size="icon-sm"
                  variant="ghost"
                >
                  <Trash2 size={12} />
                </Button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Add/Edit Form */}
      {(isAdding || editingId) && (
        <div className="p-4 rounded-lg border border-primary/20 bg-primary/5 flex flex-col gap-4 animate-in fade-in slide-in-from-top-2">
          <div className="text-xs font-bold text-primary mb-2 uppercase">
            {editingId ? "Edit Client" : "New Local Client"}
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] uppercase text-muted-foreground font-bold">
                Client Name
              </label>
              <input
                className="bg-background border border-node-border rounded px-2 py-1.5 text-xs text-text-color outline-none focus:border-primary"
                onChange={(e) => {
                  setFormData({ ...formData, name: e.target.value });
                }}
                placeholder="e.g. LM Studio"
                type="text"
                value={formData.name}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] uppercase text-muted-foreground font-bold">
                Model ID
              </label>
              <input
                className="bg-background border border-node-border rounded px-2 py-1.5 text-xs text-text-color outline-none focus:border-primary"
                onChange={(e) => {
                  setFormData({ ...formData, model: e.target.value });
                }}
                placeholder="e.g. llama-3"
                type="text"
                value={formData.model}
              />
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-[10px] uppercase text-muted-foreground font-bold">
              Base URL
            </label>
            <input
              className="bg-background border border-node-border rounded px-2 py-1.5 text-xs text-text-color outline-none focus:border-primary"
              onChange={(e) => {
                setFormData({ ...formData, baseUrl: e.target.value });
              }}
              placeholder="http://localhost:1234/v1"
              type="text"
              value={formData.baseUrl}
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-[10px] uppercase text-muted-foreground font-bold">
              API Key (Optional)
            </label>
            <input
              className="bg-background border border-node-border rounded px-2 py-1.5 text-xs text-text-color outline-none focus:border-primary"
              onChange={(e) => {
                setFormData({ ...formData, apiKey: e.target.value });
              }}
              placeholder="lm-studio"
              type="password"
              value={formData.apiKey}
            />
          </div>

          <div className="flex justify-end gap-2 mt-2">
            <Button onClick={cancelEdit} size="sm" variant="ghost">
              Cancel
            </Button>
            <Button
              disabled={!formData.name || !formData.baseUrl}
              onClick={handleSave}
              size="sm"
            >
              {editingId ? "Update" : "Create Client"}
            </Button>
          </div>
        </div>
      )}

      {settings.localClients.length > 0 && !settings.activeLocalClientId && (
        <div className="text-[10px] text-center text-orange-500 font-bold bg-orange-500/10 p-2 rounded border border-orange-500/20">
          No active client selected. Direct local inference is disabled.
        </div>
      )}
    </div>
  );
};
