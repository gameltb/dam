import { type RFState } from "@/store/types";

export interface BindingBackend<T> {
  setValue: (lens: SyncedLens<T>, newValue: T, options: BindingOptions<T>) => void;
  useValue: (lens: SyncedLens<T>, options: BindingOptions<T>) => T;
}

export interface BindingOptions<T> {
  backend?: "custom" | "store" | "table";
  debounce?: number;
  onIncoming?: (newValue: T, oldValue: T) => void;
  transient?: boolean;
  undoable?: boolean;
}

export interface SyncedLens<T> {
  category?: "edge" | "node" | "task" | "ui" | "viewport";
  description?: string;
  get: (state: RFState) => T;
  // New: Unique identifier for precise subscription
  id?: string;

  set: (draft: RFState, newValue: T) => void;
}
