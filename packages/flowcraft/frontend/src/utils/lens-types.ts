import { type RFState } from "@/store/types";

export interface SyncedLens<T> {
  get: (state: RFState) => T;
  set: (draft: RFState, newValue: T) => void;
  description?: string;
  category?: 'node' | 'edge' | 'viewport' | 'ui' | 'task';
  
  // New: Unique identifier for precise subscription
  id?: string; 
}

export interface BindingOptions<T> {
  undoable?: boolean;
  transient?: boolean;
  debounce?: number;
  backend?: 'store' | 'table' | 'custom';
  onIncoming?: (newValue: T, oldValue: T) => void;
}

export interface BindingBackend<T> {
  useValue: (lens: SyncedLens<T>, options: BindingOptions<T>) => T;
  setValue: (lens: SyncedLens<T>, newValue: T, options: BindingOptions<T>) => void;
}
