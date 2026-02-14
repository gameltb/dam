declare global {
  interface Window {
    // Since Zustand store types are complex, we'll keep the base definition here for now
    flowStore?: any;
    lastProcessedMousePos?: { x: number; y: number };
  }
}

export {};
