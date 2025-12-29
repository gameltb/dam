type Listener<T> = (data: T) => void;

class EventBus {
  private listeners: Record<string, Listener<unknown>[]> = {};

  on<T>(event: string, listener: Listener<T>) {
    this.listeners[event] ??= [];
    this.listeners[event].push(listener as Listener<unknown>);
    return () => {
      this.off(event, listener);
    };
  }

  off<T>(event: string, listener: Listener<T>) {
    if (!this.listeners[event]) return;
    this.listeners[event] = this.listeners[event].filter(
      (l) => l !== (listener as Listener<unknown>),
    );
  }

  emit(event: string, data: unknown) {
    if (!this.listeners[event]) return;
    this.listeners[event].forEach((l) => {
      l(data);
    });
  }
}

export const mockEventBus = new EventBus();
