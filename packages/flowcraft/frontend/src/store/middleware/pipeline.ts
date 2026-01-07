import { type GraphMiddleware, type GraphMutationEvent } from "./types";

export class MutationPipeline {
  private middlewares: GraphMiddleware[] = [];

  clear() {
    this.middlewares = [];
    return this;
  }

  execute(
    event: GraphMutationEvent,
    finalAction: (event: GraphMutationEvent) => void,
  ) {
    let index = -1;

    const next = (i: number, currentEvent: GraphMutationEvent) => {
      if (i <= index) return;
      index = i;
      const middleware = this.middlewares[i];
      if (middleware) {
        middleware(currentEvent, (evt) => {
          next(i + 1, evt);
        });
      } else {
        finalAction(currentEvent);
      }
    };

    next(0, event);
  }

  use(middleware: GraphMiddleware) {
    this.middlewares.push(middleware);
    return this;
  }
}

export const pipeline = new MutationPipeline();
