import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Ensures exhaustive matching in switch/if-else blocks.
 * If this is reached, it means a case was missed at compile time.
 */
export function assertNever(x: never): never {
  throw new Error(`Unreachable code reached with value: ${JSON.stringify(x)}`);
}

/**
 * Rust-like Result type for explicit error handling.
 */
export type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E };

export const Ok = <T>(value: T): Result<T, never> => ({ ok: true, value });
export const Err = <E>(error: E): Result<never, E> => ({ ok: false, error });
