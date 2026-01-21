import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Rust-like Result type for explicit error handling.
 */
export type Result<T, E = Error> = { error: E; ok: false } | { ok: true; value: T };

/**
 * Ensures exhaustive matching in switch/if-else blocks.
 * If this is reached, it means a case was missed at compile time.
 */
export function assertNever(x: never): never {
  throw new Error(`Unreachable code reached with value: ${JSON.stringify(x)}`);
}

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const Ok = <T>(value: T): Result<T, never> => ({ ok: true, value });
export const Err = <E>(error: E): Result<never, E> => ({ error, ok: false });
