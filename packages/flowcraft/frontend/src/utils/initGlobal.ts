/**
 * initGlobal
 * Performs global environment patches and polyfills.
 * This file should be imported as early as possible in both frontend and server entry points.
 */

export const initGlobal = () => {
  // --- BigInt JSON Serialization Patch ---
  // Standard JSON.stringify does not know how to serialize BigInt.
  // This polyfill adds a toJSON method to the BigInt prototype to convert it to a string.
  if (typeof BigInt !== "undefined" && !(BigInt.prototype as any).toJSON) {
    (BigInt.prototype as any).toJSON = function () {
      return this.toString();
    };
  }

  // Add other global polyfills or initializations here if needed.
};
