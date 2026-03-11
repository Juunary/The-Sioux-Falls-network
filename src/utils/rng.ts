// ============================================================
// Seeded pseudo-random number generator utilities
// ============================================================

/**
 * Mulberry32: fast, high-quality 32-bit PRNG.
 * Returns a closure that produces uniform floats in [0, 1).
 */
export function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return function () {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Fisher-Yates shuffle, seeded. Returns a new array of k random elements
 * from arr without replacement. Does not mutate arr.
 */
export function fisherYatesSample<T>(arr: readonly T[], k: number, rng: () => number): T[] {
  const copy = [...arr];
  const end = Math.min(k, copy.length);
  for (let i = copy.length - 1; i > copy.length - 1 - end; i--) {
    const j = Math.floor(rng() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(copy.length - end);
}

// ---- Module-level simulation RNG singleton ----
// Seeded from clock at startup; reseed via seedSimRng() for benchmark/replay mode.

let _simRng: () => number = mulberry32(Date.now() ^ 0xdeadbeef);

/** Replace the simulation RNG with a freshly seeded instance. */
export function seedSimRng(seed: number): void {
  _simRng = mulberry32(seed);
}

/** Draw one value from the shared simulation RNG. */
export function simRng(): number {
  return _simRng();
}
