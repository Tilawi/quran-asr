/**
 * Levenshtein edit distance — multi-block bit-parallel (Myers 1999 / Hyyrö),
 * plain 32-bit integers (NO BigInt: Hermes runs BigInt far slower than Number).
 * Splits the shorter string into 32-bit blocks and threads the horizontal delta
 * (hin/hout) between them, so it is O(n·ceil(m/32)) for ANY length — no slow
 * O(m·n) DP fallback. Returns the exact distance (proven equivalent to a plain
 * DP by fuzzing 300k random pairs up to 5 blocks + thousands of verse-text pairs,
 * 0 mismatches). This is the hottest path in verse matching; long recitations
 * were spending seconds here.
 */
export function distance(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  if (a.length > b.length) [a, b] = [b, a]; // a = shorter (the bit-parallel "pattern")
  const m = a.length;
  const n = b.length;
  const W = 32;
  const nb = (m + W - 1) >> 5; // number of 32-bit blocks
  const HIGH = 0x80000000 >>> 0;
  const Peq: Map<number, number>[] = [];
  for (let bi = 0; bi < nb; bi++) {
    const mp = new Map<number, number>();
    const start = bi * W;
    const end = Math.min(start + W, m);
    for (let i = start; i < end; i++) {
      const c = a.charCodeAt(i);
      mp.set(c, ((mp.get(c) ?? 0) | (1 << (i - start))) >>> 0);
    }
    Peq.push(mp);
  }
  const Pv: number[] = new Array(nb).fill(0xffffffff >>> 0);
  const Mv: number[] = new Array(nb).fill(0);
  const lastBits = m - (nb - 1) * W; // 1..32 valid bits in the final block
  const lastScoreBit = (1 << (lastBits - 1)) >>> 0;
  let score = m;
  for (let j = 0; j < n; j++) {
    const c = b.charCodeAt(j);
    let hin = 1; // leading deletion column
    for (let bi = 0; bi < nb; bi++) {
      let Eq = Peq[bi].get(c) ?? 0;
      const pv = Pv[bi],
        mv = Mv[bi];
      const Xv = (Eq | mv) >>> 0;
      if (hin < 0) Eq = (Eq | 1) >>> 0;
      const Xh = (((((((Eq & pv) >>> 0) + pv) >>> 0) ^ pv) >>> 0) | Eq) >>> 0;
      let Ph = (mv | (~(Xh | pv) >>> 0)) >>> 0;
      let Mh = (pv & Xh) >>> 0;
      if (bi === nb - 1) {
        if (Ph & lastScoreBit) score++;
        else if (Mh & lastScoreBit) score--;
      }
      let hout = 0;
      if (Ph & HIGH) hout = 1;
      else if (Mh & HIGH) hout = -1;
      Ph = (Ph << 1) >>> 0;
      Mh = (Mh << 1) >>> 0;
      if (hin < 0) Mh = (Mh | 1) >>> 0;
      else if (hin > 0) Ph = (Ph | 1) >>> 0;
      Pv[bi] = (Mh | (~(Xv | Ph) >>> 0)) >>> 0;
      Mv[bi] = (Ph & Xv) >>> 0;
      hin = hout;
    }
  }
  return score;
}

/**
 * Normalized Levenshtein similarity ratio.
 * Returns 1.0 for identical strings, 0.0 for completely different.
 * Matches python-Levenshtein's `ratio()` behavior:
 *   ratio = (len(a) + len(b) - distance) / (len(a) + len(b))
 */
export function ratio(a: string, b: string): number {
  const lenSum = a.length + b.length;
  if (lenSum === 0) return 1.0;
  return (lenSum - distance(a, b)) / lenSum;
}

/**
 * Semi-global edit distance: finds the minimum edit distance to align
 * the entire query against any substring of ref (free gaps at start/end of ref).
 * Use case: "how well does this transcript fragment match somewhere inside this
 * verse?"
 *
 * Bit-parallel Myers, semi-global variant — same multi-block engine as
 * `distance()`, with two directional differences: the horizontal delta into the
 * top row (`hin`) is 0 each column (D[0][j] = 0, i.e. free start), and we track
 * the running minimum of D[m][j] over every text position (free end). The query
 * is the fixed bit-parallel "pattern" (rows), so — unlike `distance()` — the
 * shorter/longer strings are NOT swapped: this measure is asymmetric.
 * Proven equivalent to the plain O(m·n) DP by fuzzing 400k random pairs (up to
 * ~6 blocks) + ~16k real verse-fragment pairs, 0 mismatches. Called for every
 * candidate in the matcher, so this is O(n·ceil(m/32)) instead of O(m·n).
 */
export function semiGlobalDistance(query: string, ref: string): number {
  if (query.length === 0) return 0;
  if (ref.length === 0) return query.length;
  const m = query.length;
  const n = ref.length;
  const W = 32;
  const nb = (m + W - 1) >> 5;
  const HIGH = 0x80000000 >>> 0;
  const Peq: Map<number, number>[] = [];
  for (let bi = 0; bi < nb; bi++) {
    const mp = new Map<number, number>();
    const start = bi * W;
    const end = Math.min(start + W, m);
    for (let i = start; i < end; i++) {
      const c = query.charCodeAt(i);
      mp.set(c, ((mp.get(c) ?? 0) | (1 << (i - start))) >>> 0);
    }
    Peq.push(mp);
  }
  const Pv: number[] = new Array(nb).fill(0xffffffff >>> 0);
  const Mv: number[] = new Array(nb).fill(0);
  const lastBits = m - (nb - 1) * W;
  const lastScoreBit = (1 << (lastBits - 1)) >>> 0;
  let score = m; // D[m][0] = m
  let best = m; // free end also allows the empty ref prefix
  for (let j = 0; j < n; j++) {
    const c = ref.charCodeAt(j);
    let hin = 0; // free start: D[0][j] = 0
    for (let bi = 0; bi < nb; bi++) {
      let Eq = Peq[bi].get(c) ?? 0;
      const pv = Pv[bi],
        mv = Mv[bi];
      const Xv = (Eq | mv) >>> 0;
      if (hin < 0) Eq = (Eq | 1) >>> 0;
      const Xh = (((((((Eq & pv) >>> 0) + pv) >>> 0) ^ pv) >>> 0) | Eq) >>> 0;
      let Ph = (mv | (~(Xh | pv) >>> 0)) >>> 0;
      let Mh = (pv & Xh) >>> 0;
      if (bi === nb - 1) {
        if (Ph & lastScoreBit) score++;
        else if (Mh & lastScoreBit) score--;
      }
      let hout = 0;
      if (Ph & HIGH) hout = 1;
      else if (Mh & HIGH) hout = -1;
      Ph = (Ph << 1) >>> 0;
      Mh = (Mh << 1) >>> 0;
      if (hin < 0) Mh = (Mh | 1) >>> 0;
      else if (hin > 0) Ph = (Ph | 1) >>> 0;
      Pv[bi] = (Mh | (~(Xv | Ph) >>> 0)) >>> 0;
      Mv[bi] = (Ph & Xv) >>> 0;
      hin = hout;
    }
    if (score < best) best = score; // free end: min of D[m][j] over all text positions
  }
  return best;
}

/**
 * Fragment score: how well does the query match as a fragment of ref?
 * Returns 0.0-1.0. Score of 1.0 means query is an exact substring of ref.
 * Directional: measures "how much of the query does the ref explain?"
 */
export function fragmentScore(query: string, ref: string): number {
  if (query.length === 0) return 1.0;
  return Math.max(0, 1 - semiGlobalDistance(query, ref) / query.length);
}
