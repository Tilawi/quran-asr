/**
 * Word-by-word memorization feedback: align what the reciter said against the
 * expected verse words and judge each word. A port of the checker Tilawi ran
 * on its (since retired) server, so on-device results match what was
 * validated there. Pure TypeScript, no platform imports.
 *
 * Statuses are deliberately few and honest: correct / apparent error /
 * uncertain / unattempted. The golden rule: NEVER mark a correct recitation
 * wrong. Tajweed rules, diacritics scoring and repetition/false-start
 * detection are out of scope.
 *
 * IMPORTANT: the expected words are the fixed canonical Quran text. This
 * module compares the USER's recitation to it; it never generates or grades
 * the Quran text itself.
 */

// --- alignment tuning (same values as the validated server checker) ---
const MATCH_SCORE = 2.0;
const GAP_OPEN = -3.0;
const GAP_EXTEND = -0.5;
const FUZZY_ALIGNMENT_THRESHOLD = 0.7;

// ---------------------------------------------------------------------------
// Arabic normalization (port of ArabicNormalizer)
// ---------------------------------------------------------------------------

const UNIFY: Record<string, string> = {
  'آ': 'ا', // alef madda -> alef
  'أ': 'ا', // alef hamza above -> alef
  'إ': 'ا', // alef hamza below -> alef
  'ٱ': 'ا', // alef wasla -> alef
  'ؤ': 'و', // waw hamza -> waw
  'ئ': 'ي', // yaa hamza -> yaa
  'ى': 'ي', // alef maqsura -> yaa
  'ة': 'ه', // taa marbuta -> haa
  'ـ': '', // tatweel removed
};

const PHONETIC: Record<string, string> = {
  'ص': 'س', // saad -> seen
  'ض': 'د', // daad -> dal
  'ط': 'ت', // taa -> ta
  'ظ': 'ذ', // dhaa -> dhal
  'ق': 'ك', // qaf -> kaf
  'ء': 'ا', // hamza -> alef
  'غ': 'خ', // ghain -> khaa
  'ث': 'س', // thaa -> seen
  'ذ': 'ز', // dhal -> zayn
};

function isDiacriticOrMark(ch: string): boolean {
  return (
    (ch >= 'ؐ' && ch <= 'ؚ') ||
    (ch >= 'ً' && ch <= 'ٟ') ||
    ch === 'ٰ' ||
    (ch >= 'ۖ' && ch <= 'ۭ')
  );
}

/** Full normalization for comparison: strip diacritics, unify letters, keep Arabic letters + single spaces. */
export function normalizeArabic(text: string): string {
  if (!text) return '';
  let out = '';
  for (const ch of text) {
    if (isDiacriticOrMark(ch)) continue;
    const unified = UNIFY[ch];
    const c = unified !== undefined ? unified : ch;
    for (const cc of c) {
      if (cc === ' ' || /\s/.test(cc)) out += ' ';
      else if (cc >= '؀' && cc <= 'ۿ') out += cc;
    }
  }
  return out.split(/\s+/).filter(Boolean).join(' ').trim();
}

/** Phonetic normalization: map similar-sounding letters for lenient matching. */
export function phoneticNormalize(text: string): string {
  const norm = normalizeArabic(text);
  let out = '';
  for (const ch of norm) out += PHONETIC[ch] ?? ch;
  return out;
}

// ---------------------------------------------------------------------------
// Fuzzy similarity (LCS ratio == rapidfuzz fuzz.ratio; plus partial_ratio)
// ---------------------------------------------------------------------------

function lcsLength(a: string, b: string): number {
  const n = a.length, m = b.length;
  if (!n || !m) return 0;
  let prev = new Array(m + 1).fill(0);
  let cur = new Array(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      cur[j] = a[i - 1] === b[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
    }
    [prev, cur] = [cur, prev];
  }
  return prev[m];
}

/** rapidfuzz-style ratio in [0,1]: 2*LCS/(|a|+|b|). */
export function ratio(a: string, b: string): number {
  if (a === b) return 1;
  const la = a.length, lb = b.length;
  if (!la && !lb) return 1;
  if (!la || !lb) return 0;
  return (2 * lcsLength(a, b)) / (la + lb);
}

/** Best ratio of the shorter string against any equal-length window of the longer. */
export function partialRatio(a: string, b: string): number {
  if (!a.length || !b.length) return 0;
  const [short, long] = a.length <= b.length ? [a, b] : [b, a];
  let best = 0;
  for (let i = 0; i + short.length <= long.length; i++) {
    best = Math.max(best, ratio(short, long.slice(i, i + short.length)));
    if (best === 1) break;
  }
  return best;
}

// Common Arabic-Quran ASR confusion pairs (port of WordMatcher.CONFUSION_SETS).
const CONFUSION_SETS: string[][] = [
  ['ق', 'ك'], ['ص', 'س'], ['ض', 'د'],
  ['ط', 'ت'], ['ظ', 'ذ'], ['ه', 'ة'],
  ['ع', 'ا'], ['ح', 'ه'], ['غ', 'خ'],
  ['ث', 'س'], ['ذ', 'ز'], ['ع', 'أ'],
];
const CONFUSION_MAP: Record<string, Set<string>> = {};
for (const pair of CONFUSION_SETS) {
  for (const c of pair) {
    (CONFUSION_MAP[c] ??= new Set()).add(pair[0] === c ? pair[1] : pair[0]);
  }
}

/** Simplified confusion check: equal length, exactly one differing char that is a known confusion pair. */
function isConfusion(expNorm: string, heardNorm: string): boolean {
  if (expNorm.length !== heardNorm.length || expNorm === heardNorm) return false;
  let diffs = 0, ei = -1;
  for (let i = 0; i < expNorm.length; i++) {
    if (expNorm[i] !== heardNorm[i]) { diffs++; ei = i; if (diffs > 1) return false; }
  }
  if (diffs !== 1) return false;
  return CONFUSION_MAP[expNorm[ei]]?.has(heardNorm[ei]) ?? false;
}

// ---------------------------------------------------------------------------
// Word matcher (port of WordMatcher.match, minus the tajweed-variant layer)
// ---------------------------------------------------------------------------

export type RichStatus = 'correct' | 'likely_correct' | 'uncertain' | 'likely_wrong' | 'wrong';

function thresholds(asrConfidence?: number | null) {
  if (asrConfidence == null || asrConfidence >= 0.8) {
    return { correct: 0.9, likely_correct: 0.75, uncertain: 0.6, likely_wrong: 0.45 };
  }
  if (asrConfidence >= 0.5) {
    return { correct: 0.85, likely_correct: 0.7, uncertain: 0.55, likely_wrong: 0.4 };
  }
  return { correct: 0.8, likely_correct: 0.65, uncertain: 0.5, likely_wrong: 0.35 };
}

export function matchWord(expected: string, heard: string, asrConfidence?: number | null): {
  status: RichStatus;
  confidence: number;
} {
  const expNorm = normalizeArabic(expected);
  const heardNorm = normalizeArabic(heard);

  if (expNorm === heardNorm) return { status: 'correct', confidence: 1 }; // golden rule

  if (phoneticNormalize(expected) === phoneticNormalize(heard)) {
    return { status: 'likely_correct', confidence: 0.85 };
  }

  const bestFuzzy = Math.max(ratio(expNorm, heardNorm), partialRatio(expNorm, heardNorm));
  const confusion = isConfusion(expNorm, heardNorm);
  const th = thresholds(asrConfidence);

  if (bestFuzzy >= th.correct) return { status: 'correct', confidence: bestFuzzy };
  if (bestFuzzy >= th.likely_correct || confusion) return { status: 'likely_correct', confidence: bestFuzzy };
  if (bestFuzzy >= th.uncertain) return { status: 'uncertain', confidence: bestFuzzy };
  if (bestFuzzy >= th.likely_wrong) return { status: 'likely_wrong', confidence: bestFuzzy };
  return { status: 'wrong', confidence: bestFuzzy };
}

// ---------------------------------------------------------------------------
// Sequence alignment (port of SequenceAligner.align — affine gap NW)
// ---------------------------------------------------------------------------

export type AlignOp = 'match' | 'substitute' | 'delete' | 'insert';
export interface AlignEntry { ei: number | null; hi: number | null; op: AlignOp; }

function alignmentScore(exp: string, heard: string): number {
  if (exp === heard) return MATCH_SCORE;
  const r = ratio(exp, heard);
  if (r > FUZZY_ALIGNMENT_THRESHOLD) return MATCH_SCORE * r - 1.0;
  return -1.0;
}

export function alignWords(
  expected: string[],
  heard: string[],
  verseBoundaries: Set<number> = new Set()
): AlignEntry[] {
  const m = expected.length, n = heard.length;
  const NEG = Number.NEGATIVE_INFINITY, EPS = 1e-9;
  const mk = () => Array.from({ length: m + 1 }, () => new Array<number>(n + 1).fill(NEG));
  const M = mk(), X = mk(), Y = mk();
  M[0][0] = 0;
  for (let i = 1; i <= m; i++) X[i][0] = GAP_OPEN + (i - 1) * GAP_EXTEND;
  for (let j = 1; j <= n; j++) Y[0][j] = GAP_OPEN + (j - 1) * GAP_EXTEND;

  for (let i = 1; i <= m; i++) {
    const gapOpenI = verseBoundaries.has(i) ? GAP_OPEN * 0.5 : GAP_OPEN;
    for (let j = 1; j <= n; j++) {
      const s = alignmentScore(expected[i - 1], heard[j - 1]);
      M[i][j] = Math.max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]) + s;
      X[i][j] = Math.max(M[i - 1][j] + gapOpenI, X[i - 1][j] + GAP_EXTEND);
      Y[i][j] = Math.max(M[i][j - 1] + GAP_OPEN, Y[i][j - 1] + GAP_EXTEND);
    }
  }

  const best = Math.max(M[m][n], X[m][n], Y[m][n]);
  let state: 'M' | 'X' | 'Y' =
    Math.abs(best - M[m][n]) < EPS ? 'M' : Math.abs(best - X[m][n]) < EPS ? 'X' : 'Y';

  const out: AlignEntry[] = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (state === 'M' && i > 0 && j > 0) {
      const prevBest = Math.max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]);
      const prev: 'M' | 'X' | 'Y' =
        Math.abs(prevBest - M[i - 1][j - 1]) < EPS ? 'M'
          : Math.abs(prevBest - X[i - 1][j - 1]) < EPS ? 'X' : 'Y';
      out.push({ ei: i - 1, hi: j - 1, op: expected[i - 1] === heard[j - 1] ? 'match' : 'substitute' });
      state = prev; i--; j--;
    } else if (state === 'X' && i > 0) {
      const gapOpenI = verseBoundaries.has(i) ? GAP_OPEN * 0.5 : GAP_OPEN;
      const fromX = X[i - 1][j] + GAP_EXTEND;
      state = Math.abs(X[i][j] - fromX) < EPS ? 'X' : 'M';
      void gapOpenI;
      out.push({ ei: i - 1, hi: null, op: 'delete' });
      i--;
    } else if (j > 0) {
      const fromY = Y[i][j - 1] + GAP_EXTEND;
      state = Math.abs(Y[i][j] - fromY) < EPS ? 'Y' : 'M';
      out.push({ ei: null, hi: j - 1, op: 'insert' });
      j--;
    } else break;
  }
  out.reverse();
  return out;
}

// ---------------------------------------------------------------------------
// Attempt judgment (simple honest surface + plan §8.5 scoring)
// ---------------------------------------------------------------------------

export type WordJudgment = 'correct' | 'apparent-error' | 'uncertain' | 'unattempted';
export type WordOperation = 'match' | 'substitution' | 'omission' | 'insertion' | 'unattempted';

export interface WordResult {
  expectedIndex: number | null;
  recognizedIndex: number | null;
  expected?: string;
  observed?: string;
  operation: WordOperation;
  judgment: WordJudgment;
}

export interface AttemptResult {
  words: WordResult[];
  attempted: number; // N = expected words the user actually attempted
  correct: number;
  apparentErrors: number; // substitutions + internal omissions (expected words)
  uncertain: number;
  insertions: number;
  coveragePercent: number;
  /**
   * Accuracy over the ATTEMPTED words only (correct / (correct + errors +
   * insertions)); null when too many attempted words were uncertain. It does
   * not penalize stopping early: reciting 1 correct word of a page scores 100.
   */
  scorePercent: number | null;
  /**
   * Accuracy over the WHOLE expected passage: correct / (expected words +
   * insertions). Stopping early lowers it (1 correct word of 29 -> ~3%). Use
   * this for a score shown for the selected passage.
   */
  overallPercent: number | null;
}

export interface JudgeOptions {
  confidences?: (number | null)[]; // per-heard-word ASR confidence
  verseBoundaries?: Set<number>;
}

function collapse(status: RichStatus): WordJudgment {
  if (status === 'correct' || status === 'likely_correct') return 'correct';
  if (status === 'uncertain') return 'uncertain';
  return 'apparent-error'; // likely_wrong | wrong
}

export function judgeAttempt(
  expectedWords: string[],
  heardWords: string[],
  opts: JudgeOptions = {}
): AttemptResult {
  const expNorm = expectedWords.map(normalizeArabic);
  const heardNorm = heardWords.map(normalizeArabic);
  const alignment = alignWords(expNorm, heardNorm, opts.verseBoundaries);

  // Last expected word the reciter actually reached (any match/substitute).
  let lastAttemptedEi = -1;
  for (const a of alignment) {
    if ((a.op === 'match' || a.op === 'substitute') && a.ei != null) {
      lastAttemptedEi = Math.max(lastAttemptedEi, a.ei);
    }
  }

  const words: WordResult[] = [];
  let correct = 0, apparentErrors = 0, uncertain = 0, insertions = 0;

  for (const a of alignment) {
    if (a.op === 'match') {
      words.push({ expectedIndex: a.ei, recognizedIndex: a.hi, expected: expectedWords[a.ei!],
        observed: heardWords[a.hi!], operation: 'match', judgment: 'correct' });
      correct++;
    } else if (a.op === 'substitute') {
      const conf = opts.confidences?.[a.hi!] ?? null;
      const judgment = collapse(matchWord(expectedWords[a.ei!], heardWords[a.hi!], conf).status);
      words.push({ expectedIndex: a.ei, recognizedIndex: a.hi, expected: expectedWords[a.ei!],
        observed: heardWords[a.hi!], operation: 'substitution', judgment });
      if (judgment === 'correct') correct++;
      else if (judgment === 'uncertain') uncertain++;
      else apparentErrors++;
    } else if (a.op === 'delete') {
      if (a.ei! <= lastAttemptedEi) {
        words.push({ expectedIndex: a.ei, recognizedIndex: null, expected: expectedWords[a.ei!],
          operation: 'omission', judgment: 'apparent-error' });
        apparentErrors++;
      } else {
        words.push({ expectedIndex: a.ei, recognizedIndex: null, expected: expectedWords[a.ei!],
          operation: 'unattempted', judgment: 'unattempted' });
      }
    } else {
      // insert
      words.push({ expectedIndex: null, recognizedIndex: a.hi, observed: heardWords[a.hi!],
        operation: 'insertion', judgment: 'apparent-error' });
      insertions++;
    }
  }

  const decided = correct + apparentErrors; // C + S + D (uncertain is undecided)
  const attempted = decided + uncertain; // N
  const coveragePercent = attempted > 0 ? (decided / attempted) * 100 : 0;
  const scorePercent =
    attempted > 0 && coveragePercent >= 80
      ? (100 * correct) / (correct + apparentErrors + insertions)
      : null;

  const overallDenominator = expectedWords.length + insertions;
  const overallPercent = overallDenominator > 0 ? (100 * correct) / overallDenominator : null;

  return { words, attempted, correct, apparentErrors, uncertain, insertions, coveragePercent, scorePercent, overallPercent };
}
