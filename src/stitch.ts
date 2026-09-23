/**
 * Stitch per-window transcripts from overlapping audio chunks into one
 * transcript.
 *
 * Long recitations (a full page) are transcribed in overlapping windows so peak
 * memory stays bounded (one-shot inference memory grows with clip length and
 * would OOM a phone on multi-minute audio). Consecutive windows overlap by a few
 * seconds, so they repeat the words spoken in that overlap; this module removes
 * the duplicates by finding the best suffix/prefix word alignment between
 * neighbouring chunks. It only DROPS duplicated words in the overlap — it never
 * invents or reorders text (Quran-integrity: the transcript is the user's
 * recitation, graded against the fixed canonical verse text elsewhere).
 */
import { ratio } from './levenshtein';

/** Two transcript words are "the same" if identical or a close near-miss. */
const WORD_SIM_THRESHOLD = 0.8;
/** Cap the overlap search; a few seconds of speech is well under this. */
const MAX_OVERLAP_WORDS = 30;
/** Fraction of the overlap window that must agree to accept it as the seam. */
const MIN_OVERLAP_AGREEMENT = 0.6;

function wordsClose(a: string, b: string): boolean {
  return a === b || ratio(a, b) >= WORD_SIM_THRESHOLD;
}

/**
 * Append `next` onto `prev`, dropping the leading `next` words that repeat the
 * trailing `prev` words (the audio overlap). Chooses the longest overlap whose
 * words agree by at least MIN_OVERLAP_AGREEMENT; if none agree, concatenates
 * outright (better a rare duplicate than dropped recitation).
 */
export function overlapMerge(prev: string[], next: string[]): string[] {
  if (prev.length === 0) return next.slice();
  if (next.length === 0) return prev.slice();
  const maxK = Math.min(prev.length, next.length, MAX_OVERLAP_WORDS);
  let bestK = 0;
  let bestMatched = 0;
  for (let k = 1; k <= maxK; k++) {
    const a = prev.slice(prev.length - k);
    const b = next.slice(0, k);
    let matched = 0;
    for (let i = 0; i < k; i++) if (wordsClose(a[i], b[i])) matched++;
    if (matched >= Math.ceil(k * MIN_OVERLAP_AGREEMENT) && matched >= bestMatched) {
      bestMatched = matched;
      bestK = k;
    }
  }
  return prev.concat(next.slice(bestK));
}

/** Stitch window transcripts (in order) into one space-joined transcript. */
export function stitchTranscripts(chunks: string[]): string {
  const nonEmpty = chunks.map((c) => c.trim()).filter(Boolean);
  if (nonEmpty.length === 0) return '';
  let merged = nonEmpty[0].split(/\s+/);
  for (let i = 1; i < nonEmpty.length; i++) {
    merged = overlapMerge(merged, nonEmpty[i].split(/\s+/));
  }
  return merged.join(' ');
}
