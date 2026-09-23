/**
 * @tilawi/quran-asr — on-device Quran recitation recognition core.
 *
 * Runtime-agnostic: you inject a model runner (onnxruntime-react-native,
 * onnxruntime-node, ...) and the text-side assets; the core decodes the CTC
 * output, identifies the recited verse(s), and applies a conservative trust
 * gate so uncertain matches are reported as "no match" rather than guessed.
 *
 * The recognizer only transcribes the user's recitation. It never generates or
 * alters Quran text; canonical text is only used to compare against.
 */
import { QuranDB } from './quran-db';
import type { QuranChampionMatch } from './quran-db';
import { TextCTCDecoder } from './text-ctc-decode';
import { adaptQuranTextData, validateCtcTokenRoundTrip, type CtcTokenTable, type RawQuranVerse } from './quran-text-adapter';
import type { SessionRunner } from './types';

export type { SessionRunner, SessionOutput, QuranVerse } from './types';
export { QuranDB, partialRatio } from './quran-db';
export type { QuranChampionMatch, QuranTokenEncoder, QuranCtcTokenTable, QuranCandidate } from './quran-db';
export { TextCTCDecoder } from './text-ctc-decode';
export type { TextCTCResult } from './text-ctc-decode';
export { adaptQuranTextData, validateCtcTokenRoundTrip } from './quran-text-adapter';
export type { CtcTokenTable, RawQuranVerse } from './quran-text-adapter';
export { normalizeArabic } from './normalizer';
export { distance, ratio, semiGlobalDistance, fragmentScore } from './levenshtein';
export { detectSpeech, voicedMilliseconds, SPEECH_DEFAULTS } from './speech';
export type { SpeechDetectionOptions } from './speech';
export { overlapMerge, stitchTranscripts } from './stitch';

// Memorization judge (word alignment + honest per-word feedback). Its
// normalizer and LCS-based ratios differ from the verse matcher's
// normalizeArabic / edit-distance ratio above, so they are exported under
// distinct names. They are intentionally NOT unified: on 600 real recitations
// (7,628 word judgments, scripts/asr-eval/run-memo-eval.mjs) switching the
// judge to the matcher's normalizer changed no user-visible judgment, and the
// judge's version also strips punctuation the model occasionally emits
// ("النجوم."). Changing the matcher's instead would alter verse matching,
// which the golden suite guards.
export {
  judgeAttempt,
  matchWord,
  alignWords,
  phoneticNormalize,
  normalizeArabic as normalizeArabicForJudge,
  ratio as lcsRatio,
  partialRatio as lcsPartialRatio,
} from './judge';
export type {
  RichStatus,
  AlignOp,
  AlignEntry,
  WordJudgment,
  WordOperation,
  WordResult,
  AttemptResult,
  JudgeOptions,
} from './judge';

/** Text-side assets handed to the session (model bytes go into the runner). */
export interface AsrAssets {
  /** CTC token id -> string map. */
  vocab: Record<string, string>;
  /** Verse key -> CTC token id sequence. */
  quranCtcTokens: CtcTokenTable;
  /** Raw verse records (surah/ayah/text_uthmani/...). */
  quran: unknown[];
  /** Optional CTC blank id (defaults to 1024). */
  blankId?: number;
}

/** Best candidate (even when not trusted) — for diagnostics on the no-match screen. */
export interface AsrBestGuess {
  surah: number;
  ayah: number;
  ayah_end: number | null;
  score: number;
  margin: number;
  frag_fit: number;
  fit_margin: number;
  trusted: boolean;
}

/** One-shot verse-match result. surah/ayah are 0 when nothing matched. */
export interface AsrPrediction {
  surah: number;
  ayah: number;
  ayah_end: number | null;
  score: number;
  transcript: string;
  /** Best candidate the matcher found even if it was rejected (diagnostics). */
  bestGuess?: AsrBestGuess;
}

/** Low-level one-shot result: transcript, tokens, acoustic log-probs, champion. */
export interface AsrTranscribeResult {
  text: string;
  tokenIds: number[];
  acoustic: {
    logprobs: Float32Array;
    timeSteps: number;
    vocabSize: number;
    blankId: number;
  };
  championMatch?: QuranChampionMatch;
  bestGuess?: AsrBestGuess;
}

export interface AsrSession {
  /** One-shot: transcribe a full clip and return the best verse match. */
  transcribe(audio: Float32Array): Promise<AsrPrediction>;
  /** Low-level one-shot returning the full result (acoustic + champion). */
  transcribeRaw(audio: Float32Array): Promise<AsrTranscribeResult>;
  /** Underlying verse database (lookup, search). */
  readonly db: QuranDB;
  /** Underlying CTC decoder. */
  readonly decoder: TextCTCDecoder;
}

// A champion verse match at/above this score is trusted outright.
const CHAMPION_TRUST_THRESHOLD = 0.8;

// Fragment rescue: a short fragment of a long verse scores BELOW the flat
// threshold even when the recognizer heard it correctly — its whole-verse ratio
// is tiny and the fragment blend caps the score near ~0.84, so realistic
// recitation noise drags a correct match under 0.80 ("analysis failed").
//
// The right confidence signal is NOT the score gap: a short verse with
// coincidental whole-string overlap scores about as high as the true long verse
// (observed on-device: correct verse 2:102 at score 0.69 with an unrelated verse
// only 0.006 behind). It is the FRAGMENT-FIT gap — how much better the query
// fits the winner as a substring than it fits any TRULY DIFFERENT verse. In that
// same case the fit gap was 0.375 (winner fit 0.81 vs rivals ~0.31-0.44).
//
// Accept a sub-0.80 match when it is a strong, unambiguous fragment:
//   - it genuinely fits the verse as a substring (frag_fit),
//   - it fits that verse MUCH better than any different verse (fit_margin), and
//   - it is long enough to be distinctive (guards 1-2 word fragments common to
//     many verses).
// Parameters tuned on a corpus-wide noisy-fragment sweep (1,638 samples over 182
// verses at 6/9/12% simulated CER): this branch lifts correct-fragment recall
// from 78% to ~92% while adding ZERO net-new confident-wrong matches versus the
// flat threshold alone, across every setting tested. It never lowers the bar for
// normal (>=0.80) recitations.
const FRAGMENT_TRUST_FLOOR = 0.62; // gate out only near-random matches
const FRAGMENT_FIT_MIN = 0.75; // query must fit the verse as a near-substring
const FRAGMENT_FIT_MARGIN_MIN = 0.15; // fits the winner far better than any rival
const FRAGMENT_MIN_CHARS = 12; // >= ~3 words

// A trusted match needs at least this many Arabic letters in the transcript,
// on every path. On silence the model can hallucinate a stray token ("لا." was
// observed) that scores >= 0.80 against a short verse; a 2-letter transcript
// must never become a confident verse identification. Cost: reciting ONLY a
// 2-4 letter opening (e.g. "الم", "حم", "يس") is no longer identified; most of
// those open several surahs, so they were ambiguous anyway.
export const MIN_TRUSTED_LETTERS = 5;
const ARABIC_LETTER_RE = /[\u0621-\u064A\u0671-\u06D3]/g;

/** Number of Arabic letters (not diacritics, spaces or punctuation) in `text`. */
export function arabicLetterCount(text: string): number {
  return text.match(ARABIC_LETTER_RE)?.length ?? 0;
}

/**
 * Trust gate for a champion verse match. Returns true when the match is either
 * at/above the flat trust threshold, or is a strong, unambiguous fragment
 * (margin + fragment fit + length) that the flat threshold alone would reject.
 * `noSpaceLen` is the transcript length without spaces.
 */
export function isTrustedChampion(match: QuranChampionMatch | null | undefined, noSpaceLen: number): boolean {
  if (!match || !match.score) return false;
  if (match.score >= CHAMPION_TRUST_THRESHOLD) return true;
  return (
    match.score >= FRAGMENT_TRUST_FLOOR &&
    (match.frag_fit ?? 0) >= FRAGMENT_FIT_MIN &&
    (match.fit_margin ?? 0) >= FRAGMENT_FIT_MARGIN_MIN &&
    noSpaceLen >= FRAGMENT_MIN_CHARS
  );
}

/**
 * Build an on-device ASR session over an injected runtime `runner` (which owns
 * the ONNX model) plus the text-side assets. One-shot only: feed a full 16 kHz
 * mono PCM clip, get back a verse match and/or the raw transcript.
 */
export function createAsrSession(runner: SessionRunner, assets: AsrAssets): AsrSession {
  const decoder = new TextCTCDecoder(assets.vocab, assets.blankId ?? 1024);
  const quranData = adaptQuranTextData(assets.quran as RawQuranVerse[], assets.quranCtcTokens, decoder);
  validateCtcTokenRoundTrip(quranData, decoder);
  const db = new QuranDB(quranData, undefined, assets.quranCtcTokens);

  const transcribeRaw = async (audio: Float32Array): Promise<AsrTranscribeResult> => {
    const { logprobs, timeSteps, vocabSize } = await runner.run(audio);
    const greedy = decoder.decode(logprobs, timeSteps, vocabSize);
    const champion = db.bestJoint03Match(greedy.text);
    const noSpaceLen = greedy.text.replace(/ /g, '').length;
    const trustedChampion =
      arabicLetterCount(greedy.text) >= MIN_TRUSTED_LETTERS && isTrustedChampion(champion, noSpaceLen) ? champion : null;
    // Best (even untrusted) guess, for diagnostics on the "no match" screen.
    const bestGuess: AsrBestGuess | undefined = champion
      ? {
          surah: champion.surah,
          ayah: champion.ayah,
          ayah_end: champion.ayah_end ?? null,
          score: champion.score,
          margin: champion.margin ?? 0,
          frag_fit: champion.frag_fit ?? 0,
          fit_margin: champion.fit_margin ?? 0,
          trusted: !!trustedChampion,
        }
      : undefined;
    return {
      text: greedy.text,
      tokenIds: greedy.tokenIds,
      acoustic: { logprobs, timeSteps, vocabSize, blankId: decoder.getBlankId() },
      championMatch: trustedChampion ?? undefined,
      bestGuess,
    };
  };

  return {
    db,
    decoder,
    /** One-shot: transcribe a clip and return the best verse match (0/0 on none). */
    async transcribe(audio: Float32Array): Promise<AsrPrediction> {
      const result = await transcribeRaw(audio);
      const m = result.championMatch;
      if (!m) {
        return {
          surah: 0,
          ayah: 0,
          ayah_end: null,
          score: 0,
          transcript: result.text,
          bestGuess: result.bestGuess,
        };
      }
      return {
        surah: m.surah,
        ayah: m.ayah,
        ayah_end: m.ayah_end ?? null,
        score: m.score,
        transcript: result.text,
        bestGuess: result.bestGuess,
      };
    },
    /** Low-level one-shot: full result (acoustic log-probs + champion match). */
    transcribeRaw,
  };
}
