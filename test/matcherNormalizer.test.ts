/**
 * The matcher's normalizeArabic (src/normalizer.ts) runs on BOTH sides of every
 * verse match: the model's transcript and every canonical verse. It must strip
 * marks and fold letter variants, and never drop a base letter. (The judge has
 * its own normalizer, tested in judge.test.ts.)
 */
import { describe, expect, it } from 'vitest';
import { normalizeArabic } from '../src/index';
import { loadAsrAssets } from './golden-runner.mjs';

const MARKS = /[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]/;
const FOLDED_AWAY = /[\u0622\u0623\u0625\u0671\u0629\u0649]/;
const BASE_LETTER = /[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]/g;

describe('matcher normalizeArabic', () => {
  it('strips harakat, shadda, sukun, superscript alef and tatweel', () => {
    // bismi with kasra/sukun, lillah with shadda + superscript alef, a tatweel
    expect(normalizeArabic('\u0628\u0650\u0633\u0652\u0645\u0650')).toBe('\u0628\u0633\u0645');
    expect(normalizeArabic('\u0644\u0651\u064E\u0670\u0647')).toBe('\u0644\u0647');
    expect(normalizeArabic('\u0643\u0640\u0640\u062A\u0627\u0628')).toBe('\u0643\u062A\u0627\u0628');
  });

  it('folds alef variants, ta marbuta and alef maqsura', () => {
    expect(normalizeArabic('\u0623\u0625\u0622\u0671')).toBe('\u0627\u0627\u0627\u0627');
    expect(normalizeArabic('\u0631\u062D\u0645\u0629')).toBe('\u0631\u062D\u0645\u0647');
    expect(normalizeArabic('\u0647\u062F\u0649')).toBe('\u0647\u062F\u064A');
  });

  it('removes the BOM and collapses whitespace', () => {
    expect(normalizeArabic('\uFEFF  \u0645\u0646 \n\t \u0631\u0628  ')).toBe('\u0645\u0646 \u0631\u0628');
    expect(normalizeArabic('')).toBe('');
  });

  it('on every word of the Quran: no marks or folded letters left, no base letter lost', () => {
    const verses = loadAsrAssets().quran as { text_uthmani: string; text_clean?: string }[];
    expect(verses).toHaveLength(6236);
    const words = verses.flatMap((v) =>
      `${v.text_uthmani} ${v.text_clean ?? ''}`.replace(/\uFEFF/g, '').split(/\s+/).filter(Boolean),
    );
    expect(words.length).toBeGreaterThan(150_000);
    for (const word of words) {
      const out = normalizeArabic(word);
      expect(out, word).not.toMatch(MARKS);
      expect(out, word).not.toMatch(FOLDED_AWAY);
      expect(out.match(BASE_LETTER)?.length ?? 0, word).toBe(word.match(BASE_LETTER)?.length ?? 0);
    }
  });
});
