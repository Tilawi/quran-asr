import { describe, it, expect } from 'vitest';
import {
  normalizeArabicForJudge as normalizeArabic,
  lcsRatio as ratio,
  judgeAttempt,
} from '../src/index';

// Al-Fatiha 1:1 (uthmani, with diacritics) — the fixed canonical "expected" words.
const BISMILLAH = ['بِسْمِ', 'ٱللَّهِ', 'ٱلرَّحْمَٰنِ', 'ٱلرَّحِيمِ'];

describe('judge normalizeArabic', () => {
  it('collapses uthmani, imlaei and bare spellings of the same word', () => {
    expect(normalizeArabic('ٱللَّهِ')).toBe(normalizeArabic('اللَّهِ'));
    expect(normalizeArabic('اللَّهِ')).toBe(normalizeArabic('الله'));
    expect(normalizeArabic('ٱللَّهِ')).toBe('الله');
  });
  it('unifies alef/hamza/taa-marbuta forms and strips tatweel', () => {
    expect(normalizeArabic('أَحَد')).toBe(normalizeArabic('احد'));
    expect(normalizeArabic('قُـــلْ')).toBe('قل');
    expect(normalizeArabic('صلاة')).toBe('صلاه'); // taa marbuta -> haa
  });
});

describe('ratio (LCS-based, == rapidfuzz fuzz.ratio)', () => {
  it('is 1 for identical and lower for near words', () => {
    expect(ratio('الرحيم', 'الرحيم')).toBe(1);
    expect(ratio('الرحيم', 'الرحين')).toBeGreaterThan(0.7); // one-letter near miss
    expect(ratio('الرحيم', 'محمد')).toBeLessThan(0.5); // unrelated
  });
});

describe('judgeAttempt — honest word-by-word feedback', () => {
  it('marks all words correct when recitation matches (diacritics aside)', () => {
    const r = judgeAttempt(BISMILLAH, ['بسم', 'الله', 'الرحمن', 'الرحيم']);
    expect(r.correct).toBe(4);
    expect(r.apparentErrors).toBe(0);
    expect(r.attempted).toBe(4);
    expect(r.scorePercent).toBe(100);
  });

  it('flags a clearly wrong word as an apparent error', () => {
    const r = judgeAttempt(BISMILLAH, ['بسم', 'الله', 'الرحمن', 'محمد']); // 4th word clearly wrong
    const w = r.words.find((x) => x.expectedIndex === 3)!;
    expect(w.judgment).toBe('apparent-error');
    expect(r.correct).toBe(3);
    expect(r.apparentErrors).toBe(1);
    expect(r.scorePercent!).toBeLessThan(100);
  });

  it('does NOT falsely accuse a near-miss (golden rule: never mark correct as wrong)', () => {
    const r = judgeAttempt(BISMILLAH, ['بسم', 'الله', 'الرحمن', 'الرحين']); // one-letter slip
    const w = r.words.find((x) => x.expectedIndex === 3)!;
    expect(w.judgment).toBe('correct'); // lenient — a near miss is not an apparent error
    expect(r.apparentErrors).toBe(0);
  });

  it('flags an internal omission (aligns the later word, deletes the skipped one)', () => {
    const r = judgeAttempt(BISMILLAH, ['بسم', 'الله', 'الرحيم']); // skipped الرحمن, said الرحيم
    const skipped = r.words.find((x) => x.expectedIndex === 2)!;
    expect(skipped.operation).toBe('omission');
    expect(skipped.judgment).toBe('apparent-error');
    expect(r.correct).toBe(3);
  });

  it('flags an inserted (extra) word without penalizing the real words', () => {
    const r = judgeAttempt(BISMILLAH, ['بسم', 'الله', 'امين', 'الرحمن', 'الرحيم']);
    expect(r.insertions).toBe(1);
    expect(r.correct).toBe(4);
  });

  it('treats an early stop as unattempted, not as errors', () => {
    const r = judgeAttempt(BISMILLAH, ['بسم', 'الله']); // stopped after two words
    expect(r.correct).toBe(2);
    expect(r.apparentErrors).toBe(0);
    expect(r.attempted).toBe(2);
    expect(r.words.filter((w) => w.judgment === 'unattempted').length).toBe(2);
  });

  it('detects a wrong verse as mostly errors / low score', () => {
    const r = judgeAttempt(BISMILLAH, ['قل', 'هو', 'الله', 'احد']); // 112:1, not 1:1
    expect(r.correct).toBeLessThanOrEqual(1);
    expect(r.scorePercent === null || r.scorePercent < 50).toBe(true);
  });

  it('keeps an exact match correct even at low ASR confidence (golden rule)', () => {
    const r = judgeAttempt(BISMILLAH, ['بسم', 'الله', 'الرحمن', 'الرحيم'], {
      confidences: [0.2, 0.2, 0.2, 0.2],
    });
    expect(r.correct).toBe(4);
    expect(r.uncertain).toBe(0);
  });

  it('returns a null score for an empty recitation (no speech)', () => {
    const r = judgeAttempt(BISMILLAH, []);
    expect(r.scorePercent).toBeNull();
    expect(r.attempted).toBe(0);
  });
});
