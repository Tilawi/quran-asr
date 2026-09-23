import { describe, it, expect } from 'vitest';
import { isTrustedChampion } from '../src/index';
import type { QuranChampionMatch } from '../src/index';

// A minimal champion-match shape; the trust gate only reads score/frag_fit/margin.
function match(partial: Partial<QuranChampionMatch>): QuranChampionMatch {
  return {
    surah: 2,
    ayah: 102,
    ayah_end: null,
    text: '',
    phonemes_joined: '',
    score: 0,
    raw_score: 0,
    bonus: 0,
    ...partial,
  };
}

describe('isTrustedChampion — trust gate', () => {
  it('trusts any match at/above the flat 0.80 threshold, regardless of fragment signals', () => {
    expect(isTrustedChampion(match({ score: 0.8 }), 40)).toBe(true);
    expect(isTrustedChampion(match({ score: 0.95, margin: 0, frag_fit: 0 }), 300)).toBe(true);
  });

  it('rejects a null/zero champion', () => {
    expect(isTrustedChampion(null, 40)).toBe(false);
    expect(isTrustedChampion(undefined, 40)).toBe(false);
    expect(isTrustedChampion(match({ score: 0 }), 40)).toBe(false);
  });

  it('RESCUES a distinctive fragment below 0.80 (real 2:102 case: fits verse far better than any rival)', () => {
    // On-device: correct verse at score 0.69, score-margin ~0, but fit_margin 0.375.
    expect(isTrustedChampion(match({ score: 0.69, frag_fit: 0.81, fit_margin: 0.375 }), 32)).toBe(true);
  });

  it('does NOT rescue an ambiguous fragment (fits rival verses about as well)', () => {
    expect(isTrustedChampion(match({ score: 0.78, frag_fit: 0.9, fit_margin: 0.05 }), 20)).toBe(false);
  });

  it('score-margin alone does NOT rescue (score gap is the wrong signal for fragments)', () => {
    // A big score-margin but a tiny fit-margin must still be rejected.
    expect(isTrustedChampion(match({ score: 0.78, frag_fit: 0.9, margin: 0.3, fit_margin: 0.04 }), 20)).toBe(false);
  });

  it('does NOT rescue a too-short fragment (few chars = appears in many verses)', () => {
    expect(isTrustedChampion(match({ score: 0.78, frag_fit: 0.99, fit_margin: 0.4 }), 8)).toBe(false);
  });

  it('does NOT rescue a match below the fragment floor', () => {
    expect(isTrustedChampion(match({ score: 0.55, frag_fit: 0.99, fit_margin: 0.4 }), 40)).toBe(false);
  });

  it('does NOT rescue when the fragment fit is poor (query is not really a substring)', () => {
    expect(isTrustedChampion(match({ score: 0.75, frag_fit: 0.6, fit_margin: 0.4 }), 20)).toBe(false);
  });

  it('treats missing fragment signals as failing (never rescues on absence of evidence)', () => {
    expect(isTrustedChampion(match({ score: 0.78 }), 20)).toBe(false);
  });
});
