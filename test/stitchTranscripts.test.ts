import { describe, it, expect } from 'vitest';
import { stitchTranscripts, overlapMerge } from '../src/index';

describe('overlapMerge — drop duplicated overlap words', () => {
  it('merges an exact overlap without duplicating', () => {
    expect(overlapMerge(['a', 'b', 'c', 'd', 'e'], ['d', 'e', 'f', 'g'])).toEqual([
      'a', 'b', 'c', 'd', 'e', 'f', 'g',
    ]);
  });

  it('concatenates when there is no overlap', () => {
    expect(overlapMerge(['a', 'b', 'c'], ['x', 'y', 'z'])).toEqual(['a', 'b', 'c', 'x', 'y', 'z']);
  });

  it('handles a near-miss word inside the overlap (ASR wobble)', () => {
    // "kitaba" vs "kitabu" is a close near-miss; the overlap should still merge.
    const merged = overlapMerge(['alif', 'lam', 'mim', 'kitabu'], ['kitaba', 'la', 'rayba']);
    expect(merged).toEqual(['alif', 'lam', 'mim', 'kitabu', 'la', 'rayba']);
  });

  it('handles empty inputs', () => {
    expect(overlapMerge([], ['a', 'b'])).toEqual(['a', 'b']);
    expect(overlapMerge(['a', 'b'], [])).toEqual(['a', 'b']);
  });

  it('prefers the longest agreeing overlap (repeated words)', () => {
    expect(overlapMerge(['x', 'a', 'a', 'a'], ['a', 'a', 'a', 'y'])).toEqual(['x', 'a', 'a', 'a', 'y']);
  });
});

describe('stitchTranscripts — full window sequence', () => {
  it('stitches three overlapping windows into one clean transcript', () => {
    const out = stitchTranscripts([
      'wattabau ma tatlu ashshayatin ala',
      'ashshayatin ala mulki sulayman wama',
      'mulki sulayman wama kafara sulaymanu',
    ]);
    expect(out).toBe('wattabau ma tatlu ashshayatin ala mulki sulayman wama kafara sulaymanu');
  });

  it('returns a single chunk unchanged', () => {
    expect(stitchTranscripts(['alif lam mim'])).toBe('alif lam mim');
  });

  it('ignores empty/whitespace chunks', () => {
    expect(stitchTranscripts(['', '  ', 'alif lam mim', ''])).toBe('alif lam mim');
    expect(stitchTranscripts([])).toBe('');
  });
});
