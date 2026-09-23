/**
 * Guards against confident verse matches from silence / noise.
 *
 * Observed on an Android emulator: digital silence made the model emit "لا.",
 * which matched 77:12 at score 0.83 and was TRUSTED (voice search jumped to a
 * verse nobody recited). Two independent defenses:
 *  - detectSpeech() rejects clips without real signal before inference;
 *  - the session never trusts a transcript shorter than MIN_TRUSTED_LETTERS.
 */
import { describe, expect, it } from 'vitest';
import { arabicLetterCount, createAsrSession, detectSpeech, MIN_TRUSTED_LETTERS, voicedMilliseconds } from '../src/index';
// Plain .mjs helper shared with the golden generator (eval/make-goldens.mjs).
import { loadAsrAssets, oneHotRunner } from './golden-runner.mjs';

const SR = 16000;
const tone = (seconds: number, amplitude: number) =>
  Float32Array.from({ length: Math.round(seconds * SR) }, (_, i) => amplitude * Math.sin((2 * Math.PI * 220 * i) / SR));
const noise = (seconds: number, amplitude: number) =>
  Float32Array.from({ length: Math.round(seconds * SR) }, () => (Math.random() * 2 - 1) * amplitude);
const concat = (...parts: Float32Array[]) => {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
};

describe('detectSpeech', () => {
  it('rejects digital silence (the emulator case: peak ~8/32768)', () => {
    expect(detectSpeech(noise(5, 8 / 32768))).toBe(false);
    expect(voicedMilliseconds(noise(5, 8 / 32768))).toBe(0);
  });

  it('rejects quiet room noise around -60 dBFS', () => {
    expect(detectSpeech(noise(5, 0.0017))).toBe(false);
  });

  it('accepts a quiet voice-like signal (-30 dBFS) lasting 1 s', () => {
    expect(detectSpeech(concat(noise(2, 0.0005), tone(1, 0.045), noise(2, 0.0005)))).toBe(true);
  });

  it('rejects a single short blip (< 300 ms) in silence', () => {
    expect(detectSpeech(concat(noise(2, 0.0005), tone(0.15, 0.3), noise(2, 0.0005)))).toBe(false);
  });

  it('handles empty and tiny inputs', () => {
    expect(detectSpeech(new Float32Array(0))).toBe(false);
    expect(detectSpeech(new Float32Array(10))).toBe(false);
  });
});

describe('minimum transcript length for a trusted match', () => {
  it('counts only Arabic letters', () => {
    expect(arabicLetterCount('لا.')).toBe(2);
    expect(arabicLetterCount('كهيعص')).toBe(5);
    expect(arabicLetterCount('والفجر')).toBe(6);
    expect(MIN_TRUSTED_LETTERS).toBe(5);
  });

  it('never trusts the hallucinated "لا." that matched 77:12 on silence', async () => {
    const assets = loadAsrAssets();
    const runner = oneHotRunner(Object.keys(assets.vocab).length, assets.blankId);
    const session = createAsrSession(runner, assets);
    // Vocab ids: 64 = '▁لا', 30 = '.'  (the model's actual output on silence)
    runner.setTokens([64, 30]);
    const result = await session.transcribe(new Float32Array(1));
    expect(result.transcript).toBe('لا.');
    expect(result.surah).toBe(0);
    expect(result.ayah).toBe(0);
    // The matcher may still have a (rejected) best guess for diagnostics.
    expect(result.bestGuess?.trusted ?? false).toBe(false);
  }, 60_000);
});
