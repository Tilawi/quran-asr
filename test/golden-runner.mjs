/**
 * Shared helpers for the tests, the golden generator and the eval harness.
 *
 * assetsDir / loadAsrAssets: the model-side data files. Looked up in order:
 *   1. $TILAWI_ASR_ASSETS
 *   2. <package>/.assets          (filled by `npm run fetch-assets`, from Hugging Face)
 *   3. <package>/../../assets/asr (inside the Tilawi app monorepo)
 * Files may be named *.json (Hugging Face) or *.bin (the app bundles JSON as .bin).
 *
 * oneHotRunner: a SessionRunner that ignores the audio and emits a one-hot
 * log-prob matrix spelling the given CTC token ids, with a blank frame between
 * tokens so repeated ids survive greedy CTC collapse. The core's greedy
 * decoder therefore yields exactly those ids.
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const CTC_BLANK_ID = 1024;

const PKG = join(dirname(fileURLToPath(import.meta.url)), '..');

export function assetsDir() {
  const candidates = [process.env.TILAWI_ASR_ASSETS, join(PKG, '.assets'), join(PKG, '..', '..', 'assets', 'asr')];
  for (const dir of candidates) {
    if (dir && (existsSync(join(dir, 'vocab.json')) || existsSync(join(dir, 'vocab.bin')))) return dir;
  }
  throw new Error('ASR assets not found. Run `npm run fetch-assets` (downloads them from Hugging Face) or set TILAWI_ASR_ASSETS.');
}

/** Path of one asset, accepting either the .json or the .bin spelling. */
export function assetPath(stem, dir = assetsDir()) {
  const json = join(dir, `${stem}.json`);
  return existsSync(json) ? json : join(dir, `${stem}.bin`);
}

export function loadAsrAssets(dir = assetsDir()) {
  const read = (stem) => JSON.parse(readFileSync(assetPath(stem, dir), 'utf8').replace(/^﻿/, ''));
  return {
    vocab: read('vocab'),
    quranCtcTokens: read('quran_ctc_tokens'),
    quran: read('quran'),
    blankId: CTC_BLANK_ID,
  };
}

export function oneHotRunner(vocabSize, blankId) {
  let tokens = [];
  return {
    setTokens(ids) {
      tokens = ids;
    },
    async run() {
      const frames = [];
      for (const id of tokens) {
        frames.push(id, blankId);
      }
      if (!frames.length) frames.push(blankId);
      const timeSteps = frames.length;
      const logprobs = new Float32Array(timeSteps * vocabSize).fill(-20);
      frames.forEach((id, t) => {
        logprobs[t * vocabSize + id] = 0;
      });
      return { logprobs, timeSteps, vocabSize };
    },
  };
}
