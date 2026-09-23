#!/usr/bin/env node
/**
 * Identify the recited verse in an audio file, fully offline.
 *
 *   npm run build && npm run fetch-assets
 *   node examples/transcribe.mjs recitation.mp3
 *
 * Needs ffmpeg on PATH (to decode the file) and onnxruntime-node.
 */
import { join } from 'node:path';
import { createAsrSession } from '@tilawi/quran-asr';
import { assetsDir, loadAsrAssets } from '../test/golden-runner.mjs';
import { createNodeRunner, decodePcm } from '../eval/node.mjs';

const file = process.argv[2];
if (!file) {
  console.error('usage: node examples/transcribe.mjs <audio file>');
  process.exit(2);
}

const runner = await createNodeRunner(join(assetsDir(), 'fastconformer_full_mixed.onnx'));
const session = createAsrSession(runner, loadAsrAssets());

const result = await session.transcribe(decodePcm(file));
console.log('heard:', result.transcript);
if (result.surah > 0) {
  const span = result.ayah_end ? `${result.ayah}-${result.ayah_end}` : `${result.ayah}`;
  console.log(`verse: ${result.surah}:${span} (score ${result.score.toFixed(2)})`);
} else {
  console.log('verse: no confident match');
}
