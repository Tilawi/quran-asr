#!/usr/bin/env node
/**
 * Download the model and its data files from Hugging Face into .assets/
 * (about 110 MB, once). Tests, the eval harness and the examples read them
 * from there (see test/golden-runner.mjs for the lookup order).
 *
 * Usage: node scripts/fetch-assets.mjs [--force]
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const HF_REPO = 'muhdur/tilawi-fastconformer-quran';
const FILES = ['fastconformer_full_mixed.onnx', 'vocab.json', 'quran_ctc_tokens.json', 'quran.json', 'SHA256SUMS', 'TANZIL-NOTICE.txt'];

const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', '.assets');
const force = process.argv.includes('--force');

mkdirSync(OUT, { recursive: true });
for (const name of FILES) {
  const file = join(OUT, name);
  if (existsSync(file) && !force) continue;
  const url = `https://huggingface.co/${HF_REPO}/resolve/main/${name}`;
  process.stdout.write(`downloading ${name} ... `);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed: ${res.status} ${url}`);
  writeFileSync(file, Buffer.from(await res.arrayBuffer()));
  console.log('ok');
}

// Verify every file against the published checksums.
for (const line of readFileSync(join(OUT, 'SHA256SUMS'), 'utf8').trim().split('\n')) {
  const [expected, name] = line.trim().split(/\s+/);
  const actual = createHash('sha256').update(readFileSync(join(OUT, name))).digest('hex');
  if (actual !== expected) throw new Error(`checksum mismatch for ${name}; rerun with --force`);
}
console.log(`assets ready in ${OUT}`);
