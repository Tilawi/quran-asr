#!/usr/bin/env node
/**
 * Build the golden regression suite for the on-device ASR core.
 *
 * Each case is a greedy CTC token sequence. A stand-in runner turns it into a
 * one-hot log-prob matrix (blank between tokens), so session.transcribe() runs
 * the core's real decode -> match -> trust-gate path with only the neural
 * network replaced. The core's full output is recorded; a refactored core must
 * reproduce it exactly (see test/asrGoldens.test.ts).
 *
 * Case sources:
 *  - real:      token ids recorded by run-eval.mjs on EveryAyah recitations
 *  - synthetic: deterministic cases from the quran_ctc_tokens asset
 *               (verses, multi-verse spans, prefixes/suffixes/fragments,
 *               dropped/substituted tokens, Bismillah openings, noise)
 *
 * Quran-integrity note: canonical token sequences are only used as recognizer
 * OUTPUT stand-ins to exercise the matcher; nothing here alters Quran text.
 *
 * Usage: npx tsx eval/make-goldens.mjs [eval/results/*.json ...]
 *
 * Only regenerate when a matching change is INTENDED and justified with
 * run-eval.mjs numbers; the goldens exist to catch unintended changes.
 */
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createAsrSession } from '../src/index.ts';
import { oneHotRunner, loadAsrAssets } from '../test/golden-runner.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, '..', 'test', 'fixtures', 'asr-goldens.json');

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function syntheticCases(ctc, vocabSize, blankId, seed = 42) {
  const rand = rng(seed);
  const pick = (arr) => arr[Math.floor(rand() * arr.length)];
  const keys = Object.keys(ctc);
  const single = keys.filter((k) => {
    const [, a, b] = k.split(':');
    return a === b;
  });
  const multi = keys.filter((k) => {
    const [, a, b] = k.split(':');
    return a !== b;
  });
  const cases = [];
  const add = (kind, source, ids) => {
    if (ids.length) cases.push({ id: `syn-${cases.length}`, kind, source, tokenIds: ids });
  };
  for (let i = 0; i < 80; i++) {
    const k = pick(single);
    add('verse', k, ctc[k]);
  }
  for (let i = 0; i < 60; i++) {
    const k = pick(multi);
    add('span', k, ctc[k]);
  }
  for (let i = 0; i < 60; i++) {
    const k = pick(single);
    const t = ctc[k];
    const cut = Math.max(1, Math.floor(t.length * (0.3 + rand() * 0.5)));
    add('prefix', k, t.slice(0, cut));
  }
  for (let i = 0; i < 40; i++) {
    const k = pick(single);
    const t = ctc[k];
    const cut = Math.floor(t.length * (0.2 + rand() * 0.5));
    add('suffix', k, t.slice(cut));
  }
  for (let i = 0; i < 40; i++) {
    const k = pick(single);
    const t = ctc[k];
    const a = Math.floor(t.length * rand() * 0.4);
    const b = a + Math.max(3, Math.floor(t.length * (0.3 + rand() * 0.3)));
    add('fragment', k, t.slice(a, b));
  }
  for (let i = 0; i < 50; i++) {
    const k = pick(single);
    const t = [...ctc[k]];
    const drops = 1 + Math.floor(rand() * 3);
    for (let d = 0; d < drops && t.length > 2; d++) t.splice(Math.floor(rand() * t.length), 1);
    add('dropped', k, t);
  }
  for (let i = 0; i < 50; i++) {
    const k = pick(single);
    const t = [...ctc[k]];
    const subs = 1 + Math.floor(rand() * 3);
    for (let d = 0; d < subs; d++) {
      let v;
      do v = Math.floor(rand() * vocabSize);
      while (v === blankId);
      t[Math.floor(rand() * t.length)] = v;
    }
    add('substituted', k, t);
  }
  const bsm = ctc['1:1:1'];
  for (let i = 0; i < 30; i++) {
    const surah = 2 + Math.floor(rand() * 113);
    const k = `${surah}:1:1`;
    if (ctc[k] && surah !== 9) add('bismillah-opening', k, [...bsm, ...ctc[k]]);
  }
  for (let i = 0; i < 30; i++) {
    const len = 5 + Math.floor(rand() * 60);
    const t = [];
    while (t.length < len) {
      const v = Math.floor(rand() * vocabSize);
      if (v !== blankId) t.push(v);
    }
    add('noise', null, t);
  }
  return cases;
}

function realCases(files) {
  const cases = [];
  for (const f of files) {
    const { summary, clips } = JSON.parse(readFileSync(f, 'utf8'));
    for (const c of clips) {
      if (c.tokenIds?.length) cases.push({ id: `real-${summary.reciter}-${c.key}`, kind: 'real', source: c.key, tokenIds: c.tokenIds });
    }
  }
  return cases;
}

async function main() {
  let files = process.argv.slice(2);
  if (!files.length) {
    const dir = join(HERE, 'results');
    files = readdirSync(dir).filter((f) => f.endsWith('.json')).map((f) => join(dir, f));
  }
  const assets = loadAsrAssets();
  const vocabSize = Object.keys(assets.vocab).length;
  const cases = [...realCases(files), ...syntheticCases(assets.quranCtcTokens, vocabSize, assets.blankId)];

  const runner = oneHotRunner(vocabSize, assets.blankId);
  const session = createAsrSession(runner, assets);
  for (const c of cases) {
    runner.setTokens(c.tokenIds);
    c.expected = await session.transcribe(new Float32Array(1));
  }
  mkdirSync(dirname(OUT), { recursive: true });
  writeFileSync(OUT, JSON.stringify({ vocabSize, blankId: assets.blankId, cases }));
  const byKind = {};
  for (const c of cases) byKind[c.kind] = (byKind[c.kind] ?? 0) + 1;
  console.log('cases', cases.length, byKind, '->', OUT);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
