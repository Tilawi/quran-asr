#!/usr/bin/env node
/**
 * On-device ASR evaluation harness (Node).
 *
 * Runs this package's recognition core (unmodified) with the real model through onnxruntime-node, on real recitations from
 * EveryAyah, and reports verse-identification accuracy. It also writes golden
 * per-clip outputs (greedy CTC token ids, transcript, prediction) so a refactor
 * of the core can be checked for byte-identical behaviour.
 *
 * The runner (eval/node.mjs) has the same contract as the React Native one.
 *
 * Usage:
 *   npx tsx eval/run-eval.mjs [--reciter Minshawy_Murattal_128kbps]
 *        [--n 100] [--seed 1] [--out eval/results/<name>.json]
 *
 * Requirements: ffmpeg on PATH, network access to everyayah.com (first run
 * only; audio is cached in ~/.cache/tilawi-asr-eval), the model assets
 * (`npm run fetch-assets`) and devDependency onnxruntime-node.
 *
 * Quran-integrity note: the expected verse is used ONLY to score the result
 * after recognition. It is never given to the recognizer.
 */
import { mkdirSync, existsSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createAsrSession, normalizeArabic } from '../src/index.ts';
import { assetsDir, CTC_BLANK_ID, loadAsrAssets } from '../test/golden-runner.mjs';
import { createNodeRunner, decodePcm, ortVersion, SAMPLE_RATE } from './node.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const args = { reciter: 'Minshawy_Murattal_128kbps', n: 100, seed: 1, out: null };
  for (let i = 2; i < argv.length; i += 2) {
    const k = argv[i].replace(/^--/, '');
    args[k] = k === 'n' || k === 'seed' ? Number(argv[i + 1]) : argv[i + 1];
  }
  args.out ??= join(HERE, 'results', `${args.reciter}-n${args.n}-s${args.seed}.json`);
  return args;
}

/** Deterministic PRNG (mulberry32) so a seed always selects the same verses. */
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

function sampleVerses(quran, n, seed) {
  const rand = rng(seed);
  const idx = new Set();
  while (idx.size < Math.min(n, quran.length)) idx.add(Math.floor(rand() * quran.length));
  return [...idx].sort((a, b) => a - b).map((i) => ({ surah: Number(quran[i].surah), ayah: Number(quran[i].ayah) }));
}

const pad3 = (x) => String(x).padStart(3, '0');

async function fetchAudio(reciter, surah, ayah) {
  const dir = join(homedir(), '.cache', 'tilawi-asr-eval', reciter);
  mkdirSync(dir, { recursive: true });
  const file = join(dir, `${pad3(surah)}${pad3(ayah)}.mp3`);
  if (!existsSync(file)) {
    const url = `https://everyayah.com/data/${reciter}/${pad3(surah)}${pad3(ayah)}.mp3`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`download failed ${res.status} ${url}`);
    writeFileSync(file, Buffer.from(await res.arrayBuffer()));
  }
  return file;
}

/** Greedy CTC ids (argmax per frame, collapse repeats, drop blank) for golden files. */
function greedyIds(logprobs, timeSteps, vocabSize) {
  const ids = [];
  let prev = -1;
  for (let t = 0; t < timeSteps; t++) {
    let best = 0;
    let bestV = -Infinity;
    const base = t * vocabSize;
    for (let v = 0; v < vocabSize; v++) {
      const x = logprobs[base + v];
      if (x > bestV) {
        bestV = x;
        best = v;
      }
    }
    if (best !== prev && best !== CTC_BLANK_ID) ids.push(best);
    prev = best;
  }
  return ids;
}

async function main() {
  const args = parseArgs(process.argv);
  const assets = loadAsrAssets();
  const quran = assets.quran;
  const baseRunner = await createNodeRunner(join(assetsDir(), 'fastconformer_full_mixed.onnx'));
  // Capture the raw model output of each run for the golden token ids.
  let last = null;
  const runner = {
    async run(audio) {
      last = await baseRunner.run(audio);
      return last;
    },
  };
  const session = createAsrSession(runner, assets);

  // Some verses are word-for-word identical to others (e.g. 55:13 recurs 31
  // times in Ar-Rahman; 38:77 = 15:34). Audio cannot distinguish them, so a
  // prediction of an identical-text verse is scored separately, not as WRONG.
  const normText = new Map(quran.map((q) => [`${q.surah}:${q.ayah}`, normalizeArabic(q.text_clean ?? q.text_uthmani)]));
  const sameText = (s, a, v) => normText.get(`${s}:${a}`) === normText.get(`${v.surah}:${v.ayah}`);

  const verses = sampleVerses(quran, args.n, args.seed);
  const clips = [];
  let exact = 0;
  let covered = 0;
  let wrong = 0;
  let none = 0;
  let identical = 0;
  const t0 = Date.now();
  for (const [i, v] of verses.entries()) {
    const key = `${v.surah}:${v.ayah}`;
    const file = await fetchAudio(args.reciter, v.surah, v.ayah);
    const audio = decodePcm(file);
    const start = Date.now();
    const pred = await session.transcribe(audio);
    const ms = Date.now() - start;
    const ids = greedyIds(last.logprobs, last.timeSteps, last.vocabSize);
    const matched = pred.surah > 0 && pred.ayah > 0;
    const end = pred.ayah_end || pred.ayah;
    const isExact = matched && pred.surah === v.surah && pred.ayah === v.ayah;
    const isCovered = matched && pred.surah === v.surah && pred.ayah <= v.ayah && v.ayah <= end;
    const isIdentical = matched && !isCovered && pred.ayah_end == null && sameText(pred.surah, pred.ayah, v);
    if (!matched) none++;
    else if (isCovered) {
      covered++;
      if (isExact) exact++;
    } else if (isIdentical) identical++;
    else wrong++;
    clips.push({
      key,
      seconds: +(audio.length / SAMPLE_RATE).toFixed(2),
      inferenceMs: ms,
      tokenIds: ids,
      transcript: pred.transcript,
      prediction: { surah: pred.surah, ayah: pred.ayah, ayah_end: pred.ayah_end, score: pred.score },
      result: !matched ? 'no-match' : isCovered ? (isExact ? 'exact' : 'covered') : isIdentical ? 'identical-text' : 'WRONG',
    });
    const status = !matched ? 'no-match' : isCovered ? 'ok' : isIdentical ? `identical-text -> ${pred.surah}:${pred.ayah}` : `WRONG -> ${pred.surah}:${pred.ayah}`;
    process.stdout.write(`[${i + 1}/${verses.length}] ${key} ${(audio.length / SAMPLE_RATE).toFixed(1)}s ${ms}ms ${status}\n`);
  }
  const n = verses.length;
  const summary = {
    reciter: args.reciter,
    n,
    seed: args.seed,
    exact,
    covered,
    identicalText: identical,
    wrong,
    noMatch: none,
    accuracyTextCorrect: +((covered + identical) / n).toFixed(4),
    accuracyCovered: +(covered / n).toFixed(4),
    accuracyExact: +(exact / n).toFixed(4),
    wrongRate: +(wrong / n).toFixed(4),
    totalSeconds: +((Date.now() - t0) / 1000).toFixed(1),
    ortVersion: ortVersion(),
  };
  mkdirSync(dirname(args.out), { recursive: true });
  writeFileSync(args.out, JSON.stringify({ summary, clips }, null, 1));
  console.log('\nSUMMARY', JSON.stringify(summary));
  console.log('written', args.out);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
