/**
 * Golden regression suite for the on-device ASR core (decode -> match -> trust
 * gate). Cases and expected outputs come from scripts/asr-eval/make-goldens.mjs:
 * real recitations (token ids recorded from the real model on EveryAyah audio)
 * plus deterministic synthetic cases. The model is replaced by a one-hot runner,
 * so everything after the neural network is checked for byte-identical output.
 *
 * A refactor of @tilawi/quran-asr (packages/quran-asr) must keep every
 * case identical. If a change is INTENDED to alter matching, regenerate the
 * goldens deliberately and justify it with scripts/asr-eval/run-eval.mjs
 * accuracy numbers; never regenerate just to make this pass.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { createAsrSession, type AsrPrediction } from '../src/index';
// Plain .mjs helper shared with the golden generator (scripts/asr-eval).
import { loadAsrAssets, oneHotRunner } from './golden-runner.mjs';

interface GoldenCase {
  id: string;
  kind: string;
  source: string | null;
  tokenIds: number[];
  expected: AsrPrediction;
}

const goldens = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'asr-goldens.json'), 'utf8')) as {
  vocabSize: number;
  blankId: number;
  cases: GoldenCase[];
};

// Matching costs ~0.5-1 s per case, so the default run checks a stratified
// subset (every kind, deterministic). ASR_GOLDENS=full checks every case:
// run it for any change to the ASR core and before releases.
const FULL = process.env.ASR_GOLDENS === 'full';
const PER_KIND = 3;
const selected: GoldenCase[] = FULL
  ? goldens.cases
  : Object.values(
      goldens.cases.reduce<Record<string, GoldenCase[]>>((acc, c) => {
        (acc[c.kind] ??= []).push(c);
        return acc;
      }, {}),
    ).flatMap((group) => group.filter((_, i) => i % Math.max(1, Math.floor(group.length / PER_KIND)) === 0).slice(0, PER_KIND));

// History: this suite first certified the TypeScript core byte-identical to
// the earlier JavaScript implementation on all 640 cases. The
// MIN_TRUSTED_LETTERS trust gate then deliberately changed 2 cases (4-letter
// transcripts: one wrong verse prevented, one one-word match dropped);
// real-recitation accuracy was unchanged (600 evaluated, 0 wrong).
describe('ASR core golden regression (@tilawi/quran-asr)', () => {
  it(
    `reproduces ${selected.length}${FULL ? '' : ` (of ${goldens.cases.length}; ASR_GOLDENS=full for all)`} golden cases exactly`,
    async () => {
      const assets = loadAsrAssets();
      const runner = oneHotRunner(goldens.vocabSize, goldens.blankId);
      const session = createAsrSession(runner, assets);
      const mismatches: string[] = [];
      for (const c of selected) {
        runner.setTokens(c.tokenIds);
        const actual = await session.transcribe(new Float32Array(1));
        // JSON round-trip: goldens are stored as JSON (undefined fields dropped).
        if (JSON.stringify(JSON.parse(JSON.stringify(actual))) !== JSON.stringify(c.expected)) {
          mismatches.push(`${c.id} (${c.kind} ${c.source ?? ''})`);
        }
      }
      expect(mismatches, `mismatching cases:\n${mismatches.slice(0, 20).join('\n')}`).toEqual([]);
    },
    FULL ? 3_600_000 : 180_000,
  );

  it('covers every case kind', () => {
    const kinds = new Set(goldens.cases.map((c) => c.kind));
    for (const k of ['real', 'verse', 'span', 'prefix', 'suffix', 'fragment', 'dropped', 'substituted', 'bismillah-opening', 'noise']) {
      expect(kinds.has(k), `missing kind ${k}`).toBe(true);
    }
  });
});
