# @tilawi/quran-asr

[![npm](https://img.shields.io/npm/v/@tilawi/quran-asr.svg)](https://www.npmjs.com/package/@tilawi/quran-asr) [![license](https://img.shields.io/npm/l/@tilawi/quran-asr.svg)](LICENSE) [![ci](https://github.com/Tilawi/quran-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/Tilawi/quran-asr/actions/workflows/ci.yml)

[npm](https://www.npmjs.com/package/@tilawi/quran-asr) · [Model on Hugging Face](https://huggingface.co/muhdur/tilawi-fastconformer-quran) · [Tilawi app](https://tilawi.ai) · [All Tilawi open source](https://github.com/Tilawi)

On-device Quran recitation recognition: turn a recitation into the verse(s) being recited, privately, with no server.

This is the recognition core behind the [Tilawi](https://tilawi.ai) app. It takes the output of a CTC speech model, decodes it, identifies the recited verse or verse span, and applies a conservative **trust gate**. When the gate isn't confident, it returns "no match" instead of guessing, because naming the wrong verse is worse than naming none.

- **Runtime-agnostic.** You inject a model runner: `onnxruntime-react-native` in an app, `onnxruntime-node` on a server or in tests.
- **Pure TypeScript** with zero runtime dependencies.
- **Never generates Quran text.** The model transcribes the *user's* recitation, and the core only compares that transcript against the fixed canonical text.

## Accuracy

The numbers below come from [`eval/run-eval.mjs`](eval/run-eval.mjs): the real model through this core on 600 verses (200 per reciter, randomly sampled across the whole Quran, seed 1) from [EveryAyah](https://everyayah.com). Real live tests, with people reciting into phones, were done as well.

| Reciter | Correct verse | No match | Wrong verse |
|---|---|---|---|
| Mohamed Siddiq al-Minshawi (murattal) | 200 / 200 | 0 | 0 |
| Mahmoud Khalil al-Husary | 200 / 200 | 0 | 0 |
| Mishary Rashid Alafasy | 199 / 200 | 1 | 0 |

"Correct" includes 12 cases where the predicted verse is **word-for-word identical** to the recited one (e.g. 55:13, which recurs 31 times in Ar-Rahman). Audio alone cannot tell those apart. The one "no match" is a very short verse (104:5) that the trust gate declined to guess.

**About these numbers:** they come from studio recordings because those can be scored automatically. The model has also been tested with live recitation recorded on phones. It isn't perfect, but it is free and open source in the hope that it helps others; the harness is included so you can measure it on your own audio.

### Memorization feedback

`judgeAttempt` was checked against the real transcripts of those 600 recitations (7,551 expected words), comparing to the quran.com (KFGQPC) word text the Tilawi app uses; results in [`eval/results/memo/memo-eval.json`](eval/results/memo/memo-eval.json). (That harness lives in the app repo because it reads the app's text database.) Mistakes are planted only in what the reciter "said"; the canonical text is never modified.

| Scenario | Result |
|---|---|
| Correct recitation | 99.6% of words judged correct; 13 words (0.17%, in 9 verses) wrongly flagged |
| A wrong word was said | 91.7% flagged as an error, 97.8% flagged as an error or "uncertain" |
| A word was skipped | 99.5% flagged as missed |
| An extra word was said | 99.7% reported as extra |

All 13 false flags on correct recitations trace back to the model mishearing or dropping words, not to the judging logic.

**Scoring a passage:** `judgeAttempt` returns two scores. `overallPercent` is correct ÷ (all expected words + extra words), so stopping early lowers it; show this one for a passage. `scorePercent` only covers the words the reciter reached, so 1 correct word of a page scores 100. Words after the point where the reciter stopped come back as `unattempted` (not errors). Words skipped *before* a recited word come back as `omission`.

## Install

```bash
npm install @tilawi/quran-asr
```

The model and data files are downloaded separately (see below).

## Quick start (Node)

The model and its data files live on Hugging Face: [`muhdur/tilawi-fastconformer-quran`](https://huggingface.co/muhdur/tilawi-fastconformer-quran).

```bash
git clone https://github.com/Tilawi/quran-asr && cd quran-asr
npm install
npm run build
npm run fetch-assets                          # ~110 MB, verified by SHA-256
node examples/transcribe.mjs recitation.mp3   # needs ffmpeg on PATH
# e.g. with https://everyayah.com/data/Husary_128kbps/112001.mp3:
# heard: قل هو الله احد
# verse: 112:1 (score 1.00)
```

## Usage

```ts
import { createAsrSession, type SessionRunner } from '@tilawi/quran-asr';

// 1. A runner wrapping your ONNX runtime. The model takes raw 16 kHz mono PCM
//    (audio_signal float32 [1, N] + length int64 [1]) and returns [1, T, vocab]
//    log-probabilities; feature extraction is inside the graph.
//    eval/node.mjs has a complete onnxruntime-node runner.
const runner: SessionRunner = {
  async run(audio) {
    /* run the model, return { logprobs, timeSteps, vocabSize } */
  },
};

// 2. The text-side assets from Hugging Face: vocab.json, quran_ctc_tokens.json, quran.json.
const session = createAsrSession(runner, { vocab, quranCtcTokens, quran, blankId: 1024 });

// 3. Recognize a clip.
const result = await session.transcribe(pcm16kMono);
// { surah: 2, ayah: 255, ayah_end: null, score: 0.93, transcript: '...' }
// surah/ayah are 0 when nothing was trusted.
```

Other exports: `detectSpeech` (reject silence before running the model), `judgeAttempt` (word-by-word memorization feedback against the fixed canonical text), `stitchTranscripts` (long clips in overlapping windows).

In React Native / Expo, use [`@tilawi/react-native-quran-asr`](https://github.com/Tilawi/react-native-quran-asr), which wraps this package with `onnxruntime-react-native` and a microphone hook.

## Tests

```bash
npm run fetch-assets && npm test
```

- `test/asrGoldens.test.ts`: a golden regression suite (real recitations plus synthetic verses, spans, fragments, dropped/substituted tokens, Bismillah openings and noise). The model is replaced by a one-hot runner, so everything after the neural network must produce byte-identical output. Set `ASR_GOLDENS=full` to run all 640 cases.
- `npm run eval -- --reciter Husary_128kbps --n 200`: the end-to-end accuracy harness above (downloads EveryAyah audio; needs ffmpeg).

## License

The code is MIT. The model and data on Hugging Face have their own terms: the model is fine-tuned from NVIDIA's [`stt_ar_fastconformer_hybrid_large_pcd_v1.0`](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0) (CC-BY-4.0), and the Quran text is from the [Tanzil Project](https://tanzil.net) (CC-BY-3.0, verbatim).
