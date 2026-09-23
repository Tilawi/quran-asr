/**
 * Node helpers shared by the eval harness and the examples: decode audio with
 * ffmpeg and run the ONNX model with onnxruntime-node. The runner has the same
 * contract as the React Native one (@tilawi/react-native-quran-asr):
 * audio_signal float32 [1, N], length int64 [1], graphOptimizationLevel 'all'.
 */
import { spawnSync } from 'node:child_process';
import ort from 'onnxruntime-node';

export const SAMPLE_RATE = 16000;
export const ortVersion = () => ort.env.versions?.common ?? 'unknown';

/** Decode any audio file to 16 kHz mono float32 PCM with ffmpeg. */
export function decodePcm(file) {
  const r = spawnSync('ffmpeg', ['-v', 'error', '-i', file, '-ac', '1', '-ar', String(SAMPLE_RATE), '-f', 'f32le', '-'], {
    maxBuffer: 1 << 30,
  });
  if (r.status !== 0) throw new Error(`ffmpeg failed for ${file}: ${r.stderr}`);
  const buf = r.stdout;
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4).slice();
}

export async function createNodeRunner(modelPath) {
  const session = await ort.InferenceSession.create(modelPath, { graphOptimizationLevel: 'all' });
  const audioName = session.inputNames.find((n) => /audio|signal|input/i.test(n)) ?? session.inputNames[0];
  const lengthName = session.inputNames.find((n) => /len/i.test(n)) ?? session.inputNames[1];
  const outputName = session.outputNames[0];
  return {
    async run(audio) {
      const audioTensor = new ort.Tensor('float32', audio, [1, audio.length]);
      const lengthTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(audio.length)]), [1]);
      const results = await session.run({ [audioName]: audioTensor, [lengthName]: lengthTensor });
      const out = results[outputName];
      const dims = out.dims;
      return {
        logprobs: out.data,
        timeSteps: Number(dims[dims.length - 2]),
        vocabSize: Number(dims[dims.length - 1]),
      };
    },
  };
}
