/** A verse record as used by the matcher (canonical text plus derived CTC text/tokens). */
export interface QuranVerse {
  surah: number;
  ayah: number;
  text_uthmani: string;
  text_clean?: string;
  surah_name: string;
  surah_name_en: string;
  /** For the text CTC pipeline these contain normalized Arabic text/tokens. */
  phonemes: string;
  phonemes_joined: string;
  phoneme_tokens?: string[];
  phoneme_tokens_no_bsm?: string[] | null;
  phoneme_token_ids?: number[];
  phoneme_token_ids_no_bsm?: number[] | null;
  word_token_ends?: number[];
  phonemes_joined_no_bsm?: string | null;
  phonemes_joined_ns?: string;
  phonemes_joined_no_bsm_ns?: string | null;
  phoneme_words: string[];
}

/**
 * The model-runtime injection seam. The core never imports an ONNX runtime;
 * the caller wires one in behind this interface (onnxruntime-react-native in
 * an app, onnxruntime-node for tests and evaluation).
 *
 * The contract mirrors the model graph exactly: feed mono 16kHz float32 PCM
 * as `audio_signal` `[1, N]` plus its `length` `[1]`, receive `[1, T, vocab]`
 * log-probabilities flattened row-major into `logprobs` (length `T * vocab`).
 * Preprocessing (mel, normalization) is baked into the graph.
 */
export interface SessionRunner {
  /**
   * Run one forward pass of the acoustic model.
   *
   * @param audio - mono 16kHz PCM, float32, shape `[N]`.
   * @returns flattened log-probs (`timeSteps * vocabSize`) plus the two dims
   *   needed to reshape them into `[T, vocab]`.
   */
  run(audio: Float32Array): Promise<SessionOutput>;
}

export interface SessionOutput {
  /** Row-major `[timeSteps, vocabSize]` log-probabilities. */
  logprobs: Float32Array;
  timeSteps: number;
  vocabSize: number;
}
