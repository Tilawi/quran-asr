/**
 * Minimal speech-presence check, run BEFORE inference.
 *
 * On silence or near-silence the acoustic model can emit a stray token or two
 * (observed: "لا." from digital silence), which the matcher may then match to a
 * short verse. Recognizing nothing when nobody spoke is always correct, so
 * clips without a stretch of real signal energy are rejected up front.
 *
 * A clip "has speech" when at least `minVoicedMs` of its fixed-size frames have
 * an RMS above `rmsThreshold` (default -50 dBFS). Normal recitation into a
 * phone is roughly -35 to -15 dBFS, digital silence well below -70 dBFS.
 */
export interface SpeechDetectionOptions {
  /** Sample rate of `audio` in Hz. Default 16000. */
  sampleRate?: number;
  /** Frame length in milliseconds. Default 30. */
  frameMs?: number;
  /** Linear RMS a frame must exceed to count as voiced. Default 0.003 (~ -50 dBFS). */
  rmsThreshold?: number;
  /** Total voiced duration required, in milliseconds. Default 300. */
  minVoicedMs?: number;
}

export const SPEECH_DEFAULTS: Required<SpeechDetectionOptions> = {
  sampleRate: 16000,
  frameMs: 30,
  rmsThreshold: 0.003,
  minVoicedMs: 300,
};

/** Milliseconds of `audio` whose frame RMS exceeds the threshold. */
export function voicedMilliseconds(audio: Float32Array, options: SpeechDetectionOptions = {}): number {
  const { sampleRate, frameMs, rmsThreshold } = { ...SPEECH_DEFAULTS, ...options };
  const frame = Math.max(1, Math.round((sampleRate * frameMs) / 1000));
  const thresholdSq = rmsThreshold * rmsThreshold;
  let voicedFrames = 0;
  for (let start = 0; start + frame <= audio.length; start += frame) {
    let sum = 0;
    for (let i = start; i < start + frame; i++) sum += audio[i] * audio[i];
    if (sum / frame > thresholdSq) voicedFrames++;
  }
  return voicedFrames * frameMs;
}

/** True when the clip contains at least `minVoicedMs` of signal above the threshold. */
export function detectSpeech(audio: Float32Array, options: SpeechDetectionOptions = {}): boolean {
  const { minVoicedMs } = { ...SPEECH_DEFAULTS, ...options };
  return voicedMilliseconds(audio, options) >= minVoicedMs;
}
