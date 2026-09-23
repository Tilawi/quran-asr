import type { AsrAssets, SessionRunner } from '../src/index';

export const CTC_BLANK_ID: number;
export function assetsDir(): string;
export function assetPath(stem: string, dir?: string): string;
export function loadAsrAssets(dir?: string): AsrAssets & { quran: unknown[]; blankId: number };
export function oneHotRunner(vocabSize: number, blankId: number): SessionRunner & { setTokens(ids: number[]): void };
