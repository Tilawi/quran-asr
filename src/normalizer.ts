// Harakat and Quranic annotation marks (U+0610-061A, U+064B-065F, superscript
// alef U+0670, U+06D6-06ED) and tatweel U+0640. No base letters.
const DIACRITICS_RE = /[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]/g;
const NORM_MAP: Record<string, string> = {
  '\u0623': '\u0627', // أ -> ا
  '\u0625': '\u0627', // إ -> ا
  '\u0622': '\u0627', // آ -> ا
  '\u0671': '\u0627', // ٱ -> ا
  '\u0629': '\u0647', // ة -> ه
  '\u0649': '\u064A', // ى -> ي
};

/**
 * Normalize Arabic for matching recognizer output against the canonical text:
 * strip the BOM, diacritics and tatweel, fold alef/ta-marbuta/alef-maqsura
 * variants, and collapse whitespace. Never used to alter displayed Quran text.
 */
export function normalizeArabic(text: string): string {
  text = text.replace(/\uFEFF/g, '');
  text = text.replace(DIACRITICS_RE, '');
  text = text.replace(/./g, (ch) => NORM_MAP[ch] ?? ch);
  text = text.split(/\s+/).filter(Boolean).join(' ');
  return text;
}
