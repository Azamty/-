const ROOTS = ["C", "C#", "Db", "D", "Eb", "E", "F", "F#", "Gb", "G", "Ab", "A", "Bb", "B"];
export const NOTATION_KEYS = [...ROOTS, ...ROOTS.map(root => `${root}m`)];
const RELATIVE_MAJOR: Record<string, string> = { Cm: "Eb", "C#m": "E", Dbm: "E", Dm: "F", Ebm: "Gb", Em: "G", Fm: "Ab", "F#m": "A", Gbm: "A", Gm: "Bb", Abm: "B", Am: "C", Bbm: "Db", Bm: "D" };
export function keyLabel(key: string): string {
  const root = key.replace(/m$/, "").replace("#", "♯").replace("b", "♭");
  return `${root}${key.endsWith("m") ? "小调" : "大调"}（${key}）· 1=${RELATIVE_MAJOR[key] || key}`;
}
export function isScoreDownload(kind: string): boolean {
  return /^(instrument|vocal|main_melody|melody_harmony)_score_(pdf|midi|svg|svg_long)$/.test(kind);
}
