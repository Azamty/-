export type ScoreArtifact = {
  artifact_id: string;
  kind: string;
  page?: number | null;
};

export type ScoreArtifactBuckets = {
  long: ScoreArtifact[];
  paged: ScoreArtifact[];
};

const SCORE_KINDS = {
  instrument: new Set([
    "instrument_score_svg_long",
    "main_melody_score_svg_long",
    "instrument_score_svg",
    "main_melody_score_svg",
    // Kept for one compatibility cycle with pre-9B artifacts.
    "main_melody_svg_long",
    "main_melody_svg",
  ]),
  vocal: new Set([
    "vocal_score_svg_long",
    "main_melody_score_svg_long",
    "vocal_score_svg",
    "main_melody_score_svg",
    // Kept for one compatibility cycle with pre-9B artifacts.
    "score_svg_long",
    "score_svg",
  ]),
} as const;

export function classifyScoreArtifacts<T extends ScoreArtifact>(
  artifacts: readonly T[],
  family: keyof typeof SCORE_KINDS,
): { long: T[]; paged: T[] } {
  const matched = artifacts
    .filter((artifact) => SCORE_KINDS[family].has(artifact.kind))
    .sort((left, right) => {
      const longDelta = Number(right.kind.endsWith("_long")) - Number(left.kind.endsWith("_long"));
      return longDelta
        || (Number(left.page || 0) - Number(right.page || 0))
        || left.artifact_id.localeCompare(right.artifact_id);
    });
  return {
    long: matched.filter((artifact) => artifact.kind.endsWith("_long")),
    paged: matched.filter((artifact) => !artifact.kind.endsWith("_long")),
  };
}
