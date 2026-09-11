export type ScoreArtifact = {
  artifact_id: string;
  kind: string;
  page?: number | null;
  relative_path?: string | null;
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

const SELECTION_ARTIFACT_ID_REVISION = /^v2-selection-r(\d+)(?:-|$)/;
const SELECTION_ARTIFACT_PATH_REVISION = /(?:^|[/\\])rev-(\d+)(?:[/\\]|$)/;

/**
 * Read the selection revision encoded by the V2 artifact id or output path.
 * Older artifacts have neither marker and intentionally return null so they
 * remain available through the compatibility path.
 */
export function selectionArtifactRevision(artifact: Pick<ScoreArtifact, "artifact_id" | "relative_path">): number | null {
  const idMatch = SELECTION_ARTIFACT_ID_REVISION.exec(artifact.artifact_id);
  if (idMatch) return Number(idMatch[1]);
  const pathMatch = artifact.relative_path ? SELECTION_ARTIFACT_PATH_REVISION.exec(artifact.relative_path) : null;
  return pathMatch ? Number(pathMatch[1]) : null;
}

/**
 * Keep artifacts for the active selection revision while retaining artifacts
 * from the pre-revision API, which have no revision marker at all.
 */
export function filterArtifactsForSelectionRevision<T extends ScoreArtifact>(
  artifacts: readonly T[],
  selectionRevision?: number | null,
): T[] {
  if (selectionRevision === undefined || selectionRevision === null) return [...artifacts];
  return artifacts.filter((artifact) => {
    const artifactRevision = selectionArtifactRevision(artifact);
    return artifactRevision === null || artifactRevision === selectionRevision;
  });
}

export function classifyScoreArtifacts<T extends ScoreArtifact>(
  artifacts: readonly T[],
  family: keyof typeof SCORE_KINDS,
  selectionRevision?: number | null,
): { long: T[]; paged: T[] } {
  const matched = filterArtifactsForSelectionRevision(artifacts, selectionRevision)
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
