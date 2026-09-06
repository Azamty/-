export type SelectionSuggestion = {
  bpm?: number | string | null;
  key?: string | null;
  time_signature?: string | null;
};

export type SavedSelection = {
  bpm?: number | string | null;
  key?: string | null;
  meter?: string | null;
};

export type SelectionOverrides = {
  bpm?: number | string | null;
  key?: string | null;
  time_signature?: string | null;
};

export type ResolvedSelectionValues = {
  bpm: string;
  key: string;
  meter: string;
};

const firstValue = <T>(...values: Array<T | null | undefined>): T | undefined => {
  const found = values.find((value) => value !== undefined && value !== null && String(value).trim() !== "");
  return found === null || found === undefined ? undefined : found;
};

export function selectionStorageKey(jobId: string): string {
  return `jianpu-v2-selection:${jobId}`;
}

export function canPersistSelection(status: string | null | undefined, hydrated: boolean): boolean {
  return hydrated && (status === "selection_ready" || status === "completed");
}

/** Resolve server overrides first, then explicit saved values, then analysis advice. */
export function resolveSelectionValues(
  suggestion: SelectionSuggestion | null | undefined,
  saved: SavedSelection | null | undefined,
  overrides: SelectionOverrides | null | undefined,
): ResolvedSelectionValues {
  return {
    bpm: String(firstValue(overrides?.bpm, saved?.bpm, suggestion?.bpm, 120)),
    key: String(firstValue(overrides?.key, saved?.key, suggestion?.key, "C")),
    meter: String(firstValue(overrides?.time_signature, saved?.meter, suggestion?.time_signature, "4/4")),
  };
}
