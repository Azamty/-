export type SelectionSuggestion = {
  bpm?: number | string | null;
  key?: string | null;
  time_signature?: string | null;
};

export type SavedSelection = {
  schema?: string;
  version?: number;
  job_id?: string;
  hydrated?: boolean;
  bpm?: number | string | null;
  key?: string | null;
  meter?: string | null;
  manual?: ManualSelectionFields;
  selectedTrackIds?: string[];
  rollVisible?: boolean;
};

export type ManualSelectionFields = {
  bpm?: boolean;
  key?: boolean;
  meter?: boolean;
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

export const SELECTION_DRAFT_SCHEMA = "jianpu-v2-selection";
export const SELECTION_DRAFT_VERSION = 1;

const JOB_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export type InitialJobId = {
  jobId: string | null;
  invalidQuery: boolean;
};

export function isValidJobId(value: string | null | undefined): value is string {
  return typeof value === "string" && JOB_ID_PATTERN.test(value.trim());
}

/** Resolve a direct job link before the last job saved by the local app. */
export function resolveInitialJobId(search: string, storedJobId: string | null): InitialJobId {
  const queryJobId = new URLSearchParams(search).get("job")?.trim() || "";
  if (queryJobId) {
    return {
      jobId: isValidJobId(queryJobId) ? queryJobId : null,
      invalidQuery: !isValidJobId(queryJobId),
    };
  }
  return { jobId: isValidJobId(storedJobId) ? storedJobId.trim() : null, invalidQuery: false };
}

const firstValue = <T>(...values: Array<T | null | undefined>): T | undefined => {
  const found = values.find((value) => value !== undefined && value !== null && String(value).trim() !== "");
  return found === null || found === undefined ? undefined : found;
};

export function selectionStorageKey(jobId: string): string {
  return `jianpu-v2-selection:${jobId}`;
}

export function parseSelectionDraft(raw: string | null, jobId: string): SavedSelection | null {
  if (!raw) return null;
  try {
    const value = JSON.parse(raw) as SavedSelection;
    if (
      value?.schema !== SELECTION_DRAFT_SCHEMA
      || value.version !== SELECTION_DRAFT_VERSION
      || value.job_id !== jobId
      || value.hydrated !== true
    ) return null;
    return value;
  } catch {
    return null;
  }
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
  const savedBpm = saved?.manual?.bpm ? saved.bpm : undefined;
  const savedKey = saved?.manual?.key ? saved.key : undefined;
  const savedMeter = saved?.manual?.meter ? saved.meter : undefined;
  return {
    bpm: String(firstValue(overrides?.bpm, savedBpm, suggestion?.bpm, 120)),
    key: String(firstValue(overrides?.key, savedKey, suggestion?.key, "C")),
    meter: String(firstValue(overrides?.time_signature, savedMeter, suggestion?.time_signature, "4/4")),
  };
}
