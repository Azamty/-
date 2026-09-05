"""V2 MuScriptor routing and post-transcription selection policy.

The model always receives the complete instrumental mix.  Instrument choices
are applied after full decoding so the UI can offer the model's complete
instrument inventory without running one inference per selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence


SourceKind = Literal["instrumental", "vocal"]
EngineName = Literal["muscriptor", "game"]
DRUMS = "drums"


@dataclass(frozen=True)
class MuscriptorNote:
    """A normalized MuScriptor event suitable for policy and artifact code."""

    instrument: str
    pitch: int
    start_sec: float
    end_sec: float

    @property
    def is_drum(self) -> bool:
        return self.instrument == DRUMS


@dataclass(frozen=True)
class TranscriptionPlan:
    """The V2 choices that follow a complete model decode."""

    source_kind: SourceKind
    engine: EngineName
    use_demucs: bool
    detected_instruments: tuple[str, ...]
    selected_pitched_instruments: tuple[str, ...]
    drum_preview_enabled: bool
    merge_main_melody: bool

    @property
    def score_instruments(self) -> tuple[str, ...]:
        """Pitched instruments eligible for jianpu score generation."""

        return self.selected_pitched_instruments


def route_for_source(source_kind: str) -> tuple[EngineName, bool]:
    """Return the V2 engine and separation choice for an input source.

    V2 intentionally does not call Demucs.  MuScriptor receives an
    instrumental mix directly, while GAME handles a vocal source directly.
    """

    if source_kind == "instrumental":
        return "muscriptor", False
    if source_kind == "vocal":
        return "game", False
    raise ValueError("source_kind must be 'instrumental' or 'vocal'")


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value).strip()
        if name and name not in seen:
            result.append(name)
            seen.add(name)
    return tuple(result)


def make_transcription_plan(
    source_kind: str,
    detected_instruments: Iterable[str],
    selected_instruments: Sequence[str] | None = None,
    *,
    include_drums: bool = True,
    merge_main_melody: bool = False,
) -> TranscriptionPlan:
    """Build post-decode instrument and score selection.

    ``selected_instruments=None`` means every detected pitched instrument is
    selected.  Drums remain available for playback and MIDI, but are never in
    ``score_instruments``.
    """

    engine, use_demucs = route_for_source(source_kind)
    detected = _ordered_unique(detected_instruments)
    detected_pitched = tuple(name for name in detected if name != DRUMS)
    detected_has_drums = DRUMS in detected

    if selected_instruments is None:
        selected_pitched = detected_pitched
    else:
        requested = _ordered_unique(selected_instruments)
        unknown = tuple(name for name in requested if name not in detected)
        if unknown:
            raise ValueError(
                "selected instruments were not detected: " + ", ".join(unknown)
            )
        selected_pitched = tuple(name for name in requested if name != DRUMS)

    return TranscriptionPlan(
        source_kind=source_kind,  # type: ignore[arg-type]
        engine=engine,
        use_demucs=use_demucs,
        detected_instruments=detected,
        selected_pitched_instruments=selected_pitched,
        drum_preview_enabled=bool(include_drums and detected_has_drums),
        merge_main_melody=merge_main_melody,
    )


def partition_notes(
    notes: Iterable[MuscriptorNote], plan: TranscriptionPlan
) -> Mapping[str, tuple[MuscriptorNote, ...]]:
    """Partition full decode events into score stems and optional drum MIDI."""

    partitions: dict[str, list[MuscriptorNote]] = {
        name: [] for name in plan.selected_pitched_instruments
    }
    if plan.drum_preview_enabled:
        partitions[DRUMS] = []
    for note in notes:
        if note.instrument == DRUMS:
            if plan.drum_preview_enabled:
                partitions[DRUMS].append(note)
        elif note.instrument in partitions:
            partitions[note.instrument].append(note)
    return {name: tuple(values) for name, values in partitions.items()}
