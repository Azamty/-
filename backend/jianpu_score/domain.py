"""Stable data contracts shared by analysis, quantization and rendering."""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAJOR_KEYS = frozenset({"C", "C#", "Db", "D", "Eb", "E", "F", "F#", "Gb", "G", "Ab", "A", "Bb", "B"})
MINOR_KEYS = frozenset(f"{root}m" for root in MAJOR_KEYS)
VALID_KEYS = MAJOR_KEYS | MINOR_KEYS
VALID_TIME_SIGNATURES = frozenset({"2/4", "3/4", "4/4", "6/8"})
RELATIVE_MAJOR_KEYS = {
    "Cm": "Eb",
    "C#m": "E",
    "Dbm": "E",
    "Dm": "F",
    "Ebm": "Gb",
    "Em": "G",
    "Fm": "Ab",
    "F#m": "A",
    "Gbm": "A",
    "Gm": "Bb",
    "Abm": "B",
    "Am": "C",
    "Bbm": "Db",
    "Bm": "D",
}
_KEY_PATTERN = re.compile(r"^([A-Ga-g])([#b]?)(m?)$")
_KEY_ROOTS = {"C": "C", "C#": "C#", "Cb": None, "D": "D", "D#": "D#", "Db": "Db", "E": "E", "Eb": "Eb", "F": "F", "F#": "F#", "Fb": None, "G": "G", "G#": "G#", "Gb": "Gb", "A": "A", "A#": "A#", "Ab": "Ab", "B": "B", "Bb": "Bb"}


def normalize_key(value: str) -> str:
    """Return a jianpu-safe key from the explicit supported whitelist."""

    if not isinstance(value, str):
        raise ValueError("key must be a string")
    text = value.strip()
    match = _KEY_PATTERN.fullmatch(text)
    if not match:
        raise ValueError(f"unsupported key: {value!r}")
    root, accidental, minor = match.groups()
    canonical_root = _KEY_ROOTS.get(root.upper() + accidental)
    if canonical_root is None:
        raise ValueError(f"unsupported key: {value!r}")
    key = canonical_root + minor
    if key not in VALID_KEYS:
        raise ValueError(f"unsupported key: {value!r}")
    return key


def relative_major_key(value: str) -> str:
    """Return the renderer-safe relative major for a supported key."""

    key = normalize_key(value)
    return RELATIVE_MAJOR_KEYS.get(key, key)


def normalize_time_signature(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("time_signature must be a string")
    text = value.strip()
    if text not in VALID_TIME_SIGNATURES:
        raise ValueError(f"unsupported time signature: {value!r}; choose one of {sorted(VALID_TIME_SIGNATURES)}")
    return text


def sanitize_title(value: str) -> str:
    """Make a title a single safe jianpu metadata line."""

    if not isinstance(value, str):
        value = str(value)
    unsafe = set("\\\"{}%#;|[]~()<>=")
    safe = "".join(" " if not char.isprintable() or char in unsafe else char for char in value)
    return " ".join(safe.split()) or "Untitled"


def confidence_value(value: float | None) -> float:
    """Neutral ordering score for engines that do not expose calibrated confidence."""

    return value if value is not None and math.isfinite(value) else 0.5


class NoteEvent(BaseModel):
    """A single source note with unquantized seconds and MIDI pitch."""

    model_config = ConfigDict(extra="forbid")

    start_sec: float = Field(ge=0, allow_inf_nan=False)
    end_sec: float = Field(gt=0, allow_inf_nan=False)
    midi: int = Field(ge=0, le=127)
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    voice_id: str = "voice-0"
    source: str = "unknown"
    velocity: int | None = Field(default=None, ge=1, le=127)
    raw_pitch: float | None = Field(default=None, allow_inf_nan=False)
    stem_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_interval(self) -> "NoteEvent":
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        return self

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


class MusicAnalysis(BaseModel):
    """Audio-level analysis before beat quantization."""

    model_config = ConfigDict(extra="forbid")

    sample_rate: int = Field(gt=0)
    duration_sec: float = Field(gt=0, allow_inf_nan=False)
    bpm: float = Field(gt=0, allow_inf_nan=False)
    time_signature: str = "4/4"
    key: str = "C"
    beat_times: list[float] = Field(default_factory=list)
    note_events: list[NoteEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return normalize_key(value)

    @field_validator("time_signature")
    @classmethod
    def validate_time_signature(cls, value: str) -> str:
        return normalize_time_signature(value)

    @field_validator("beat_times")
    @classmethod
    def validate_beat_times(cls, value: list[float]) -> list[float]:
        if any(not math.isfinite(beat) for beat in value):
            raise ValueError("beat_times must contain only finite values")
        if any(right <= left for left, right in zip(value, value[1:])):
            raise ValueError("beat_times must be strictly increasing")
        return value


class TempoEvent(BaseModel):
    """A tempo change on the renderer-independent score timeline."""

    model_config = ConfigDict(extra="forbid")

    start_tick: int = Field(ge=0)
    bpm: float = Field(gt=0, allow_inf_nan=False)


class ScoreNote(BaseModel):
    """A quantized note or rest on the shared score timeline."""

    model_config = ConfigDict(extra="forbid")

    start_tick: int = Field(ge=0)
    duration_tick: int = Field(gt=0)
    midi: int | None = Field(default=None, ge=0, le=127)
    voice_id: str = "voice-0"
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    source: str = "unknown"
    velocity: int | None = Field(default=None, ge=1, le=127)
    raw_pitch: float | None = Field(default=None, allow_inf_nan=False)
    stem_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.duration_tick

    @property
    def is_rest(self) -> bool:
        return self.midi is None


class ScoreVoice(BaseModel):
    """One independently timed monophonic layer in a score."""

    model_config = ConfigDict(extra="forbid")

    voice_id: str
    events: list[ScoreNote] = Field(default_factory=list)
    label: str | None = None
    stem_id: str | None = None


class Score(BaseModel):
    """Renderer-independent score contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    title: str = "Untitled"
    bpm: float = Field(gt=0, allow_inf_nan=False)
    key: str = "C"
    time_signature: str = "4/4"
    quarter_ticks: int = Field(default=12, gt=0)
    total_ticks: int = Field(gt=0)
    voices: list[ScoreVoice] = Field(min_length=1)
    tempo_events: list[TempoEvent] = Field(default_factory=list)
    source: str = "unknown"
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return sanitize_title(value)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return normalize_key(value)

    @field_validator("time_signature")
    @classmethod
    def validate_time_signature(cls, value: str) -> str:
        return normalize_time_signature(value)

    @model_validator(mode="after")
    def validate_voice_timelines(self) -> "Score":
        for voice in self.voices:
            cursor = 0
            for event in voice.events:
                if event.start_tick != cursor:
                    raise ValueError(f"voice {voice.voice_id} has a timeline gap or overlap at {cursor}")
                cursor = event.end_tick
            if cursor != self.total_ticks:
                raise ValueError(f"voice {voice.voice_id} ends at {cursor}, expected {self.total_ticks}")
        previous_tick = -1
        for tempo in self.tempo_events:
            if tempo.start_tick < previous_tick or tempo.start_tick > self.total_ticks:
                raise ValueError("tempo_events must be ordered and lie on the score timeline")
            previous_tick = tempo.start_tick
        return self
