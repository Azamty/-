"""Core audio analysis and deterministic jianpu rendering pipeline."""

from .domain import MusicAnalysis, NoteEvent, Score, ScoreNote, ScoreVoice, TempoEvent, relative_major_key
from .capabilities import capabilities, get_capabilities
from .models.adapter import EngineAdapter, EngineError, EngineExecutionError, EngineResult, EngineUnavailableError
from .quantize import NoNotesError
from .performance_midi import PerformanceMidiArtifact, PerformanceTrack, build_performance_midi, write_performance_midi, write_performance_midi_bundle
from .musescore_import import MusicXMLArtifact, MuseScoreImportError, convert_performance_midi, convert_selected_performance_tracks
from .musicxml_standardize import (
    MusicXMLStandardizationError,
    StandardizedScoreArtifact,
    standardize_musicxml,
    standardize_musicxml_payload,
    write_standardized_score,
)

__all__ = [
    "MusicAnalysis",
    "NoteEvent",
    "Score",
    "ScoreNote",
    "ScoreVoice",
    "TempoEvent",
    "relative_major_key",
    "NoNotesError",
    "PerformanceMidiArtifact",
    "PerformanceTrack",
    "build_performance_midi",
    "write_performance_midi",
    "write_performance_midi_bundle",
    "MusicXMLArtifact",
    "MuseScoreImportError",
    "convert_performance_midi",
    "convert_selected_performance_tracks",
    "MusicXMLStandardizationError",
    "StandardizedScoreArtifact",
    "standardize_musicxml",
    "standardize_musicxml_payload",
    "write_standardized_score",
    "EngineAdapter",
    "EngineError",
    "EngineExecutionError",
    "EngineResult",
    "EngineUnavailableError",
    "capabilities",
    "get_capabilities",
]
