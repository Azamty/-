"""Core audio analysis and deterministic jianpu rendering pipeline."""

from .domain import MusicAnalysis, NoteEvent, Score, ScoreNote, ScoreVoice, TempoEvent, relative_major_key
from .capabilities import capabilities, get_capabilities
from .models.adapter import EngineAdapter, EngineError, EngineExecutionError, EngineResult, EngineUnavailableError
from .quantize import NoNotesError
from .performance_midi import PerformanceMidiArtifact, PerformanceTrack, build_performance_midi, write_performance_midi, write_performance_midi_bundle

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
    "EngineAdapter",
    "EngineError",
    "EngineExecutionError",
    "EngineResult",
    "EngineUnavailableError",
    "capabilities",
    "get_capabilities",
]
