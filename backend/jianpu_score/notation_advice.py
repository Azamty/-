"""Rank readable notation interpretations independently of onset detection."""
from __future__ import annotations

import math
import numpy as np

from .domain import MusicAnalysis

ROOTS = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
PROFILES = {
    "": np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
    "m": np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
}


def recommend_notation(analysis: MusicAnalysis, notes: list[dict]) -> dict:
    pitched = [n for n in notes if not n.get("is_drum") and n.get("end_sec", 0) > n.get("start_sec", 0)]
    tempo = analysis.metadata.get("beat_grid", {}).get("tempo", {})
    detected = float(tempo.get("detected_bpm") or analysis.bpm)
    candidates = []
    durations = np.array([float(n["end_sec"])-float(n["start_sec"]) for n in pitched])
    starts = np.array([float(n["start_sec"]) for n in pitched])
    gaps = np.maximum(0, np.diff(np.sort(np.unique(starts))))
    for factor in (0.5, 1, 2):
        bpm = detected * factor
        if not 30 <= bpm <= 300:
            continue
        # Dense subdivisions should not count as evidence for a faster tactus.
        # Compare quantization loss and notation complexity, with a soft tempo prior.
        units = durations * bpm / 60 * 4
        short = float(np.mean(units < .75)) if len(units) else 0
        long = float(np.mean(units > 16)) if len(units) else 0
        error = float(np.mean(abs(units-np.round(units)))) if len(units) else 0
        if len(analysis.beat_times) >= 2 and len(starts):
            positions = np.interp(starts, analysis.beat_times, np.arange(len(analysis.beat_times))) * bpm / analysis.bpm
            placement = float(np.mean(abs(positions*4-np.round(positions*4))))
            ties = float(np.mean(np.maximum(0, np.floor(positions+units/4)-np.ceil(positions))))
        else:
            placement, ties = 0., 0.
        tiny_gaps = float(np.mean(gaps*bpm/60*4 < .75)) if len(gaps) else 0.
        score = .45 * abs(math.log2(bpm/100)) + 1.2*short + .3*long + .4*error + .25*placement + .03*ties + .1*tiny_gaps
        candidates.append({"bpm": round(bpm, 4), "score": round(score, 6),
                           "short_note_fraction": short, "long_note_fraction": long,
                           "duration_grid_error": error, "onset_grid_error": placement,
                           "beat_crossings": ties, "short_attack_gap_fraction": tiny_gaps})
    candidates.sort(key=lambda c: c["score"])
    hist = np.zeros(12)
    bass = np.zeros(12)
    endings = np.zeros(12)
    attacks: dict[float, list[dict]] = {}
    for n in pitched:
        p = int(n["pitch"])
        hist[p % 12] += min(2., float(n["end_sec"])-float(n["start_sec"]))
        attacks.setdefault(round(float(n["start_sec"])*20)/20, []).append(n)
    ordered = sorted(attacks.items())
    for i, (start, chord) in enumerate(ordered):
        low = min(chord, key=lambda n: n["pitch"])
        bass[int(low["pitch"]) % 12] += min(2., low["end_sec"]-low["start_sec"])
        end = max(n["end_sec"] for n in chord)
        if i == len(ordered)-1 or ordered[i+1][0] - end >= .3:
            endings[int(low["pitch"]) % 12] += 1
    keys = []
    if len(pitched) >= 8 and np.count_nonzero(hist) >= 3:
        for root in range(12):
            for mode, profile in PROFILES.items():
                corr = float(np.corrcoef(hist, np.roll(profile, root))[0, 1]) if np.std(hist) else 0.
                tonic = bass[root] / max(1e-9, bass.sum())
                # One final note alone is weak tonal evidence.
                cadence = endings[root] / (4. + endings.sum())
                score = corr + .65*tonic + .2*cadence
                keys.append({"key": ROOTS[root]+mode, "score": round(float(score), 6),
                             "profile_correlation": corr, "bass_tonic_share": float(tonic),
                             "phrase_end_share": float(cadence)})
        keys.sort(key=lambda c: c["score"], reverse=True)
    return {"version": 1, "bpm": candidates[0]["bpm"] if candidates else analysis.bpm,
            "key": keys[0]["key"] if keys else analysis.key,
            "tempo_candidates": candidates, "key_candidates": keys[:4] or [{"key": analysis.key, "score": 0.}],
            "key_ambiguous": len(keys) > 1 and keys[0]["score"]-keys[1]["score"] < .12,
            "note_count": len(pitched), "reason": "结合记谱复杂度、音高时长分布、低音与乐句停顿选择；支持手动覆盖"}
