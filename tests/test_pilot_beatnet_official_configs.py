from __future__ import annotations

import numpy as np

from scripts.pilot_beatnet_official_configs import METER_CONFIGS, _config_id, _decoded_records, _f1, _tree_hash


def test_config_id_is_global_and_explicit() -> None:
    assert _config_id(2, (2, 3, 4)) == "model-2_meter-2-3-4"
    assert (6,) in METER_CONFIGS
    assert (2, 3, 4, 6) in METER_CONFIGS


def test_decoded_records_preserve_official_beat_numbers() -> None:
    records = _decoded_records(np.asarray([[0.5, 1.0], [1.0, 2.0], [1.5, 3.0]]))
    assert records[0]["downbeat"] is True
    assert records[2]["beat_number"] == 3


def test_f1_matches_acceptance_tolerance() -> None:
    assert _f1([0.0, 0.5], [0.06, 0.56])["f1"] == 1.0
    assert _f1([0.0, 0.5], [0.08, 0.58])["f1"] == 0.0


def test_tree_hash_includes_relative_names_and_contents(tmp_path) -> None:
    (tmp_path / "a").write_text("one", encoding="utf-8")
    first = _tree_hash(tmp_path)
    (tmp_path / "a").write_text("two", encoding="utf-8")
    assert _tree_hash(tmp_path) != first
