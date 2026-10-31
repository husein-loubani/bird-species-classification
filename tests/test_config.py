"""Sanity checks on the constants every module reads."""

from __future__ import annotations

from bird_species import config


def test_paths_live_inside_the_project():
    for path in (config.RAW_DIR, config.INTERIM_DIR, config.PROCESSED_DIR, config.MODEL_DIR, config.FIGURE_DIR):
        assert config.PROJECT_ROOT in path.parents


def test_the_start_respects_the_brief_cap_of_five():
    assert 1 <= config.START_SHOTS <= config.MAX_SHOTS == 5
    assert max(config.SHOT_CURVE) <= config.MAX_SHOTS


def test_thresholds_are_in_range():
    assert 0 < config.TARGET_ACCURACY < 1
    assert config.SIMILARITY_FLOOR < config.DUPLICATE_SIMILARITY < 1
    assert config.SIMILARITY_BANDS[0] == config.SIMILARITY_FLOOR
    assert list(config.SIMILARITY_BANDS) == sorted(config.SIMILARITY_BANDS)
    assert config.RESIZE_SIZE >= config.IMAGE_SIZE


def test_the_fewshot_backbone_is_one_of_the_compared_backbones():
    assert config.FEWSHOT_BACKBONE in config.BACKBONES
