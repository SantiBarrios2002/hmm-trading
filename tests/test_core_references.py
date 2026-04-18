"""Tests for the paper reference helpers and module category registry."""

from __future__ import annotations

import types

import pytest

from hft_hmm.core.references import (
    ALL_CATEGORIES,
    ENGINEERING_APPROXIMATION,
    EVALUATION_LAYER,
    PAPER_FAITHFUL,
    PaperReference,
    module_category,
    reference,
)


def test_category_constants_distinct_and_covered():
    assert ALL_CATEGORIES == frozenset(
        {PAPER_FAITHFUL, ENGINEERING_APPROXIMATION, EVALUATION_LAYER}
    )
    assert len(ALL_CATEGORIES) == 3


def test_reference_factory_produces_paper_reference():
    ref = reference("§3.2", "forward filtering")
    assert isinstance(ref, PaperReference)
    assert ref.section == "§3.2"
    assert ref.topic == "forward filtering"


def test_reference_strips_whitespace():
    ref = reference("  §3  ", "  topic  ")
    assert ref.section == "§3"
    assert ref.topic == "topic"


def test_reference_str_format():
    ref = reference("§3.2", "forward filtering")
    assert str(ref) == "§3.2 — forward filtering"


@pytest.mark.parametrize(
    "section,topic",
    [("", "topic"), ("   ", "topic"), ("§3", ""), ("§3", "   ")],
)
def test_reference_rejects_empty_fields(section, topic):
    with pytest.raises(ValueError):
        reference(section, topic)


def test_paper_reference_is_frozen():
    ref = reference("§3", "t")
    with pytest.raises(AttributeError):
        ref.section = "§4"  # type: ignore[misc]


def test_module_category_missing_returns_none():
    module = types.ModuleType("fake_module")
    assert module_category(module) is None


@pytest.mark.parametrize(
    "category",
    [PAPER_FAITHFUL, ENGINEERING_APPROXIMATION, EVALUATION_LAYER],
)
def test_module_category_valid(category):
    module = types.ModuleType("fake_module")
    module.__category__ = category
    assert module_category(module) == category


def test_module_category_rejects_unknown_value():
    module = types.ModuleType("fake_module")
    module.__category__ = "unknown-category"
    with pytest.raises(ValueError, match="unknown __category__"):
        module_category(module)


def test_existing_modules_declare_valid_categories():
    from hft_hmm import data, preprocessing, project

    for module in (data, preprocessing, project):
        category = module_category(module)
        assert category in ALL_CATEGORIES, f"{module.__name__} missing __category__"
