"""Banners and chronological ordering (Design Document §7)."""

import pytest

from domain import Banner, sorted_chronologically
from domain.versions import parse_version


class TestVersionParsing:
    def test_simple_version(self):
        assert parse_version("7.0") == (7, 0)

    @pytest.mark.parametrize(
        "bad", ["7", "v7.0", "7.0.1", "seven.0", "7.o", "", ".5", "7."]
    )
    def test_malformed_versions_are_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_version(bad)


class TestBannerValidation:
    def test_phase_starts_at_one(self):
        with pytest.raises(ValueError, match="phase"):
            Banner("Vesna", "7.0", 0)

    def test_malformed_version_rejected(self):
        with pytest.raises(ValueError, match="version"):
            Banner("Vesna", "7", 1)


class TestChronologicalOrdering:
    def test_order_key_is_major_minor_phase(self):
        assert Banner("Vesna", "7.0", 2).order_key == (7, 0, 2)

    def test_versions_order_numerically_across_decades(self):
        """Plain string sorting would put "10.0" before "9.5"."""
        banners = [
            Banner("Vodynista", "10.0", 1),
            Banner("Tsaritsa", "9.5", 1),
        ]
        ordered = sorted_chronologically(banners)
        assert [b.version for b in ordered] == ["9.5", "10.0"]

    def test_phase_breaks_version_ties(self):
        banners = [
            Banner("Vesna", "7.2", 2),
            Banner("Vesna", "7.2", 1),
        ]
        ordered = sorted_chronologically(banners)
        assert [b.phase for b in ordered] == [1, 2]

    def test_input_order_is_not_mutated(self):
        banners = [Banner("Tsaritsa", "7.1", 1), Banner("Vesna", "7.0", 1)]
        sorted_chronologically(banners)
        assert [b.character for b in banners] == ["Tsaritsa", "Vesna"]
