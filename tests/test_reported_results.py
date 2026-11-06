"""REPORT.md, the committed output of the full run, against the paper's claims."""

from __future__ import annotations

import re

import pytest


def row(report: str, name: str) -> tuple[float, ...]:
    match = re.search(rf"\| {re.escape(name)} \| ([\d.]+)% \| ([\d.]+)% \| ([\d.]+)% \|", report)
    assert match, f"{name} row missing from REPORT.md"
    return tuple(float(g) for g in match.groups())


def test_gender_predictability_drops(report):
    assert row(report, "Private (Lp1+Lp2)")[2] < row(report, "Original")[2]


def test_verification_is_kept(report):
    assert row(report, "Private (Lp1+Lp2)")[0] >= row(report, "Original")[0] - 1.0


def test_pic_summary_matches_the_table(report):
    original, private = row(report, "Original"), row(report, "Private (Lp1+Lp2)")
    match = re.search(r"\(PG=(-?[\d.]+), IL=(-?[\d.]+)\)", report)
    assert match
    assert float(match.group(1)) * 100 == pytest.approx(original[2] - private[2], abs=0.02)
    assert float(match.group(2)) * 100 == pytest.approx(private[1] - original[1], abs=0.02)


def test_published_values(report, targets):
    if targets["published_values"] is None:
        pytest.skip("paper tables not recorded in tests/paper_targets.yaml")
    for name, expected in targets["published_values"].items():
        assert row(report, name) == pytest.approx(expected, abs=1.0)
