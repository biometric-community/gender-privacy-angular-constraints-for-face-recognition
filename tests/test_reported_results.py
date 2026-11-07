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


def test_pic_follows_eqs_6_and_7(report):
    """PG and IL are changes relative to the original embeddings, not plain differences."""
    original, private = row(report, "Original"), row(report, "Private (Lp1+Lp2)")
    match = re.search(r"\(PG=(-?[\d.]+), IL=(-?[\d.]+)\)", report)
    assert match
    privacy_gain = ((100 - private[2]) - (100 - original[2])) / (100 - original[2])
    identity_loss = (private[1] - original[1]) / original[1]
    assert float(match.group(1)) == pytest.approx(privacy_gain, abs=0.01)
    assert float(match.group(2)) == pytest.approx(identity_loss, abs=0.01)


def test_gender_predictability_reaches_the_published_level(report, targets):
    expected = targets["elasticface_lfw"]["accg"]
    assert row(report, "Private (Lp1+Lp2)")[2] == pytest.approx(expected, abs=targets["tolerance_pp"])


def test_verification_error_reaches_the_published_level(report, targets):
    expected = targets["elasticface_lfw"]["eer"]
    assert row(report, "Private (Lp1+Lp2)")[1] == pytest.approx(expected, abs=targets["tolerance_pp"])
