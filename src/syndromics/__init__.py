"""Syndromics: read a quantum chip's health from its error-correction data."""
__version__ = "0.4.3"

from .diagnose import Finding, Report, diagnose, score_against_truth  # noqa: F401
from .model import NoiseFit, fit_noise  # noqa: F401
