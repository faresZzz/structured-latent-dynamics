"""Factorial runner: expand a YAML grid (dataset x arm x seed) into Trainer runs."""

from sld.experiments.factorial import expand, main

__all__ = ["expand", "main"]
