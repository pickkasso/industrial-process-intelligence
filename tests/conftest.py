"""Shared pytest fixtures."""

import random

import pytest


@pytest.fixture
def random_seed() -> int:
    """Provide a fixed seed and initialize Python's RNG for reproducible tests."""
    seed = 42
    random.seed(seed)
    return seed
