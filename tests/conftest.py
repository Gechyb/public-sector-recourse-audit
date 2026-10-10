"""Shared fixtures: load the data and train the model once per test session."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_config, prepare  # noqa: E402
from src.model import split, train  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def data(cfg):
    X, y, groups = prepare(cfg)
    return X, y, groups


@pytest.fixture(scope="session")
def splits(cfg, data):
    """(X_train, X_test, y_train, y_test, g_train, g_test)"""
    return split(*data, cfg)


@pytest.fixture(scope="session", params=["logistic", "gbm"])
def model(request, cfg, splits):
    X_train, _, y_train, *_ = splits
    return train(request.param, X_train, y_train, seed=cfg["data"]["seed"])
