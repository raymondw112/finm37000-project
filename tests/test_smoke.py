"""Import smoke test: a fresh clone + documented setup can import everything (Issue #3)."""
import importlib

import pytest

import gcsi


@pytest.mark.parametrize("module", gcsi.__all__)
def test_gcsi_modules_import(module):
    importlib.import_module(f"gcsi.{module}")


@pytest.mark.parametrize("module", ["finm37000.continuous", "finm37000.db_env_util"])
def test_course_package_imports(module):
    importlib.import_module(module)
