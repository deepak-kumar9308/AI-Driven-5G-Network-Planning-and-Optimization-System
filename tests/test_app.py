"""Smoke test: every dashboard page renders without errors (uses Streamlit's AppTest)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")
PAGES = ["Home / Network Overview", "RF Planning Analysis", "Beamforming Simulation", "Handover Prediction",
         "Network Slicing", "Energy Optimization", "ML Model Performance", "Before vs After Comparison",
         "About / Methodology"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
