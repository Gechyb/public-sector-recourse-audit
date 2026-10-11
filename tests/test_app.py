"""Headless checks that the Streamlit app runs for every dataset and model."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")
DATASETS = ["Adult 1994 (UCI benchmark)", "ACS North Carolina 2023, $103K bar",
            "ACS North Carolina 2023, $50K bar"]


def _run(dataset=None, model=None):
    at = AppTest.from_file(APP, default_timeout=300).run()
    if dataset:
        at.sidebar.selectbox[0].select(dataset).run()
    if model:
        at.sidebar.selectbox[1].select(model).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("model", ["gbm", "logistic"])
def test_app_runs(dataset, model):
    at = _run(dataset, model)
    labels = [m.label for m in at.metric]
    assert "Denied applicants with realistic recourse" in labels  # findings tab
    assert "Approval probability" in labels                       # applicant tab
    assert any("vs" in md.value for md in at.markdown)            # generated findings


def test_both_recourse_panels_render():
    at = _run()
    headers = [h.value for h in at.subheader]
    assert {"Realistic recourse", "Off-the-shelf recourse"} <= set(headers)


def test_group_filter_and_budget_slider():
    at = _run()
    at.selectbox[0].select("age_group").run()
    at.selectbox[1].select("<30").run()
    at.sidebar.slider[0].set_value(12.0).run()
    assert not at.exception
    ages = [int(o.split("age ")[1].split(" ")[0]) for o in at.selectbox[2].options]
    assert ages and max(ages) < 30  # filter keeps only under-30 applicants
