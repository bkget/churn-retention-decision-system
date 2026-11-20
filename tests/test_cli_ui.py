"""cli_ui must stay clean under pytest / redirected output and honour --quiet."""

from __future__ import annotations

import types

import pandas as pd
import pytest

from src import cli_ui


@pytest.fixture(autouse=True)
def _reset_quiet():
    cli_ui.set_quiet(False)
    yield
    cli_ui.set_quiet(False)


def test_not_interactive_when_stdout_is_not_a_tty(capsys):
    assert cli_ui.interactive() is False
    with cli_ui.spinner("Working"):
        pass
    out = capsys.readouterr().out
    assert "Working" in out and "\x1b" not in out  # one static line, no ANSI/animation


def test_quiet_suppresses_everything(capsys):
    cli_ui.set_quiet(True)
    cli_ui.banner("Title")
    with cli_ui.spinner("Working"):
        pass
    cli_ui.df_table(pd.DataFrame({"a": [1.0]}), "T")
    cli_ui.summary_panel("Done", ["x.csv"])
    assert capsys.readouterr().out == ""


def test_df_table_formats_values(capsys):
    df = pd.DataFrame({"name": ["a", "b"], "net": [1234.5, -50.0], "p": [0.12345, None]})
    cli_ui.df_table(df, "Demo", {"net": cli_ui.money})
    out = capsys.readouterr().out
    assert "$1,234" in out or "$1,235" in out
    assert "-$50" in out and "0.123" in out


def test_trial_progress_callback_reports_best_without_animation(capsys):
    study = types.SimpleNamespace(best_value=0.8123)
    with cli_ui.trial_progress(10) as cb:
        for n in range(10):
            cb(study, types.SimpleNamespace(number=n))
    out = capsys.readouterr().out
    assert "trial 10/10" in out and "0.8123" in out


def test_summary_panel_lists_runtime_and_artifacts(capsys):
    cli_ui.summary_panel("Done", ["artifacts/metrics.json"])
    out = capsys.readouterr().out
    assert "Total runtime" in out and "artifacts/metrics.json" in out
