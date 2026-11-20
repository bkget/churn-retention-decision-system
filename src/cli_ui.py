"""Shared terminal UI for the pipeline CLIs (train / explain / roi), built on ``rich``.

Animations (spinners, progress bars) run only on an interactive TTY and when
``--quiet`` is not set. Redirected output and CI get plain, static lines;
``--quiet`` suppresses everything except warnings and errors.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Console resolves sys.stdout lazily, so pytest's capture and shell redirects are honoured.
console = Console(highlight=False)
_state = {"quiet": False}
_T0 = time.perf_counter()


def set_quiet(quiet: bool) -> None:
    _state["quiet"] = bool(quiet)


def is_quiet() -> bool:
    return _state["quiet"]


def interactive() -> bool:
    """True when live animations are appropriate (TTY and not --quiet)."""
    return not is_quiet() and console.is_terminal


def base_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="no banners, progress or tables (errors only)"
    )
    return parser


def setup(quiet: bool = False) -> None:
    """Configure quiet mode and route ``logging`` through the shared console."""
    set_quiet(quiet)
    if not console.is_terminal and "COLUMNS" not in os.environ:
        console.width = 140  # redirected logs: avoid truncating wide tables at the 80-col default
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
        force=True,
    )


def _print(*args: Any, **kwargs: Any) -> None:
    if not is_quiet():
        console.print(*args, **kwargs)


def banner(title: str, subtitle: str = "") -> None:
    _print()
    _print(f"[bold cyan]{title}[/]" + (f"  [dim]{subtitle}[/]" if subtitle else ""), justify="left")
    _print("[dim]" + "─" * min(console.width, 72) + "[/]")


@contextmanager
def spinner(label: str) -> Iterator[None]:
    """Spinner with elapsed time on a TTY; a single static line otherwise."""
    t0 = time.perf_counter()
    if interactive():
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        with progress:
            progress.add_task(label, total=None)
            yield
    else:
        yield
    _print(f"[green]✔[/] {label} [dim]({time.perf_counter() - t0:.1f}s)[/]")


@contextmanager
def trial_progress(n_trials: int, label: str = "Optuna tuning") -> Iterator[Callable]:
    """Yield an Optuna callback that advances a bar and shows the best CV PR-AUC so far."""
    best = {"value": float("nan")}
    bar = None
    task = None
    t0 = time.perf_counter()

    def callback(study: Any, trial: Any) -> None:
        best["value"] = float(study.best_value)
        done = trial.number + 1
        if bar is not None:
            bar.update(task, completed=done, best=f"{best['value']:.4f}")
        elif not is_quiet() and (done % 10 == 0 or done == n_trials):
            console.print(f"  trial {done}/{n_trials}  best CV PR-AUC {best['value']:.4f}")

    if interactive():
        bar = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("best CV PR-AUC [cyan]{task.fields[best]}[/]"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        )
        task = bar.add_task(label, total=n_trials, best="--")
        with bar:
            yield callback
    else:
        _print(f"{label}: {n_trials} trials")
        yield callback
    _print(
        f"[green]✔[/] {label}: best CV PR-AUC {best['value']:.4f} "
        f"[dim]({time.perf_counter() - t0:.1f}s)[/]"
    )


def money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def df_table(
    df: pd.DataFrame,
    title: str,
    formats: dict[str, Callable[[Any], str] | str] | None = None,
    caption: str | None = None,
) -> None:
    """Print a DataFrame as a rich table. Floats default to 3 dp; numbers right-align."""
    if is_quiet():
        return
    formats = formats or {}
    table = Table(title=title, caption=caption, title_justify="left", header_style="bold")
    numeric = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    for col in df.columns:
        table.add_column(str(col), justify="right" if col in numeric else "left", overflow="fold")
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            v, fmt = row[col], formats.get(col)
            if pd.isna(v):
                cells.append("-")
            elif callable(fmt):
                cells.append(fmt(v))
            elif fmt:
                cells.append(format(v, fmt))
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        table.add_row(*cells)
    console.print(table)


def _display_path(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)


def summary_panel(title: str, artifacts: Sequence[str | Path]) -> None:
    """Closing panel: total runtime of this process plus the artifacts it wrote."""
    elapsed = time.perf_counter() - _T0
    minutes, seconds = divmod(elapsed, 60)
    runtime = f"{int(minutes)}m {seconds:04.1f}s" if minutes else f"{seconds:.1f}s"
    lines = [f"[bold]Total runtime:[/] {runtime}", "", "[bold]Artifacts[/]"]
    lines += [f"  {_display_path(a)}" for a in artifacts]
    _print()
    _print(Panel("\n".join(lines), title=f"[green]{title}[/]", title_align="left", expand=False))
