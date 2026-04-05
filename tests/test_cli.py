"""Tests for the agent-observatory CLI entrypoint."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_main_importable():
    """from agent_diagnostics.cli import main succeeds."""
    from agent_diagnostics.cli import main

    assert callable(main)


def test_help_output_includes_all_subcommands():
    """python -m agent_diagnostics --help lists all 8 subcommands."""
    result = subprocess.run(
        [sys.executable, "-m", "agent_diagnostics", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    help_text = result.stdout

    expected_subcommands = [
        "extract",
        "annotate",
        "report",
        "llm-annotate",
        "train",
        "predict",
        "ensemble",
        "validate",
    ]
    for sub in expected_subcommands:
        assert sub in help_text, f"Subcommand '{sub}' not found in --help output"


def test_no_judge_flag_in_help():
    """--judge flag must not appear in llm-annotate help."""
    result = subprocess.run(
        [sys.executable, "-m", "agent_diagnostics", "llm-annotate", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert (
        "--judge" not in result.stdout
    ), "--judge flag should not exist on llm-annotate"


def test_cli_imports_use_agent_diagnostics():
    """All imports in cli.py use agent_diagnostics, not observatory."""
    cli_path = (
        Path(__file__).resolve().parent.parent / "src" / "agent_diagnostics" / "cli.py"
    )
    content = cli_path.read_text()

    # Should not have bare 'from observatory.' imports
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("from observatory.") or stripped.startswith(
            "import observatory."
        ):
            pytest.fail(f"Found bare observatory import: {stripped}")

    # Should have agent_diagnostics imports
    assert "from agent_diagnostics." in content


def test_dunder_main_imports_from_agent_diagnostics():
    """__main__.py imports from agent_diagnostics.cli."""
    main_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "agent_diagnostics"
        / "__main__.py"
    )
    content = main_path.read_text()
    assert "from agent_diagnostics.cli import main" in content


def test_pyproject_has_scripts_entry():
    """pyproject.toml has [project.scripts] observatory entry."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()
    assert "[project.scripts]" in content
    assert 'observatory = "agent_diagnostics.cli:main"' in content
