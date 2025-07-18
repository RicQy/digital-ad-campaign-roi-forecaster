#!/usr/bin/env python3
"""
Script to simulate CI environment locally
"""
import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print("=" * 60)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print("STDOUT:")
        print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    if result.returncode != 0:
        print(f"❌ FAILED with return code {result.returncode}")
        return False
    else:
        print("✅ PASSED")
        return True


def main():
    """Main CI simulation"""
    print("Starting CI simulation...")

    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    commands = [
        ("uv sync --dev", "Install dependencies"),
        ("uv run python --version", "Check Python version"),
        ("uv run black --version", "Check Black version"),
        ("uv run flake8 --version", "Check flake8 version"),
        ("uv run pytest --version", "Check pytest version"),
        ("uv run black --check --diff --color .", "Black formatting check"),
        ("uv run flake8 . --show-source --statistics", "Flake8 linting"),
        ("uv run pytest tests/ --maxfail=5 --disable-warnings -v", "Run tests"),
    ]

    all_passed = True
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            all_passed = False
            break

    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 All CI checks PASSED!")
    else:
        print("❌ Some CI checks FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
