"""Smoke test for all example scripts in examples/ directory."""
import os
import sys
import pytest
import glob
import subprocess

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")

example_scripts = glob.glob(os.path.join(EXAMPLES_DIR, "*.py"))

@pytest.mark.parametrize("script_path", example_scripts)
def test_example_script_runs(script_path):
	"""Smoke test: run each example script and check for errors."""
	print(f"Running example: {script_path}")
	result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
	assert result.returncode == 0, f"Script {script_path} failed with error:\n{result.stderr}"


