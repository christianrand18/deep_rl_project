"""
Minimal regression tests for the Deep RL AMoD project.
Tests that the core simulation loop runs without errors.

Run from project root:
    python -m pytest test_regression.py -v
"""

import os
import subprocess
import sys


def _run(script, **kwargs):
    cmd = [sys.executable, script]
    for k, v in kwargs.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, env=env
    )


def test_single_agent_mode3():
    """Single-agent baseline: no rebalancing, fixed price (no CPLEX needed)."""
    r = _run("main_a2c.py", mode=3, max_episodes=2, max_steps=5)
    assert r.returncode == 0, (
        f"STDERR:\n{r.stderr[-800:]}\nSTDOUT:\n{r.stdout[-500:]}"
    )


def test_multi_agent_mode3():
    """Multi-agent baseline: no rebalancing, fixed price (no CPLEX needed)."""
    r = _run("main_a2c_multi_agent.py", mode=3, max_episodes=2, max_steps=5)
    assert r.returncode == 0, (
        f"STDERR:\n{r.stderr[-800:]}\nSTDOUT:\n{r.stdout[-500:]}"
    )


def test_single_agent_mode4():
    """Single-agent baseline: uniform rebalancing, fixed price (needs CPLEX)."""
    r = _run("main_a2c.py", mode=4, max_episodes=2, max_steps=5)
    assert r.returncode == 0, (
        f"STDERR:\n{r.stderr[-800:]}\nSTDOUT:\n{r.stdout[-500:]}"
    )


def test_multi_agent_mode4():
    """Multi-agent baseline: uniform rebalancing, fixed price (needs CPLEX)."""
    r = _run("main_a2c_multi_agent.py", mode=4, max_episodes=2, max_steps=5)
    assert r.returncode == 0, (
        f"STDERR:\n{r.stderr[-800:]}\nSTDOUT:\n{r.stdout[-500:]}"
    )
