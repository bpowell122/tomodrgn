"""
Module containing functions for conveniently running pytest-console-scripts
"""

from pytest_console_scripts import ScriptRunner, RunResult


def run_assert_no_error(script_runner: ScriptRunner,
                        command: str) -> RunResult:
    command_list = command.split(' ')
    result = script_runner.run(command_list)
    assert result.returncode == 0
    return result
