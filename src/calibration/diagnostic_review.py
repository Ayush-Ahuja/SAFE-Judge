"""Compatibility imports for experiments saved before the SAFE-Judge rename."""

from src.calibration.safe_judge import (
    SafeJudgeConfig as ReviewConfig,
    SafeJudgeModel as DiagnosticModel,
    run_safe_judge as run_diagnostic_review,
)
