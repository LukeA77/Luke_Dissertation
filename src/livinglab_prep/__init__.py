"""LivingLab CodeBrain-ready EEG preprocessing pipeline.

Implements README.md Stages 0-9 (config -> preflight -> load/filter/uV/PSD ->
alignment -> task-locked windowing -> dual labels -> rejection -> serialization)
plus reports and the window/manifest-level validation checks. Stages 10-11
(LOPO folds + Dataset loader) are deferred to the 30-patient phase.
"""

__all__ = ["config"]
