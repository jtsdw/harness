"""Configure tau2's NL-assertion judge from the outer adapter layer.

The pinned tau2 release copies its default judge model and arguments into the
evaluator module at import time. Configure those runtime values explicitly here
so the adapter does not depend on edits to the provisioned tau2 source tree.
"""

from __future__ import annotations

import os

from tau2.evaluator import evaluator_nl_assertions

from tau2_adapter.runtime import json_object_from_env


def configure_tau2_nl_assertions() -> None:
    """Apply optional local-judge settings to tau2's evaluator module."""
    model = os.environ.get("TAU2_LLM_NL_ASSERTIONS", "").strip()
    if model:
        evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = model

    if os.environ.get("TAU2_LLM_NL_ASSERTIONS_ARGS", "").strip():
        evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = (
            json_object_from_env("TAU2_LLM_NL_ASSERTIONS_ARGS")
        )
