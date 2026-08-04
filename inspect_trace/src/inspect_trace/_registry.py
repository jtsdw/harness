"""Entry point target for the `inspect_ai` setuptools entry-point group.

Importing this module (which `inspect_ai`'s extension loader does automatically once this
package is installed) registers `TraceHooks` via the `@hooks` decorator.
"""

from inspect_ai.hooks import Hooks, hooks


@hooks(
    name="inspect_trace",
    description=(
        "Records goal-1 execution-trace derived facts (repeated-prefill detection, "
        "segment-level token estimates, retry attempt grouping) alongside the eval log."
    ),
)
def trace_hooks() -> type[Hooks]:
    from .hooks import TraceHooks

    return TraceHooks
