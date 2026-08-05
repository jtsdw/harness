"""Entry point target for the `inspect_ai` setuptools entry-point group.

Registers `toolspec-hf`, a from-scratch `ModelAPI` (not an HTTP provider subclass -- see
provider.py's module docstring for why) that drives ToolSpec's own baseline/toolspec forward
functions in-process. Model string: `toolspec-hf/<model-name>`, e.g.
`toolspec-hf/Qwen/Qwen2.5-3B-Instruct`. Mode selected via `-M mode=baseline|toolspec`.
"""

from inspect_ai.model import ModelAPI, modelapi


@modelapi(name="toolspec-hf")
def toolspec_hf() -> type[ModelAPI]:
    from .provider import ToolSpecHFModelAPI

    return ToolSpecHFModelAPI
