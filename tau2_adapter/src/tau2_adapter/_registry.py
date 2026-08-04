"""Entry point target for the `inspect_ai` setuptools entry-point group.

Registers `tau2-agent-vllm`, a thin `OpenAICompatibleAPI` subclass that omits the `"strict"`
field inspect_ai's stock `openai-api` provider always adds to tool definitions
(`openai_compatible.py`'s `tools_to_openai()`). vLLM 0.6.3.post1's request schema rejects that
field as unrecognized -- see `docs/tau2_bench_integration_findings.md`'s Bug 3 for the full
story. This lets the tau2-bench adapter's agent use real native tool-calling instead of falling
back to `emulate_tools=true`, matching the mechanism tau2's own native CLI path already uses (its
tool construction never had this field), so a native-vs-adapter comparison isn't comparing two
different tool-calling mechanisms alongside whatever else differs.

Model string: `tau2-agent-vllm/<service>/<model>` -- same shape as the stock `openai-api`
provider (e.g. `tau2-agent-vllm/vllm/Qwen/Qwen2.5-3B-Instruct`), same `<SERVICE>_BASE_URL`/
`<SERVICE>_API_KEY` env var lookup, inherited unchanged from `OpenAICompatibleAPI`.
"""

from inspect_ai.model import ModelAPI, modelapi


@modelapi(name="tau2-agent-vllm")
def tau2_agent_vllm() -> type[ModelAPI]:
    from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
    from inspect_ai.tool import ToolInfo

    class NoStrictOpenAICompatibleAPI(OpenAICompatibleAPI):
        def tools_to_openai(self, tools: list[ToolInfo]) -> list[dict]:
            openai_tools = super().tools_to_openai(tools)
            for tool in openai_tools:
                tool["function"].pop("strict", None)
            return openai_tools

    return NoStrictOpenAICompatibleAPI
