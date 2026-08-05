"""Custom inspect_ai ModelAPI provider that drives ToolSpec's own forward functions in-process.

ToolSpec (/home/liuyingen/code/ToolSpec) is not an OpenAI-compatible HTTP service like
tau2-bench was -- it is a raw HuggingFace `transformers` generation loop with patched modeling
code (custom KV-cache handling for tree-based speculative verification). There is no wire
protocol to plug into, so this provider loads the (patched) model once and calls ToolSpec's own
`baseline_forward()`/`toolspec_forward()` functions directly, instead of making HTTP requests.

Because this provider IS the thing inspect_ai calls to generate (we are not calling inspect_ai's
Model from inside some foreign framework's own async loop, unlike the tau2-bench adapter), Hooks
fire without needing any anyio thread-bridging trick -- this is architecturally simpler than the
tau2-bench integration in that specific respect.

ToolSpec's repo has no packaging (no pyproject.toml/setup.py, just `model/` and `evaluation/`
importable only when its repo root is on sys.path, matching how `eval.sh` runs
`python -m evaluation.inference_toolspec` from the repo root). We inject that path at import time
via TOOLSPEC_REPO_DIR rather than restructuring someone else's repo -- documented wrinkle, not a
bug. `evaluation/inference_toolspec.py` also imports fastchat/shortuuid/tqdm/numpy at module
level even though we only use the one `toolspec_forward` function from it, so those are declared
as real dependencies in pyproject.toml rather than reimplemented.

Model string: `toolspec-hf/<model-name>` (no "service" concept -- there's no HTTP base_url).
Mode is selected via `-M mode=baseline|toolspec` (default: toolspec).

ToolSpec's retrieval-augmented drafting is stateful and order-dependent: `output_memory` grows
in dataset order across the whole eval run (see doc/project_explanation.md and
evaluation/eval_toolspec.py's `get_model_answers()`). This provider replicates that by keeping
one `output_memory` list per provider instance, appended to after each `generate()` call --
correct only when the eval runs with `--max-connections 1` (this project's standing convention
for local GPU work) so sample order is not interleaved by concurrent generation.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

TOOLSPEC_REPO_DIR = os.environ.get("TOOLSPEC_REPO_DIR", "/home/liuyingen/code/ToolSpec")
if TOOLSPEC_REPO_DIR not in sys.path:
    sys.path.insert(0, TOOLSPEC_REPO_DIR)


def _stub_fastchat() -> None:
    """`evaluation/inference_{baseline,toolspec}.py` do `from fastchat.utils import
    str_to_torch_dtype` at module level, but we never call that function (we do our own dtype
    mapping) and never call anything else from those modules that needs fastchat. The real
    fschat==0.2.31 package pins pydantic<2, which conflicts with inspect_ai's pydantic>=2 --
    so rather than carry that real, avoidable dependency conflict, register a stub module in
    sys.modules that satisfies the import statement without installing the real package.
    """
    import types

    if "fastchat" in sys.modules and "fastchat.utils" in sys.modules:
        return
    fastchat_module = types.ModuleType("fastchat")
    fastchat_utils_module = types.ModuleType("fastchat.utils")
    fastchat_utils_module.str_to_torch_dtype = lambda dtype_str: _DTYPE_MAP[dtype_str]  # noqa: E731
    fastchat_module.utils = fastchat_utils_module
    sys.modules["fastchat"] = fastchat_module
    sys.modules["fastchat.utils"] = fastchat_utils_module


import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from inspect_ai.model import ChatMessage, GenerateConfig, ModelAPI, ModelCall, ModelOutput  # noqa: E402
from inspect_ai.model._model_output import ModelUsage  # noqa: E402
from inspect_ai.tool import ToolChoice, ToolInfo  # noqa: E402

_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_stub_fastchat()

# Keyed by (model_path, dtype, mode) -> (model, tokenizer, adj_matrix | None). Loading a 3B model
# takes real seconds; inspect_ai may construct more than one ModelAPI instance pointing at the
# same underlying model (e.g. one per role), so this cache avoids reloading weights redundantly.
_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, Any, Any]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_model_and_tokenizer(model_path: str, dtype: str, mode: str, output_id_topk: int) -> tuple[Any, Any, Any]:
    key = (model_path, dtype, mode)
    with _MODEL_CACHE_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        torch_dtype = _DTYPE_MAP[dtype]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        adj_matrix = None

        if mode == "baseline":
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map="auto"
            )
        elif mode == "toolspec":
            from model.toolspec.modeling_llama_kv import LlamaForCausalLM
            from model.toolspec.modeling_qwen_kv import Qwen2ForCausalLM

            model_config = AutoConfig.from_pretrained(model_path)
            if model_config.model_type == "llama":
                model_cls = LlamaForCausalLM
            elif model_config.model_type == "qwen2":
                model_cls = Qwen2ForCausalLM
            else:
                raise ValueError(f"Unsupported model_type for ToolSpec: {model_config.model_type}")
            model = model_cls.from_pretrained(
                model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map="auto"
            )
            adj_matrix = torch.zeros(
                (model.vocab_size, output_id_topk), dtype=torch.long, device=model.device, requires_grad=False
            )
        else:
            raise ValueError(f"Unsupported mode for toolspec-hf provider: {mode!r} (expected baseline|toolspec)")

        _MODEL_CACHE[key] = (model, tokenizer, adj_matrix)
        return _MODEL_CACHE[key]


def _chat_message_text(message: ChatMessage) -> str:
    return message.text


class ToolSpecHFModelAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, [], config)
        self.mode = str(model_args.pop("mode", "toolspec"))
        self.dtype = str(model_args.pop("dtype", "float16"))
        self.max_new_tokens = int(model_args.pop("max_new_tokens", 1024))
        self.output_id_topk = int(model_args.pop("output_id_topk", 8))

        self.model, self.tokenizer, self.adj_matrix = _load_model_and_tokenizer(
            model_name, self.dtype, self.mode, self.output_id_topk
        )
        self.model.eval()

        # Retrieval memory: persists across generate() calls for this provider instance's
        # lifetime, appended to in call order -- mirrors ToolSpec's own eval loop statewise.
        self._output_memory: list[dict[str, Any]] = []
        self._call_index = 0

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput, ModelCall]:
        import anyio

        system_text = ""
        other_messages = []
        for m in input:
            if m.role == "system" and not system_text:
                system_text = _chat_message_text(m)
            else:
                other_messages.append({"role": m.role, "content": _chat_message_text(m)})

        messages = ([{"role": "system", "content": system_text}] if system_text else []) + other_messages
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids

        request = {"model": self.model_name, "mode": self.mode, "messages": messages}

        start_time = time.time()
        (
            output_text,
            new_token,
            step,
            accept_length_list,
        ) = await anyio.to_thread.run_sync(self._forward, inputs, system_text)
        wall_time = time.time() - start_time
        self._call_index += 1

        output = ModelOutput.from_content(model=self.model_name, content=output_text)
        output.usage = ModelUsage(
            input_tokens=input_ids.size(1),
            output_tokens=new_token,
            total_tokens=input_ids.size(1) + new_token,
        )

        response = {
            "output": output_text,
            "new_tokens": new_token,
            "decoding_steps": step,
            "wall_time": wall_time,
            "accept_lengths": accept_length_list,
            "mode": self.mode,
        }
        model_call = ModelCall.create(request, response, time=wall_time)
        return output, model_call

    def _forward(self, inputs: Any, system_text: str) -> tuple[str, int, int, list[int]]:
        """Runs synchronously in a worker thread (blocking CUDA calls) -- see generate()."""
        input_ids = inputs.input_ids
        start_len = input_ids.size(1)

        if self.mode == "baseline":
            from evaluation.inference_baseline import baseline_forward

            output_ids, new_token, step, accept_length_list = baseline_forward(
                inputs, self.model, self.tokenizer, self.max_new_tokens, temperature=0.0, do_sample=False
            )
        elif self.mode == "toolspec":
            from model.toolspec.schema_fsm import SchemaFSM

            from evaluation.inference_toolspec import toolspec_forward

            schema_fsm = SchemaFSM(system_text, self.tokenizer)
            torch.manual_seed(self._call_index)
            (
                output_ids,
                new_token,
                step,
                accept_length_list,
                question_hidden_state,
            ) = toolspec_forward(
                inputs,
                self._output_memory,
                schema_fsm,
                self.model,
                self.tokenizer,
                self.max_new_tokens,
                output_id_topk=self.output_id_topk,
                adj_matrix=self.adj_matrix,
            )
            gen_ids = output_ids[0][start_len:]
            self._output_memory.append(
                {
                    "question_id": self._call_index,
                    "question_hidden_state": question_hidden_state,
                    "output_ids": gen_ids.tolist(),
                }
            )
        else:
            raise ValueError(f"Unsupported mode: {self.mode!r}")

        gen_ids = output_ids[0][start_len:]
        output_text = self.tokenizer.decode(gen_ids, spaces_between_special_tokens=False)
        for special_token in self.tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for tok in special_token:
                    output_text = output_text.replace(tok, "")
            else:
                output_text = output_text.replace(special_token, "")
        output_text = output_text.strip()

        return output_text, new_token, step, accept_length_list
