"""Assembles the inspect_ai Task: API-Bank dataset + stock generate() solver (single-turn, no
tool execution loop -- ToolSpec's own benchmark never calls execute_tools(), it only predicts the
tool call) + a scorer that compares against ToolSpec's own native-repo baseline output for the
same question_id, loaded from a reference JSONL (TOOLSPEC_REFERENCE_JSONL env var). This is the
same "compare against a known reference run" methodology used for tau2-bench and BFCL, and is
also exactly the correctness check ("is speculative decoding actually lossless here?") that
phase 1's native reproduction already found isn't quite 100%.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

from toolspec_adapter.dataset import api_bank_samples


@lru_cache(maxsize=4)
def _load_reference(path: str) -> dict[int, str]:
    ref: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ref[obj["question_id"]] = obj["choices"]["output"]
    return ref


@scorer(metrics=[accuracy(), stderr()])
def matches_reference_baseline() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        ref_path = os.environ.get("TOOLSPEC_REFERENCE_JSONL")
        question_id = state.metadata.get("question_id")
        completion = state.output.completion.strip()

        if not ref_path or not Path(ref_path).exists():
            return Score(value="N", explanation="TOOLSPEC_REFERENCE_JSONL not set/found -- no reference to compare against")

        reference = _load_reference(ref_path).get(question_id)
        if reference is None:
            return Score(value="N", explanation=f"question_id {question_id} not found in reference file")

        match = completion == reference.strip()
        return Score(
            value="C" if match else "I",
            answer=completion,
            explanation=f"reference: {reference!r}\ngot: {completion!r}",
            metadata={"question_id": question_id, "matches_reference": match},
        )

    return score


@task
def toolspec_apibank(limit: int | None = None) -> Task:
    samples: list[Sample] = api_bank_samples(limit=limit)
    return Task(
        dataset=samples,
        solver=generate(),
        scorer=matches_reference_baseline(),
    )
