"""目标二: offline analysis over already-recorded inspect_trace JSONL + real `.eval` logs.

Nothing here is a Hooks callback -- these are pure functions consuming what `hooks.py`'s
real-time capture already wrote to disk (see `../config.py` for the on-disk layout). See
`/home/liuyingen/code/doc/efficient-harness/goal2_design.md` for the design rationale.
"""
