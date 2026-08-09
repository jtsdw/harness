#!/usr/bin/env python3
"""Validate the newest Inspect log in one tau2 run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
from zipfile import ZipFile

import zstandard


_LOCAL_FILE_HEADER = struct.Struct("<4s5H3I2H")
_ZSTANDARD_METHODS = {20, 93}


def _read_member(archive: Path, zip_file: ZipFile, name: str) -> bytes:
    """Read a ZIP member, including method-93 Zstandard on Python 3.12."""
    info = zip_file.getinfo(name)
    if info.compress_type not in _ZSTANDARD_METHODS:
        return zip_file.read(info)

    with archive.open("rb") as source:
        source.seek(info.header_offset)
        fields = _LOCAL_FILE_HEADER.unpack(source.read(_LOCAL_FILE_HEADER.size))
        if fields[0] != b"PK\x03\x04":
            raise ValueError(f"invalid local ZIP header for {name}")
        name_length, extra_length = fields[-2:]
        source.seek(name_length + extra_length, 1)
        compressed = source.read(info.compress_size)
    return zstandard.ZstdDecompressor().decompress(
        compressed, max_output_size=info.file_size
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("expected_samples", type=int)
    args = parser.parse_args()

    paths = list((args.run_dir / "logs").glob("*.eval"))
    if not paths:
        parser.error(f"no .eval log found under {args.run_dir / 'logs'}")

    path = max(paths, key=lambda item: item.stat().st_mtime)
    with ZipFile(path) as zip_file:
        names = zip_file.namelist()
        header = (
            json.loads(_read_member(path, zip_file, "header.json"))
            if "header.json" in names
            else {}
        )
        sample_names = [
            name
            for name in names
            if name.startswith("samples/") and name.endswith(".json")
        ]
        samples = [
            json.loads(_read_member(path, zip_file, name)) for name in sample_names
        ]

    errors = [sample for sample in samples if sample.get("error") is not None]
    status = header.get("status", "incomplete")
    reported_total = (header.get("results") or {}).get("total_samples")

    print(
        f"validation: {path} status={status} "
        f"samples={len(samples)} errors={len(errors)}"
    )
    if str(status).lower() != "success":
        raise SystemExit(f"eval status is {status!r}, expected 'success'")
    if len(samples) != args.expected_samples:
        raise SystemExit(
            f"sample count is {len(samples)}, expected {args.expected_samples}"
        )
    if errors:
        raise SystemExit(f"{len(errors)} sample(s) contain errors")
    if reported_total is not None and reported_total != len(samples):
        raise SystemExit(
            f"header reports {reported_total} samples, archive contains {len(samples)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
