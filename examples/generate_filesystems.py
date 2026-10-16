# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Generate a few file system models with a throwaway Refinery server in Docker.

Run with: uv run examples/generate_filesystems.py
"""

from pathlib import Path

from pyfinery import RefineryServer, UnsatisfiableError

PROBLEM = Path(__file__).with_name("filesystem.problem")

with RefineryServer() as server, server.client() as refinery:
    print(f"Refinery is running at {server.url}")

    result = refinery.generate_many(
        PROBLEM,
        3,
        scopes=["File=2..*"],
        on_status=lambda status: print("  ...", getattr(status, "message", "model received")),
    )
    print(f"Stopped because: {result.stop_reason}")
    for i, model in enumerate(result.models, start=1):
        assert model.json_ is not None
        print(f"\nModel {i}:")
        for parent, child in model.json_.true_tuples("contents"):
            print(f"  {parent} contains {child}")

    try:
        refinery.generate(PROBLEM, scopes=["Dir=0"])
    except UnsatisfiableError as e:
        print(f"\nAs expected, a file system without directories is impossible: {e}")
