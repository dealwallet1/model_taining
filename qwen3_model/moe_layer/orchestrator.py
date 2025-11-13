# orchestrator.py
#
# Run unit tests + optional MoE demo

import argparse
import pathlib
import sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parent

def run_pytest(test_file: str):
    print(f"\n>>> Running tests: {test_file}")
    #pytest_args = ["-q", str(ROOT / test_file)]
    pytest_args = ["-v", "-s", str(ROOT / test_file)]
    return_code = pytest.main(pytest_args)
    if return_code != 0:
        sys.exit(return_code)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="run a tiny MoE demo")
    args = p.parse_args()

    # 1) Unit tests
    run_pytest("tests/test_gate_shapes.py")
    run_pytest("tests/test_moe_forward.py")
    run_pytest("tests/test_hybrid_block.py")

    # 2) Optional demo
    if args.demo:
        import subprocess, shlex
        cmd = "python demo_moe.py --tokens 6 --hidden 128 --experts 4 --top_k 1"
        print(f"\n>>> Running demo: {cmd}")
        res = subprocess.run(shlex.split(cmd), cwd=ROOT, shell=True)
        if res.returncode != 0:
            sys.exit(res.returncode)

    print("\nPart 5 checks complete. ✅")
