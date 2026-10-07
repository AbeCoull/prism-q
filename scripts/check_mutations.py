import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile


@dataclass(frozen=True)
class Mutation:
    name: str
    path: str
    before: str
    after: str
    binary: str
    test: str


MUTATIONS = (
    Mutation(
        "phase_sign",
        "src/circuit/fusion_phase.rs",
        "entry.1 *= phase;",
        "entry.1 *= phase.conj();",
        "fusion_correctness",
        "fusion_batch_phase_duplicate_pair",
    ),
    Mutation(
        "target_order",
        "src/circuit/fusion.rs",
        "        swap_order_4x4(mat)\n",
        "        *mat\n",
        "fusion_correctness",
        "fusion_same_pair_reversed_targets_20q",
    ),
    Mutation(
        "parameter_replay",
        "src/circuit/plan.rs",
        "d.set_theta(theta);",
        "d.set_theta(d.theta());",
        "parameter_binding",
        "a_bound_pauli_rotation_reaches_the_fused_stream",
    ),
)


def test_command(cases):
    command = ["cargo", "nextest", "run", "--features", "parallel"]
    for binary in sorted({case.binary for case in cases}):
        command.extend(["--test", binary])
    filters = " | ".join(
        f"(binary(={case.binary}) & test(={case.test}))" for case in cases
    )
    return command + ["-E", filters, "--no-tests", "fail", "--status-level", "fail"]


def run_command(command, workspace, env, log, timeout):
    with log.open("wb") as output:
        with subprocess.Popen(
            command,
            cwd=workspace,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        ) as process:
            try:
                return process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return None


def mutation_status(exit_code):
    if exit_code == 100:
        return "caught"
    if exit_code == 0:
        return "survived"
    return "timeout" if exit_code is None else "error"


def check_mutation(case, workspace, env, output, timeout):
    path = workspace / case.path
    original = path.read_bytes()
    source = original.decode("utf-8").replace("\r\n", "\n")
    if source.count(case.before) != 1:
        return {"name": case.name, "status": "source_mismatch"}
    try:
        path.write_bytes(source.replace(case.before, case.after).encode("utf-8"))
        exit_code = run_command(
            test_command([case]), workspace, env, output / f"{case.name}.log", timeout
        )
        return {
            "name": case.name,
            "status": mutation_status(exit_code),
            "exit_code": exit_code,
            "test": f"{case.binary}::{case.test}",
        }
    finally:
        path.write_bytes(original)


def run_mutations(root, output, timeout):
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
    ).decode("utf-8").split("\0")
    with tempfile.TemporaryDirectory(prefix="source-", dir=output) as scratch:
        workspace = Path(scratch).resolve()
        assert workspace.is_relative_to(output.resolve())
        for name in set(files) - {""}:
            source = root / name
            if source.is_file():
                destination = workspace / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        env = dict(os.environ, CARGO_TARGET_DIR=str(output / "build"))
        print("Baseline: build and run the three selected tests.", flush=True)
        baseline = run_command(
            test_command(MUTATIONS), workspace, env, output / "baseline.log", timeout
        )
        report = {"baseline_exit_code": baseline, "mutations": []}
        if baseline != 0:
            report["status"] = "baseline_failed"
            return report
        for case in MUTATIONS:
            print(f"{case.name}: build and test the mutated source.", flush=True)
            result = check_mutation(case, workspace, env, output, timeout)
            report["mutations"].append(result)
            print(f"{case.name}: {result['status']}", flush=True)
        report["status"] = (
            "pass" if all(case["status"] == "caught" for case in report["mutations"]) else "fail"
        )
        return report


def main():
    parser = argparse.ArgumentParser(description="Check three seeded faults in an isolated source copy.")
    parser.add_argument("--output", type=Path, default=Path("target/mutation-check"))
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per build and test invocation.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = run_mutations(root, output, args.timeout)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Mutation check: {report['status']}. Logs: {output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
