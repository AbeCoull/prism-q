"""Benchmark matrix: run a profile, aggregate its raw observations, draw its figures.

    python comparison/matrix.py list --profile reference
    python comparison/matrix.py run --profile smoke --out comparison/results/smoke
    python comparison/matrix.py aggregate comparison/results/smoke
    python comparison/matrix.py figures comparison/results/smoke
    python comparison/matrix.py report comparison/results/reference-i7-6700k --docs docs/comparison

`run` writes `observations.jsonl` (one record per simulator, workload and
thread count, failures included), `manifest.json` (host, toolchain, versions,
profile, code hashes) and `circuits.jsonl.gz` (every gate list with its SHA-256). A rerun
with `--append --replace` moves the records it re-measures to
`superseded.jsonl` instead of deleting them. The later commands read only those
files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import profiles, provenance, runners, stats, validate, workloads  # noqa: E402

SCHEMA = "prism-q-matrix/1"
ORDER_SEED = 42
PROBS_REFERENCES = ("aer-statevector", "quest")
COUNTS_FALLBACKS = ("aer-statevector", "aer-stabilizer", "aer-mps", "aer-automatic", "stim", "prismq")


def build_runner() -> None:
    print("building compare_runner ...", flush=True)
    proc = subprocess.run(
        ["cargo", "build", "--release", "--features", "parallel", "--example", "compare_runner"],
        cwd=runners.REPO_ROOT,
        check=False,
    )
    if proc.returncode != 0:
        sys.exit("cargo build failed")


def load_profile(args: argparse.Namespace) -> profiles.Profile:
    factory = profiles.PROFILES[args.profile]
    profile = factory(runners.RUNNER, args.threads)
    if args.groups:
        keep = set(args.groups.split(","))
        profile.cells = [c for c in profile.cells if c.group in keep]
    if args.match:
        profile.cells = [c for c in profile.cells if args.match in c.workload.workload_id]
    if getattr(args, "simulators", None):
        keep = set(args.simulators.split(","))
        for cell in profile.cells:
            cell.simulators = [s for s in cell.simulators if s in keep]
        profile.cells = [c for c in profile.cells if c.simulators]
    return profile


def cmd_list(args: argparse.Namespace) -> int:
    profile = load_profile(args)
    runs = skips = 0
    for cell in profile.cells:
        print(f"[{cell.group}] {cell.workload.workload_id}  ({cell.workload.num_operations} ops)")
        for name in cell.simulators:
            reason = runners.applicable(runners.resolve(name), cell.workload)
            for threads in cell.threads:
                if reason:
                    skips += 1
                else:
                    runs += 1
            mark = f"skip: {reason}" if reason else f"threads {cell.threads}"
            print(f"    {name:28s} {mark}")
    print(f"\n{len(profile.cells)} workloads, {runs} measurements, {skips} recorded as not applicable")
    return 0


def _sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hashes() -> dict[str, str | None]:
    """SHA-256 of every binary and script a measurement runs through."""
    here = os.path.dirname(os.path.abspath(__file__))
    paths = {"compare_runner": runners.RUNNER, "peers": runners.PEERS, "quest_runner": runners.QUEST}
    paths["matrix.py"] = os.path.abspath(__file__)
    for path in sorted(glob.glob(os.path.join(here, "harness", "*.py"))):
        paths[f"harness/{os.path.basename(path)}"] = path
    return {name: _sha256(path) for name, path in paths.items()}


def _family(name: str) -> str:
    return name.partition(":")[0].partition("-")[0]


def _manifest(args: argparse.Namespace, profile: profiles.Profile, run_id: str, versions: dict) -> dict:
    prov = provenance.collect("parallel", args.threads, versions)
    prov["host"].update(provenance.host_details())
    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "profile": profile.name,
        "profile_description": profile.description,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "order_seed": ORDER_SEED,
        "memory_method": runners.MEMORY_METHOD,
        "timed_region": {
            "probs": "execute the circuit and materialize the full 2^n probability vector",
            "counts": (
                "execute the circuit and return the simulator's own shot result (PRISM-Q's count map, "
                "Aer's Result, qsim's measurement array)"
            ),
            "excluded": "process start, imports, building the circuit object from the gate list, "
            "conversion to the harness file format",
        },
        "repetition_policy": (
            "one cold call, then up to `iterations` warm calls in the same process; stop early once the "
            "warm calls have used `budget_s`, never below `min_iterations`"
        ),
        "validation": {
            "probs": "total variation distance against the fp64 reference simulator's vector",
            "counts": validate.__doc__.strip().split("\n\n")[0].replace("\n", " "),
            "family_alpha": validate.FAMILY_ALPHA,
        },
        "provenance": prov,
        "code_sha256": _code_hashes(),
        "machine_warnings": provenance.quiet_machine_warnings(),
        "simulators": {
            name: {
                "label": s.label,
                "kind": s.kind,
                "precision": s.precision,
                "representation": s.representation,
                "modes": sorted(s.modes),
                "version": versions.get(name),
            }
            for name, s in runners.REGISTRY.items()
        },
    }


CIRCUITS = "circuits.jsonl.gz"


def load_circuits(results_dir: str) -> dict[str, str]:
    """Gate list text by SHA-256 from a results directory's circuit store."""
    path = os.path.join(results_dir, CIRCUITS)
    if not os.path.exists(path):
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {entry["sha256"]: entry["text"] for entry in map(json.loads, handle)}


def _save_circuit(out_dir: str, workload: workloads.Workload, known: set[str]) -> str:
    if workload.sha256 not in known:
        with gzip.open(os.path.join(out_dir, CIRCUITS), "at", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"sha256": workload.sha256, "text": workload.text}) + "\n")
        known.add(workload.sha256)
    return CIRCUITS


def _validate_row(cell: profiles.Cell, records: list[dict], outputs: dict[str, str]) -> None:
    wl = cell.workload
    ok = [r for r in records if r["status"] == "ok" and r["simulator"] in outputs]
    if wl.mode == "probs":
        ref_name = next((n for n in PROBS_REFERENCES if any(r["simulator"] == n for r in ok)), None)
        if ref_name is None:
            for r in ok:
                r["validation"] = {"pass": None, "reason": "no fp64 reference ran on this row"}
            return
        ref = np.fromfile(outputs[ref_name], dtype="<f8")
        for r in ok:
            if r["simulator"] == ref_name:
                r["validation"] = {"pass": None, "role": "reference", "reason": "reference vector for this row"}
                continue
            vec = np.fromfile(outputs[r["simulator"]], dtype="<f8")
            distance = validate.tvd(vec, ref)
            tolerance = 1e-5 if r["precision"] == "single" else 1e-8
            r["validation"] = {
                "method": "total variation distance against the reference vector",
                "reference": ref_name,
                "tvd": distance,
                "classical_fidelity": validate.classical_fidelity(vec, ref),
                "tolerance": tolerance,
                "pass": distance is not None and distance <= tolerance,
            }
            if r["group"] == "accuracy":
                r["validation"]["pass_required"] = False
            del vec
        return

    _, _, instructions = workloads.parse(wl.text)
    singles, pairs = validate.select_statistics(wl.num_clbits)
    t0 = time.perf_counter()
    exact = validate.exact_statistics(instructions, singles, pairs)
    exact_s = time.perf_counter() - t0
    observed: dict[str, tuple[dict, int]] = {}
    for r in ok:
        bits, counts = validate.read_counts(outputs[r["simulator"]], wl.num_clbits)
        observed[r["simulator"]] = validate.observed(bits, counts, singles, pairs)
    for r in ok:
        others = [n for n in observed if _family(n) != _family(r["simulator"])]
        fallback_name = next((n for n in COUNTS_FALLBACKS if n in others), others[0] if others else None)
        fallback = observed.get(fallback_name) if fallback_name else None
        obs, shots = observed[r["simulator"]]
        check = validate.check_counts(obs, shots, exact, fallback)
        check["fallback_simulator"] = fallback_name if check["against_simulator"] else None
        check["exact_reference_s"] = round(exact_s, 3)
        r["validation"] = check


def _supersede(out_dir: str, profile: profiles.Profile, whole_workloads: bool, retire: list[str], run_id: str) -> int:
    """Move the records an appended run re-measures or retires to `superseded.jsonl`; return how many moved."""
    obs_path = os.path.join(out_dir, "observations.jsonl")
    targets = {
        (c.group, c.workload.workload_id, name, threads)
        for c in profile.cells
        for name in c.simulators
        for threads in c.threads
    }
    workloads_ = {(c.group, c.workload.workload_id) for c in profile.cells}
    kept, moved = [], []
    with open(obs_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["group"], r["workload"]["workload_id"])
            if (
                (*key, r["simulator"], r["threads"]) in targets
                or (whole_workloads and key in workloads_)
                or any(text in key[1] for text in retire)
            ):
                moved.append({**r, "superseded_by": run_id})
                continue
            kept.append(line if line.endswith("\n") else line + "\n")
    with open(os.path.join(out_dir, "superseded.jsonl"), "a", encoding="utf-8", newline="\n") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in moved)
    with open(obs_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(kept)
    return len(moved)


def cmd_run(args: argparse.Namespace) -> int:
    if not args.no_build:
        build_runner()
    profile = load_profile(args)
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    run_id = f"{dt.datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"

    names = sorted({n for c in profile.cells for n in c.simulators})
    versions = {n: runners.version(runners.resolve(n)) for n in sorted({n.partition(":")[0] for n in names})}
    manifest = _manifest(args, profile, run_id, versions)
    for warning in manifest["machine_warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    manifest_path = os.path.join(out_dir, "manifest.json")
    obs_path = os.path.join(out_dir, "observations.jsonl")
    base_manifest = None
    if args.append and os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            base_manifest = json.load(handle)
        replaced = _supersede(out_dir, profile, args.drop_other_simulators, args.retire, run_id) if args.replace else 0
        supplement = {
            "run_id": run_id,
            "command": manifest["command"],
            "started_utc": manifest["started_utc"],
            "machine_warnings": manifest["machine_warnings"],
            "code_sha256": manifest["code_sha256"],
            "replaced_records": replaced,
            "retired": args.retire,
            "workloads": [c.workload.workload_id for c in profile.cells],
        }
        base_manifest.setdefault("supplementary_runs", []).append(supplement)
        for name, entry in manifest["simulators"].items():
            if entry.get("version") and not base_manifest["simulators"].get(name, {}).get("version"):
                base_manifest["simulators"][name] = entry
        for name, version in manifest["provenance"]["simulators"].items():
            base_manifest["provenance"]["simulators"].setdefault(name, version)
        base_manifest["provenance"]["python"]["packages"].update(
            {k: v for k, v in manifest["provenance"]["python"]["packages"].items() if v}
        )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(base_manifest or manifest, handle, indent=2)

    mode = "a" if args.append else "w"
    if not args.append and os.path.exists(os.path.join(out_dir, CIRCUITS)):
        os.unlink(os.path.join(out_dir, CIRCUITS))
    known = set(load_circuits(out_dir))
    order_rng = np.random.Generator(np.random.PCG64(ORDER_SEED))
    started = time.perf_counter()
    with open(obs_path, mode, encoding="utf-8", newline="\n") as sink:
        for index, cell in enumerate(profile.cells, start=1):
            wl = cell.workload
            circuit_file = _save_circuit(out_dir, wl, known)
            elapsed = (time.perf_counter() - started) / 60
            print(f"[{index}/{len(profile.cells)} {elapsed:5.1f} min] [{cell.group}] {wl.workload_id}", flush=True)
            jobs = [(name, threads) for name in cell.simulators for threads in cell.threads]
            order = order_rng.permutation(len(jobs))
            records: list[dict] = []
            outputs: dict[str, str] = {}
            for position, job_index in enumerate(order):
                name, threads = jobs[int(job_index)]
                sim = runners.resolve(name)
                record: dict[str, Any] = {
                    "schema": SCHEMA,
                    "run_id": run_id,
                    "profile": profile.name,
                    "group": cell.group,
                    "workload": {**wl.summary(), "circuit_file": circuit_file},
                    "simulator": name,
                    "label": sim.label,
                    "adapter": sim.kind,
                    "precision": sim.precision,
                    "representation": sim.representation,
                    "threads": threads,
                    "order_in_row": position,
                    "policy": {
                        "iterations": cell.iterations,
                        "min_iterations": cell.min_iterations,
                        "budget_s": cell.budget_s,
                        "timeout_s": cell.timeout_s,
                    },
                    "version": versions.get(name.partition(":")[0]),
                }
                reason = runners.applicable(sim, wl)
                if reason:
                    record.update({"status": "not-applicable", "reason": reason})
                else:
                    suffix = ".f64" if wl.mode == "probs" else ".counts"
                    out_path = runners.temp_output(suffix)
                    result = runners.execute(
                        sim, wl, threads, cell.iterations, cell.min_iterations, cell.budget_s, cell.timeout_s, out_path
                    )
                    record.update(result)
                    record["stats"] = stats.summarize(record.get("samples_ms", []))
                    key = name if len(cell.threads) == 1 else f"{name}@{threads}"
                    if record["status"] == "ok" and threads == max(cell.threads):
                        outputs[name] = out_path
                    elif os.path.exists(out_path):
                        os.unlink(out_path)
                    median = record["stats"].get("median_ms")
                    shown = f"{median:10.2f} ms" if median is not None else f"{record['status']:>13s}"
                    print(f"      {key:30s} {shown}  {record.get('reason') or ''}"[:160], flush=True)
                    if args.cooldown_s:
                        time.sleep(args.cooldown_s)
                record["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                records.append(record)
            try:
                _validate_row(cell, [r for r in records if r["threads"] == max(cell.threads)], outputs)
            finally:
                for path in outputs.values():
                    if os.path.exists(path):
                        os.unlink(path)
            for record in records:
                v = record.get("validation")
                if v and v.get("pass") is False and v.get("pass_required", True):
                    print(f"      VALIDATION FAILED {record['simulator']}: {v.get('failures') or v.get('tvd')}")
                sink.write(json.dumps(record) + "\n")
            sink.flush()
    minutes = (time.perf_counter() - started) / 60
    finished = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if base_manifest is not None:
        base_manifest["supplementary_runs"][-1].update({"finished_utc": finished, "wall_minutes": round(minutes, 2)})
        manifest = base_manifest
    else:
        manifest["finished_utc"] = finished
        manifest["wall_minutes"] = round(minutes, 2)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nwrote {obs_path} in {minutes:.1f} min")
    return 0


def cmd_fusion_counts(args: argparse.Namespace) -> int:
    """PRISM-Q's fused instruction count for every dense headline circuit in a results directory."""
    seen: dict[str, dict] = {}
    texts = load_circuits(args.results)
    with open(os.path.join(args.results, "observations.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            wl = r["workload"]
            if r["group"] != "headline" or wl["mode"] != "probs" or wl["workload_id"] in seen:
                continue
            text = texts[wl["sha256"]]
            proc = subprocess.run([runners.RUNNER, "fusion"], input=text, capture_output=True, text=True, check=True)
            seen[wl["workload_id"]] = json.loads(proc.stdout.strip().splitlines()[-1])
    path = os.path.join(args.results, "fusion.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(seen, handle, indent=2)
    print(f"wrote {path} ({len(seen)} circuits)")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    from harness import aggregate

    aggregate.write(args.results)
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from harness import figures

    figures.write_all(args.results, args.figures_dir)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from harness import aggregate, figures, pages

    aggregate.write(args.results)
    fig_dir = os.path.join(args.docs, "figures") if args.docs else None
    figures.write_all(args.results, fig_dir)
    if args.docs:
        pages.write(args.results, args.docs, args.landing)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def profile_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--profile", choices=sorted(profiles.PROFILES), default="smoke")
        p.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="headline thread count")
        p.add_argument("--groups", default=None, help="comma separated groups to keep")
        p.add_argument("--match", default=None, help="keep workloads whose id contains this text")

    p = sub.add_parser("list", help="print the matrix a profile would run")
    profile_args(p)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("run", help="run a profile and write raw observations")
    profile_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--append", action="store_true", help="append to an existing observations file")
    p.add_argument(
        "--replace", action="store_true", help="with --append, move the records being re-measured to superseded.jsonl"
    )
    p.add_argument(
        "--drop-other-simulators",
        action="store_true",
        help="with --replace, also move records of simulators no longer in a re-measured workload's list",
    )
    p.add_argument(
        "--retire",
        action="append",
        default=[],
        help="with --replace, also move records whose workload id contains this text (a changed definition)",
    )
    p.add_argument("--simulators", default=None, help="comma separated: run only these simulators")
    p.add_argument("--cooldown-s", type=float, default=0.5, help="pause between measurements")
    p.add_argument("--no-build", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("fusion-counts", help="PRISM-Q fused instruction counts for the dense circuits")
    p.add_argument("results")
    p.set_defaults(func=cmd_fusion_counts)

    p = sub.add_parser("aggregate", help="summaries and tables from observations.jsonl")
    p.add_argument("results")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("figures", help="SVG figures from observations.jsonl")
    p.add_argument("results")
    p.add_argument("--figures-dir", default=None)
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("report", help="aggregate, draw, and write the documentation pages")
    p.add_argument("results")
    p.add_argument("--docs", default=None, help="documentation directory for the report pages")
    p.add_argument("--landing", default=None, help="landing page path to regenerate")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
