"""Benchmark matrix: run a profile, aggregate its raw observations, write the Performance page.

    python comparison/matrix.py list --profile reference
    python comparison/matrix.py run --profile smoke --out comparison/results/smoke
    python comparison/matrix.py aggregate comparison/results/smoke
    python comparison/matrix.py report comparison/results/matrix-reference-i7-6700k --docs docs/performance

`run` writes `observations.jsonl` (one record per simulator, workload and
thread count, failures included), `manifest.json` (host, toolchain, versions,
profile, code hashes) and `circuits.jsonl.gz` (every gate list with its SHA-256). The
outputs (up to 64 MiB each) of the simulators a row can validate against are kept gzipped under `outputs/`,
so a later rerun of part of the row can still be checked. A rerun with `--append
--replace` moves the records it re-measures to `superseded.jsonl` once each row's new
records are written. The later commands read only those files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import aggregate, profiles, provenance, runners, stats, validate, workloads  # noqa: E402

SCHEMA = "prism-q-matrix/1"
ORDER_SEED = 42
PROBS_REFERENCES = ("aer-statevector", "quest")
COUNTS_FALLBACKS = ("aer-statevector", "aer-stabilizer", "aer-mps", "aer-automatic", "stim", "prismq")
FP64_TOLERANCE = 1e-8
FP32_TOLERANCE = 1e-5
NOT_PASS_FAIL = ("accuracy", "bondsweep")
OUTPUTS = "outputs"
MAX_STORED_BYTES = 1 << 26
OUTPUT_SUFFIX = {"probs": ".f64", "counts": ".counts", "expval": ".values", "stats": ".stats"}
FRESH_RUN_REMOVES = ("superseded.jsonl", "fusion.json", "samples.csv", "summary.csv", "derived.json", OUTPUTS)


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
    skipped = getattr(args, "skip_peer", None) or []
    if skipped:
        for cell in profile.cells:
            cell.simulators = [s for s in cell.simulators if not _skipped(s, skipped)]
    profile.cells = [c for c in profile.cells if c.simulators]
    return profile


def _skipped(name: str, skipped: list[str]) -> bool:
    return name in skipped or name.partition(":")[0] in skipped


def not_applicable(cell: profiles.Cell, name: str, sim: runners.Simulator) -> str | None:
    """The profile's declared exclusion of `name` on this row, else the registry's applicability rule."""
    return cell.excluded.get(name) or runners.applicable(sim, cell.workload)


def cmd_list(args: argparse.Namespace) -> int:
    profile = load_profile(args)
    runs = skips = 0
    for cell in profile.cells:
        print(f"[{cell.group}] {cell.workload.workload_id}  ({cell.workload.num_operations} ops)")
        for name in cell.simulators:
            reason = not_applicable(cell, name, runners.resolve(name))
            threads = cell.threads_for(name)
            if reason:
                skips += len(threads)
            else:
                runs += len(threads)
            mark = f"skip: {reason}" if reason else f"threads {threads}"
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
    child = runners.child_env(args.threads)
    prov = provenance.collect("parallel", args.threads, versions, {k: child[k] for k in runners.THREAD_ENV_KEYS})
    prov["host"].update(provenance.host_details())
    prov["toolchain"]["native_builds"] = runners.native_builds()
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
                "Aer's Result, qsim's measurement array, Stim's bit-packed sample array, quimb's sample list)"
            ),
            "expval": (
                "execute the circuit and return every requested expectation value (PRISM-Q's "
                "expectation_values, Aer's saved expectation values, quimb's local_expectation per "
                "observable); quimb's tensor network finds a fresh contraction path for every observable "
                "inside the timed call, and quimb's CircuitMPS applies the gates inside it"
            ),
            "stats": (
                "compile the sampler and draw every shot into the simulator's packed shot buffer "
                "(PRISM-Q's compile_measurements and CompiledSampler::try_sample_bulk_packed, Stim's "
                "compile_sampler and sample with bit_packed=True), in buffers of at most 64 MiB; reducing "
                "each buffer to the selected statistics happens outside the timer on both sides"
            ),
            "excluded": "process start, imports, building the circuit object from the gate list, "
            "conversion to the harness file format",
        },
        "repetition_policy": (
            "one cold call, then up to `iterations` warm calls in the same process; stop early once the "
            "warm calls have used `budget_s`, never below `min_iterations`"
        ),
        "validation": {
            "probs": (
                "total variation distance against the fp64 reference simulator's vector; Aer is the reference "
                f"only within {FP64_TOLERANCE:g} of QuEST where QuEST ran on the row"
            ),
            "counts": validate.COUNTS_METHOD,
            "stats": validate.STATS_METHOD,
            "expval": (
                f"{validate.EXPVAL_METHOD}; tolerance {FP64_TOLERANCE:g} for fp64 and {FP32_TOLERANCE:g} for fp32 "
                "simulators; the bond-cap sweep reports the largest error without a pass or fail"
            ),
            "family_alpha": validate.FAMILY_ALPHA,
            "stored_outputs": f"gzipped outputs of the reference candidates under `{OUTPUTS}/`, named by `output_file`",
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
                "env": dict(s.env),
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


def _read_vector(path: str) -> np.ndarray:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return np.frombuffer(handle.read(), dtype="<f8")
    return np.fromfile(path, dtype="<f8")


def _stored_sources(stored: list[dict], names: tuple[str, ...] | None, out_dir: str) -> dict[str, tuple[str, dict]]:
    """Kept output path and record of each stored record of `names` (None: any) that passed or was the reference."""
    found: dict[str, tuple[str, dict]] = {}
    for r in stored:
        path = os.path.join(out_dir, r["output_file"]) if r.get("output_file") else None
        if (names is None or r["simulator"] in names) and aggregate.validated(r) and path and os.path.exists(path):
            found[r["simulator"]] = (path, r)
    return found


def _validate_row(
    cell: profiles.Cell,
    records: list[dict],
    outputs: dict[str, str],
    stored: list[dict] | None = None,
    out_dir: str = "",
) -> None:
    """Attach a `validation` block to every ok record of one row at its top thread count.

    `stored` holds earlier records of the same row and circuit for simulators this run did
    not measure; their kept outputs stand in for a reference the rerun left out.
    """
    ok = [r for r in records if r["status"] == "ok" and r["simulator"] in outputs]
    if cell.workload.mode == "probs":
        _validate_probs(ok, outputs, stored or [], out_dir)
    elif cell.workload.mode == "expval":
        _validate_expval(cell.workload, ok, outputs, stored or [], out_dir)
    else:
        _validate_counts(cell.workload, ok, outputs, stored or [], out_dir)


def _validate_probs(ok: list[dict], outputs: dict[str, str], stored: list[dict], out_dir: str) -> None:
    sources: dict[str, tuple[str, dict | None]] = {r["simulator"]: (outputs[r["simulator"]], None) for r in ok}
    for name, (path, r) in _stored_sources(stored, PROBS_REFERENCES, out_dir).items():
        sources.setdefault(name, (path, r))
    ref_name = next((n for n in PROBS_REFERENCES if n in sources), None)
    if ref_name is None:
        for r in ok:
            r["validation"] = {"pass": None, "reason": "no fp64 reference ran on this row"}
        return
    ref_path, ref_record = sources[ref_name]
    ref = _read_vector(ref_path)
    origin = {"reference_run_id": ref_record["run_id"]} if ref_record else {}
    cross_check = None
    if ref_name == "aer-statevector" and "quest" in sources:
        distance = validate.tvd(ref, _read_vector(sources["quest"][0]))
        if distance is None or distance > FP64_TOLERANCE:
            shown = "different lengths" if distance is None else f"{distance:.2g}"
            reason = f"Aer and QuEST disagree (distance {shown}, tolerance {FP64_TOLERANCE:g}); no trusted reference"
            for r in ok:
                r["validation"] = {"pass": None, "reason": reason}
            return
        cross_check = {"simulator": "quest", "tvd": distance, "tolerance": FP64_TOLERANCE}
    for r in ok:
        if r["simulator"] == ref_name:
            r["validation"] = {"pass": None, "role": "reference", "reason": "reference vector for this row"}
            if cross_check:
                r["validation"]["cross_check"] = cross_check
            continue
        vec = _read_vector(outputs[r["simulator"]])
        distance = validate.tvd(vec, ref)
        tolerance = FP32_TOLERANCE if r["precision"] == "single" else FP64_TOLERANCE
        r["validation"] = {
            "method": "total variation distance against the reference vector",
            "reference": ref_name,
            **origin,
            "tvd": distance,
            "classical_fidelity": validate.classical_fidelity(vec, ref),
            "tolerance": tolerance,
            "pass": distance is not None and distance <= tolerance,
        }
        if r["group"] in NOT_PASS_FAIL:
            r["validation"]["pass_required"] = False
        del vec


def _preference(names: list[str]) -> list[str]:
    """Candidates ordered by `COUNTS_FALLBACKS` on their base name, the rest after in their own order."""
    rank = {n: i for i, n in enumerate(COUNTS_FALLBACKS)}
    return sorted(names, key=lambda n: rank.get(n.partition(":")[0], len(rank)))


def _kept(name: str, mode: str) -> bool:
    """Whether a row keeps this simulator's output for a later partial rerun to validate against."""
    if mode == "probs":
        return name in PROBS_REFERENCES
    return mode in ("expval", "stats") or name.partition(":")[0] in COUNTS_FALLBACKS


_EXACT: dict[str, dict] = {}


def _validate_counts(
    wl: workloads.Workload, ok: list[dict], outputs: dict[str, str], stored: list[dict], out_dir: str
) -> None:
    _, _, instructions = workloads.parse(wl.text)
    singles, pairs = validate.select_statistics(wl.num_clbits)
    method = validate.STATS_METHOD if wl.mode == "stats" else validate.COUNTS_METHOD
    t0 = time.perf_counter()
    if wl.sha256 not in _EXACT:
        _EXACT[wl.sha256] = validate.exact_statistics(instructions, singles, pairs)
    exact = _EXACT[wl.sha256]
    exact_s = time.perf_counter() - t0

    def observe(path: str) -> tuple[dict, int]:
        if wl.mode == "stats":
            return validate.read_stats(path)
        bits, counts = validate.read_counts(path, wl.num_clbits)
        return validate.observed(bits, counts, singles, pairs)

    observed: dict[str, tuple[dict, int]] = {}
    for r in ok:
        try:
            observed[r["simulator"]] = observe(outputs[r["simulator"]])
        except (OSError, ValueError) as exc:
            r["validation"] = {"method": method, "pass": False, "failures": [f"unreadable: {exc}"]}
    stored_runs: dict[str, str] = {}
    for name, (path, r) in _stored_sources(stored, None, out_dir).items():
        if name in observed or not _kept(name, wl.mode):
            continue
        try:
            observed[name] = observe(path)
        except (OSError, ValueError):
            continue
        stored_runs[name] = r["run_id"]
    for r in ok:
        if "validation" in r:
            continue
        others = _preference([n for n in observed if _family(n) != _family(r["simulator"]) and observed[n][1]])
        fallback_name = others[0] if others else None
        fallback = observed.get(fallback_name) if fallback_name else None
        obs, shots = observed[r["simulator"]]
        check = validate.check_counts(obs, shots, exact, fallback, wl.shots)
        if wl.mode == "stats":
            check["method"] = method
            missing = len(singles) + len(pairs) - len(obs)
            if missing:
                check["failures"].insert(0, f"{missing} requested statistics missing from the output")
                check["pass"] = False
        used = fallback_name if check["against_simulator"] else None
        check["fallback_simulator"] = used
        if used in stored_runs:
            check["fallback_run_id"] = stored_runs[used]
        check["exact_reference_s"] = round(exact_s, 3)
        r["validation"] = check


def _validate_expval(
    wl: workloads.Workload, ok: list[dict], outputs: dict[str, str], stored: list[dict], out_dir: str
) -> None:
    _, _, instructions = workloads.parse(wl.text)
    keys = validate.keys_for(wl)
    t0 = time.perf_counter()
    exact = validate.exact_expectations(instructions, keys)
    exact_s = time.perf_counter() - t0
    values: dict[str, dict[str, float]] = {}
    for r in ok:
        try:
            values[r["simulator"]] = validate.read_values(outputs[r["simulator"]])
        except (OSError, ValueError) as exc:
            r["validation"] = {"method": validate.EXPVAL_METHOD, "pass": False, "failures": [f"unreadable: {exc}"]}
    stored_runs: dict[str, str] = {}
    for name, (path, r) in _stored_sources(stored, None, out_dir).items():
        if name in values:
            continue
        try:
            values[name] = validate.read_values(path)
        except (OSError, ValueError):
            continue
        stored_runs[name] = r["run_id"]
    sweep = any(r["group"] in NOT_PASS_FAIL for r in ok)
    for r in ok:
        if "validation" in r:
            continue
        others = [] if sweep else [n for n in values if _family(n) != _family(r["simulator"])]
        other = (others[0], values[others[0]]) if others else None
        tolerance = FP32_TOLERANCE if r["precision"] == "single" else FP64_TOLERANCE
        check = validate.check_expectations(values[r["simulator"]], keys, exact, other, tolerance)
        if check["agreement_simulator"] in stored_runs:
            check["agreement_run_id"] = stored_runs[check["agreement_simulator"]]
        check["exact_reference_s"] = round(exact_s, 3)
        if r["group"] in NOT_PASS_FAIL:
            check["pass_required"] = False
        r["validation"] = check


def _store_output(out_dir: str, record: dict, path: str) -> str:
    """Gzip one output under `outputs/` and return its path relative to the results directory."""
    tag = hashlib.sha256(json.dumps(aggregate.identity(record)).encode()).hexdigest()[:12]
    name = f"{record['run_id']}-{tag}{os.path.splitext(path)[1]}.gz"
    os.makedirs(os.path.join(out_dir, OUTPUTS), exist_ok=True)
    with open(path, "rb") as src, gzip.open(os.path.join(out_dir, OUTPUTS, name), "wb", compresslevel=1) as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    return f"{OUTPUTS}/{name}"


def _supersede(out_dir: str, cells: list[profiles.Cell], drop_other: bool, retire: list[str], run_id: str) -> int:
    """Move earlier records the given rows re-measured, or `retire` names, to `superseded.jsonl`.

    Records written by `run_id` stay. With `drop_other`, a row's records of simulators its
    profile no longer lists, and of thread counts it no longer runs, move too. Returns how
    many moved; `observations.jsonl` is replaced in one step.
    """
    obs_path = os.path.join(out_dir, "observations.jsonl")
    targets = {
        (c.group, c.workload.workload_id, name, threads)
        for c in cells
        for name in c.simulators
        for threads in c.threads_for(name)
    }
    rows = {(c.group, c.workload.workload_id): c for c in cells}
    kept, moved = [], []
    with open(obs_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["group"], r["workload"]["workload_id"])
            cell = rows.get(key)
            dropped = (
                drop_other
                and cell is not None
                and (
                    r["simulator"] not in cell.profile_simulators
                    or (r["simulator"] in cell.simulators and r["threads"] not in cell.threads_for(r["simulator"]))
                )
            )
            if r["run_id"] != run_id and (
                (*key, r["simulator"], r["threads"]) in targets or dropped or any(text in key[1] for text in retire)
            ):
                moved.append({**r, "superseded_by": run_id})
                continue
            kept.append(line if line.endswith("\n") else line + "\n")
    if not moved:
        return 0
    for r in moved:
        output = r.pop("output_file", None)
        if output and os.path.exists(os.path.join(out_dir, output)):
            os.unlink(os.path.join(out_dir, output))
    with open(os.path.join(out_dir, "superseded.jsonl"), "a", encoding="utf-8", newline="\n") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in moved)
    temp = obs_path + ".tmp"
    with open(temp, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(kept)
    os.replace(temp, obs_path)
    return len(moved)


def _stored_rows(obs_path: str) -> dict[tuple[str, str], list[dict]]:
    """The newest record per identity already in `observations.jsonl`, by group and workload id."""
    if not os.path.exists(obs_path):
        return {}
    with open(obs_path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    rows: dict[tuple[str, str], list[dict]] = {}
    for r in aggregate.newest(records):
        rows.setdefault((r["group"], r["workload"]["workload_id"]), []).append(r)
    return rows


def cmd_run(args: argparse.Namespace) -> int:
    if not args.no_build:
        build_runner()
    profile = load_profile(args)
    checks = smoke_check(profile)
    failed = sorted(name for name, check in checks.items() if not check["ok"])
    if failed:
        print(
            f"peer smoke check failed for {', '.join(failed)}; fix them or pass --skip-peer NAME to run without them",
            file=sys.stderr,
        )
        return 1
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    run_id = f"{dt.datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"

    names = sorted({n for c in profile.cells for n in c.simulators})
    versions = {n: runners.version(runners.resolve(n)) for n in sorted({n.partition(":")[0] for n in names})}
    manifest = _manifest(args, profile, run_id, versions)
    manifest["skipped_peers"] = list(args.skip_peer)
    manifest["smoke"] = checks
    for warning in manifest["machine_warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    manifest_path = os.path.join(out_dir, "manifest.json")
    obs_path = os.path.join(out_dir, "observations.jsonl")
    base_manifest = None
    if not args.append:
        for name in (CIRCUITS, "observations.jsonl", *FRESH_RUN_REMOVES):
            path = os.path.join(out_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.unlink(path)
    elif os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            base_manifest = json.load(handle)
        supplement = {
            "run_id": run_id,
            "command": manifest["command"],
            "started_utc": manifest["started_utc"],
            "machine_warnings": manifest["machine_warnings"],
            "code_sha256": manifest["code_sha256"],
            "replaced_records": 0,
            "retired": args.retire,
            "skipped_peers": list(args.skip_peer),
            "smoke": checks,
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

    known = set(load_circuits(out_dir))
    stored_rows = _stored_rows(obs_path) if args.append else {}
    order_rng = np.random.Generator(np.random.PCG64(ORDER_SEED))
    started = time.perf_counter()
    replaced = 0
    for index, cell in enumerate(profile.cells, start=1):
        wl = cell.workload
        circuit_file = _save_circuit(out_dir, wl, known)
        elapsed = (time.perf_counter() - started) / 60
        print(f"[{index}/{len(profile.cells)} {elapsed:5.1f} min] [{cell.group}] {wl.workload_id}", flush=True)
        jobs = [(name, threads) for name in cell.simulators for threads in cell.threads_for(name)]
        order = order_rng.permutation(len(jobs))
        top = {name: max(cell.threads_for(name)) for name in cell.simulators}
        records: list[dict] = []
        outputs: dict[str, str] = {}
        extra_outputs: dict[tuple[str, int], str] = {}
        temps: list[str] = []
        keys_path = None
        if wl.mode in ("expval", "stats"):
            keys_path = runners.temp_output(".keys")
            temps.append(keys_path)
            with open(keys_path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write("\n".join(validate.keys_for(wl)) + "\n")
        try:
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
                reason = not_applicable(cell, name, sim)
                if reason:
                    status = "skipped" if name in cell.excluded else "not-applicable"
                    record.update({"status": status, "reason": reason})
                else:
                    out_path = runners.temp_output(OUTPUT_SUFFIX[wl.mode])
                    temps.append(out_path)
                    try:
                        record.update(
                            runners.execute(
                                sim,
                                wl,
                                threads,
                                cell.iterations,
                                cell.min_iterations,
                                cell.budget_s,
                                cell.timeout_s,
                                out_path,
                                keys_path,
                            )
                        )
                    except Exception as exc:
                        record.update({"status": "error", "reason": f"harness: {type(exc).__name__}: {exc}"})
                    record["stats"] = stats.summarize(record.get("samples_ms", []))
                    if record["status"] == "ok" and threads == top[name]:
                        outputs[name] = out_path
                    elif record["status"] == "ok" and name in cell.sim_threads:
                        extra_outputs[(name, threads)] = out_path
                    elif os.path.exists(out_path):
                        os.unlink(out_path)
                    key = name if len(cell.threads_for(name)) == 1 else f"{name}@{threads}"
                    median = record["stats"].get("median_ms")
                    shown = f"{median:10.2f} ms" if median is not None else f"{record['status']:>13s}"
                    print(f"      {key:30s} {shown}  {record.get('reason') or ''}"[:160], flush=True)
                    if args.cooldown_s:
                        time.sleep(args.cooldown_s)
                record["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                records.append(record)
            at_top = [r for r in records if r["threads"] == top[r["simulator"]]]
            measured = {r["simulator"] for r in at_top}
            stored = [
                r
                for r in stored_rows.get((cell.group, wl.workload_id), [])
                if r["simulator"] not in measured
                and r["threads"] == max(cell.threads_for(r["simulator"]))
                and r["workload"]["sha256"] == wl.sha256
            ]
            extra = [r for r in records if (r["simulator"], r["threads"]) in extra_outputs]
            try:
                _validate_row(cell, at_top, outputs, stored, out_dir)
                for r in extra:
                    context = [{k: v for k, v in x.items() if k != "validation"} for x in at_top]
                    context = [x for x in context if x["simulator"] != r["simulator"]]
                    paths = {**outputs, r["simulator"]: extra_outputs[(r["simulator"], r["threads"])]}
                    _validate_row(cell, [r, *context], paths, stored, out_dir)
            except Exception as exc:
                for r in at_top + extra:
                    if r["status"] == "ok" and "validation" not in r:
                        r["validation"] = {"pass": None, "reason": f"validation raised {type(exc).__name__}: {exc}"}
            for r in at_top:
                if (
                    r["simulator"] in outputs
                    and _kept(r["simulator"], wl.mode)
                    and os.path.getsize(outputs[r["simulator"]]) <= MAX_STORED_BYTES
                ):
                    r["output_file"] = _store_output(out_dir, r, outputs[r["simulator"]])
        finally:
            for path in temps:
                if os.path.exists(path):
                    os.unlink(path)
        for record in records:
            v = record.get("validation")
            if v and v.get("pass") is False and v.get("pass_required", True):
                print(f"      VALIDATION FAILED {record['simulator']}: {v.get('failures') or v.get('tvd')}")
        with open(obs_path, "a", encoding="utf-8", newline="\n") as sink:
            sink.writelines(json.dumps(r) + "\n" for r in records)
        if args.replace:
            replaced += _supersede(out_dir, [cell], args.drop_other_simulators, [], run_id)
    if args.replace and args.retire:
        replaced += _supersede(out_dir, [], False, args.retire, run_id)
    minutes = (time.perf_counter() - started) / 60
    finished = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if base_manifest is not None:
        base_manifest["supplementary_runs"][-1].update(
            {"finished_utc": finished, "wall_minutes": round(minutes, 2), "replaced_records": replaced}
        )
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


SMOKE_SHOTS = 100
SMOKE_TIMEOUT_S = 180.0
SMOKE_CIRCUITS = {
    # Sixteen qubits: Aer's single-precision statevector crashes on registers of 12 qubits or fewer.
    # Depth 2 keeps a bond cap of 2 exact.
    "probs": workloads.rcs(16, 2).text,
    "clifford": "qubits 3\nh 0\ncx 0 1\ns 1\nh 2\ncx 1 2\nsdg 0\nh 1\n",
    # Deterministic: classical bits 0 and 2 read 1, bit 1 reads 0.
    "counts": "qubits 3\nclbits 3\nx 0\ncx 0 1\nmeasure 0 2\nmeasure 1 0\nmeasure 2 1\n",
    # One CX per cut, so a bond cap of 2 is exact.
    "expval": "qubits 4\nry 0 0.7\nry 1 1.3\ncx 0 1\nrz 1 0.4\nry 2 0.9\ncx 1 2\nry 3 2.1\ncx 2 3\nry 1 0.5\n",
}


GATE_SMOKE_QUBITS = 16
GATE_SMOKE_PAIR = (3, 1)


def gate_smoke_text(gate: str | None) -> str:
    """A seeded product state, one `gate` on qubit 3 (and 1), then Ry on every qubit.

    The control sits above the target so a swapped orientation shows, and the closing Ry
    turns a phase into a change in the probabilities. `gate` None is the bare preparation.
    """
    n = GATE_SMOKE_QUBITS
    lines = [f"qubits {n}"]
    for q in range(n):
        lines += [f"ry {q} {0.3 + 0.17 * q!r}", f"rz {q} {0.5 + 0.23 * q!r}"]
    a, b = GATE_SMOKE_PAIR
    if gate in ("h", "x", "z", "s", "sdg"):
        lines.append(f"{gate} {a}")
    elif gate in ("ry", "rz"):
        lines.append(f"{gate} {a} 1.1")
    elif gate in ("cx", "cz", "swap"):
        lines.append(f"{gate} {a} {b}")
    elif gate == "cp":
        lines.append(f"cp {a} {b} 0.9")
    lines += [f"ry {q} 0.7" for q in range(n)]
    return "\n".join(lines) + "\n"


def _gate_smoke(sim: runners.Simulator) -> list[str | None]:
    """The gates a dense probability peer declares, each smoked on its own, after the bare preparation."""
    if "probs" not in sim.modes or not {"ry", "rz"} <= sim.gates:
        return []
    return [None] + [g for g in sorted(runners.UNITARY) if g in sim.gates]


def _smoke_workload(mode: str, sim: runners.Simulator) -> workloads.Workload:
    if mode == "probs":
        text = SMOKE_CIRCUITS["probs" if "ry" in sim.gates else "clifford"]
    else:
        text = SMOKE_CIRCUITS["counts" if mode in ("counts", "stats") else "expval"]
    n = int(text.split()[1])
    return workloads.Workload("smoke", n, mode, text, {}, SMOKE_SHOTS if mode in ("counts", "stats") else 0)


def _smoke_output(mode: str, path: str, num_bits: int):
    if mode == "probs":
        return _read_vector(path)
    if mode == "counts":
        bits, counts = validate.read_counts(path, num_bits)
        return sorted((tuple(int(b) for b in row), int(c)) for row, c in zip(bits, counts))
    if mode == "stats":
        return validate.read_stats(path)
    return validate.read_values(path)


def _smoke_compare(mode: str, ours, theirs, precision: str) -> tuple[bool, str]:
    tolerance = FP32_TOLERANCE if precision == "single" else FP64_TOLERANCE
    if mode == "probs":
        distance = validate.tvd(ours, theirs)
        return (
            distance is not None and distance <= tolerance,
            f"distance {distance:.2g}" if distance is not None else "length differs",
        )
    if mode == "expval":
        gaps = [abs(theirs[k] - v) for k, v in ours.items() if k in theirs]
        if len(gaps) != len(ours):
            return False, "missing values"
        return max(gaps) <= tolerance, f"largest difference {max(gaps):.2g}"
    return ours == theirs, "identical output" if ours == theirs else "output differs from PRISM-Q"


def smoke_check(profile: profiles.Profile, quiet: bool = False) -> dict[str, dict]:
    """Run a tiny circuit in each mode every named simulator serves on the profile, against PRISM-Q.

    Peers run at their top thread count on the profile, so a multithreaded kernel is the one checked;
    a dense probability peer also runs one circuit per gate it declares.

    Returns one entry per simulator name: `ok`, `version` and the per-mode `checks`. A missing
    executable or package, a failed run, or an output that disagrees with PRISM-Q fails the entry;
    a mode the simulator cannot serve on the tiny circuit is reported as skipped.
    """
    modes: dict[str, set[str]] = {}
    top_threads: dict[str, int] = {}
    for cell in profile.cells:
        for name in cell.simulators:
            modes.setdefault(name, set()).add(cell.workload.mode)
            top_threads[name] = max(top_threads.get(name, 1), *cell.threads_for(name))
    versions: dict[str, str | None] = {}
    reference: dict[tuple[str, str], Any] = {}
    temps: list[str] = []

    def run(name: str, wl: workloads.Workload, threads: int) -> tuple[Any, str | None]:
        sim = runners.resolve(name)
        out_path = runners.temp_output(OUTPUT_SUFFIX[wl.mode])
        temps.append(out_path)
        keys_path = None
        if wl.mode in ("expval", "stats"):
            keys_path = runners.temp_output(".keys")
            temps.append(keys_path)
            with open(keys_path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write("\n".join(validate.keys_for(wl)) + "\n")
        obs = runners.execute(sim, wl, threads, 1, 1, 0.0, SMOKE_TIMEOUT_S, out_path, keys_path)
        if obs["status"] != "ok":
            return None, f"{obs['status']}: {obs.get('reason') or ''}".strip()
        try:
            return _smoke_output(wl.mode, out_path, wl.num_clbits), None
        except (OSError, ValueError) as exc:
            return None, f"unreadable output: {exc}"

    results: dict[str, dict] = {}
    try:
        for name in sorted(modes, key=lambda n: (n != "prismq", n)):
            sim = runners.resolve(name)
            base = name.partition(":")[0]
            if base not in versions:
                versions[base] = runners.version(sim)
            entry: dict[str, Any] = {"ok": True, "version": versions[base], "checks": {}}
            results[name] = entry
            executable = runners._executable(sim)
            if executable is not None and not os.path.exists(executable):
                entry.update({"ok": False, "reason": f"executable not built: {executable}"})
                continue
            if versions[base] is None:
                entry.update({"ok": False, "reason": "not installed or does not start (no version reported)"})
                continue
            for mode in sorted(modes[name]):
                wl = _smoke_workload(mode, sim)
                reason = runners.applicable(sim, wl)
                if reason:
                    entry["checks"][mode] = {"result": "skipped", "detail": reason}
                    continue
                key = (mode, wl.sha256)
                if key not in reference:
                    reference[key] = run("prismq", wl, 1)
                expected, ref_error = reference[key]
                got, error = (expected, ref_error) if name == "prismq" else run(name, wl, top_threads[name])
                if error:
                    entry["checks"][mode] = {"result": "fail", "detail": error[:200]}
                elif ref_error:
                    entry["checks"][mode] = {"result": "fail", "detail": f"PRISM-Q reference failed: {ref_error[:160]}"}
                else:
                    agree, detail = _smoke_compare(mode, got, expected, sim.precision)
                    entry["checks"][mode] = {"result": "pass" if agree else "fail", "detail": detail}
            if "probs" in modes[name]:
                for gate in _gate_smoke(sim):
                    text = gate_smoke_text(gate)
                    wl = workloads.Workload("smoke", GATE_SMOKE_QUBITS, "probs", text, {}, 0)
                    label = f"gate {gate or 'preparation'}"
                    faults = dict(sim.gate_faults)
                    if gate in faults:
                        entry["checks"][label] = {"result": "skipped", "detail": faults[gate]}
                        continue
                    key = ("probs", wl.sha256)
                    if key not in reference:
                        reference[key] = run("prismq", wl, 1)
                    expected, ref_error = reference[key]
                    got, error = (expected, ref_error) if name == "prismq" else run(name, wl, top_threads[name])
                    if error or ref_error:
                        entry["checks"][label] = {"result": "fail", "detail": (error or ref_error)[:200]}
                    else:
                        agree, detail = _smoke_compare("probs", got, expected, sim.precision)
                        entry["checks"][label] = {"result": "pass" if agree else "fail", "detail": detail}
            entry["ok"] = all(c["result"] != "fail" for c in entry["checks"].values())
    finally:
        for path in temps:
            if os.path.exists(path):
                os.unlink(path)
    if not quiet:
        for name, entry in results.items():
            status = "ok  " if entry["ok"] else "FAIL"
            print(f"{status} {name:24s} {entry['version'] or 'n/a':12s} {entry.get('reason') or ''}".rstrip())
            for mode, check in entry["checks"].items():
                print(f"       {mode:7s} {check['result']:8s} {check['detail']}"[:200])
    return results


def cmd_smoke(args: argparse.Namespace) -> int:
    results = smoke_check(load_profile(args))
    return 0 if all(entry["ok"] for entry in results.values()) else 1


def cmd_aggregate(args: argparse.Namespace) -> int:
    from harness import aggregate

    aggregate.write(args.results)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from harness import aggregate, pages

    if args.docs:
        pages.write(args.results, args.docs)
    else:
        aggregate.write(args.results)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def profile_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--profile", choices=sorted(profiles.PROFILES), default="smoke")
        p.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="headline thread count")
        p.add_argument("--groups", default=None, help="comma separated groups to keep")
        p.add_argument("--match", default=None, help="keep workloads whose id contains this text")

    def skip_peer_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--skip-peer",
            action="append",
            default=[],
            metavar="NAME",
            help="leave this simulator (or every variant of a base name) out of the run; recorded in the manifest",
        )

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
    skip_peer_arg(p)
    p.add_argument("--cooldown-s", type=float, default=0.5, help="pause between measurements")
    p.add_argument("--no-build", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("smoke", help="check every peer a profile names is installed and agrees with PRISM-Q")
    profile_args(p)
    p.add_argument("--simulators", default=None, help="comma separated: check only these simulators")
    skip_peer_arg(p)
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("fusion-counts", help="PRISM-Q fused instruction counts for the dense circuits")
    p.add_argument("results")
    p.set_defaults(func=cmd_fusion_counts)

    p = sub.add_parser("aggregate", help="summaries and tables from observations.jsonl")
    p.add_argument("results")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("report", help="aggregate and write the Performance page")
    p.add_argument("results")
    p.add_argument("--docs", default=None, help="page directory, docs/performance")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
