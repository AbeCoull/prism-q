import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { availableParallelism, cpus, platform, release, arch } from "node:os";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

export const root = fileURLToPath(new URL("../../../", import.meta.url));
export const canonical = (text) =>
  text.toString("utf8").replaceAll("\r\n", "\n");
export const hash = (text) => createHash("sha256").update(text).digest("hex");
const audit = JSON.parse(
  readFileSync(
    new URL("../src/data/source-manifest.json", import.meta.url),
    "utf8",
  ),
);

export function verifyAuditedImplementation() {
  for (const [path, expected] of Object.entries(audit.files)) {
    let text = canonical(readFileSync(resolve(root, path)));
    if (path === "src/circuit/fusion.rs")
      text = text.replace(
        '\n#[cfg(test)]\n#[path = "fusion_atlas_tests.rs"]\nmod atlas_tests;\n',
        "",
      );
    assert.equal(
      hash(text),
      expected,
      `Audited source changed: ${path}. Re-audit before recording.`,
    );
  }
}

function git(args) {
  const result = spawnSync("git", args, {
    cwd: root,
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
  });
  assert.equal(result.status, 0, result.error?.message ?? result.stderr);
  return result.stdout;
}

function blobs(revision, paths) {
  const result = spawnSync("git", ["cat-file", "--batch"], {
    cwd: root,
    input: paths
      .map(
        (path) => `${revision}:${path}
`,
      )
      .join(""),
    maxBuffer: 256 * 1024 * 1024,
  });
  assert.equal(result.status, 0, result.error?.message ?? `${result.stderr}`);
  const out = result.stdout,
    files = new Map();
  let at = 0;
  for (const path of paths) {
    const end = out.indexOf(10, at);
    const header = out.toString("utf8", at, end).split(" ");
    assert.equal(header[1], "blob", `Missing ${path} at ${revision}`);
    const size = Number(header[2]);
    files.set(path, out.subarray(end + 1, end + 1 + size));
    at = end + 2 + size;
  }
  return files;
}

// Ignored files never enter the manifest: a clean checkout must list the same paths.
function sourcePaths(revision = null) {
  const paths = new Set(Object.keys(audit.files));
  const listed = revision
    ? git(["ls-tree", "-r", "--name-only", revision, "--", "src"])
    : git([
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "src",
      ]);
  for (const path of listed.split("\n")) if (path) paths.add(path);
  const exists = revision
    ? (path) =>
        spawnSync("git", ["cat-file", "-e", `${revision}:${path}`], {
          cwd: root,
        }).status === 0
    : (path) => existsSync(resolve(root, path));
  for (const path of [
    "Cargo.toml",
    "Cargo.lock",
    "build.rs",
    "rust-toolchain.toml",
    ".cargo/config.toml",
    "examples/atlas_trace.rs",
    "docs/atlas/scripts/generate-traces.mjs",
    "docs/atlas/scripts/generate-fusion-traces.mjs",
    "docs/atlas/scripts/recording-provenance.mjs",
  ])
    if (exists(path)) paths.add(path);
  return [...paths].sort();
}

export function recordingProvenance(environmentOverrides = {}) {
  verifyAuditedImplementation();
  const environment = { ...process.env, ...environmentOverrides };
  const compiler = spawnSync("rustc", ["--version"], {
    cwd: root,
    encoding: "utf8",
  });
  assert.equal(compiler.status, 0, compiler.error?.message ?? compiler.stderr);
  const recordingRevision = git(["rev-parse", "HEAD"]).trim();
  return {
    auditRevision: audit.revision,
    recordingRevision,
    sourceRevision: recordingRevision,
    dirtyTree:
      git([
        "status",
        "--porcelain",
        "--",
        "src",
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "examples/atlas_trace.rs",
      ]).trim().length > 0,
    auditComparison:
      "Canonical source audit hashes; only the cfg(test) atlas module registration is excluded from the fusion.rs comparison. Full recording hashes include it and all current src files, including test counter visibility.",
    rustc: compiler.stdout.trim(),
    os: `${platform()} ${release()}`,
    cpu: cpus()[0]?.model ?? "unknown",
    architecture: arch(),
    availableParallelism: availableParallelism(),
    rustflags: process.env.RUSTFLAGS ?? "",
    encodedRustflags: process.env.CARGO_ENCODED_RUSTFLAGS ?? null,
    environment: Object.fromEntries(
      [
        "PRISM_NO_REORDER",
        "PRISM_TILE_KB",
        "PRISM_MIN_TILES_PER_CPU",
        "RAYON_NUM_THREADS",
      ].map((key) => [key, environment[key] ?? null]),
    ),
    sourceHashNormalization: "UTF-8 with CRLF normalized to LF",
    sources: sourcePaths().map((path) => ({
      path,
      sha256: hash(canonical(readFileSync(resolve(root, path)))),
    })),
  };
}

/**
 * Check a recording against the sources it was made from: the commit that last wrote
 * `fixture`, or the working tree while the fixture is uncommitted. Later source edits
 * do not fail the check; they are returned as the paths the fixture no longer matches.
 */
export function verifyRecording(provenance, fixture) {
  assert.equal(
    provenance.auditRevision,
    audit.revision,
    "Recording audit differs from source evidence",
  );
  assert.match(provenance.recordingRevision, /^[a-f0-9]{40}$/);
  assert.equal(provenance.sourceRevision, provenance.recordingRevision);
  assert.equal(typeof provenance.dirtyTree, "boolean");
  const uncommitted =
    git(["status", "--porcelain", "--", fixture]).trim().length > 0;
  const revision = uncommitted
    ? null
    : git(["log", "-1", "--format=%H", "--", fixture]).trim();
  assert(
    uncommitted || /^[a-f0-9]{40}$/.test(revision),
    `No commit records ${fixture}; fetch full history`,
  );
  const committed = revision
    ? blobs(
        revision,
        provenance.sources.map((source) => source.path),
      )
    : null;
  const read = (path) =>
    canonical(committed?.get(path) ?? readFileSync(resolve(root, path)));
  assert.deepEqual(
    provenance.sources.map((source) => source.path),
    sourcePaths(revision),
    "Recording source manifest is incomplete",
  );
  for (const source of provenance.sources)
    assert.equal(
      hash(read(source.path)),
      source.sha256,
      `Recording producer changed before the fixture was written; regenerate: ${source.path}`,
    );
  if (!revision) return [];
  return provenance.sources
    .filter(
      (source) =>
        !existsSync(resolve(root, source.path)) ||
        hash(canonical(readFileSync(resolve(root, source.path)))) !==
          source.sha256,
    )
    .map((source) => source.path);
}
