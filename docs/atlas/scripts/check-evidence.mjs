import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { strict as assert } from "node:assert";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { verifyRecording } from "./recording-provenance.mjs";

const here = new URL("../", import.meta.url);
const root = new URL("../../../", import.meta.url);
const manifest = JSON.parse(
  readFileSync(new URL("src/data/source-manifest.json", here), "utf8"),
);
const records = ["math", "runtime", "hardware"].flatMap((name) =>
  JSON.parse(
    readFileSync(new URL(`src/data/${name}-evidence.json`, here), "utf8"),
  ),
);
const ids = new Set();
const sourceFiles = new Map();
const canonical = (buffer) => buffer.toString("utf8").replaceAll("\r\n", "\n");
const hash = (text) => createHash("sha256").update(text).digest("hex");
for (const record of records) {
  assert(!ids.has(record.id), `Duplicate evidence ID: ${record.id}`);
  ids.add(record.id);
  assert(
    ["implementation", "theory", "model", "empirical", "proposed"].includes(
      record.status,
    ),
    `Invalid evidence status: ${record.id}`,
  );
  assert.equal(
    record.revision,
    manifest.revision,
    `Revision mismatch: ${record.id}`,
  );
  assert(record.sources.length > 0, `Missing source: ${record.id}`);
  for (const source of record.sources) {
    assert(!source.path.includes(".."), `Invalid path: ${source.path}`);
    if (!sourceFiles.has(source.path)) {
      let text = canonical(readFileSync(new URL(source.path, root)));
      if (hash(text) !== manifest.files[source.path]) {
        const historical = spawnSync(
          "git",
          ["show", `${manifest.revision}:${source.path}`],
          {
            cwd: fileURLToPath(root),
            encoding: "utf8",
            maxBuffer: 8 * 1024 * 1024,
          },
        );
        assert.equal(
          historical.status,
          0,
          `Fetch audited revision ${manifest.revision} to verify ${source.path}`,
        );
        text = canonical(historical.stdout);
      }
      assert.equal(
        hash(text),
        manifest.files[source.path],
        `Audited source hash mismatch: ${source.path}`,
      );
      sourceFiles.set(source.path, text);
    }
    const lines = sourceFiles.get(source.path).split("\n");
    assert(
      Number.isInteger(source.startLine) &&
        source.startLine >= 1 &&
        source.endLine >= source.startLine &&
        source.endLine <= lines.length,
      `Invalid range: ${record.id}`,
    );
    assert(source.symbol.length > 0, `Missing symbol: ${record.id}`);
  }
}
const fixture = JSON.parse(
  readFileSync(new URL("fixtures/native-gates.json", here), "utf8"),
);
assert.equal(fixture.schemaVersion, 1);
const staleGates = verifyRecording(
  fixture.provenance,
  "docs/atlas/fixtures/native-gates.json",
);
assert.equal(
  fixture.provenance.timing,
  null,
  "Golden traces must not contain timing claims",
);
assert.equal(fixture.traces.length, 140);
for (const trace of fixture.traces) {
  assert.equal(trace.before.length, 2 * 2 ** trace.n);
  assert.equal(trace.steps.length, 4);
  assert.equal(trace.theta, Math.PI / 3);
  for (const step of trace.steps) {
    assert.equal(step.length, trace.before.length);
    assert(step.every(Number.isFinite));
  }
}
assert.equal(fixture.wideTraces.length, 12);
for (const trace of fixture.wideTraces) {
  assert.equal(trace.n, 16);
  assert(trace.maxAbsoluteError >= 0 && trace.maxAbsoluteError < 1e-12);
  assert(Math.abs(trace.normSquared - 1) < 1e-12);
  assert(trace.samples.every((s) => s.index < 2 ** trace.n));
}
process.stdout.write(
  `${records.length} source records, ${Object.keys(manifest.files).length} source hashes, and ${fixture.traces.length} native traces verified.\n`,
);
const compiler = JSON.parse(
  readFileSync(new URL("fixtures/native-fusion.json", here), "utf8"),
);
assert.equal(compiler.schemaVersion, 1);
const staleCompiler = verifyRecording(
  compiler.generator,
  "docs/atlas/fixtures/native-fusion.json",
);
assert.equal(compiler.records.length, 31);
assert.equal(compiler.generator.seed, 42);
assert.equal(
  hash(canonical(compiler.generator.exporterSource)),
  compiler.generator.sources.find((source) =>
    source.path.endsWith("fusion_atlas_tests.rs"),
  ).sha256,
);
let snapshots = 0;
for (const record of compiler.records) {
  assert.equal(record.finalMatchesPublicPipeline, true);
  for (const stage of record.stages) {
    snapshots++;
    assert(
      stage.state.maxAbsoluteError >= 0 && stage.state.maxAbsoluteError < 1e-12,
    );
    assert(Math.abs(stage.state.normSquared - 1) < 1e-12);
    assert.equal(
      stage.metrics.measuredKernelSweeps,
      stage.instructions.reduce(
        (sum, instruction) => sum + instruction.kernelSweeps,
        0,
      ),
    );
    assert.equal(
      stage.metrics.measuredKernelSweeps,
      stage.metrics.modeledFullStatePasses,
    );
  }
}
process.stdout.write(
  `${compiler.records.length} native compiler runs, ${snapshots} snapshots, and ${compiler.generator.sources.length} producer hashes verified.\n`,
);
const stale = [...new Set([...staleGates, ...staleCompiler])];
if (stale.length)
  process.stdout.write(
    `Fixtures predate ${stale.length} changed producer files; regenerate to refresh: ${stale.join(", ")}\n`,
  );
