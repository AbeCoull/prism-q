import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import {
  recordingProvenance,
  verifyAuditedImplementation,
} from "./recording-provenance.mjs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../../", import.meta.url));
verifyAuditedImplementation();
const fixturePath = fileURLToPath(
  new URL("../fixtures/native-fusion.json", import.meta.url),
);
mkdirSync(new URL("../fixtures/", import.meta.url), { recursive: true });
const result = spawnSync(
  "cargo",
  [
    "nextest",
    "run",
    "--features",
    "parallel",
    "--lib",
    "--run-ignored",
    "ignored-only",
    "-E",
    "test(export_atlas_fusion_traces)",
  ],
  {
    cwd: root,
    stdio: "inherit",
    env: {
      ...process.env,
      RAYON_NUM_THREADS: "8",
      PRISM_ATLAS_FUSION_OUTPUT: fixturePath,
    },
  },
);
if (result.status !== 0) {
  if (result.error) process.stderr.write(`${result.error.message}\n`);
  process.exit(result.status ?? 1);
}
const fixture = JSON.parse(readFileSync(fixturePath, "utf8"));
fixture.generator = {
  ...fixture.generator,
  ...recordingProvenance({ RAYON_NUM_THREADS: "8" }),
  command: "node docs/atlas/scripts/generate-fusion-traces.mjs",
  profile: "test",
  features: ["parallel"],
  reorder2qEnabled: process.env.PRISM_NO_REORDER === undefined,
  tileBudgetOverrideKb: process.env.PRISM_TILE_KB ?? null,
  exporterSource: readFileSync(
    new URL("../../../src/circuit/fusion_atlas_tests.rs", import.meta.url),
    "utf8",
  ).replaceAll("\r\n", "\n"),
};
const stages = fixture.records.flatMap((record) => record.stages);
for (const record of fixture.records) {
  if (!record.finalMatchesPublicPipeline)
    throw new Error(`Pipeline mismatch: ${record.id}`);
  for (const stage of record.stages) {
    if (!(stage.state.maxAbsoluteError < 1e-12))
      throw new Error(`State mismatch: ${record.id}/${stage.name}`);
    if (
      stage.metrics.measuredKernelSweeps !==
      stage.metrics.modeledFullStatePasses
    ) {
      throw new Error(`Sweep counter mismatch: ${record.id}/${stage.name}`);
    }
  }
}
writeFileSync(fixturePath, `${JSON.stringify(fixture)}\n`);
process.stdout.write(
  `Exported ${fixture.records.length} native compiler runs and ${stages.length} intermediate states. No timings collected.\n`,
);
