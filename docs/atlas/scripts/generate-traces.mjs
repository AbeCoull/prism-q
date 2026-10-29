import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  recordingProvenance,
  verifyAuditedImplementation,
} from "./recording-provenance.mjs";

const root = fileURLToPath(new URL("../../../", import.meta.url));
verifyAuditedImplementation();
const result = spawnSync(
  "cargo",
  ["run", "--quiet", "--features", "parallel", "--example", "atlas_trace"],
  { cwd: root, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
);
if (result.status !== 0) {
  process.stderr.write(
    result.stderr ?? result.error?.message ?? "Trace exporter failed",
  );
  process.exit(result.status ?? 1);
}
const fixture = JSON.parse(result.stdout);
fixture.provenance = {
  ...fixture.provenance,
  ...recordingProvenance(),
  profile: "dev",
};
fixture.provenance.exporterSource = readFileSync(
  new URL("../../../examples/atlas_trace.rs", import.meta.url),
  "utf8",
);
const directory = new URL("../fixtures/", import.meta.url);
mkdirSync(directory, { recursive: true });
writeFileSync(
  new URL("native-gates.json", directory),
  `${JSON.stringify(fixture)}\n`,
);
process.stdout.write(
  `Exported ${fixture.traces.length} native traces, four steps each. No timings collected.\n`,
);
