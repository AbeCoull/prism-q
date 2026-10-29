import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";

const root = new URL("../../../", import.meta.url);
const records = ["math", "runtime", "hardware"].flatMap((name) =>
  JSON.parse(
    readFileSync(
      new URL(`../src/data/${name}-evidence.json`, import.meta.url),
      "utf8",
    ),
  ),
);
const revision = records[0].revision;
if (!records.every((record) => record.revision === revision))
  throw new Error("Audit revisions differ");
const paths = [
  ...new Set(
    records.flatMap((record) => record.sources.map((source) => source.path)),
  ),
].sort();
const files = Object.fromEntries(
  paths.map((path) => [
    path,
    createHash("sha256")
      .update(
        readFileSync(new URL(path, root), "utf8").replaceAll("\r\n", "\n"),
      )
      .digest("hex"),
  ]),
);
writeFileSync(
  new URL("../src/data/source-manifest.json", import.meta.url),
  `${JSON.stringify({ revision, files }, null, 2)}\n`,
);
const matrix = [
  "# Source-evidence matrix",
  "",
  `Audited source revision: \`${revision}\`. Hashes are recorded in`,
  "`src/data/source-manifest.json`. A changed source file requires a new audit before updating its hash.",
  "",
  "Status describes the claim, not the presence of a citation. No empirical timing data is included.",
  "",
  "| Claim | Evidence type | Source | Derivation and limits |",
  "| --- | --- | --- | --- |",
  ...records.map(
    (record) =>
      `| ${record.claim.replaceAll("|", "\\|")} | ${record.status} | ${record.sources.map((source) => `[${source.symbol}](https://github.com/AbeCoull/prism-q/blob/${revision}/${source.path}#L${source.startLine}-L${source.endLine})`).join("<br>")} | ${(record.derivation + " " + record.limitations).replaceAll("|", "\\|").replaceAll("\n", " ")} |`,
  ),
  "",
];
writeFileSync(
  new URL("../research/source-evidence.md", import.meta.url),
  matrix.join("\n"),
);
process.stdout.write(
  `Wrote ${records.length} matrix rows and ${paths.length} source hashes.\n`,
);
