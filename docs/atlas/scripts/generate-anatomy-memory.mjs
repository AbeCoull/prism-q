import { readFile, writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolve } from "node:path";

const atlas = fileURLToPath(new URL("..", import.meta.url));
const checkout = resolve(atlas, process.argv[2] ?? ".cache/anatomy");
const kit = await import(
  pathToFileURL(resolve(checkout, "kit/iso-kit.mjs")).href
);
const commit = execFileSync("git", ["rev-parse", "HEAD"], {
  cwd: checkout,
  encoding: "utf8",
}).trim();
const projection = { origin: [0, 0], scale: 2.3, azimuth: 48, elevation: 40 };
const slab = (x, y, w, d, height) =>
  kit.slabOf({ x, y, w, d, r: 0.8 }, 0, height, projection, 4, 0.25);
const geometry = {
  provenance: {
    source: "https://github.com/wheresryan22/anatomy",
    commit,
    module: "kit/iso-kit.mjs",
    functions: ["slabOf", "iso"],
    license: "MIT",
    licenseText: await readFile(resolve(checkout, "LICENSE"), "utf8"),
  },
  projection,
  column: kit.iso([28, 0, 0], projection),
  row: kit.iso([0, 13.8, 0], projection),
  rowBase: slab(-13.6, -6.1, 111.2, 12.6, 1),
  amplitude: slab(-12.8, -5.5, 25.6, 11, 2),
  real: slab(-11.8, -4.6, 11.3, 9.2, 2.5),
  imag: slab(0.5, -4.6, 11.3, 9.2, 2.5),
  registerLane: slab(-6, -7, 12, 14, 2.5),
};
await writeFile(
  resolve(atlas, "src/data/anatomy-memory-geometry.json"),
  `${JSON.stringify(geometry, null, 2)}\n`,
);
