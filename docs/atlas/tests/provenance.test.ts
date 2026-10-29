import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { root, verifyRecording } from "../scripts/recording-provenance.mjs";

const fixture = JSON.parse(
  readFileSync(
    new URL("../fixtures/native-gates.json", import.meta.url),
    "utf8",
  ),
);
const path = "docs/atlas/fixtures/native-gates.json";
describe("recording provenance", () => {
  it("accepts disclosed export changes with audited implementation and complete producer hashes", () => {
    const disclosed = structuredClone(fixture.provenance);
    disclosed.dirtyTree = true;
    expect(() => verifyRecording(disclosed, path)).not.toThrow();
    expect(fixture.provenance.environment).toHaveProperty(
      "PRISM_MIN_TILES_PER_CPU",
    );
    expect(fixture.provenance).toHaveProperty("encodedRustflags");
  });
  it("rejects a missing SIMD producer", () => {
    const copy = structuredClone(fixture.provenance);
    copy.sources = copy.sources.filter(
      (source: { path: string }) => source.path !== "src/backend/simd.rs",
    );
    expect(() => verifyRecording(copy, path)).toThrow(/incomplete/);
  });
  it("rejects changed execution source fingerprints", () => {
    const copy = structuredClone(fixture.provenance);
    copy.sources.find(
      (source: { path: string }) => source.path === "src/backend/simd.rs",
    ).sha256 = "0".repeat(64);
    expect(() => verifyRecording(copy, path)).toThrow(/producer changed/);
  });
  it("rejects an unrelated audit revision", () => {
    const copy = structuredClone(fixture.provenance);
    copy.auditRevision = "0".repeat(40);
    expect(() => verifyRecording(copy, path)).toThrow(/audit differs/);
  });
  it("records only sources a clean checkout contains", () => {
    const ignored = spawnSync(
      "git",
      [
        "check-ignore",
        ...fixture.provenance.sources.map((s: { path: string }) => s.path),
      ],
      { cwd: root, encoding: "utf8" },
    );
    expect(ignored.stdout).toBe("");
  });
});
