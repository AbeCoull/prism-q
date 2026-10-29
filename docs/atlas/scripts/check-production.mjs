import { createServer } from "node:http";
import { existsSync, readFileSync, statSync } from "node:fs";
import { resolve, extname } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import { chromium } from "@playwright/test";

const dist = fileURLToPath(new URL("../dist/", import.meta.url));
const prefix = "/prism-q/atlas/";
const types = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".woff2": "font/woff2",
  ".woff": "font/woff",
  ".ttf": "font/ttf",
};
const server = createServer((request, response) => {
  const path = decodeURIComponent(
    new URL(request.url, "http://localhost").pathname,
  );
  const file = resolve(dist, path.slice(prefix.length) || "index.html");
  if (
    !path.startsWith(prefix) ||
    !file.startsWith(dist) ||
    !existsSync(file) ||
    !statSync(file).isFile()
  ) {
    response.writeHead(404).end();
    return;
  }
  response.writeHead(200, {
    "Content-Type": types[extname(file)] ?? "application/octet-stream",
  });
  response.end(readFileSync(file));
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const edge =
  process.platform === "win32" &&
  existsSync("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe");
const browser = await chromium.launch({
  headless: true,
  ...(edge ? { channel: "msedge" } : {}),
});
try {
  const page = await browser.newPage({ reducedMotion: "reduce" });
  const errors = [],
    failures = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("response", (response) => {
    if (response.status() >= 400) failures.push(response.url());
  });
  await page.goto(
    `http://127.0.0.1:${server.address().port}${prefix}?renderer=svg&compiler=diagonal-pair-runs-20-Host&pass=11#compiler-laboratory`,
  );
  const lab = page.getByTestId("compiler-laboratory");
  await lab
    .getByRole("combobox", { name: "Compiler width", exact: true })
    .waitFor();
  assert.equal(
    await lab
      .getByRole("combobox", { name: "Compiler width", exact: true })
      .inputValue(),
    "20",
  );
  await lab
    .getByRole("button", { name: "Copy compiler experiment link", exact: true })
    .click();
  assert.equal(new URL(page.url()).pathname, prefix);
  await page.reload();
  await lab
    .getByRole("combobox", { name: "Compiler width", exact: true })
    .waitFor();
  assert.equal(
    await lab
      .getByRole("combobox", { name: "Compiler width", exact: true })
      .inputValue(),
    "20",
  );
  assert.equal(
    await page
      .getByTestId("workspace")
      .getByText("NATIVE REPLAY", { exact: true })
      .count(),
    1,
  );
  assert.equal(await lab.getByRole("alert").count(), 0);
  assert.deepEqual(errors, []);
  assert.deepEqual(failures, []);
  console.log(
    "Production PASS: nested Pages path, built assets, native fixtures, compiler link restoration, no page or request failures.",
  );
} finally {
  await browser.close();
  server.close();
}
