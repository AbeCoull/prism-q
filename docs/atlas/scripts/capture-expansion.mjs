import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { chromium } from "@playwright/test";

const atlas = fileURLToPath(new URL("..", import.meta.url));
const destination = resolve(atlas, "public/preview");
const base = process.argv[2] ?? "http://127.0.0.1:4173";
const edge =
  process.platform === "win32" &&
  existsSync("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe");
const browser = await chromium.launch({
  headless: true,
  ...(edge ? { channel: "msedge" } : {}),
});
await mkdir(destination, { recursive: true });
const context = await browser.newContext({
  viewport: { width: 1440, height: 1500 },
  deviceScaleFactor: 1,
  reducedMotion: "reduce",
  colorScheme: "dark",
  locale: "en-US",
  timezoneId: "America/New_York",
});
const page = await context.newPage();
const pageErrors = [];
page.on("pageerror", (error) => pageErrors.push(error.message));
async function open(query) {
  await page.goto(`${base}/${query}`, { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  await page
    .getByRole("combobox", { name: "Compiler circuit", exact: true })
    .waitFor();
}
async function capture(locator, filename) {
  await locator.scrollIntoViewIfNeeded();
  await page.evaluate(() => {
    if (document.activeElement instanceof HTMLElement)
      document.activeElement.blur();
  });
  await locator.screenshot({
    path: resolve(destination, filename),
    animations: "disabled",
  });
}
try {
  await open("?renderer=svg&n=4&q=2&gate=Rx");
  const microscope = page.getByRole("region", {
    name: "Step through the kernel",
    exact: true,
  });
  await microscope.getByRole("button", { name: /^u01\b/ }).click();
  const slider = microscope.getByRole("slider", {
    name: "Execution instruction group",
  });
  await slider.focus();
  await slider.press("End");
  await capture(microscope, "execution-microscope.png");

  await open("?compiler=diagonal-pair-runs-20-Host&pass=11");
  const laboratory = page.getByTestId("compiler-laboratory");
  await laboratory.evaluate((element) =>
    element.scrollIntoView({ block: "start" }),
  );
  await page.waitForTimeout(350);
  const top = await laboratory.boundingBox();
  await page.screenshot({
    path: resolve(destination, "compiler-laboratory.png"),
    animations: "disabled",
    clip: { x: top.x, y: Math.max(0, top.y), width: top.width, height: 1100 },
  });
  const inspector = laboratory.locator(".operator-inspector");
  await inspector.scrollIntoViewIfNeeded();
  await inspector.locator("canvas").waitFor();
  await page.waitForTimeout(750);
  await capture(inspector, "operator-landscape.png");
  const operatorRenderer = (await inspector.locator("canvas").count())
    ? "webgl"
    : "table";

  await page.setViewportSize({ width: 390, height: 1200 });
  await open("?renderer=svg");
  const challenge = page.getByRole("region", {
    name: "Can the two X gates disappear?",
    exact: true,
  });
  await challenge.getByRole("button", { name: /Remove both X gates/ }).click();
  await capture(challenge, "rewrite-challenge-mobile.png");
  const report = {
    browser: edge ? "Microsoft Edge" : "Chromium",
    base,
    pageErrors,
    operatorRenderer,
    pageWidth: await page.evaluate(() => document.documentElement.scrollWidth),
    viewportWidth: 390,
    katexErrors: await page.locator(".katex-error").count(),
    captures: [
      "execution-microscope.png",
      "compiler-laboratory.png",
      "operator-landscape.png",
      "rewrite-challenge-mobile.png",
    ],
    meaning:
      "Native compiler replay and source-derived arithmetic. No runtime measurement.",
  };
  if (pageErrors.length || report.katexErrors || report.pageWidth > 391)
    throw new Error(JSON.stringify(report));
  await writeFile(
    resolve(destination, "expansion-capture-report.json"),
    `${JSON.stringify(report, null, 2)}\n`,
  );
  console.log(JSON.stringify(report));
} finally {
  await browser.close();
}
