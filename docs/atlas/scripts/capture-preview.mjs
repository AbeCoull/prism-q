import { existsSync } from "node:fs";
import { mkdir, stat, writeFile } from "node:fs/promises";
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
const failures = [];
const checks = [];
const context = await browser.newContext({
  viewport: { width: 1440, height: 1000 },
  deviceScaleFactor: 1,
  reducedMotion: "reduce",
  colorScheme: "dark",
  locale: "en-US",
  timezoneId: "America/New_York",
});
const page = await context.newPage();
page.on("pageerror", (error) => failures.push(error.message));

async function open(query = "") {
  await page.goto(`${base}/${query}`, { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  await page.getByTestId("workspace").waitFor();
  await page.waitForTimeout(300);
}

async function inspect(label) {
  checks.push({
    label,
    ...(await page.evaluate(() => ({
      viewport: innerWidth,
      scrollWidth: document.documentElement.scrollWidth,
      reducedMotion: matchMedia("(prefers-reduced-motion: reduce)").matches,
      renderer: document.querySelector(".scene-canvas canvas")
        ? "webgl"
        : "svg",
    }))),
  });
}

try {
  await open();
  await inspect("desktop 1440 by 1000");
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.screenshot({
    path: resolve(destination, "statevector-desktop.png"),
    animations: "disabled",
    clip: { x: 0, y: 0, width: 1440, height: 1100 },
  });

  await open("?n=4&q=2&gate=H&stage=2");
  await page
    .getByTestId("workspace")
    .screenshot({
      path: resolve(destination, "statevector-closeup.png"),
      animations: "disabled",
    });
  await open("?renderer=svg&stage=1");
  await page
    .getByTestId("workspace")
    .screenshot({
      path: resolve(destination, "statevector-svg.png"),
      animations: "disabled",
    });
  await inspect("desktop SVG fallback");

  await page.setViewportSize({ width: 390, height: 844 });
  await open();
  await inspect("mobile 390 by 844");
  await page.setViewportSize({ width: 390, height: 1500 });
  await page.screenshot({
    path: resolve(destination, "statevector-mobile.png"),
    animations: "disabled",
    clip: { x: 0, y: 0, width: 390, height: 1500 },
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await open("?renderer=svg&stage=1");
  await inspect("mobile SVG fallback");
  await context.close();

  const videoContext = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    reducedMotion: "reduce",
    colorScheme: "dark",
    recordVideo: {
      dir: resolve(atlas, ".cache/preview-video"),
      size: { width: 1440, height: 1000 },
    },
  });
  const videoPage = await videoContext.newPage();
  await videoPage.goto(`${base}/?n=4&q=2&gate=H`, { waitUntil: "networkidle" });
  await videoPage.evaluate(() => document.fonts.ready);
  await videoPage.getByTestId("workspace").scrollIntoViewIfNeeded();
  for (let stage = 0; stage < 5; stage += 1) {
    if (stage > 0)
      await videoPage
        .getByRole("button", { name: "Next stage", exact: true })
        .click();
    await videoPage.waitForTimeout(1200);
  }
  const video = videoPage.video();
  await videoContext.close();
  if (!video) throw new Error("The browser did not create a video artifact.");
  await video.saveAs(resolve(destination, "statevector.webm"));
  const videoBytes = (await stat(resolve(destination, "statevector.webm")))
    .size;
  const playback = await browser.newPage();
  await playback.setContent(
    `<video src="${new URL("/preview/statevector.webm", base)}" preload="metadata"></video>`,
  );
  await playback.locator("video").evaluate(
    (element) =>
      new Promise((resolve, reject) => {
        element.onloadedmetadata = resolve;
        element.onerror = reject;
        if (element.readyState >= 1) resolve();
      }),
  );
  const videoMetadata = await playback
    .locator("video")
    .evaluate((element) => ({
      duration: element.duration,
      width: element.videoWidth,
      height: element.videoHeight,
    }));
  await playback.close();
  const report = {
    browser: edge ? "Microsoft Edge" : "Chromium",
    base,
    checks,
    pageErrors: failures,
    videoBytes,
    video: videoMetadata,
    capture: "Five manual execution stages. No native timing is recorded.",
  };
  await writeFile(
    resolve(destination, "capture-report.json"),
    `${JSON.stringify(report, null, 2)}\n`,
  );
  console.log(JSON.stringify(report));
} finally {
  await browser.close();
}
