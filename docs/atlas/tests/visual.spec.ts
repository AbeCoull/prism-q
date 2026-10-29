import { expect, test, type Page } from "@playwright/test";

async function openAtlas(page: Page, query = "") {
  await page.goto(`/${query}`);
  await expect(
    page.getByRole("heading", {
      name: "Applying a one-qubit gate",
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: "Amplitude pairs and SIMD registers",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.getByTestId("workspace")).toBeVisible();
}

async function expectNoHorizontalOverflow(page: Page) {
  const dimensions = await page.evaluate(() => ({
    page: document.documentElement.scrollWidth,
    viewport: document.documentElement.clientWidth,
  }));
  expect(dimensions.page).toBeLessThanOrEqual(dimensions.viewport + 1);
}

test("controls change amplitude-pair addresses and arithmetic route", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openAtlas(page);
  await expect(
    page.getByRole("slider", { name: "Qubit count", exact: true }),
  ).toHaveValue("4");
  await expect(
    page.getByRole("combobox", { name: "Target qubit", exact: true }),
  ).toHaveValue("2");
  await expect(page.getByTestId("pair-offset")).toContainText("64");
  await expect(page.getByTestId("norm")).toContainText(/1\.0+/);

  await page
    .getByRole("combobox", { name: "Target qubit", exact: true })
    .selectOption("0");
  await expect(page.getByTestId("pair-offset")).toContainText("16");
  await expect(
    page.getByText("128-bit FMA", { exact: true }).first(),
  ).toBeVisible();
  await page
    .getByRole("combobox", { name: "Target qubit", exact: true })
    .selectOption("3");
  await expect(page.getByTestId("pair-offset")).toContainText("128");
  await page
    .getByRole("combobox", { name: "Architecture", exact: true })
    .selectOption("neon");
  await expect(
    page.getByText("AArch64 NEON", { exact: true }).first(),
  ).toBeVisible();
  await page
    .getByRole("combobox", { name: "Architecture", exact: true })
    .selectOption("scalar");
  await expect(
    page.getByText("Scalar complex128", { exact: true }).first(),
  ).toBeVisible();

  await page
    .getByRole("combobox", { name: "Gate", exact: true })
    .selectOption("Ry");
  const depth = page.getByRole("slider", {
    name: "Circuit depth",
    exact: true,
  });
  await depth.focus();
  await depth.press("End");
  await expect(depth).toHaveValue("4");
  await expect(page.getByTestId("norm")).toContainText(/1\.0+/);
  await expectNoHorizontalOverflow(page);
  expect(errors).toEqual([]);
});

test("execution stages can be traversed in both directions with keyboard input", async ({
  page,
}) => {
  await openAtlas(page);
  const next = page.getByRole("button", { name: "Next stage", exact: true });
  const previous = page.getByRole("button", {
    name: "Previous stage",
    exact: true,
  });
  await expect(previous).toBeDisabled();
  await next.focus();
  await next.press("Enter");
  await expect(previous).toBeEnabled();
  await previous.click();
  await expect(previous).toBeDisabled();
  for (let step = 0; step < 20 && (await next.isEnabled()); step++)
    await next.click();
  await expect(next).toBeDisabled();
  await expect(previous).toBeEnabled();
});

test("source evidence is available as a keyboard-dismissable dialog", async ({
  page,
}) => {
  await openAtlas(page);
  const trigger = page.getByRole("button", {
    name: "Source evidence",
    exact: true,
  });
  await trigger.click();
  const drawer = page.getByRole("dialog");
  await expect(drawer).toBeVisible();
  await expect(drawer.getByRole("link").first()).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(drawer).not.toBeVisible();
  await expect(trigger).toBeFocused();
});

test("native validation and share controls remain available", async ({
  page,
}) => {
  await openAtlas(page);
  const error = page.getByTestId("native-error");
  await expect(error).toBeVisible();
  await expect(error).toContainText(/0|[eE]-\d+/);
  await page.evaluate(() =>
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (text: string) => {
          document.documentElement.dataset.sharedExperiment = text;
        },
      },
    }),
  );
  await page
    .getByRole("combobox", { name: "Gate", exact: true })
    .selectOption("Rx");
  await page
    .getByRole("button", { name: "Copy experiment link", exact: true })
    .click();
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.dataset.sharedExperiment ?? "",
      ),
    )
    .toContain("http://127.0.0.1:4173");
  const shared = await page.evaluate(
    () => document.documentElement.dataset.sharedExperiment!,
  );
  await page.goto(shared);
  await expect(
    page.getByRole("combobox", { name: "Gate", exact: true }),
  ).toHaveValue("Rx");
});

test("mobile controls fit a 390-pixel viewport and respect reduced motion", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openAtlas(page);
  expect(
    await page.evaluate(
      () => matchMedia("(prefers-reduced-motion: reduce)").matches,
    ),
  ).toBe(true);
  await expectNoHorizontalOverflow(page);
  const qubits = page.getByRole("slider", { name: "Qubit count", exact: true });
  await qubits.focus();
  await qubits.press("Home");
  await expect(qubits).toHaveValue("2");
  await qubits.press("ArrowRight");
  await expect(qubits).toHaveValue("3");
  const target = page.getByRole("combobox", {
    name: "Target qubit",
    exact: true,
  });
  await target.selectOption("2");
  await expect(page.getByTestId("pair-offset")).toContainText("64");
  await expect(page.getByTestId("norm")).toContainText(/1\.0+/);
  await expectNoHorizontalOverflow(page);
});

test("SVG fallback preserves scientific controls without a WebGL canvas", async ({
  page,
}) => {
  await openAtlas(page, "?renderer=svg");
  const workspace = page.getByTestId("workspace");
  await expect(workspace.locator("svg").first()).toBeVisible();
  await expect(workspace.locator("canvas")).toHaveCount(0);
  await page
    .getByRole("combobox", { name: "Target qubit", exact: true })
    .selectOption("1");
  await expect(page.getByTestId("pair-offset")).toContainText("32");
  await expect(page.getByTestId("norm")).toContainText(/1\.0+/);
});

test("unavailable WebGL automatically falls back to the interactive SVG", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const prototype = HTMLCanvasElement.prototype;
    const original = prototype.getContext;
    prototype.getContext = function (kind: string, ...args: unknown[]) {
      if (
        kind === "webgl" ||
        kind === "webgl2" ||
        kind === "experimental-webgl"
      )
        return null;
      return original.apply(this, [kind, ...args] as Parameters<
        typeof original
      >);
    } as typeof original;
  });
  await openAtlas(page);
  await expect(page.locator('[data-renderer="anatomy-svg"]')).toBeVisible();
  await page
    .getByRole("combobox", { name: "Target qubit", exact: true })
    .selectOption("1");
  await expect(page.getByTestId("pair-offset")).toContainText("32");
});

test("diagonal gates and unrecorded examples retain their evidence boundaries", async ({
  page,
}) => {
  await openAtlas(page);
  await page
    .getByRole("combobox", { name: "Gate", exact: true })
    .selectOption("Rz");
  await expect(
    page.getByText("128-bit FMA diagonal", { exact: true }).first(),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Hardware" }).click();
  await expect(
    page
      .getByText(
        "Each amplitude is scaled by its own entry, with no cross-row sum.",
        {
          exact: false,
        },
      )
      .first(),
  ).toBeVisible();
  const angle = page.getByRole("slider", {
    name: "Rotation angle",
    exact: true,
  });
  await angle.focus();
  await angle.press("Home");
  await expect(page.getByText("Browser model", { exact: true })).toBeVisible();
  await expect(page.getByTestId("native-error")).toContainText(
    "Not checked against native",
  );
});

test.describe("reviewed visual baselines", () => {
  test.skip(
    process.env.ATLAS_VISUAL !== "1",
    "Set ATLAS_VISUAL=1 to compare reviewed browser and platform baselines.",
  );

  test("desktop scientific workspace", async ({ page }) => {
    await openAtlas(page);
    await page.evaluate(() => document.fonts.ready);
    await expect(page.getByTestId("workspace")).toHaveScreenshot(
      "desktop-workspace.png",
    );
  });

  test("mobile scientific workspace", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await openAtlas(page, "?renderer=svg");
    await page.evaluate(() => document.fonts.ready);
    await expect(page.getByTestId("workspace")).toHaveScreenshot(
      "mobile-workspace.png",
    );
  });

  test("deterministic SVG workspace", async ({ page }) => {
    await openAtlas(page, "?renderer=svg");
    await page.evaluate(() => document.fonts.ready);
    await expect(page.getByTestId("workspace")).toHaveScreenshot(
      "svg-workspace.png",
    );
  });
});
