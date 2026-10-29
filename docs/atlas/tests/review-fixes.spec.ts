import { expect, test } from "@playwright/test";

test("Z bypasses the low input and scalar evidence links to scalar code", async ({
  page,
}) => {
  await page.goto("/?renderer=svg&gate=Z&n=4&q=2&stage=1&arch=scalar");
  const scene = page.locator('[data-renderer="anatomy-svg"]');
  await expect(scene).toHaveAttribute("aria-label", /Low half bypassed/);
  await expect(scene.locator('[data-execution="loaded"]')).toHaveCount(1);
  await expect(scene.locator('[data-execution="bypassed"]')).toHaveCount(1);
  const microscope = page.getByRole("region", {
    name: "Step through the kernel",
    exact: true,
  });
  await expect(microscope.locator(".micro-source a")).toHaveAttribute(
    "href",
    /#L922-L945$/,
  );
  await page.getByRole("button", { name: "Next stage", exact: true }).click();
  await page.getByRole("button", { name: "Next stage", exact: true }).click();
  await expect(scene.getByTestId("memory-store-marker")).toHaveCount(1);
  await page
    .getByRole("combobox", { name: "Gate", exact: true })
    .selectOption("H");
  await expect(scene.getByTestId("memory-store-marker")).toHaveCount(2);
  await expect(scene.locator('[data-execution="bypassed"]')).toHaveCount(0);
});

test("compiler selectors preserve width and unrelated variables", async ({
  page,
}) => {
  await page.goto("/?renderer=svg&compiler=diagonal-pair-runs-20-Host&pass=11");
  const lab = page.getByTestId("compiler-laboratory");
  const profile = lab.getByRole("combobox", {
    name: "Compiler profile",
    exact: true,
  });
  await expect(profile).toHaveValue("Host");
  await expect(
    profile.locator('option[value="Device"], option', { hasText: /^Device$/ }),
  ).toHaveAttribute("disabled", "");
  await expect(
    lab
      .getByRole("combobox", { name: "Compiler circuit", exact: true })
      .locator('option[value="legal-disjoint"]'),
  ).toHaveAttribute("disabled", "");
  const width = lab.getByRole("combobox", {
    name: "Compiler width",
    exact: true,
  });
  await expect(width).toHaveValue("20");
  await width.selectOption("14");
  await profile.selectOption("Device");
  await expect(width).toHaveValue("14");
  await expect(
    lab.getByRole("combobox", { name: "Compiler circuit", exact: true }),
  ).toHaveValue("diagonal-pair-runs");
});

test("compiler changes and keyboard traversal keep selection visible", async ({
  page,
}) => {
  await page.goto(
    "/?renderer=svg&compiler=diagonal-pair-runs-14-Host&pass=0#compiler-laboratory",
  );
  const lab = page.getByTestId("compiler-laboratory");
  const graph = lab.getByRole("region", {
    name: "Native circuit stream, horizontally scrollable",
    exact: true,
  });
  await graph.locator('[data-instruction="0"]').waitFor();
  await graph.evaluate((element) => (element.scrollLeft = 5000));
  await lab
    .getByRole("button", { name: "Next compiler pass", exact: true })
    .click();
  await expect
    .poll(() => graph.evaluate((element) => element.scrollLeft))
    .toBeLessThan(100);
  const first = graph.locator('[data-instruction="0"]');
  await first.focus();
  await first.press("End");
  const selected = graph.locator('[aria-pressed="true"]');
  await expect(selected).toBeFocused();
  await expect
    .poll(async () => {
      const node = await selected.boundingBox(),
        viewport = await graph.boundingBox();
      return (
        !!node &&
        !!viewport &&
        node.x >= viewport.x &&
        node.x + node.width <= viewport.x + viewport.width + 1
      );
    })
    .toBe(true);
  await expect(graph.locator('[tabindex="0"]')).toHaveCount(1);
});

test("every chapter stays reachable on mobile", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/?renderer=svg");
  const methods = page.locator("#methodology");
  await expect(methods.locator(".chapter-list button")).toHaveCount(9);
  await methods
    .getByRole("button", { name: "MPS and entanglement complexity" })
    .click();
  await expect(page.getByRole("dialog")).toContainText("MPS");
  await page.keyboard.press("Escape");
  const nav = page.getByRole("navigation", { name: "Primary" });
  await nav.getByRole("link", { name: "Fusion", exact: true }).click();
  await expect(page.getByTestId("compiler-laboratory")).toBeInViewport();
  await nav.getByRole("link", { name: "Gates", exact: true }).click();
  await expect(page.getByTestId("workspace")).toBeInViewport();
});

test("scientific text stays readable without mobile page overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/?renderer=svg");
  await page.locator(".operator-geometry-legend").waitFor();
  for (const selector of [
    ".micro-lane small",
    ".micro-lane span",
    ".operator-geometry-legend",
    ".compiler-metrics span",
  ])
    expect(
      await page
        .locator(selector)
        .first()
        .evaluate((element) => parseFloat(getComputedStyle(element).fontSize)),
    ).toBeGreaterThanOrEqual(11);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth - innerWidth,
    ),
  ).toBeLessThanOrEqual(1);
  const contrast = await page.evaluate(() => {
    const style = getComputedStyle(document.documentElement);
    const luminance = (hex: string) => {
      const [r, g, b] = [1, 3, 5].map((at) => {
        const v = parseInt(hex.slice(at, at + 2), 16) / 255;
        return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const faint = luminance(style.getPropertyValue("--faint").trim());
    const panel = luminance(style.getPropertyValue("--panel").trim());
    return (Math.max(faint, panel) + 0.05) / (Math.min(faint, panel) + 0.05);
  });
  expect(contrast).toBeGreaterThanOrEqual(4.5);
  await expect(page.locator(".micro-register.is-pending").first()).toHaveCSS(
    "opacity",
    "1",
  );
});

test("before and after show complete native streams without inventing ancestry", async ({
  page,
}) => {
  await page.goto(
    "/?renderer=svg&compiler=legal-disjoint-8-Host&pass=2#compiler-laboratory",
  );
  const lab = page.getByTestId("compiler-laboratory");
  await lab
    .getByRole("button", { name: "Before / after", exact: true })
    .click();
  const before = lab.getByRole("region", {
    name: "Previous native stream",
    exact: true,
  });
  await expect(before).toBeVisible();
  await expect(before.getByRole("button")).toHaveCount(0);
  expect(await before.locator("svg title").allTextContents()).toContain(
    "Control q4, target q5",
  );
  await expect(before.locator(".compiler-diagram")).toHaveAttribute(
    "tabindex",
    "0",
  );
  await expect(lab).toContainText(
    "an instruction in the same position is not necessarily derived from the one before",
  );
  await lab
    .getByRole("button", { name: "Before / after", exact: true })
    .click();
  await expect(before).toHaveCount(0);
});

test("shared links keep the selected instruction and both laboratories", async ({
  page,
}) => {
  await page.goto(
    "/?renderer=svg&compiler=diagonal-pair-runs-12-Host&pass=17&inst=3&view=graph#compiler-laboratory",
  );
  const lab = page.getByTestId("compiler-laboratory");
  await expect(lab).toContainText("Selected instruction 3");
  await page.evaluate(() =>
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (text: string) => {
          document.documentElement.dataset.shared = text;
        },
      },
    }),
  );
  const shared = async (name: string) => {
    await page.evaluate(() => delete document.documentElement.dataset.shared);
    await page.getByRole("button", { name, exact: true }).click();
    await expect
      .poll(() =>
        page.evaluate(() => document.documentElement.dataset.shared ?? ""),
      )
      .not.toBe("");
    return new URL(
      await page.evaluate(() => document.documentElement.dataset.shared!),
    ).searchParams;
  };
  const compiler = await shared("Copy compiler experiment link");
  expect(compiler.get("inst")).toBe("3");
  expect(compiler.get("view")).toBe("graph");
  const gate = await shared("Copy experiment link");
  expect(gate.get("compiler")).toBe("diagonal-pair-runs-12-Host");
  expect(gate.get("inst")).toBe("3");
  expect(gate.get("renderer")).toBe("svg");
});
