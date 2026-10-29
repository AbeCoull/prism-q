import { expect, test, type Locator, type Page } from "@playwright/test";

const revision = "7adde35d0bacaff71711148223ac8df06a008d01";
const plannerQuery =
  "?renderer=svg&compiler=diagonal-pair-runs-20-Host&pass=11";

async function openExpansion(page: Page, query = "?renderer=svg") {
  await page.goto(`/${query}`);
  await expect(
    page.getByRole("heading", {
      name: "The execution microscope",
      exact: true,
    }),
  ).toBeVisible();
  const laboratory = page.getByTestId("compiler-laboratory");
  await expect(
    laboratory.getByRole("combobox", { name: "Compiler circuit", exact: true }),
  ).toBeVisible();
  await expect(laboratory.getByRole("alert")).toHaveCount(0);
  return laboratory;
}

function microscope(page: Page) {
  return page.getByRole("region", {
    name: "The execution microscope",
    exact: true,
  });
}

async function endSlider(slider: Locator) {
  await slider.focus();
  await slider.press("End");
}

async function assertNoOverflow(page: Page) {
  const dimensions = await page.evaluate(() => ({
    page: document.documentElement.scrollWidth,
    viewport: document.documentElement.clientWidth,
  }));
  expect(dimensions.page).toBeLessThanOrEqual(dimensions.viewport + 1);
}

async function settleScreenshot(page: Page) {
  await page.evaluate(async () => {
    await document.fonts.ready;
    if (document.activeElement instanceof HTMLElement)
      document.activeElement.blur();
  });
}

test("microscope steps source groups, coefficients, pair addresses, and stored phases", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openExpansion(page);
  const scope = microscope(page);
  const slider = scope.getByRole("slider", {
    name: "Execution instruction group",
    exact: true,
  });
  await expect(slider).toHaveValue("0");
  await expect(
    scope.getByRole("button", { name: "Previous", exact: true }),
  ).toBeDisabled();
  await scope.getByRole("button", { name: "Next", exact: true }).click();
  await expect(slider).toHaveValue("1");
  await scope.getByRole("button", { name: "Previous", exact: true }).click();
  await expect(slider).toHaveValue("0");
  await scope.getByRole("button", { name: /^u11\b/ }).click();
  await expect(scope.getByRole("button", { name: /^u11\b/ })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(
    scope.getByRole("img", {
      name: "Complex amplitude 4, before and after the gate",
      exact: true,
    }),
  ).toBeVisible();
  await page
    .getByRole("spinbutton", { name: "Pair ordinal", exact: true })
    .fill("2");
  await expect(
    scope.getByRole("img", {
      name: "Complex amplitude 6, before and after the gate",
      exact: true,
    }),
  ).toBeVisible();
  await endSlider(slider);
  await expect(slider).toHaveValue("9");
  await expect(
    scope.getByRole("button", { name: "Next", exact: true }),
  ).toBeDisabled();
  await expect(scope.locator(".micro-register").last()).toHaveClass(/is-ready/);
  await expect(scope.locator(".micro-phase-readout")).toContainText(
    "probability",
  );
  await expect(scope.locator(".micro-phase-readout")).toContainText("phase");
  await expect(scope.locator(".micro-phase > p")).toContainText(
    "stored output",
  );
  await expect(scope).not.toContainText(/NaN|Infinity/);
  await expect(
    scope.getByRole("link", { name: /Read this intrinsic sequence/ }),
  ).toHaveAttribute(
    "href",
    new RegExp(`${revision}/src/backend/simd\\.rs#L258`),
  );
  await expect(scope).toContainText(
    "not a native instruction trace or disassembly",
  );
  expect(errors).toEqual([]);
});

test("microscope switches to NEON diagonal scaling and identity-half bypass", async ({
  page,
}) => {
  await openExpansion(page);
  await page
    .getByRole("combobox", { name: "Architecture", exact: true })
    .selectOption("neon");
  await page
    .getByRole("combobox", { name: "Gate", exact: true })
    .selectOption("Rz");
  const scope = microscope(page);
  await expect(scope.locator(".micro-route")).toContainText("128-bit NEON");
  await expect(scope.locator(".micro-coefficients button")).toHaveCount(2);
  await expect(scope.getByRole("button", { name: /^d1\b/ })).toBeVisible();
  await expect(scope.locator(".micro-timeline li")).toHaveCount(8);
  await scope
    .getByRole("button", { name: /Finish complex multiplication/ })
    .click();
  await expect(scope.locator(".micro-step-heading code")).toContainText(
    "vfmaq_f64",
  );
  await expect(
    scope
      .locator(".micro-register")
      .filter({ hasText: "c.im signed broadcast" }),
  ).toBeVisible();
  await expect(scope).not.toContainText("two-term row sum");
  await page
    .getByRole("combobox", { name: "Gate", exact: true })
    .selectOption("Z");
  await scope.getByRole("button", { name: /^d0\b/ }).click();
  await expect(
    scope.getByText("Low half bypassed", { exact: true }),
  ).toBeVisible();
  await expect(scope.locator(".micro-bypass")).toContainText(
    "no low-half load, arithmetic, or store",
  );
  await expect(scope).not.toContainText(/NaN|Infinity/);
});

test("native compiler width and profile controls reach the measured cache planner", async ({
  page,
}) => {
  const lab = await openExpansion(page);
  await lab
    .getByRole("combobox", { name: "Compiler circuit", exact: true })
    .selectOption("diagonal-pair-runs");
  await lab
    .getByRole("combobox", { name: "Compiler profile", exact: true })
    .selectOption("Host");
  await lab
    .getByRole("combobox", { name: "Compiler width", exact: true })
    .selectOption("20");
  await expect(lab.locator(".compiler-stage-heading h3")).toHaveText(
    "Plan cache stages",
  );
  await expect(lab.getByTestId("compiler-sweeps")).toHaveText(/150\s*→\s*8/);
  await expect(
    lab.locator(".compiler-metrics > div").first().locator("strong"),
  ).toHaveText(/150\s*→\s*8/);
  await expect(
    lab.getByRole("slider", { name: "Compiler pass", exact: true }),
  ).toHaveValue("11");
  await expect(
    lab.getByRole("button", { name: "Next compiler pass", exact: true }),
  ).toBeDisabled();
  await expect(lab.locator(".compiler-source a")).toHaveAttribute(
    "href",
    `https://github.com/AbeCoull/prism-q/blob/${revision}/src/circuit/fusion_stage.rs`,
  );
  await expect(lab.getByTestId("compiler-error")).toContainText(
    /e-1[3-9]|0\.00e\+0/,
  );
  await lab
    .getByRole("button", { name: "Previous compiler pass", exact: true })
    .click();
  await expect(lab.getByTestId("compiler-sweeps")).toHaveText(/150\s*→\s*41/);
  await expect(lab.locator(".compiler-stage-heading h3")).toHaveText(
    "Fuse same-pair blocks",
  );
  await lab
    .getByRole("button", { name: "Next compiler pass", exact: true })
    .click();
  await expect(lab.getByTestId("compiler-sweeps")).toHaveText(/150\s*→\s*8/);
  await expect(lab).toContainText("NATIVE COUNTER · NOT TIMING");
  await expect(lab).toContainText("A COUNT IS NOT A BANDWIDTH MEASUREMENT");
});

test("dependency graph selection and matrix cells expose recorded operator values", async ({
  page,
}) => {
  const lab = await openExpansion(page, plannerQuery);
  await lab.getByRole("button", { name: "Dependencies", exact: true }).click();
  const dag = lab.getByRole("region", {
    name: "Qubit-order dependency graph, horizontally scrollable",
    exact: true,
  });
  const node = dag.getByRole("button", { name: /^Inspect instruction 1:/ });
  await node.focus();
  await node.press("Enter");
  await expect(node).toHaveAttribute("aria-pressed", "true");
  await expect(lab.locator(".operator-inspector-title h3")).toContainText(
    "Instruction 1:",
  );
  const cells = lab.getByRole("group", {
    name: "Native complex matrix coefficients",
    exact: true,
  });
  const original = await lab
    .locator(".coefficient-detail > code")
    .textContent();
  await cells
    .getByRole("button", { name: /^Coefficient row 0 column 1:/ })
    .click();
  await expect(lab.locator(".coefficient-detail > span")).toHaveText("u[0,1]");
  await expect(lab.locator(".coefficient-detail > code")).not.toHaveText(
    original!,
  );
  await expect(
    cells.getByRole("button", { name: /^Coefficient row 0 column 1:/ }),
  ).toHaveClass(/selected/);
  await expect(lab.locator(".coefficient-detail")).toContainText("Magnitude");
  await expect(lab.locator(".coefficient-detail")).toContainText("Phase");
  const imaginaryValues = await cells.locator(".matrix-imag").allTextContents();
  expect(imaginaryValues.some((value) => value.startsWith("-"))).toBe(true);
  for (const value of imaginaryValues) expect(value).toMatch(/^[+-]\d+\.\d+i$/);
  await expect(lab).not.toContainText(/NaN|Infinity/);
});

test("compiler sharing preserves the record and selected pass", async ({
  page,
}) => {
  const lab = await openExpansion(page, plannerQuery);
  await lab
    .getByRole("button", { name: "Previous compiler pass", exact: true })
    .click();
  await page.evaluate(() =>
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (text: string) => {
          document.documentElement.dataset.sharedCompiler = text;
        },
      },
    }),
  );
  await lab
    .getByRole("button", { name: "Copy compiler experiment link", exact: true })
    .click();
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.dataset.sharedCompiler ?? "",
      ),
    )
    .toContain("compiler=diagonal-pair-runs-20-Host");
  const shared = await page.evaluate(
    () => document.documentElement.dataset.sharedCompiler!,
  );
  expect(new URL(shared).searchParams.get("pass")).toBe("10");
  expect(new URL(shared).hash).toBe("#compiler-laboratory");
  await page.goto(shared);
  await expect(
    lab.getByRole("combobox", { name: "Compiler width", exact: true }),
  ).toHaveValue("20");
  await expect(
    lab.getByRole("combobox", { name: "Compiler profile", exact: true }),
  ).toHaveValue("Host");
  await expect(
    lab.getByRole("slider", { name: "Compiler pass", exact: true }),
  ).toHaveValue("10");
  await expect(lab.getByTestId("compiler-sweeps")).toHaveText(/150\s*→\s*41/);
});

test("rewrite challenge exposes a basis counterexample and the favorable plus input", async ({
  page,
}) => {
  await openExpansion(page);
  const challenge = page.getByRole("region", {
    name: "Can the two X gates disappear?",
    exact: true,
  });
  await expect(challenge.getByTestId("rewrite-fidelity")).toHaveText(
    "0.000000",
  );
  await challenge.getByRole("button", { name: /Remove both X gates/ }).click();
  await expect(challenge).toContainText("ILLEGAL REWRITE");
  await expect(challenge.getByTestId("rewrite-fidelity")).toHaveText(
    "0.000000",
  );
  await challenge.getByRole("button", { name: /^X eigenstate/ }).click();
  await expect(challenge.getByTestId("rewrite-fidelity")).toHaveText(
    "1.000000",
  );
  await expect(challenge).toContainText("This input hides the error");
  await challenge.getByRole("button", { name: /^Basis/ }).click();
  await expect(challenge.getByTestId("rewrite-fidelity")).toHaveText(
    "0.000000",
  );
  await expect(challenge).toContainText(
    "Normalization alone cannot establish circuit equivalence",
  );
});

test("new scientific sections fit a 390-pixel mobile viewport", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const lab = await openExpansion(page, plannerQuery);
  await expect
    .poll(() =>
      page.evaluate(
        () => matchMedia("(prefers-reduced-motion: reduce)").matches,
      ),
    )
    .toBe(true);
  await microscope(page).scrollIntoViewIfNeeded();
  await assertNoOverflow(page);
  await microscope(page)
    .getByRole("button", { name: /^u01\b/ })
    .click();
  await endSlider(
    microscope(page).getByRole("slider", {
      name: "Execution instruction group",
      exact: true,
    }),
  );
  await assertNoOverflow(page);
  await lab.getByRole("button", { name: "Dependencies", exact: true }).click();
  await lab.locator(".operator-inspector").scrollIntoViewIfNeeded();
  await expect(
    lab.getByRole("group", {
      name: "Native complex matrix coefficients",
      exact: true,
    }),
  ).toBeVisible();
  await assertNoOverflow(page);
  await lab.locator(".rewrite-challenge").scrollIntoViewIfNeeded();
  await assertNoOverflow(page);
});

test("operator matrix remains interactive when real WebGL creation is unavailable", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (
      kind: string,
      ...args: unknown[]
    ) {
      if (["webgl", "webgl2", "experimental-webgl"].includes(kind)) return null;
      return original.apply(this, [kind, ...args] as Parameters<
        typeof original
      >);
    } as typeof original;
  });
  const lab = await openExpansion(page, plannerQuery);
  await expect(
    lab.getByText(
      "WebGL is unavailable. Every coefficient remains accessible in the matrix table.",
      { exact: true },
    ),
  ).toBeVisible();
  await lab
    .getByRole("button", { name: /^Coefficient row 0 column 1:/ })
    .click();
  await expect(lab.locator(".coefficient-detail > span")).toHaveText("u[0,1]");
  await expect(lab.locator(".coefficient-detail")).not.toContainText(
    /NaN|Infinity/,
  );
  await assertNoOverflow(page);
});

test.describe("reviewed expansion visual baselines", () => {
  test.skip(
    process.env.ATLAS_VISUAL !== "1",
    "Set ATLAS_VISUAL=1 to compare reviewed expansion screenshots.",
  );

  test("execution microscope at the store group", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1600 });
    await openExpansion(page);
    await page
      .getByRole("combobox", { name: "Gate", exact: true })
      .selectOption("Rx");
    await microscope(page)
      .getByRole("button", { name: /^u01\b/ })
      .click();
    await expect(
      page.getByRole("combobox", { name: "Gate", exact: true }),
    ).toHaveValue("Rx");
    await expect(
      microscope(page).getByRole("button", { name: /^u01\b/ }),
    ).toHaveAttribute("aria-pressed", "true");
    await endSlider(
      microscope(page).getByRole("slider", {
        name: "Execution instruction group",
        exact: true,
      }),
    );
    await settleScreenshot(page);
    await expect(microscope(page)).toHaveScreenshot("execution-microscope.png");
  });

  test("native matrix inspector with deterministic table view", async ({
    page,
  }) => {
    const lab = await openExpansion(page, plannerQuery);
    await lab
      .getByRole("button", { name: "Use matrix table", exact: true })
      .click();
    await lab
      .getByRole("button", { name: /^Coefficient row 0 column 1:/ })
      .click();
    await settleScreenshot(page);
    await expect(lab.locator(".operator-inspector")).toHaveScreenshot(
      "native-operator-inspector.png",
    );
  });

  test("rewrite challenge counterexample on mobile", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 1200 });
    const lab = await openExpansion(page, plannerQuery);
    await lab.getByRole("button", { name: /Remove both X gates/ }).click();
    await settleScreenshot(page);
    await expect(
      page.getByRole("link", { name: "Skip to experiment", exact: true }),
    ).not.toBeFocused();
    await lab
      .locator(".rewrite-challenge")
      .evaluate((element) =>
        element.scrollIntoView({ block: "center", behavior: "instant" }),
      );
    await expect(lab.locator(".rewrite-challenge")).toHaveScreenshot(
      "rewrite-challenge-mobile.png",
    );
  });
});
