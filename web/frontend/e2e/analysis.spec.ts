import { expect, test } from "@playwright/test";


async function runAnalysis(page: import("@playwright/test").Page) {
  await page.getByRole("button", { name: "Run analysis" }).click();
  await expect(page.getByRole("heading", { name: "Pipeline running" })).toBeVisible();
  await expect(page.getByText("Workload Graph")).toBeVisible();
  await expect(page.getByText("Hybrid Operator IR")).toBeVisible();
  await expect(page.getByText("Cost IR")).toBeVisible();
  await expect(page.getByText("Performance", { exact: true })).toBeVisible();
}


test.describe.serial("SOLAR analysis lifecycle", () => {
  test("shows editing, real pipeline progress and completed evidence on desktop", async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto("/");

    await expect(page.getByRole("heading", { name: "New analysis" })).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath("desktop-editing.png"), fullPage: true });

    await runAnalysis(page);
    await page.screenshot({ path: testInfo.outputPath("desktop-running.png"), fullPage: true });

    await expect(page.getByRole("heading", { name: "Analysis complete" })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText("0.2049 µs")).toBeVisible();
    await expect(page.getByText("Memory bound")).toBeVisible();
    await expect(page.getByText("supported")).toBeVisible();

    await page.getByRole("button", { name: /Stage 2 Hybrid Operator IR/ }).click();
    await expect(page.getByRole("heading", { name: "Stage 2 · Hybrid Operator IR" })).toBeVisible();
    await expect(page.locator(".react-flow__node")).toHaveCount(3);
    await expect(page.locator(".react-flow__edge")).toHaveCount(2);
    await expect(page.getByText("hidden-tensor [32x256]", { exact: true })).toBeVisible();
    await expect(page.getByText("hidden-tensor_1 [32x128]", { exact: true })).toBeVisible();
    const graph = await page.locator(".graph-view").boundingBox();
    expect(graph?.width).toBeGreaterThan(300);
    expect(graph?.height).toBeGreaterThan(250);
    await page.screenshot({ path: testInfo.outputPath("desktop-completed.png"), fullPage: true });
  });

  test("keeps the complete workflow usable at mobile width", async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");

    await expect(page.getByRole("heading", { name: "New analysis" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Run analysis" })).toBeVisible();
    await runAnalysis(page);
    await expect(page.getByRole("heading", { name: "Analysis complete" })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByLabel("Four-stage pipeline")).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath("mobile-completed.png"), fullPage: true });
  });
});
