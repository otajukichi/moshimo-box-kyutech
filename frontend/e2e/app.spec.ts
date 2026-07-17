import { expect, test } from "@playwright/test";

async function waitForTitle(page: import("@playwright/test").Page) {
  await page.goto("./");
  await expect(page.getByRole("button", { name: "始める" })).toBeVisible();
}

async function expectNoHorizontalOverflow(
  page: import("@playwright/test").Page
) {
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth
  );
  expect(overflow).toBeLessThanOrEqual(1);
}

test.beforeEach(async ({ request }) => {
  await request.post("api/control/reset");
});

test("preparation screen inherits the title scene without a bitmap background", async ({
  page
}, testInfo) => {
  const preparation = {
    state: "loading",
    message: "インタビューに必要なAIを準備しています",
    retry_available: true,
    groups: [
      {
        group: "interview",
        state: "loading",
        error_code: null,
        roles: [
          {
            role: "audio_preprocess_worker",
            group: "interview",
            state: "ready",
            model_id: "foundation/unimplemented",
            backend: "stub",
            progress: 1,
            message: "準備完了",
            load_time_ms: 40,
            processing_time_ms: null,
            peak_vram_mb: null,
            peak_cpu_memory_mb: null,
            error_code: null
          },
          {
            role: "streaming_asr_worker",
            group: "interview",
            state: "loading",
            model_id: "kotoba-tech/kotoba-whisper-v2.0-faster",
            backend: "faster-whisper",
            progress: 0.4,
            message: "モデルを読み込んでいます",
            load_time_ms: null,
            processing_time_ms: null,
            peak_vram_mb: null,
            peak_cpu_memory_mb: null,
            error_code: null
          }
        ]
      }
    ]
  };
  await page.route("**/api/runtime/status", async (route) => {
    await route.fulfill({ json: { preparation } });
  });
  await page.route("**/api/session/current", async (route) => {
    await route.fulfill({ json: { session: null } });
  });

  await page.goto("./");
  await expect(
    page.getByRole("heading", { name: /もしもボックス.*九工大出張所/ })
  ).toBeVisible();
  await expect(page.getByText("インタビューに必要なAIを準備しています")).toBeVisible();
  const backgroundImage = await page.locator(".preparation-page").evaluate(
    (element) => getComputedStyle(element).backgroundImage
  );
  expect(backgroundImage).toBe("none");
  await expectNoHorizontalOverflow(page);
  await page.screenshot({ path: testInfo.outputPath("preparation-title-scene.png") });
});

test("title and staff settings stay within the desktop viewport", async ({
  page
}, testInfo) => {
  await waitForTitle(page);
  await expect(page.getByRole("heading", { name: /もしもボックス.*九工大出張所/ })).toBeVisible();
  await expectNoHorizontalOverflow(page);
  await page.screenshot({ path: testInfo.outputPath("title-desktop.png") });

  await page.getByRole("button", { name: "運営設定を開く" }).click();
  await expect(page.getByRole("heading", { name: "運営設定" })).toBeVisible();
  await expect(page.getByText("動画生成の制限")).toBeVisible();
  await expectNoHorizontalOverflow(page);
  await page.screenshot({ path: testInfo.outputPath("settings-desktop.png") });
});

test("mobile title and consent remain readable", async ({ page }, testInfo) => {
  await waitForTitle(page);
  await expectNoHorizontalOverflow(page);
  await page.screenshot({ path: testInfo.outputPath("title-mobile.png") });

  await page.getByRole("button", { name: "始める" }).click();
  await expect(page.getByText("本人の声を模倣したAI音声の生成に同意します")).toBeVisible();
  await expectNoHorizontalOverflow(page);
  await page.screenshot({ path: testInfo.outputPath("consent-mobile.png"), fullPage: true });
});

test.describe("captured demo flow", () => {

  test("moves from consent through capture to the unimplemented video review", async ({
    page
  }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop-edge-engine", "full media flow runs once");
    const failedMediaStatuses: number[] = [];
    page.on("response", (response) => {
      if (response.url().includes("/media/chunk") && !response.ok()) {
        failedMediaStatuses.push(response.status());
      }
    });
    await page.addInitScript(() => {
      class TestFaceDetector {
        async detect(source: HTMLVideoElement) {
          const width = source.videoWidth || 1280;
          const height = source.videoHeight || 720;
          return [
            {
              boundingBox: new DOMRectReadOnly(
                width * 0.35,
                height * 0.2,
                width * 0.3,
                height * 0.55
              )
            }
          ];
        }
      }
      Object.defineProperty(window, "FaceDetector", {
        configurable: true,
        value: TestFaceDetector
      });

      const originalGetContext = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function (...args: Parameters<typeof originalGetContext>) {
        const context = originalGetContext.apply(this, args) as CanvasRenderingContext2D | null;
        if (context && args[0] === "2d") {
          context.getImageData = ((x: number, y: number, width: number, height: number) => {
            const data = new Uint8ClampedArray(width * height * 4);
            for (let index = 0; index < data.length; index += 4) {
              data[index] = 128;
              data[index + 1] = 128;
              data[index + 2] = 128;
              data[index + 3] = 255;
            }
            return new ImageData(data, width, height);
          }) as typeof context.getImageData;
        }
        return context;
      } as typeof HTMLCanvasElement.prototype.getContext;

      class TestUtterance extends EventTarget {
        text: string;
        lang = "ja-JP";
        rate = 1;
        pitch = 1;
        voice: SpeechSynthesisVoice | null = null;
        onend: (() => void) | null = null;
        onerror: (() => void) | null = null;
        constructor(text: string) {
          super();
          this.text = text;
        }
      }
      Object.defineProperty(window, "SpeechSynthesisUtterance", {
        configurable: true,
        value: TestUtterance
      });
      Object.defineProperty(window, "speechSynthesis", {
        configurable: true,
        value: {
          cancel() {},
          getVoices() { return []; },
          speak(utterance: TestUtterance) {
            window.setTimeout(() => utterance.onend?.(), 1_250);
          }
        }
      });
    });

    await waitForTitle(page);
    await page.getByRole("button", { name: "始める" }).click();
    await page.getByRole("checkbox", { name: /本人の声を模倣/ }).check();
    await page.getByRole("button", { name: "内容に同意して進む" }).click();

    await expect(page.getByRole("heading", { name: "あなたの声を聞いています" })).toBeVisible({
      timeout: 20_000
    });
    await expect(page.getByLabel("最新のASR文字起こし")).toHaveValue("");
    await expectNoHorizontalOverflow(page);
    await page.screenshot({ path: testInfo.outputPath("conversation-desktop.png") });

    await page.getByRole("button", { name: "会話を終了" }).click();
    await expect(page.getByText("未来からのメッセージを", { exact: false })).toBeVisible();
    await page.getByRole("button", { name: "生成を完了" }).click();

    await expect(page.getByText("動画生成ワーカーは未接続です")).toBeVisible();
    await expect(page.getByText("AI生成映像")).toBeVisible();
    await expect(page.getByText("保存接続を再試行しています")).toHaveCount(0);
    expect(failedMediaStatuses).toEqual([]);
    await expectNoHorizontalOverflow(page);
    await page.screenshot({ path: testInfo.outputPath("review-desktop.png") });

    page.once("dialog", async (dialog) => {
      expect(dialog.message()).toContain("エピソードと追加演出を再抽選");
      await dialog.accept();
    });
    await page
      .getByRole("button", { name: "エピソードを再抽選して再生成" })
      .click();
    await expect(page.getByText("未来からのメッセージを", { exact: false })).toBeVisible();
    await page.getByRole("button", { name: "生成を完了" }).click();
    await expect(page.getByText("動画生成ワーカーは未接続です")).toBeVisible();
    await expect(page.getByText("保存接続を再試行しています")).toHaveCount(0);

    await page.getByRole("button", { name: "終了してデータを削除" }).click();
    await expect(page.getByRole("button", { name: "始める" })).toBeVisible();
  });
});
