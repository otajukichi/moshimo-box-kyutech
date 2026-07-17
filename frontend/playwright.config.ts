import { defineConfig, devices } from "@playwright/test";

const externalBaseURL = process.env.MOSHIMO_E2E_BASE_URL;
const baseURL = externalBaseURL ?? "http://127.0.0.1:8790/";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 90_000,
  expect: { timeout: 60_000 },
  outputDir: "test-results",
  webServer: externalBaseURL
    ? undefined
    : {
        command:
          "MOSHIMO__STORAGE__SESSION_ROOT=/tmp/moshimo-box-e2e/sessions " +
          "MOSHIMO__STORAGE__STAFF_SETTINGS_PATH=/tmp/moshimo-box-e2e/staff-settings.json " +
          "MOSHIMO__STORAGE__METRICS_DB_PATH=/tmp/moshimo-box-e2e/metrics.sqlite3 " +
          "MOSHIMO__STORAGE__LOG_ROOT=/tmp/moshimo-box-e2e/logs " +
          "MOSHIMO__APP__DEBUG_MODE=true " +
          "../start-app.sh 8790",
        url: "http://127.0.0.1:8790/api/health",
        reuseExistingServer: true,
        timeout: 90_000
      },
  use: {
    baseURL,
    headless: true,
    permissions: ["camera", "microphone"],
    launchOptions: {
      args: [
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
        "--autoplay-policy=no-user-gesture-required"
      ]
    }
  },
  projects: [
    {
      name: "desktop-edge-engine",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 900 }
      }
    },
    {
      name: "mobile-edge-engine",
      use: {
        ...devices["Pixel 7"],
        viewport: { width: 412, height: 915 }
      }
    }
  ]
});
