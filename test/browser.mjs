// Browser check of the web page (local or the live GitHub Pages site):
// loads it in headless Chrome, waits for the auto-booted model to start typing,
// checks the emulator picture keeps changing (text being generated), uses the
// on-screen prompt box and buttons, and saves screenshots.
// Runs twice: the plain Model B page and ?machine=2p (6502 Second Processor).
// usage: node test/browser.mjs [url]   (default http://localhost:8766/)
import { chromium } from "../vendor/jsbeeb/node_modules/playwright/index.mjs";
import { mkdirSync } from "fs";
const url = process.argv[2] || "http://localhost:8766/";
const tag = url.includes("github.io") ? "live" : "local";
mkdirSync("screenshots", { recursive: true });
const browser = await chromium.launch({ executablePath: process.env.CHROME || "/usr/bin/google-chrome", args: ["--autoplay-policy=no-user-gesture-required"] });
let allOk = true;
for (const mach of ["b", "2p"]) {
const page = await browser.newPage({ viewport: { width: 1280, height: 1100 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const u = mach === "2p" ? url + (url.includes("?") ? "&" : "?") + "machine=2p" : url;
await page.goto(u, { waitUntil: "load" });
const src = await page.locator("#beeb").getAttribute("src");
const iframe = page.locator("#beeb");
const snap = async (name) => iframe.screenshot({ path: `screenshots/browser-${tag}-${mach}-${name}.png` });
await page.waitForTimeout(6000);
await snap("00-menu");
await page.waitForTimeout(16000);
const a = await snap("01-boot");
await page.waitForTimeout(8000);
const b = await snap("02-typing");
const changed1 = !a.equals(b);
await page.fill("#prompt", "the river was");
await page.click("#promptform button");
await page.waitForTimeout(6000 + 15 * 900);
await page.waitForTimeout(15000);
const c = await snap("03-prompt");
await page.locator("[data-code=Period]").click();
await page.waitForTimeout(2000);
await page.locator("[data-code=KeyM]").click();
await page.waitForTimeout(3000);
await snap("04-menu-again");
await page.screenshot({ path: `screenshots/browser-${tag}-${mach}-05-page.png`, fullPage: true });
const changed2 = !b.equals(c);
const tubeOk = (mach === "2p") === /coProcessor=true/.test(src);
console.log(`${u}: iframe ${src}; picture changing while generating: ${changed1}, after prompt: ${changed2}, page errors: ${errors.length ? errors.join("; ") : "none"}`);
allOk &&= changed1 && changed2 && tubeOk && !errors.length;
await page.close();
}
await browser.close();
process.exit(allOk ? 0 : 1);
