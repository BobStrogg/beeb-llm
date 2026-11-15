// Boot checks: SHIFT+BREAK autoboot, Q (and ESCAPE) back to BASIC with BASIC
// still working (immediate mode and a program), then *RUN LLM and *LLM again.
// Saves screenshots to screenshots/.
import { MachineSession } from "../vendor/jsbeeb/src/machine-session.js";
import { BBC } from "../vendor/jsbeeb/src/keymap.js";
import { writeFileSync, mkdirSync } from "fs";
import path from "path";
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const out = path.join(root, "screenshots"); mkdirSync(out, { recursive: true });
const s = new MachineSession("B-DFS1.2", { discImage: path.join(root, "build/beeb-llm.ssd") });
await s.initialise(); await s.autoboot();
const text = () => { const m = s.readMemory(0x7c00, 1000); let o = []; for (let r = 0; r < 25; r++) { let l = ""; for (let c = 0; c < 40; c++) { const b = m[r * 40 + c] & 0x7f; l += b < 0x20 ? " " : String.fromCharCode(b); } o.push(l); } return o.join("\n"); };
const running = () => text().includes("DON'T PANIC");
async function waitRun() { for (let i = 0; i < 400; i++) { await s.runFrames(5); if (running()) return true; } return false; }
async function shot(n) { writeFileSync(path.join(out, n + ".png"), await s.screenshotActive({ scale: 1 })); writeFileSync(path.join(out, n + ".txt"), text() + "\n"); }
async function tap(k, f = 4) { s.keyDownRaw(BBC[k]); await s.runFrames(f); s.keyUpRaw(BBC[k]); await s.runFrames(3); }
async function quitWith(k) { await tap(k, 40); return s.runUntilPrompt(20); }
async function basicOK() {
    await s.type("PRINT 6*7"); const a = JSON.stringify(await s.runUntilPrompt(5)).includes("42");
    await s.type('10 PRINT "HELLO";2+2'); await s.runUntilPrompt(5);
    await s.type("RUN"); const b = JSON.stringify(await s.runUntilPrompt(5)).includes("HELLO4");
    await s.type("NEW"); await s.runUntilPrompt(5);
    return a && b;
}
let ok = await waitRun(); console.log("SHIFT+BREAK started:", ok);
await s.runFrames(50 * 20); await shot("01-running");
await s.runFrames(50 * 60); await shot("02-after-a-minute");
await tap("PERIOD"); await tap("PERIOD"); await tap("K"); await s.runFrames(100); await shot("03-settings");
let b = (await quitWith("Q"), await basicOK()); console.log("Q -> BASIC works:", b); ok &&= b;
await shot("04-basic");
for (const cmd of ["*RUN LLM", "*LLM"]) {
    await s.type(cmd); const st = await waitRun(); await s.runFrames(200);
    await quitWith(cmd === "*LLM" ? "ESCAPE" : "Q"); const back = await basicOK();
    console.log(`${cmd}: started=${st} back to BASIC=${back}`); ok &&= st && back;
}
console.log(ok ? "BOOT OK" : "BOOT FAILED"); process.exit(ok ? 0 : 1);
