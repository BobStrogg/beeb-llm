// Shared helpers for the headless jsbeeb tests: boot build/beeb-llm.ssd on a
// Model B with or without a 6502 Second Processor, read the right listing's
// symbols and the right CPU's memory, drive the boot menu.
import { MachineSession } from "../vendor/jsbeeb/src/machine-session.js";
import { BBC } from "../vendor/jsbeeb/src/keymap.js";
import { readFileSync } from "fs";
import path from "path";

export const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
export const models = JSON.parse(readFileSync(path.join(root, "build/models.json"), "utf8"));

export function symbols(tube) {
    const sym = {};
    for (const l of readFileSync(path.join(root, tube ? "build/listing-2p.txt" : "build/listing.txt"), "utf8").split("\n")) {
        const m = l.match(/^SYM (\w+)\s*&([0-9A-F]+)/); if (m) sym[m[1]] = parseInt(m[2], 16);
    }
    return sym;
}

export async function beeb(tube, disc = path.join(root, "build/beeb-llm.ssd")) {
    const s = new MachineSession("B-DFS1.2", { discImage: disc, tube });
    await s.initialise();
    const b = { s, tube, sym: symbols(tube) };
    // program memory: the parasite's when the program runs in the second processor
    b.read = (a, n) => tube ? s._machine.processor.tube.memory.slice(a, a + n) : s.readMemory(a, n);
    b.write = (a, bytes) => { if (tube) s._machine.processor.tube.memory.set(bytes, a); else s.writeMemory(a, bytes); };
    b.text = () => { const m = s.readMemory(0x7c00, 1000); const o = []; for (let r = 0; r < 25; r++) { let l = ""; for (let c = 0; c < 40; c++) { const v = m[r * 40 + c] & 0x7f; l += v < 0x20 ? " " : String.fromCharCode(v); } o.push(l); } return o.join("\n"); };
    b.tap = async (k, f = 3) => { s.keyDownRaw(BBC[k]); await s.runFrames(f); s.keyUpRaw(BBC[k]); await s.runFrames(2); };
    b.waitFor = async (re, frames = 3000) => { for (let i = 0; i < frames / 5; i++) { await s.runFrames(5); if (re.test(b.text())) return true; } return false; };
    b.inMenu = () => /Model {2}Params/.test(b.text());
    b.inApp = () => /tok\/s/.test(b.text());
    // pick model i (0-based) in the menu and wait until it is generating
    b.start = async (i) => {
        if (!await b.waitFor(/Model {2}Params/)) throw new Error("no menu:\n" + b.text());
        await b.tap("K" + (i + 1)); await b.tap("RETURN");
        if (!await b.waitFor(/tok\/s/)) throw new Error("did not start:\n" + b.text());
    };
    b.boot = async () => { await s.autoboot(); };
    return b;
}
export { BBC };
