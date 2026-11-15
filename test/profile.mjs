// Cycle profile of one generated token, by routine (uses the listing's labels).
import { MachineSession } from "../vendor/jsbeeb/src/machine-session.js";
import { BBC } from "../vendor/jsbeeb/src/keymap.js";
import { readFileSync } from "fs";
import path from "path";
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const labels = [];
for (const l of readFileSync(path.join(root, "build/listing.txt"), "utf8").split("\n")) {
    const m = l.match(/^SYM (\w+)\s*&([0-9A-F]+)/); if (m) labels.push([parseInt(m[2], 16), m[1]]);
}
labels.sort((a, b) => a[0] - b[0]);
const s = new MachineSession("B-DFS1.2", { discImage: path.join(root, "build/beeb-llm.ssd") });
await s.initialise(); await s.autoboot();
await s.runFrames(600);
const cpu = s._machine.processor;
const hist = new Map(); let last = s.elapsedCycles, lastName = "?";
const name = (pc) => { if (pc >= 0xc000) return "OS"; let n = "?"; for (const [a, l] of labels) { if (pc >= a) n = l; else break; } return n; };
const cache = new Map();
cpu.debugInstruction.add((pc) => {
    const now = s.elapsedCycles; hist.set(lastName, (hist.get(lastName) || 0) + now - last); last = now;
    let n = cache.get(pc); if (n === undefined) { n = name(pc); cache.set(pc, n); } lastName = n;
});
const t0 = s.elapsedCycles; await s.runFrames(150); const tot = s.elapsedCycles - t0;
const toks = 150 * 40000 / tot;
for (const [k, v] of [...hist].sort((a, b) => b[1] - a[1])) console.log(k.padEnd(12), (100 * v / tot).toFixed(1) + "%");
