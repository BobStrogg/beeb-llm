// Cycle profile of generation, by routine (uses the listing's SYM labels).
// usage: node test/profile.mjs [model number, default 1 = Small]   (Model B)
import { beeb } from "./lib.mjs";
const b = await beeb(false); await b.boot();
await b.start(+(process.argv[2] || 1));
await b.s.runFrames(200);
const { s, sym } = b;
const labels = Object.entries(sym).map(([n, a]) => [a, n]).sort((x, y) => x[0] - y[0]);
const cpu = s._machine.processor;
const hist = new Map(); let last = s.elapsedCycles, lastName = "?";
const name = (pc) => { if (pc >= 0xc000) return "OS"; let n = "?"; for (const [a, l] of labels) { if (pc >= a) n = l; else break; } return n; };
const cache = new Map();
cpu.debugInstruction.add((pc) => {
    const now = s.elapsedCycles; hist.set(lastName, (hist.get(lastName) || 0) + now - last); last = now;
    let n = cache.get(pc); if (n === undefined) { n = name(pc); cache.set(pc, n); } lastName = n;
});
const t0 = s.elapsedCycles; await s.runFrames(150); const tot = s.elapsedCycles - t0;
for (const [k, v] of [...hist].sort((a, b) => b[1] - a[1])) console.log(k.padEnd(12), (100 * v / tot).toFixed(1) + "%");
