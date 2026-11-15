// Headless jsbeeb check: boot build/beeb-llm.ssd with SHIFT+BREAK on a Model B,
// generate with fixed seeds / temperatures / top-k / prompts and compare every
// sampled token (and the logits it was sampled from) with tools/intref.py.
// usage: node test/verify.mjs [ntokens]
import { MachineSession } from "../vendor/jsbeeb/src/machine-session.js";
import { BBC } from "../vendor/jsbeeb/src/keymap.js";
import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { execFileSync } from "child_process";
import path from "path";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const N = Math.min(200, +(process.argv[2] || 120));
const PY = process.env.PYTHON || "python3";
const sym = {};
for (const l of readFileSync(path.join(root, "build/listing.txt"), "utf8").split("\n")) {
    const m = l.match(/^SYM (\w+)\s*&([0-9A-F]+)/); if (m) sym[m[1]] = parseInt(m[2], 16);
}
const s = new MachineSession("B-DFS1.2", { discImage: path.join(root, "build/beeb-llm.ssd") });
await s.initialise();
await s.autoboot();
const cpu = s._machine.processor;
const scr = () => { const m = s.readMemory(0x7c00, 1000); let o = []; for (let r = 0; r < 25; r++) { let l = ""; for (let c = 0; c < 40; c++) { const b = m[r * 40 + c] & 0x7f; l += b < 0x20 ? " " : String.fromCharCode(b); } o.push(l); } return o.join("\n"); };
let booted = false;
for (let i = 0; i < 600 && !booted; i++) { await s.runFrames(5); booted = scr().includes("DON'T PANIC"); }
if (!booted) { console.log(scr()); throw new Error("did not boot"); }
console.log("booted at frame", s.frameCount);

let rec = null, lastCycle = 0;
const HOOK = !!process.env.HOOK;   // HOOK=1: also capture the logits at every sample (slow)
if (HOOK) cpu.debugInstruction.add((pc) => {
    if (pc === sym.emit && rec) {
        rec.tokens.push(cpu.a);
        rec.logits.push(Array.from(s.readMemory(sym.LU, 47)).map((b) => (b << 24) >> 24));
        rec.cycles.push(s.elapsedCycles - lastCycle); lastCycle = s.elapsedCycles;
    }
});
async function tap(key) { s.keyDownRaw(BBC[key]); await s.runFrames(3); s.keyUpRaw(BBC[key]); await s.runFrames(2); }

async function run(seed, kt, topk, prompt = "") {
    s.writeMemory(sym.testseed, [seed & 255, seed >> 8]);
    s.writeMemory(sym.kt, [kt]); s.writeMemory(sym.topk, [topk]);
    rec = { tokens: [], logits: [], cycles: [] };
    if (prompt) {
        await tap("RETURN");
        await s.type(prompt);           // types the prompt and RETURN
        rec = { tokens: [], logits: [], cycles: [] };
    } else {
        await tap("SPACE");
        rec = { tokens: [], logits: [], cycles: [] };
    }
    lastCycle = s.elapsedCycles;
    if (HOOK) {
        while (rec.tokens.length < N) await s.runFrames(25);
    } else {
        // the Beeb keeps the last 256 sampled tokens in hist[] (histn = count mod 256)
        const h0 = s.readMemory(sym.histn, 1)[0];
        let n = 0, t0 = 0, t1 = 0;
        while (n < N) {
            await s.runFrames(2);
            const h = (s.readMemory(sym.histn, 1)[0] - h0) & 255;
            if (h !== n) { if (n === 0) t0 = s.elapsedCycles; n = h; t1 = s.elapsedCycles; if (n > 200) break; }
        }
        const all = []; for (let i = 0; i < N; i++) all.push(s.readMemory(sym.hist + ((h0 + i) & 255), 1)[0]);
        rec.tokens = all; rec.logits = null;
        const per = (t1 - t0) / (n - 1); rec.cycles = new Array(N).fill(per);
    }
    const got = rec; rec = null;
    const ref = JSON.parse(execFileSync(PY, [path.join(root, "tools/intref.py"), "json", path.join(root, "build/model.npz"), seed, kt, topk, N, prompt], { cwd: root }).toString());
    let firstBad = -1;
    for (let i = 0; i < N; i++) if (got.tokens[i] !== ref.tokens[i] || (got.logits && got.logits[i].join() !== ref.logits[i].join())) { firstBad = i; break; }
    const cyc = got.cycles.slice(1);
    const avg = cyc.reduce((a, b) => a + b, 0) / cyc.length;
    console.log(`seed ${seed} kt ${kt} topk ${topk} prompt ${JSON.stringify(prompt)}: ${firstBad < 0 ? "MATCH" : "MISMATCH at " + firstBad} ` +
        `(${N} tokens, ${Math.round(avg)} cycles/token = ${(2e6 / avg).toFixed(2)} tok/s)`);
    if (firstBad >= 0) {
        if (got.logits) console.log(" beeb logits", got.logits[firstBad].join(" "));
        console.log(" ref  logits", ref.logits[firstBad].join(" "));
        console.log(" beeb tokens", got.tokens.slice(0, firstBad + 3).join(" "));
        console.log(" ref  tokens", ref.tokens.slice(0, firstBad + 3).join(" "));
    }
    console.log(" text:", JSON.stringify(ref.text));
    return { ok: firstBad < 0, avg };
}
const results = [];
const cases = (process.env.CASES || "1234,20,10|777,16,0|4242,27,5|99,32,1|31337,13,20|555,20,10,the meaning of life").split("|");
for (const c of cases) { const [a, b, d, ...p] = c.split(","); results.push(await run(+a, +b, +d, p.join(","))); }
mkdirSync(path.join(root, "screenshots"), { recursive: true });
writeFileSync(path.join(root, "screenshots/verify.png"), await s.screenshotActive({ scale: 1 }));
writeFileSync(path.join(root, "screenshots/verify.txt"), scr() + "\n");
console.log(scr());
const ok = results.every((r) => r.ok);
console.log(ok ? "ALL MATCH" : "FAILURES");
process.exit(ok ? 0 : 1);
