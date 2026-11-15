// Headless jsbeeb check: boot build/beeb-llm.ssd (SHIFT+BREAK) on a Model B and on a
// Model B + 6502 Second Processor, pick each model in the boot menu, generate with
// fixed seeds / temperatures / top-k / prompts and compare every sampled token with
// tools/intref.py (HOOK=1, Model B only: the logits too).
// usage: node test/verify.mjs [ntokens]      env: MACHINES=b,2p  MODELS=0,1,2,3  CASES=...
import { beeb, models, root } from "./lib.mjs";
import { writeFileSync, mkdirSync } from "fs";
import { execFileSync } from "child_process";
import path from "path";

const N = Math.min(200, +(process.argv[2] || 120));
const PY = process.env.PYTHON || "python3";
const HOOK = !!process.env.HOOK;
const machines = (process.env.MACHINES || "b,2p").split(",");
const cases = (process.env.CASES || "1234,20,10|777,16,0|4242,27,5|99,32,1|31337,13,20|555,20,10,the meaning of life").split("|");
mkdirSync(path.join(root, "screenshots"), { recursive: true });
const results = [], speeds = {};

for (const mach of machines) {
    const tube = mach === "2p";
    const want = (process.env.MODELS || models.map((_, i) => i).join(",")).split(",").map(Number);
    const fits = models.map((m) => m.need <= (tube ? 54000 : 18900) && m.rows <= (tube ? 128 : 64));
    const b = await beeb(tube); await b.boot();
    let first = true;
    for (const mi of want) {
        const M = models[mi];
        if (!fits[mi]) { console.log(`${mach} ${M.name}: does not fit (skipped)`); continue; }
        if (!first) { await b.tap("M"); }
        first = false;
        await b.start(mi);
        const { s, sym } = b;
        let rec = null;
        const cpu = s._machine.processor;
        const hook = HOOK && !tube ? (pc) => { if (pc === sym.emit && rec) { rec.tokens.push(cpu.a); rec.logits.push(Array.from(s.readMemory(sym.LU, 47)).map((v) => (v << 24) >> 24)); } } : null;
        if (hook) cpu.debugInstruction.add(hook);
        let allPer = [];
        for (const c of cases) {
            const [a, kt, topk, ...p] = c.split(","); const seed = +a, prompt = p.join(",");
            b.write(sym.testseed, [seed & 255, seed >> 8]);
            b.write(sym.kt, [+kt]); b.write(sym.topk, [+topk]);
            if (prompt) { await b.tap("RETURN"); await s.type(prompt); } else await b.tap("SPACE");
            rec = { tokens: [], logits: [] };
            // the program keeps the last 64 sampled tokens in hist[] (histn counts them)
            const h0 = b.read(sym.histn, 1)[0];
            const toks = []; let t0 = 0, t1 = 0, n = 0;
            while (n < N) {
                await s.runFrames(1);
                const h = (b.read(sym.histn, 1)[0] - h0) & 255;
                if (h - n > 60) throw new Error("fell behind");
                while (n < h && n < N) { toks.push(b.read(sym.hist + ((h0 + n) & 63), 1)[0]); n++; if (n === 1) t0 = s.elapsedCycles; t1 = s.elapsedCycles; }
            }
            if (hook) { while (rec.tokens.length < N) await s.runFrames(5); }
            const per = (t1 - t0) / (N - 1);
            allPer.push(per);
            const ref = JSON.parse(execFileSync(PY, [path.join(root, "tools/intref.py"), "json", path.join(root, M.source), seed, kt, topk, N, prompt], { cwd: root }).toString());
            let bad = -1;
            for (let i = 0; i < N; i++) if (toks[i] !== ref.tokens[i] || (hook && rec.logits[i].join() !== ref.logits[i].join())) { bad = i; break; }
            console.log(`${mach} ${M.name} seed ${seed} kt ${kt} topk ${topk} prompt ${JSON.stringify(prompt)}: ${bad < 0 ? "MATCH" : "MISMATCH at " + bad} (${N} tokens, ${(2e6 / per).toFixed(2)} tok/s)`);
            if (bad >= 0) { console.log(" beeb", toks.slice(0, bad + 3).join(" ")); console.log(" ref ", ref.tokens.slice(0, bad + 3).join(" ")); }
            if (c === cases[0]) console.log(" text:", JSON.stringify(ref.text));
            results.push(bad < 0);
            rec = null;
        }
        if (hook) cpu.debugInstruction.remove(hook);
        const avg = allPer.reduce((x, y) => x + y, 0) / allPer.length;
        (speeds[M.name] ||= {})[tube ? "2p" : "b"] = +(2e6 / avg).toFixed(2);
        writeFileSync(path.join(root, `screenshots/verify-${mach}-${M.name.toLowerCase()}.png`), await s.screenshotActive({ scale: 1 }));
    }
}
console.log("speeds (chars/s):", JSON.stringify(speeds));
writeFileSync(path.join(root, "build/speeds-measured.json"), JSON.stringify(speeds, null, 1) + "\n");
const ok = results.length > 0 && results.every(Boolean);
console.log(ok ? "ALL MATCH" : "FAILURES");
process.exit(ok ? 0 : 1);
