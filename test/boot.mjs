// Boot checks on a Model B and on a Model B + 6502 Second Processor:
// SHIFT+BREAK -> boot menu (screenshot) -> auto-start of the default model,
// M back to the menu, pick another model, Q (and ESCAPE, and Q in the menu)
// back to BASIC with BASIC still working (immediate mode and a program),
// then *RUN LLM / *LLM (*RUN LLMP / *LLMP on the second processor) again.
// Screenshots go to screenshots/.     usage: node test/boot.mjs [b|2p ...]
import { beeb, models, root } from "./lib.mjs";
import { writeFileSync, mkdirSync } from "fs";
import path from "path";
const out = path.join(root, "screenshots"); mkdirSync(out, { recursive: true });
const which = process.argv.slice(2).length ? process.argv.slice(2) : ["b", "2p"];
let ok = true;
const check = (what, v) => { console.log(`  ${what}: ${v ? "ok" : "FAILED"}`); ok &&= !!v; };

for (const mach of which) {
    const tube = mach === "2p";
    console.log(mach === "2p" ? "Model B + 6502 Second Processor" : "Model B");
    const b = await beeb(tube); const { s } = b;
    const shot = async (n) => { writeFileSync(path.join(out, `${mach}-${n}.png`), await s.screenshotActive({ scale: 1 })); writeFileSync(path.join(out, `${mach}-${n}.txt`), b.text() + "\n"); };
    const prompt = async () => { for (let i = 0; i < 200; i++) { await s.runFrames(5); const t = b.text().trimEnd().split("\n"); if (/^>\s*$/.test(t[t.length - 1] || "")) return true; } return false; };
    const basicOK = async () => {
        await s.type("PRINT 6*7"); await s.runFrames(30); const a = b.text().includes("42");
        await s.type('10 PRINT "HELLO";2+2'); await s.runFrames(20);
        await s.type("RUN"); await s.runFrames(30); const c = b.text().includes("HELLO4");
        await s.type("NEW"); await s.runFrames(20);
        return a && c;
    };
    await b.boot();
    check("SHIFT+BREAK shows the menu", await b.waitFor(/Model {2}Params/, 4000));
    await s.runFrames(25); await shot("01-menu");
    const def = tube ? "Medium" : "Small";
    check(`auto-starts the default (${def})`, await b.waitFor(new RegExp(`BEEB-LLM ${def}`), 600) && await b.waitFor(/tok\/s/, 600));
    await s.runFrames(50 * 30); await shot("02-running");
    await b.tap("PERIOD"); await b.tap("PERIOD"); await b.tap("K"); await s.runFrames(100); await shot("03-settings");
    await b.tap("M");
    check("M returns to the menu", await b.waitFor(/Model {2}Params/, 1500) && !/Starting in/.test(b.text()));
    await b.tap("K1"); await s.runFrames(10); await shot("04-menu-pick");
    await b.tap("RETURN");
    check(`picks ${models[0].name}`, await b.waitFor(new RegExp(`BEEB-LLM ${models[0].name}`), 1500));
    await s.runFrames(50 * 10);
    await b.tap("Q", 40);
    check("Q -> BASIC", await prompt()); check("BASIC works", await basicOK()); await shot("05-basic");
    const P = tube ? "LLMP" : "LLM";
    for (const [cmd, how] of [[`*RUN ${P}`, "ESCAPE"], [`*${P}`, "menuQ"]]) {
        await s.type(cmd);
        const m = await b.waitFor(/Model {2}Params/, 2000);
        if (how === "menuQ") { await b.tap("Q", 30); }
        else { await b.tap("RETURN"); await b.waitFor(/tok\/s/, 1500); await s.runFrames(200); await b.tap("ESCAPE", 40); }
        const back = await prompt() && await basicOK();
        check(`${cmd}, ${how === "menuQ" ? "Q in the menu" : "ESCAPE"} -> BASIC works`, m && back);
    }
}
console.log(ok ? "BOOT OK" : "BOOT FAILED"); process.exit(ok ? 0 : 1);
