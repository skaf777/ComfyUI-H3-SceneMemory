const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const file = require("node:path").join(__dirname, "../web/scene_memory.js");
const source = fs.readFileSync(file, "utf8").replace(/^import .*;\s*/, "");
let extension;
vm.runInNewContext(source, { app: { registerExtension(x) { extension = x; } } });
class Node {
    constructor() {
        this.widgets = [
            { name: "enabled", value: false }, { name: "memory_frame", value: 187 },
            { name: "memory_mode", value: "Uniform" }, { name: "memory_preset", value: "Balanced" },
            { name: "max_references", value: 3 },
        ];
    }
    addWidget(type, name, value) { const w = { name, value }; this.widgets.push(w); return w; }
    setDirtyCanvas() {}
}
(async () => {
    await extension.beforeRegisterNodeDef(Node, { name: "H3SceneMemoryReferenceToVideo" });
    const node = new Node(); node.onNodeCreated();
    const preset = node.widgets.find(w => w.name === "memory_preset");
    const max = node.widgets.find(w => w.name === "max_references");
    preset.callback("Geometry"); assert.equal(max.value, 6);
    max.value = 4; preset.callback("Custom"); assert.equal(max.value, 4);
    node.onConfigure({ widgets_values: [false, 187] });
    assert.equal(node.widgets[2].value, "Single");
    assert.equal(node.widgets[1].value, 187);
    preset.callback("Dialogue"); assert.equal(node.widgets[2].value, "Uniform"); assert.equal(max.value, 2);
    node.onExecuted({ text: ["Selected scene memory: 1:0, 2:187", "warning"] });
    const report = node.widgets.find(w => w.name === "Selected memory windows");
    assert.equal(report.value, "Selected scene memory: 1:0, 2:187 | warning");
    assert.equal(report.serialize, false);
})().catch(error => { console.error(error); process.exit(1); });
