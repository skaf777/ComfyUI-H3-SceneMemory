const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
    constructor(tag) { this.tag = tag; this.style = {}; this.children = []; this.value = ''; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    setAttribute() {}
    remove() { this.removed = true; }
    focus() {}
}
const body = new Element('body');
const document = { body, createElement: tag => new Element(tag), addEventListener() {}, removeEventListener() {} };
let extension;
const app = { registerExtension(value) { extension = value; } };
let calls = 0;
const api = { apiURL: x => x, fetchApi: async () => { calls++; return { ok: true, json: async () => [
    { filename: 'scene_0001.safetensors', scene: 'scene', take: '0001', preview: true },
    { filename: 'old.safetensors', scene: 'old', take: null, preview: false },
] }; } };
const source = fs.readFileSync(require('node:path').join(__dirname, '../web/gallery.js'), 'utf8').replace(/^import .*;\s*/gm, '');
vm.runInNewContext(source, { app, api, document, encodeURIComponent });
class Node {
    constructor() { this.widgets = [{ name: 'filename', value: '', options: { values: [''] } }]; }
    addWidget(type, name, value, callback) { this.widgets.push({ type, name, callback }); }
    setDirtyCanvas() {}
}
(async () => {
    await extension.beforeRegisterNodeDef(Node, { name: 'H3SceneMemoryLoad' });
    const node = new Node(); node.onNodeCreated();
    await node.widgets[1].callback();
    const overlay = body.children[0], dialog = overlay.children[0], toolbar = dialog.children[0], grid = dialog.children[2];
    assert.equal(grid.children.length, 2);
    assert.equal(grid.children[1].children[0].textContent, 'No Preview');
    toolbar.children[0].value = 'scene'; toolbar.children[0].oninput();
    assert.equal(grid.children.length, 1);
    grid.children[0].onclick();
    assert.equal(node.widgets[0].value, 'scene_0001.safetensors');
    assert.ok(node.widgets[0].options.values.includes('scene_0001.safetensors'));
    assert.ok(overlay.removed); assert.equal(calls, 1);
})().catch(error => { console.error(error); process.exit(1); });
