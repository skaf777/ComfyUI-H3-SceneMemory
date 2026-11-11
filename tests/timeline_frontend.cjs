const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../web/timeline.js'), 'utf8').replace(/^import .*;\s*/gm, '');
class Element {
    constructor() { this.style = {}; this.children = []; }
    append(...items) { this.children.push(...items); }
    replaceChildren() { this.children = []; }
}
let extension, submitted;
const app = { registerExtension(x) { extension = x; }, async graphToPrompt() { return { output: { '1': { inputs: {} }, '2': { inputs: { memory: ['1', 0] } }, '3': { inputs: {} } }, workflow: {} }; } };
const api = { async queuePrompt(n, p) { submitted = p; } };
vm.runInNewContext(source, { app, api, document: { createElement: () => new Element() } });
class Node {
    constructor() { this.id = 2; this.size = [200, 200]; this.widgets = [{ name: 'selection', value: '{"version":1,"items":[]}' }]; }
    addDOMWidget(name, type, root) { this.root = root; }
    setDirtyCanvas() {}
}
(async () => {
    await extension.beforeRegisterNodeDef(Node, {name: 'H3SceneMemoryTimeline'});
    const node = new Node(); node.onNodeCreated();
    const item = {slot: 1, frame: 17, source: 'hash', png: 'x'};
    node.onExecuted({previews: [item], warning: []});
    const [toolbar, , strip] = node.root.children;
    strip.children[0].onclick();
    assert.equal(JSON.parse(node.widgets[0].value).items[0].frame, 17);
    strip.children[0].onclick(); assert.equal(JSON.parse(node.widgets[0].value).items.length, 0);
    toolbar.children[0].onclick(); assert.equal(JSON.parse(node.widgets[0].value).items.length, 1);
    const persisted = node.widgets[0].value;
    const restored = new Node(); restored.widgets[0].value = persisted; restored.onNodeCreated(); restored.onExecuted({previews: [item]});
    assert.match(restored.root.children[2].children[0].style.border, /42db70/);
    toolbar.children[1].onclick(); assert.equal(JSON.parse(node.widgets[0].value).items.length, 0);
    await toolbar.children[2].onclick(); assert.deepEqual(Object.keys(submitted.output).sort(), ['1', '2']);
    class Main { constructor() { this.widgets = [{name:'memory_mode', options:{}}]; this.inputs = [{name:'user_selection', link:null}]; } }
    await extension.beforeRegisterNodeDef(Main, {name:'H3SceneMemoryReferenceToVideo'});
    const main = new Main(); main.onNodeCreated(); assert.equal(main.widgets[0].options.values.includes('User'), false);
    main.inputs[0].link = 10; main.onConnectionsChange(); assert.equal(main.widgets[0].options.values.includes('User'), true);
    main.inputs[0].link = null; main.onConnectionsChange(); assert.equal(main.widgets[0].options.values.includes('User'), false);
})().catch(error => { console.error(error); process.exit(1); });
