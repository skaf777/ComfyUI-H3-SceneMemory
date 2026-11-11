import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const empty = () => ({ version: 1, items: [] });
const key = item => `${item.slot}:${item.frame}:${item.source}`;

app.registerExtension({
    name: "H3SceneMemory.Timeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "H3SceneMemoryReferenceToVideo") {
            const updateMode = node => {
                const mode = node.widgets?.find(w => w.name === "memory_mode");
                const input = node.inputs?.find(i => i.name === "user_selection");
                if (mode?.options) {
                    const values = ["Single", "Uniform", "Interval", "Smart", "Manual"];
                    if (input?.link != null) values.push("User");
                    mode.options.values = values;
                }
            };
            for (const hook of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
                const previous = nodeType.prototype[hook];
                nodeType.prototype[hook] = function (...args) {
                    const result = previous?.apply(this, args);
                    updateMode(this);
                    return result;
                };
            }
            return;
        }
        if (nodeData.name !== "H3SceneMemoryTimeline") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            const node = this;
            const selection = this.widgets.find(w => w.name === "selection");
            selection.type = "hidden";
            selection.computeSize = () => [0, -4];
            // This flag is injected into the isolated prompt only, never the workflow.
            const request = this.widgets.find(w => w.name === "preview_request");
            if (request) {
                request.type = "hidden";
                request.value = "";
                request.computeSize = () => [0, -4];
                request.serializeValue = () => "";
            }
            const root = document.createElement("div");
            root.className = "h3-timeline";
            Object.assign(root.style, { background: "#20252a", color: "#eee", height: "100%", padding: "8px", boxSizing: "border-box", display: "flex", flexDirection: "column", gap: "8px", overflow: "hidden" });
            const toolbar = document.createElement("div");
            Object.assign(toolbar.style, { display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" });
            const count = document.createElement("span");
            const warning = document.createElement("div"); warning.style.color = "#ffbf69";
            const strip = document.createElement("div");
            Object.assign(strip.style, { display: "flex", gap: "8px", overflow: "auto", flex: "1", alignItems: "flex-start", minHeight: "100px" });
            let previews = [];
            const read = () => { try { const value = JSON.parse(selection.value); return value?.version === 1 && Array.isArray(value.items) ? value : empty(); } catch { return empty(); } };
            const write = state => { selection.value = JSON.stringify(state); node.setDirtyCanvas(true, true); render(); };
            const render = () => {
                const state = read(); const chosen = new Set(state.items.map(key));
                count.textContent = `${state.items.length} selected`;
                strip.replaceChildren();
                for (const item of previews) {
                    const tile = document.createElement("button");
                    tile.type = "button";
                    Object.assign(tile.style, { flex: "0 0 auto", padding: "3px", color: "#eee", background: "#333", border: chosen.has(key(item)) ? "3px solid #42db70" : "3px solid transparent", cursor: "pointer" });
                    const img = document.createElement("img"); img.src = `data:image/png;base64,${item.png}`; img.style.display = "block"; img.style.width = "128px";
                    const label = document.createElement("div"); label.textContent = `File ${item.slot} · Frame ${item.frame}`;
                    tile.append(img, label);
                    tile.onclick = () => { const k = key(item); const items = state.items.filter(x => key(x) !== k); if (!chosen.has(k)) items.push({ slot: item.slot, frame: item.frame, source: item.source }); write({ version: 1, items }); };
                    strip.append(tile);
                }
            };
            const button = (label, action) => { const b = document.createElement("button"); b.type = "button"; b.textContent = label; b.onclick = action; toolbar.append(b); };
            button("Select All", () => { const items = read().items.slice(); const keys = new Set(items.map(key)); for (const x of previews) if (!keys.has(key(x))) { items.push({ slot: x.slot, frame: x.frame, source: x.source }); keys.add(key(x)); } write({ version: 1, items }); });
            button("Clear Selection", () => write(empty()));
            button("Refresh Previews", async () => {
                try {
                    warning.textContent = "Generating previews…";
                    const prompt = await app.graphToPrompt();
                    const all = prompt.output; const id = String(node.id); const output = {};
                    const visit = current => { if (output[current]) return; const entry = all[current]; if (!entry) throw new Error(`Missing upstream node ${current}`); output[current] = entry; for (const value of Object.values(entry.inputs || {})) if (Array.isArray(value) && value.length === 2 && all[String(value[0])] && Number.isInteger(value[1])) visit(String(value[0])); };
                    visit(id);
                    // Give each button press a unique cache key without changing the graph.
                    output[id] = { ...output[id], inputs: { ...output[id].inputs,
                        preview_request: `${Date.now()}-${Math.random()}` } };
                    await api.queuePrompt(0, { output, workflow: prompt.workflow });
                } catch (error) { warning.textContent = `Preview failed: ${error.message || error}`; }
            });
            toolbar.append(count); root.append(toolbar, warning, strip);
            this.addDOMWidget("timeline", "div", root, { serialize: false, hideOnZoom: false });
            this.size = [Math.max(this.size[0], 520), Math.max(this.size[1], 340)];
            this.h3TimelineUpdate = message => {
                if (!Array.isArray(message?.previews)) return;
                previews = message.previews;
                warning.textContent = (message.warning || []).join(" ");
                render();
            };
            render();
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) { const result = executed?.call(this, message); this.h3TimelineUpdate?.(message); return result; };
    },
});
