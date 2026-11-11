import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "H3SceneMemory.Gallery",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3SceneMemoryLoad") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            const node = this;
            this.addWidget("button", "Browse Scene Memories", null, () => openGallery(node), { serialize: false });
            return result;
        };
    },
});

async function openGallery(node) {
    const overlay = document.createElement("div");
    Object.assign(overlay.style, { position: "fixed", inset: "0", zIndex: "10000", background: "#000b", display: "flex", alignItems: "center", justifyContent: "center" });
    const dialog = document.createElement("div");
    dialog.setAttribute("role", "dialog"); dialog.setAttribute("aria-label", "Scene memories");
    Object.assign(dialog.style, { background: "#20252a", color: "#eee", padding: "16px", width: "min(1000px, 90vw)", height: "min(720px, 85vh)", display: "flex", flexDirection: "column", gap: "10px" });
    const toolbar = document.createElement("div");
    Object.assign(toolbar.style, { display: "flex", gap: "8px" });
    const filter = document.createElement("input"); filter.placeholder = "Filter scenes"; filter.setAttribute("aria-label", "Filter scenes");
    const refresh = document.createElement("button"); refresh.textContent = "Refresh";
    const close = document.createElement("button"); close.textContent = "Close";
    const status = document.createElement("div");
    const grid = document.createElement("div");
    Object.assign(grid.style, { overflow: "auto", display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))", gap: "10px", alignContent: "start", flex: "1" });
    toolbar.append(filter, refresh, close); dialog.append(toolbar, status, grid); overlay.append(dialog); document.body.append(overlay);
    let entries = [];
    const remove = () => { document.removeEventListener("keydown", escape); overlay.remove(); };
    const escape = e => { if (e.key === "Escape") remove(); };
    document.addEventListener("keydown", escape); close.onclick = remove;
    overlay.onclick = e => { if (e.target === overlay) remove(); };
    const render = () => {
        grid.replaceChildren();
        const query = filter.value.toLocaleLowerCase();
        const matches = entries.filter(item => item.filename.toLocaleLowerCase().includes(query));
        status.textContent = `${matches.length} memories`;
        for (const item of matches) {
            const tile = document.createElement("button"); tile.type = "button";
            Object.assign(tile.style, { background: "#333", color: "#eee", border: "1px solid #666", padding: "6px", cursor: "pointer", overflowWrap: "anywhere", textAlign: "left" });
            const preview = document.createElement("div");
            Object.assign(preview.style, { height: "110px", background: "#171a1d", display: "flex", alignItems: "center", justifyContent: "center" });
            const placeholder = () => { preview.replaceChildren(); preview.textContent = "No Preview"; };
            if (item.preview) {
                const img = document.createElement("img"); img.loading = "lazy"; img.alt = "";
                img.src = api.apiURL(`/h3_scene_memory/thumbnail?filename=${encodeURIComponent(item.filename)}`);
                Object.assign(img.style, { maxWidth: "100%", maxHeight: "100%", objectFit: "contain" });
                img.onerror = placeholder; preview.append(img);
            } else placeholder();
            const title = document.createElement("div"); title.textContent = item.scene + (item.take ? ` · Take ${item.take}` : "");
            const filename = document.createElement("small"); filename.textContent = item.filename;
            tile.append(preview, title, filename);
            tile.onclick = () => {
                const widget = node.widgets?.find(w => w.name === "filename");
                if (!widget) { status.textContent = "Filename widget unavailable"; return; }
                if (Array.isArray(widget.options?.values) && !widget.options.values.includes(item.filename)) widget.options.values.push(item.filename);
                widget.value = item.filename; widget.callback?.call(widget, item.filename); node.setDirtyCanvas(true, true); remove();
            };
            grid.append(tile);
        }
    };
    const load = async () => {
        try {
            status.textContent = "Loading…";
            const response = await api.fetchApi("/h3_scene_memory/list");
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            entries = await response.json(); render();
        } catch (error) { status.textContent = `Could not load memories: ${error.message || error}`; }
    };
    filter.oninput = render; refresh.onclick = load; await load(); filter.focus();
}
