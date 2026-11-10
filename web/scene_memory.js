import { app } from "../../scripts/app.js";

const counts = { Dialogue: 2, Balanced: 3, Geometry: 6, Detailed: 8, Dense: 10 };

app.registerExtension({
    name: "H3SceneMemory.Controls",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3SceneMemoryReferenceToVideo") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            const preset = this.widgets?.find((widget) => widget.name === "memory_preset");
            const maximum = this.widgets?.find((widget) => widget.name === "max_references");
            const mode = this.widgets?.find((widget) => widget.name === "memory_mode");
            if (preset && maximum) {
                const previous = preset.callback;
                preset.callback = function (value, ...rest) {
                    const returned = previous?.call(this, value, ...rest);
                    if (Object.hasOwn(counts, value)) {
                        maximum.value = counts[value];
                        // A multi-window preset must not silently stay in Single mode.
                        if (mode?.value === "Single") mode.value = "Uniform";
                    }
                    return returned;
                };
            }
            return result;
        };
        // Old node versions saved no memory_mode widget: preserve Single and
        // its existing memory_frame instead of applying the new-node default.
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data, ...args) {
            const result = configured?.call(this, data, ...args);
            const index = this.widgets?.findIndex((widget) => widget.name === "memory_mode") ?? -1;
            if (index >= 0 && Array.isArray(data?.widgets_values) && data.widgets_values.length <= index) {
                this.widgets[index].value = "Single";
            }
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = executed?.call(this, message);
            const text = message?.text;
            if (Array.isArray(text) && text.length) {
                let widget = this.widgets?.find((item) => item.name === "Selected memory windows");
                if (!widget) {
                    widget = this.addWidget("text", "Selected memory windows", "", () => {}, { serialize: false });
                    widget.serialize = false;
                    widget.disabled = true;
                }
                widget.value = text.join(" | ");
                this.setDirtyCanvas(true, true);
            }
            return result;
        };
    },
});
