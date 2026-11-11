# ComfyUI-H3-SceneMemory

Save and load an original MiniMax H3 **stage 1 video latent**, then use a short window as an additional reference for an independent H3 generation. The original latent is preserved for the diffusion model; the video VAE decodes a presentation for the Qwen encoder. Consistency improvements and lip-sync behavior must be evaluated experimentally.

## Install

1. Place this repository in `ComfyUI/custom_nodes/ComfyUI-H3-SceneMemory` (for example, clone it into `custom_nodes`).
2. Restart ComfyUI. Find **Save Scene Memory**, **Load Scene Memory**, and **MiniMax H3 Reference to Video + Scene Memory** under `MiniMax H3/Scene Memory`.

The extension uses PyTorch, NumPy and safetensors already required by a normal ComfyUI installation. No models or extra model weights are installed. Install and use a revision of this repository containing the desired changes.

## Save a scene

Connect the **video_latent** output of **Separate AV Latent** after the first H3 sampler to **Save Scene Memory**. In the workflow discussed in the issue this is node **#156**, downstream of sampler **#150**. Leave its existing connections in place: this is an additional branch.

Enter a scene name such as `Airlock_Corridor`, enable **auto_numbering**, and run the graph. Each execution saves the next take in `ComfyUI/output/scene_memory`, for example `Airlock_Corridor_0001.safetensors`, then `Airlock_Corridor_0002.safetensors`. Numbering continues from existing files after a restart and concurrent saves cannot overwrite each other. `saved_path` is the full path of the file created; `video_latent` passes through unchanged. Names must be single filenames without path separators.

Optionally connect the MiniMax H3 video **vae** to save a matching PNG thumbnail. Only the first two latent steps are decoded and the first frame is reduced to at most 256 pixels wide; no sampler runs. Without a VAE no thumbnail is made. If thumbnail generation fails, the memory is still saved and the error is logged. Audio is not saved.

With **auto_numbering** off (the default for existing workflows), the original filename behavior is preserved: `room_take_01` saves `room_take_01.safetensors`, and an existing file causes an error. Use a new filename or enable numbering for subsequent runs. Existing memories are never overwritten.

Use only an original Stage 1 H3 video latent. The node verifies H3 tensor structure, but the tensor itself cannot prove its model, source sampler, or stage: `source=stage1` records what the user connected. Do not connect the AV pair, an LTX latent, or the Stage 2 output. Stage 2 support and source identification are outside this first stage.

## Load a scene

Click **Browse Scene Memories** in **Load Scene Memory** to open the gallery without running the graph. Filter by name, use **Refresh** to find new files, and click a thumbnail to select its file. Older files without PNGs appear as **No Preview**. The original filename dropdown remains available; refresh/reopen the node list in ComfyUI if a newly saved file is not yet shown there. The output is a standard video-only `LATENT` (`{"samples": tensor}`), on CPU. Connect it to the `scene_memory` input described below. It is not a complete audio/video sampling latent and must not replace the target latent in the existing H3 sampler.

Files contain a single `video_latent` tensor and string metadata: format version, source, shape, dtype, latent dimensions, inferred pixel resolution and frame count, FPS and SHA-256 of the exact tensor bytes. The loader checks safetensors parsing, tensor structure and finite values, metadata agreement and checksum. Files with altered or missing information are rejected; no pixel or VAE fallback is performed. The checksum detects accidental data changes, not malicious modification of both tensor and checksum. The format does not identify the actual H3 checkpoint or VAE, so it cannot guarantee model compatibility or provenance.

The supported stage-one geometry is batch 1, 24 channels, `[1, 24, T, H/16, W/16]`, floating-point `float16`, `bfloat16`, or `float32`, even positive latent spatial dimensions, and H3's full-clip `T = 5k + 2` temporal grid. Frame count is inferred as `17k + 5` at 24 fps; this is the generated H3 frame count, not necessarily the number of frames in a separately trimmed/exported video. Output resolution is inferred from latent dimensions, not probed from a video file.

## Apply scene memory

Replace only **MiniMax H3 Reference to Video** (node **#121** in the supplied workflow) with **MiniMax H3 Reference to Video + Scene Memory**. Reconnect its original CLIP, video/audio VAE, prompt, dimensions, image/video/audio references and both outputs exactly as before. Feed `positive` to the existing guiders and `LATENT` to the Stage 1 sampler. Connect **Load Scene Memory** to `scene_memory`; optionally connect up to three more individually selected files to `scene_memory_2` through `scene_memory_4`. Only files you explicitly connect are combined: choose memories of the same scene yourself.

Set `enabled` to true. Each memory window is five source frames (two H3 latent steps), beginning at a multiple of 17 frames. Frame numbers are zero-based at 24 fps. The node snaps requested frames down to the nearest preceding boundary, removes duplicate windows, and displays the actual `file:frame` selections and warnings on the node after execution; they are also logged. The node does not yet include an image preview. A 243-frame H3 clip has 15 phase-aligned windows (0 through 238); the actual number is calculated from the saved latent.

| Memory Mode | Selection |
| --- | --- |
| Single | The existing `memory_frame` input selects one moment. |
| Uniform | Spread windows across all explicitly connected files. |
| Interval | Walk each source at `frame_interval` frames; 5, 10, 17, 34, 68 and other positive values are accepted. After snapping, duplicates are removed. |
| Smart | Decode candidates and deterministically favor visual difference, edge structure, temporal spread and edge sharpness. Visually near-identical candidates are skipped. |
| Manual | Use `manual_frames`, for example `0, 187` for file 1 or `1:0, 2:187` for specific inputs. |

`pinned_frames` uses the same syntax and takes priority in every mode. If pins exceed `max_references`, the requested count is increased to preserve them, subject to the multi-file safety cap. `max_references` is a target, not a promise: a source may have fewer distinct windows, and Smart can discard similar views. One file has no arbitrary 32-window cap. For multiple connected files this extension caps the combined result at **32**, removes byte-identical file/window duplicates and warns when the request or pins are truncated. This is an extension safeguard, **not a validated H3 reference limit**. Each input slot remains numbered even if an intermediate socket is unconnected.

`memory_preset` sets the starting `max_references` when changed in the ComfyUI interface: Dialogue = 2, Balanced = 3, Geometry = 6, Detailed = 8 and Dense = 10. These are experimental starting points, not proven optima; edit `max_references` afterwards or select Custom. Selecting a multi-window preset while in Single mode also switches to Uniform; you can change the mode afterwards. API callers should set `max_references` explicitly: the preset changes the widget value in the frontend, not backend generation parameters. A new node defaults to Uniform with three windows; prior workflows retain Single and their `memory_frame`. When disabled, execution delegates to the original H3 node and lazy memory inputs are not needed.

Example for a dialogue: choose **Dialogue**, set mode to **Smart**, pin `0`, and use a quiet source moment with neutral mouth positions when possible. For an interior, start with **Geometry** or **Detailed**, use **Uniform** or **Smart**, and pin an establishing view plus a frame with an important feature, e.g. `0, 187`. For views saved in separate files, use pins such as `1:0, 2:187`. These choices require A/B testing with your existing references and audio.

Smart analyzes a small spatial grid only for its explainable color, contour, temporal-coverage and sharpness heuristics. It does **not** recognize characters, objects, geometry, camera pose or actual semantic usefulness; a different composition is only a proxy for a different view. Decodes are cached for the duration of one execution, so analysis and Qwen presentation do not decode the same selected window twice. New executions may decode again. No new neural model is used.

The video VAE decodes each selected window for Qwen; at H3's 2 fps sampling rate, Qwen sees its first frame. DiT receives the corresponding original two-step latent, without re-encoding. Each memory becomes the next `<Video N>` after existing video references. The original prompt is preserved and a brief instruction is appended in enabled mode. The node does not modify the original H3 node, audio inputs, AV latent output, sampler settings or saved memory. Use the compatible H3 video VAE and Qwen encoder that created the source; shape checks cannot establish model or VAE identity.

Each memory adds `2 × (latent_h/2) × (latent_w/2)` DiT visual rows, in addition to existing references, text, target video/audio and Qwen vision tokens. For example, at 864×480 this is about 810 rows per memory, before the other inputs. Warnings use heuristic thresholds and known reference-block geometry, not measured VRAM or a confirmed maximum context. Time, VRAM and attention cost increase, and third-party patch compatibility is not guaranteed; test your exact versions and hardware. No full-resolution reference is downscaled for generation.

A short window reduces temporal context but cannot guarantee that camera movement, mouth positions or old motion will not influence the new shot. Memory carries no old audio, but it does not guarantee speaker identity, scene geometry, dialogue or lip-sync. Evaluate those properties on paired seeds in the target workflow.

## Tests

In an environment with ComfyUI's Python dependencies installed:

```bash
python -m unittest discover -s tests -v
node tests/frontend_check.cjs  # optional frontend contract check
```

Apply tests use lightweight ComfyUI, Qwen and VAE doubles. Set `COMFYUI_H3_SOURCE` to the installed `comfy_extras/nodes_minimax_h3.py` to additionally exercise the official node method with doubles. Tests use real PyTorch when available, including a bit-exact bfloat16 test. If PyTorch is absent, the tests use a clearly identified NumPy tensor double with the real safetensors serializer; this exercises the file format but is **not** a substitute for running the tests with real PyTorch and an H3/ComfyUI workflow. No model is needed for the unit tests.

## Design references

The video-only original-latent approach follows the architecture described in [H3 Continuity Studio Visual Memory](https://github.com/KadoBOT/ComfyUI-H3-Continuity-Studio/blob/e6a37dcf36d9be541a5cbde0667720836e51b615/guide/continuity/visual-memory.md). The separated AV safetensors and metadata pattern was informed by [OBVPM MCtx](https://github.com/chanon/comfyui-obvpm-timeline/blob/068a97b4786daef8d67277fe323297fb86d54a3f/mctx.py). This repository implements its own scene-memory format and does not copy OBVPM source or claim `.mctx.safetensors` compatibility.

## Visual timeline (Scene Memory 3.0)

Add **Scene Memory Timeline** and connect the same **Load Scene Memory** output to its `scene_memory` and the main H3 node's `scene_memory`. Connect the MiniMax H3 video VAE to the timeline. For multiple files, connect the same files to matching numbered slots on both nodes (up to four). Connect `USER_SELECTION` to the main node's `user_selection`, enable memory and choose **User** in `memory_mode`.

Click **Refresh Previews** to queue the timeline and its upstream inputs only; this does not queue the H3 generation branch or samplers. A normal Queue Prompt only passes the saved selection and never generates previews; previously displayed thumbnails stay visible. Thumbnails represent complete H3 windows on the 17-frame grid. The preview interval defaults to 17; custom intervals round up to the next multiple of 17. Click thumbnails to toggle them, or use **Select All** and **Clear Selection**. Resize the node and scroll horizontally for long memories. The saved workflow stores source hashes, slot numbers and selected frames, not the images; refresh previews after reopening. A changed or mismatched source produces a warning, and User mode refuses to run until the selection matches. Clear and select again when intentionally replacing a source.

User mode uses every selected unique window, regardless of the preset, maximum, pins or the multi-file safeguard used by automatic modes. Identical windows from identical source content are sent only once. Each original latent window goes to DiT, and its decoded first frame goes to Qwen. No latent is re-encoded. Other modes retain their existing selection rules and do not require a timeline connection. Cached small previews are kept in process memory (up to 512) and are rebuilt after restart; each window is decoded separately. Selecting many windows may substantially increase inference time and memory use.

Run the frontend contract checks with `node tests/frontend_check.cjs` and `node tests/timeline_frontend.cjs`. These and the Python tests use doubles and do not replace testing with a real ComfyUI installation and H3 VAE.
