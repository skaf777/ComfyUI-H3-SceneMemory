# ComfyUI-H3-SceneMemory

Save and load the original MiniMax H3 **stage 1 video latent** without decoding or re-encoding it. This first stage provides two ComfyUI nodes; it does not yet apply memory to conditioning or claim to improve scene consistency.

## Install

1. Place this repository in `ComfyUI/custom_nodes/ComfyUI-H3-SceneMemory` (for example, clone it into `custom_nodes`).
2. Restart ComfyUI. Find **Save Scene Memory** and **Load Scene Memory** under `MiniMax H3/Scene Memory`.

The extension uses PyTorch, NumPy and safetensors already required by a normal ComfyUI installation. No models or extra model weights are installed. Install and use a revision of this repository containing the desired changes.

## Save a scene

Connect the **video_latent** output of **Separate AV Latent** after the first H3 sampler to **Save Scene Memory**. In the workflow discussed in the issue this is node **#156**, downstream of sampler **#150**. Leave its existing connections in place: this is an additional branch.

Enter a filename, such as `room_take_01`, and run the graph. The node writes `ComfyUI/output/scene_memory/room_take_01.safetensors` and returns the input latent unchanged plus the saved path. It does not save audio, run a VAE, or modify the supplied tensor. Filenames are restricted to a single filename and existing files are never overwritten. Use a new filename for each take. This is an output node: disable/bypass it or change the name before a later queue run to avoid an existing-file error.

Use only an original Stage 1 H3 video latent. The node verifies H3 tensor structure, but the tensor itself cannot prove its model, source sampler, or stage: `source=stage1` records what the user connected. Do not connect the AV pair, an LTX latent, or the Stage 2 output. Stage 2 support and source identification are outside this first stage.

## Load a scene

Select the file in **Load Scene Memory**; refresh/reopen the node list in ComfyUI if a newly saved file is not yet shown. The output is a standard video-only `LATENT` (`{"samples": tensor}`), on CPU. It is intended for a future **Apply Scene Memory** node. It is not a complete audio/video sampling latent and must not replace the target latent in the existing H3 sampler.

Files contain a single `video_latent` tensor and string metadata: format version, source, shape, dtype, latent dimensions, inferred pixel resolution and frame count, FPS and SHA-256 of the exact tensor bytes. The loader checks safetensors parsing, tensor structure and finite values, metadata agreement and checksum. Files with altered or missing information are rejected; no pixel or VAE fallback is performed. The checksum detects accidental data changes, not malicious modification of both tensor and checksum. The format does not identify the actual H3 checkpoint or VAE, so it cannot guarantee model compatibility or provenance.

The supported stage-one geometry is batch 1, 24 channels, `[1, 24, T, H/16, W/16]`, floating-point `float16`, `bfloat16`, or `float32`, even positive latent spatial dimensions, and H3's full-clip `T = 5k + 2` temporal grid. Frame count is inferred as `17k + 5` at 24 fps; this is the generated H3 frame count, not necessarily the number of frames in a separately trimmed/exported video. Output resolution is inferred from latent dimensions, not probed from a video file.

## Tests

In an environment with ComfyUI's Python dependencies installed:

```bash
python -m unittest discover -s tests -v
```

Tests use real PyTorch when available, including a bit-exact bfloat16 test. If PyTorch is absent, the tests use a clearly identified NumPy tensor double with the real safetensors serializer; this exercises the file format but is **not** a substitute for running the tests with real PyTorch and an H3/ComfyUI workflow. No model is needed for the unit tests.

## Design references

The video-only original-latent approach follows the architecture described in [H3 Continuity Studio Visual Memory](https://github.com/KadoBOT/ComfyUI-H3-Continuity-Studio/blob/e6a37dcf36d9be541a5cbde0667720836e51b615/guide/continuity/visual-memory.md). The separated AV safetensors and metadata pattern was informed by [OBVPM MCtx](https://github.com/chanon/comfyui-obvpm-timeline/blob/068a97b4786daef8d67277fe323297fb86d54a3f/mctx.py). This repository implements its own scene-memory format and does not copy OBVPM source or claim `.mctx.safetensors` compatibility.
