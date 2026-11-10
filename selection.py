"""Deterministic selection of phase-aligned H3 scene-memory windows."""

import re
import numpy as np

from .scene_memory import SceneMemoryError

MODES = ("Single", "Uniform", "Interval", "Smart", "Manual")
PRESETS = {"Dialogue": 2, "Balanced": 3, "Geometry": 6, "Detailed": 8, "Dense": 10}
MULTI_FILE_LIMIT = 32  # Extension safeguard, not a verified H3 model limit.


def candidates(sources):
    # sources are (slot, video, identity); only complete, phase-zero windows.
    return [(slot, frame) for slot, video, _ in sources
            for frame in range(0, 17 * ((video.shape[2] - 2) // 5) + 1, 17)]


def parse_frames(text, sources, warnings, label):
    slots = {slot: video for slot, video, _ in sources}
    if not text or not text.strip():
        return []
    result = []
    duplicates = 0
    for token in re.split(r"[,;\s]+", text.strip()):
        match = re.fullmatch(r"(?:(\d+):)?(\d+)", token)
        if not match:
            raise SceneMemoryError(f"{label}: use frame numbers like 0, 187 or file:frame like 2:187.")
        slot = int(match.group(1) or 1)
        frame = int(match.group(2))
        if slot not in slots:
            raise SceneMemoryError(f"{label}: memory file {slot} is not connected.")
        video = slots[slot]
        total = 5 + 17 * ((video.shape[2] - 2) // 5)
        if frame >= total:
            raise SceneMemoryError(f"{label}: frame {frame} is outside file {slot} (0–{total - 1}).")
        key = (slot, (frame // 17) * 17)
        if key in result:
            duplicates += 1
        else:
            result.append(key)
    if duplicates:
        warnings.append(f"{label}: {duplicates} duplicate aligned window(s) removed.")
    return result


def _features(image):
    # Analyze a fixed small grid only; the Qwen and DiT references are not resized.
    a = np.asarray(image, dtype=np.float32)
    if a.ndim != 3 or a.shape[-1] < 3 or not np.isfinite(a).all():
        raise SceneMemoryError("Smart mode requires finite RGB frames from the H3 video VAE.")
    y = np.linspace(0, a.shape[0] - 1, 32).astype(np.int64)
    x = np.linspace(0, a.shape[1] - 1, 32).astype(np.int64)
    rgb = np.clip(a[y][:, x, :3], 0, 1)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = np.clip(dx + dy, 0, 1)
    # Composition, colour and structure; sharpness is a simple edge-energy proxy.
    descriptor = np.concatenate((rgb.reshape(-1), edges.reshape(-1)))
    return descriptor, float(edges.mean())


def _smart(pool, chosen, wanted, render, warnings):
    if not pool or len(chosen) >= wanted:
        return chosen
    feature = {key: _features(render(key)) for key in dict.fromkeys(chosen + pool)}
    sharp_values = [feature[k][1] for k in pool]
    lo, hi = min(sharp_values), max(sharp_values)
    while pool and len(chosen) < wanted:
        ranked = []
        for key in pool:
            vector, sharp = feature[key]
            distances = [float(np.mean(np.abs(vector - feature[prev][0]))) for prev in chosen]
            diversity = min(distances) if distances else 0.0
            # Similarity is only a low-resolution heuristic, not recognition.
            if chosen and diversity < 0.008:
                continue
            spans = [abs(key[1] - prev[1]) for prev in chosen if prev[0] == key[0]]
            file_frames = [k[1] for k in pool + chosen if k[0] == key[0]]
            length = max(file_frames, default=0) or 17
            coverage = min(spans) / length if spans else 1.0
            sharpness = (sharp - lo) / (hi - lo) if hi > lo else 0.0
            score = 0.65 * diversity + 0.20 * coverage + 0.15 * sharpness
            ranked.append((score, -key[0], -key[1], key))
        if not ranked:
            warnings.append("Smart mode skipped visually near-identical windows; fewer references were selected.")
            break
        key = max(ranked)[-1]
        chosen.append(key)
        pool.remove(key)
    return chosen


def select(sources, mode, max_references, memory_frame, frame_interval, manual_frames, pinned_frames, render):
    if mode not in MODES:
        raise SceneMemoryError(f"Unknown memory mode: {mode}.")
    if isinstance(max_references, bool) or not isinstance(max_references, int) or max_references < 1:
        raise SceneMemoryError("Max References must be a positive integer.")
    if isinstance(frame_interval, bool) or not isinstance(frame_interval, int) or frame_interval < 1:
        raise SceneMemoryError("Frame Interval must be a positive integer.")
    warnings = []
    all_keys = candidates(sources)
    # Never select the same window twice from byte-identical source files.
    identity = {slot: source_id for slot, _, source_id in sources}
    canonical = {}
    for key in all_keys:
        canonical.setdefault((identity[key[0]], key[1]), key)
    def dedup(keys):
        out, seen = [], set()
        duplicates = 0
        for key in keys:
            sig = (identity[key[0]], key[1])
            if sig in seen:
                duplicates += 1
            else:
                seen.add(sig); out.append(key)
        if duplicates:
            warnings.append(f"{duplicates} duplicate window(s) removed.")
        return out
    pins = dedup(parse_frames(pinned_frames, sources, warnings, "Pinned Frames"))
    unique = list(canonical.values())
    if len(unique) < len(all_keys):
        warnings.append(f"{len(all_keys) - len(unique)} duplicate window(s) from identical files removed.")
    cap = MULTI_FILE_LIMIT if len(sources) > 1 else len(unique)
    wanted = max_references if mode != "Single" else 1
    if len(pins) > wanted:
        warnings.append("Pinned windows exceed Max References; pins take priority.")
        wanted = len(pins)
    if wanted > cap:
        warnings.append(f"Requested {wanted} windows; selecting at most {cap} available or allowed windows.")
        wanted = cap
    if len(pins) > wanted:
        warnings.append(f"Pinned windows exceed the {cap}-window limit; later pins are omitted.")
    chosen = pins[:wanted]
    seen = {(identity[s], f) for s, f in chosen}
    if len(chosen) >= wanted:
        return chosen, warnings
    pool = [k for k in unique if (identity[k[0]], k[1]) not in seen]
    if mode == "Single":
        automatic = parse_frames(str(memory_frame), sources, warnings, "Memory Frame")
    elif mode == "Manual":
        automatic = parse_frames(manual_frames, sources, warnings, "Manual Frames")
        if not automatic and not chosen:
            raise SceneMemoryError("Manual mode requires Manual Frames or Pinned Frames.")
    elif mode == "Interval":
        automatic = []
        for slot, video, _ in sources:
            total = 5 + 17 * ((video.shape[2] - 2) // 5)
            automatic.extend((slot, (f // 17) * 17) for f in range(0, total, frame_interval))
        automatic = dedup(automatic)
    elif mode == "Uniform":
        # Spread target positions over the complete original candidate sequence.
        automatic = [unique[i] for i in np.linspace(0, len(unique) - 1, min(wanted, len(unique))).round().astype(int)] if unique else []
        automatic += unique  # fill gaps created by pins or rounding
    else:
        return _smart(pool, chosen, wanted, render, warnings), warnings
    for key in automatic:
        sig = (identity[key[0]], key[1])
        if sig not in seen and len(chosen) < wanted:
            chosen.append(key); seen.add(sig)
    if len(chosen) < wanted:
        warnings.append(f"Selected {len(chosen)} of {wanted} requested windows.")
    return chosen, warnings
