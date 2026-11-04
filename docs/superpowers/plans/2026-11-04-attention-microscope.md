# Attention Heatmap & Logits Probability Microscope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement character-level Attention Heatmap and Logits Top-3 Probability Microscope for the CRT trash-talk generator, allowing users to hover over characters on screen to see neural attention backlighting and inspect candidate token predictions without altering the single-page layout height.

**Architecture:** 
1. **Model Backends (`MLXBackend` & `TorchBackend`)**: Provide an `inspect_sequence(ids: list[int]) -> dict` method that performs a single causal forward pass to extract the attention probability matrix from the final transformer block (averaged across attention heads) and the next-token probability distribution (Top-3 candidates) at each step.
2. **Runtime (`ModelRuntime`)**: During `generate()`, attach a `microscope` payload to the generation result containing filtered visible characters, normalized character-to-character attention weights, and top-3 candidate predictions.
3. **Frontend CRT Display (`index.html`, `style.css`, `app.js`)**: Render generated characters as interactive `.attn-char` spans. Hovering illuminates preceding tokens with phosphor-glow heatmaps according to attention weights and updates the CRT status with top attended tokens; clicking displays a vintage CRT popover showing the model's top 3 candidate predictions.

**Tech Stack:** Python 3.14, MLX (Apple Silicon Metal), PyTorch, Vanilla JavaScript, CSS3.

---

### File Structure Map
- `trash_talk/mlx_backend.py`: Add `inspect_sequence(token_ids)` to compute last-block attention and logits probabilities.
- `trash_talk/torch_backend.py` & `trash_talk/garbage_model.py`: Add `inspect_sequence` for PyTorch backend consistency.
- `trash_talk/model_runtime.py`: Extract and package `microscope` metadata during `generate()`.
- `tests/test_microscope.py`: Comprehensive unit tests verifying attention matrix shapes, normalization, and candidate mapping.
- `web/index.html`: Add `.microscope-bar` within the CRT glass area.
- `web/style.css`: Vintage CRT attention heatmap glow, focus states, and mini popover styling.
- `web/app.js`: Wire character hover and click interactions, tooltip popovers, and preserve text-editing fallback.

---

## Tasks

- [x] **Task 1: Implement `inspect_sequence` in MLX Backend**
  - [x] Add `inspect_sequence` to `trash_talk/mlx_backend.py`.
  - [x] Extract last-block attention map `[length, length]` averaged over attention heads.
  - [x] Extract top-3 predictions and probabilities from logits.

- [x] **Task 2: Implement `inspect_sequence` in PyTorch Model / Backend**
  - [x] Update `trash_talk/garbage_model.py` and `trash_talk/torch_backend.py` with `inspect_sequence`.
  - [x] Ensure parity between MLX and PyTorch outputs.

- [x] **Task 3: Integrate Microscope in `ModelRuntime.generate`**
  - [x] In `trash_talk/model_runtime.py`, call `backend.inspect_sequence(ids)` after generation.
  - [x] Map special tokens, isolate visible characters, and normalize attention rows over visible characters.
  - [x] Map top token IDs to readable characters with probabilities.
  - [x] Write unit tests in `tests/test_microscope.py` and ensure 100% test pass.

- [x] **Task 4: Build Frontend CRT Heatmap & Microscope UI**
  - [x] Update `web/index.html` to add a compact microscope readout in CRT display meta.
  - [x] Update `web/style.css` with `.attn-char`, `--attn-alpha` phosphor heatmap glow, and vintage popup bubble.
  - [x] Update `web/app.js` to render spans, bind hover attention heatmaps, and handle click candidate inspector.
  - [x] Preserve single-page viewport height and contenteditable fallback.

- [x] **Task 5: End-to-End Verification**
  - [x] Run full test suite (`python -m unittest discover tests`).
  - [x] Verify live interaction on `http://127.0.0.1:8765` using Playwright: test hover, click, and screenshot visual appearance.
