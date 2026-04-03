# AI-Assisted Annotation Tool -- Next Session Brief

## What exists
- Flask app (`annotation_server.py`) + Canvas frontend (`templates/index.html`)
- Loads a BacDETR model, runs inference on a microscopy frame, shows detections as green overlays
- User can left-click to add cells, right-click to remove, 1/2/3 to reclassify (SC/clump/debris)
- Export to COCO JSON format
- Runs on localhost:5000

## What works
- Model inference + overlay rendering
- Pan (shift+drag), zoom (scroll), fit-to-screen (F)
- Class selection buttons, opacity slider, detection stats
- Claude (via Chrome MCP) CAN see the frame, identify real cells vs ghost cells, and spot missed detections

## What needs fixing (priority order)

### 1. Click coordinate precision (CRITICAL)
- Claude's clicks land ~13px off from intended target
- Root cause: the `screenToImage()` function doesn't properly account for sidebar width (260px) and canvas transform (pan/zoom state)
- Current workaround: hover to get coordinates via status bar, then call JS API
- Fix: audit the coordinate transform chain (screen -> canvas -> image space)

### 2. Undo system
- Z key / "Undo Last Action" button exists in UI but may not work reliably
- Need: proper undo stack that tracks add/remove/reclassify actions

### 3. Persistence
- Annotations are lost on page refresh
- Need: auto-save to localStorage or server-side JSON

### 4. Better cell markers
- Current: small green circles (hard to see on bright cells)
- Better: semi-transparent polygon masks or outlined contours

## How Claude annotates (current workflow)
1. Open tool in Chrome tab (Chrome MCP)
2. Take screenshot to see the frame
3. Identify cells the model missed (real cells with dark phase-contrast halo, NOT ghost cells which are bright white blobs)
4. Left-click to add annotation at cell location
5. Right-click to remove false positives

## Key insight from testing
- Claude's cell DETECTION is good (correctly identifies missed cells)
- Claude's click PLACEMENT is off (coordinate mapping bug)
- Fix the mapping and this becomes a powerful semi-automatic annotation workflow

## Files
- `annotation_server.py` -- Flask backend (model loading, inference, COCO export)
- `templates/index.html` -- Canvas frontend (all UI + interaction logic)
- Model checkpoint: configurable via `--checkpoint` flag (defaults to v13)

## To run
```bash
conda activate hitmictools
cd C:/Users/sergi/ExperimentsWindows/MyProjects/ai_assisted_labelling
python annotation_server.py
# Opens at http://127.0.0.1:5000
```
