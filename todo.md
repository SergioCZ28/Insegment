# Insegment - TODO

Tracking remaining work and what's a good fit for the cloud-VM dev session.

## Phase 3 status

Done:
- Proper undo/redo stack (commit 23368ef)
- Progress bar for inference (SSE from backend)
- Multi-image navigation (previous/next)
- Image loading from common formats

Remaining:
- Coordinate precision bug (screenToImage transform) — needs Canvas/visual verification, not cloud-VM friendly
- Online learning (train while labeling) — needs GPU + model + real images, not cloud-VM friendly

## Cloud-VM friendly tasks

The cloud VM has no display, no GPU and no local images. Good fit: backend, refactor, tests, docs.

- [x] Unit tests for `exporters/` (COCO, YOLO, CSV, VOC) — first task
- [x] Add a new export format (LabelMe JSON) + tests + README update
- [ ] Refactor `app.py` (1429 lines, STATE god-object) into smaller modules: `state.py`, `routes_labels.py`, `routes_annotations.py`, etc.
- [ ] Add API input validation on the JSON endpoints (pydantic or manual) with clear error messages
- [ ] Replace `print()` calls with proper `logging` and add a `--verbose` CLI flag
- [ ] Tests for polygon/shape generators (pure math)
- [ ] Expand README with screenshots and example workflow (install/formats already covered)

## Local-only tasks (do NOT assign to cloud VM)

- Canvas UI bugs (drawing, colors, click behavior, coordinate precision)
- Anything that needs to see the annotation tool running in a browser
- Inference / model / training work
