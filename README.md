# CAD2Graph

`CAD2Graph` is a Python toolkit for parsing architectural CAD drawings / floor plans. It supports DXF-to-SVG conversion, spatial element extraction, topological graph construction, semantic enrichment, and generation of JSON-LD results with visualizations.

## Highlights

- ✅ DXF → SVG conversion
- ✅ Structured extraction of drawing elements and text labels
- ✅ Construction of geometric and topological knowledge graphs
- ✅ Pluggable spatial algorithms (contour extraction & space-type recognition)
- ✅ Semantic enrichment with optional LLM inference
- ✅ Batch and single-file processing modes
- ✅ Ground truth generation and model evaluation
- ✅ A reproducible two-task benchmark (`scripts/run_benchmark.py` + `scripts/evaluate_benchmark.py`),
  with the protocol in [docs/benchmark_protocol.md](docs/benchmark_protocol.md)
- ✅ Web graphical interface for single-file workflows (analysis / comparison / evaluation overview)

## Project Structure

```text
CAD2Graph/
├── input_data/
│   ├── dxf/                 # Raw DXF files
│   ├── dxf_gt/              # Manually annotated Ground Truth DXF files
│   ├── pickle/              # External instance segmentation pickle cache
│   └── svg/                 # SVG files converted from DXF
├── output/
│   ├── jsonld/              # Semantic-enriched JSON-LD output
│   ├── bench/               # Benchmark matrix: one dir per (contour algo, classifier)
│   ├── viz/                 # All visualization images (floor plan / SVG instance / experiment / GT)
│   ├── gt/                  # Ground truth JSON-LD annotations
│   ├── html/                # HTML evaluation reports
│   └── processed/           # Intermediate SVGs after modification
├── data/                    # Datasets, trained weights and evaluation artifacts
├── docs/                    # Design and protocol documents
├── prompt/
│   └── prompt_config.txt    # LLM prompt configuration
├── scripts/                 # Dataset export, training, benchmark and reporting entry points
├── src/
│   ├── config/              # Configuration and directory constants
│   ├── enricher/            # Graph enrichment and semantic extensions
│   ├── experiment/          # Experiment entries (parsing / GT / evaluation / bench metrics)
│   ├── io/                  # DXF/SVG reading, writing, and conversion
│   ├── spatial/             # Pluggable spatial layer: contour extraction, space typing, primitives, viz
│   ├── topology/            # Component aggregation, topology construction, graph analysis
│   ├── utils/               # Visualization and utility tools
│   ├── main.py              # Main entry point
│   └── processor.py         # Drawing processing pipeline
└── README.md
```

## Environment and Dependencies

### Setting Up a Virtual Environment

It is **highly recommended** to use a virtual environment to isolate project dependencies. Create and activate one as follows:

```bash
# Create a virtual environment (e.g., named myvenv or cadruler)
python -m venv myvenv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

After activation, install the required packages (see below). To deactivate the virtual environment later, simply run `deactivate`.

### Python Version

Python **3.10+** is recommended. The project has been tested with **Python 3.13**.

### Important: Python 3.13 Removed the `cgi` Module

Starting from **Python 3.13**, the standard library modules `cgi` and `cgitb` have been **removed** (as per [PEP 594](https://peps.python.org/pep-0594/)). This project's `src/web_ui_server.py` imports `cgi`, so **if you are using Python 3.13 or later**, you must install the `legacy-cgi` compatibility package:

```bash
pip install legacy-cgi
```

The existing `cadruler/` virtual environment in this repository already has `legacy-cgi` installed.

### Installing Dependencies

Install the necessary packages:

```bash
pip install ezdxf openai pyyaml matplotlib lxml numpy pandas svgpathtools opencv-python triangle networkx pyvis shapely scikit-learn
```

If you plan to run ground truth generation and evaluation scripts, you may also need:

```bash
pip install shapely
```

> **Note**: A [`requirements.txt`](requirements.txt) is provided at the project root. You can install all dependencies at once with:
>
> ```bash
> pip install -r requirements.txt
> ```

### Environment Variables Configuration (Security Best Practice)

This project uses **environment variables** to manage sensitive information like API keys. This is a security best practice that prevents sensitive data from being committed to version control.

#### Setup Environment Variables

1. **Copy the example environment file:**

```bash
cp .env.example .env
```

2. **Edit the `.env` file and add your configuration:**

```bash
# Required: OpenAI-compatible API key for LLM services
CAD2GRAPH_LLM_API_KEY=your_actual_api_key_here

# Optional: Custom API base URL (overrides settings.yaml)
CAD2GRAPH_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/

# Optional: Custom model name (overrides settings.yaml)
CAD2GRAPH_LLM_MODEL=glm-4-flash
```

**Why manage all LLM configs together?**
- 🔒 **Security**: Prevents sensitive API keys from being committed to version control
- 🚀 **Flexibility**: Easy to switch between different environments (dev/staging/prod)
- 🎯 **Consistency**: All LLM-related configurations in one place
- 🧪 **Experimentation**: Easy to test different models and endpoints

**Loading is automatic** — no `export` required. [src/config/config.py](src/config/config.py) reads
the project-root `.env` the first time the config module is imported, so every entry point
([src/main.py](src/main.py), [src/web_ui_server.py](src/web_ui_server.py), the scripts under
[src/experiment/](src/experiment/)) picks it up. The parsing follows the usual dotenv rules:

- keys already present in the process environment are **never** overwritten, so a shell `export`
  or a CI variable wins over `.env`;
- `#` starts a comment on its own line, and an unquoted value is truncated at a trailing ` #`;
- `export KEY=value` lines and single/double-quoted values are accepted;
- a missing `.env` is silently ignored.

To bypass `.env` entirely — for instance to force the no-LLM path (semantic enrichment
disabled) — set
`CAD2GRAPH_SKIP_DOTENV=1`. Use that flag rather than blanking the key: in PowerShell
`$env:X=""` *deletes* the variable instead of emptying it.

#### Alternative Configuration Methods

If you prefer not to use a `.env` file, you can also set environment variables directly:

**Windows PowerShell:**
```powershell
# Set all LLM configurations
$env:CAD2GRAPH_LLM_API_KEY="your_api_key_here"
$env:CAD2GRAPH_LLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
$env:CAD2GRAPH_LLM_MODEL="glm-4-flash"
```

**Linux/macOS:**
```bash
# Set all LLM configurations
export CAD2GRAPH_LLM_API_KEY="your_api_key_here"
export CAD2GRAPH_LLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
export CAD2GRAPH_LLM_MODEL="glm-4-flash"
```

**Or modify the configuration file** (not recommended for production):
- Edit [src/config/settings.yaml](src/config/settings.yaml) and set values directly in the `llm` section
- ⚠️ This approach is less secure and not recommended for production environments

#### Configuration Priority

The system reads LLM configurations in the following priority order:
1. **Environment variables** (highest priority) - overrides everything; the `.env` file is merged in at this level, filling only the keys that are not already set
2. **Configuration file** ([src/config/settings.yaml](src/config/settings.yaml)) - fallback defaults
3. **Hardcoded defaults** - if neither above is available

## Quick Start

### 1. Prepare DXF Input

Place the DXF files to be processed in:

```text
input_data/dxf/
```

### 2. Convert DXF to SVG

The conversion script reads DXF files from `input_data/dxf/` and outputs SVG to `input_data/svg/`.

From the project root, run the conversion script:

```bash
python -m src.io.dxf_to_svg
```

The converted SVG files will be output to:

```text
input_data/svg/
```

### 3. Run the Drawing Parsing Pipeline (Exp 1-4)

`src/main.py` performs the end-to-end parsing — it extracts elements, builds the
topological knowledge graph, enriches semantics, generates the JSON-LD results
and renders visualizations. This corresponds to the first four evaluation
experiments (Exp 1-4).

You can control the input path and run mode via command-line arguments:

```bash
python -m src.main --mode SINGLE --target-file sample.svg
```

#### BATCH Mode

To process **all** SVG files in a directory at once, use `--mode BATCH` and
point `--target-dir` at the folder containing the SVG files:

```bash
python -m src.main --mode BATCH
```

Optionally, specify a custom output directory with `--output-dir`:

```bash
python -m src.main --mode BATCH --target-dir input_data/svg --output-dir output/jsonld
```

In BATCH mode `src/main.py` will:

- Scan the target directory and pick up every `*.svg` file (files are
  processed in directory order)
- Run the parsing pipeline on each drawing (the same Exp 1-4 steps as
  SINGLE mode)
- Print a summary report at the end with the total number of drawings,
  how many parsed successfully, and how many failed

Alternatively, modify the `runtime` section in the configuration file [src/config/settings.yaml](src/config/settings.yaml) (set `run_mode: BATCH` and the target directory) and then run:

```bash
python -m src.main
```

The program will execute:

- Primitive extraction and element visualization
- Topological graph construction
- Semantic enrichment and JSON-LD output
- Visualization generation

### 4. Generate Ground Truth Data

Place manually annotated or extended DXF files in:

```text
input_data/dxf_gt/
```

Run the ground truth creation script (single interactive mode, or batch over all
DXF files in the directory):

```bash
python -m src.experiment.ground_truth_creator
python -m src.experiment.ground_truth_creator --mode BATCH
```

### 5. Parse All Drawings

Run the parsing entry point to produce the enriched JSON-LD for the whole corpus:

```bash
python -m src.experiment.parsing_pipeline --mode BATCH
```

> The former `dataset_evaluator.py` (Exp 1-4 batch metrics → `output/html/`) has been
> **retired**. It could only ever score a single pipeline configuration, so it could not
> express "contour algorithm × classifier" comparisons. Use the benchmark below instead.
> The files it used to produce (`output/html/*`) are kept on disk but nothing reads them.

### 6. Experiment Entry Points

- `parsing_pipeline.py` — Drawing parsing: element extraction, topology construction,
  semantic enrichment, and visualization. Produces enriched JSON-LD in `output/jsonld/`.
- `ground_truth_creator.py` — Generates human-annotated Ground Truth JSON-LD from
  annotated DXF files (single interactive or batch mode).

```bash
# Ground truth generation (interactive / batch)
python -m src.experiment.ground_truth_creator
python -m src.experiment.ground_truth_creator --mode BATCH

# Parsing (batch)
python -m src.experiment.parsing_pipeline --mode BATCH
```

### 7. Reproducible Benchmark (local · end-to-end)

The pluggable spatial algorithms are compared by `scripts/run_benchmark.py` +
`scripts/evaluate_benchmark.py`, which write one run directory per configuration into
`output/bench/` and a single authoritative `output/bench/scores.json`.

```bash
python scripts/run_benchmark.py --status     # what is already complete
python scripts/run_benchmark.py              # run everything still incomplete
python scripts/evaluate_benchmark.py --out-json output/bench/scores.json
```

**The end-to-end matrix (9 runs of 39 drawings)** is the cascade: *task 2 runs on top of
whatever task 1 produced*, so contour error and classification error stack up.

| Task | Candidates | Everything else held fixed |
| --- | --- | --- |
| 1 · space **contour** extraction | `CDT` / `RGP` / `VecFloorSeg` | classifier = `NoOp`; metrics read `<base>_raw.jsonld` |
| 2 · space **type** recognition | `LLMMultiStage` / `SAGEE` / `TextMatching` | contours = `GT` (this is the **upper bound**) |
| **end-to-end** | the **3 × 3 product** `e2e_<contour>_<classifier>` | task 2 simply consumes task 1's output |

The reported end-to-end accuracy is decomposed, which is what makes the matrix useful:

```
端到端准确率 = 轮廓覆盖 × 条件准确率
             (任务1 找出该 GT 空间了吗) × (找出之后类型判对了吗)
```

A `GT(自检)` row scores GT contours against GT and is **not** a method — it must come out at
≈1.0, proving the matching and metric code are correct.

Full rationale, metric definitions and the numbers: **[docs/benchmark_protocol.md](docs/benchmark_protocol.md)**.
The two interactive views are `overall_report.html` (end-to-end matrix) and
`exp_analysis.html` (algorithm-level + per-drawing inspection).

### 8. Knowledge Graph Browser (multi-view)

`src/utils/kg_browser.py` renders **one self-contained interactive HTML per
drawing** with three tabs, so you can inspect both the whole knowledge graph and
how the enrichment pipeline changes it step by step (replaces the former
`suite_viz.py`):

- **全图 Knowledge Graph** — the whole `@graph` as a force-directed graph: all
  node types (Space / Suite / Door / Window / FunctionalElement) and all relations
  (`bot:adjacentZone` / `bot:containsElement` / `bot:interfaceOf` /
  `bot:hasSpace` / `bot:hasSubZone`), switchable by enrichment stage and filterable
  by edge type; click a node to see its id/group.
- **富化过程 Stages** — step through the enrichment pipeline (raw → semantic →
  geometry → ACD → geometry² → topology), highlighting nodes/edges **added
  (green) / removed (red)** versus the previous stage, plus per-stage stats.
- **套型从属 Suite** — the suite containment report: color-coded floor plan (one
  color per suite; public/unassigned hatched gray), membership table and
  auto-detected anomaly flags (suite count vs. the `<N>suite` expectation, a suite
  swallowing >75% of private spaces, 1-space suites, unassigned spaces).

The stage browser replays the pipeline from `<base>_raw.jsonld` + `<base>.svg`;
single-file mode uses the configured LLM client for a faithful replay
(`--no-llm` falls back to the rule-based sandbox).

```bash
# Single file (faithful LLM replay)
python -m src.utils.kg_browser --base "2suite (1)"

# Sandbox (rule-based) replay
python -m src.utils.kg_browser --base "2suite (1)" --no-llm

# Batch over all drawings
python -m src.utils.kg_browser --mode BATCH --out-dir output/viz
```

Output: `<base>_kg_browser.html` (plus `<base>_suites.png` for the Suite tab) in
`output/viz/`. Requires `vis.js` from CDN (consistent with the existing pyvis
usage).

## Web Graphical Interface

A lightweight web UI is provided for interactive, single-file workflows. Start the server from the project root:

```bash
python -m src.web_ui_server
```

Then open `http://localhost:8001` in a browser. The GUI is one entry page plus three tools:

### Category 1 · Normal Use

- **`normal_use.html`** — End-to-End Drawing Analysis: input a single DXF file, the parsing pipeline runs automatically (DXF→SVG → extraction → topology → semantic enrichment → visualization), and the input drawing, topology image, instance image and enriched JSON-LD are displayed.

### Category 2 · Local / per-drawing analysis

- **`exp_analysis.html`** — Drawing Analysis: pick a single DXF and see both tasks **at algorithm level**.

  **① library-wide comparison** renders the full tables first, so you see the overall picture
  before drilling in. **② Task 1 · contour extraction** gives each algorithm its own row with
  that drawing's metrics and a **zoomable, pannable** contour image; **③ Task 2 · type
  recognition** gives each classifier its own row with a folded type-count table and that
  method's **knowledge graph**; **④ System vs GT** shows the topology pair, a difference
  summary and a side-by-side KG comparison.

  There is **no algorithm dropdown**: the top bar runs *every* combination for the selected
  drawing (see `RUN_PLAN` in the page, which mirrors `scripts/run_benchmark.py`). SAGEE is
  dispatched to `/api/run-holdout-drawing` so it always uses the out-of-fold model.
- **`kg_browser.html`** — Knowledge Graph Browser (**standalone tool**): pick a processed drawing and a source (`System` or `GT`) and load **one knowledge graph at a time** into the interactive browser produced by `src/utils/kg_browser.py` (whole graph / stages / suite tabs). Tick “包含富化过程” to replay the six enrichment stages for the system side (uses the LLM, slower). System vs GT comparison is intentionally left to `exp_analysis.html`.

### Evaluation Overview · end-to-end

- **`overall_report.html`** — **End-to-end comparison**: the 3 × 3 product of *task-1 contour
  algorithms* × *task-2 classifiers*, where **task 2 runs on whatever task 1 produced**. It
  shows the accuracy matrix, the `轮廓覆盖 × 条件准确率` decomposition (so you can tell whether
  the loss comes from extraction or from classification), the two single-task reference
  tables, and per-class F1 for a selected combination.

  It reads `output/bench/scores.json` through `/api/preview` — run
  `scripts/evaluate_benchmark.py` first. The distinction from `exp_analysis.html` is the whole
  point: **this page is the cascade, that page is the isolated algorithms.**

All pages accept a **single file input**; none run batch processing. Images (SVG / PNG) support **zoom & pan**: scroll to zoom, drag to pan, and double-click or the `⟲` button to reset.

### Backend API

| Method & path | Purpose |
|---|---|
| `GET /api/available-dxfs` | List DXF files under `input_data/dxf` |
| `GET /api/spatial-algos` | List selectable spatial algorithms (contour extraction / space-type classification), their defaults and the LLM status |
| `POST /api/run-analysis` | Single DXF end-to-end analysis (Category 1) |
| `POST /api/experiment-draw` | System vs GT comparison |
| `POST /api/upload-dxf` | Upload DXF files |
| `GET /api/preview?path=` | Preview a file (SVG / PNG / HTML / JSON) |
| `GET /api/kg-files` | List datasets for the KG browser (system JSON-LD + matching GT) |
| `GET /api/kg-browser?base=&src=&stages=&refresh=&dir=&tag=` | Generate/serve the KG browser HTML. `src=system`\|`gt`; `stages=1` replays the enrichment pipeline; `dir=` overrides the directory holding the system `<base>.jsonld` (e.g. `output/bench/type_SAGEE`); `tag=` suffixes the generated file so several runs of the same drawing do not overwrite each other |
| `POST /api/run-holdout-drawing` | Re-run one drawing with the **out-of-fold** SAGEE model that never saw it. Never use `/api/experiment-draw` for SAGEE — that would produce an in-sample result |

The single-DXF endpoints accept optional `contourAlgo`, `classifierAlgo`
and `forceReparse` fields (the same names as the CLI `--contour-algo` /
`--classifier-algo` arguments); an unknown algorithm name is rejected with HTTP
400. Responses echo back `contourAlgo` / `classifierAlgo` / `llmEnabled` /
`llmModel` together with `reused`, which tells the UI whether the shown result
was re-parsed or reused from the cache.

> Note: `manual.html` is a legacy pure-frontend prototype (canvas graph editor) and retains its built-in zh/en toggle.

## Key Configuration Files

- [src/config/settings.yaml](src/config/settings.yaml) — Input, output, prompt, and `spatial` algorithm settings
- [prompt/prompt_config.txt](prompt/prompt_config.txt) — LLM prompt template

## Pluggable Spatial Algorithms

Two pipeline tasks are isolated behind abstract interfaces so that new algorithms of
the same kind can be added without touching the pipeline:

| Task | Interface | Registry | Default |
| --- | --- | --- | --- |
| Spatial contour extraction | `ISpatialContourExtractor` | `CONTOUR_EXTRACTORS` | `CDT` |
| Space type recognition | `ISpaceTypeClassifier` | `SPACE_TYPE_CLASSIFIERS` | `LLMMultiStage` |
| Text label normalization | `ITextLabelNormalizer` | `TEXT_LABEL_NORMALIZERS` | `LLM` |

All of them live in [src/spatial/](src/spatial/); the pipeline only talks to the
factory in [src/spatial/factory.py](src/spatial/factory.py) and to the algorithm-neutral
domain objects in [src/spatial/domain.py](src/spatial/domain.py). JSON-LD details are
confined to the two adapters, [src/topology/builder.py](src/topology/builder.py) and
[src/enricher/semantic_enricher.py](src/enricher/semantic_enricher.py).

Everything the spatial pipeline needs sits in that package, so the algorithm-adjacent
code is no longer scattered over `topology/` and `utils/`:

| Module | Responsibility |
| --- | --- |
| [contracts.py](src/spatial/contracts.py) / [domain.py](src/spatial/domain.py) | Interfaces and the plain objects algorithms exchange |
| [registry.py](src/spatial/registry.py) / [factory.py](src/spatial/factory.py) / [config.py](src/spatial/config.py) | Name resolution, construction, `settings.yaml` parsing |
| [primitives.py](src/spatial/primitives.py) | Boundary-primitive preparation (`clean_lines`) — the shared input of every contour algorithm |
| [contours/](src/spatial/contours/) | Contour extraction algorithms (`cdt.py`, `rgp.py`, `vecfloorseg.py`) plus the shared `filters.py` |
| [vecfloorseg/](src/spatial/vecfloorseg/) | Input/output bridge for the VecFloorSeg baseline — runs in a separate conda environment, never imports torch |
| [classifiers/](src/spatial/classifiers/) | Space type recognition algorithms (`llm_multistage.py`, `text_matching.py`, `sagee.py`) |
| [sagee/](src/spatial/sagee/) | SAGE-E baseline: graph construction, 8/5-dim features, label space, and a **pure-numpy** forward pass |
| [visualization.py](src/spatial/visualization.py) | Algorithm-neutral plot of an extraction result (`plot_floor_plan`) |

[src/topology/builder.py](src/topology/builder.py) is now pure orchestration
(boundary primitives → extractor → BOT graph), and component aggregation moved out of it
into [src/topology/components.py](src/topology/components.py).

The old module paths below were removed rather than forwarded, so any historical script
using them must be pointed at the new location:

| Removed path | Where the code lives now |
| --- | --- |
| `src.topology.preprocessing.clean_lines` | [src/spatial/primitives.py](src/spatial/primitives.py) |
| `src.topology.generate_virtual_wall.FloorPlanMeshBuilderCDT` | dropped — use `CDTContourExtractor` in [src/spatial/contours/cdt.py](src/spatial/contours/cdt.py), or `create_contour_extractor("CDT")` |
| `src.utils.cdt_viz.plot_floor_plan` | [src/spatial/visualization.py](src/spatial/visualization.py) |

Select an algorithm from the command line (any registered name or alias):

```bash
# list everything that is registered
python -m src.experiment.parsing_pipeline --list-algos

# use the LLM-driven classifier (default) or the geometry-only dictionary matcher
python -m src.main --mode SINGLE --target-file sample.svg --classifier-algo LLMMultiStage
python -m src.main --mode SINGLE --target-file sample.svg --classifier-algo TextMatching

# use the SAGE-E graph neural network classifier (needs a trained model, see below)
python -m src.main --mode SINGLE --target-file sample.svg \
    --contour-algo RGP --classifier-algo SAGEE

# use the CDT triangulation extractor (default) or rule-based geometric polygonization
python -m src.main --mode SINGLE --target-file sample.svg --contour-algo CDT
python -m src.main --mode SINGLE --target-file sample.svg --contour-algo RGP

# use the VecFloorSeg neural baseline (needs a separate conda env + trained .ckpt,
# see "VecFloorSeg Baseline" below)
python -m src.main --mode SINGLE --target-file sample.svg --contour-algo VecFloorSeg
```

Or configure it in the `spatial` section of [src/config/settings.yaml](src/config/settings.yaml):

```yaml
spatial:
  contour:
    algorithm: "CDT"
    params:                     # parameters shared by every contour algorithm
      min_area_mm2: 2000000.0
      # …
    algorithm_params:           # per-algorithm block, only merged when that algorithm is selected
      RGP:
        snap_tol_mm: 10.0
        max_gap_mm: 3000.0
      VecFloorSeg:
        device: "cuda:0"
        canvas_px: 512
  classification:
    algorithm: "LLMMultiStage"
    params:
      match_tolerance_mm: 0.0
      min_confidence: 0.5
```

`params` keys must be accepted by the constructor of the *selected* algorithm, otherwise
creation fails fast with a `TypeError`. Algorithm-specific knobs therefore belong in
`algorithm_params.<NAME>` (matched case-insensitively, aliases included) — that way you can
switch between algorithms freely without having to comment parameters in and out.

The factory resolves `algorithm_params` against the **final** algorithm name, so the block
still applies when the algorithm is overridden at runtime (`--contour-algo `, the Web UI
dropdown, or `create_contour_extractor("RGP")`) and not just when `settings.yaml` selects it.
`scripts/check_vecfloorseg_config.py` guards this behaviour.

### Available Contour Extractors

| Name | Aliases | How it works |
| --- | --- | --- |
| `CDT` | `CDT_MESH`, `TRIANGLE_CDT` | Constrained Delaunay triangulation of the wall/door-constrained mesh; rooms are grown over the triangle graph and door/window openings are sealed with virtual blocker edges. |
| `RGP` | `RULE_BASED`, `POLYGONIZE`, `GEOMETRIC_POLYGONIZATION` | Rule-based geometric polygonization — purely combinatorial, no CDT mesh. |
| `VecFloorSeg` | `VFS`, `VECTORFLOORSEG` | Neural baseline (third-party): a two-stream graph attention network over a wireframe image + Delaunay-derived region graph. Runs out-of-process in its own conda environment. |
| `GT` | `GROUNDTRUTH`, `GT_CONTOUR` | **Not a method** — reads `output/gt/<base>_gt.jsonld`, converts the annotated WKT polygons into spaces, and **relabels every space `Unknown`** so no GT type can leak to a classifier. Exists to give Task 2 a common starting point, and to self-check the metrics. |

The `GT` extractor needs `context['base_name']` (supplied by `TopologyBuilder`) and raises
without it. It also checks that its bounding box aligns with the extracted walls
(`min_align_iou`), so a mis-paired GT cannot silently produce a plausible-looking result.
Its self-check row in the benchmark must come out at ≈1.0 on coverage / one-to-one / mIoU;
if it does not, the evaluator is wrong, not the algorithm.

`RGP` ([src/spatial/contours/rgp.py](src/spatial/contours/rgp.py)) consumes the same predicted
boundary primitives (wall segments + door/window patches) and runs:

1. **Coordinate normalization** — quantize to `coord_precision_mm`.
2. **Duplicate removal** — drop zero-length/degenerate segments (`dup_tol_mm`).
3. **Endpoint snapping** — weld endpoints that are within `snap_tol_mm` (with cluster arithmetic means).
4. **Collinear segment merging** — unify touching/overlapping intervals on one line (`collinear_angle_tol_deg`, `collinear_offset_tol_mm`).
5. **Short-gap completion** — two routes: bridge a gap between two collinear wall lines whose endpoints keep going away in the bridge direction, and seal door/window openings with an edge parallel to the opening edge (`max_gap_mm`, `gap_angle_tol_deg`).
6. **Segment intersection and splitting** — insert every crossing point so segments only meet at endpoints.
7. **Planar graph construction** — build a rotation system (sorted neighbours per node).
8. **Polygonization** — trace every inner face of the planar embedding.
9. **Small/slender-region filtering** — drop implausible rooms via the shared `SpaceShapeFilter` (`min_area_mm2`, `erode_mm`, `min_width_mm`, `min_compactness`, `min_solidity`).
10. **Exterior-face removal** — discard unbounded faces so only real rooms remain.

All three contour algorithms share the room-plausibility defences in
[src/spatial/contours/filters.py](src/spatial/contours/filters.py), so the same
`params` block applies to either one.

Trade-off to be aware of: on the drawings tested, RGP's room set is a strict refinement of
CDT's (it never invents a room outside a CDT room), and it correctly splits several rooms
that CDT merges through unsealed openings. Because step 9 *filters* rather than *absorbs*,
the sub-`min_area_mm2` leftovers (wall cavities, stair/duct cells) are reported as no room
instead of being folded into a neighbour, so the total extracted area is a few percent
smaller than CDT's.

### VecFloorSeg Baseline

`VecFloorSeg` is a **neural** baseline kept alongside the two geometric ones so that the
same drawings can be compared across paradigms. It is a two-stream graph attention network
over (a) a rasterized wireframe image and (b) a region graph derived from a constrained
Delaunay triangulation; the region graph is then merged along walls before being labelled.

It is the only contour algorithm that is **not self-contained**, because it needs PyTorch:

| Aspect | `CDT` / `RGP` | `VecFloorSeg` |
| --- | --- | --- |
| Dependencies | numpy + shapely (already in the venv) | PyTorch + `torch_geometric` + `triangle` |
| Execution | in-process | **out-of-process**, in its own conda env |
| Extra artefact | — | a trained `.ckpt` |

The whole third-party stack is therefore quarantined behind a subprocess boundary:
[src/spatial/contours/vecfloorseg.py](src/spatial/contours/vecfloorseg.py) only orchestrates,
and the code that actually touches torch lives in [src/spatial/vecfloorseg/](src/spatial/vecfloorseg/)
under `build_dataset.py`. The CAD2Graph environment itself never imports torch.

```
CAD2Graph env                                   vecfloorseg env
──────────────                                  ────────────────────────
walls/doors/windows (mm)
   │
   ├─ geometry.build_spec()  ── spec.json ──▶   build_dataset.py
   │                                                 │
   │                                        (Delaunay + region merge)
   │                                                 ▼
   │                                            pkl + PNG
   │                                                 │
   │                                          graphgym/main.py --eval
   │                                                 ▼
   │◀── result.pkl + triangles.npz ─────────────────┘
   │
   ├─ postprocess.regions_to_contours()  → list[SpatialContour]
   └─ SpaceShapeFilter  → same plausibility defences as CDT/RGP
```

The original pipeline consumes SVG (`model.svg → SVGParserCUBI → …`), but that parser
only understands the CubiCasa SVG convention. Since the triangulation really only needs a
PSLG — and CAD2Graph's wall centrelines *are* a PSLG — the bridge skips SVG parsing and
feeds the geometry straight into the author's `triangulate → _genTriangleGraph →
buildDualRelationship → graphCrune` chain. The upstream algorithm code is reused verbatim.

**Setup.** Install the environment, then point the adapter at it (three variables, all
machine-specific, so keep them in `.env` rather than in `settings.yaml`):

| Variable | Meaning |
| --- | --- |
| `VECFLOORSEG_PYTHON` | Interpreter of the vecfloorseg conda env |
| `VECFLOORSEG_CKPT` | Path to the trained `best1.ckpt` |
| `VECFLOORSEG_ROOT` | VecFloorSeg checkout (defaults to `third_party/VecFloorSeg`) |

Run `python scripts/check_vecfloorseg_config.py` to confirm the wiring without running any
inference. See [docs/vecfloorseg_adapter_design.md](docs/vecfloorseg_adapter_design.md) for
the full design, the verified I/O contract, and the upstream pitfalls.

⚠️ The upstream `load_ckpt()` returns silently when the checkpoint file is missing, which
would run the model with **random weights** and no error. The adapter validates the path
up front instead.

### Available Space Type Classifiers

| Name | Aliases | What it uses | Needs training |
| --- | --- | --- | --- |
| `LLMMultiStage` | `LLM`, `MULTISTAGE` | drawing text + LLM common-sense reasoning + furniture | no |
| `TextMatching` | `TEXT`, `GEOMETRY_ONLY` | drawing text only (offline dictionary) | no |
| `SAGEE` | `SAGE`, `SAGE_E` | **geometry + topology only — never reads the drawing text** | **yes** |
| `NoOp` | `NONE`, `NOCLS`, `CONTOUR_ONLY` | nothing — returns an empty prediction list | no |

`NoOp` exists for **Task 1** (contour extraction) runs. The pipeline needs *some* classifier,
and falling back to the `settings.yaml` default (`LLMMultiStage`) would burn a full LLM pass
for an experiment that is not about typing — worse, the enricher's ACD step would then
**split** multi-typed spaces and change the very contours being measured. `NoOp` keeps the
run pure. It accepts `match_tolerance_mm` / `min_confidence` / `verbose` so it is drop-in
compatible with shared config blocks.

`SAGEE` is a port of the SAGE-E graph neural network (`SPACE_TYPE_CLASSIFIERS` name
`SAGEE`), which classifies rooms from a room-adjacency graph. It fills the gap the
other two leave: spaces with **no text label at all** (stairwells, elevator shafts,
corridors) can still be typed.

* Architecture: 4 layers, 15,394 parameters, ~15 lines of maths. Runs **in-process
  with numpy** — no torch, no DGL, no subprocess.
* Nodes = extracted space contours, edges = `bot:adjacentZone`. Door/window edges come
  from `bot:interfaceOf` on component nodes.
* Features (our own 8/5-dim schema, defined in [src/spatial/sagee/features.py](src/spatial/sagee/features.py)):
  node = `log_area`, `area_ratio`, `perimeter_m`, `short_side_m`, `aspect_ratio`,
  `n_doors`, `n_windows`, `n_furniture`; edge = `is_door`, `is_window`,
  `boundary_len_m`, `boundary_ratio`, `center_dist_m`.

**A trained model is a bundle that must travel together:**

| File | Purpose | Env var | Required |
| --- | --- | --- | --- |
| `<name>.npz` | weights | `SAGEE_WEIGHTS` | yes |
| `<name>.labels.json` | class table — index misalignment is silent otherwise | `SAGEE_LABELS` | yes |
| `<name>.meta.json` | training recipe stamp (`normalize`, `class_weights`, `epochs`, …) | — | yes |
| `<name>.norm.json` | per-column z-score — **only if trained with `--normalize`** | `SAGEE_NORM` | no |

The classifier refuses to run when the label file is missing, and refuses when
`meta.json` says `normalize: true` while the norm file is absent. Conversely it
**ignores** a norm file when `meta.json` says `normalize: false` — that guards a trap
that bit us once: `SageeNumpy` auto-loads the `<weights>.norm.json` sidecar, so a
leftover file from an earlier `--normalize` run gets silently applied to a model that
never trained with it. `train_sagee.py` deletes the stale sidecar on export
(`drop_stale_norm`) and `check_sagee_config.py` self-checks the bundle.

⚠️ **Train and inference must use the same contour algorithm.** SAGE-E's ceiling is set
by the contour partition: on CDT the extractor merges the whole open-plan area into one
contour, which makes several ground-truth classes unreachable. Feeding an RGP-trained
model GT contours is an out-of-distribution evaluation and unfairly deflates it
(measured `0.607` acc / `0.357` macro-F1, versus `0.754` / `0.565` after retraining on
GT contours). The shipped model is trained on **GT** contours.

⚠️ **Reproduce the paper's recipe, do not "improve" it.** The paper's notebook uses no
z-score and plain cross-entropy, so `--normalize` / `--class-weights` are **off by
default** and their help text says so. Measured, the faithful recipe is also *better*:
`0.6004 ± 0.0366` accuracy vs `0.5551 ± 0.0921` for the tuned one.

```bash
# 1) export a training set from the ground truth (GT contours)
#    export_sagee_dataset.py reads <name>_raw.jsonld from --jsonld-dir, so point it at
#    a run directory produced with --contour-algo GT.
python scripts/export_sagee_dataset.py --jsonld-dir output/bench/type_SAGEE \
    --out data/cad2graph_sagee_gtc --space CORE

# 2) 5-fold cross-validation + export the per-fold models and the deployment model
python scripts/train_sagee.py --dataset data/cad2graph_sagee_gtc.npz --cv 5 \
    --epochs 200 --select final \
    --labels-from data/cad2graph_sagee_gtc.labels.json \
    --emit-folds data/holdout_gtc \
    --export-weights data/sagee_gtc.npz

# 3) self-check the deployed bundle
python scripts/check_sagee_config.py
```

See [docs/sagee_baseline_plan.md](docs/sagee_baseline_plan.md) for the full design and
the verified data contract. **Never quote an in-sample number** — the tell-tale sign is
that the predicted class distribution matches the training set exactly. The honest
protocol is the 5-fold holdout: 5 folds cover all 39 drawings, each drawing is predicted
by a model that never saw it, and the concatenation is a full out-of-sample **online**
result directly comparable to the zero-shot classifiers.

#### Comparing classifiers and contours honestly

Evaluation lives behind two scripts and **one document**:

* `scripts/run_benchmark.py` — runs the matrix and drops every run into `output/bench/<run>/`,
  so a directory name alone identifies the exact configuration.
* `scripts/evaluate_benchmark.py` — discovers those directories by name prefix and writes
  `output/bench/scores.json`, the single authoritative result file.
* **[docs/benchmark_protocol.md](docs/benchmark_protocol.md)** — the protocol itself:
  why the two tasks are kept separate, why Task 1 uses `NoOp` and Task 2 uses `GT`
  contours, how the metrics are defined, and the current measured results.

```bash
python scripts/run_benchmark.py --status          # where am I
python scripts/run_benchmark.py                   # run the 6 incomplete runs
python scripts/evaluate_benchmark.py --out-json output/bench/scores.json
```

The rules below are enforced in code because breaking any one of them silently flips the
conclusion:

1. **One drawing set.** A benchmark run is only complete when it covers all 39 keys.
   `sample` is excluded by user request and `6suite (5)` because its GT file is empty
   (`{"@graph": []}`). Both scripts default to `--exclude sample` — keep them identical.
2. **Abstentions count as false negatives.** Dropping the abstain column from the F1
   denominator inflated `TextMatching`'s macro-F1 from `0.462` to `0.722` (**+56%**) —
   a classifier that says nothing looked better. `accuracy` is always reported with
   `coverage`.
3. **Fold the GT labels first.** `CORE` unifies the three corridor classes. Comparing
   against raw GT types reported `71.4%` and counted correct predictions as errors
   (true: `82.9%`). The folding table ships inside `scores.json` (`label_space.folding`)
   so the UI folds both sides with the same rule.
4. **No cross-product contamination.** Task 1 reads `<base>_raw.jsonld` (written *before*
   enrichment, so its spaces and geometry are the same whichever classifier ran — verified:
   identical file size, `@id` set, space count and WKT across the three `type_*` runs, with
   only the *order* of the unordered relation lists differing, as recorded under **Notes**);
   Task 2 reads `<base>.jsonld`. The two tasks never share a run directory.
5. **The innovation stays with the innovator.** ACD composite-space splitting is this
   project's contribution and SAGE-E has no such capability, so giving it to a baseline
   for free would not be a comparison between methods. It is gated by
   `ISpaceTypeClassifier.supports_composite_split` — `True` only for `LLMMultiStage`.
   A/B check (same RGP contours + `TextMatching`): 126 spaces with ACD off vs **142**
   with it on.

The `GT(自检)` row in the Task-1 table scores GT contours against GT and is deliberately
**not** counted as a method. It must come out at ≈1.0 — if it does not, the evaluator is
broken, not the algorithm:

```
GT(自检)   n_sys 1916   coverage 0.9995   one_to_one 0.9990   mIoU 1.0000
```

**Headline results** (39 drawings, 1916 GT spaces, `CORE` 16 classes — full tables and the
per-class F1 breakdown are in [docs/benchmark_protocol.md](docs/benchmark_protocol.md)):

| Task 1 · contour | n_sys | count ratio | coverage | 1-to-1 | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| `CDT` | 1205 | 0.629 | 0.547 | 0.483 | **0.860** |
| `RGP` | 1606 | **0.838** | **0.656** | **0.545** | 0.754 |

| Task 2 · type (on GT contours) | accuracy | coverage | MacroF1 | WeightedF1 |
| --- | ---: | ---: | ---: | ---: |
| `LLMMultiStage` | 0.731 | 0.902 | **0.699** | **0.778** |
| `SAGEE` (out-of-fold) | **0.755** | **0.999** | 0.566 | 0.732 |
| `TextMatching` | 0.449 | 0.469 | 0.429 | 0.532 |

CDT and RGP are a **trade-off, not a ranking**: CDT gets the shapes right (mIoU +10.6
points) while RGP finds more of the rooms (coverage +10.9 points, one-to-one +6.2). Which
one to pick depends on whether the downstream stage fears *missing a space* or *getting its
shape wrong*. Likewise, `SAGEE` wins on accuracy but loses on macro-F1 because it is weak
on rare classes, while `LLMMultiStage` is the other way round — they are complementary,
not competing.

#### Secondary harnesses

Two earlier evaluators are kept because they answer different questions. Their published
numbers come from the **RGP-contour era** and are superseded — re-run them before quoting.

* `scripts/evaluate_per_gt_space.py` — the original per-GT-space evaluator, and the only
  one with `--ensemble "A,B"` (first non-abstaining classifier wins) plus a
  complementarity matrix. Its most durable finding **still holds**: `TextMatching` and
  `SAGEE` are almost orthogonal. Out of 1839 GT spaces, 391 were right in both,
  **304 only in `SAGEE`, 301 only in `TextMatching`** — nearly equal, almost
  non-overlapping gains. A one-line "text first, SAGE-E as fallback" rule reached 0.523
  against an oracle union ceiling of 0.542, i.e. **96% of the available combination gain**,
  with SAGE-E's contribution concentrated on spaces that carry no text at all.
* `scripts/evaluate_space_types.py` — the earliest harness; takes `--run NAME=DIR`
  (repeatable) and `--restrict-to DIR` to force an identical drawing set.
  `scripts/inspect_space_sets.py` reproduces the split-behaviour measurement that
  motivated the per-GT-space denominator.

⚠️ `SAGEE`'s `min_confidence` default must stay at **0.0**. SAGE-E's softmax has a long
low tail, so thresholding at `0.5` only discards correct answers (measured: accuracy
`0.378 → 0.321`, coverage `0.680 → 0.461`).

### Adding a New Algorithm

1. Implement the interface in `src/spatial/contours/` (for contour extraction) or
   `src/spatial/classifiers/` (for space typing), and decorate the class with the
   matching registry, e.g. `@CONTOUR_EXTRACTORS.register("MyAlgo", aliases=("MY",))`.
   Constructor parameters must be keyword arguments; unknown keyword arguments raise
   `TypeError` at creation time, so typos in `settings.yaml` fail fast. Set a class-level
   `name` and, for contour extractors, a `visualization_suffix` (e.g. `"myalgo"`) — the
   pipeline uses it to name its floor-plan plot `output/viz/<drawing>_myalgo.png`.
2. Import the module from the package `__init__.py` so the registration runs.
3. Point `spatial.contour.algorithm` / `spatial.classification.algorithm` (or the
   `--contour-algo` / `--classifier-algo` arguments) at the new name, and put any
   algorithm-specific parameters under `spatial.contour.algorithm_params.<NAME>`.

Algorithm instances can also be injected directly — `TopologyBuilder(contour_extractor=…)`,
`GraphEnrichmentPipeline(…, classifier=…)` and `process_single_drawing(…, classifier=…)`
all accept a pre-built object for callers that manage the lifecycle themselves. The
historical `FloorPlanMeshBuilderCDT` class and its module (`src/topology/generate_virtual_wall.py`)
were both removed; use `CDTContourExtractor` itself.

## Run Modes

The main entry point supports the following modes, configurable via the settings file or command-line arguments:

- `SINGLE` — Process a single SVG file
- `BATCH` — Process all SVG files in a specified directory

### Common Arguments

- `--mode` — Run mode
- `--target-dir` — SVG input directory name
- `--target-file` — File name for single-file mode
- `--output-dir` — Output directory name
- `--contour-algo` — Spatial contour extraction algorithm (default from `settings.yaml`)
- `--classifier-algo` — Space type recognition algorithm (default from `settings.yaml`)
- `--list-algos` — Print the registered spatial algorithms and exit (`parsing_pipeline` only)
- Environment variables: `CAD2GRAPH_RUN_MODE`, `CAD2GRAPH_TARGET_DIR`, `CAD2GRAPH_TARGET_FILE`, `CAD2GRAPH_OUTPUT_DIR`, `CAD2GRAPH_CONTOUR_ALGO`, `CAD2GRAPH_CLASSIFIER_ALGO`
  (the legacy `CAD_RULE_CHECKER_*` names are still accepted for backward compatibility)

## Output Directory Overview

- `output/jsonld/` — Semantic-enriched JSON-LD output
- `output/bench/` — **Benchmark matrix**: one directory per configuration —
  `contour_<algo>/`, `type_<classifier>/` and `e2e_<contour>_<classifier>/` — each covering all
  39 drawings with `<base>.jsonld` (enriched) and `<base>_raw.jsonld` (pre-enrichment).
  `scores.json` is the authoritative evaluation result.
  See [docs/benchmark_protocol.md](docs/benchmark_protocol.md).
- `output/viz/` — All visualization images (floor-plan contours `<drawing>_<algorithm>.png` — e.g. `_cdt.png` / `_rgp.png` / `_vecfloorseg.png` / `_gt.png` — plus SVG instance, experiment, and GT previews)
- `output/gt/` — Ground truth JSON-LD annotations
- `output/html/` — **Legacy** reports from the retired `dataset_evaluator.py`; nothing reads them any more
- `output/processed/` — Intermediate SVGs after svg_modifier processing

## Notes

- `src/main.py` contains LLM client initialization code. Replace or refactor it to use a secure API key management approach before deployment.
- This project is currently designed for **experimental** drawing parsing and result visualization. The pipeline can be extended for production use cases as needed.
- Without an LLM API key the semantic stage falls back to a built-in sandbox mapping
  (`次卧`/`主卫`/`餐客厅` plus a fixed stage-3 inference table), which is why the default
  `LLMMultiStage` classifier recognises very few space types on real drawings.
- The BOT topology relations (`bot:adjacentZone`, `bot:interfaceOf`, `bot:containsElement`)
  are built from unordered sets, so their order — and therefore `*_topology.png` — is
  **not reproducible between runs**. Compare JSON-LD by node content, not by byte.
- In `LLMMultiStage`, the three `[Stage n]` headers are printed while the classifier runs,
  so all `[+] …` result lines appear after the last header rather than interleaved.

## Utility Scripts

- `scripts/clean_generated.py` — delete all code-generated files under `output/`
  in one shot (JSON-LD, visualizations, reports, GT JSON-LD,
  intermediate SVGs) while keeping the directory structure. `input_data/` is
  never touched (raw DXF, annotated GT DXF, converted SVGs, pickle cache all
  preserved).

  ```bash
  python scripts/clean_generated.py --dry-run   # preview only
  python scripts/clean_generated.py --yes       # delete immediately
  ```

- `scripts/check_ui_endpoints.py` — check the web UI backend endpoints.
- `scripts/run_step_test.py` — run a single pipeline step.

### Benchmark

- `scripts/run_benchmark.py` — run the 6-run benchmark matrix into `output/bench/<run>/`.
  `--status` shows coverage per directory, `--dry-run` shows the plan, `--task contour|type`
  and `--run NAME` narrow it down, `--force` re-runs a complete run.
- `scripts/evaluate_benchmark.py` — evaluate the matrix into `output/bench/scores.json`.
  Discovers runs by directory prefix, emits per-drawing and aggregate metrics plus a
  `GT(自检)` self-check row.
- `src/experiment/bench_metrics.py` — the shared evaluation core (matching, per-GT-space
  pooling, confusion matrix, metrics). Import it rather than re-deriving the rules.

### SAGE-E / VecFloorSeg plumbing

- `scripts/export_sagee_dataset.py` — build a SAGE-E training set from a run directory's
  `_raw.jsonld` plus the GT annotations.
- `scripts/train_sagee.py` — 5-fold CV, per-fold export (`--emit-folds`) and deployment
  export (`--export-weights`). `--normalize` / `--class-weights` deviate from the paper
  and are off by default.
- `scripts/run_holdout_eval.py` — leave-one-fold-out online evaluation; `--out-name` writes
  straight into a benchmark run directory.
- `scripts/check_sagee_config.py` — 6-point self-check of the deployed bundle.
- `scripts/inspect_vecfloorseg_ckpt.py` / `scripts/convert_vecfloorseg_ckpt.py` — read and
  repair GraphGym checkpoints (their keys can carry a redundant middle `.module.` segment,
  which `load_ckpt(strict=True)` rejects).
- `scripts/audit_label_vocabulary.py` — count raw vs folded `bldg:` types per run directory
  and flag anything outside the label space.
- `scripts/report_cv_f1.py` — per-class F1 straight from the CV confusion matrix, which
  separates "the model cannot do it" from "the threshold hid it".
- `scripts/compare_space_types.py` / `scripts/inspect_sagee_matching.py` — single-drawing
  sanity checks for the SAGE-E predictions and its system↔GT matching.
- `scripts/check_vecfloorseg_config.py` / `scripts/patch_vecfloorseg_pyg23_compat.py` /
  `setup_vecfloorseg_env.ps1` / `setup_vecfloorseg_server.sh` — the third-party baseline's
  config guard, its PyG 2.3 compatibility patch, and environment setup.

## Suggested Improvements

- Add sample data and result demonstrations

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
