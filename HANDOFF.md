# GFM Coasts — Handoff

A research-preview project applying Geographic Foundation Models (GFMs) to coastal erosion along the Catalan coast (Ebro Delta + Maresme). This document is a **standing brief** — a future Claude session (or any reader) can read it once to get oriented quickly without re-litigating decisions that took several iterations to arrive at.

---

## Where we are

| Notebook | Status | One-liner |
|---|---|---|
| `01_setup_and_framing.ipynb` | Done | Project framing (exploratory, not benchmark), AOIs, what a GFM is, May-2026 model landscape, mermaid workflow |
| `02_data_acquisition.ipynb` | Done | AOIs from `src/aois.py` → GeoJSON; S2 trend + Gloria pre/post via STAC; lazy `stackstac`; ICGC MDT 2 m via WCS; CMEMS waves; CHE discharge |
| `03_classical_baseline.ipynb` | Done (5 iterations) | CoastSat-style waterline extraction (re-implemented, not the GEE package); Punta de la Banya demo; transect-based change rates; literature side-by-side |
| `04_gfm_embeddings.ipynb` | **NEXT** | Clay v1.5 embeddings; clustering; embedding-distance change signal |
| `05_gfm_finetune.ipynb` | Conditional | Prithvi-EO-2.0 fine-tune *if Clay justifies it* |
| `06_validation.ipynb` | Pending | ICGC LiDAR-derived shoreline ↔ algorithm shoreline; literature reference values |
| `07_storm_gloria.ipynb` | Pending | Pre/post Gloria; SAR via TerraMind |
| `08_synthesis_and_maps.ipynb` | Pending | Interactive maps, write-up |

---

## Project framing — read this first

We are **not** running a benchmark. We are *not* claiming "GFMs beat CoastSat". We describe what each pipeline produces and place numbers next to *published* CoastSat / DSAS reference values from the literature. The user (James) is new to GFMs and explicitly asked not to make claims he can't defend.

The substantive question is: *what can a GFM reveal about this coast, and how do its outputs sit alongside what classical methods produce?*

This framing matters because it shapes every methodological choice below.

---

## Key methodological decisions (the non-obvious ones)

These took iteration on real data to arrive at. Don't relitigate without reason.

### 1. Re-implement the algorithm, don't install CoastSat the package
CoastSat is built around Google Earth Engine. We already have a clean STAC + `stackstac` pipeline from notebook 02; using it keeps the env light, the data flow consistent, and the algorithm visible on the page. We use the *Vos et al. 2019 algorithm* (MNDWI + Otsu + subpixel `find_contours`) and cite CoastSat's published RMSE figures as the relevant reference point.

### 2. MGRS tile boundaries — mosaic by overpass
Sentinel-2 is delivered in 100 × 100 km MGRS tiles. Our 11 × 11 km Banya AOI straddles `31TBF` / `31TCF`. Picking the cleanest *single granule* by cloud cover gave us only the western half. Fix: group items by date, pass *all* same-overpass granules to `stackstac.stack`, mosaic across time via `ds.median(dim='time', skipna=True)`. Selection logic prefers 2-granule overpasses for full coverage.

### 3. Delta water masks need TWO priors
Punta de la Banya is surrounded by Encanyissada lagoon (~20 km²), Trinitat saltworks, rice paddies, wet ground. The lagoon is bigger than the visible-in-AOI sea, AND the **bocana** (the narrow tidal channel) keeps the lagoon and sea pixel-connected.

- **Morphological opening** (`disk(opening_radius=3)`, erode then dilate, in `isolate_sea_by_edge()`): erodes the bocana so the lagoon and sea become *different* connected components. Reference: Soille (2003) *Morphological Image Analysis*.
- **Seaward-edge prior**: `DEMO_SEAWARD_EDGES = ('south', 'east')` for the Banya demo; `SEAWARD_EDGES` dict in `src/aois.py` for the full AOIs. The open sea must extend offshore beyond the bbox; lagoons don't.

Both priors are needed. Either alone fails on Banya.

### 4. NaN-edge trim, not just AOI-edge trim
Granule cutoffs (single-MGRS-tile coverage) introduce square shoreline residuals because the contour traces the boundary between sea and NaN. Fix in `extract_shoreline_from_items()`: dilate the NaN mask by `edge_buffer` pixels and drop contour points that land on it, alongside the existing AOI-bbox-edge trim.

### 5. Curving baseline, not straight ruler
The Banya spit curves from W-E in the north to NW-SE in the south. A *straight* baseline gives transects that don't face the local coast. Fix: take the most-recent shoreline, `simplify(50)`, use as baseline. Orient transect normals seaward via the `seaward_edges` hint. Position is a signed projection onto the seaward normal — positive = seaward, negative = retreat. This is essentially what DSAS does.

### 6. ICGC MDT 2 m via WCS, not raw LiDAR LAZ
Raw LiDAR LAZ for the Catalan coast across 3 campaigns is tens of GB. We don't need point clouds — we need elevation at the coast for shoreline-from-DEM validation. ICGC publishes MDT 2 m as a derived raster (~30–80 MB per tile, 5 × 5 km tiles, much fewer than raw). `owslib` for WCS endpoint = no manual tile-by-tile clicks. Fall back to MDT 5 m for campaigns that don't publish 2 m (notably 2008).

### 7. Single source of truth for AOIs is `src/aois.py`
Polygons defined as `shapely.box` in Python; notebook 02 generates the GeoJSON at run time. Diffs cleanly in git. AOI extras (descriptions, wave-buoy locations, seaward-edges) all live in `aois.py` too.

### 8. Pandas footgun to remember
Don't name a DataFrame column `item` — `pandas.Series.item` is a method and `r.item` will return the bound method instead of the column value. We use `stac_item` instead.

---

## Repo conventions

- `data/aois/` is git-tracked (small AOI polygons); `data/raw|interim|processed/` ignored
- `src/aois.py` = single source of truth for AOI geometries + metadata
- Notebooks add `src/` to path via `sys.path.insert(0, str((Path('..') / 'src').resolve()))`
- Environment managed by **UV** (`uv sync`). Heavy GFM deps behind `--extra gfm` to keep base install light. Python 3.11. Beyond the initial spec we've added `scikit-image`, `owslib`, and (transitively) `scipy`
- Public repo on GitHub. Intended for the user's website / LinkedIn write-up. Keep documentation lift-and-shift suitable
- Strip pre-run cell outputs when patching .ipynb files — otherwise the file balloons with base64-encoded matplotlib images. The patch scripts use:
  ```python
  for c in nb["cells"]:
      if c["cell_type"] == "code":
          c["outputs"] = []
          c["execution_count"] = None
  ```

---

## User collaboration style

- **Cite the literature** for methodological choices. Already-useful: Boak & Turner 2005 (shoreline definitions), Vos et al. 2019 (CoastSat algorithm), Sánchez-Arcilla et al. 2008/2014 (Ebro Delta), Anthony 2015 (wave-influenced deltas), Soille 2003 (morphological image analysis)
- **Lead with analogies** for new concepts (bocana = corridor between rooms; MNDWI = coloured contact lens; foundation model = polyglot interpreter; STAC catalog = library card catalog for satellites; transect = tape measure; find_contours = tape walker)
- **Bugs reported by what's seen** ("v1 and v2 look identical", "image is in the bottom-left"). Trust the user's geographical intuition — they have spotted real things I missed. Work from observation back to cause
- **Surgical notebook patches**, not full rewrites. Python build scripts that target cells by content predicate
- **Concise responses**; rely on memory and this file for the long brief. The user explicitly tracks tokens

---

## What's next: notebook 04 — Clay v1.5 embeddings

Same scenes as notebook 03 (Banya demo, then full AOIs). Compute per-tile embeddings. Cluster to discover coastal types automatically. Use cosine distance over time as a *labels-free* change signal. Goal: place embedding-distance trends *next to* (not against) the classical change-rate output from notebook 03.

Pre-checks before starting:
1. `uv sync --extra gfm` to install `claymodel` + `terratorch` + `torchgeo`
2. Notebook 03 should run end-to-end first — confirm with the user
3. Strip pre-run outputs in nb 03 if they've ballooned the file

The pedagogical hypothesis the user already expressed (worth carrying through):
> *We had to add a delta-specific fix (morphological opening + seaward-edges prior) to make the classical algorithm work here. Will Clay's embedding-based change detection need a similar geographical prior, or does the foundation model's pre-training already "know" the difference between a sea and a salt pan?*

Author's guess (mine): the embedding distance will be relatively unaffected by inland water bodies — the embedding compresses everything about a tile, so the *difference* over time is what we read, not the absolute embedding value. Worth testing directly.

---

## Loose ends to surface in future notebooks

- For other AOIs (e.g. Sant Pere Pescador, Costa Brava sites with wider bocanas), `opening_radius` may need to differ. Currently a function parameter
- For a *coastal-system* study (sediment budget, ecology of the whole spit-lagoon complex), the lagoon shoreline IS coast. Would need a different algorithm: "all water touching the seaward edge" PLUS "all water connected via the bocana channel". Different question, different design — flag if it ever comes up
- Tide correction is *not yet* applied to the extracted instantaneous waterlines. Mediterranean tides are tiny (~20–30 cm range) but Storm Gloria's surge was ~1 m. Will need to handle for the Gloria zoom in nb 07
- Validation against ICGC MDT (notebook 06) is conditioned on having the WCS endpoint URL nailed down. Currently pyproject has `owslib` but the exact ICGC WCS service URL isn't yet set in code

---

## Pointers
- Project root: `~/Documents/Documents/GitHub/GFM_coasts`
- Public-facing audience: user's website / LinkedIn ("mindfully vibe coded" is their phrase for the project)
- Auto-memory: project + reference + user + feedback files in the user's persistent memory directory; auto-loads in every Claude session
