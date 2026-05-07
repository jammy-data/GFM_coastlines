# GFM Coasts

Applying Geographic Foundation Models to coastal erosion along the Catalan coast.

## What this is

A research project — not a benchmark — exploring what Geographic Foundation Models (GFMs) can reveal about coastal change in two areas of the Catalan coast:

- **Ebro Delta** — sediment-starved delta retreating at metres per year
- **Maresme** — chronic-deficit pocket beaches north of Barcelona

We look at the decadal trend (2017–2025, Sentinel-2 era) plus a focused zoom on **Storm Gloria** (January 2020).

The goal isn't to claim a GFM is better or worse than established tools. It's to *see what GFMs see*, describe their outputs honestly, and place numerical results next to published reference values from CoastSat / DSAS / DEM-differencing studies so a reader can judge.

## Models

- **Clay v1.5** — open, embeddings-first, S2-native; run first to ask "what does the model see?"
- **Prithvi-EO-2.0** (IBM/NASA) — fine-tuning for water/sand/vegetation segmentation if Clay justifies it
- **TerraMind** (IBM) — multimodal incl. SAR; reach for the storm zoom where SAR matters

## Reference comparisons

For each numerical output (shoreline RMSE, change-rate transects, etc.) we cite a published CoastSat or DSAS figure for the same or analogous coast as a *reference point*, not a target to beat.

## Validation data

We use **ICGC MDT 2 m** (LiDAR-derived bare-earth DEM rasters, three campaigns) — not the raw LiDAR point clouds — to derive ground-truth shorelines via DEM ∩ tidal datum. This keeps total validation storage to a few GB while giving us multi-temporal elevation for change detection. See `notebooks/02_data_acquisition.ipynb` for the rationale.

## Layout

```
GFM_coasts/
├── notebooks/         # Numbered chapters; 01 is the entry point
├── src/               # Reusable Python modules called from notebooks
├── data/
│   ├── raw/           # Untouched downloads (S2, LiDAR, …)
│   ├── interim/       # Cleaned, cropped, reprojected
│   └── processed/     # Analysis-ready (shorelines, embeddings, masks)
├── maps/              # Interactive HTML maps (Folium / Leafmap)
├── reports/           # Markdown / PDF write-ups
├── pyproject.toml     # UV-managed environment
├── .python-version    # 3.11
└── README.md
```

## Getting started

This repo uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
uv sync                                                    # install base deps
uv run jupyter lab notebooks/01_setup_and_framing.ipynb    # open the entry-point notebook
```

To pull in the heavier GFM frameworks (Clay, Prithvi via terratorch, torchgeo):

```bash
uv sync --extra gfm
```

## Status

Active research preview. The framing, AOIs, and model order are all open to revision — see the "Choices" section in `notebooks/01_setup_and_framing.ipynb`.
