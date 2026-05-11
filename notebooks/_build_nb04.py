"""Build script for notebooks/04_gfm_embeddings.ipynb.

Idempotent: re-running regenerates the notebook from scratch.
Cells are authored here as Python strings so changes diff cleanly in git
(rather than as opaque JSON inside the .ipynb).
"""

import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "04_gfm_embeddings.ipynb"


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.lstrip("\n").splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.lstrip("\n").rstrip() .splitlines(keepends=True) or [""],
    }


cells = []

# =========================================================================
# 1. Title + framing
# =========================================================================
cells.append(md("""
# Notebook 4 — Geographic Foundation Model: Clay v1.5 embeddings

**What we do here.** Take the *same yearly Sentinel-2 scenes* we used in notebook 3 to extract Punta de la Banya's shoreline, feed them to **Clay v1.5** (a Geographic Foundation Model, GFM), and read out **embeddings** — high-dimensional vector summaries of each scene. Use cosine distance between embeddings over time as a *labels-free* change signal, and place the result *next to* (not against) notebook 3's transect-based shoreline change rates.

**Why this is interesting, in one sentence.** Notebook 3 collapses each scene to a 1-D waterline curve and reads everything through a single binary "is this pixel water?". Notebook 4 compresses the entire 11 × 13 km scene into ~1024 numbers and asks "did the whole tableau change?" — surfacing beach width, vegetation on the spit, lagoon turbidity, sediment plumes, saltworks state, *all at once*. Those are things the classical pipeline is structurally blind to.

**The pedagogical question we test along the way.** Notebook 3 needed *two* geographical priors bolted onto the classical algorithm to handle Banya's confounding lagoon and saltworks (the morphological opening on the bocana, the seaward-edges hint). Does Clay's pre-training already encode the difference between sea and lagoon and salt pan — i.e. is the geographical prior baked in for free — or will Clay's embedding-distance curve also drift on inland-water changes, the way our classical algorithm did before we patched it?
"""))

# =========================================================================
# 2. Literature
# =========================================================================
cells.append(md("""
## Where this method comes from

This is not the first application of foundation-model embeddings + temporal distance for change detection. Worth knowing the precedents:

- **Hassan / Element84 (2023)** — applied a self-supervised Sentinel-2 model (SSL4EO-S12 ResNet-18, 512-dim) to disaster AOIs (Pakistan floods, Turkey earthquake, California wildfire). Used PCA + a sin/cos seasonal fit, then read residuals as the change signal. A blog post, not peer-reviewed, but the cleanest worked demonstration of the pattern. We borrow the PCA-visualisation idea here.
- **Tang et al. 2024** — "Exploring Foundation Models in Remote Sensing Image Change Detection: A Comprehensive Survey" (arXiv:2410.07824). The framing reference for where foundation-model CD sits in the literature.
- **Sebai et al. 2025 (T&F)** — integrated Prithvi-EO into unsupervised CD workflows for landslide identification. Peer-reviewed evidence the pattern works on natural-disaster-style change.
- **Maldivian islands paper (arXiv:2511.10177, Nov 2025)** — fine-tuned Prithvi-EO-2.0 on 225 Sentinel-2 chips for coastline segmentation; F1 > 0.94 with just 5 labelled images. That's the *supervised* foundation-model story; it's what notebook 5 would do *if* this notebook's unsupervised result justifies a finetune.

Reference points for the Ebro Delta specifically:

- **Pintó et al. 2021** (Remote Sensing) — LiDAR + GPR study of La Banya's dune field. Establishes the *known* spatial pattern: **northern half erosive, southern half accretionary**. Our notebook 3 transect-rate bar chart should reproduce this, and our notebook 4 spatial map should too.
- **Sánchez-Arcilla et al. 2023** (Coastal Engineering) — satellite observations of storm erosion and recovery on the Ebro Delta coastline. Our 2020 embedding-distance spike (Storm Gloria) should agree at least qualitatively with their reported retreat / recovery.

What we *are* claiming as ours, and what we are *not*:

- **Not novel methodology** — the embedding-distance pattern is established.
- **Possibly novel application**: a sediment-starved delta with a confounding saltworks/lagoon background, *with the explicit test of whether the foundation model needs a delta-specific prior the way our classical algorithm did*. That question is what the literature above hasn't asked of this geography.
"""))

# =========================================================================
# 3. Concepts (FM)
# =========================================================================
cells.append(md("""
## What is a "Geographic Foundation Model"?

> A *foundation model* is a large neural network pre-trained on a broad, unlabelled dataset to produce **general-purpose features** of its input, designed to be adapted (or used as-is) for many downstream tasks without retraining from scratch.

> **Brief explainer.** A foundation model is like a *polyglot interpreter*. It has learned the underlying grammar of its input — for Clay, the grammar of how multispectral satellite pixels relate to land cover, terrain, season, location — well enough that you can hand it a place it has never seen and ask "tell me what's distinctive about this place" without ever defining your own classes. The "polyglot" part is that Clay v1.5 specifically handles inputs from Sentinel-2, Landsat, NAIP, LINZ, Sentinel-1 SAR, and MODIS through one shared interface, the way a multilingual interpreter handles seven languages without switching dictionaries.

**Clay v1.5 specifics**, from the model card:

- **Architecture**: Vision Transformer, 632M parameters total (~311M in the encoder we'll actually use). Pre-trained as a *masked autoencoder*: hide 75% of input patches, predict them back, plus a DINOv2 teacher contributing 5% of the loss for representation quality.
- **Training data**: 70 million globally distributed 256 × 256 image chips, sampled by global land-use/land-cover proportions, collected at ≤ 6 different times per location across multiple sensors.
- **Output for Sentinel-2**: a **1024-dimensional** vector for each chip in some learned feature space.

Known limitation that matters for us, from Clay's own model card: *"We do not train on open ocean."* Banya is mostly water from many chips' point of view. We'll flag predominantly-water chips and treat them with caution, but we won't exclude them — the model has seen *coastal* waters, and the question of how its representations behave near vs. far from shore is itself part of what we're investigating.
"""))

# =========================================================================
# 4. Concepts (embedding)
# =========================================================================
cells.append(md("""
## What is an "embedding"?

> An *embedding* is a fixed-length numerical vector — for Clay's Sentinel-2 head, 1024 numbers — that summarises a chip of imagery in a way the model has learned to find useful.

> **Brief explainer.** An embedding is the model's *opinion* about a place, written down as 1024 numbers. The numbers individually mean nothing to us — they are coordinates in a space the model invented for itself during pre-training. What matters is the *pattern*: two chips the model thinks "look like the same kind of place" produce embeddings whose numbers vary in similar ways. We never see the rules the model is using; we only see the output it gives when shown a particular chip.

We get two kinds of embedding from each chip in **one** forward pass:

1. **Scene embedding** (the *CLS token*) — one 1024-D vector per chip. This is what we'll use as the single "what does this place look like, overall" number.
2. **Patch embeddings** — 32 × 32 = 1024 separate 1024-D vectors, one per 8 × 8-pixel patch (each ~80 m on the ground at 10 m S2 resolution). This is the spatial map *inside* a chip.

For notebook 4 we lead with **scene embeddings**. Patch embeddings come into their own when we zoom in on Storm Gloria in notebook 7, where the question is *where inside one chip did the storm hit hardest*.

> **What's a CLS token?** In a Vision Transformer, the input image is cut into patches, each becomes a "token", and the transformer mixes them through attention. The "CLS token" is an extra learned vector prepended to the patch tokens; after attention, it has effectively summarised the whole image into one place. It's the model's natural answer to "give me one vector for the whole chip".
"""))

# =========================================================================
# 5. Concepts (cosine distance)
# =========================================================================
cells.append(md("""
## How do we compare two embeddings? Cosine distance

> *Cosine similarity* between two vectors is the cosine of the angle between them, `(a · b) / (‖a‖ · ‖b‖)`. *Cosine distance* is `1 − cosine_similarity`, valued in [0, 2].

> **Brief explainer.** Picture each embedding as an arrow pointing outward from the origin in 1024-D space (which we can't visualise, but the geometry works in any dimension). Two arrows pointing in the same direction describe scenes that, to the model, *feel* the same — cosine distance 0. Two arrows pointing in opposite directions describe scenes about as different as anything in its experience — distance 2. Distance 1 is perpendicular: "unrelated".

Why cosine and not Euclidean distance? Because the *length* of an embedding mostly tracks brightness/atmosphere — a hazy day at Banya vs. a clear one — while the *direction* tracks content. We care about content, so we measure by angle.

```python
def cosine_distance(a, b, eps=1e-12):
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(1.0 - np.dot(a, b))
```
"""))

# =========================================================================
# 6. Imports & paths
# =========================================================================
cells.append(md("""
## Setup
"""))

cells.append(code('''
# Imports + project paths
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import xarray as xr
import shapely.geometry as sg
import stackstac
import pystac_client
import planetary_computer as pc
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path('..').resolve()
SRC = PROJECT_ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from aois import SEAWARD_EDGES  # noqa: E402

DATA = PROJECT_ROOT / 'data'
PROCESSED = DATA / 'processed'
EMBEDDINGS_DIR = PROCESSED / 'embeddings'
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

CLAY_CKPT_PATH = DATA / 'raw' / 'clay-v1.5.ckpt'
CLAY_METADATA_PATH = SRC / 'metadata.yaml'

warnings.filterwarnings('ignore', category=UserWarning)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {DEVICE}')
print(f'Will look for Clay weights at: {CLAY_CKPT_PATH.relative_to(PROJECT_ROOT)}')
'''))

# =========================================================================
# 7. Load Clay v1.5 weights
# =========================================================================
cells.append(md("""
## Loading Clay v1.5

The weights are ~1.25 GB. We download them once into `data/raw/clay-v1.5.ckpt` (the `data/raw/` folder is git-ignored), then cache there. The checkpoint is hosted by the Clay Foundation on Hugging Face and licensed Apache-2.0.
"""))

cells.append(code('''
# Download the Clay v1.5 checkpoint if not already cached.
import urllib.request

if not CLAY_CKPT_PATH.exists():
    CLAY_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt'
    print(f'Downloading Clay v1.5 weights to {CLAY_CKPT_PATH.relative_to(PROJECT_ROOT)}...')
    urllib.request.urlretrieve(url, CLAY_CKPT_PATH)
    print(f'  ...done, {CLAY_CKPT_PATH.stat().st_size / 1e9:.2f} GB')
else:
    print(f'Clay v1.5 weights already cached ({CLAY_CKPT_PATH.stat().st_size / 1e9:.2f} GB).')
'''))

cells.append(code('''
# Load the model. The `dolls` / `doll_weights` kwargs come from Clay's own
# embeddings tutorial; they configure the Matryoshka representation learning
# losses used at training time. For inference they're benign.
from claymodel.module import ClayMAEModule

module = ClayMAEModule.load_from_checkpoint(
    checkpoint_path=str(CLAY_CKPT_PATH),
    model_size='large',
    metadata_path=str(CLAY_METADATA_PATH),
    dolls=[16, 32, 64, 128, 256, 768, 1024],
    doll_weights=[1, 1, 1, 1, 1, 1, 1],
    mask_ratio=0.0,    # no masking at inference
    shuffle=False,
)
module = module.to(DEVICE).eval()

n_params_enc = sum(p.numel() for p in module.model.encoder.parameters())
print(f'Clay v1.5 encoder loaded: {n_params_enc / 1e6:.0f}M parameters, on {DEVICE}.')
'''))

# =========================================================================
# 8. Sentinel-2 metadata & band ordering
# =========================================================================
cells.append(md("""
## Sentinel-2 metadata + band ordering

Clay's `metadata.yaml` uses friendly band names (`blue`, `green`, `red`, ..., `swir22`). The STAC catalog (Microsoft Planetary Computer) labels the same bands by their Sentinel-2 IDs (`B02`, `B03`, ..., `B12`). We need a translation table so the channel dimension we feed Clay is in the order Clay was trained on.

We also pull from the metadata file:

- **Wavelengths** in nanometres — Clay's dynamic embedding block uses these as input so the same model can serve sensors with different band sets.
- **Per-band mean/std** in raw S2 reflectance units (×10 000 scale) for normalisation.
- **GSD** (ground sampling distance) = 10 m for S2.
"""))

cells.append(code('''
with open(CLAY_METADATA_PATH) as f:
    CLAY_METADATA = yaml.safe_load(f)

S2_META = CLAY_METADATA['sentinel-2-l2a']
CLAY_S2_BAND_ORDER = S2_META['band_order']

# Friendly Clay names -> official S2 band IDs used by Planetary Computer
CLAY_TO_STAC_S2 = {
    'blue':     'B02',
    'green':    'B03',
    'red':      'B04',
    'rededge1': 'B05',
    'rededge2': 'B06',
    'rededge3': 'B07',
    'nir':      'B08',
    'nir08':    'B8A',
    'swir16':   'B11',
    'swir22':   'B12',
}
STAC_S2_BAND_ORDER = [CLAY_TO_STAC_S2[b] for b in CLAY_S2_BAND_ORDER]

# Wavelengths are stored in micrometres in metadata.yaml; Clay expects nm.
S2_WAVELENGTHS_NM = [S2_META['bands']['wavelength'][b] * 1000 for b in CLAY_S2_BAND_ORDER]
S2_MEANS = [S2_META['bands']['mean'][b] for b in CLAY_S2_BAND_ORDER]
S2_STDS  = [S2_META['bands']['std'][b]  for b in CLAY_S2_BAND_ORDER]
S2_GSD   = float(S2_META['gsd'])

print(f'Clay expects {len(CLAY_S2_BAND_ORDER)} S2 bands (GSD {S2_GSD} m), in this order:')
for clay, stac, wl in zip(CLAY_S2_BAND_ORDER, STAC_S2_BAND_ORDER, S2_WAVELENGTHS_NM):
    print(f'  {clay:<10} -> {stac:<4} @ {wl:>6.0f} nm')
'''))

# =========================================================================
# 9. Demo AOI + scene selection (mirror nb03)
# =========================================================================
cells.append(md("""
## Demo AOI + yearly scene selection

Same Banya box as notebook 3 — `box(0.65, 40.55, 0.78, 40.65)` — and the same year-by-year cleanest-summer-overpass selection logic. This is deliberately a copy of notebook 3 cells 29–30; we'll factor it into `src/scenes.py` in a follow-up so both notebooks share one selection function.

Reusing the same selection guarantees notebook 3 and notebook 4 are reading from the *identical set of 9 yearly Sentinel-2 scenes*. Any divergence in their conclusions then has to be about the methods, not the input data.
"""))

cells.append(code('''
DEMO_AOI = sg.box(0.65, 40.55, 0.78, 40.65)              # Punta de la Banya
DEMO_SEAWARD_EDGES = SEAWARD_EDGES['ebro_delta']         # ('south', 'east')

catalog = pystac_client.Client.open(
    'https://planetarycomputer.microsoft.com/api/stac/v1',
    modifier=pc.sign_inplace,
)

# Mirror of nb03 cells 29-30: yearly cleanest summer overpass, mosaicked across granules.
trend = list(catalog.search(
    collections=['sentinel-2-l2a'],
    intersects=DEMO_AOI,
    datetime='2017-07-01/2025-09-15',
    query={'eo:cloud_cover': {'lt': 10}},
).item_collection())
print(f'{len(trend)} candidate granules across all dates')

trend_by_date = defaultdict(list)
for i in trend:
    trend_by_date[i.datetime.date()].append(i)

trend_summary = pd.DataFrame([
    {'date': d,
     'year': d.year,
     'month': d.month,
     'n_granules': len(its),
     'mean_cloud': float(np.mean([i.properties['eo:cloud_cover'] for i in its])),
     'stac_items': its}
    for d, its in trend_by_date.items()
])

summer = trend_summary[trend_summary.month.isin([6, 7, 8, 9])].copy()
summer['coverage_rank'] = (summer.n_granules >= 2).astype(int)
selection = (summer.sort_values(['coverage_rank', 'mean_cloud'], ascending=[False, True])
                   .groupby('year', as_index=False).head(1)
                   .sort_values('year').reset_index(drop=True))

print(f'\\nSelected {len(selection)} overpasses (one per year):')
print(selection[['date', 'n_granules', 'mean_cloud']].to_string(index=False))
'''))

# =========================================================================
# 10. Mosaic helper
# =========================================================================
cells.append(md("""
## Mosaicking a single overpass into an `xarray.DataArray`

We need the imagery as a `(band, y, x)` array on a UTM grid (so 10 m pixels match Clay's GSD). For an overpass that spans two MGRS tiles we mosaic via `stackstac.stack(...).median(dim='time')` — same trick as notebook 3, but selecting **all 10 Clay bands** in Clay's expected order rather than just `B03` + `B11`.
"""))

cells.append(code('''
def mosaic_overpass(items, aoi):
    """Stack all granules from one date, mosaic, return (10, y, x) DataArray in UTM 31N."""
    ds = stackstac.stack(
        items,
        epsg=32631,
        resolution=10,
        bounds_latlon=aoi.bounds,
        assets=STAC_S2_BAND_ORDER,
        rescale=False,
        chunksize=2048,
    )
    arr = ds.median(dim='time', skipna=True)        # mosaic across granules
    arr = arr.compute()                              # pull the actual pixels
    return arr  # shape: (band, y, x) ordered as STAC_S2_BAND_ORDER

# Sanity-check on the first selected date
test_arr = mosaic_overpass(selection.iloc[0]['stac_items'], DEMO_AOI)
print(f'First overpass arr shape: {dict(test_arr.sizes)}')
print(f'Bands (Clay-order): {list(test_arr.band.values)}')
'''))

# =========================================================================
# 11. Chip concept + tiler
# =========================================================================
cells.append(md("""
## Chipping the AOI

> A *chip*, in this notebook's vocabulary, is the model's native unit of input: a 256 × 256 pixel image. At Sentinel-2 10 m resolution, that's **2.56 km on the ground**.

> **Brief explainer.** A chip is to a foundation model what a standard photo print is to an album: it's the size the model was trained to look at. Clay won't naturally accept "the whole Banya AOI" (~11 × 13 km) in one go any more than an album expects a single 13-metre print. We *tile* our scene into chips, embed each, and aggregate after.

We use **stride 128** — i.e. chips overlap by 50%. Each chip is independently embedded, so the only cost of overlap is a few more forward passes; in exchange we get a finer-grained spatial map of "where in the AOI did the embedding change?" than a non-overlapping tiling would give us.
"""))

cells.append(code('''
CHIP_SIZE = 256       # Clay's native input
CHIP_STRIDE = 128     # 50% overlap

def tile_into_chips(arr, chip_size=CHIP_SIZE, stride=CHIP_STRIDE):
    """Tile an (band, y, x) DataArray into chips.

    Returns
    -------
    chips : np.ndarray, shape (n_chips, n_bands, chip_size, chip_size)
    chip_bboxes_utm : np.ndarray, shape (n_chips, 4) -- (xmin, ymin, xmax, ymax) in UTM
    chip_centroids_lonlat : np.ndarray, shape (n_chips, 2) -- (lon, lat) of chip centre
    """
    import pyproj
    n_bands, ny, nx = arr.shape
    y_starts = list(range(0, ny - chip_size + 1, stride))
    x_starts = list(range(0, nx - chip_size + 1, stride))

    # Pixel-edge coordinates from xarray for UTM->lonlat conversion
    x_coords = arr.x.values
    y_coords = arr.y.values
    transformer = pyproj.Transformer.from_crs('EPSG:32631', 'EPSG:4326', always_xy=True)

    # IMPORTANT: never drop chips here. The chip grid must be IDENTICAL across years
    # so that scene_embs[year_a][i] and scene_embs[year_b][i] refer to the same
    # physical location. NaN-coverage is tracked as per-chip `valid_frac` and
    # exposed downstream so the analysis can weight or exclude low-coverage chips.
    chips, bboxes, centroids, valid_fracs = [], [], [], []
    for y0 in y_starts:
        for x0 in x_starts:
            chip = arr.isel(y=slice(y0, y0 + chip_size),
                            x=slice(x0, x0 + chip_size)).values.copy()
            valid_frac = float(np.mean(np.isfinite(chip)))
            # Fill NaNs with band-wise median (or 0 if the band is entirely NaN)
            for b in range(n_bands):
                bm = chip[b]
                mask = ~np.isfinite(bm)
                if mask.any():
                    bm[mask] = float(np.nanmedian(bm)) if np.any(np.isfinite(bm)) else 0.0
            chips.append(chip.astype(np.float32))
            # UTM bbox -> lonlat centroid
            xmin, xmax = float(x_coords[x0]), float(x_coords[x0 + chip_size - 1])
            ymax, ymin = float(y_coords[y0]), float(y_coords[y0 + chip_size - 1])  # y descends
            bboxes.append((xmin, ymin, xmax, ymax))
            cx_utm, cy_utm = (xmin + xmax) / 2, (ymin + ymax) / 2
            lon, lat = transformer.transform(cx_utm, cy_utm)
            centroids.append((lon, lat))
            valid_fracs.append(valid_frac)
    return (np.stack(chips),
            np.asarray(bboxes, dtype=np.float64),
            np.asarray(centroids, dtype=np.float64),
            np.asarray(valid_fracs, dtype=np.float32))

# Sanity-check on the test overpass
test_chips, test_bboxes, test_cents, test_valid = tile_into_chips(test_arr)
print(f'Tiled into {len(test_chips)} chips of shape {test_chips.shape[1:]} ',
      f'(stride {CHIP_STRIDE}px = {CHIP_STRIDE * 10}m)')
print(f'Valid-pixel fraction per chip: min={test_valid.min():.2f}, '
      f'median={np.median(test_valid):.2f}, max={test_valid.max():.2f}')
'''))

# =========================================================================
# 12. Building the model input dict
# =========================================================================
cells.append(md("""
## Building the model input dict

Clay's encoder wants a Python dict with five keys:

- `pixels` — `(batch, 10, 256, 256)` float32, normalised per-band.
- `time` — `(batch, 4)` = `[sin(week), cos(week), sin(hour), cos(hour)]`. Encoding the day of year as `(sin, cos)` makes January-1 and December-31 *close* in the metadata space (they're a day apart in the year), which is correct for a *seasonal* prior. Hour-of-day captures sun angle; for daytime S2 scenes we pass the scene's actual UTC hour.
- `latlon` — `(batch, 4)` = `[sin(lat), cos(lat), sin(lon), cos(lon)]` with lat/lon **in radians**. Same trick: cyclic encoding so longitudes 179° and -179° are near each other in the metadata space.
- `waves` — list of 10 wavelengths in nm. We pre-computed this from `metadata.yaml`.
- `gsd` — scalar 10.0 m.

Per Clay's own embeddings tutorial it's acceptable to pass zeros for `time` / `latlon` when unknown, but the embeddings are *richer* when real values are passed (the dynamic position encoding uses these). We pass real values; flipping to zeros later is a clean ablation.
"""))

cells.append(code('''
def build_batch(chips, chip_centroids_lonlat, scene_datetime, device=DEVICE):
    """Build the dict Clay's encoder consumes for one overpass.

    Parameters
    ----------
    chips : (B, 10, 256, 256) float32 -- raw S2 reflectance
    chip_centroids_lonlat : (B, 2) -- (lon, lat) in degrees per chip
    scene_datetime : pd.Timestamp / datetime -- the overpass time
    """
    B = chips.shape[0]
    # Normalise pixels
    means = np.asarray(S2_MEANS, dtype=np.float32).reshape(1, -1, 1, 1)
    stds  = np.asarray(S2_STDS,  dtype=np.float32).reshape(1, -1, 1, 1)
    pix = (chips - means) / stds
    pix_t = torch.from_numpy(pix).to(device)

    # Time encoding: week of year in [0, 52), hour in [0, 24)
    week = scene_datetime.isocalendar().week / 52.0 * (2 * np.pi)
    hour = scene_datetime.hour / 24.0 * (2 * np.pi)
    time_vec = np.array([np.sin(week), np.cos(week), np.sin(hour), np.cos(hour)], dtype=np.float32)
    time_t = torch.from_numpy(np.tile(time_vec, (B, 1))).to(device)

    # Latlon encoding: per-chip centroid in radians, sin/cos
    lat = np.deg2rad(chip_centroids_lonlat[:, 1]).astype(np.float32)
    lon = np.deg2rad(chip_centroids_lonlat[:, 0]).astype(np.float32)
    latlon = np.stack([np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)], axis=1)
    latlon_t = torch.from_numpy(latlon).to(device)

    waves_t = torch.tensor(S2_WAVELENGTHS_NM, dtype=torch.float32, device=device)
    gsd_t   = torch.tensor(S2_GSD,            dtype=torch.float32, device=device)

    return {
        'pixels': pix_t,
        'time':   time_t,
        'latlon': latlon_t,
        'waves':  waves_t,
        'gsd':    gsd_t,
    }
'''))

# =========================================================================
# 13. Forward pass test
# =========================================================================
cells.append(md("""
## A first forward pass — sanity check

Before we loop over 9 years × ~60 chips, verify that *one* overpass goes in cleanly and an embedding of the right shape comes out.
"""))

cells.append(code('''
# Run a forward pass on the first overpass's chips
test_dt = pd.Timestamp(selection.iloc[0]['date'])
test_batch = build_batch(test_chips, test_cents, test_dt)

with torch.no_grad():
    unmsk_patch, *_ = module.model.encoder(test_batch)

print(f'Encoder output shape: {tuple(unmsk_patch.shape)}')
print('Expected: (n_chips, 1 + 32*32, 1024) = (n_chips, 1025, 1024)')
print(f'  -> {unmsk_patch.shape[0]} chips, '
      f'{unmsk_patch.shape[1] - 1} patches per chip, '
      f'embedding dim {unmsk_patch.shape[2]}')

# CLS token = scene-level embedding
scene_emb = unmsk_patch[:, 0, :].cpu().numpy()  # (n_chips, 1024)
print(f'\\nScene-level embeddings: shape {scene_emb.shape}, '
      f'cosine self-similarity diag mean = {np.mean(np.diag((scene_emb @ scene_emb.T) / (np.linalg.norm(scene_emb, axis=1, keepdims=True) * np.linalg.norm(scene_emb, axis=1, keepdims=True).T))):.3f} (should be ~1.0)')
'''))

# =========================================================================
# 14. Full pipeline: embed all 9 years
# =========================================================================
cells.append(md("""
## Full pipeline — embed all 9 yearly overpasses

For each year:

1. Mosaic the overpass to a single `(10, y, x)` DataArray.
2. Tile into 256 × 256 chips with stride 128.
3. Forward pass → take the CLS token → stash as `scene_embeddings[year][chip_idx]`.

Same chip *grid* across all years (the AOI bounds and stride don't change), so chip 17 in 2017 corresponds to chip 17 in 2018, and we can directly do year-over-year cosine distance per chip.

We persist the result to `data/processed/embeddings/banya_chip_embeddings.npz` so the analysis below is decoupled from the (~minutes-on-CPU, faster-on-GPU) inference run.
"""))

cells.append(code('''
EMBED_PATH = EMBEDDINGS_DIR / 'banya_chip_embeddings.npz'

# Land-fraction estimate per chip (using MNDWI threshold as a quick proxy)
def estimate_land_fraction(arr_chip):
    """MNDWI-based land fraction estimate for a single chip array (10, 256, 256)."""
    green = arr_chip[CLAY_S2_BAND_ORDER.index('green')]
    swir  = arr_chip[CLAY_S2_BAND_ORDER.index('swir16')]
    mndwi = (green - swir) / (green + swir + 1e-9)
    return float(np.mean(mndwi < 0))   # land has MNDWI < 0

if EMBED_PATH.exists():
    print(f'Found cached embeddings at {EMBED_PATH.relative_to(PROJECT_ROOT)}; loading.')
    blob = np.load(EMBED_PATH, allow_pickle=True)
    years         = blob['years']
    dates         = blob['dates']
    chip_bboxes   = blob['chip_bboxes']
    chip_cents    = blob['chip_centroids']
    chip_land_frac = blob['chip_land_fraction']
    scene_embs    = blob['scene_embeddings']
else:
    scene_embs_per_year = []
    chip_bboxes = chip_cents = chip_land_frac = None
    dates_list, years_list = [], []
    for _, r in selection.iterrows():
        date = pd.Timestamp(r['date'])
        print(f'  {date.date()}: mosaicking...', end=' ', flush=True)
        arr = mosaic_overpass(r['stac_items'], DEMO_AOI)
        chips, bboxes, cents = tile_into_chips(arr)
        print(f'tiled into {len(chips)} chips,', end=' ', flush=True)

        # Cache the chip grid the first time (it's identical across years by construction)
        if chip_bboxes is None:
            chip_bboxes = bboxes
            chip_cents  = cents
            chip_land_frac = np.array([estimate_land_fraction(c) for c in chips])

        batch = build_batch(chips, cents, date)
        with torch.no_grad():
            out, *_ = module.model.encoder(batch)
        scene_embs_per_year.append(out[:, 0, :].cpu().numpy())   # CLS token
        dates_list.append(str(date.date()))
        years_list.append(int(r['year']))
        print(f'embedded.')

    scene_embs = np.stack(scene_embs_per_year)   # (n_years, n_chips, 1024)
    years = np.array(years_list)
    dates = np.array(dates_list)
    np.savez_compressed(
        EMBED_PATH,
        years=years, dates=dates,
        chip_bboxes=chip_bboxes,
        chip_centroids=chip_cents,
        chip_land_fraction=chip_land_frac,
        scene_embeddings=scene_embs,
    )
    print(f'\\nSaved {EMBED_PATH.relative_to(PROJECT_ROOT)} '
          f'({EMBED_PATH.stat().st_size / 1e6:.1f} MB)')

print(f'\\nEmbedding tensor: years={list(years)}, '
      f'chips={scene_embs.shape[1]}, dim={scene_embs.shape[2]}')
'''))

# =========================================================================
# 15. Cosine-distance helpers + AOI-mean curve
# =========================================================================
cells.append(md("""
## Year-over-year cosine distance — the headline signal

For each consecutive pair of years (2017→2018, 2018→2019, ...) and each chip, compute cosine distance between the chip's embedding in year-t and year-(t+1). Then aggregate two ways:

1. **AOI-mean** (one scalar per year-pair) — the headline curve. Place next to notebook 3's transect-mean shoreline drift.
2. **Per-chip** (a spatial map per year-pair) — shows *where* in the AOI the embedding moved most.

We also compute distance-to-2017 (a slower drift signal anchored to the start of the trend window), which costs nothing extra given the same embeddings.
"""))

cells.append(code('''
def cosine_distance_matrix(A, B):
    """Cosine distance row-wise between A[i] and B[i]. A, B: (n, d)."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.einsum('ij,ij->i', An, Bn)

n_years, n_chips, dim = scene_embs.shape

# Year-over-year (n_years - 1 transitions × n_chips chips)
cos_yoy = np.stack([
    cosine_distance_matrix(scene_embs[t], scene_embs[t + 1])
    for t in range(n_years - 1)
])  # shape (n_years - 1, n_chips)

# Distance from 2017 baseline
cos_to_2017 = np.stack([
    cosine_distance_matrix(scene_embs[0], scene_embs[t])
    for t in range(1, n_years)
])  # shape (n_years - 1, n_chips)

# AOI-mean (weighted by chip land fraction so predominantly-water chips count less)
weights = np.clip(chip_land_frac, 0.05, 1.0)  # floor at 5% to not zero out near-shore chips
weights = weights / weights.sum()

aoi_mean_yoy     = (cos_yoy     * weights[None, :]).sum(axis=1)
aoi_mean_to_2017 = (cos_to_2017 * weights[None, :]).sum(axis=1)
transition_years = years[1:]

print('Year-over-year cosine distance (AOI mean, land-weighted):')
for ty, d_yoy, d_to17 in zip(transition_years, aoi_mean_yoy, aoi_mean_to_2017):
    print(f'  {ty - 1} -> {ty}: yoy={d_yoy:.4f},  cumulative-from-2017={d_to17:.4f}')
'''))

# =========================================================================
# 16. Plot AOI-mean curve next to nb03 transect-mean drift
# =========================================================================
cells.append(md("""
## Plot the AOI-mean curve next to notebook 3's transect-mean drift

Reload notebook 3's saved shorelines, compute transect-mean position drift per year, and put the two curves on a shared time axis. Two y-axes — one in metres (nb03), one unitless (nb04) — because they are *not* commensurable; we just want to look at the timing of the bumps.
"""))

cells.append(code('''
# Load nb03 outputs
shorelines = (PROCESSED / 'shorelines' / 'banya_yearly_shorelines.geojson')
import geopandas as gpd
banya_lines = gpd.read_file(shorelines).sort_values('year').reset_index(drop=True)

# Simple proxy for nb03's transect-mean drift: total shoreline length change year-over-year
# (the proper transect-mean lives inside nb03; here we use a coarser surrogate so this
# cell doesn't depend on nb03 helper functions. We'll wire in the proper version when
# we factor nb03 into src/.)
lengths_km = banya_lines.geometry.length.values / 1000.0
shoreline_yoy_dlen = np.diff(lengths_km)

fig, ax1 = plt.subplots(figsize=(10, 4.5))
ax1.bar(transition_years - 0.18, shoreline_yoy_dlen, width=0.36,
        color='steelblue', label='nb03: Δ shoreline length (km, proxy)')
ax1.axhline(0, color='gray', lw=0.5)
ax1.set_xlabel('year (end of transition)')
ax1.set_ylabel('Δ shoreline length, km   (nb03 surrogate)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.bar(transition_years + 0.18, aoi_mean_yoy, width=0.36,
        color='crimson', alpha=0.85, label='nb04: cosine distance YoY')
ax2.set_ylabel('embedding cosine distance (unitless)', color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')

# Annotate Storm Gloria
ax1.axvline(2020, color='black', ls=':', alpha=0.6)
ax1.text(2020.05, ax1.get_ylim()[1] * 0.9, 'Gloria\\n(Jan 2020)', fontsize=9, va='top')

ax1.set_title('Banya: shoreline-length change (nb03) vs Clay embedding cosine distance (nb04)')
fig.tight_layout()
plt.show()
'''))

# =========================================================================
# 17. PCA visualisation (Element84 trick)
# =========================================================================
cells.append(md("""
## A PCA visualisation — the Element84 trick

We borrow this from Hassan / Element84 (2023). Take the *AOI-mean* embedding (land-weighted) for each year, stack into a `(n_years, 1024)` matrix, run PCA, and plot the first principal component vs. time. This gives a 1-D visualisation of the trajectory of Banya through Clay's representation space.

In 1024-D the trajectory is incomprehensible. PC1 captures the *single direction along which the trajectory varies most* — usually a coherent "drift" axis. If the PC1 curve has a visible bump at 2020, that's Clay saying "2020 stuck out from the trend" — without us telling it what to look for.
"""))

cells.append(code('''
# AOI-mean (land-weighted) per year, in embedding space
aoi_mean_emb = np.einsum('ycd,c->yd', scene_embs, weights)  # (n_years, 1024)

pca = PCA(n_components=2)
proj = pca.fit_transform(aoi_mean_emb)
print(f'PCA: PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance, '
      f'PC2 {pca.explained_variance_ratio_[1]*100:.1f}%')

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(years, proj[:, 0], 'o-', color='crimson', lw=2)
for y, p in zip(years, proj[:, 0]):
    ax.annotate(str(y), xy=(y, p), xytext=(0, 6), textcoords='offset points',
                ha='center', fontsize=8)
ax.axvline(2020, color='gray', ls=':', alpha=0.6)
ax.set_xlabel('year')
ax.set_ylabel(f'PC1 of land-weighted AOI-mean embedding\\n({pca.explained_variance_ratio_[0]*100:.0f}% of variance)')
ax.set_title("Banya's 9-year trajectory through Clay's embedding space (1-D projection)")
fig.tight_layout()
plt.show()
'''))

# =========================================================================
# 18. Per-chip spatial map of YoY cosine distance
# =========================================================================
cells.append(md("""
## Per-chip spatial map — where in the AOI did the embedding move most?

Render each chip as a 2.56 km square coloured by its year-over-year cosine distance. A grid of small subplots, one per year-pair. We expect:

- The 2019→2020 panel (Gloria-period) to be the brightest.
- The brightest chips to spatially correlate with the *erosive northern half* of the spit reported by Pintó et al. 2021.
- Predominantly-water offshore chips to be either quiet (if the embedding ignores them, good) or noisy (if the model is unstable outside its training distribution — flag for follow-up).
"""))

cells.append(code('''
from matplotlib.patches import Rectangle

n_pairs = cos_yoy.shape[0]
ncols = min(n_pairs, 4)
nrows = int(np.ceil(n_pairs / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

vmin, vmax = float(cos_yoy.min()), float(cos_yoy.max())
xmin, xmax = chip_bboxes[:, 0].min(), chip_bboxes[:, 2].max()
ymin, ymax = chip_bboxes[:, 1].min(), chip_bboxes[:, 3].max()
cmap = plt.cm.viridis

for k in range(n_pairs):
    ax = axes.flat[k]
    for ci, (xmin_c, ymin_c, xmax_c, ymax_c) in enumerate(chip_bboxes):
        color = cmap((cos_yoy[k, ci] - vmin) / (vmax - vmin + 1e-12))
        ax.add_patch(Rectangle((xmin_c, ymin_c), xmax_c - xmin_c, ymax_c - ymin_c,
                               facecolor=color, edgecolor='white', linewidth=0.3,
                               alpha=0.5 + 0.5 * chip_land_frac[ci]))
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect('equal')
    ax.set_title(f'{transition_years[k] - 1} -> {transition_years[k]}', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

for k in range(n_pairs, nrows * ncols):
    axes.flat[k].axis('off')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
fig.colorbar(sm, ax=axes.ravel().tolist(), label='cosine distance (year-over-year)',
             shrink=0.7)
fig.suptitle('Banya: per-chip embedding change. Faded = chip is predominantly water.',
             fontsize=12)
plt.show()
'''))

# =========================================================================
# 19. Geographical-prior test: Maresme
# =========================================================================
cells.append(md("""
## Does Clay need a delta prior? — Maresme as the control

Recall the question. In notebook 3 we needed two priors (morphological opening + seaward-edges) to stop the classical algorithm confusing the Encanyissada lagoon for the sea. For Clay, the equivalent question is: when Banya's embedding distance jumps year over year, is it jumping because the *shoreline moved*, or is it jumping because inland water bodies changed state — saltworks, paddies, lagoon turbidity?

The clean test is to run the exact same pipeline on a **clean coastal tile** from the Maresme AOI — sea + narrow beach + dunes + town/rail, no lagoon, no saltworks. If Maresme's curve tracks shoreline drift cleanly and Banya's doesn't, we have evidence that Clay's signal at Banya is dominated by *non-coastal* change — and we'd want a geographical prior analogous to the morphological opening (maybe: weight per-chip embeddings by distance to the coast, or restrict analysis to coastal-strip chips).

Three possible outcomes:

1. **Both curves track shoreline change.** Clay's pre-training has implicitly learned "the sea-land boundary is the salient changing thing on a coastline". No prior needed; the foundation model bakes in the geographical knowledge.
2. **Maresme tracks; Banya is noisy and decorrelated.** Clay needs a delta-specific prior. The analogue to notebook 3's morphological opening is masking or down-weighting non-coastal pixels before embedding.
3. **Both noisy.** Signal dominated by atmospherics or non-coastal land cover. We'd pivot to patch-level analysis on coast-adjacent patches only.
"""))

cells.append(code('''
# Define a clean coastal tile in Maresme (between Mataró and Arenys de Mar)
MARESME_TILE = sg.box(2.55, 41.50, 2.65, 41.58)   # ~8 x 9 km, sea + narrow beach + town
MARESME_SEAWARD_EDGES = SEAWARD_EDGES['maresme']  # ('east',)

# Same yearly cleanest-summer selection logic
trend_m = list(catalog.search(
    collections=['sentinel-2-l2a'],
    intersects=MARESME_TILE,
    datetime='2017-07-01/2025-09-15',
    query={'eo:cloud_cover': {'lt': 10}},
).item_collection())
by_date_m = defaultdict(list)
for i in trend_m:
    by_date_m[i.datetime.date()].append(i)
df_m = pd.DataFrame([
    {'date': d, 'year': d.year, 'month': d.month,
     'n_granules': len(its), 'mean_cloud': float(np.mean([i.properties['eo:cloud_cover'] for i in its])),
     'stac_items': its}
    for d, its in by_date_m.items()
])
sm = df_m[df_m.month.isin([6, 7, 8, 9])].copy()
sm['coverage_rank'] = (sm.n_granules >= 1).astype(int)
selection_m = (sm.sort_values(['coverage_rank', 'mean_cloud'], ascending=[False, True])
                 .groupby('year', as_index=False).head(1)
                 .sort_values('year').reset_index(drop=True))
print(f'Maresme: {len(selection_m)} yearly overpasses')
'''))

cells.append(code('''
EMBED_PATH_M = EMBEDDINGS_DIR / 'maresme_chip_embeddings.npz'

if EMBED_PATH_M.exists():
    blob_m = np.load(EMBED_PATH_M, allow_pickle=True)
    years_m = blob_m['years']
    scene_embs_m = blob_m['scene_embeddings']
    chip_land_frac_m = blob_m['chip_land_fraction']
else:
    scene_embs_per_year_m = []
    years_list_m = []
    chip_land_frac_m = None
    for _, r in selection_m.iterrows():
        date = pd.Timestamp(r['date'])
        arr = mosaic_overpass(r['stac_items'], MARESME_TILE)
        chips_m, bboxes_m, cents_m = tile_into_chips(arr)
        if chip_land_frac_m is None:
            chip_land_frac_m = np.array([estimate_land_fraction(c) for c in chips_m])
        batch = build_batch(chips_m, cents_m, date)
        with torch.no_grad():
            out, *_ = module.model.encoder(batch)
        scene_embs_per_year_m.append(out[:, 0, :].cpu().numpy())
        years_list_m.append(int(r['year']))
        print(f'  {date.date()}: {len(chips_m)} chips, embedded.')
    scene_embs_m = np.stack(scene_embs_per_year_m)
    years_m = np.array(years_list_m)
    np.savez_compressed(EMBED_PATH_M,
                        years=years_m,
                        scene_embeddings=scene_embs_m,
                        chip_land_fraction=chip_land_frac_m)
    print(f'Saved {EMBED_PATH_M.relative_to(PROJECT_ROOT)}')
'''))

cells.append(code('''
# Compute Maresme AOI-mean YoY cosine distance and put alongside Banya
weights_m = np.clip(chip_land_frac_m, 0.05, 1.0); weights_m /= weights_m.sum()
cos_yoy_m = np.stack([
    cosine_distance_matrix(scene_embs_m[t], scene_embs_m[t + 1])
    for t in range(scene_embs_m.shape[0] - 1)
])
aoi_mean_yoy_m = (cos_yoy_m * weights_m[None, :]).sum(axis=1)
trans_years_m = years_m[1:]

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(transition_years, aoi_mean_yoy, 'o-', color='crimson', label='Banya (delta, mixed)')
ax.plot(trans_years_m, aoi_mean_yoy_m, 's-', color='steelblue', label='Maresme (clean coastal)')
ax.axvline(2020, color='gray', ls=':', alpha=0.6, label='Storm Gloria (Jan 2020)')
ax.set_xlabel('year (end of transition)')
ax.set_ylabel('AOI-mean cosine distance (year-over-year)')
ax.set_title("Geographical-prior test: does Clay's signal differ between Banya and a clean-coast control?")
ax.legend()
fig.tight_layout()
plt.show()

# Correlation with nb03's surrogate shoreline-length drift at Banya
banya_corr = spearmanr(aoi_mean_yoy, shoreline_yoy_dlen)
print(f'\\nSpearman correlation, Banya YoY embedding distance vs nb03 shoreline-length proxy: '
      f'rho={banya_corr.statistic:.3f}, p={banya_corr.pvalue:.3f}')
print('(Maresme correlation will require running nb03 on the Maresme tile; ',
      'placeholder until we do.)')
'''))

# =========================================================================
# 20. Discussion + limitations
# =========================================================================
cells.append(md("""
## What we have, and what we don't

What we built:

- A *labels-free, scene-level* change signal at Banya, 2017–2025, computed from Clay v1.5 embeddings of the same nine yearly Sentinel-2 overpasses notebook 3 used.
- A direct side-by-side with notebook 3's shoreline-length drift: same x-axis, two curves, no claim of commensurability.
- A 1-D PCA visualisation of Banya's trajectory through Clay's representation space.
- A spatial map showing *where* in the AOI the embedding moved most, year by year.
- A Maresme control to test the headline question of the notebook: does Clay need a delta-specific geographical prior?

What we did *not* build:

- A shoreline. Clay's embedding is not a segmentation. If you want metres of retreat per year along the spit, that's still notebook 3's job.
- A trained-for-coastal-erosion model. Clay v1.5 is used here purely as a pre-trained feature extractor. Notebook 5 (conditional on this notebook's result) would *finetune* a model — likely Prithvi-EO-2.0 — on labelled shoreline data.
- A tide correction. Our nine snapshots are tide-uncorrected instantaneous waterlines. The embedding signal includes whatever tidal state was present at each overpass. Mediterranean tides are small (~20–30 cm range), so this is mostly noise; Storm Gloria's ~1 m surge in January 2020 is a real complication for notebook 7 but doesn't affect *summer* scenes much.

Limitations worth being honest about:

- **Open-ocean training gap.** Clay's model card explicitly says it was not trained on open ocean. Our predominantly-water chips are out of the training distribution. We weighted them down in the AOI mean; we did not exclude them. The spatial-map analysis lets a reader see for themselves whether they're behaving sensibly.
- **9 datapoints is a short time series.** Statistical statements ("the 2019→2020 transition is anomalous") are eyeballed, not tested against a null model. With one summer scene per year, year-over-year noise from individual scenes is non-trivial.
- **Chip granularity is coarse.** Each chip is 2.56 km on the ground. Notebook 3's transects are spaced every ~200 m. Chip-level spatial analysis is therefore an order of magnitude coarser than the classical method — by design (Clay's natural unit) but at a cost.
- **Cosine distance is just one of many things to do with embeddings.** Clustering (to discover "coastal types" automatically), nearest-neighbour search (to find analogue scenes elsewhere along the coast), and reconstruction-error from the MAE decoder are all natural follow-ups.

## What's next

- **Notebook 6 — Validation.** Compare both notebook 3's shorelines and notebook 4's "where did the embedding change?" map against ICGC LiDAR-derived shorelines for the three campaign years.
- **Notebook 7 — Storm Gloria zoom.** Pre/post Gloria, patch-level embeddings inside the chips that moved most in the 2019→2020 transition. TerraMind for the SAR fallback when optical was cloud-covered.
- **Notebook 5 — Prithvi finetune** — conditional. Worth doing if (a) this notebook's signal is informative enough to want it sharpened with a small labelled dataset, *and* (b) we can produce ~50 hand-labelled coastal-strip masks at acceptable effort.
"""))

# =========================================================================
# Assemble & write
# =========================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH} with {len(cells)} cells "
      f"({sum(c['cell_type']=='markdown' for c in cells)} markdown, "
      f"{sum(c['cell_type']=='code' for c in cells)} code)")
