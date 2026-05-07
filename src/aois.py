"""AOI definitions for the GFM_coasts project.

Single source of truth for the polygons used in every notebook. Notebooks
that need them do:

    import sys; sys.path.insert(0, str(Path('../src').resolve()))
    from aois import AOIS, DESCRIPTIONS

Coordinates are WGS84 (EPSG:4326). Bounding boxes are deliberately generous
on the inland side; analysis later will clip to a coastal strip.
"""

from shapely.geometry import box

# (lon_min, lat_min, lon_max, lat_max)
EBRO_DELTA = box(0.50, 40.55, 0.95, 40.85)
MARESME = box(2.25, 41.40, 2.90, 41.75)

AOIS = {
    "ebro_delta": EBRO_DELTA,
    "maresme": MARESME,
}

DESCRIPTIONS = {
    "ebro_delta": (
        "Ebro Delta — sediment-starved delta with active retreat in the "
        "southern hemidelta (Punta de la Banya) and accretion in the "
        "northern (Punta del Fangar)."
    ),
    "maresme": (
        "Maresme coast — Mongat to Blanes, north of Barcelona; chronic-deficit "
        "pocket beaches with the railway line at the back of the beach."
    ),
}

# Reference points where we'll later sample wave climate.
# Lat/lon of buoys offshore of each AOI.
WAVE_BUOYS = {
    "tarragona": {"lon": 1.47, "lat": 41.07, "near": "ebro_delta"},
    "begur":     {"lon": 3.66, "lat": 41.92, "near": "maresme"},
}

# Which AOI bounding-box edges face the open sea — used by the
# edge-touching connected-component sea isolator (see notebook 03)
# to disambiguate the sea from inland water bodies (lagoons, salt pans,
# rice paddies). On the Ebro Delta this is critical — La Encanyissada
# lagoon is bigger than the visible sea inside many sub-AOIs.
SEAWARD_EDGES = {
    "ebro_delta": ("south", "east"),
    "maresme":    ("east",),
}
