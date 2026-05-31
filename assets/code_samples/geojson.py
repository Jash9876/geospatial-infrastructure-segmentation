import numpy as np
import rasterio
from rasterio.features import shapes
import json

# ======================
# PATHS
# ======================
MASK_PATH = r"C:\stitched\Village1.npy"
IMG_PATH = r"C:\raw\Village1.tif"
OUT_GEOJSON = r"C:\output.geojson"

# ======================
# LOAD DATA
# ======================
mask = np.load(MASK_PATH)

with rasterio.open(IMG_PATH) as src:
    transform = src.transform

# ======================
# CLASS LABELS
# ======================
class_map = {
    1: "road",
    2: "roof",
    3: "water",
    4: "utility"
}

features = []

# ======================
# CONVERT EACH CLASS
# ======================
for class_id, class_name in class_map.items():

    binary_mask = (mask == class_id).astype(np.uint8)

    for geom, value in shapes(binary_mask, transform=transform):

        if value == 1:
            feature = {
                "type": "Feature",
                "properties": {
                    "class": class_name,
                    "class_id": int(class_id)
                },
                "geometry": geom
            }
            features.append(feature)

# ======================
# SAVE GEOJSON
# ======================
geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open(OUT_GEOJSON, "w") as f:
    json.dump(geojson, f)

print("✅ GeoJSON saved:", OUT_GEOJSON)