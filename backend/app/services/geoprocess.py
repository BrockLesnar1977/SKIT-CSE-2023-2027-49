import json
import logging
import os
from typing import Dict, Optional
from uuid import UUID
from pathlib import Path

import ee

logger = logging.getLogger(__name__)

LOCAL_OUTPUT_DIR = Path(__file__).parent.parent.parent / "geojson_outputs"
LOCAL_OUTPUT_DIR.mkdir(exist_ok=True)

from google.cloud import storage as gcs_storage
from ..core.config import settings

def process_and_store_results(
    loss_mask_image: ee.Image,
    aoi_geometry: ee.Geometry,
    request_id: UUID,
) -> Dict[str, Optional[str]]:
    logger.info(f"[{request_id}] Geoprocess: starting")


    map_id = loss_mask_image.getMapId({"palette": ["#FF4136"], "min": 0, "max": 1})
    tile_url = map_id["tile_fetcher"].url_format
    logger.info(f"[{request_id}] Geoprocess: tile URL generated")

    min_px = 15
    px_count = loss_mask_image.connectedPixelCount(min_px, False)
    cleaned = loss_mask_image.updateMask(px_count.gte(min_px))

    vectors = cleaned.reduceToVectors(
        geometry=aoi_geometry,
        scale=60,          
        geometryType="polygon",
        eightConnected=False,
        maxPixels=1e9,     
    )
    simplified = vectors.map(lambda f: f.simplify(maxError=100))
    logger.info(f"[{request_id}] Geoprocess: vectorization complete")

    geojson_data = simplified.getInfo()
    feature_count = len(geojson_data.get("features", []))
    logger.info(f"[{request_id}] Geoprocess: {feature_count} features fetched")
    
    if feature_count == 0:
        logger.info(f"[{request_id}] No change features found, skipping upload")
        return {"geojson_url": None, "tile_url": tile_url}

    geojson_str = json.dumps(geojson_data, separators=(",", ":"))
    geojson_url = None

    return {"geojson_url": geojson_url, "tile_url": tile_url}
