import logging
import datetime
from typing import Dict, Any
import ee

logger = logging.getLogger(__name__)

ee.Initialize()
logger.info("Google Earth Engine initialized successfully.")


def _today_ee() -> ee.Date:
    return ee.Date(datetime.datetime.utcnow().strftime("%Y-%m-%d"))


def _cap_end_date(end_date: str) -> ee.Date:
    today = _today_ee()
    end = ee.Date(end_date)
    return ee.Date(ee.Algorithms.If(end.difference(today, "day").gt(0), today, end))


def _polygon_geometry(aoi: Dict[str, Any]) -> ee.Geometry:
    geo_type = aoi.get("type")
    if geo_type == "Feature":
        aoi = aoi["geometry"]
        geo_type = aoi.get("type")
    if geo_type == "FeatureCollection":
        aoi = aoi["features"][0]["geometry"]
        geo_type = aoi.get("type")
    if geo_type == "Polygon":
        return ee.Geometry.Polygon(aoi["coordinates"])
    if geo_type == "MultiPolygon":
        return ee.Geometry.MultiPolygon(aoi["coordinates"])
    raise ValueError(f"Unsupported geometry type: {geo_type}")


def _require_start_after(start_date: str, cutoff: str, dataset_name: str):
    start = ee.Date(start_date)
    cutoff_ee = ee.Date(cutoff)
    if start.difference(cutoff_ee, "day").lt(0).getInfo():
        raise ValueError(
            f"{dataset_name} data is only available from {cutoff}. "
            f"Please choose a start date on or after {cutoff}."
        )





def _sentinel2_ndvi_composite(region: ee.Geometry, start: ee.Date, end: ee.Date) -> ee.Image:
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .map(_mask_s2_clouds)
    )
    if col.size().getInfo() == 0:
        raise ValueError(f"No cloud-free Sentinel-2 imagery found for {start.format('YYYY-MM-dd').getInfo()} – {end.format('YYYY-MM-dd').getInfo()}.")
    composite = col.median()
    required = ee.List(["B4", "B8"])
    if not composite.bandNames().containsAll(required).getInfo():
        raise ValueError("Composite is missing required bands B4/B8. Try a wider date range.")
    return composite.normalizedDifference(["B8", "B4"]).rename("NDVI")





def _calc_area_km2(mask: ee.Image, region: ee.Geometry, scale: int = 100) -> float:
    result = (
        ee.Image.pixelArea()
        .updateMask(mask)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,   
        )
        .get("area")
        .getInfo()
    ) or 0
    return result / 1_000_000

def compute_vegetation_change(aoi: Dict, start_date: str, end_date: str) -> Dict:
    logger.info(f"GEE: vegetation_change {start_date} → {end_date}")
    _require_start_after(start_date, "2015-06-23", "Sentinel-2")
    end_ee = _cap_end_date(end_date)
    region = _polygon_geometry(aoi)
    start_ee = ee.Date(start_date)

    start_ndvi = _sentinel2_ndvi_composite(region, start_ee, start_ee.advance(1, "year"))
    end_ndvi = _sentinel2_ndvi_composite(region, end_ee.advance(-1, "year"), end_ee)

    threshold = 0.2
    start_veg = start_ndvi.gte(threshold).unmask(0)
    end_veg = end_ndvi.gte(threshold).unmask(0)
    loss_mask = start_veg.And(end_veg.Not())
    gain_mask = end_veg.And(start_veg.Not())

    loss_km2 = _calc_area_km2(loss_mask, region)
    gain_km2 = _calc_area_km2(gain_mask, region)
    initial_km2 = _calc_area_km2(start_veg, region)
    loss_pct = (loss_km2 / initial_km2 * 100) if initial_km2 > 0 else 0

    logger.info(f"GEE: vegetation loss={loss_km2:.2f} km², gain={gain_km2:.2f} km²")
    return {
        "metrics": {
            "vegetation_loss_km2": round(loss_km2, 4),
            "vegetation_gain_km2": round(gain_km2, 4),
            "initial_vegetation_km2": round(initial_km2, 4),
            "net_change_km2": round(gain_km2 - loss_km2, 4),
            "loss_pct": round(loss_pct, 4),
        },
        "ee_image": loss_mask,
        "ee_geometry": region,
    }


def compute_builtup_change(aoi: Dict, start_date: str, end_date: str) -> Dict:
    logger.info(f"GEE: builtup_change {start_date} → {end_date}")
    _require_start_after(start_date, "2015-06-27", "Dynamic World")
    end_ee = _cap_end_date(end_date)
    region = _polygon_geometry(aoi)
    start_ee = ee.Date(start_date)
    BUILT_UP = 6

    dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region)

    start_img = dw.filterDate(start_ee, start_ee.advance(1, "year")).mode()
    end_img = dw.filterDate(end_ee.advance(-1, "year"), end_ee).mode()

    start_mask = start_img.select("label").eq(BUILT_UP)
    end_mask = end_img.select("label").eq(BUILT_UP)
    gain_mask = end_mask.And(start_mask.Not())
    loss_mask = start_mask.And(end_mask.Not())

    gain_km2 = _calc_area_km2(gain_mask, region)
    loss_km2 = _calc_area_km2(loss_mask, region)
    initial_km2 = _calc_area_km2(start_mask, region)
    gain_pct = (gain_km2 / initial_km2 * 100) if initial_km2 > 0 else 0

    logger.info(f"GEE: builtup gain={gain_km2:.2f} km²")
    return {
        "metrics": {
            "builtup_gain_km2": round(gain_km2, 4),
            "builtup_loss_km2": round(loss_km2, 4),
            "initial_builtup_km2": round(initial_km2, 4),
            "final_builtup_km2": round(initial_km2 + gain_km2 - loss_km2, 4),
            "gain_pct": round(gain_pct, 4),
        },
        "ee_image": gain_mask,
        "ee_geometry": region,
    }


