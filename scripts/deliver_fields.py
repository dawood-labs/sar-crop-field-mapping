"""Client deliverable: rice + cane-cassava field polygons in one clean layer (user, 2026-10-09).

Inputs : data/model/rice_v5/fields/fields_rice_v5_clip_cane_v3.parquet  (Rice, cane taken out)
         data/model/cane_v3/fields/fields_cane_v3_clip_rice.parquet       (Cane-Cassava, 0.1 m from rice)
Steps  : predicted = 7 rice, 1 cane-cassava -> topology.refine (closing 1.5 m: slits / notches; holes < 0.05 ac filled;
         opening 2 m: tails / spikes; neighbours' original area protected) -> simplify 0.5 m -> separate 0.1 m (no
         overlap, no touching, so no dissolve can ever merge two fields) -> multi-part features split into one feature
         per polygon -> QC (fails loudly) -> write.
Outputs: data/deliverables/<NAME>.parquet   GeoParquet for internal work (all attributes, bbox covering column)
         data/deliverables/<NAME>.zip       zipped shapefile for the client: field_id, predicted, class, area_ac
         data/deliverables/<NAME>_qc.json   every QC number (the shapefile is read back and checked too)
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")

import json, sys, time, zipfile
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import safe_resources  # noqa: E402,F401
import topology as tp  # noqa: E402

NAME = "rice_cane_fields"
OUT = PROJ / "data/deliverables"
RICE = PROJ / "data/model/rice_v5/fields/fields_rice_v5_clip_cane_v3.parquet"
CANE = PROJ / "data/model/cane_v3/fields/fields_cane_v3_clip_rice.parquet"
ACRE_M2 = 4046.856
PREDICTED = {"Rice": 7, "Cane-Cassava": 1}


def qc(g, aoi):
    """All checks on a GeoDataFrame; returns (dict, list of failures)."""
    import shapely
    G = np.asarray(g.geometry.values, dtype=object)
    P = shapely.get_parts(G)
    ni, mg = tp.check(G)
    holes = [shapely.Polygon(r).area for p in P for r in p.interiors]
    small_holes = int(sum(h < 0.05 * ACRE_M2 for h in holes))
    thin = int(shapely.is_empty(tp._threaded(lambda x: shapely.buffer(x, -1.5, quad_segs=2), G)).sum())
    wkb = shapely.to_wkb(shapely.normalize(G))
    r = {"features": int(len(g)), "invalid": int((~shapely.is_valid(G)).sum()), "empty": int(shapely.is_empty(G).sum()),
         "multipart": int((shapely.get_num_geometries(G) > 1).sum()), "non_polygon": int((shapely.get_type_id(G) != 3).sum()),
         "intersecting_or_touching_pairs": ni, "min_gap_m": mg, "duplicates": int(len(wkb) - len(set(wkb))),
         "holes": len(holes), "holes_lt_0.05ac": small_holes, "spikes_lt_15deg": tp.spikes(G), "thinner_than_3m": thin,
         "outside_aoi": int((~shapely.within(G, aoi.buffer(1.0))).sum()), "min_area_m2": float(shapely.area(G).min()),
         "crs": g.crs.to_string()}
    fail = [k for k in ("invalid", "empty", "multipart", "non_polygon", "intersecting_or_touching_pairs", "duplicates",
                        "holes_lt_0.05ac", "outside_aoi") if r[k] > 0]
    return r, fail


def main():
    import geopandas as gpd, pandas as pd, shapely
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["uid", "source", "class", "area_acres", "Shp_Label", "Confidence"]
    r = gpd.read_parquet(RICE); c = gpd.read_parquet(CANE)
    r = r[r["class"] == "Rice"]; c = c[c["class"] == "Cane-Cassava"]
    g = pd.concat([r[cols + ["rice_frac", "geometry"]].rename(columns={"rice_frac": "class_frac"}),
                   c[cols + ["cane_frac", "geometry"]].rename(columns={"cane_frac": "class_frac"})], ignore_index=True)
    g = gpd.GeoDataFrame(g, crs=r.crs)
    before = {k: float(v) for k, v in (g.area / ACRE_M2).groupby(g["class"]).sum().items()}
    G0 = shapely.make_valid(shapely.set_precision(np.asarray(g.geometry.values, dtype=object), tp.GRID))   # snap to 1 cm
    s0 = {"holes": int(shapely.get_num_interior_rings(shapely.get_parts(G0)).sum()), "spikes_lt_15deg": tp.spikes(G0)}
    print(f"input: {len(g):,} polygons {before} {s0} ({time.time() - t0:.0f} s)", flush=True)

    G = tp.refine(G0)
    G = tp.simplify(G)
    G = tp.separate(G)
    g["geometry"] = G
    g = g[~g.is_empty].explode(index_parts=False).reset_index(drop=True)
    g = g[g.geom_type == "Polygon"]
    g = g[g.area >= 100.0].reset_index(drop=True)             # parts left < 1 px by the cut / opening
    # closing can push an outline a little past the AOI: clip, then one more separate pass on the final shapes
    aoi0 = gpd.read_file(PROJ / "data/aoi/aoi.shp").to_crs(g.crs).union_all()
    out_aoi = ~g.within(aoi0)
    g.loc[out_aoi, "geometry"] = tp.safe_difference(np.asarray(g.geometry.values[out_aoi], dtype=object),
                                                    np.array([aoi0.envelope.buffer(1000).difference(aoi0)] * int(out_aoi.sum()), dtype=object))
    for _ in range(3):                                         # repeat until no pair touches
        g["geometry"] = tp.separate(np.asarray(g.geometry.values, dtype=object), gap=tp.GAP)
        g = g[~g.is_empty].explode(index_parts=False).reset_index(drop=True)
        g = g[(g.geom_type == "Polygon") & (g.area >= 100.0)].reset_index(drop=True)
        if tp.check(np.asarray(g.geometry.values, dtype=object))[0] == 0:
            break
    print(f"refined: {len(g):,} polygons; GEOS fallbacks {tp.FALLBACKS} ({time.time() - t0:.0f} s)", flush=True)

    aoi = gpd.read_file(PROJ / "data/aoi/aoi.shp").to_crs(g.crs).union_all()
    g["predicted"] = g["class"].map(PREDICTED).astype("int32")
    g["area_acres"] = (g.area / ACRE_M2).round(4)
    g = g.sort_values(["predicted", "area_acres"], ascending=[False, False]).reset_index(drop=True)
    g.insert(0, "field_id", np.arange(1, len(g) + 1, dtype="int32"))
    rep, fail = qc(g, aoi)
    after = {k: float(v) for k, v in g.groupby("class").area_acres.sum().items()}
    # a dissolve of everything must keep every field separate
    # (0 intersecting/touching pairs is the proof: a dissolve can only merge polygons that touch; a full union_all of
    # 55k polygons took ~3 min and splits pinched polygons, so it is not used)
    rep["dissolve_ok"] = rep["intersecting_or_touching_pairs"] == 0
    if not rep["dissolve_ok"]:
        fail.append("dissolve")
    print("QC:", json.dumps(rep, default=float), "FAIL:" if fail else "all checks passed", fail or "", flush=True)
    if fail:
        raise SystemExit(f"QC failed: {fail}")

    pq = OUT / f"{NAME}.parquet"
    g[["field_id", "predicted", "class", "area_acres", "class_frac", "source", "uid", "Shp_Label", "Confidence", "geometry"]] \
        .to_parquet(pq, compression="zstd", write_covering_bbox=True)
    shp_dir = OUT / f"{NAME}_shp"; shp_dir.mkdir(exist_ok=True)
    for f in shp_dir.glob(f"{NAME}.*"):
        f.unlink()
    client = g[["field_id", "predicted", "class", "area_acres", "geometry"]].rename(columns={"area_acres": "area_ac"})
    client.to_file(shp_dir / f"{NAME}.shp", encoding="UTF-8")
    zp = OUT / f"{NAME}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(shp_dir.glob(f"{NAME}.*")):
            z.write(f, f.name)
    # read the shapefile back from the zip and check it again
    back = gpd.read_file(f"zip://{zp}!{NAME}.shp")
    rb, fb = qc(back, aoi)
    rep["shapefile_readback"] = {"features": rb["features"], "invalid": rb["invalid"], "touching_pairs": rb["intersecting_or_touching_pairs"],
                                 "fields": list(back.columns), "predicted_values": back.predicted.value_counts().to_dict(),
                                 "acres_by_class": back.groupby("class").area_ac.sum().round(1).to_dict(), "fail": fb}
    rep.update(geos_fallbacks=dict(tp.FALLBACKS), acres_before=before, acres_after=after, input_holes=s0["holes"], input_spikes=s0["spikes_lt_15deg"],
               polygons_by_class=g["class"].value_counts().to_dict(), predicted_code=PREDICTED,
               files={"parquet": str(pq.relative_to(PROJ)), "zip": str(zp.relative_to(PROJ))},
               sizes_mb={"parquet": round(pq.stat().st_size / 1e6, 1), "zip": round(zp.stat().st_size / 1e6, 1)})
    (OUT / f"{NAME}_qc.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("acres_before", "acres_after", "polygons_by_class", "shapefile_readback", "sizes_mb")},
                     indent=1, default=float))
    if fb:
        raise SystemExit(f"shapefile read-back QC failed: {fb}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
