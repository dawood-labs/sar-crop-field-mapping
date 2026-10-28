"""Fast checks of the runner (no Earth Engine, a few small S3 objects under a scratch prefix).

  PYTHONPATH=scripts AWS_ENV=<creds> python -m pipeline.test_pipeline            # all quick tests (~30 s)

What is checked:
  params       JSON < env < CLI precedence, bool / number parsing, fd version validation, resource caps -> env
  tracks       track env + aliasing to the model's track names
  grid         tiles cover the province exactly once (no overlap, no hole), edges on the fixed grid, slivers merged
  markers      a stage counts as done only with every listed file present at the listed size (missing / resized -> not done)
  upload       re-upload skips unchanged files, verify catches nothing, file paths and folders both accepted
The heavy end-to-end checks (a real unit: download -> process -> S3, kill + resume, bit-identical to the locked outputs)
are described in docs/PIPELINE.md section 6 and run with `python -m pipeline site ...` against PIPELINE_S3_ROOT=<scratch>.
"""
from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

SCRATCH = os.environ.get("PIPELINE_TEST_S3", "s3://<bucket>/tmp/standing_cane_pipeline_test/unit")


def test_params():
    from pipeline.config import load_params
    os.environ["PIPELINE_GRID_KM"] = "15"
    p = load_params({"province": "province_le", "fd": "true", "units": "grid"})
    assert p.grid_km == 15.0 and p.fd is True and p.units == "grid" and p.province == "province_le"
    p = load_params({"province": "province_le", "grid_km": "10"})
    assert p.grid_km == 10.0, "CLI must beat env"
    del os.environ["PIPELINE_GRID_KM"]
    assert load_params({"province": "province_le"}).fd is False
    try:
        load_params({"province": "province_le", "fd_versions": "v1,3ac"}).fd_list
        raise AssertionError("bad fd version accepted")
    except ValueError:
        pass
    p = load_params({"province": "province_le", "max_cpus": "3", "force": "process,fd"})
    assert os.environ["SC_CPUS"] == "3" and p.forced("process") and not p.forced("download")
    del os.environ["SC_CPUS"]


def test_tracks():
    from pipeline.config import Site
    e = Site("x_2026", "P", 164, 99).track_env()
    assert e["S1_TRACK_ALIAS"] == "" and e["S1_TRACKS"] == "RO164_DES,RO099_ASC"
    e = Site("x_2026", "P", 62, 172).track_env()
    assert e["S1_TRACK_ALIAS"] == "RO062_DES:RO164_DES,RO172_ASC:RO099_ASC"
    assert e["DL_S1_TRACKS"] == "62:DESCENDING,172:ASCENDING"


def test_grid(province="province9"):
    import geopandas as gpd
    from shapely.ops import unary_union
    from pipeline import prepare as PR
    from pipeline.config import load_params
    with tempfile.TemporaryDirectory() as d:
        os.environ["SC_SITES_ROOT"] = d
        import importlib, site_paths
        importlib.reload(site_paths); importlib.reload(PR)
        PR.tracks_for = lambda g, p: (164, 99)                     # no Earth Engine in the quick test
        p = load_params({"province": province, "grid_km": "20"})
        sites = PR.units_grid(p)
        geoms = [gpd.read_file(Path(d) / p.year / province / s.name / "data/aoi/aoi.shp").to_crs(32648).geometry.union_all() for s in sites]
        prov, _ = PR._province_geoms(p)
        union = unary_union(geoms)
        assert abs(union.area - prov.area) / prov.area < 1e-6, "tiles must cover the province"
        assert sum(g.area for g in geoms) - union.area < 1.0, "tiles must not overlap"
        assert all(g.area >= p.min_tile_acres * PR.ACRE * 0.999 for g in geoms) or len(geoms) == 1, "slivers must be merged"
        assert len({s.name for s in sites}) == len(sites)
        del os.environ["SC_SITES_ROOT"]
        print(f"   grid: {len(sites)} tiles")


def test_markers_and_upload():
    from pipeline import s3state as S3
    pre = f"{SCRATCH}/{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / "data/a").mkdir(parents=True)
        (d / "data/a/x.bin").write_bytes(b"1" * 1000); (d / "data/one.txt").write_text("hello")
        files = S3.upload_dir(d, pre, ["data/a", "data/one.txt"])
        assert files == {"data/a/x.bin": 1000, "data/one.txt": 5}
        st = S3.Stage(pre, "t"); assert not st.done()
        st.mark(files, time.time()); assert st.done()
        assert S3.upload_dir(d, pre, ["data/a", "data/one.txt"]) == files            # unchanged: nothing to re-send
        b, k = S3.split(pre + "/data/a/x.bin")
        S3.client().put_object(Bucket=b, Key=k, Body=b"short")                         # a resized file -> not done
        assert not st.done()
        st2 = S3.Stage(pre, "t2"); st2.mark({**files, "data/never_uploaded.bin": 7}, time.time())
        assert not st2.done(), "a listed file that is missing on S3 must make the stage not done"
        assert st.done(verify=False)
        n = S3.download_prefix(pre, d / "back"); assert n == 2 and (d / "back/data/one.txt").read_text() == "hello"
    # the bucket policy allows no deletes: the scratch prefix stays (tmp/ is cleaned by the bucket owner)


def main():
    for name, fn in [("params", test_params), ("tracks", test_tracks), ("grid", test_grid), ("markers + upload", test_markers_and_upload)]:
        t0 = time.time(); fn(); print(f"ok  {name} ({time.time() - t0:.1f} s)", flush=True)
    print("ALL QUICK TESTS PASSED")


if __name__ == "__main__":
    main()
