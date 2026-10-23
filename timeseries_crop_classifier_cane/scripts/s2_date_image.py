"""Write per-date Sentinel-2 images (B2 B3 B4 B5 B8, uint16 x 10000) for QGIS from the band stacks, for the clearest October
and September dates of a district (raw per-date files are deleted after processing; the stacks hold every date).
usage: python scripts/s2_date_image.py sites/<site> [YYYYMMDD ...]   (no dates = clearest Oct + clearest Sep date over the AOI)
       python scripts/s2_date_image.py sites/<site> --monthly 202606 202610   (clearest date of every month in the range,
       listed in <site>/data/s2_dates/monthly_clearest.csv; a month with no date in the stacks is reported as missing)
Output: <site>/data/s2_dates/S2_<date>.tif + .qml (5-3-2 false colour: bands 4,2,1)"""
import sys
from pathlib import Path
import numpy as np, rasterio
import site_paths as SP
P = Path(__file__).resolve().parents[1]
QML = """<!DOCTYPE qgis><qgis version="3.34"><pipe><rasterrenderer type="multibandcolor" redBand="4" greenBand="2" blueBand="1" opacity="1">
<redContrastEnhancement><minValue>200</minValue><maxValue>3500</maxValue><algorithm>StretchToMinimumMaximum</algorithm></redContrastEnhancement>
<greenContrastEnhancement><minValue>200</minValue><maxValue>3500</maxValue><algorithm>StretchToMinimumMaximum</algorithm></greenContrastEnhancement>
<blueContrastEnhancement><minValue>200</minValue><maxValue>3500</maxValue><algorithm>StretchToMinimumMaximum</algorithm></blueContrastEnhancement>
</rasterrenderer></pipe></qgis>"""


def clear_share(site, keep=lambda d: True):
    with rasterio.open(site / "data/stacks/S2_cs_cdf.tif") as c, rasterio.open(site / "data/grid/index.tif") as g:
        aoi = g.read(2) > 0; out = {}
        sel = [(i + 1, d) for i, d in enumerate(c.descriptions) if keep(d)]
        if not sel: return out
        arr = c.read([i for i, _ in sel])                      # ONE read: the stack is pixel-interleaved, per-band reads decode it every time
        for (i, d), cs in zip(sel, arr):
            out[d] = float(((cs >= 60) & (cs <= 100))[aoi].mean())
    return out


def write_many(site, dates):
    """All dates in one read per band stack (pixel-interleaved stacks: one decode instead of one per date)."""
    st = site / "data/stacks"; o = site / "data/s2_dates"; o.mkdir(exist_ok=True)
    with rasterio.open(st / "S2_B8.tif") as r: desc = list(r.descriptions); prof = r.profile.copy()
    idx = [desc.index(d) + 1 for d in dates]
    bands = {}
    for b in ("B2", "B3", "B4", "B5", "B8"):
        with rasterio.open(st / f"S2_{b}.tif") as r: bands[b] = r.read(idx)
    prof.update(count=5, dtype="uint16", nodata=0, compress="deflate", predictor=2, interleave="pixel")
    out = []
    for j, date in enumerate(dates):
        f = o / f"S2_{date}.tif"
        with rasterio.open(f, "w", **prof) as d:
            for k, b in enumerate(("B2", "B3", "B4", "B5", "B8"), 1): d.write(bands[b][j], k); d.set_band_description(k, b)
        f.with_suffix(".qml").write_text(QML); out.append(f)
    return out


def write(site, date):
    return write_many(site, [date])[0]


if __name__ == "__main__":
    site = SP.site_dir(sys.argv[1]); dates = sys.argv[2:]
    if dates[:1] == ["--monthly"]:
        a, b = dates[1], dates[2]
        months = [m for m in (f"{y}{mo:02d}" for y in range(int(a[:4]), int(b[:4]) + 1) for mo in range(1, 13)) if a <= m <= b]
        cs = clear_share(site, lambda d: d[:6] in months); rows = []
        for m in months:
            ds = [d for d in cs if d.startswith(m)]
            if not ds: rows.append(f"{m},,,missing"); print(site.name, m, "MISSING (no date in the stacks)"); continue
            d = max(ds, key=lambda d: cs[d]); rows.append(f"{m},{d},{cs[d]:.3f},ok")
            print(site.name, m, d, f"{cs[d]:.0%} clear")
        write_many(site, [r.split(",")[1] for r in rows if r.endswith(",ok")])
        (site / "data/s2_dates").mkdir(exist_ok=True)
        (site / "data/s2_dates/monthly_clearest.csv").write_text("month,date,clear_share,status\n" + "\n".join(rows) + "\n")
        sys.exit()
    if not dates:
        cs = clear_share(site)
        best = lambda pre: max((d for d in cs if d.startswith(pre)), key=lambda d: cs[d], default=None)
        dates = [d for d in (best("202610"), best("202609")) if d]
        print(site.name, "clearest:", {d: f"{cs[d]:.0%}" for d in dates})
    for d in dates: print("  ->", write(site, d))
