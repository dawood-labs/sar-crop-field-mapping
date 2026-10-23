"""Write QGIS style files (.qml, same basename as the raster -> QGIS loads them automatically).

    from qgis_style import paletted, pseudocolor
    paletted("map.tif", {0: ("#D5D8DC", "non-cane"), 1: ("#33ef2a", "Sugarcane")})
    pseudocolor("prob.tif", [(0, "#f7f7f7", "0"), (50, "#fee08b", "50"), (100, "#1a9850", "100")])
usage (styles for the standard map rasters of this project): python scripts/qgis_style.py
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

PROJ = Path(__file__).resolve().parents[1]

HEAD = '<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n<qgis version="3.28" styleCategories="AllStyleCategories">\n <pipe>\n'
TAIL = ' </pipe>\n</qgis>\n'


def _rgba(hex_):
    h = hex_.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},255"


def paletted(raster, classes: dict, band=1):
    """classes: {value: (hex colour, label)}; nodata stays transparent."""
    rows = "\n".join(f'    <paletteEntry value="{v}" color="{c}" alpha="255" label="{escape(lb)}"/>' for v, (c, lb) in classes.items())
    xml = (HEAD + f'  <rasterrenderer type="paletted" band="{band}" opacity="1" alphaBand="-1" nodataColor="">\n'
           f'   <colorPalette>\n{rows}\n   </colorPalette>\n  </rasterrenderer>\n' + TAIL)
    out = Path(raster).with_suffix(".qml")
    out.write_text(xml)
    return out


def pseudocolor(raster, stops, band=1, vmin=None, vmax=None):
    """stops: [(value, hex colour, label)], linear interpolation between stops."""
    vmin = stops[0][0] if vmin is None else vmin
    vmax = stops[-1][0] if vmax is None else vmax
    items = "\n".join(f'       <item value="{v}" color="{c}" alpha="255" label="{escape(lb)}"/>' for v, c, lb in stops)
    xml = (HEAD + f'  <rasterrenderer type="singlebandpseudocolor" band="{band}" opacity="1" alphaBand="-1" '
           f'classificationMin="{vmin}" classificationMax="{vmax}" nodataColor="">\n'
           f'   <rastershader>\n    <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0" '
           f'minimumValue="{vmin}" maximumValue="{vmax}">\n{items}\n    </colorrampshader>\n   </rastershader>\n'
           f'  </rasterrenderer>\n' + TAIL)
    out = Path(raster).with_suffix(".qml")
    out.write_text(xml)
    return out


PROB_STOPS = [(0, "#f7f7f7", "0 = surely non-cane"), (30, "#fdd49e", "30"), (50, "#fee08b", "50 (threshold)"),
              (70, "#a6d96a", "70"), (100, "#1a9850", "100 = surely cane")]
MAP_CLASSES = {0: ("#D5D8DC", "Non-cane"), 1: ("#33ef2a", "Sugarcane (company predicted 1)")}
COMPARE_CLASSES = {0: ("#e6e6e6", "non-cane both"), 1: ("#33bf33", "cane both"),
                   2: ("#e63333", "label cane, map non-cane"), 3: ("#1a66ff", "label non-cane, map cane")}


def main():
    done = []
    for f in sorted((PROJ / "data/model").rglob("*.tif")):
        n = f.stem
        if n.startswith("cane_map_"):
            done.append(paletted(f, MAP_CLASSES))
        elif n.startswith("cane_prob_"):
            done.append(pseudocolor(f, PROB_STOPS))
        elif n.startswith("map_vs_label_"):
            done.append(paletted(f, COMPARE_CLASSES))
    for f in (PROJ / "data/labels").glob("cane2025_qc.tif"):
        done.append(paletted(f, {0: ("#D5D8DC", "non-cane (clean)"), 1: ("#33ef2a", "cane (clean)")}))   # 255 (excluded or outside) not in the palette = transparent
    print("\n".join(str(p) for p in done))


if __name__ == "__main__":
    main()
