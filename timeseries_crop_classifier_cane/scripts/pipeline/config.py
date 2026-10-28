"""Run parameters + province plan.

Params (one place for every knob): defaults in config/standing_cane_2026.json, overridden by PIPELINE_* env vars, overridden
by CLI flags. A DAG passes only what it wants to change; `province` is the one required value.

Plan (what to process for a province): written once by `prepare` to S3  <s3_root>/<year>/<Province>/_plan/plan.json
(sites with AOIs + S1 tracks, the params that shaped them). Every later task reads it from S3, so a fresh pod needs nothing local.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJ / "config"
MODEL_DES, MODEL_ASC = "RO164_DES", "RO099_ASC"      # the track names the layer-1 model was trained with
FD_ALL = ("v1", "1ac", "1p5ac", "2ac")


@dataclass
class Params:
    province: str = ""                    # REQUIRED, folder style with "_" for spaces, e.g. "province5"
    units: str = "districts"              # "districts" (GAUL district AOIs + gap AOIs) | "grid" (square tiles clipped to the province)
    grid_km: float = 20.0                 # grid mode: tile size (20 km ~ 100k acres, the size of an average district)
    min_tile_acres: float = 5000.0        # grid mode: a tile with less province inside is merged into its biggest neighbour
    fd: bool = False                      # also make the field-delineation sugarcane polygons for this province
    fd_versions: str = ",".join(FD_ALL)   # which FD versions (v1 = cut below 75 % cane; Xac = keep whole while non-cane <= X ac)
    year: str = "2026"
    season_start: str = "2025-11-01"      # S1 / S2 download window (end exclusive)
    season_end: str = "2026-10-15"
    s3_root: str = "s3://$S3_BUCKET/$S3_PREFIX"   # outputs + state
    inputs_root: str = ""                 # models / source tables (default = s3_root); tests write elsewhere but read here
    layer1_model: str = ""                # S3 URI; default from the JSON (locked s2on_r2)
    colour_model: str = ""                # S3 URI; default from the JSON (locked colour v8)
    delineation_root: str = ""            # S3 prefix holding <Province>/ *_fields_2026.parquet + *Boundary.gpkg
    keep_local: bool = False              # keep raw + stacks on the pod after upload (debugging)
    force: str = ""                       # comma list of stages to redo although done: download,process,province,fd
    min_free_gb: float = 35.0             # refuse to start a download below this free disk
    disk_wait_s: int = 0                  # >0: wait up to this long for disk instead of failing (shared pod)
    max_cpus: int = 0                     # 0 = all the pod gives (cgroup quota aware); >0 caps it
    max_mem_gb: float = 0.0               # 0 = the pod's request (MEM_GUARANTEE) or the free RAM; >0 caps it
    ee_key_file: str = ""                 # Earth Engine service-account json (path or s3:// URI); env EE_KEY_FILE
    ee_project: str = __import__("os").environ.get("EE_PROJECT", "")
    workload_tag: str = "standing-cane"   # Earth Engine billing label
    dry_run: bool = False                 # print the plan / task list only
    extra: dict = field(default_factory=dict)   # rest of the JSON (shared inputs, download env)

    def forced(self, stage: str) -> bool:
        return stage in {s.strip() for s in self.force.split(",") if s.strip()}

    @property
    def fd_list(self) -> list[str]:
        v = [x.strip() for x in self.fd_versions.split(",") if x.strip()]
        bad = [x for x in v if x not in FD_ALL]
        if bad:
            raise ValueError(f"unknown fd versions {bad}; choose from {FD_ALL}")
        return v

    def prefix(self) -> str:
        return f"{self.s3_root.rstrip('/')}/{self.year}/{self.province}"

    def apply_resources(self):
        """Pod caps go to the env the science scripts read (safe_resources.py)."""
        if self.max_cpus:
            os.environ["SC_CPUS"] = str(self.max_cpus)
        if self.max_mem_gb:
            os.environ["MEM_GUARANTEE"] = str(int(self.max_mem_gb * 2**30))
        if self.ee_key_file:
            os.environ["EE_KEY_FILE"] = self.ee_key_file
        os.environ["EE_PROJECT"] = self.ee_project


def _coerce(t, v):
    if t in (bool, "bool"):
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "y", "on")
    if t in (int, "int"):
        return int(v)
    if t in (float, "float"):
        return float(v)
    return v


def load_params(cli: dict | None = None) -> Params:
    js = json.loads(Path(os.environ.get("PIPELINE_CONFIG", CONFIG_DIR / "standing_cane_2026.json")).read_text())
    p = Params()
    names = {f.name: f.type for f in fields(Params)}
    for k, v in js.items():                                  # 1) JSON defaults
        if k in names and k != "extra":
            setattr(p, k, _coerce(names[k], v))
    p.extra = {k: v for k, v in js.items() if k not in names}
    for k, t in names.items():                               # 2) env PIPELINE_<NAME>
        e = os.environ.get(f"PIPELINE_{k.upper()}")
        if e is not None and k != "extra":
            setattr(p, k, _coerce(t, e))
    for k, v in (cli or {}).items():                         # 3) CLI flags
        if v is not None and k in names:
            setattr(p, k, _coerce(names[k], v))
    p.inputs_root = p.inputs_root or p.s3_root
    p.extra["shared_inputs"] = {k: v.replace("{inputs_root}", p.inputs_root.rstrip("/")) for k, v in p.extra.get("shared_inputs", {}).items()}
    p.layer1_model = p.layer1_model or p.extra["shared_inputs"]["data/model/s2on_r2/model.joblib"]
    p.colour_model = p.colour_model or p.extra["shared_inputs"]["sites/layer2/colour_model_v8.joblib"]
    p.delineation_root = p.delineation_root or f"{p.inputs_root.rstrip('/')}/shared/delineation"
    if p.units not in ("districts", "grid"):
        raise ValueError("units must be 'districts' or 'grid'")
    p.apply_resources()
    return p


@dataclass
class Site:
    name: str                 # folder / key name, e.g. "site19_2026", "site34_2026", "province6_g007_2026"
    province: str
    des: int                  # relative orbit of the descending track
    asc: int                  # relative orbit of the ascending track
    kind: str = "district"    # "district" | "gap" | "grid"

    def track_env(self) -> dict:
        """S1 env for download.py / s1_composites.py: own tracks, aliased to the model's names when they differ."""
        d, a = f"RO{self.des:03d}_DES", f"RO{self.asc:03d}_ASC"
        alias = [f"{x}:{m}" for x, m in ((d, MODEL_DES), (a, MODEL_ASC)) if x != m]
        return {"DL_S1_TRACKS": f"{self.des}:DESCENDING,{self.asc}:ASCENDING", "S1_TRACKS": f"{d},{a}",
                "S1_TRACK_ALIAS": ",".join(alias)}


@dataclass
class Province:
    name: str
    units: str = "districts"
    sites: list[Site] = field(default_factory=list)

    @property
    def gaps(self) -> list[Site]:
        return [s for s in self.sites if s.kind == "gap"]

    def site(self, name: str) -> Site:
        name = name if name.endswith("_2026") else f"{name}_2026"
        for s in self.sites:
            if s.name == name:
                return s
        raise KeyError(f"{name} is not in the plan of {self.name}")

    def to_json(self, params: Params | None = None) -> dict:
        return {"province": self.name, "units": self.units,
                "params": {k: v for k, v in asdict(params).items() if k != "extra"} if params else {},
                "sites": [{"name": s.name, "des": s.des, "asc": s.asc, "kind": s.kind} for s in self.sites]}

    @staticmethod
    def from_json(d: dict) -> "Province":
        return Province(d["province"], d.get("units", "districts"), [Site(province=d["province"], **s) for s in d["sites"]])


def plan_uri(p: Params) -> str:
    return f"{p.prefix()}/_plan/plan.json"


def load_plan(p: Params) -> Province:
    """The province plan from S3 (falls back to config/provinces/<P>.json made on the JupyterHub pod)."""
    from . import s3state as S3
    d = S3.read_json(plan_uri(p))
    if d is None:
        f = CONFIG_DIR / "provinces" / f"{p.province}.json"
        if not f.exists():
            raise RuntimeError(f"no plan for {p.province}: run `python -m pipeline prepare --province {p.province}` first")
        d = json.loads(f.read_text())
    return Province.from_json(d)


def save_plan(prov: Province, p: Params):
    from . import s3state as S3
    body = prov.to_json(p)
    S3.write_json(plan_uri(p), body)
    out = CONFIG_DIR / "provinces" / f"{prov.name}.json"          # local copy for people reading the repo / the pod
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(body, indent=1)); os.replace(tmp, out)
