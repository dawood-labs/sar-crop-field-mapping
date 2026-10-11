"""Earth Engine + GCS auth (service account). Key file path is relative to cfg['_root']."""
from __future__ import annotations

from pathlib import Path

import ee
from google.cloud import storage
from google.oauth2 import service_account


SCOPES = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/cloud-platform",
]

_CREDS = None


def root(cfg: dict) -> Path:
    return Path(cfg["_root"])


def credentials(cfg: dict):
    global _CREDS
    if _CREDS is None:
        key = root(cfg) / cfg["auth"]["key_file"]
        _CREDS = service_account.Credentials.from_service_account_file(str(key)).with_scopes(SCOPES)
    return _CREDS


def init_ee(cfg: dict, high_volume: bool = False) -> None:
    kwargs = dict(credentials=credentials(cfg), project=cfg["auth"]["project"])
    if high_volume:
        # for many interactive getInfo/computePixels calls; batch exports do not need it
        kwargs["opt_url"] = "https://earthengine-highvolume.googleapis.com"
    ee.Initialize(**kwargs)


def gcs_bucket(cfg: dict) -> storage.Bucket:
    client = storage.Client(credentials=credentials(cfg), project=cfg["auth"]["project"])
    return client.bucket(cfg["gcs"]["bucket"])
