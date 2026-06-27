from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path(os.environ.get("YCYGL_DATA_ROOT", "data/raw"))


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    sample_dir: Path
    image_dir: Path


DATASETS = {
    "weibo": DatasetPaths(
        name="weibo",
        sample_dir=DATA_ROOT / "weibo" / "text",
        image_dir=DATA_ROOT / "weibo" / "images",
    ),
    "twitter": DatasetPaths(
        name="twitter",
        sample_dir=DATA_ROOT / "twitter" / "text",
        image_dir=DATA_ROOT / "twitter" / "images",
    ),
}


DEFAULT_WORK_DIR = Path(os.environ.get("YCYGL_WORK_DIR", "runs"))


@dataclass
class WandbConfig:
    use_wandb: bool = False
    project: str = "gcn_ycygl_three_feature"
    entity: str | None = None
    run_name: str | None = None
