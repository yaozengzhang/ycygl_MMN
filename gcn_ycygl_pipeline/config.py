from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path(r"F:\原电脑深度学习相关\多模态谣言检测数据")


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    sample_dir: Path
    image_dir: Path


DATASETS = {
    "weibo": DatasetPaths(
        name="weibo",
        sample_dir=DATA_ROOT / "微博" / "ALL_textimage",
        image_dir=DATA_ROOT / "微博" / "ALL_pic",
    ),
    "twitter": DatasetPaths(
        name="twitter",
        sample_dir=DATA_ROOT / "推特" / "new_twitter_list",
        image_dir=DATA_ROOT / "推特" / "ALLPIL",
    ),
}


DEFAULT_WORK_DIR = Path(r"E:\task1\gcn_ycygl_pipeline\runs")


@dataclass
class WandbConfig:
    use_wandb: bool = False
    project: str = "gcn_ycygl_three_feature"
    entity: str | None = None
    run_name: str | None = None

