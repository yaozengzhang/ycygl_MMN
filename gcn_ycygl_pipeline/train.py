from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .config import DATASETS, DEFAULT_WORK_DIR
from .model import ThreeFeatureRumorModel
from .prepare_dataset import Sample, load_samples, make_split


@dataclass
class TrainingSetup:
    model: ThreeFeatureRumorModel
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader
    optimizer: torch.optim.Optimizer
    loss_fn: BCEWithLogitsLoss
    device: torch.device


class ThreeFeatureDataset(Dataset):
    def __init__(self, samples: list[Sample], gcn_dir: Path, image_dir: Path, ela_dir: Path):
        self.samples = samples
        self.gcn_dir = gcn_dir
        self.image_dir = image_dir
        self.ela_dir = ela_dir
        self._check_feature_files()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        return {
            "data_id": sample.data_id,
            "gcn": self.load_feature(self.gcn_dir, sample.data_id),
            "image": self.load_feature(self.image_dir, sample.data_id),
            "ela": self.load_feature(self.ela_dir, sample.data_id),
            "label": torch.tensor(sample.label, dtype=torch.float32),
        }

    def _check_feature_files(self) -> None:
        missing: list[str] = []
        for sample in self.samples:
            if not (self.gcn_dir / f"{sample.data_id}.pt").exists():
                missing.append(f"GCN:{sample.data_id}")
            if not (self.image_dir / f"{sample.data_id}.pt").exists():
                missing.append(f"IMG:{sample.data_id}")
            if not (self.ela_dir / f"{sample.data_id}.pt").exists():
                missing.append(f"ELA:{sample.data_id}")
        if missing:
            preview = ", ".join(missing[:20])
            raise FileNotFoundError(f"Missing {len(missing)} feature files. First items: {preview}")

    @staticmethod
    def load_feature(folder: Path, data_id: str) -> torch.Tensor:
        tensor = torch.load(folder / f"{data_id}.pt", map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        return tensor.float().view(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="weibo")
    parser.add_argument("--feature_root", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="gcn_ycygl_three_feature")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def split_samples(samples: list[Sample], split: dict[str, str]) -> tuple[list[Sample], list[Sample], list[Sample]]:
    train_samples = [sample for sample in samples if split[sample.data_id] == "train"]
    valid_samples = [sample for sample in samples if split[sample.data_id] == "valid"]
    test_samples = [sample for sample in samples if split[sample.data_id] == "test"]
    if not valid_samples:
        valid_samples = test_samples
    if not train_samples:
        raise ValueError("No train samples found in dataset split.")
    if not test_samples:
        raise ValueError("No test samples found in dataset split.")
    return train_samples, valid_samples, test_samples


def build_dataloader(
    samples: list[Sample],
    gcn_dir: Path,
    image_dir: Path,
    ela_dir: Path,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ThreeFeatureDataset(samples, gcn_dir, image_dir, ela_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def build_model(hidden_dim: int, num_heads: int, dropout: float, device: torch.device) -> ThreeFeatureRumorModel:
    model = ThreeFeatureRumorModel(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
    return model.to(device)


def build_optimizer(model: ThreeFeatureRumorModel, learning_rate: float) -> AdamW:
    return AdamW(model.parameters(), lr=learning_rate)


def move_batch_to_device(batch: dict[str, torch.Tensor | list[str]], device: torch.device):
    return (
        batch["gcn"].to(device),
        batch["image"].to(device),
        batch["ela"].to(device),
        batch["label"].to(device),
    )


def compute_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    preds = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }


def train_epoch(
    model: ThreeFeatureRumorModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: BCEWithLogitsLoss,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    labels: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []

    for batch in train_loader:
        gcn, image, ela, target = move_batch_to_device(batch, device)
        logits = model(gcn, image, ela)
        loss = loss_fn(logits, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        labels.append(target.detach().cpu().numpy())
        logits_list.append(logits.detach().cpu().numpy())

    return summarize_epoch(labels, logits_list, losses)


def eval_epoch(
    model: ThreeFeatureRumorModel,
    valid_loader: DataLoader,
    loss_fn: BCEWithLogitsLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    labels: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in valid_loader:
            gcn, image, ela, target = move_batch_to_device(batch, device)
            logits = model(gcn, image, ela)
            loss = loss_fn(logits, target)
            losses.append(float(loss.detach().cpu()))
            labels.append(target.detach().cpu().numpy())
            logits_list.append(logits.detach().cpu().numpy())

    return summarize_epoch(labels, logits_list, losses)


def test_epoch(
    model: ThreeFeatureRumorModel,
    test_loader: DataLoader,
    loss_fn: BCEWithLogitsLoss,
    device: torch.device,
) -> dict[str, float]:
    return eval_epoch(model, test_loader, loss_fn, device)


def summarize_epoch(
    labels: list[np.ndarray],
    logits_list: list[np.ndarray],
    losses: list[float],
) -> dict[str, float]:
    if not labels:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "f1_weighted": 0.0,
            "f1_macro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
        }
    y_true = np.concatenate(labels).astype(int)
    y_logit = np.concatenate(logits_list)
    metrics = compute_metrics(y_true, y_logit)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def test_score_model(
    model: ThreeFeatureRumorModel,
    test_loader: DataLoader,
    loss_fn: BCEWithLogitsLoss,
    device: torch.device,
) -> dict[str, float]:
    return test_epoch(model, test_loader, loss_fn, device)


def prep_for_training(
    samples: list[Sample],
    split: dict[str, str],
    gcn_dir: Path,
    image_dir: Path,
    ela_dir: Path,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
    device: str | None,
) -> TrainingSetup:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_samples, valid_samples, test_samples = split_samples(samples, split)
    train_loader = build_dataloader(train_samples, gcn_dir, image_dir, ela_dir, batch_size, shuffle=True)
    valid_loader = build_dataloader(valid_samples, gcn_dir, image_dir, ela_dir, batch_size, shuffle=False)
    test_loader = build_dataloader(test_samples, gcn_dir, image_dir, ela_dir, batch_size, shuffle=False)
    model = build_model(hidden_dim, num_heads, dropout, device_obj)
    optimizer = build_optimizer(model, learning_rate)
    loss_fn = BCEWithLogitsLoss()
    return TrainingSetup(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device_obj,
    )


def write_log_header(log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "valid_loss", "valid_accuracy", "valid_f1_weighted"])


def write_log_row(log_path: Path, epoch: int, train_metrics: dict[str, float], valid_metrics: dict[str, float]) -> None:
    with log_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                epoch,
                train_metrics["loss"],
                valid_metrics["loss"],
                valid_metrics["accuracy"],
                valid_metrics["f1_weighted"],
            ]
        )


def log_epoch_to_wandb(
    wandb_run,
    epoch: int,
    train_metrics: dict[str, float],
    valid_metrics: dict[str, float],
) -> None:
    if wandb_run is None:
        return
    wandb_run.log(
        {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_f1_weighted": valid_metrics["f1_weighted"],
            "valid_f1_macro": valid_metrics["f1_macro"],
        }
    )


def train(
    setup: TrainingSetup,
    out_dir: Path,
    epochs: int,
    patience: int,
    wandb_run=None,
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"
    write_log_header(log_path)

    best_state = None
    best_valid_loss = float("inf")
    stale_epochs = 0

    for epoch in range(epochs):
        train_metrics = train_epoch(setup.model, setup.train_loader, setup.optimizer, setup.loss_fn, setup.device)
        valid_metrics = eval_epoch(setup.model, setup.valid_loader, setup.loss_fn, setup.device)
        write_log_row(log_path, epoch, train_metrics, valid_metrics)
        log_epoch_to_wandb(wandb_run, epoch, train_metrics, valid_metrics)

        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in setup.model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        setup.model.load_state_dict(best_state)

    torch.save(setup.model.state_dict(), out_dir / "best_three_feature_model.pt")
    test_metrics = test_score_model(setup.model, setup.test_loader, setup.loss_fn, setup.device)
    (out_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if wandb_run is not None:
        wandb_run.log({f"test_{key}": value for key, value in test_metrics.items()})
    return test_metrics


def train_three_feature_model(
    samples: list[Sample],
    split: dict[str, str],
    gcn_dir: Path,
    image_dir: Path,
    ela_dir: Path,
    out_dir: Path,
    batch_size: int = 128,
    epochs: int = 50,
    patience: int = 10,
    learning_rate: float = 1e-4,
    hidden_dim: int = 256,
    num_heads: int = 4,
    dropout: float = 0.5,
    seed: int = 100,
    device: str | None = None,
    wandb_run=None,
) -> dict[str, float]:
    set_random_seed(seed)
    setup = prep_for_training(
        samples=samples,
        split=split,
        gcn_dir=gcn_dir,
        image_dir=image_dir,
        ela_dir=ela_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        device=device,
    )
    return train(setup=setup, out_dir=out_dir, epochs=epochs, patience=patience, wandb_run=wandb_run)


def maybe_init_wandb(args: argparse.Namespace):
    if not args.use_wandb:
        return None
    import wandb

    return wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name)


def main() -> None:
    args = parse_args()
    dataset_paths = DATASETS[args.dataset]
    feature_root = args.feature_root or (DEFAULT_WORK_DIR / args.dataset)
    out_dir = args.out_dir or (feature_root / "three_feature_train")
    samples = load_samples(dataset_paths.sample_dir, limit=args.limit)
    split = make_split(samples, seed=args.seed)
    wandb_run = maybe_init_wandb(args)
    try:
        metrics = train_three_feature_model(
            samples=samples,
            split=split,
            gcn_dir=feature_root / "gcn_features",
            image_dir=feature_root / "image_features",
            ela_dir=feature_root / "ela_features",
            out_dir=out_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            seed=args.seed,
            device=args.device,
            wandb_run=wandb_run,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
