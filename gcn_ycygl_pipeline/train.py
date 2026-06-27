from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from .config import DATASETS, DEFAULT_WORK_DIR
from .model import FocalLoss, ThreeFeatureRumorModel
from .prepare_dataset import Sample, load_samples, make_split


class ThreeFeatureDataset(Dataset):
    def __init__(self, samples: list[Sample], gcn_dir: Path, image_dir: Path, ela_dir: Path):
        self.samples = samples
        self.gcn_dir = gcn_dir
        self.image_dir = image_dir
        self.ela_dir = ela_dir
        missing: list[str] = []
        for sample in samples:
            if not (gcn_dir / f"{sample.data_id}.pt").exists():
                missing.append(f"GCN:{sample.data_id}")
            if not (image_dir / f"{sample.data_id}.pt").exists():
                missing.append(f"IMG:{sample.data_id}")
            if not (ela_dir / f"{sample.data_id}.pt").exists():
                missing.append(f"ELA:{sample.data_id}")
        if missing:
            preview = ", ".join(missing[:20])
            raise FileNotFoundError(f"Missing {len(missing)} feature files. First items: {preview}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_feature(self, folder: Path, data_id: str) -> torch.Tensor:
        tensor = torch.load(folder / f"{data_id}.pt", map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        return tensor.float().view(-1)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return {
            "data_id": sample.data_id,
            "gcn": self._load_feature(self.gcn_dir, sample.data_id),
            "image": self._load_feature(self.image_dir, sample.data_id),
            "ela": self._load_feature(self.ela_dir, sample.data_id),
            "label": torch.tensor(sample.label, dtype=torch.float32),
        }


def subset(samples: list[Sample], split: dict[str, str], name: str) -> list[Sample]:
    return [sample for sample in samples if split[sample.data_id] == name]


def build_loader(
    samples: list[Sample],
    gcn_dir: Path,
    image_dir: Path,
    ela_dir: Path,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ThreeFeatureDataset(samples, gcn_dir, image_dir, ela_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def compute_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    preds = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    labels: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    for batch in loader:
        gcn = batch["gcn"].to(device)
        image = batch["image"].to(device)
        ela = batch["ela"].to(device)
        target = batch["label"].to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(gcn, image, ela)
            loss = loss_fn(logits, target)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        labels.append(target.detach().cpu().numpy())
        logits_list.append(logits.detach().cpu().numpy())

    y_true = np.concatenate(labels)
    y_logit = np.concatenate(logits_list)
    metrics = compute_metrics(y_true.astype(int), y_logit)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


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
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_samples = subset(samples, split, "train")
    valid_samples = subset(samples, split, "valid")
    test_samples = subset(samples, split, "test")
    if not valid_samples:
        valid_samples = test_samples

    train_loader = build_loader(train_samples, gcn_dir, image_dir, ela_dir, batch_size, True)
    valid_loader = build_loader(valid_samples, gcn_dir, image_dir, ela_dir, batch_size, False)
    test_loader = build_loader(test_samples, gcn_dir, image_dir, ela_dir, batch_size, False)

    model = ThreeFeatureRumorModel(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout).to(device_obj)
    loss_fn = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_state = None
    best_valid_loss = float("inf")
    stale_epochs = 0
    log_path = out_dir / "training_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "valid_loss", "valid_accuracy", "valid_f1_weighted"])
        for epoch in range(epochs):
            train_metrics = run_epoch(model, train_loader, loss_fn, device_obj, optimizer)
            valid_metrics = run_epoch(model, valid_loader, loss_fn, device_obj)
            writer.writerow(
                [
                    epoch,
                    train_metrics["loss"],
                    valid_metrics["loss"],
                    valid_metrics["accuracy"],
                    valid_metrics["f1_weighted"],
                ]
            )
            if wandb_run is not None:
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
            if valid_metrics["loss"] < best_valid_loss:
                best_valid_loss = valid_metrics["loss"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "best_three_feature_model.pt")
    test_metrics = run_epoch(model, test_loader, loss_fn, device_obj)
    metrics_path = out_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.log({f"test_{key}": value for key, value in test_metrics.items()})
    return test_metrics


def maybe_init_wandb(args):
    if not args.use_wandb:
        return None
    import wandb

    return wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name)


def main() -> None:
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
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="gcn_ycygl_three_feature")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

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
            wandb_run=wandb_run,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

