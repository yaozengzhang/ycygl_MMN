from __future__ import annotations

import argparse
import json
from pathlib import Path

from gcn_ycygl_pipeline.config import DATASETS, DEFAULT_WORK_DIR
from gcn_ycygl_pipeline.image_features import generate_image_features
from gcn_ycygl_pipeline.prepare_dataset import load_samples, make_split, write_textgcn_files
from gcn_ycygl_pipeline.textgcn_features import train_textgcn_and_save
from gcn_ycygl_pipeline.train import train_three_feature_model


def all_feature_files_exist(samples, folder: Path) -> bool:
    return folder.exists() and all((folder / f"{sample.data_id}.pt").exists() for sample in samples)


def init_wandb(args, config: dict):
    if not args.use_wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name or f"{args.dataset}_three_feature",
        config=config,
    )


def save_split(samples, split: dict[str, str], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(f"{sample.data_id}\t{split[sample.data_id]}\t{sample.label}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="weibo")
    parser.add_argument("--work_dir", type=Path, default=None)
    parser.add_argument("--stages", nargs="+", choices=["gcn", "image", "train", "all"], default=["all"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image_pretrained", action="store_true")

    parser.add_argument("--gcn_epochs", type=int, default=200)
    parser.add_argument("--gcn_hidden_dim", type=int, default=200)
    parser.add_argument("--gcn_lr", type=float, default=0.02)
    parser.add_argument("--gcn_dropout", type=float, default=0.226)
    parser.add_argument("--gcn_patience", type=int, default=10)
    parser.add_argument("--no_pmi", action="store_true")
    parser.add_argument("--max_vocab", type=int, default=None)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="gcn_ycygl_three_feature")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    dataset_paths = DATASETS[args.dataset]
    work_dir = args.work_dir or (DEFAULT_WORK_DIR / args.dataset)
    work_dir.mkdir(parents=True, exist_ok=True)
    stages = {"gcn", "image", "train"} if "all" in args.stages else set(args.stages)

    samples = load_samples(dataset_paths.sample_dir, limit=args.limit)
    split = make_split(samples, seed=args.seed)
    save_split(samples, split, work_dir / "split.tsv")
    write_textgcn_files(samples, split, work_dir, args.dataset)

    wandb_run = init_wandb(
        args,
        {
            "dataset": args.dataset,
            "sample_count": len(samples),
            "work_dir": str(work_dir),
            "sample_dir": str(dataset_paths.sample_dir),
            "image_dir": str(dataset_paths.image_dir),
            "gcn_hidden_dim": args.gcn_hidden_dim,
            "fusion_hidden_dim": args.hidden_dim,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
    )

    try:
        if "gcn" in stages:
            gcn_dir = work_dir / "gcn_features"
            if args.overwrite or not all_feature_files_exist(samples, gcn_dir):
                train_textgcn_and_save(
                    samples=samples,
                    split=split,
                    out_dir=work_dir,
                    hidden_dim=args.gcn_hidden_dim,
                    epochs=args.gcn_epochs,
                    lr=args.gcn_lr,
                    dropout=args.gcn_dropout,
                    patience=args.gcn_patience,
                    seed=args.seed,
                    device=args.device,
                    use_pmi=not args.no_pmi,
                    max_features=args.max_vocab,
                    wandb_run=wandb_run,
                )
            else:
                print(f"Skip GCN: all files already exist in {gcn_dir}")

        if "image" in stages:
            generate_image_features(
                samples=samples,
                image_dir=dataset_paths.image_dir,
                out_dir=work_dir,
                overwrite=args.overwrite,
                pretrained=args.image_pretrained,
                device=args.device,
            )

        if "train" in stages:
            metrics = train_three_feature_model(
                samples=samples,
                split=split,
                gcn_dir=work_dir / "gcn_features",
                image_dir=work_dir / "image_features",
                ela_dir=work_dir / "ela_features",
                out_dir=work_dir / "three_feature_train",
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
