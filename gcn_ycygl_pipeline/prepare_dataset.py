from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Sample:
    data_id: str
    text: str
    image_id: str
    label: int
    split: str | None = None


def parse_label(raw: str) -> int:
    value = raw.strip().lower()
    if value in {"0", "nonrumor", "non-rumor", "real", "true", "truth"}:
        return 0
    if value in {"1", "rumor", "fake", "false"}:
        return 1
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"Unsupported label: {raw!r}") from exc


def read_sample_file(path: Path) -> Sample:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 4:
        raise ValueError(f"Expected at least 4 lines in {path}")
    first_field = lines[1].strip()
    second_field = lines[2].strip()
    if looks_like_image_id(first_field) and not looks_like_image_id(second_field):
        image_id = first_field
        text = second_field
    else:
        text = first_field
        image_id = second_field
    return Sample(
        data_id=lines[0].strip(),
        text=text,
        image_id=image_id,
        label=parse_label(lines[3]),
        split=parse_split(lines[4]) if len(lines) >= 5 else None,
    )


def looks_like_image_id(value: str) -> bool:
    if not value or len(value) > 180:
        return False
    if re.search(r"\s", value):
        return False
    if re.search(r"[\u4e00-\u9fff]{4,}", value):
        return False
    return True


def parse_split(raw: str) -> str | None:
    value = raw.strip().lower()
    if value in {"train", "training"}:
        return "train"
    if value in {"valid", "val", "dev", "validation"}:
        return "valid"
    if value in {"test", "testing"}:
        return "test"
    return None


def load_samples(sample_dir: Path, limit: int | None = None) -> list[Sample]:
    files = sorted(sample_dir.glob("*.txt"))
    if limit is not None:
        files = files[:limit]
    samples: list[Sample] = []
    bad_files: list[str] = []
    for path in files:
        try:
            samples.append(read_sample_file(path))
        except Exception:
            bad_files.append(str(path))
    if bad_files:
        preview = "\n".join(bad_files[:5])
        raise RuntimeError(f"Failed to parse {len(bad_files)} sample files. First files:\n{preview}")
    return samples


def simple_tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower())
    return tokens or ["<empty>"]


def tokenized_text(text: str) -> str:
    return " ".join(simple_tokenize(text))


def make_split(
    samples: Iterable[Sample],
    seed: int = 100,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
) -> dict[str, str]:
    samples = list(samples)
    if samples and all(sample.split is not None for sample in samples):
        return {sample.data_id: sample.split or "train" for sample in samples}
    missing = [sample.data_id for sample in samples if sample.split is None]
    preview = ", ".join(missing[:10])
    raise ValueError(f"Samples must include dataset-provided split labels. Missing split for: {preview}")


def write_textgcn_files(samples: list[Sample], split: dict[str, str], out_dir: Path, dataset: str) -> None:
    text_dir = out_dir / "data" / "text_dataset"
    clean_dir = text_dir / "clean_corpus"
    text_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    target_path = text_dir / f"{dataset}.txt"
    corpus_path = clean_dir / f"{dataset}.txt"
    with target_path.open("w", encoding="utf-8", newline="\n") as target_file, corpus_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as corpus_file:
        for sample in samples:
            target_file.write(f"{sample.data_id}\t{split[sample.data_id]}\t{sample.label}\n")
            corpus_file.write(tokenized_text(sample.text) + "\n")
