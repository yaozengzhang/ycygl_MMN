from __future__ import annotations

from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageChops, ImageEnhance
from torch import nn
from torchvision import models, transforms

from .prepare_dataset import Sample


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_image_path(image_dir: Path, sample: Sample) -> Path | None:
    raw = sample.image_id.strip()
    candidates: list[Path] = []
    if raw:
        raw_path = Path(raw)
        candidates.append(image_dir / raw_path.name)
        if raw_path.suffix:
            candidates.append(image_dir / raw_path.stem)
        stem = raw_path.stem if raw_path.suffix else raw_path.name
        for ext in IMAGE_EXTS:
            candidates.append(image_dir / f"{stem}{ext}")
    for ext in IMAGE_EXTS:
        candidates.append(image_dir / f"{sample.data_id}{ext}")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def make_ela_image(image: Image.Image, quality: int = 90, scale: float = 12.0) -> Image.Image:
    rgb = image.convert("RGB")
    buffer = BytesIO()
    rgb.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    diff = ImageChops.difference(rgb, compressed)
    extrema = diff.getextrema()
    max_diff = max(channel[1] for channel in extrema)
    factor = scale if max_diff == 0 else min(255.0 / max_diff, scale)
    return ImageEnhance.Brightness(diff).enhance(factor)


class ResNetLayer3Feature(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            self.transform = weights.transforms()
        else:
            model = models.resnet50(weights=None)
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        self.backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, image: Image.Image, device: torch.device) -> torch.Tensor:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = self.backbone(tensor).flatten(1).squeeze(0)
        return feature.detach().cpu().float()


def generate_image_features(
    samples: list[Sample],
    image_dir: Path,
    out_dir: Path,
    batch_size: int = 1,
    overwrite: bool = False,
    pretrained: bool = False,
    device: str | None = None,
) -> tuple[Path, Path]:
    del batch_size
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    original_dir = out_dir / "image_features"
    ela_dir = out_dir / "ela_features"
    original_dir.mkdir(parents=True, exist_ok=True)
    ela_dir.mkdir(parents=True, exist_ok=True)

    extractor = ResNetLayer3Feature(pretrained=pretrained).to(device_obj).eval()
    missing: list[str] = []
    for sample in samples:
        original_out = original_dir / f"{sample.data_id}.pt"
        ela_out = ela_dir / f"{sample.data_id}.pt"
        if not overwrite and original_out.exists() and ela_out.exists():
            continue

        image_path = find_image_path(image_dir, sample)
        if image_path is None:
            missing.append(sample.data_id)
            continue

        with Image.open(image_path) as image:
            original_feature = extractor(image, device_obj)
            ela_feature = extractor(make_ela_image(image), device_obj)
        torch.save(original_feature, original_out)
        torch.save(ela_feature, ela_out)

    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Missing {len(missing)} images in {image_dir}. First ids: {preview}")
    return original_dir, ela_dir
