from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class ManualMVTecDataset(Dataset):
    """Manual train/validation/test dataloader for MVTec AD2."""

    def __init__(self, class_root, split="train", resize=256):
        self.class_root = Path(class_root)
        self.split = split
        self.resize = resize
        self.split_dir = self.class_root / split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
            ]
        )

        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"No images found in split: {self.split_dir}")

    def _collect_samples(self):
        samples = []
        for defect_dir in sorted(self.split_dir.iterdir()):
            if not defect_dir.is_dir() or defect_dir.name == "ground_truth":
                continue

            defect_type = defect_dir.name
            mask_dir = None
            if defect_type != "good":
                candidate = self.split_dir / "ground_truth" / defect_type
                if candidate.exists():
                    mask_dir = candidate

            for image_path in sorted(defect_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue

                mask_path = None
                if mask_dir is not None:
                    stem = image_path.stem
                    candidate_mask = mask_dir / f"{stem}_mask.png"
                    if candidate_mask.exists():
                        mask_path = candidate_mask

                samples.append(
                    {
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "defect_type": defect_type,
                        "is_anomaly": defect_type != "good",
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.image_transform(image)

        if sample["mask_path"] is not None:
            mask = Image.open(sample["mask_path"]).convert("L")
            mask = self.mask_transform(mask)
        else:
            mask = torch.zeros((1, self.resize, self.resize), dtype=torch.float32)

        return {
            "image": image,
            "image_path": str(sample["image_path"]),
            "mask": mask,
            "mask_path": str(sample["mask_path"]) if sample["mask_path"] is not None else "",
            "is_anomaly": sample["is_anomaly"],
            "defect_type": sample["defect_type"],
            "split": self.split,
        }


def create_manual_mvtec_dataloaders(
    class_root,
    resize=256,
    batch_size=2,
    test_split="test_public",
    num_workers=0,
):
    """Create train, validation, and test dataloaders for one AD2 class."""

    class_root = Path(class_root)

    dataloaders = {
        "train": DataLoader(
            ManualMVTecDataset(class_root, split="train", resize=resize),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "validation": DataLoader(
            ManualMVTecDataset(class_root, split="validation", resize=resize),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            ManualMVTecDataset(class_root, split=test_split, resize=resize),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return dataloaders


def describe_batch(title, batch):
    print(f"\n{title}")
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch image paths: {list(batch['image_path'])}")
    print(f"Batch anomaly labels: {batch['is_anomaly']}")
    print(f"Batch defect types: {list(batch['defect_type'])}")
    print(f"Batch mask shape: {batch['mask'].shape}")
    print(f"Batch mask paths: {list(batch['mask_path'])}")


def main():
    class_root = Path(
        r"C:\Users\fateh\Desktop\Comet-project\CoMet\mvtec_ad_2\can"
    )

    dataloaders = create_manual_mvtec_dataloaders(
        class_root=class_root,
        resize=256,
        batch_size=2,
        test_split="test_public",
        num_workers=0,
    )

    print(f"Dataset root: {class_root}")
    print(f"Train samples: {len(dataloaders['train'].dataset)}")
    print(f"Validation samples: {len(dataloaders['validation'].dataset)}")
    print(f"Test samples: {len(dataloaders['test'].dataset)}")

    describe_batch("Train batch", next(iter(dataloaders["train"])))
    describe_batch("Validation batch", next(iter(dataloaders["validation"])))
    describe_batch("Test batch", next(iter(dataloaders["test"])))


if __name__ == "__main__":
    main()
