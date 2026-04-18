# This script adapts feature extraction from:
# https://github.com/caiyu6666/MedIAnomaly

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "reconstruction"))

from dataloaders.dataload import (
    MedAD,
    BraTSAD,
    Camelyon16AD,
    ISIC2018,
    OCT2017,
    ColonAD,
    CpChildA,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "ssl", "two_stage"))
from model import ProjectionNet


# Utilities
class GrayToRGB:
    def __call__(self, img):
        return img.convert("RGB")

# Change as per your dataset location
def get_data_path(dataset):
    data_root = r"C:\Users\user\Pritam\MediAnomaly\MedIAnomaly-Data"

    mapping = {
        'rsna': "RSNA",
        'vin': "VinCXR",
        'brain': "BrainTumor",
        'lag': "LAG",
        'brats': "BraTS2021",
        'c16': "Camelyon16",
        'isic': "ISIC2018_Task3",
    }

    if dataset in mapping:
        return os.path.join(data_root, mapping[dataset])
    else:
        raise ValueError(f"Invalid dataset: {dataset}")


def get_dataset_class(dataset):
    if dataset in ['rsna', 'vin', 'brain', 'lag']:
        return MedAD
    elif dataset == 'brats':
        return BraTSAD
    elif dataset == 'c16':
        return Camelyon16AD
    elif dataset == 'isic':
        return ISIC2018
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def get_transform():
    return transforms.Compose([
        GrayToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])


def build_dataset(dataset_name, mode, transform):
    dataset_class = get_dataset_class(dataset_name)
    data_path = get_data_path(dataset_name)

    return dataset_class(
        main_path=data_path,
        img_size=224,
        transform=transform,
        mode=mode
    )


# FEATURE EXTRACTOR
def get_feature_extractor(backbone, device, dataset):
    print(f"\n[INFO] Using backbone: {backbone}")

    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])

    elif backbone == "anatpaste":
        model = ProjectionNet(
            pretrained=False,
            head_layers=[512] * 1 + [128],
            num_classes=2
        )

        ckpt_path = os.path.join(
            "ssl",
            "two_stage",
            "models",
            dataset,
            "AnatPaste",
            "model.tch"
        )

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model not found: {ckpt_path}")

        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint)

        # IMPORTANT: use ONLY encoder
        feature_extractor = model.resnet18

    else:
        raise ValueError("Invalid backbone")

    feature_extractor.to(device)
    feature_extractor.eval()

    return feature_extractor


# Extraction
@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    all_features, all_labels, all_names = [], [], []

    for batch in tqdm(dataloader, desc="Extracting"):
        imgs = batch["img"].to(device)
        labels = batch["label"]
        names = batch["name"]

        feats = model(imgs).flatten(1)

        all_features.append(feats.cpu().numpy())
        all_labels.append(np.array(labels))
        all_names.extend(list(names))

    return (
        np.concatenate(all_features),
        np.concatenate(all_labels),
        np.array(all_names)
    )


def save_arrays(save_dir, split, features, labels, names, save_csv=True):
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, f"{split}_features.npy"), features)
    np.save(os.path.join(save_dir, f"{split}_labels.npy"), labels)
    np.save(os.path.join(save_dir, f"{split}_names.npy"), names)

    print(f"[Saved] {split}_features.npy -> {features.shape}")

    if save_csv:
        df = pd.DataFrame(features)
        df.insert(0, "label", labels)
        df.insert(0, "name", names)
        df.to_csv(os.path.join(save_dir, f"{split}_features.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "anatpaste"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-root", default="features")
    parser.add_argument("--save-csv", action="store_true")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = get_transform()

    train_set = build_dataset(args.dataset, "train", transform)
    test_set = build_dataset(args.dataset, "test", transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = get_feature_extractor(args.backbone, device, args.dataset)

    print("\nExtracting train features...")
    train_features, train_labels, train_names = extract_embeddings(model, train_loader, device)

    print("\nExtracting test features...")
    test_features, test_labels, test_names = extract_embeddings(model, test_loader, device)

    save_dir = os.path.join(args.save_root, args.dataset, args.backbone)

    save_arrays(save_dir, "train", train_features, train_labels, train_names, args.save_csv)
    save_arrays(save_dir, "test", test_features, test_labels, test_names, args.save_csv)

    print(f"\n Features saved at: {save_dir}")


if __name__ == "__main__":
    main()
