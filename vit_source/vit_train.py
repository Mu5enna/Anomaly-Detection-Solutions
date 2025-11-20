import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vit_dataset import TrainImageDataset
from vit_model import PatchCoreViT
import vit_config


def train_patchcore():
    device = vit_config.DEVICE

    print(f"\nDevice in use: {device}")
    model = PatchCoreViT(device=device, coreset_size=vit_config.CORESET, img_size=vit_config.IMGSIZE)
    model.to(device)
    
    train_ds = TrainImageDataset(vit_config.TRAIN_DATA_FOLDER)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    feature_list = []

    print(f"Extracting features from {len(train_ds)} good images...")

    with torch.no_grad():
        for img in tqdm(train_loader):
            img = img.to(device)

            feats, H, W = model.extract_features(img)
            feature_list.append(feats[0].cpu())

    model.build_memory_bank(feature_list)

    save_path = vit_config.MODEL_PATH
    checkpoint = {
        "memory_bank": model.memory_bank.cpu(),
        "coreset_size": model.coreset_size
    }

    torch.save(checkpoint, save_path)
    print(f"\nTraining finished. Memory bank saved to:\n  {save_path}\n")

    return save_path