import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from enet_dataset import TrainImageDataset
from enet_model import PatchCoreEffB3
import enet_config


def train_patchcore():
    device = enet_config.DEVICE

    print(f"\nDevice in use: {device}")
    model = PatchCoreEffB3(device=device, coreset_size=enet_config.CORESET)
    model.to(device)
    
    train_ds = TrainImageDataset(enet_config.TRAIN_DATA_FOLDER)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    feature_list = []

    print(f"Extracting features from {len(train_ds)} good images...")

    with torch.no_grad():
        for img in tqdm(train_loader):
            img = img.to(device)

            feats, H, W = model.extract_features(img)
            feats_0 = feats[0]  

            feature_list.append(feats_0.cpu())

    model.build_memory_bank(feature_list)

    save_path = enet_config.MODEL_PATH
    checkpoint = {
        "memory_bank": model.memory_bank.cpu(),
        "coreset_size": model.coreset_size
    }

    torch.save(checkpoint, save_path)
    print(f"\nTraining finished. Memory bank saved to:\n  {save_path}\n")

    return save_path