import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from torchvision import transforms
from glob import glob
from PIL import Image
import numpy as np

train_data_folder = "your_good_path"
test_good_folder = "your_good_path"
test_defect_folder = "your_defect_path"
model_path = "your_model_path"

lrngrate = 1e-3
nof_epoch = 50

class ParquetDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        file_paths = glob(os.path.join(folder_path, "*.bmp"))
        self.file_paths = file_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        with Image.open(self.file_paths[index]) as img:
            if self.transform is not None:
                img_transformed = self.transform(img)
                return img_transformed
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Output (0-1)
        )
        
    def forward(self, x):
        z=self.encoder(x)
        out=self.decoder(z)
        return out
    
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    ])

def reconstruct_error(img_tensor):
    model.eval()
    with torch.no_grad():
        img_error = img_tensor.unsqueeze(0).to(device)
        out = model(img_error)
        error = torch.mean(torch.abs(img_error - out)).item()
        return error
    
def compute_auc(test_good_dl, test_defect_dl):
    errors = []
    labels = []

    for imgs in test_good_dl:
        for img in imgs:
            err = reconstruct_error(img)
            errors.append(err)
            labels.append(0)

    for imgs in test_defect_dl:
        for img in imgs:
            err = reconstruct_error(img)
            errors.append(err)
            labels.append(1)

    auc = roc_auc_score(labels, errors)
    f1, thr = compute_f1(errors, labels)
    return auc, errors, labels, f1

def compute_f1(errors, labels):
    good_errs = [e for e,l in zip(errors, labels) if l == 0]
    threshold = np.mean(good_errs) + np.std(good_errs)

    preds = (np.array(errors) > threshold).astype(int)
    return f1_score(labels, preds), threshold

train_ds = ParquetDataset(train_data_folder, transform=transform)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

test_good_ds = ParquetDataset(test_good_folder, transform=transform)
test_defect_ds = ParquetDataset(test_defect_folder, transform=transform)

test_good_dl = DataLoader(test_good_ds, batch_size=1)
test_defect_dl = DataLoader(test_defect_ds, batch_size=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoEncoder()
model.to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lrngrate)

best_f1 = 0
best_auc = 0

checkpoint = {
    "model_state_dict":model.state_dict(),
    "optimizer_state_dict":optimizer.state_dict(),
    "best_auc":best_auc}

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_auc = checkpoint["best_f1"]
else:
    model.to(device)

for epoch in range(nof_epoch):
    model.train()
    train_loss = 0
    
    for img in train_dl:
        img = img.to(device)
        
        output = model.forward(img)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_dl)
    
    auc, _, _, f1 = compute_auc(test_good_dl, test_defect_dl)

    if f1 > best_auc:
        best_auc = f1
        best_f1 = auc
        checkpoint = {
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "best_f1":best_auc}
        torch.save(checkpoint, model_path)
        
print(f"Best f1: {best_auc}, AUC: {best_f1}")

