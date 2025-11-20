import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from enet_config import TOPK
import torch.nn.functional as F


class PatchCoreEffB3(nn.Module):
    def __init__(self, device="cuda", coreset_size=0.2, use_multiscale=True):
        super().__init__()
        self.device = device
        self.coreset_size = coreset_size
        self.use_multiscale = use_multiscale

        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.memory_bank = None

    def extract_features(self, images):
        with torch.no_grad():
            x = images
            x = self.backbone.features[0](x)
            x = self.backbone.features[1](x)
            x2 = self.backbone.features[2](x)
            x3 = self.backbone.features[3](x2)
            x4 = self.backbone.features[4](x3)

        h_t, w_t = x3.shape[2], x3.shape[3]
        x2_up = F.interpolate(x2, size=(h_t, w_t), mode="bilinear", align_corners=False)
        x4_up = F.interpolate(x4, size=(h_t, w_t), mode="bilinear", align_corners=False)
        feat = torch.cat([x2_up, x3, x4_up], dim=1)

        B, C, H, W = feat.shape
        fmap_flat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return fmap_flat, H, W

    def build_memory_bank(self, feature_list):
        all_features = torch.cat(feature_list, dim=0)
        total = all_features.shape[0]
        keep = int(total * self.coreset_size)

        print(f"Total patches: {total}, CoreSet keep: {keep}")

        idxs = torch.randperm(total)[:keep]
        self.memory_bank = all_features[idxs].to(self.device)

    def anomaly_score_patchwise(self, patch_feats, chunk_size=512):
        patch_feats = patch_feats.to(self.device)
        memory = self.memory_bank.to(self.device)
    
        patch_feats = patch_feats / (patch_feats.norm(dim=1, keepdim=True) + 1e-6)
        memory = memory / (memory.norm(dim=1, keepdim=True) + 1e-6)
    
        min_dists = []
    
        for i in range(0, memory.shape[0], chunk_size):
            mb_chunk = memory[i:i + chunk_size]
            sim_chunk = torch.mm(patch_feats, mb_chunk.t())
            dist_chunk = 1.0 - sim_chunk
            min_dists.append(dist_chunk.min(dim=1)[0].cpu())
    
        return torch.stack(min_dists, dim=1).min(dim=1)[0]

    #takes top k anormal patches scores to send them to metrics
    def forward(self, img, topk=TOPK):
        feats, H, W = self.extract_features(img)
        feats_0 = feats[0]
        patch_scores = self.anomaly_score_patchwise(feats_0)

        if topk is None:
            image_score = patch_scores.max().item()
        else:
            k = min(topk, patch_scores.numel())
            topk_vals, _ = torch.topk(patch_scores, k)
            image_score = topk_vals.mean().item()

        return patch_scores, image_score, H, W