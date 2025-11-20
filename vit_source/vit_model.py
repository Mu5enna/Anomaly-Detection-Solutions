import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
import vit_config


class PatchCoreViT(nn.Module):
    def __init__(self, device=vit_config.DEVICE, coreset_size=vit_config.CORESET, img_size=vit_config.IMGSIZE):
        super().__init__()
        self.device = device
        self.coreset_size = coreset_size
        self.img_size = img_size

        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad = False

        self.memory_bank = None

    def extract_features(self, images):
        B, C, H, W = images.shape

        if H != self.img_size or W != self.img_size:
            images = F.interpolate(images, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        with torch.no_grad():
            x = self.vit._process_input(images)

            B, N, D = x.shape
            pos_embed = self.vit.encoder.pos_embedding

            pe_patches = pos_embed[:, 1:, :]
            if pe_patches.shape[1] != N:
                pe_patches = F.interpolate(
                    pe_patches.permute(0, 2, 1),
                    size=N,
                    mode="linear",
                    align_corners=False
                ).permute(0, 2, 1)

            x = x + pe_patches
            x = self.vit.encoder.dropout(x)
            x = self.vit.encoder.layers(x)
            x = self.vit.encoder.ln(x)

        patch_tokens = x
        N = patch_tokens.shape[1]

        H_p = W_p = int(math.sqrt(N))
        if H_p * W_p != N:
            H_p = H_p
            W_p = N // H_p

        fmap_flat = patch_tokens
        return fmap_flat, H_p, W_p

    def build_memory_bank(self, feature_list):
        all_features = torch.cat(feature_list, dim=0)
        total = all_features.shape[0]
        keep = int(total * self.coreset_size)
        print(f"Total patches: {total}, CoreSet keep: {keep}")

        idxs = torch.randperm(total)[:keep]
        self.memory_bank = all_features[idxs].to(self.device)

    def anomaly_score_patchwise(self, patch_feats, chunk_size=2048):
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

        min_dists = torch.stack(min_dists, dim=1)
        patch_scores = min_dists.min(dim=1)[0]
        return patch_scores

    def forward(self, img, topk=vit_config.TOPK):
        fmap_flat, H_p, W_p = self.extract_features(img)
        feats_0 = fmap_flat[0]
        patch_scores = self.anomaly_score_patchwise(feats_0)

        if topk is not None:
            k = min(topk, patch_scores.numel())
            topk_vals, _ = torch.topk(patch_scores, k)
            image_score = topk_vals.mean().item()
        else:
            image_score = patch_scores.max().item()

        return patch_scores, image_score, H_p, W_p