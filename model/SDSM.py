import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from model.Mamba_scan import MSBlock, SimpleSpecMambaBlock


def random_seed_setting(seed: int = 42):
    import random
    import os
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def img2seq(x):
    [b, c, h, w] = x.shape
    x = x.reshape((b, c, h * w))
    return x


def seq2img(x):
    [b, c, d] = x.shape
    h = w = int(d ** 0.5)
    return x.reshape((b, c, h, w))

class CNN_Encoder(nn.Module):
    def __init__(self, l1, l2):
        super(CNN_Encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(l1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(l2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        x1 = self.conv1_1(x1)
        x2 = self.conv2_1(x2)
        return x1, x2


class FuseMamba(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, patch_size=None, d_state=16,
                 expand=0.5):
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size
        self.MBh = MSBlock(hidden_dim=dim, d_state=d_state, expand=expand)
        self.simple_spectral = SimpleSpecMambaBlock(hidden_dim=dim, d_state=d_state, expand=expand, dropout=dropout)
        self.MBx = MSBlock(hidden_dim=dim, d_state=d_state, expand=expand)
        self.linear1 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.relu = nn.ReLU()

    def forward(self, x, is_hsi=True, mask=None):
        b, n, d = x.shape
        patch_size = self.patch_size if self.patch_size else int(n ** 0.5)

        x_proj = self.linear1(x)

        if is_hsi:
            x_spatial_input = x_proj.permute(0, 2, 1).reshape(b, d, patch_size, patch_size)
            x_spatial = self.MBh(x_spatial_input)
            x_spectral = self.simple_spectral(x_spatial)
            x_mamba = x_spectral.reshape(b, d, -1).permute(0, 2, 1)

        else:
            x_reshaped = x_proj.permute(0, 2, 1).reshape(b, d, patch_size, patch_size)
            x_spatial = self.MBx(x_reshaped)
            x_mamba = x_spatial.reshape(b, d, -1).permute(0, 2, 1)
        x = x + x_mamba
        x = self.relu(self.norm(x))

        return x

class SpatialBranch(nn.Module):
    def __init__(self, l1, l2, patch_size, encoder_embed_dim, en_depth, en_heads, dim_head, mlp_dim, dropout,
                 d_state=16, expand=1):
        super(SpatialBranch, self).__init__()
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.standard_size = 7
        self.cnn_encoder = CNN_Encoder(l1, l2)
        self.adaptive_pool1 = nn.AdaptiveAvgPool2d((self.standard_size, self.standard_size))
        self.adaptive_pool2 = nn.AdaptiveAvgPool2d((self.standard_size, self.standard_size))
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.standard_size ** 2 + 1, encoder_embed_dim))
        self.encoder_embedding1 = nn.Linear(self.standard_size ** 2, self.standard_size ** 2)
        self.encoder_embedding2 = nn.Linear(self.standard_size ** 2, self.standard_size ** 2)
        self.dropout = nn.Dropout(dropout)
        self.FuseMamba1 = FuseMamba(encoder_embed_dim, en_depth, en_heads, dim_head, mlp_dim, dropout,
                                    self.standard_size ** 2, self.standard_size, d_state, expand)
        self.FuseMamba2 = FuseMamba(encoder_embed_dim, en_depth, en_heads, dim_head, mlp_dim, dropout,
                                    self.standard_size ** 2, self.standard_size, d_state, expand)

        self.fusion_layer = nn.Linear(encoder_embed_dim * 2, encoder_embed_dim)

    def forward(self, x1, x2):
        x_fuse1, x_fuse2 = self.cnn_encoder(x1, x2)
        x_fuse1 = self.adaptive_pool1(x_fuse1)  # [batch, 64, standard_size, standard_size]
        x_fuse2 = self.adaptive_pool2(x_fuse2)  # [batch, 64, standard_size, standard_size]

        x_flat1 = x_fuse1.flatten(2)  # [batch, 64, standard_size^2]
        x_flat2 = x_fuse2.flatten(2)  # [batch, 64, standard_size^2]

        x1 = self.encoder_embedding1(x_flat1)  # [batch, 64, standard_size^2]
        x1 = torch.einsum('nld->ndl', x1)  # [batch, standard_size^2, 64]

        x2 = self.encoder_embedding2(x_flat2)  # [batch, 64, standard_size^2]
        x2 = torch.einsum('nld->ndl', x2)  # [batch, standard_size^2, 64]

        b1, n1, _ = x1.shape
        b2, n2, _ = x2.shape

        x1 += self.encoder_pos_embed[:, :n1]
        x1 = self.dropout(x1)

        x2 += self.encoder_pos_embed[:, :n2]
        x2 = self.dropout(x2)

        x1 = self.FuseMamba1(x1, is_hsi=True, mask=None)
        x2 = self.FuseMamba2(x2, is_hsi=False, mask=None)

        x_concat = torch.cat([x1, x2], dim=-1)
        x_fused = self.fusion_layer(x_concat)
        return x_fused

class GlobalFourierBranch(nn.Module):
    def __init__(self, l1, l2, patch_size, encoder_embed_dim):
        super(GlobalFourierBranch, self).__init__()
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.input_conv1 = nn.Conv2d(l1, 64, 3, 1, 1)
        self.input_conv2 = nn.Conv2d(l2, 64, 3, 1, 1)
        self.pre1 = nn.Conv2d(64, 64, 1, 1, 0)
        self.pre2 = nn.Conv2d(64, 64, 1, 1, 0)
        self.amp_fuse = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(64, 64, 1, 1, 0)
        )
        self.pha_fuse = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(64, 64, 1, 1, 0)
        )
        self.post = nn.Conv2d(64, 64, 1, 1, 0)
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, encoder_embed_dim)
        )

    def forward(self, x1, x2):
        x1 = self.input_conv1(x1)
        x2 = self.input_conv2(x2)

        _, _, H, W = x1.shape

        x1_pre = self.pre1(x1)
        x2_pre = self.pre2(x2)

        x1F = torch.fft.rfft2(x1_pre + 1e-8, norm='backward')
        x2F = torch.fft.rfft2(x2_pre + 1e-8, norm='backward')

        x1F_amp = torch.abs(x1F)
        x1F_pha = torch.angle(x1F)
        x2F_amp = torch.abs(x2F)
        x2F_pha = torch.angle(x2F)

        amp_fuse = self.amp_fuse(torch.cat([x1F_amp, x2F_amp], 1))
        pha_fuse = self.pha_fuse(torch.cat([x1F_pha, x2F_pha], 1))

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
        out = torch.complex(real, imag) + 1e-8

        freq_feat = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        freq_feat = self.post(freq_feat)

        global_feat = self.feature_projection(freq_feat)  # [batch, embed_dim]
        return global_feat

class LocalFourierBranch(nn.Module):
    def __init__(self, l1, l2, patch_size, encoder_embed_dim):
        super(LocalFourierBranch, self).__init__()

        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.input_conv1 = nn.Conv2d(l1, 64, 3, 1, 1)
        self.input_conv2 = nn.Conv2d(l2, 64, 3, 1, 1)
        self.pre1 = nn.Conv2d(64, 64, 1, 1, 0)
        self.pre2 = nn.Conv2d(64, 64, 1, 1, 0)
        self.amp_fuse_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 64, 1, 1, 0),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(64, 64, 1, 1, 0)
            ) for _ in range(4)
        ])
        self.pha_fuse_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 64, 1, 1, 0),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(64, 64, 1, 1, 0)
            ) for _ in range(4)
        ])
        self.post = nn.Conv2d(64, 64, 1, 1, 0)
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, encoder_embed_dim)
        )

    def forward(self, x1, x2):
        x1 = self.input_conv1(x1)
        x2 = self.input_conv2(x2)
        _, _, H, W = x1.shape
        x1_pre = self.pre1(x1)
        x2_pre = self.pre2(x2)

        s1 = H // 2
        s2 = W // 2
        overlap = min(s1, s2) // 2

        x1_regions = [
            x1_pre[:, :, :s1 + overlap, :s1 + overlap],
            x1_pre[:, :, :s1 + overlap, s1 - overlap:],
            x1_pre[:, :, s1 - overlap:, :s1 + overlap],
            x1_pre[:, :, s1 - overlap:, s1 - overlap:]
        ]

        x2_regions = [
            x2_pre[:, :, :s1 + overlap, :s1 + overlap],
            x2_pre[:, :, :s1 + overlap, s1 - overlap:],
            x2_pre[:, :, s1 - overlap:, :s1 + overlap],
            x2_pre[:, :, s1 - overlap:, s1 - overlap:]
        ]

        out_regions = []

        for i in range(4):
            h, w = x1_regions[i].shape[2:]

            x1F = torch.fft.rfft2(x1_regions[i] + 1e-8, norm='backward')
            x2F = torch.fft.rfft2(x2_regions[i] + 1e-8, norm='backward')

            x1F_amp = torch.abs(x1F)
            x1F_pha = torch.angle(x1F)
            x2F_amp = torch.abs(x2F)
            x2F_pha = torch.angle(x2F)

            amp_fuse = self.amp_fuse_blocks[i](torch.cat([x1F_amp, x2F_amp], 1))
            pha_fuse = self.pha_fuse_blocks[i](torch.cat([x1F_pha, x2F_pha], 1))

            real = amp_fuse * torch.cos(pha_fuse) + 1e-8
            imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
            out = torch.complex(real, imag) + 1e-8

            out = torch.abs(torch.fft.irfft2(out, s=(h, w), norm='backward'))
            out_regions.append(out)

        output = torch.zeros((x1.shape[0], 64, H, W), device=x1.device)
        weight_map = torch.zeros((1, 1, H, W), device=x1.device)

        positions = [
            (0, 0),
            (0, s1 - overlap),
            (s1 - overlap, 0),
            (s1 - overlap, s1 - overlap)
        ]

        for i, (out_region, (h_start, w_start)) in enumerate(zip(out_regions, positions)):
            h_end = h_start + out_region.shape[2]
            w_end = w_start + out_region.shape[3]

            h_end = min(h_end, H)
            w_end = min(w_end, W)

            output[:, :, h_start:h_end, w_start:w_end] += out_region[:, :, :h_end - h_start, :w_end - w_start]
            weight_map[:, :, h_start:h_end, w_start:w_end] += 1

        freq_feat = output / (weight_map + 1e-8)
        freq_feat = self.post(freq_feat)

        local_feat = self.feature_projection(freq_feat)  # [batch, embed_dim]

        return local_feat


class CNN_Classifier(nn.Module):
    def __init__(self, Classes, patch_size):
        super(CNN_Classifier, self).__init__()
        input_channels = patch_size ** 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.FC = nn.Sequential(
            Linear(32, Classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.FC(x)
        return F.softmax(x, dim=1)


class BCFM(nn.Module):
    """
    Bidirectional Cross-Fusion Module
    严格按照论文公式(24)-(29)实现
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.WQ1 = nn.Linear(embed_dim, embed_dim)
        self.WK1 = nn.Linear(embed_dim, embed_dim)
        self.WV1 = nn.Linear(embed_dim, embed_dim)

        self.WQ2 = nn.Linear(embed_dim, embed_dim)
        self.WK2 = nn.Linear(embed_dim, embed_dim)
        self.WV2 = nn.Linear(embed_dim, embed_dim)

        self.alpha1 = nn.Parameter(torch.tensor(0.1))
        self.alpha2 = nn.Parameter(torch.tensor(0.05))

    def forward(self, H, Ff):
        Q1 = self.WQ1(H)
        K1 = self.WK1(H)
        V1 = self.WV1(H)

        Q2 = self.WQ2(Ff)
        K2 = self.WK2(Ff)
        V2 = self.WV2(Ff)

        A12 = F.softmax(Q1 @ K2.transpose(-2, -1) / (Q1.size(-1) ** 0.5), dim=-1)
        A21 = F.softmax(Q2 @ K1.transpose(-2, -1) / (Q2.size(-1) ** 0.5), dim=-1)
        Fh = H + self.alpha1 * (A21 @ V1)
        Fof = Ff + self.alpha2 * (A12 @ V2)
        return torch.cat([Fh, Fof], dim=-1)


class SDSM(nn.Module):
    def __init__(self, l1, l2, patch_size, num_patches, num_classes, encoder_embed_dim,
                 en_depth, en_heads, mlp_dim, dim_head=16, dropout=0., emb_dropout=0.):
        super().__init__()
        d_state = 16
        expand = 1
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.spatial_branch = SpatialBranch(l1, l2, patch_size, encoder_embed_dim, en_depth, en_heads,
                                            dim_head, mlp_dim, dropout, d_state, expand)
        self.global_fourier_branch = GlobalFourierBranch(l1, l2, patch_size, encoder_embed_dim)
        self.local_fourier_branch = LocalFourierBranch(l1, l2, patch_size, encoder_embed_dim)
        self.freq_projection = nn.Sequential(
            nn.Linear(encoder_embed_dim * 2, encoder_embed_dim),
            nn.LayerNorm(encoder_embed_dim)
        )
        self.bcfm = BCFM(encoder_embed_dim)
        self.bcfm_scale = nn.Parameter(torch.tensor(0.01))
        self.fusion_layer = nn.Sequential(
            nn.Linear(encoder_embed_dim * 2, encoder_embed_dim),
            nn.LayerNorm(encoder_embed_dim)
        )
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(encoder_embed_dim),
            nn.Linear(encoder_embed_dim, num_classes)
        )

        random_seed_setting(6)

    def forward(self, img1, img2):
        spatial_feat = self.spatial_branch(img1, img2)
        global_freq_feat = self.global_fourier_branch(img1, img2)
        local_freq_feat = self.local_fourier_branch(img1, img2)
        batch_size, seq_len, embed_dim = spatial_feat.shape
        freq_combined = torch.cat([global_freq_feat, local_freq_feat], dim=-1)
        freq_feat = self.freq_projection(freq_combined)
        freq_feat_seq = freq_feat.unsqueeze(1).expand(-1, seq_len, -1)
        bcfm_output = self.bcfm(spatial_feat, freq_feat_seq)
        scale = (torch.tanh(self.bcfm_scale) + 1) / 2
        bcfm_output_scaled = scale * bcfm_output
        fused_feat = self.fusion_layer(bcfm_output_scaled)
        fused_feat = fused_feat.mean(dim=1)
        x_cls = self.to_latent(fused_feat)
        x_cls = self.mlp_head(x_cls)
        return x_cls

