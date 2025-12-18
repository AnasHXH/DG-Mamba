# models/dg_mamba.py
# DG-Mamba: Dual-Branch Gated Mamba for Nighttime Remote Sensing Image Enhancement
# 
# Component Naming Convention:
# - DGMamba: Dual-Branch Gated Mamba (full network)
# - DGMambaBranch: Single branch of the dual-branch architecture
# - GMB: Gated Mamba Block (outer block with dual residuals)
# - CAGM: Context-Aware Gated Mixer (inner module)
# - GSM: Gated Selective Modulation (content-adaptive gating)
# - LFP: Learned Feature Projection (linear projection with positional embedding)
# - MSCM: Multi-Scale Context Module (ASPP-based context aggregation)
# - ElementWiseGating: Split-and-multiply gating (x1 * x2)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ElementWiseGating(nn.Module):
    """
    Element-wise Gating mechanism.
    
    Splits input along channel dimension and computes element-wise product.
    Widely used in efficient architectures like NAFNet.
    
    Operation: y = x1 ⊙ x2, where [x1, x2] = split(x)
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * x2


class MSCM(nn.Module):
    """
    Multi-Scale Context Module (MSCM).
    
    Based on Atrous Spatial Pyramid Pooling (ASPP) for multi-scale 
    context aggregation. Uses dilated convolutions with rates {1, 6, 12, 18}
    plus global average pooling to capture context at multiple scales.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(MSCM, self).__init__()
        
        # Multi-scale dilated convolutions
        self.conv_d1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv_d6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv_d12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv_d18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        
        # Global context branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_global = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        # Fusion layer
        self.conv_fuse = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Multi-scale context features of shape (B, C, H, W)
        """
        size = x.shape[-2:]
        
        # Multi-scale feature extraction
        feat_d1 = self.conv_d1(x)
        feat_d6 = self.conv_d6(x)
        feat_d12 = self.conv_d12(x)
        feat_d18 = self.conv_d18(x)
        
        # Global context
        feat_global = self.conv_global(self.global_pool(x))
        feat_global = F.interpolate(feat_global, size=size, mode='bilinear', align_corners=False)
        
        # Concatenate and fuse
        out = torch.cat((feat_d1, feat_d6, feat_d12, feat_d18, feat_global), dim=1)
        out = self.conv_fuse(out)
        
        # Normalize and activate
        out = out.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        
        return self.act(out)


class LFP(nn.Module):
    """
    Learned Feature Projection (LFP).
    
    Applies learned linear projections with positional embedding to modulate
    features at each spatial location. Uses query-key-value formulation
    for feature transformation.
    
    Args:
        dim: Feature dimension (number of channels)
    """
    def __init__(self, dim):
        super(LFP, self).__init__()
        self.dim = dim
        
        # Learnable projections
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        
        # Scaling factor
        self.scale = dim ** -0.5
        
        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, 1, dim))

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, H, W, C)
            
        Returns:
            Projected features of shape (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # Add positional embedding
        x = x + self.pos_embedding
        x = x.view(B, H * W, C)
        
        # Compute projections
        q = self.proj_q(x)  # (B, N, C)
        k = self.proj_k(x)  # (B, N, C)
        v = self.proj_v(x)  # (B, N, C)
        
        # Compute position-wise attention scores
        q = q.view(B, H * W, 1, C)  # (B, N, 1, C)
        k = k.view(B, H * W, C, 1)  # (B, N, C, 1)
        attn = torch.matmul(q, k).squeeze(2) * self.scale  # (B, N, C)
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out = attn * v
        out = out.view(B, H, W, C)
        
        return out


class CAGM(nn.Module):
    """
    Context-Aware Gated Mixer (CAGM).
    
    Core mixing module that combines:
    - Path 1: GSM → SiLU-Gating → Element-wise Gating → LFP
    - Path 2: MSCM (Multi-Scale Context Module)
    
    The two paths are fused via residual addition to integrate
    content-adaptive gating with multi-scale spatial context.
    
    Args:
        d_model: Model dimension (number of channels)
        d_state: State dimension (unused, kept for compatibility)
        d_conv: Convolution kernel size for depth-wise conv
        expand: Channel expansion factor (default: 2.0)
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2.0, 
                 dt_rank=64, dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  # 2 * d_model
        self.dt_rank = dt_rank

        # Input projection: d_model → 4 * d_model
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        
        # Depth-wise convolution
        self.dwconv = nn.Conv2d(
            self.d_inner, self.d_inner, 
            kernel_size=d_conv, 
            padding=(d_conv - 1) // 2, 
            groups=self.d_inner
        )
        self.act = nn.SiLU()

        # GSM projections
        self.gsm_proj = nn.Linear(self.d_inner, self.d_inner * 2)
        self.delta_proj = nn.Linear(self.d_inner, self.d_inner)

        # Normalization
        self.out_norm = nn.LayerNorm(self.d_inner)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner // 2, d_model)

        # Sub-modules
        self.element_gate = ElementWiseGating()
        self.mscm = MSCM(d_model, d_model)
        self.lfp = LFP(d_model)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, H, W, C)
            
        Returns:
            Output tensor of shape (B, H, W, C)
        """
        B, H, W, C = x.shape

        # ============ Path 2: Multi-Scale Context ============
        x_mscm = self.mscm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # ============ Path 1: Gated Processing ============
        
        # Input projection and split
        x_proj = self.in_proj(x)
        x_main, z = x_proj.chunk(2, dim=-1)  # x_main: 2C, z: 2C (gating signal)
        
        # Depth-wise convolution
        x_main = x_main.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        x_main = self.dwconv(x_main)
        x_main = x_main.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
        x_main = self.act(x_main)
        
        # Gated Selective Modulation (GSM)
        y = self.gated_selective_modulation(x_main)
        
        # Layer normalization
        y = self.out_norm(y)
        
        # SiLU-Gating: y = y ⊙ SiLU(z) = y ⊙ z ⊙ σ(z)
        y = y * F.silu(z)

        # Element-wise Gating: split and multiply
        y = self.element_gate(y)

        # Learned Feature Projection
        y = self.lfp(y)

        # ============ Path Fusion ============
        y = y + x_mscm

        # Output projection
        out = self.out_proj(y)
        
        return out

    def gated_selective_modulation(self, x):
        """
        Gated Selective Modulation (GSM).
        
        Content-adaptive dual-path gating inspired by Mamba's selective mechanism.
        Replaces sequential state-space recurrence with parallel gating.
        
        Formula: y = x ⊙ σ(Δ) + x_content ⊙ tanh(x_content)
        
        Args:
            x: Input tensor of shape (B, H, W, C)
            
        Returns:
            Modulated features of shape (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # Flatten spatial dimensions
        x_flat = x.reshape(B, H * W, C)
        
        # Project to 2C dimensions
        x_dbl = self.gsm_proj(x_flat)
        x_dbl = x_dbl.view(B, H, W, -1)
        
        # Split into delta (gate) and content
        delta_raw, x_content = x_dbl.chunk(2, dim=-1)
        
        # Compute gating coefficients: Δ = softplus(W_Δ · delta_raw)
        delta = F.softplus(self.delta_proj(delta_raw))
        
        # Dual-path gating:
        # - Path A: Input gated by learned delta → x ⊙ σ(Δ)
        # - Path B: Self-gated content → x_content ⊙ tanh(x_content)
        y = x * torch.sigmoid(delta) + x_content * torch.tanh(x_content)
        
        return y


class GMB(nn.Module):
    """
    Gated Mamba Block (GMB).
    
    Building block of DG-Mamba with dual-residual structure:
    1. LayerNorm → CAGM → Residual Add
    2. LayerNorm → Conv Block → Residual Add
    
    Args:
        d_model: Model dimension (number of channels)
        d_state: State dimension for CAGM
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        
        # First normalization and CAGM
        self.norm1 = nn.LayerNorm(d_model)
        self.cagm = CAGM(d_model, d_state)
        
        # Second normalization and conv refinement
        self.norm2 = nn.LayerNorm(d_model)
        self.conv_refine = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        Forward pass with dual residual connections.
        
        Args:
            x: Input tensor of shape (B, H, W, C)
            
        Returns:
            Output tensor of shape (B, H, W, C)
        """
        # First residual: CAGM path
        residual = x
        x = self.norm1(x)
        x = residual + self.cagm(x)
        
        # Second residual: Conv refinement path
        residual = x
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        x = self.conv_refine(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
        x = residual + x
        
        return x


class DGMambaBranch(nn.Module):
    """
    Single branch of DG-Mamba.
    
    U-Net style encoder-decoder architecture with GMB blocks.
    Uses skip connections between encoder and decoder stages.
    
    Args:
        img_channel: Number of input image channels (default: 3)
        width: Base channel width (default: 32)
        middle_blk_num: Number of GMB blocks in bottleneck (default: 1)
        enc_blk_nums: Number of GMB blocks per encoder stage (default: [1,1,1,1])
        dec_blk_nums: Number of GMB blocks per decoder stage (default: [1,1,1,1])
        d_state: State dimension for CAGM (default: 64)
    """
    def __init__(self, img_channel=3, width=32, middle_blk_num=1, 
                 enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1], d_state=64):
        super().__init__()
        
        # Input projection
        self.intro = nn.Conv2d(
            img_channel, width, 
            kernel_size=3, padding=1, stride=1, 
            groups=1, bias=True
        )
        
        # Output projection
        self.ending = nn.Conv2d(
            width, img_channel, 
            kernel_size=3, padding=1, stride=1, 
            groups=1, bias=True
        )

        # Build encoder, decoder, and bottleneck
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        
        # Encoder stages
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[GMB(chan, d_state) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan = chan * 2

        # Bottleneck
        self.middle_blks = nn.Sequential(
            *[GMB(chan, d_state) for _ in range(middle_blk_num)]
        )

        # Decoder stages
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[GMB(chan, d_state) for _ in range(num)])
            )

        # Padding size for input alignment
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        """
        Forward pass.
        
        Args:
            inp: Input image of shape (B, C, H, W)
            
        Returns:
            Enhanced image of shape (B, C, H, W)
        """
        B, C, H, W = inp.shape
        
        # Pad input to be divisible by padder_size
        inp = self.check_image_size(inp)
        
        # Input projection
        x = self.intro(inp)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)

        # Encoder with skip connections
        enc_features = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            enc_features.append(x)
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
            x = down(x)
            x = x.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)

        # Bottleneck
        x = self.middle_blks(x)

        # Decoder with skip connections
        for decoder, up, enc_skip in zip(self.decoders, self.ups, enc_features[::-1]):
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
            x = up(x)
            x = x.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
            x = x + enc_skip  # Skip connection
            x = decoder(x)

        # Output projection with global residual
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        x = self.ending(x)
        x = x + inp  # Global residual connection

        # Crop to original size
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        """Pad image to be divisible by padder_size."""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class DGMamba(nn.Module):
    """
    DG-Mamba: Dual-Branch Gated Mamba.
    
    Dual-branch architecture for nighttime remote sensing image enhancement.
    Two parallel DGMambaBranch networks process the input independently,
    and their outputs are fused via a learned combination layer.
    
    Architecture:
        Input → [Branch A] → Output A ─┐
                                       ├→ Concat → Conv1x1 → Sigmoid → Output
        Input → [Branch B] → Output B ─┘
    
    Args:
        img_channel: Number of input image channels (default: 3)
        width: Base channel width for each branch (default: 32)
        middle_blk_num: Number of GMB blocks in bottleneck (default: 1)
        enc_blk_nums: Number of GMB blocks per encoder stage (default: [1,1,1,1])
        dec_blk_nums: Number of GMB blocks per decoder stage (default: [1,1,1,1])
        d_state: State dimension for CAGM (default: 64)
    """
    def __init__(
        self,
        img_channel=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 1],
        dec_blk_nums=[1, 1, 1, 1],
        d_state=64
    ):
        super().__init__()

        # Branch A
        self.branch_a = DGMambaBranch(
            img_channel=img_channel,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
            d_state=d_state
        )

        # Branch B
        self.branch_b = DGMambaBranch(
            img_channel=img_channel,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
            d_state=d_state
        )

        # Fusion layer: Concatenate + Conv1x1 + Sigmoid
        self.fusion = nn.Sequential(
            nn.Conv2d(img_channel * 2, img_channel, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input image of shape (B, C, H, W)
            
        Returns:
            Enhanced image of shape (B, C, H, W)
        """
        # Process through both branches
        out_a = self.branch_a(x)  # (B, C, H, W)
        out_b = self.branch_b(x)  # (B, C, H, W)

        # Fuse branch outputs
        fused = torch.cat([out_a, out_b], dim=1)  # (B, 2C, H, W)
        fused = self.fusion(fused)                 # (B, C, H, W)

        return fused
