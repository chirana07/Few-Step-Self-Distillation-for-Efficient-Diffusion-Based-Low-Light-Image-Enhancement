import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / max(half_dim - 1, 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_emb = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        self.apply(init_weights)

    def forward(self, x, t_emb=None):
        h = self.conv1(self.act1(self.norm1(x)))
        if t_emb is not None:
            h = h + self.time_emb(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
        self.apply(init_weights)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)

        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads

        q = q.view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)

        attn = torch.matmul(q, k) * (head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)

        return x + self.proj(out)


class SpatialFeatureTransform(nn.Module):
    def __init__(self, feat_ch, cond_ch, hidden_ch=None):
        super().__init__()
        hidden_ch = hidden_ch or max(feat_ch, cond_ch)

        self.shared = nn.Sequential(
            nn.Conv2d(cond_ch, hidden_ch, 3, padding=1),
            nn.SiLU(),
        )
        self.gamma = nn.Conv2d(hidden_ch, feat_ch, 3, padding=1)
        self.beta = nn.Conv2d(hidden_ch, feat_ch, 3, padding=1)

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, cond):
        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)

        h = self.shared(cond)
        gamma = self.gamma(h)
        beta = self.beta(h)
        return x * (1.0 + gamma) + beta


class ResBlockSFT(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_channels):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = Swish()
        self.sft = SpatialFeatureTransform(out_channels, cond_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

        self.apply(init_weights)

    def forward(self, x, t_emb, cond):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.act2(self.norm2(h))
        h = self.sft(h, cond)
        h = self.conv2(h)
        return h + self.shortcut(x)


class IlluminationPrior(nn.Module):
    """
    Retinex-style illumination estimate.

    L(y_low) = GaussianBlur( max_c y_low[c] )

    Operates on images normalized to [-1, 1]. Returns a single-channel map in [-1, 1].
    No learnable parameters — kept as an nn.Module so it moves with .to(device) naturally.
    This is the "illumination prior" used as extra conditioning for LuminaDiff-R.
    """

    def __init__(self, kernel_size=15, sigma=3.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Precompute 2D Gaussian kernel
        half = kernel_size // 2
        coords = torch.arange(kernel_size, dtype=torch.float32) - half
        g1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g1d = g1d / g1d.sum()
        g2d = g1d[:, None] @ g1d[None, :]
        kernel = g2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        # x: (B, 3, H, W) in [-1, 1] -> illum: (B, 1, H, W) in [-1, 1]
        x01 = (x + 1.0) / 2.0
        illum01 = x01.max(dim=1, keepdim=True).values
        pad = self.kernel_size // 2
        illum01 = F.conv2d(illum01, self.kernel, padding=pad)
        return illum01 * 2.0 - 1.0


class ConditionEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, ch_mult=(1, 2, 4, 8)):
        super().__init__()
        self.in_proj = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.out_channels = []

        curr_ch = base_ch
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(curr_ch, out_ch, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.SiLU(),
                )
            )
            self.out_channels.append(out_ch)
            curr_ch = out_ch

            if i != len(ch_mult) - 1:
                self.downs.append(Downsample(curr_ch))

        self.apply(init_weights)

    def forward(self, low_light):
        feats = []
        h = self.in_proj(low_light)

        for i, block in enumerate(self.blocks):
            h = block(h)
            feats.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)

        return feats


# ----- losses -----

class TVLoss(nn.Module):
    def forward(self, x):
        dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return dh + dw


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer("window", self.create_window(window_size, channel))

    def gaussian(self, window_size, sigma):
        gauss = torch.tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
             for x in range(window_size)],
            dtype=torch.float32
        )
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1d = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2d = (_1d @ _1d.t()).unsqueeze(0).unsqueeze(0)
        return _2d.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2):
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0

        window = self.window.to(img1.device, dtype=img1.dtype)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
        )
        return 1.0 - ssim_map.mean()


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_ids=(4, 9, 18), layer_weights=(1.0, 1.0, 1.0)):
        super().__init__()

        vgg = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.DEFAULT
        ).features.eval()

        self.blocks = nn.ModuleList()
        prev = 0
        for lid in layer_ids:
            self.blocks.append(nn.Sequential(*[vgg[i] for i in range(prev, lid)]))
            prev = lid

        self.layer_weights = layer_weights

        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize(self, x):
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)

        loss = 0.0
        for w, block in zip(self.layer_weights, self.blocks):
            x = block(x)
            y = block(y)
            loss = loss + w * F.l1_loss(x, y)
        return loss


class ColorConsistencyLoss(nn.Module):
    def forward(self, pred, target):
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0

        pred_mean = pred.mean(dim=(2, 3))
        target_mean = target.mean(dim=(2, 3))

        pred_std = pred.std(dim=(2, 3))
        target_std = target.std(dim=(2, 3))

        return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("kx", kernel_x)
        self.register_buffer("ky", kernel_y)

    def sobel(self, x):
        grads = []
        for c in range(x.shape[1]):
            xc = x[:, c:c+1]
            gx = F.conv2d(xc, self.kx, padding=1)
            gy = F.conv2d(xc, self.ky, padding=1)
            grads.append(torch.sqrt(gx * gx + gy * gy + 1e-6))
        return torch.cat(grads, dim=1)

    def forward(self, pred, target):
        gp = self.sobel(pred)
        gt = self.sobel(target)
        return F.l1_loss(gp, gt)


class CompositeEnhancementLoss(nn.Module):
    def __init__(
        self,
        w_char=1.0,
        w_ssim=0.4,
        w_perc=0.05,
        w_color=0.05,
        w_grad=0.2,
        w_tv=0.0,
    ):
        super().__init__()
        self.char = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.perc = VGGPerceptualLoss()
        self.color = ColorConsistencyLoss()
        self.grad = GradientLoss()
        self.tv = TVLoss()

        self.w_char = w_char
        self.w_ssim = w_ssim
        self.w_perc = w_perc
        self.w_color = w_color
        self.w_grad = w_grad
        self.w_tv = w_tv

    def forward(self, pred_img, target_img, pred_residual=None):
        loss_char = self.char(pred_img, target_img)
        loss_ssim = self.ssim(pred_img, target_img)
        loss_perc = self.perc(pred_img, target_img)
        loss_color = self.color(pred_img, target_img)
        loss_grad = self.grad(pred_img, target_img)

        if pred_residual is None:
            loss_tv = pred_img.new_tensor(0.0)
        else:
            loss_tv = self.tv(pred_residual)

        total = (
            self.w_char * loss_char +
            self.w_ssim * loss_ssim +
            self.w_perc * loss_perc +
            self.w_color * loss_color +
            self.w_grad * loss_grad +
            self.w_tv * loss_tv
        )

        logs = {
            "char": loss_char.detach(),
            "ssim": loss_ssim.detach(),
            "perc": loss_perc.detach(),
            "color": loss_color.detach(),
            "grad": loss_grad.detach(),
            "tv": loss_tv.detach(),
            "total": total.detach(),
        }
        return total, logs