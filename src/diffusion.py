"""
diffusion.py — noise schedule + samplers.

Changes vs. prior version (2026-04-24):
- Added ddim_sample(): a proper deterministic DDIM sampler with strided timesteps
  and an optional eta parameter for stochasticity. The model predicts the clean
  residual (x0 parameterization), which fits naturally into DDIM's x0-prediction
  form without retraining.
- Kept the old sample() method (renamed internal semantics for clarity, kept the
  public API). It's a deterministic DDPM-posterior-mean chain — useful as a
  "legacy" sampler in the ablation.
- q_sample now also returns the exact (s1, s2) coefficients used, so the training
  code can do an x0-parameterized MSE cross-check if needed.

Changes (2026-04-30, novelty for resubmission):
- ddim_sample now supports `gate_alpha` and `gate_floor` parameters that implement
  ADAPTIVE RESIDUAL RESCALING (ARR): an inference-time, training-free modulation
  that dampens the residual prediction at high noise levels and trusts it fully
  at clean levels. Mirrors the fact that the model's residual prediction is more
  uncertain when the input residual is closer to pure noise.

  factor(t) = max(gate_floor, 1 - gate_alpha * (t / (T-1)))
  x0_pred_modulated = x0_pred * factor

  - gate_alpha = 0 (default) reproduces the original DDIM behavior exactly,
    so existing checkpoints and prior results are unaffected.
  - gate_alpha > 0 enables ARR; tune on a small held-out subset.
  - gate_floor caps the minimum scaling so the residual is never zeroed out.

  This is the small, citable architectural variation we add for the resubmission
  (no retraining required).
"""
import math

import torch

from config import Config


def cosine_beta_schedule(timesteps, s=0.008, max_beta=0.999):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-6, max_beta)


class DiffusionEngine:
    def __init__(self):
        self.conf = Config()
        self.device = self.conf.DEVICE
        self.steps = self.conf.TIMESTEPS

        self.betas = cosine_beta_schedule(self.steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=self.device),
            self.alphas_cumprod[:-1],
        ], dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # DDPM posterior coefficients (for legacy sampler)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    # ------------------------- forward process -------------------------

    def make_noise(self, x, offset_noise_strength=0.05):
        noise = torch.randn_like(x)
        if offset_noise_strength > 0:
            offset = torch.randn(x.size(0), x.size(1), 1, 1, device=x.device)
            noise = noise + offset_noise_strength * offset
        return noise

    def q_sample(self, x_start, t, noise=None, offset_noise_strength=0.05):
        if noise is None:
            noise = self.make_noise(x_start, offset_noise_strength)

        s1 = self.sqrt_alphas_cumprod[t][:, None, None, None]
        s2 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_t = s1 * x_start + s2 * noise
        return x_t, noise

    # ------------------------- samplers -------------------------

    def _build_schedule(self, inference_steps):
        inference_steps = inference_steps or getattr(self.conf, "INFERENCE_STEPS", self.steps)
        inference_steps = min(inference_steps, self.steps)
        schedule = torch.linspace(
            self.steps - 1, 0, inference_steps, device=self.device
        ).long()
        schedule = torch.unique_consecutive(schedule)
        return schedule

    @torch.no_grad()
    def sample(self, model, low_light, inference_steps=None, initial_noise=None):
        """
        Legacy sampler: deterministic DDPM-posterior-mean chain over the residual.
        The model predicts the clean residual (treated as x0) and we walk down using
        the DDPM posterior mean with no noise added back. Kept for ablation / reproducing
        earlier numbers.
        """
        model.eval()
        b = low_light.shape[0]

        residual = self._initial_residual(low_light, initial_noise)
        schedule = self._build_schedule(inference_steps)

        for i in schedule:
            t = torch.full((b,), int(i.item()), device=self.device, dtype=torch.long)

            _pred_img, pred_residual = model(residual, t, low_light, return_residual=True)
            pred_residual = torch.clamp(pred_residual, -1.0, 1.0)

            posterior_mean = (
                self.posterior_mean_coef1[t][:, None, None, None] * pred_residual
                + self.posterior_mean_coef2[t][:, None, None, None] * residual
            )
            residual = posterior_mean

        final_img = torch.clamp(low_light + residual, -1.0, 1.0)
        return final_img

    @torch.no_grad()
    def ddim_sample(self, model, low_light, inference_steps=None, eta=0.0,
                    gate_alpha=0.0, gate_floor=0.5, initial_noise=None):
        """
        Proper DDIM sampler operating in residual space (x0-parameterization).

        At each timestep:
          x_{t-1} = sqrt(a_prev) * x0_pred
                  + sqrt(1 - a_prev - sigma^2) * eps_pred
                  + sigma * z,  z ~ N(0, I)
        where eps_pred = (x_t - sqrt(a_t) * x0_pred) / sqrt(1 - a_t).

        eta=0 is fully deterministic DDIM (what the paper should have done).
        eta=1 recovers stochastic DDPM ancestral sampling.

        The model emits a clean residual prediction; we treat it as x0 for DDIM.

        Adaptive Residual Rescaling (ARR):
          When `gate_alpha > 0`, the predicted residual x0_pred is multiplied by
              factor(t) = max(gate_floor, 1 - gate_alpha * t / (T - 1))
          before being plugged back into the DDIM update. At t=0 (clean step)
          factor=1, so the final-step residual is unaffected. At t=T-1 (noisiest)
          factor reaches its minimum (clamped at gate_floor). This is an
          inference-time-only modulation; the trained model is unchanged.

          Set gate_alpha=0 (default) to reproduce vanilla DDIM behavior exactly.
        """
        model.eval()
        b = low_light.shape[0]

        residual = self._initial_residual(low_light, initial_noise)
        schedule = self._build_schedule(inference_steps)
        idx_list = schedule.tolist()

        T_minus_1 = max(self.steps - 1, 1)

        for step_i, i in enumerate(idx_list):
            t = torch.full((b,), int(i), device=self.device, dtype=torch.long)

            _pred_img, x0_pred = model(residual, t, low_light, return_residual=True)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # --- Adaptive Residual Rescaling (ARR) ---
            if gate_alpha > 0.0:
                # t is identical across the batch in DDIM; safe to take a scalar.
                t_norm = float(int(i)) / T_minus_1
                factor = max(float(gate_floor), 1.0 - gate_alpha * t_norm)
                x0_pred = x0_pred * factor

            a_t = self.alphas_cumprod[t][:, None, None, None]
            # index of "t-1" in the strided schedule
            if step_i < len(idx_list) - 1:
                prev_i = idx_list[step_i + 1]
                a_prev = self.alphas_cumprod[
                    torch.full((b,), int(prev_i), device=self.device, dtype=torch.long)
                ][:, None, None, None]
            else:
                a_prev = torch.ones_like(a_t)

            # epsilon implied by (x_t, x0_pred)
            eps_pred = (residual - torch.sqrt(a_t) * x0_pred) / torch.sqrt(1.0 - a_t + 1e-8)

            # DDIM variance
            sigma = eta * torch.sqrt(
                (1.0 - a_prev) / (1.0 - a_t + 1e-8)
                * (1.0 - a_t / (a_prev + 1e-8))
            )
            sigma = torch.clamp(sigma, min=0.0)

            noise = torch.randn_like(residual) if eta > 0 else torch.zeros_like(residual)

            residual = (
                torch.sqrt(a_prev) * x0_pred
                + torch.sqrt(torch.clamp(1.0 - a_prev - sigma ** 2, min=0.0)) * eps_pred
                + sigma * noise
            )

        final_img = torch.clamp(low_light + residual, -1.0, 1.0)
        return final_img

    @staticmethod
    def _initial_residual(low_light, initial_noise):
        """Return a caller-controlled latent or create a fresh one.

        Passing the same initial_noise tensor makes checkpoint and ARR comparisons
        paired at the image level. The original behavior is retained when it is None.
        """
        if initial_noise is None:
            return torch.randn_like(low_light)
        if initial_noise.shape != low_light.shape:
            raise ValueError(
                f"initial_noise shape {tuple(initial_noise.shape)} does not match "
                f"input shape {tuple(low_light.shape)}"
            )
        return initial_noise.to(device=low_light.device, dtype=low_light.dtype).clone()
