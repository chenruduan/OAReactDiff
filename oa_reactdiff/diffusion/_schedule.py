"""t schedule used in diffusion process."""
from typing import Tuple
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    r"""
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def ccosine_schedule(timesteps, start=0, end=1, tau=1, clip_min=1e-9):
    t = np.linspace(0, 1, timesteps + 1)
    v_start = np.cos(start * np.pi / 2) ** (2 * tau)
    v_end = np.cos(end * np.pi / 2) ** (2 * tau)
    output = np.cos((t * (end - start) + start) * np.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1 - clip_min)


def linear_schedule(timesteps, clip_min=1e-9):
    t = np.linspace(0, 1, timesteps + 1)
    output = 1 - t
    return np.clip(output, clip_min, 1 - clip_min)


def clip_noise_schedule(alphas2, clip_value=0.001):
    r"""
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1.
    This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.0):
    r"""
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


class PredefinedNoiseSchedule(nn.Module):
    r"""
    Predefined noise schedule. Essentially creates a lookup array for predefined
    (non-learned) noise schedules.
    """

    def __init__(
        self,
        noise_schedule: str,
        timesteps: int,
        precision: float,
    ):
        super().__init__()
        self.timesteps = timesteps

        if "cosine" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) <= 2
            power = 1 if len(splits) == 1 else float(splits[1])
            alphas2 = cosine_beta_schedule(timesteps, raise_to_power=power)
        elif "polynomial" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        elif "csin" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) == 4
            start, end, tau = float(splits[1]), float(splits[2]), float(splits[3])
            alphas2 = ccosine_schedule(timesteps, start=start, end=end, tau=tau)
        elif "linear" in noise_schedule:
            alphas2 = linear_schedule(timesteps)
        else:
            raise ValueError(noise_schedule)

        # print("alphas2", alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        # print("gamma", -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False
        )

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class DiffSchedule(nn.Module):
    def __init__(self, gamma_module: nn.Module, norm_values: Tuple[float]) -> None:
        super().__init__()
        self.gamma_module = gamma_module
        self.norm_values = norm_values
        self.check_issues_norm_values()

    @staticmethod
    def inflate_batch_array(array, target):
        r"""
        Inflates the batch array (array) with only a single axis
        (i.e. shape = (batch_size,), or possibly more empty axes
        (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        r"""Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        r"""Computes alpha given gamma."""
        return self.inflate_batch_array(
            torch.sqrt(torch.sigmoid(-gamma)), target_tensor
        )

    @staticmethod
    def SNR(gamma):
        r"""Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s(
        self, gamma_t: Tensor, gamma_s: Tensor, target_tensor: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.
        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma_module(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        norm_value = self.norm_values[1]

        if sigma_0 * num_stdevs > 1.0 / norm_value:
            raise ValueError(
                f"Value for normalization value {norm_value} probably too "
                f"large with sigma_0 {sigma_0:.5f} and "
                f"1 / norm_value = {1. / norm_value}"
            )


def get_repaint_schedule(resamplings, jump_length, timesteps):
    """
    Each integer in the schedule list describes how many denoising steps
    need to be applied before jumping back.

    sum(out) - (len(out) -1) * jump_length = timesteps

    """
    repaint_schedule = []
    curr_t = 0
    while curr_t < timesteps:
        if curr_t + jump_length < timesteps:
            if len(repaint_schedule) > 0:
                repaint_schedule[-1] += jump_length
                repaint_schedule.extend([jump_length] * (resamplings - 1))
            else:
                repaint_schedule.extend([jump_length] * resamplings)
            curr_t += jump_length
        else:
            residual = timesteps - curr_t
            if len(repaint_schedule) > 0:
                repaint_schedule[-1] += residual
            else:
                repaint_schedule.append(residual)
            curr_t += residual

    return list(reversed(repaint_schedule))
