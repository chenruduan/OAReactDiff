from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_scatter import scatter_mean

from oa_reactdiff.dynamics import EGNNDynamics
from oa_reactdiff.utils import (
    get_n_frag_switch,
    get_mask_for_frag,
    get_edges_index,
)

import oa_reactdiff.diffusion._utils as utils
from oa_reactdiff.diffusion._schedule import DiffSchedule, get_repaint_schedule
from oa_reactdiff.diffusion._normalizer import Normalizer, FEATURE_MAPPING


class EnVariationalDiffusion(nn.Module):
    """
    The E(n) Diffusion Module.
    """

    def __init__(
        self,
        dynamics: EGNNDynamics,
        schdule: DiffSchedule,
        normalizer: Normalizer,
        size_histogram: Optional[Dict] = None,
        loss_type: str = "l2",
        pos_only: bool = False,
        fixed_idx: Optional[List] = None,
    ):
        super().__init__()
        assert loss_type in {"vlb", "l2"}

        self.dynamics = dynamics
        self.schedule = schdule
        self.normalizer = normalizer
        self.size_histogram = size_histogram
        self.loss_type = loss_type
        self.pos_only = pos_only
        self.fixed_idx = fixed_idx or []

        self.pos_dim = dynamics.pos_dim
        self.node_nfs = dynamics.node_nfs
        self.fragment_names = dynamics.fragment_names
        self.T = schdule.gamma_module.timesteps
        self.norm_values = normalizer.norm_values
        self.norm_biases = normalizer.norm_biases

    # ------ FORWARD PASS ------

    def forward(
        self,
        representations: List[Dict],
        conditions: Tensor,
        return_pred: bool = False,
    ):
        r"""
        Computes the loss and NLL terms.

        #TODO: edge_attr not considered at all
        """
        num_sample = representations[0]["size"].size(0)
        n_nodes = torch.stack(
            [repr["size"] for repr in representations],
            dim=0,
        ).sum(dim=0)
        device = representations[0]["pos"].device
        masks = [repre["mask"] for repre in representations]
        combined_mask = torch.cat(masks)
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        fragments_nodes = [repr["size"] for repr in representations]
        n_frag_switch = get_n_frag_switch(fragments_nodes)

        # Normalize data, take into account volume change in x.
        representations = self.normalizer.normalize(representations)

        # Likelihood change due to normalization
        delta_log_px = self.delta_log_px(n_nodes.sum())

        # Sample a timestep t for each example in batch
        # At evaluation time, loss_0 will be computed separately to decrease
        # variance in the estimator (costs two forward passes)
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(num_sample, 1), device=device
        ).float()
        s_int = t_int - 1  # previous timestep

        # Masks: important to compute log p(x | z0).
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.schedule.inflate_batch_array(
            self.schedule.gamma_module(s), representations[0]["pos"]
        )
        gamma_t = self.schedule.inflate_batch_array(
            self.schedule.gamma_module(t), representations[0]["pos"]
        )

        # Concatenate x, and h[categorical].
        xh = [
            torch.cat(
                [repre[feature_type] for feature_type in FEATURE_MAPPING],
                dim=1,
            )
            for repre in representations
        ]

        # Find noised representation
        z_t, eps_xh = self.noised_representation(xh, masks, gamma_t)

        # Neural net prediction.
        net_eps_xh, net_eps_edge_attr = self.dynamics(
            xh=z_t,
            edge_index=edge_index,
            t=t,
            conditions=conditions,
            n_frag_switch=n_frag_switch,
            combined_mask=combined_mask,
            edge_attr=None,  # TODO: no edge_attr is considered now
        )

        if return_pred:
            return eps_xh, net_eps_xh

        # TODO: LJ term not implemented
        # xh_lig_hat = self.xh_given_zt_and_epsilon(z_t_lig, net_out_lig, gamma_t,
        #                                           ligand['mask'])
        if self.pos_only:
            for ii in range(len(masks)):
                net_eps_xh[ii][:, self.pos_dim :] = torch.zeros_like(
                    net_eps_xh[ii][:, self.pos_dim :],
                    device=device,
                )
        # Compute the L2 error.
        error_t: List[Tensor] = [
            utils.sum_except_batch(
                (eps_xh[ii] - net_eps_xh[ii]) ** 2,
                masks[ii],
                dim_size=num_sample,
            )
            for ii in range(len(masks))
        ]  # TODO: no edge_attr contribution

        # Compute weighting with SNR: (1 - SNR(s-t)) for epsilon parametrization
        SNR_weight = (1 - self.schedule.SNR(gamma_s - gamma_t)).squeeze(1)
        assert error_t[0].size() == SNR_weight.size()

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(
            n_nodes=n_nodes, device=device
        )

        # The KL between q(zT | x) and p(zT) = Normal(0, 1).
        # Should be close to zero.
        # kl_prior = self.kl_prior_with_pocket(
        #     xh_lig, xh_pocket, ligand['mask'], pocket['mask'],
        #     ligand['size'] + pocket['size'])
        # TODO: approximate KL prior with zero now, which should not influence training.
        kl_prior = torch.zeros_like(neg_log_constants)

        if self.training:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            log_p_h_given_z0 = self.log_pxh_given_z0_without_constants(
                representations=representations,
                z_t=z_t,
                eps_xh=eps_xh,
                net_eps_xh=net_eps_xh,
                gamma_t=gamma_t,
                epsilon=1e-10,
            )
            loss_0_x = [
                -_log_p_fragment * t_is_zero.squeeze()
                for _log_p_fragment in log_p_h_given_z0[0]
            ]
            loss_0_cat = [
                -_log_p_fragment * t_is_zero.squeeze()
                for _log_p_fragment in log_p_h_given_z0[1]
            ]
            loss_0_charge = [
                -_log_p_fragment * t_is_zero.squeeze()
                for _log_p_fragment in log_p_h_given_z0[2]
            ]

            # apply t_is_zero mask
            error_t = [_error_t * t_is_not_zero.squeeze() for _error_t in error_t]

        else:
            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.schedule.inflate_batch_array(
                self.schedule.gamma_module(t_zeros), representations[0]["pos"]
            )

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            z_0, eps_0_xh = self.noised_representation(xh, masks, gamma_0)
            net_eps_0_xh, net_eps_0_edge_attr = self.dynamics(
                xh=z_0,
                edge_index=edge_index,
                t=t_zeros,
                conditions=conditions,
                n_frag_switch=n_frag_switch,
                combined_mask=combined_mask,
                edge_attr=None,  # TODO: no edge_attr is considered now
            )

            log_p_h_given_z0 = self.log_pxh_given_z0_without_constants(
                representations=representations,
                z_t=z_0,
                eps_xh=eps_0_xh,
                net_eps_xh=net_eps_0_xh,
                gamma_t=gamma_0,
                epsilon=1e-10,
            )
            loss_0_x = [-_log_p_fragment for _log_p_fragment in log_p_h_given_z0[0]]
            loss_0_cat = [-_log_p_fragment for _log_p_fragment in log_p_h_given_z0[1]]
            loss_0_charge = [
                -_log_p_fragment for _log_p_fragment in log_p_h_given_z0[2]
            ]

        loss_terms = {
            "delta_log_px": delta_log_px,
            "error_t": error_t,
            "SNR_weight": SNR_weight,
            "loss_0_x": loss_0_x,
            "loss_0_cat": loss_0_cat,
            "loss_0_charge": loss_0_charge,
            "neg_log_constants": neg_log_constants,
            "kl_prior": kl_prior,
            "log_pN": torch.zeros_like(kl_prior),
            "t_int": t_int.squeeze(),
            "net_eps_xh": net_eps_xh,
            "eps_xh": eps_xh,
        }
        return loss_terms

    def delta_log_px(self, num_nodes):
        return -self.subspace_dimensionality(num_nodes) * np.log(self.norm_values[0])

    def subspace_dimensionality(self, input_size):
        r"""
        Compute the dimensionality on translation-invariant linear subspace
        where distributions on x are defined.
        """
        return (input_size - 1) * self.pos_dim

    def noised_representation(
        self,
        xh: List[Tensor],
        masks: List[Tensor],
        gamma_t: Tensor,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.schedule.alpha(gamma_t, xh[0])
        sigma_t = self.schedule.sigma(gamma_t, xh[0])

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps_xh = self.sample_combined_position_feature_noise(masks)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = [
            alpha_t[masks[ii]] * xh[ii] + sigma_t[masks[ii]] * eps_xh[ii]
            for ii in range(len(masks))
        ]

        return z_t, eps_xh

    def sample_combined_position_feature_noise(
        self,
        masks: List[Tensor],
    ) -> List[Tensor]:
        r"""
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        Note that we only need to put the center of gravity of *each fragment* to the origin.
        """
        eps_xh = []
        for ii, mask in enumerate(masks):
            _eps_x = utils.sample_center_gravity_zero_gaussian_batch(
                size=(len(mask), self.pos_dim),
                indices=[mask],
            )
            _eps_h = utils.sample_gaussian(
                size=(len(mask), self.node_nfs[ii] - self.pos_dim),
                device=mask.device,
            )
            if self.pos_only:
                _eps_h = torch.zeros_like(_eps_h, device=mask.device)
            eps_xh.append(torch.cat([_eps_x, _eps_h], dim=1))
        for idx in self.fixed_idx:
            eps_xh[idx] = torch.zeros_like(eps_xh[idx], device=mask.device)
        return eps_xh

    def log_constants_p_x_given_z0(self, n_nodes, device):
        r"""Computes p(x|z0)."""

        batch_size = len(n_nodes)
        degrees_of_freedom_x = self.subspace_dimensionality(n_nodes).to(device)

        zeros = torch.zeros((batch_size, 1), device=device)
        gamma_0 = self.schedule.gamma_module(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)
        return degrees_of_freedom_x * (-log_sigma_x - 0.5 * np.log(2 * np.pi))

    def kl_prior(self):
        return NotImplementedError

    @staticmethod
    def gaussian_KL(q_mu_minus_p_mu_squared, q_sigma, p_sigma, d):
        """Computes the KL distance between two normal distributions.
        Args:
            q_mu_minus_p_mu_squared: Squared difference between mean of
                distribution q and distribution p: ||mu_q - mu_p||^2
            q_sigma: Standard deviation of distribution q.
            p_sigma: Standard deviation of distribution p.
            d: dimension
        Returns:
            The KL distance
        """
        return (
            d * torch.log(p_sigma / q_sigma)
            + 0.5 * (d * q_sigma**2 + q_mu_minus_p_mu_squared) / (p_sigma**2)
            - 0.5 * d
        )

    def log_pxh_given_z0_without_constants(
        self,
        representations: List[Dict],
        z_t: List[Tensor],
        eps_xh: List[Tensor],
        net_eps_xh: List[Tensor],
        gamma_t: Tensor,
        epsilon: float = 1e-10,
    ) -> List[List[Tensor]]:
        # Compute sigma_0 and rescale to the integer scale of the data.
        # for pos
        log_p_x_given_z0_without_constants = [
            -0.5
            * (
                utils.sum_except_batch(
                    (eps_xh[ii][:, : self.pos_dim] - net_eps_xh[ii][:, : self.pos_dim])
                    ** 2,
                    representations[ii]["mask"],
                    dim_size=representations[0]["size"].size(0),
                )
            )
            for ii in range(len(representations))
        ]

        # only keep first several elements
        z_t = [_z_t[:, : 3 + 5 + 1] for _z_t in z_t]
        for ii, repr in enumerate(representations):
            representations[ii]["charge"] = representations[ii]["charge"][:, :1]
        # for ohe of atom types
        sigma_0 = self.schedule.sigma(gamma_t, target_tensor=z_t[0])
        sigma_0_cat = sigma_0 * self.normalizer.norm_values[1]
        atoms = [
            self.normalizer.unnormalize(repr["one_hot"], ind=1)
            for repr in representations
        ]
        est_atoms = [
            self.normalizer.unnormalize(_z_t[:, self.pos_dim : -1], ind=1)
            for _z_t in z_t
        ]
        centered_atoms = [_est_atoms - 1 for _est_atoms in est_atoms]
        log_ph_cat_proportionals = [
            torch.log(
                utils.cdf_standard_gaussian(
                    (centered_atoms[ii] + 0.5)
                    / sigma_0_cat[representations[ii]["mask"]]
                )
                - utils.cdf_standard_gaussian(
                    (centered_atoms[ii] - 0.5)
                    / sigma_0_cat[representations[ii]["mask"]]
                )
                + epsilon
            )
            for ii in range(len(representations))
        ]
        log_probabilities = [
            _log_ph_cat_proportionals
            - torch.logsumexp(
                _log_ph_cat_proportionals,
                dim=1,
                keepdim=True,
            )
            for _log_ph_cat_proportionals in log_ph_cat_proportionals
        ]
        log_p_hcat_given_z0 = [
            utils.sum_except_batch(
                log_probabilities[ii] * atoms[ii],
                representations[ii]["mask"],
                dim_size=representations[0]["size"].size(0),
            )
            for ii in range(len(representations))
        ]

        # for atom charge
        sigma_0_charge = sigma_0 * self.normalizer.norm_values[2]
        charges = [
            self.normalizer.unnormalize(repr["charge"], ind=2)
            for repr in representations
        ]
        est_charges = [
            self.normalizer.unnormalize(_z_t[:, -1:], ind=2).long() for _z_t in z_t
        ]
        for ii in range(len(representations)):
            assert charges[ii].size() == est_charges[ii].size()
        centered_charges = [
            charges[ii] - est_charges[ii] for ii in range(len(representations))
        ]
        log_ph_charge_proportionals = [
            torch.log(
                utils.cdf_standard_gaussian(
                    (centered_charges[ii] + 0.5)
                    / sigma_0_charge[representations[ii]["mask"]]
                )
                - utils.cdf_standard_gaussian(
                    (centered_charges[ii] - 0.5)
                    / sigma_0_charge[representations[ii]["mask"]]
                )
                + epsilon
            )
            for ii in range(len(representations))
        ]
        log_p_hcharge_given_z0 = [
            utils.sum_except_batch(
                log_ph_charge_proportionals[ii],
                representations[ii]["mask"],
                dim_size=representations[0]["size"].size(0),
            )
            for ii in range(len(representations))
        ]

        log_p_h_given_z0 = [
            log_p_x_given_z0_without_constants,
            log_p_hcat_given_z0,
            log_p_hcharge_given_z0,
        ]
        return log_p_h_given_z0

    # ------ INVERSE PASS ------

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        fragments_nodes: List[torch.tensor],
        conditions: Optional[Tensor] = None,
        return_frames: int = 1,
        timesteps: Optional[int] = None,
        h0: Optional[List[Tensor]] = None,
    ):
        r"""
        Draw samples from the generative model. Optionally, return intermediate
        states for visualization purposes.
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0
        assert h0 is not None if self.pos_only else True

        fragments_masks = [
            get_mask_for_frag(natm_nodes) for natm_nodes in fragments_nodes
        ]
        combined_mask = torch.cat(fragments_masks)
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        n_frag_switch = get_n_frag_switch(fragments_nodes)

        zt_xh = self.sample_combined_position_feature_noise(masks=fragments_masks)
        if self.pos_only:
            zt_xh = [
                torch.cat([zt_xh[ii][:, : self.pos_dim], h0[ii]], dim=1)
                for ii in range(len(h0))
            ]

        utils.assert_mean_zero_with_mask(
            torch.cat(
                [_zt_xh[:, : self.pos_dim] for _zt_xh in zt_xh],
                dim=0,
            ),
            combined_mask,
        )

        out_samples = [
            [
                torch.zeros((return_frames,) + _zt_xh.size(), device=_zt_xh.device)
                for _zt_xh in zt_xh
            ]
            for _ in range(return_frames)
        ]

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, timesteps)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=zt_xh[0].device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps

            # print(s, zt_xh)

            zt_xh = self.sample_p_zs_given_zt(
                s=s_array,
                t=t_array,
                zt_xh=zt_xh,
                edge_index=edge_index,
                n_frag_switch=n_frag_switch,
                masks=fragments_masks,
                conditions=conditions,
                fix_noise=False,
            )
            if self.pos_only:
                zt_xh = [
                    torch.cat([zt_xh[ii][:, : self.pos_dim], h0[ii]], dim=1)
                    for ii in range(len(h0))
                ]

            # save frame
            if (s * return_frames) % timesteps == 0:
                idx = (s * return_frames) // timesteps
                out_samples[idx] = self.normalizer.unnormalize_z(zt_xh)

        pos, cat, charge = self.sample_p_xh_given_z0(
            z0_xh=zt_xh,
            edge_index=edge_index,
            n_frag_switch=n_frag_switch,
            masks=fragments_masks,
            batch_size=n_samples,
            conditions=conditions,
        )
        if self.pos_only:
            cat = [_h0[:, :-1] for _h0 in h0]
            charge = [_h0[:, -1:] for _h0 in h0]
        utils.assert_mean_zero_with_mask(
            torch.cat(
                [_pos[:, : self.pos_dim] for _pos in pos],
                dim=0,
            ),
            combined_mask,
        )

        # Overwrite last frame with the resulting x and h.
        out_samples[0] = [
            torch.cat([pos[ii], cat[ii], charge[ii]], dim=1) for ii in range(len(pos))
        ]
        return out_samples, fragments_masks

    def sample_p_zs_given_zt(
        self,
        s: Tensor,
        t: Tensor,
        zt_xh: List[Tensor],
        edge_index: Tensor,
        n_frag_switch: Tensor,
        masks: List[Tensor],
        conditions: Optional[Tensor] = None,
        fix_noise: bool = False,
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.schedule.gamma_module(s)
        gamma_t = self.schedule.gamma_module(t)

        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.schedule.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_xh[0])

        sigma_s = self.schedule.sigma(gamma_s, target_tensor=zt_xh[0])
        sigma_t = self.schedule.sigma(gamma_t, target_tensor=zt_xh[0])

        # Neural net prediction.
        combined_mask = torch.cat(masks)
        net_eps_xh, net_eps_edge_attr = self.dynamics(
            xh=zt_xh,
            edge_index=edge_index,
            t=t,
            conditions=conditions,
            n_frag_switch=n_frag_switch,
            combined_mask=combined_mask,
            edge_attr=None,  # TODO: no edge_attr is considered now
        )
        utils.assert_mean_zero_with_mask(
            torch.cat(
                [_zt_xh[:, : self.pos_dim] for _zt_xh in zt_xh],
                dim=0,
            ),
            combined_mask,
        )
        utils.assert_mean_zero_with_mask(
            torch.cat(
                [_net_eps_xh[:, : self.pos_dim] for _net_eps_xh in net_eps_xh],
                dim=0,
            ),
            combined_mask,
        )

        # Note: mu_{t->s} = 1 / alpha_{t|s} z_t - sigma_{t|s}^2 / sigma_t / alpha_{t|s} epsilon
        # follows from the definition of mu_{t->s} and Equ. (7) in the EDM paper
        mu = [
            zt_xh[ii] / alpha_t_given_s[masks[ii]]
            - net_eps_xh[ii] * (sigma2_t_given_s / alpha_t_given_s / sigma_t)[masks[ii]]
            for ii in range(len(zt_xh))
        ]

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs_xh = self.sample_normal(mu=mu, sigma=sigma, masks=masks, fix_noise=fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        for ii in range(len(masks)):
            zs_xh[ii][:, : self.pos_dim] = utils.remove_mean_batch(
                zs_xh[ii][:, : self.pos_dim],
                masks[ii],
            )
        return zs_xh

    def sample_normal(
        self,
        mu: List[Tensor],
        sigma: Tensor,
        masks: List[Tensor],
        fix_noise: bool = False,
    ) -> List[Tensor]:
        r"""Samples from a Normal distribution."""
        if fix_noise:
            # bs = 1 if fix_noise else mu.size(0)
            raise NotImplementedError("fix_noise option isn't implemented yet")
        eps_xh = self.sample_combined_position_feature_noise(masks=masks)
        zs_xh = [mu[ii] + sigma[masks[ii]] * eps_xh[ii] for ii in range(len(masks))]
        return zs_xh

    def sample_p_xh_given_z0(
        self,
        z0_xh: List[Tensor],
        edge_index: Tensor,
        n_frag_switch: Tensor,
        masks: List[Tensor],
        batch_size: int,
        conditions: Optional[Tensor] = None,
        fix_noise: bool = False,
    ) -> Tuple[List[Tensor]]:
        """Samples x ~ p(x|z0)."""
        t_zeros = torch.zeros(size=(batch_size, 1), device=z0_xh[0].device)
        gamma_0 = self.schedule.gamma_module(t_zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.schedule.SNR(-0.5 * gamma_0)
        net_eps_xh, net_eps_edge_attr = self.dynamics(
            xh=z0_xh,
            edge_index=edge_index,
            t=t_zeros,
            conditions=conditions,
            n_frag_switch=n_frag_switch,
            combined_mask=torch.cat(masks),
            edge_attr=None,  # TODO: no edge_attr is considered now
        )

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(
            net_eps_xh=net_eps_xh,
            zt_xh=z0_xh,
            gamma_t=gamma_0,
            masks=masks,
        )
        x0_xh = self.sample_normal(
            mu=mu_x, sigma=sigma_x, masks=masks, fix_noise=fix_noise
        )

        pos_0 = [
            self.normalizer.unnormalize(x0_xh[ii][:, : self.pos_dim], ii)
            for ii in range(len(masks))
        ]
        cat_0 = [
            self.normalizer.unnormalize(x0_xh[ii][:, self.pos_dim : -1], ii)
            for ii in range(len(masks))
        ]
        charge_0 = [
            torch.round(self.normalizer.unnormalize(x0_xh[ii][:, -1:], ii)).long()
            for ii in range(len(masks))
        ]

        cat_0 = [
            F.one_hot(torch.argmax(cat_0[ii], dim=1), self.node_nfs[ii] - 4).long()
            for ii in range(len(masks))
        ]
        return pos_0, cat_0, charge_0

    def compute_x_pred(
        self,
        net_eps_xh: List[Tensor],
        zt_xh: List[Tensor],
        gamma_t: Tensor,
        masks: List[Tensor],
    ) -> List[Tensor]:
        """Commputes x_pred, i.e. the most likely prediction of x."""
        sigma_t = self.schedule.sigma(gamma_t, target_tensor=net_eps_xh[0])
        alpha_t = self.schedule.alpha(gamma_t, target_tensor=net_eps_xh[0])
        x_pred = [
            1.0 / alpha_t[masks[ii]] * (zt_xh[ii] - sigma_t[masks[ii]] * net_eps_xh[ii])
            for ii in range(len(masks))
        ]
        return x_pred

    # ------ INPAINT ------
    @torch.no_grad()
    def inpaint(
        self,
        n_samples: int,
        fragments_nodes: List[torch.tensor],
        conditions: Optional[Tensor] = None,
        return_frames: int = 1,
        resamplings: int = 1,
        jump_length: int = 1,
        timesteps: Optional[int] = None,
        xh_fixed: Optional[List[Tensor]] = None,
        frag_fixed: Optional[List] = None,
    ):
        r"""
        Draw samples from the generative model. Optionally, return intermediate
        states for visualization purposes.
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0
        assert len(xh_fixed)

        fragments_masks = [
            get_mask_for_frag(natm_nodes) for natm_nodes in fragments_nodes
        ]
        combined_mask = torch.cat(fragments_masks)
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        n_frag_switch = get_n_frag_switch(fragments_nodes)

        h0 = [_xh_fixed[:, self.pos_dim :].long() for _xh_fixed in xh_fixed]

        for ii, _ in enumerate(xh_fixed):
            xh_fixed[ii][:, : self.pos_dim] = utils.remove_mean_batch(
                xh_fixed[ii][:, : self.pos_dim],
                fragments_masks[ii],
            )
        utils.assert_mean_zero_with_mask(
            torch.cat(
                [_xh_fixed[:, : self.pos_dim] for _xh_fixed in xh_fixed],
                dim=0,
            ),
            combined_mask,
        )

        zt_xh = self.sample_combined_position_feature_noise(masks=fragments_masks)
        if self.pos_only:
            zt_xh = [
                torch.cat([zt_xh[ii][:, : self.pos_dim], h0[ii]], dim=1)
                for ii in range(len(h0))
            ]

        utils.assert_mean_zero_with_mask(
            torch.cat(
                [_zt_xh[:, : self.pos_dim] for _zt_xh in zt_xh],
                dim=0,
            ),
            combined_mask,
        )

        out_samples = [
            [
                torch.zeros((return_frames,) + _zt_xh.size(), device=_zt_xh.device)
                for _zt_xh in zt_xh
            ]
            for _ in range(return_frames)
        ]

        schedule = get_repaint_schedule(resamplings, jump_length, timesteps)
        s = timesteps - 1
        for i, n_denoise_steps in enumerate(schedule):
            for j in range(n_denoise_steps):
                s_array = torch.full(
                    (n_samples, 1), fill_value=s, device=zt_xh[0].device
                )
                t_array = s_array + 1
                s_array = s_array / timesteps
                t_array = t_array / timesteps

                gamma_s = self.schedule.inflate_batch_array(
                    self.schedule.gamma_module(s_array), xh_fixed[0]
                )

                zt_known, _ = self.noised_representation(
                    xh_fixed, fragments_masks, gamma_s
                )
                zt_unknown = self.sample_p_zs_given_zt(
                    s=s_array,
                    t=t_array,
                    zt_xh=zt_xh,
                    edge_index=edge_index,
                    n_frag_switch=n_frag_switch,
                    masks=fragments_masks,
                    conditions=conditions,
                    fix_noise=False,
                )

                if self.pos_only:
                    zt_known = [
                        torch.cat([zt_known[ii][:, : self.pos_dim], h0[ii]], dim=1)
                        for ii in range(len(h0))
                    ]
                    zt_unknown = [
                        torch.cat([zt_unknown[ii][:, : self.pos_dim], h0[ii]], dim=1)
                        for ii in range(len(h0))
                    ]

                zt_xh = [
                    zt_known[ii] if ii in frag_fixed else zt_unknown[ii]
                    for ii in range(len(h0))
                ]

                # Noise combined representation, i.e., resample
                if j == n_denoise_steps - 1 and i < len(schedule) - 1:
                    # Go back jump_length steps
                    t = s + jump_length
                    t_array = torch.full(
                        (n_samples, 1), fill_value=t, device=zt_xh[0].device
                    )
                    t_array = t_array / timesteps

                    gamma_s = self.schedule.inflate_batch_array(
                        self.schedule.gamma_module(s_array), xh_fixed[0]
                    )
                    gamma_t = self.schedule.inflate_batch_array(
                        self.schedule.gamma_module(t_array), xh_fixed[0]
                    )

                    zt_xh = self.sample_p_zt_given_zs(
                        zt_xh, fragments_masks, gamma_t, gamma_s
                    )
                    s = t

                s = s - 1

                # # save frame
                # if (s * return_frames) % timesteps == 0:
                #     idx = (s * return_frames) // timesteps
                #     out_samples[idx] = self.normalizer.unnormalize_z(zt_xh)

        pos, cat, charge = self.sample_p_xh_given_z0(
            z0_xh=zt_xh,
            edge_index=edge_index,
            n_frag_switch=n_frag_switch,
            masks=fragments_masks,
            batch_size=n_samples,
            conditions=conditions,
        )
        if self.pos_only:
            cat = [_h0[:, :-1] for _h0 in h0]
            charge = [_h0[:, -1:] for _h0 in h0]
        utils.assert_mean_zero_with_mask(
            torch.cat(
                [_pos[:, : self.pos_dim] for _pos in pos],
                dim=0,
            ),
            combined_mask,
        )

        # Overwrite last frame with the resulting x and h.
        out_samples[0] = [
            torch.cat([pos[ii], cat[ii], charge[ii]], dim=1) for ii in range(len(pos))
        ]
        return out_samples, fragments_masks

    # ------ INPAINT ------
    @torch.no_grad()
    def inpaint_fixed(
        self,
        n_samples: int,
        fragments_nodes: List[torch.tensor],
        conditions: Optional[Tensor] = None,
        return_frames: int = 1,
        resamplings: int = 1,
        jump_length: int = 1,
        timesteps: Optional[int] = None,
        xh_fixed: Optional[List[Tensor]] = None,
        frag_fixed: Optional[List] = None,
    ):
        r"""
        Draw samples from the generative model. Optionally, return intermediate
        states for visualization purposes.
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0
        assert len(xh_fixed)

        fragments_masks = [
            get_mask_for_frag(natm_nodes) for natm_nodes in fragments_nodes
        ]
        combined_mask = torch.cat(fragments_masks)
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        n_frag_switch = get_n_frag_switch(fragments_nodes)

        h0 = [_xh_fixed[:, self.pos_dim :].long() for _xh_fixed in xh_fixed]

        for ii, _ in enumerate(xh_fixed):
            xh_fixed[ii][:, : self.pos_dim] = utils.remove_mean_batch(
                xh_fixed[ii][:, : self.pos_dim],
                fragments_masks[ii],
            )
        utils.assert_mean_zero_with_mask(
            torch.cat(
                [_xh_fixed[:, : self.pos_dim] for _xh_fixed in xh_fixed],
                dim=0,
            ),
            combined_mask,
        )

        zt_xh = self.sample_combined_position_feature_noise(masks=fragments_masks)
        if self.pos_only:
            zt_xh = [
                torch.cat([zt_xh[ii][:, : self.pos_dim], h0[ii]], dim=1)
                for ii in range(len(h0))
            ]

        utils.assert_mean_zero_with_mask(
            torch.cat(
                [_zt_xh[:, : self.pos_dim] for _zt_xh in zt_xh],
                dim=0,
            ),
            combined_mask,
        )

        out_samples = [
            [
                torch.zeros((return_frames,) + _zt_xh.size(), device=_zt_xh.device)
                for _zt_xh in zt_xh
            ]
            for _ in range(return_frames)
        ]

        schedule = get_repaint_schedule(resamplings, jump_length, timesteps)
        s = timesteps - 1
        for i, n_denoise_steps in enumerate(schedule):
            for j in range(n_denoise_steps):
                s_array = torch.full(
                    (n_samples, 1), fill_value=s, device=zt_xh[0].device
                )
                t_array = s_array + 1
                s_array = s_array / timesteps
                t_array = t_array / timesteps

                gamma_s = self.schedule.inflate_batch_array(
                    self.schedule.gamma_module(s_array), xh_fixed[0]
                )

                zt_known, _ = self.noised_representation(
                    xh_fixed, fragments_masks, gamma_s
                )
                zt_unknown = self.sample_p_zs_given_zt(
                    s=s_array,
                    t=t_array,
                    zt_xh=zt_xh,
                    edge_index=edge_index,
                    n_frag_switch=n_frag_switch,
                    masks=fragments_masks,
                    conditions=conditions,
                    fix_noise=False,
                )

                if self.pos_only:
                    zt_known = [
                        torch.cat([zt_known[ii][:, : self.pos_dim], h0[ii]], dim=1)
                        for ii in range(len(h0))
                    ]
                    zt_unknown = [
                        torch.cat([zt_unknown[ii][:, : self.pos_dim], h0[ii]], dim=1)
                        for ii in range(len(h0))
                    ]

                zt_xh = [
                    zt_known[ii] if ii in frag_fixed else zt_unknown[ii]
                    for ii in range(len(h0))
                ]

                # Noise combined representation, i.e., resample
                if j == n_denoise_steps - 1 and i < len(schedule) - 1:
                    # Go back jump_length steps
                    t = s + jump_length
                    t_array = torch.full(
                        (n_samples, 1), fill_value=t, device=zt_xh[0].device
                    )
                    t_array = t_array / timesteps

                    gamma_s = self.schedule.inflate_batch_array(
                        self.schedule.gamma_module(s_array), xh_fixed[0]
                    )
                    gamma_t = self.schedule.inflate_batch_array(
                        self.schedule.gamma_module(t_array), xh_fixed[0]
                    )

                    zt_xh = self.sample_p_zt_given_zs(
                        zt_xh, fragments_masks, gamma_t, gamma_s
                    )
                    s = t

                s = s - 1

                # # save frame
                # if (s * return_frames) % timesteps == 0:
                #     idx = (s * return_frames) // timesteps
                #     out_samples[idx] = self.normalizer.unnormalize_z(zt_xh)

        pos, cat, charge = self.sample_p_xh_given_z0(
            z0_xh=zt_xh,
            edge_index=edge_index,
            n_frag_switch=n_frag_switch,
            masks=fragments_masks,
            batch_size=n_samples,
            conditions=conditions,
        )
        if self.pos_only:
            cat = [_h0[:, :-1] for _h0 in h0]
            charge = [_h0[:, -1:] for _h0 in h0]
        utils.assert_mean_zero_with_mask(
            torch.cat(
                [_pos[:, : self.pos_dim] for _pos in pos],
                dim=0,
            ),
            combined_mask,
        )

        # Overwrite last frame with the resulting x and h.
        out_samples[0] = [
            torch.cat([pos[ii], cat[ii], charge[ii]], dim=1) for ii in range(len(pos))
        ]
        return out_samples, fragments_masks

    def sample_p_zt_given_zs(
        self,
        zs: List[Tensor],
        masks: List[Tensor],
        gamma_t: Tensor,
        gamma_s: Tensor,
        fix_noise: bool = False,
    ) -> List[Tensor]:
        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.schedule.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zs[0])

        mu = [alpha_t_given_s[masks[ii]] * zs[ii] for ii in range(len(masks))]
        zt = self.sample_normal(
            mu=mu, sigma=sigma_t_given_s, masks=masks, fix_noise=fix_noise
        )

        for ii in range(len(masks)):
            zt[ii][:, : self.pos_dim] = utils.remove_mean_batch(
                zt[ii][:, : self.pos_dim],
                masks[ii],
            )
        return zt
