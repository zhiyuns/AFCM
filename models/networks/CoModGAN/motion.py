from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .torch_utils import misc
from .torch_utils import persistence
from .layers import (
    MappingNetwork,
    EqLRConv1d,
    FullyConnectedLayer,
)


# ----------------------------------------------------------------------------

@persistence.persistent_class
class MotionMappingNetwork(torch.nn.Module):
    def __init__(self,
                 max_num_frames,
                 motion_v_dim,
                 motion_z_dim,
                 motion_z_distance,
                 motion_kernel_size,
                 fourier=True,
                 motion_gen_strategy='conv',
                 time_encoder_kwargs={},
                 ):
        super().__init__()
        self.max_num_frames = max_num_frames
        self.motion_v_dim = motion_v_dim
        self.motion_z_dim = motion_z_dim
        self.motion_z_distance = motion_z_distance
        self.fourier = fourier
        self.motion_gen_strategy = motion_gen_strategy

        assert motion_gen_strategy in ["autoregressive", "conv"], f"Unknown generation strategy: {motion_gen_strategy}"

        if fourier:
            self.time_encoder = AlignedTimeEncoder(
                latent_dim=motion_v_dim,
                **time_encoder_kwargs
            )
        else:
            self.mapping = MappingNetwork(
                z_dim=motion_z_dim,
                w_dim=motion_v_dim,
                num_ws=None,
                num_layers=2,
                activation='lrelu',
                w_avg_beta=None
            )

        if self.motion_gen_strategy == 'autoregressive':
            self.rnn = nn.LSTM(
                input_size=motion_z_dim,
                hidden_size=motion_z_dim,
                bidirectional=False,
                batch_first=True)
            self._parameters_flattened = False
            self.num_additional_codes = 0
        elif motion_gen_strategy == 'conv':
            # Using Conv1d without paddings instead of LSTM makes the generations good for any time in t \in (0, +\infty),
            # while LSTM would diverge for large `t`
            # Also, this allows us to use equalized learning rates
            self.conv = nn.Sequential(
                EqLRConv1d(motion_z_dim, motion_z_dim, motion_kernel_size,
                           padding=0, activation='lrelu', lr_multiplier=0.01),
                EqLRConv1d(motion_z_dim, motion_v_dim, motion_kernel_size, padding=0,
                           activation='lrelu', lr_multiplier=0.01),
            )
            self.num_additional_codes = (motion_kernel_size - 1) * 2
        else:
            raise NotImplementedError(f'Unknown generation strategy: {motion_gen_strategy}')

    def get_max_traj_len(self, t: torch.Tensor) -> int:
        max_t = max(self.max_num_frames - 1, t.max().item())  # [1]
        max_traj_len = np.ceil(max_t / self.motion_z_distance).astype(int).item() + 2  # [1]
        return max_traj_len

    def generate_motion_u_codes(self, t: torch.Tensor, motion_z: torch.Tensor = None) -> Dict:
        """
        Arguments:
            - t of shape [batch_size, num_frames]
            - w of shape [batch_size, w_dim]
            - motion_z of shape [batch_size, max_traj_len, motion_z_dim] --- in case we want to reuse some existing motion noise
        """
        out = {}
        batch_size, num_frames = t.shape

        # Consutruct trajectories (from code idx for now)
        max_traj_len = self.get_max_traj_len(t) + self.num_additional_codes  # [1]

        if motion_z is None:
            motion_z = torch.randn(batch_size, max_traj_len, self.motion_z_dim,
                                   device=t.device)  # [batch_size, max_traj_len, motion.z_dim]

        # Input motion trajectories are just random noise
        input_trajs = motion_z[:batch_size, :max_traj_len, :self.motion_z_dim].to(
            t.device)  # [batch_size, max_traj_len, motion.z_dim]

        if self.motion_gen_strategy == 'autoregressive':
            # Somehow, RNN parameters do not get flattened on their own and we get a lot of warnings...
            if not self._parameters_flattened:
                self.rnn.flatten_parameters()
                self._parameters_flattened = True
            trajs, _ = self.rnn(input_trajs)  # [batch_size, max_traj_len, motion.z_dim]
        elif self.motion_gen_strategy == 'conv':
            trajs = self.conv(input_trajs.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, max_traj_len, motion.v_dim]
        else:
            raise NotImplementedError(f'Unknown generation strategy: {self.motion_gen_strategy}')

        # Now, we should select neighbouring codes for each frame
        left_idx = (t / self.motion_z_distance).floor().long()  # [batch_size, num_frames]
        batch_idx = torch.arange(batch_size, device=t.device).unsqueeze(1).repeat(1,
                                                                                  num_frames)  # [batch_size, num_frames]
        motion_u_left = trajs[batch_idx, left_idx]  # [batch_size, num_frames, motion.z_dim]
        motion_u_right = trajs[batch_idx, left_idx + 1]  # [batch_size, num_frames, motion.z_dim]

        # Compute `u` codes as the interpolation between `u_left` and `u_right`
        t_left = t - t % self.motion_z_distance  # [batch_size, num_frames]
        t_right = t_left + self.motion_z_distance  # [batch_size, num_frames]
        # Compute interpolation weights `alpha` (we'll use them later)
        interp_weights = ((t % self.motion_z_distance) / self.motion_z_distance).unsqueeze(2).to(
            torch.float32)  # [batch_size, num_frames, 1]
        motion_u = motion_u_left * (
                    1 - interp_weights) + motion_u_right * interp_weights  # [batch_size, num_frames, motion.z_dim]
        motion_u = motion_u.view(batch_size * num_frames, motion_u.shape[2]).to(
            torch.float32)  # [batch_size * num_frames, motion.z_dim]

        # Save the results into the output dict
        out['motion_u_left'] = motion_u_left  # [batch_size, num_frames, motion.z_dim]
        out['motion_u_right'] = motion_u_right  # [batch_size, num_frames, motion.z_dim]
        out['t_left'] = t_left  # [batch_size, num_frames]
        out['t_right'] = t_right  # [batch_size, num_frames]
        out['interp_weights'] = interp_weights  # [batch_size, num_frames, 1]
        out['motion_u'] = motion_u  # [batch_size * num_frames, motion.z_dim]
        out['motion_z'] = motion_z  # [batch_size+, max_traj_len+, motion.z_dim+]

        return out

    def get_dim(self) -> int:
        return self.motion_v_dim if self.time_encoder is None else self.time_encoder.get_dim()

    def forward(self, t: torch.Tensor, motion_z: Dict = None) -> Dict:
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        out = {}  # We'll be aggregating the result here
        motion_u_info: Dict = self.generate_motion_u_codes(t, motion_z=motion_z)  # Dict of tensors
        motion_u = motion_u_info['motion_u'].view(t.shape[0] * t.shape[1],
                                                  -1)  # [batch_size * num_frames, motion.z_dim]

        # Compute the `v` motion codes
        if self.fourier:
            motion_v = self.time_encoder(
                t=t,
                motion_u_left=motion_u_info['motion_u_left'],
                motion_u_right=motion_u_info['motion_u_right'],
                t_left=motion_u_info['t_left'],
                t_right=motion_u_info['t_right'],
                interp_weights=motion_u_info['interp_weights'],
            )  # [batch_size * num_frames, motion_v_dim]
        else:
            motion_v = self.mapping(z=motion_u)  # [batch_size * num_frames, motion.v_dim]

        out['motion_v'] = motion_v  # [batch_size * num_frames, motion.v_dim]
        out['motion_z'] = motion_u_info['motion_z']  # (Any shape)

        return out


# ----------------------------------------------------------------------------

@persistence.persistent_class
class AlignedTimeEncoder(nn.Module):
    def __init__(self,
                 dim,
                 min_period_len,
                 max_period_len,
                 latent_dim: int = 512,
                 ):
        super().__init__()
        self.latent_dim = latent_dim

        freqs = construct_linspaced_frequencies(dim, min_period_len,
                                                max_period_len)
        self.register_buffer('freqs', freqs)  # [1, num_fourier_feats]

        # Creating the affine without bias to prevent motion mode collapse
        self.periods_predictor = FullyConnectedLayer(latent_dim, freqs.shape[1], activation='linear', bias=False)
        self.phase_predictor = FullyConnectedLayer(latent_dim, freqs.shape[1], activation='linear', bias=False)
        period_lens = 2 * np.pi / self.freqs  # [1, num_fourier_feats]
        phase_scales = max_period_len / period_lens  # [1, num_fourier_feats]
        self.register_buffer('phase_scales', phase_scales)

        self.aligners_predictor = FullyConnectedLayer(latent_dim, self.freqs.shape[1] * 2, activation='linear',
                                                      bias=False)

    def get_dim(self) -> int:
        return self.freqs.shape[1] * 2

    def forward(self, t: torch.Tensor, motion_u_left: torch.Tensor, motion_u_right: torch.Tensor,
                interp_weights: torch.Tensor, t_left: torch.Tensor, t_right: torch.Tensor):
        batch_size, num_frames, motion_u_dim = motion_u_left.shape  # [1], [1], [1]

        misc.assert_shape(t, [batch_size, num_frames])
        misc.assert_shape(motion_u_left, [batch_size, num_frames, None])
        misc.assert_shape(motion_u_right, [batch_size, num_frames, None])
        misc.assert_shape(interp_weights, [batch_size, num_frames, 1])
        assert t.shape == t_left.shape == t_right.shape, f"Wrong shape: {t.shape} vs {t_left.shape} vs {t_right.shape}"

        motion_u_left = motion_u_left.view(batch_size * num_frames,
                                           motion_u_dim)  # [batch_size * num_frames, motion_u_dim]
        motion_u_right = motion_u_right.view(batch_size * num_frames,
                                             motion_u_dim)  # [batch_size * num_frames, motion_u_dim]
        periods = self.periods_predictor(motion_u_left).tanh() + 1  # [batch_size * num_frames, feat_dim]
        phases = self.phase_predictor(motion_u_left)  # [batch_size * num_frames, feat_dim]
        aligners_left = self.aligners_predictor(motion_u_left)  # [batch_size * num_frames, out_dim]
        aligners_right = self.aligners_predictor(motion_u_right)  # [batch_size * num_frames, out_dim]

        raw_pos_embs = self.freqs * periods * t.view(-1).float().unsqueeze(
            1) + phases * self.phase_scales  # [bf, feat_dim]
        raw_pos_embs_left = self.freqs * periods * t_left.view(-1).float().unsqueeze(
            1) + phases * self.phase_scales  # [bf, feat_dim]
        raw_pos_embs_right = self.freqs * periods * t_right.view(-1).float().unsqueeze(
            1) + phases * self.phase_scales  # [bf, feat_dim]

        pos_embs = torch.cat([raw_pos_embs.sin(), raw_pos_embs.cos()], dim=1)  # [bf, out_dim]
        pos_embs_left = torch.cat([raw_pos_embs_left.sin(), raw_pos_embs_left.cos()], dim=1)  # [bf, out_dim]
        pos_embs_right = torch.cat([raw_pos_embs_right.sin(), raw_pos_embs_right.cos()], dim=1)  # [bf, out_dim]

        interp_weights = interp_weights.view(-1, 1)  # [bf, 1]
        aligners_remove = pos_embs_left * (1 - interp_weights) + pos_embs_right * interp_weights  # [bf, out_dim]
        aligners_add = aligners_left * (1 - interp_weights) + aligners_right * interp_weights  # [bf, out_dim]
        time_embs = pos_embs - aligners_remove + aligners_add  # [bf, out_dim]

        return time_embs


# ----------------------------------------------------------------------------

def construct_linspaced_frequencies(num_freqs: int, min_period_len: int, max_period_len: int) -> torch.Tensor:
    freqs = 2 * np.pi / (2 ** np.linspace(np.log2(min_period_len), np.log2(max_period_len), num_freqs))  # [num_freqs]
    freqs = torch.from_numpy(freqs[::-1].copy().astype(np.float32)).unsqueeze(0)  # [1, num_freqs]

    return freqs

# ----------------------------------------------------------------------------
