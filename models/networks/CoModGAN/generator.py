# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import importlib
import numpy as np
import torch
from .torch_utils import misc
from .torch_utils import persistence
from .torch_utils.ops import upfirdn2d
from .layers import SynthesisBlock, CASynthesisBlock, MappingNetwork, FullyConnectedLayer, TemporalDifferenceEncoder, Conv2dLayer, E_block, E_fromrgb, get_mapper, PixelShuffle
from .motion import MotionMappingNetwork


#----------------------------------------------------------------------------


def get_synthesizer(name, **kwargs):
    m = importlib.import_module('models.networks.CoModGAN.generator')
    clazz = getattr(m, name)
    return clazz(**kwargs)


#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels_in,               # Number of color channels.
        img_channels_out,
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        activation = 'lrelu',
        resample_filter=[1,3,3,1],
        dropout_rate=0.5,
        skip_resolution=256,
        channel_attention=False,
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels_in = img_channels_in
        self.img_channels_out = img_channels_out
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        self.channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in range(self.img_resolution_log2, 2, -1):
            if res == self.img_resolution_log2:
                self.e_fromrgb = E_fromrgb(in_channels=self.img_channels_in, out_channels=self.channels_dict[2**res], kernel_size=1, activation='lrelu', conv_clamp=None)
            e_block = E_block(res=res, tmp_channels=self.channels_dict[2**res], out_channels=self.channels_dict[2**res//2], kernel_size=3, activation='lrelu', conv_clamp=None, resample_filter=resample_filter)
            setattr(self, f'e_b{res}', e_block)
        
        self.e_4x4 = Conv2dLayer(self.channels_dict[4], self.channels_dict[4], kernel_size=3, activation=activation, conv_clamp=None)
        self.fc_in = FullyConnectedLayer(self.channels_dict[4] * (4 ** 2), self.channels_dict[4]*2, activation=activation)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc_out = FullyConnectedLayer(self.channels_dict[4]*2, self.channels_dict[4] * (4 ** 2), activation=activation)
        self.block_early = SynthesisBlock(0, self.channels_dict[4], w_dim=w_dim, global_w_dim=self.channels_dict[4]*2, resolution=4,
                img_channels=img_channels_out, is_last=False, use_fp16=False, channel_attention=channel_attention, **block_kwargs)
        self.num_ws += self.block_early.num_conv
        
        for res in self.block_resolutions[1:]:
            in_channels = self.channels_dict[res // 2] if res > 4 else 0
            out_channels = self.channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, global_w_dim=self.channels_dict[4]*2, resolution=res,
                img_channels=img_channels_out, is_last=is_last, use_fp16=use_fp16, channel_attention=channel_attention, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

        if skip_resolution >= 4:
            final_skip = int(np.log2(skip_resolution))
            self.skip_connects = [True for i in range(2, final_skip + 1)] + [False for i in range(final_skip + 1, self.img_resolution_log2 + 1)]
        else:
            self.skip_connects = [False for i in range(self.img_resolution_log2)]

    def forward(self, ws, img_in, **block_kwargs):
        block_ws = []
        E_features = {}
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            block_ws.append(ws.narrow(1, w_idx, self.block_early.num_conv + self.block_early.num_torgb))
            w_idx += self.block_early.num_conv
            for res in self.block_resolutions[1:]:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = None
        for res in range(self.img_resolution_log2, 2, -1):
            if res == self.img_resolution_log2:
                img_in = self.e_fromrgb(x, img_in)
            block = getattr(self, f'e_b{res}')
            img_in = block(img_in, E_features)
    
        img_in = self.e_4x4(img_in)
        E_features[2**2] = img_in
        img_in = self.fc_in(img_in.flatten(1))
        img_in = self.dropout(img_in)
        img_global = img_in
        img_in = self.fc_out(img_in)
        img_in = torch.reshape(img_in, (-1, self.channels_dict[4], 4, 4))
        img_in = img_in + E_features[2**2] if self.skip_connects[0] else img_in
        
        x, img = self.block_early(img_in, None, block_ws[0], img_global, **block_kwargs)

        for res, cur_ws, skip in zip(self.block_resolutions[1:], block_ws[1:], self.skip_connects[1:]):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, img_global, E_features, skip,  **block_kwargs)
        return img


# ----------------------------------------------------------------------------

@persistence.persistent_class
class MotionSynthesisNetwork(torch.nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.
                 motion_v_dim,
                 img_resolution,  # Output image resolution.
                 img_channels_in,  # Number of color channels.
                 img_channels_out,
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=0,  # Use FP16 for the N highest resolutions.
                 activation='lrelu',
                 resample_filter=[1, 3, 3, 1],
                 dropout_rate=0.5,
                 skip_resolution=256,
                 global_enc=False,
                 relative_enc=False,
                 coord_emb=True,
                 comod_emb=False,
                 max_num_frames=128,
                 **block_kwargs,  # Arguments for SynthesisBlock.
                 ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels_in = img_channels_in
        self.img_channels_out = img_channels_out
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        self.channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        self.relative_enc = relative_enc
        self.global_enc = global_enc
        self.comod_emb = comod_emb
        self.coord_emb = coord_emb
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in range(self.img_resolution_log2, 2, -1):
            if res == self.img_resolution_log2:
                self.e_fromrgb = E_fromrgb(in_channels=self.img_channels_in, out_channels=self.channels_dict[2 ** res],
                                           kernel_size=1, activation='lrelu', conv_clamp=None)
            e_block = E_block(res=res, tmp_channels=self.channels_dict[2 ** res],
                              out_channels=self.channels_dict[2 ** res // 2], kernel_size=3, activation='lrelu',
                              conv_clamp=None, resample_filter=resample_filter)
            setattr(self, f'e_b{res}', e_block)

        self.e_4x4 = Conv2dLayer(self.channels_dict[4], self.channels_dict[4], kernel_size=3, activation=activation,
                                 conv_clamp=None)
        self.fc_in = FullyConnectedLayer(self.channels_dict[4] * (4 ** 2), self.channels_dict[4] * 2,
                                         activation=activation)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc_out = FullyConnectedLayer(self.channels_dict[4] * 2, self.channels_dict[4] * (4 ** 2),
                                          activation=activation)
        early_channels = self.channels_dict[4]

        if coord_emb:
            if self.global_enc:
                early_channels += motion_v_dim
            if self.relative_enc:
                self.time_encoder = TemporalDifferenceEncoder(max_num_frames=max_num_frames, num_frames_per_video=1)
                early_channels += self.time_encoder.get_dim()
            else:
                early_channels += 1
        global_w_dim = self.channels_dict[4] + early_channels if comod_emb else self.channels_dict[4] * 2
        if not coord_emb and comod_emb:
            global_w_dim += motion_v_dim
            # self.time_encoder = TemporalDifferenceEncoder(max_num_frames=max_num_frames, num_frames_per_video=1)
            # global_w_dim += self.time_encoder.get_dim()
        self.block_early = SynthesisBlock(0, self.channels_dict[4],
                                          w_dim=w_dim,
                                          global_w_dim=global_w_dim,
                                          resolution=4,
                                          img_channels=img_channels_out, is_last=False, use_fp16=False,
                                          early_channels=early_channels,
                                          **block_kwargs)
        self.num_ws += self.block_early.num_conv

        for res in self.block_resolutions[1:]:
            in_channels = self.channels_dict[res // 2] if res > 4 else 0
            out_channels = self.channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, global_w_dim=global_w_dim,
                                   resolution=res,
                                   img_channels=img_channels_out, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

        if skip_resolution >= 4:
            final_skip = int(np.log2(skip_resolution))
            self.skip_connects = [True for i in range(2, final_skip + 1)] + [False for i in range(final_skip + 1,
                                                                                                  self.img_resolution_log2 + 1)]
        else:
            self.skip_connects = [False for i in range(self.img_resolution_log2)]

    def forward(self, ws, img_in, motion, delta_t, **block_kwargs):
        block_ws = []
        E_features = {}
        if self.global_enc:
            motion_v = motion['motion_v']
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            block_ws.append(ws.narrow(1, w_idx, self.block_early.num_conv + self.block_early.num_torgb))
            w_idx += self.block_early.num_conv
            for res in self.block_resolutions[1:]:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = None
        for res in range(self.img_resolution_log2, 2, -1):
            if res == self.img_resolution_log2:
                img_in = self.e_fromrgb(x, img_in)
            block = getattr(self, f'e_b{res}')
            img_in = block(img_in, E_features)

        img_in = self.e_4x4(img_in)
        E_features[2 ** 2] = img_in
        img_in = self.fc_in(img_in.flatten(1))
        img_in = self.dropout(img_in)
        img_global = img_in
        img_in = self.fc_out(img_in)
        img_in = torch.reshape(img_in, (-1, self.channels_dict[4], 4, 4))
        img_in = img_in + E_features[2 ** 2] if self.skip_connects[0] else img_in
        position_emb = []

        if self.global_enc:
            position_emb.append(motion_v)
        if self.relative_enc:
            t_embs = self.time_encoder(delta_t)
            position_emb.append(t_embs)
        else:
            position_emb.append(delta_t)
        position_emb = torch.cat(position_emb, -1)

        if self.coord_emb:
            if self.global_enc:
                img_in = torch.cat([
                    img_in,
                    motion_v.unsqueeze(2).unsqueeze(3).repeat(1, 1, *img_in.shape[2:]),
                ], dim=1)
            if self.relative_enc:
                img_in = torch.cat([
                    img_in,
                    t_embs.unsqueeze(2).unsqueeze(3).repeat(1, 1, *img_in.shape[2:]),
                ], dim=1)
            else:
                img_in = torch.cat([
                    img_in,
                    delta_t.unsqueeze(2).unsqueeze(3).repeat(1, 1, *img_in.shape[2:]),
                ], dim=1)

        if self.comod_emb:
            img_global = torch.cat([img_global, position_emb], -1)

        x, img = self.block_early(img_in, None, block_ws[0], img_global, **block_kwargs)

        for res, cur_ws, skip in zip(self.block_resolutions[1:], block_ws[1:], self.skip_connects[1:]):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, img_global, E_features, skip, **block_kwargs)
        return img

# ----------------------------------------------------------------------------


@persistence.persistent_class
class PixShuffleEarlyBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        self.in_channels = in_channels
        self.n_feats = 4**depth
        self.shuffler_en = PixelShuffle(1 / 2 ** depth)
        self.headConv = torch.nn.Conv2d(in_channels * self.n_feats, out_channels, kernel_size=3, padding=1)

    def forward(self, input):
        x = []
        for img_idx in range(self.in_channels):
            x.append(self.shuffler_en(input[:, img_idx:img_idx + 1, ...]))
        x = torch.cat(x, 1)
        x = self.headConv(x)
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class PixShuffleTailBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        self.n_feats = 4**depth
        self.tailConv = torch.nn.Conv2d(in_channels, self.n_feats * out_channels, kernel_size=3, padding=1)
        self.shuffler_de = PixelShuffle(2 ** depth)

    def forward(self, out):
        out = self.tailConv(out)
        out = self.shuffler_de(out)
        return out


# ----------------------------------------------------------------------------


@persistence.persistent_class
class CASynthesisNetwork(torch.nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output image resolution.
                 img_channels_in,  # Number of color channels.
                 img_channels_out,
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=0,  # Use FP16 for the N highest resolutions.
                 activation='lrelu',
                 resample_filter=[1, 3, 3, 1],
                 dropout_rate=0.5,
                 skip_resolution=256,
                 **block_kwargs,  # Arguments for SynthesisBlock.
                 ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels_in = img_channels_in
        inter_img_channels_out = 4 ** 2
        self.img_channels_out = img_channels_out
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        self.channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        self.e_fromrgb = PixShuffleEarlyBlock(in_channels=self.img_channels_in,
                                              out_channels=self.channels_dict[self.img_resolution], depth=1)
        for res in range(self.img_resolution_log2, 2, -1):
            e_block = E_block(res=res, tmp_channels=self.channels_dict[2 ** res],
                              out_channels=self.channels_dict[2 ** res // 2], kernel_size=3, activation='lrelu',
                              conv_clamp=None, resample_filter=resample_filter, channel_attention=True)
            setattr(self, f'e_b{res}', e_block)

        self.e_4x4 = Conv2dLayer(self.channels_dict[4], self.channels_dict[4], kernel_size=3, activation=activation,
                                 conv_clamp=None)
        self.fc_in = FullyConnectedLayer(self.channels_dict[4] * (4 ** 2), self.channels_dict[4] * 2,
                                         activation=activation)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc_out = FullyConnectedLayer(self.channels_dict[4] * 2, self.channels_dict[4] * (4 ** 2),
                                          activation=activation)
        self.block_early = SynthesisBlock(0, self.channels_dict[4], w_dim=w_dim, global_w_dim=self.channels_dict[4] * 2,
                                          resolution=4,
                                          img_channels=inter_img_channels_out, is_last=False, use_fp16=False, **block_kwargs)
        self.num_ws += self.block_early.num_conv

        for res in self.block_resolutions[1:]:
            in_channels = self.channels_dict[res // 2] if res > 4 else 0
            out_channels = self.channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, global_w_dim=self.channels_dict[4] * 2,
                                   resolution=res,
                                   img_channels=inter_img_channels_out, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

        self.d_torgb = PixShuffleTailBlock(in_channels=inter_img_channels_out, out_channels=img_channels_out, depth=1)

        if skip_resolution >= 4:
            final_skip = int(np.log2(skip_resolution))
            self.skip_connects = [True for i in range(2, final_skip + 1)] + [False for i in range(final_skip + 1,
                                                                                                  self.img_resolution_log2 + 1)]
        else:
            self.skip_connects = [False for i in range(self.img_resolution_log2)]

    def forward(self, ws, img_in, **block_kwargs):
        block_ws = []
        E_features = {}
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            block_ws.append(ws.narrow(1, w_idx, self.block_early.num_conv + self.block_early.num_torgb))
            w_idx += self.block_early.num_conv
            for res in self.block_resolutions[1:]:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        img_in = self.e_fromrgb(img_in)
        for res in range(self.img_resolution_log2, 2, -1):
            block = getattr(self, f'e_b{res}')
            img_in = block(img_in, E_features)

        img_in = self.e_4x4(img_in)
        E_features[2 ** 2] = img_in
        img_in = self.fc_in(img_in.flatten(1))
        img_in = self.dropout(img_in)
        img_global = img_in
        img_in = self.fc_out(img_in)
        img_in = torch.reshape(img_in, (-1, self.channels_dict[4], 4, 4))
        img_in = img_in + E_features[2 ** 2] if self.skip_connects[0] else img_in

        x, img = self.block_early(img_in, None, block_ws[0], img_global, **block_kwargs)

        for res, cur_ws, skip in zip(self.block_resolutions[1:], block_ws[1:], self.skip_connects[1:]):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, img_global, E_features, skip, **block_kwargs)
        img = self.d_torgb(img)
        return img


# ----------------------------------------------------------------------------

@persistence.persistent_class
class CAINSynthesisNetwork(torch.nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output image resolution.
                 img_channels_in,  # Number of color channels.
                 img_channels_out,
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=0,  # Use FP16 for the N highest resolutions.
                 activation='lrelu',
                 resample_filter=[1, 3, 3, 1],
                 dropout_rate=0.5,
                 skip_resolution=256,
                 depth=3,
                 n_resgroups=5,
                 n_resblocks=12,
                 reduction=16,
                 **block_kwargs,  # Arguments for SynthesisBlock.
                 ):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels_in = img_channels_in
        self.img_channels_out = img_channels_out
        n_feats = 4**depth
        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        self.shuffler_en = PixelShuffle(1 / 2 ** depth)
        self.headConv = torch.nn.Conv2d(img_channels_in * n_feats, n_feats, kernel_size=3, padding=1)
        self.num_ws = 0

        for group_idx in range(n_resgroups):
            for block_idx in range(n_resblocks):
                in_channels = n_feats
                out_channels = n_feats
                use_fp16 = False
                is_last = False
                block = CASynthesisBlock(in_channels, out_channels, w_dim=w_dim, global_w_dim=0,
                                       resolution=int(img_resolution / (2 ** depth)), reduction=reduction,
                                       img_channels=img_channels_out, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
                self.num_ws += block.num_conv
                setattr(self, f'b{group_idx}_{block_idx}', block)

        '''
        modules_body = [
        ResidualGroup(
            RCAB,
            n_resblocks=n_resblocks,
            n_feat=n_feats,
            kernel_size=3,
            reduction=reduction,
            act=activation,
            norm=False)
        for _ in range(n_resgroups)]
        self.body = torch.nn.Sequential(*modules_body)
        '''

        self.tailConv = torch.nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.shuffler_de = PixelShuffle(2 ** depth)

    def forward(self, ws, img_in, **block_kwargs):
        x = []
        for img_idx in range(self.img_channels_in):
            x.append(self.shuffler_en(img_in[:, img_idx:img_idx+1, ...]))
        x = torch.cat(x, 1)
        x = self.headConv(x)
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for group_idx in range(self.n_resgroups):
                for block_idx in range(self.n_resblocks):
                    block = getattr(self, f'b{group_idx}_{block_idx}')
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    w_idx += block.num_conv

        ws_idx = 0
        global_res = x
        for group_idx in range(self.n_resgroups):
            res = x
            for block_idx in range(self.n_resblocks):
                block = getattr(self, f'b{group_idx}_{block_idx}')
                x, _ = block(x, None, block_ws[ws_idx], None, None, False, **block_kwargs)
                ws_idx += 1
            x = x + res

        x = x + global_res
        out = self.tailConv(x)
        out = torch.tanh(out)
        out = self.shuffler_de(out)

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class CoModGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels_in,               # Number of output color channels.
        img_channels_out,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels_in = img_channels_in
        self.img_channels_out = img_channels_out

        self.synthesis = get_synthesizer(w_dim=w_dim, img_resolution=img_resolution, img_channels_in=img_channels_in, img_channels_out=img_channels_out, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = get_mapper(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, cond_img, ref_img=None, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z=z, c=c, truncation_psi=truncation_psi, img_in=ref_img, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, cond_img, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------


@persistence.persistent_class
class StyleGANVGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        motion_v_dim,
        img_resolution,             # Output resolution.
        img_channels_in,               # Number of output color channels.
        img_channels_out,
        max_num_frames      = 8,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        motion_mapping_kwargs = {},     # Arguments for MotionMappingNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels_in = img_channels_in
        self.img_channels_out = img_channels_out
        self.synthesis = get_synthesizer(w_dim=w_dim, motion_v_dim=motion_v_dim, img_resolution=img_resolution, img_channels_in=img_channels_in,
                                         img_channels_out=img_channels_out, max_num_frames=max_num_frames, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = get_mapper(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        self.motion_mapping = MotionMappingNetwork(motion_v_dim=motion_v_dim, max_num_frames=max_num_frames, **motion_mapping_kwargs)

    def forward(self, z, c, t, delta_t, cond_img, motion_z=None, ref_img=None, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z=z, c=c, truncation_psi=truncation_psi, img_in=ref_img, truncation_cutoff=truncation_cutoff)
        motion_info = self.motion_mapping(t, motion_z=motion_z)
        img = self.synthesis(ws, cond_img, motion_info, delta_t, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class CoModDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        **kwargs
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class StyleGANVDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        num_frames          = 3,
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        num_frames_div_factor=4,  # Divide the channel dimensionality
        max_num_frames      = 128,
        concat_res          = 16,
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()

        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_frames = num_frames
        self.concat_res = concat_res
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]

        self.time_encoder = TemporalDifferenceEncoder(max_num_frames=max_num_frames, num_frames_per_video=1)
        assert self.time_encoder.get_dim() > 0

        if self.c_dim == 0 and self.time_encoder is None:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        total_c_dim = c_dim + (0 if self.time_encoder is None else self.time_encoder.get_dim())
        cur_layer_idx = 0

        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]

            if res // 2 == concat_res:
                out_channels = out_channels // num_frames_div_factor
            if res == concat_res:
                in_channels = tmp_channels = (in_channels // num_frames_div_factor) * self.num_frames

            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        if self.c_dim > 0 or not self.time_encoder is None:
            self.mapping = MappingNetwork(z_dim=0, c_dim=total_c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, delta_t, **block_kwargs):
        if not self.time_encoder is None:
            # Encoding the time distances
            t_embs = self.time_encoder(delta_t) # [batch_size, t_dim]

            # Concatenate `c` and time embeddings
            c = torch.cat([c, t_embs], dim=1) # [batch_size, c_dim + t_dim]

        img = img.view(-1, self.img_channels, *img.shape[2:]) # [batch_size * num_frames, c, h, w]
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            if res == self.concat_res:
                # Concatenating the frames
                x = x.view(-1, self.num_frames, *x.shape[1:]) # [batch_size, num_frames, c, h, w]
                x = x.view(x.shape[0], -1, *x.shape[3:]) # [batch_size, num_frames * c, h, w]
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0 or not self.time_encoder is None:
            assert c.shape[1] > 0
        if c.shape[1] > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)

        return x
