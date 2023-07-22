import torch
import numpy as np
import importlib

import torch.nn.functional as F
from .torch_utils.ops import conv2d_resample
from .torch_utils import misc
from .torch_utils import persistence
from .torch_utils.ops import fma
from .torch_utils.ops import upfirdn2d
from .torch_utils.ops import bias_act


@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()
#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


#----------------------------------------------------------------------------

@persistence.persistent_class
class EqLRConv1d(torch.nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        padding: int=0,
        stride: int=1,
        activation: str='linear',
        lr_multiplier: float=1.0,
        bias=True,
        bias_init=0.0,
    ):
        super().__init__()

        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features, kernel_size]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], float(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features * kernel_size)
        self.bias_gain = lr_multiplier
        self.padding = padding
        self.stride = stride

        assert self.activation in ['lrelu', 'linear']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"Wrong shape: {x.shape}"

        w = self.weight.to(x.dtype) * self.weight_gain # [out_features, in_features, kernel_size]
        b = self.bias # [out_features]
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        y = F.conv1d(input=x, weight=w, bias=b, stride=self.stride, padding=self.padding) # [batch_size, out_features, out_len]
        if self.activation == 'linear':
            pass
        elif self.activation == 'lrelu':
            y = F.leaky_relu(y, negative_slope=0.2) # [batch_size, out_features, out_len]
        else:
            raise NotImplementedError
        return y

#----------------------------------------------------------------------------


@persistence.persistent_class
class E_fromrgb(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, activation='lrelu', conv_clamp=None):
        super().__init__()
        self.con_layer = Conv2dLayer(in_channels, out_channels, kernel_size=1, activation=activation,
                trainable=True, conv_clamp=conv_clamp, channels_last=False)

    def forward(self, x, y):
        t=self.con_layer(y)
        return t if x is None else x + t

#----------------------------------------------------------------------------


@persistence.persistent_class
class E_block(torch.nn.Module):
    def __init__(self, res, tmp_channels, out_channels, kernel_size=3, activation='lrelu', conv_clamp=None, resample_filter=[1,3,3,1], channel_attention=False):
        super().__init__()
        self.res=res
        self.channel_attention = channel_attention
        self.conv_layer0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=kernel_size, activation=activation,
            trainable=True, conv_clamp=conv_clamp, channels_last=False)
        self.conv_layer1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=kernel_size, activation=activation, down=2,
            trainable=True, resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=False)
        if channel_attention:
            self.ca = CALayer(out_channels, reduction=16)

    def forward(self, x, E_features):
        x=self.conv_layer0(x)
        E_features[2**self.res]=x
        x=self.conv_layer1(x)
        if self.channel_attention:
            x, _ = self.ca(x)

        return x


#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

#----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        global_w_dim,
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        up                  = 2,
        to_rgb              = True,
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        cond_mod            = False,
        early_channels      = 0,
        channel_attention   = False,
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0
        self.cond_mod = cond_mod
        self.channel_attention = channel_attention

        if not cond_mod:
            global_w_dim = 0

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim+global_w_dim, resolution=resolution, up=up,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        if early_channels > 0:
            self.conv1 = SynthesisLayer(early_channels, out_channels, w_dim=w_dim + global_w_dim, resolution=resolution,
                                        conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        else:
            self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim+global_w_dim, resolution=resolution,
                conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if to_rgb and (is_last or architecture == 'skip'):
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim+global_w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=up,
                resample_filter=resample_filter, channels_last=self.channels_last)

        if channel_attention:
            self.ca = CALayer(out_channels, reduction=16)

    def forward(self, x, img, ws, global_w, E_features=None, include_skip=True, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if E_features is not None:
            x_skip = E_features[self.resolution].to(dtype=dtype, memory_format=memory_format)
        else:
            x_skip = 0
        if self.in_channels != 0:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        if self.cond_mod:
            mod_vector = torch.cat((next(w_iter), global_w),1)
        else:
            mod_vector = next(w_iter)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, mod_vector, fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, mod_vector, fused_modconv=fused_modconv, **layer_kwargs)
            if include_skip:
                x = x + x_skip
            x = self.conv1(x, mod_vector, fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            if self.channel_attention:
                x, _ = self.ca(x)
            x = y.add_(x)
        else:
            x = self.conv0(x, mod_vector, fused_modconv=fused_modconv, **layer_kwargs)
            if include_skip:
                x = x + x_skip
            x = self.conv1(x, mod_vector, fused_modconv=fused_modconv, **layer_kwargs)
            if self.channel_attention:
                x, _ = self.ca(x)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, mod_vector, fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

#----------------------------------------------------------------------------


class CASynthesisBlock(SynthesisBlock):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        global_w_dim,
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        up                  = 1,
        to_rgb              = False,
        architecture        = 'resnet',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        cond_mod            = False,
        early_channels      = 0,
        reduction           = 16,
        **layer_kwargs,                     # Arguments for SynthesisLayer.
        ):
        super(CASynthesisBlock, self).__init__(in_channels, out_channels, w_dim, global_w_dim, resolution, img_channels,
                                               is_last, up, to_rgb, architecture, resample_filter, conv_clamp, use_fp16,
                                               fp16_channels_last, cond_mod, early_channels, **layer_kwargs)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, x, img, ws, global_w, E_features=None, include_skip=False, force_fp32=False, fused_modconv=None,
                **layer_kwargs):
        assert img is None
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings():  # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if E_features is not None:
            x_skip = E_features[self.resolution].to(dtype=dtype, memory_format=memory_format)
        else:
            x_skip = 0
        if self.in_channels != 0:
            x = x.to(dtype=dtype, memory_format=memory_format)

        if self.cond_mod:
            mod_vector = torch.cat((next(w_iter), global_w), 1)
        else:
            mod_vector = next(w_iter)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, mod_vector, fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, mod_vector, fused_modconv=fused_modconv, **layer_kwargs)
            if include_skip:
                x = x + x_skip
            x = self.conv1(x, mod_vector, fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x, _ = self.ca(x)
            x = y.add_(x)
        else:
            y = x
            x = self.conv0(x, mod_vector, fused_modconv=fused_modconv, **layer_kwargs)
            if include_skip:
                x = x + x_skip
            x = self.conv1(x, mod_vector, fused_modconv=fused_modconv, **layer_kwargs)
            x, _ = self.ca(x)
            x = x + y

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------


def get_mapper(name, **kwargs):
    m = importlib.import_module('models.networks.CoModGAN.layers')
    clazz = getattr(m, name)
    return clazz(**kwargs)


#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
        **kwargs
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False, **kwargs):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x


class RefMappingNetwork(torch.nn.Module):
    def __init__(self,
                 img_resolution,
                 img_channels,
                 w_dim,
                 num_ws,
                 channel_base=32768,
                 channel_max=512,
                 activation='lrelu',
                 resample_filter=[1,3,3,1],
                 **kwargs,
                 ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_ws = num_ws
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        self.channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        for res in range(self.img_resolution_log2, 2, -1):
            if res == self.img_resolution_log2:
                self.e_fromrgb = E_fromrgb(in_channels=self.img_channels, out_channels=self.channels_dict[2 ** res],
                                           kernel_size=1, activation='lrelu', conv_clamp=None)
            e_block = E_block(res=res, tmp_channels=self.channels_dict[2 ** res],
                              out_channels=self.channels_dict[2 ** res // 2], kernel_size=3, activation='lrelu',
                              conv_clamp=None, resample_filter=resample_filter)
            setattr(self, f'e_b{res}', e_block)

        self.e_4x4 = Conv2dLayer(self.channels_dict[4], self.channels_dict[4], kernel_size=3, activation=activation,
                                 conv_clamp=None)
        self.fc_in = FullyConnectedLayer(self.channels_dict[4] * (4 ** 2), w_dim, activation=activation)

    def forward(self, img_in, **kwargs):
        E_features = {}
        x = None
        for res in range(self.img_resolution_log2, 2, -1):
            if res == self.img_resolution_log2:
                img_in = self.e_fromrgb(x, img_in)
            block = getattr(self, f'e_b{res}')
            img_in = block(img_in, E_features)
        img_in = self.e_4x4(img_in)
        x = self.fc_in(img_in.flatten(1))
        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
        return x

#----------------------------------------------------------------------------


def construct_log_spaced_freqs(max_num_frames: int, skip_small_t_freqs: int=0):
    time_resolution = 2 ** np.ceil(np.log2(max_num_frames))
    num_fourier_feats = np.ceil(np.log2(time_resolution)).astype(int)
    powers = torch.tensor([2]).repeat(num_fourier_feats).pow(torch.arange(num_fourier_feats)) # [num_fourier_feats]
    powers = powers[:len(powers) - skip_small_t_freqs] # [num_fourier_feats]
    fourier_coefs = powers.unsqueeze(0).float() * np.pi # [1, num_fourier_feats]

    return fourier_coefs / time_resolution

#----------------------------------------------------------------------------


@persistence.persistent_class
class FixedTimeEncoder(torch.nn.Module):
    def __init__(self,
            max_num_frames: int,            # Maximum T size
            skip_small_t_freqs: int=0,      # How many high frequencies we should skip
        ):
        super().__init__()

        assert max_num_frames >= 1, f"Wrong max_num_frames: {max_num_frames}"
        fourier_coefs = construct_log_spaced_freqs(max_num_frames, skip_small_t_freqs=skip_small_t_freqs)
        self.register_buffer('fourier_coefs', fourier_coefs) # [1, num_fourier_feats]

    def get_dim(self) -> int:
        return self.fourier_coefs.shape[1] * 2

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        t = t.view(-1).float() # [batch_size * num_frames]
        fourier_raw_embs = self.fourier_coefs * t.unsqueeze(1) # [bf, num_fourier_feats]

        fourier_embs = torch.cat([
            fourier_raw_embs.sin(),
            fourier_raw_embs.cos(),
        ], dim=1) # [bf, num_fourier_feats * 2]

        return fourier_embs

#----------------------------------------------------------------------------


class TemporalDifferenceEncoder(torch.nn.Module):
    def __init__(self,
                 max_num_frames,
                 num_frames_per_video
                 ):
        super().__init__()

        self.d = 256
        self.num_frames_per_video = num_frames_per_video
        self.const_embed = torch.nn.Embedding(max_num_frames, self.d)
        self.time_encoder = FixedTimeEncoder(max_num_frames, skip_small_t_freqs=0)

    def get_dim(self) -> int:
        return self.d + self.time_encoder.get_dim()

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        misc.assert_shape(delta_t, [None, self.num_frames_per_video])

        batch_size = delta_t.shape[0]

        t_diffs = delta_t.view(-1)
        # Note: float => round => long is necessary when it's originally long
        const_embs = self.const_embed(t_diffs.float().round().long()) # [batch_size * num_diffs_to_use, d]
        fourier_embs = self.time_encoder(t_diffs.unsqueeze(1)) # [batch_size * num_diffs_to_use, num_fourier_feats]
        out = torch.cat([const_embs, fourier_embs], dim=1) # [batch_size * num_diffs_to_use, d + num_fourier_feats]
        out = out.view(batch_size, -1) # [batch_size, num_diffs_to_use * (d + num_fourier_feats)]

        return out


def pixel_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(batch_size, out_channels, scale_factor, scale_factor, in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(batch_size, channels, out_height, block_size, out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)


class PixelShuffle(torch.nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return pixel_shuffle(x, self.scale_factor)

    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv = torch.nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)

        self.norm = norm
        if norm == 'IN':
            self.norm = torch.nn.InstanceNorm2d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = torch.nn.BatchNorm2d(out_feat)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        return out


## Channel Attention (CA) Layer
class CALayer(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = torch.nn.Sequential(
            torch.nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y, y


class RCAB(torch.nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction, bias=True,
            norm=False, act=torch.nn.ReLU(True), downscale=False, return_ca=False):
        super(RCAB, self).__init__()

        self.body = torch.nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=2 if downscale else 1, norm=norm),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
            CALayer(out_feat, reduction)
        )
        self.downscale = downscale
        if downscale:
            self.downConv = torch.nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1)
        self.return_ca = return_ca

    def forward(self, x):
        res = x
        out, ca = self.body(x)
        if self.downscale:
            res = self.downConv(res)
        out = res.add_(out)

        if self.return_ca:
            return out, ca
        else:
            return out


## Residual Group (RG)
class ResidualGroup(torch.nn.Module):
    def __init__(self, Block, n_resblocks, n_feat, kernel_size, reduction, act, norm=False):
        super(ResidualGroup, self).__init__()

        modules_body = [Block(n_feat, n_feat, kernel_size, reduction, bias=True, norm=norm, act=act)
            for _ in range(n_resblocks)]
        modules_body.append(ConvNorm(n_feat, n_feat, kernel_size, stride=1, norm=norm))
        self.body = torch.nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = x.add_(res)
        return res
