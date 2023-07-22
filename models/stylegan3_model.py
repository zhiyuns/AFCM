import torch
import numpy as np
from .comodgan_model import CoModGANModel, CN
from models.networks.stylegan3.torch_utils.ops import upfirdn2d
from .networks.CoModGAN.torch_utils import misc

class StyleGAN3Model(CoModGANModel):
    def __init__(self, opt):
        super(StyleGAN3Model, self).__init__(opt)
        self.modality_list = opt.loaders.raw_internal_path_out
        self.blur_sigma = 0

    def run_G(self, cond_img, update_emas=False, noise_mode='random'):
        ref_img = self.extra_B if self.extra_b else self.real_B
        ws = self.G_mapping(z=self.gen_z, c=self.gen_c, img_in=ref_img, update_emas=False)
        if self.style_mixing_prob > 0:
            cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                 torch.full_like(cutoff, ws.shape[1]))
            ws[:, cutoff:] = self.G_mapping(z=torch.randn_like(self.gen_z), c=self.gen_c, img_in=ref_img, skip_w_avg_update=True)[:, cutoff:]
        img = self.G_synthesis(ws, cond_img, update_emas=False, noise_mode=noise_mode)
        return img

    def run_D(self, img, **kwargs):
        blur_size = np.floor(self.blur_sigma * 3)
        if blur_size > 0:
            f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(self.blur_sigma).square().neg().exp2()
            img = upfirdn2d.filter2d(img, f / f.sum())

        return self.netD(img, **kwargs)

    @staticmethod
    def modify_commandline_options(config, is_train=True):
        _C = config
        _C.loaders.slice_num = 3

        _C.model.G.z_dim = 512
        _C.model.G.w_dim = 512
        _C.model.G.c_dim = 1
        _C.model.G.img_resolution = 256
        _C.model.G.img_channels_in = 1
        _C.model.G.img_channels_out = 1

        _C.model.G.synthesis_kwargs = CN()
        _C.model.G.synthesis_kwargs.channel_base = int(0.5 * 32768)
        _C.model.G.synthesis_kwargs.channel_max = 512
        _C.model.G.synthesis_kwargs.num_layers = 14
        _C.model.G.synthesis_kwargs.num_critical = 2
        _C.model.G.synthesis_kwargs.first_cutoff = 2
        _C.model.G.synthesis_kwargs.first_stopband = 2**2.1
        _C.model.G.synthesis_kwargs.last_stopband_rel = 2**0.3
        _C.model.G.synthesis_kwargs.margin_size = 10
        _C.model.G.synthesis_kwargs.output_scale = 0.25
        _C.model.G.synthesis_kwargs.skip_resolution = 128
        # layer kwargs
        _C.model.G.synthesis_kwargs.conv_kernel = 3
        _C.model.G.synthesis_kwargs.filter_size = 6
        _C.model.G.synthesis_kwargs.lrelu_upsampling = 2
        _C.model.G.synthesis_kwargs.use_radial_filters = False
        _C.model.G.synthesis_kwargs.conv_clamp = 256
        _C.model.G.synthesis_kwargs.magnitude_ema_beta = 0.5 ** (16 / (20 * 1e3))  # depend on bs
        _C.model.G.synthesis_kwargs.cond_mod = True

        _C.model.G.mapping_kwargs = CN()
        _C.model.G.mapping_kwargs.num_layers = 8

        _C.model.D.channel_base = int(0.5 * 32768)
        _C.model.D.num_fp16_res = 0
        _C.model.D.conv_clamp = None
        _C.model.D.channel_max = 512
        _C.model.D.c_dim = 0
        _C.model.D.img_resolution = 256
        _C.model.D.img_channels = 2

        _C.model.D.mapping_kwargs = CN()
        _C.model.D.mapping_kwargs.num_layers = 8
        _C.model.D.epilogue_kwargs = CN()
        _C.model.D.epilogue_kwargs.mbstd_group_size = 16

        _C.loss.blur_init_sigma = 0
        _C.loss.blur_fade_kimg = 0  # depend on bs

        return _C

    def forward(self, update_emas=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.run_G(self.real_A, update_emas=update_emas)  # G(A)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        fake_AB = torch.cat((self.real_A, self.fake_B), 1) if self.combine_ab else self.fake_B
        gen_logits = self.run_D(fake_AB, c=self.gen_c)
        self.loss_G_GAN = (torch.nn.functional.softplus(-gen_logits)).mean()
        # Second, G(A) = B
        blur_size = np.floor(self.blur_sigma * 3)

        # also blur the image when calculating L1 loss
        if blur_size > 0:
            f = torch.arange(-blur_size, blur_size + 1, device=fake_AB.device).div(self.blur_sigma).square().neg().exp2()
            fake_B = upfirdn2d.filter2d(self.fake_B, f / f.sum())
            real_B = upfirdn2d.filter2d(self.real_B, f / f.sum())
        else:
            fake_B = self.fake_B
            real_B = self.real_B

        self.loss_G_L1 = self.criterionL1(fake_B, real_B) * self.opt.loss.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self, cur_nimg, **kwargs):
        # update D
        self.blur_sigma = max(1 - cur_nimg / (self.opt.loss.blur_fade_kimg * 1e3),
                         0) * self.opt.loss.blur_init_sigma if self.opt.loss.blur_fade_kimg > 0 else 0
        self.optimizer_D.zero_grad(set_to_none=True)
        self.netD.requires_grad_(True)
        self.forward(update_emas=True)  # compute fake images: G(A)
        self.backward_D()                # calculate gradients for D
        self.netD.requires_grad_(False)
        for param in self.netD.parameters():
            if param.grad is not None:
                misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        self.optimizer_D.step()          # update D's weights
        # update G
        self.optimizer_G.zero_grad(set_to_none=True)
        self.netG.requires_grad_(True)
        self.forward(update_emas=False)  # compute fake images: G(A)
        self.backward_G()                   # calculate graidents for G
        self.netG.requires_grad_(False)
        for param in self.netG.parameters():
            if param.grad is not None:
                misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        self.optimizer_G.step()             # udpate G's weights
