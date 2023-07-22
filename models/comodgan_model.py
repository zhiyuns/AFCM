import copy
import torch
from collections import OrderedDict
from .pix2pix_model import Pix2PixModel
from configs import CfgNode as CN
from .networks.CoModGAN.torch_utils import misc
from .networks.CoModGAN.torch_utils.ops import conv2d_gradfix
from .networks.CoModGAN.torch_utils.ops import grid_sample_gradfix


class CoModGANModel(Pix2PixModel):
    def __init__(self, opt):
        super(CoModGANModel, self).__init__(opt)
        self.G_mapping = self.netG.module.mapping if hasattr(self.netG.module, 'mapping') else None
        self.G_synthesis = self.netG.module.synthesis if hasattr(self.netG.module, 'synthesis') else None
        self.netG_ema = copy.deepcopy(self.netG).eval()
        self.model_names.append('G_ema')

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=opt.optimizer.lr_G, betas=(0, 0.99), eps=1e-8)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.optimizer.lr_D, betas=(0, 0.99), eps=1e-8)
        '''
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=opt.optimizer.lr, betas=(opt.optimizer.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.optimizer.lr, betas=(opt.optimizer.beta1, 0.999))
        '''
        conv2d_gradfix.enabled = True  # Improves training speed.
        grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.
        self.style_mixing_prob = 0
        self.real_A = self.real_B = self.fake_B = None

    @staticmethod
    def modify_commandline_options(config, is_train=True):
        _C = config
        _C.loaders.slice_num = 3

        _C.model.G.z_dim = 512
        _C.model.G.w_dim = 512
        _C.model.G.c_dim = 0
        _C.model.G.img_resolution = 256
        _C.model.G.img_channels_in = 1
        _C.model.G.img_channels_out = 1

        _C.model.G.synthesis_kwargs = CN()
        _C.model.G.synthesis_kwargs.name = 'SynthesisNetwork'
        _C.model.G.synthesis_kwargs.channel_base = int(0.5 * 32768)
        _C.model.G.synthesis_kwargs.channel_max = 512
        _C.model.G.synthesis_kwargs.skip_resolution = 256
        _C.model.G.synthesis_kwargs.cond_mod = True
        _C.model.G.synthesis_kwargs.num_fp16_res = 0
        _C.model.G.synthesis_kwargs.conv_clamp = None
        _C.model.G.synthesis_kwargs.channel_attention = False

        _C.model.G.mapping_kwargs = CN()
        _C.model.G.mapping_kwargs.name = 'MappingNetwork'
        _C.model.G.mapping_kwargs.num_layers = 8
        _C.model.G.mapping_kwargs.img_resolution = 256
        _C.model.G.mapping_kwargs.img_channels = 1
        _C.model.G.mapping_kwargs.channel_base = int(0.5 * 32768)
        _C.model.G.mapping_kwargs.channel_max = 512

        _C.model.D.channel_base = int(0.5 * 32768)
        _C.model.D.num_fp16_res = 0
        _C.model.D.conv_clamp = None
        _C.model.D.channel_max = 512
        _C.model.D.c_dim = 0
        _C.model.D.img_resolution = 256
        _C.model.D.img_channels = 2

        _C.model.D.mapping_kwargs = CN()
        _C.model.D.epilogue_kwargs = CN()
        _C.model.D.epilogue_kwargs.mbstd_group_size = 16
        return _C

    def run_G(self, cond_img, noise_mode='random'):
        ref_img = self.extra_B if self.extra_b else self.real_B
        ws = self.G_mapping(z=self.gen_z, c=self.gen_c, img_in=ref_img)
        if self.style_mixing_prob > 0:
            cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                 torch.full_like(cutoff, ws.shape[1]))
            ws[:, cutoff:] = self.G_mapping(z=torch.randn_like(self.gen_z), c=self.gen_c, img_in=ref_img, skip_w_avg_update=True)[:, cutoff:]
        img = self.G_synthesis(ws, cond_img, noise_mode=noise_mode)
        return img

    def run_D(self, img, **kwargs):
        return self.netD(img, **kwargs)

    def set_input(self, input):
        super(CoModGANModel, self).set_input(input)
        z_dim = self.opt.model.G.get('z_dim', 512)
        self.gen_z = torch.randn([self.real_A.shape[0], z_dim], device=self.device)
        
        if self.opt.model.G.c_dim > 0:
            self.gen_c = input['slice_idx'].to(self.device)
        else:
            self.gen_c = torch.zeros([self.real_A.shape[0], 1]).pin_memory().to(self.device)

    def set_test_input(self, input, slice_idx, indices):
        super(CoModGANModel, self).set_test_input(input, slice_idx, indices)
        z_dim = self.opt.model.G.get('z_dim', 512)
        self.gen_z = torch.randn([self.real_A.shape[0], z_dim], device=self.device)
        if self.opt.model.G.c_dim > 0:
            self.gen_c = slice_idx.to(self.device)
        else:
            self.gen_c = torch.zeros([self.real_A.shape[0], 1]).pin_memory().to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.run_G(self.real_A)  # G(A)
        if self.sg:
            self.pred_mask = self.netSG(self.fake_B.unsqueeze(1)).squeeze(1)

    def forward_ema(self):
        ref_img = self.extra_B if self.extra_b else self.real_B
        self.fake_B = self.netG_ema(z=self.gen_z, c=self.gen_c, cond_img=self.real_A, ref_img=ref_img, noise_mode='const')  # G(A)
        if self.sg or not self.isTrain:
            self.fake_B = self.fake_B.detach()
            with torch.no_grad():
                self.pred_mask = self.netSG(self.fake_B.unsqueeze(1))
                self.pred_mask = torch.sigmoid(self.pred_mask.squeeze(1))

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward_ema()
            self.compute_visuals()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1) if self.combine_ab else self.fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator

        gen_logits = self.run_D(fake_AB.detach(), c=self.gen_c)
        self.loss_D_fake = (torch.nn.functional.softplus(gen_logits)).mean()
        self.loss_D_fake.backward()
        # Real
        '''
        if self.extra_b:
            real_AB = self.extra_B
        else:
            real_AB = torch.cat((self.real_A, self.real_B), 1) if self.combine_ab else self.real_B
        '''
        real_AB = torch.cat((self.real_A, self.real_B), 1) if self.combine_ab else self.real_B
        real_img_tmp = real_AB.detach().requires_grad_(True)
        real_logits = self.run_D(real_img_tmp, c=self.gen_c)
        self.loss_D_real = (torch.nn.functional.softplus(-real_logits)).mean()
        self.loss_D = self.loss_D_real
        if self.opt.loss.lambda_r1 > 0:
            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp],
                                           create_graph=True, only_inputs=True)[0]
            self.loss_Dr1 = (r1_grads.square().sum([1, 2, 3])).mean() * 0.5
            self.loss_D += self.loss_Dr1 * self.opt.loss.lambda_r1

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1) if self.combine_ab else self.fake_B
        gen_logits = self.run_D(fake_AB, c=self.gen_c)
        self.loss_G_GAN = (torch.nn.functional.softplus(-gen_logits)).mean()
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.loss.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        if self.sg:
            self.loss_seg = self.criterionSeg(self.pred_mask, self.label)
            self.loss_G += self.loss_seg * self.opt.loss.lambda_SG
            self.pred_mask = torch.sigmoid(self.pred_mask)
        self.loss_G.backward()

    def optimize_parameters(self, **kwargs):
        # update D
        self.optimizer_D.zero_grad(set_to_none=True)
        self.netD.requires_grad_(True)
        self.forward()  # compute fake images: G(A)
        self.backward_D()                # calculate gradients for D
        self.netD.requires_grad_(False)
        for param in self.netD.parameters():
            if param.grad is not None:
                misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        self.optimizer_D.step()          # update D's weights
        # update G
        self.optimizer_G.zero_grad(set_to_none=True)
        self.netG.requires_grad_(True)
        self.forward()  # compute fake images: G(A)
        self.backward_G()                   # calculate graidents for G
        self.netG.requires_grad_(False)
        for param in self.netG.parameters():
            if param.grad is not None:
                misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        self.optimizer_G.step()             # udpate G's weights


    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                img = getattr(self, name)
                img = img[:, 0:1, :, :]
                visual_ret[name] = img
        return visual_ret
