from .config import CfgNode as CN

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2
_C.name = 'cmsr_pix2pix'
_C.direction = 'AtoB'
_C.isTrain = True
_C.extra_b = False
_C.checkpoints_dir = './checkpoints'
_C.continue_train = False
_C.load_iter = 0
# which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter];
# otherwise, the code will load models by [epoch]
_C.epoch = None  # which epoch to load? set to latest to use latest cached model
_C.verbose = False

_C.loaders = CN()
_C.loaders.dataset_mode = 'cmsr'
_C.loaders.batch_size = 8
_C.loaders.serial_batches = False
_C.loaders.num_threads = 8
_C.loaders.mirror_padding = [0, 0, 0]
_C.loaders.raw_internal_path_in = ['T2_FLAIR', 'T1']
_C.loaders.raw_internal_path_out = ['T2_FLAIR', 'T1']
_C.loaders.rand_output = False
_C.loaders.cat_inputs = False
_C.loaders.slice_num = 4
_C.loaders.random_modality_in = False
_C.loaders.random_modality_out = False
_C.loaders.preprocess = 'none'
_C.loaders.thickness = []
_C.loaders.out_thickness = 1.0

_C.loaders.train = CN()
_C.loaders.train.file_paths = ['../cmsr_data/data_h5/train', ]

_C.loaders.train.slice_builder = CN()
_C.loaders.train.slice_builder.name = 'FilterSliceBuilder'
_C.loaders.train.slice_builder.patch_shape = [1, 128, 128]
_C.loaders.train.slice_builder.stride_shape = [1, 64, 64]
_C.loaders.train.slice_builder.threshold = 0.0
_C.loaders.train.slice_builder.slack_acceptance = 0.01
_C.loaders.train.slice_builder.gray_threshold = 200
_C.loaders.train.slice_builder.gray_percentile = 0.2

_C.loaders.train.transformer = CN()

_C.loaders.train.transformer.raw = CN()
_C.loaders.train.transformer.raw.PercentileNormalizer = CN()
_C.loaders.train.transformer.raw.PercentileNormalizer.enabled = False
_C.loaders.train.transformer.raw.PercentileNormalizer.pmin = 1.0
_C.loaders.train.transformer.raw.PercentileNormalizer.pmax = 99.6
_C.loaders.train.transformer.raw.Standardize = CN()
_C.loaders.train.transformer.raw.Standardize.enabled = False
_C.loaders.train.transformer.raw.Normalize = CN()
_C.loaders.train.transformer.raw.Normalize.enabled = False
_C.loaders.train.transformer.raw.Normalize.min_value = 0.0
_C.loaders.train.transformer.raw.Normalize.max_value = 1.0
_C.loaders.train.transformer.raw.RandomFlip = CN()
_C.loaders.train.transformer.raw.RandomFlip.enabled = False
_C.loaders.train.transformer.raw.RandomFlip.axes = [0, ]
_C.loaders.train.transformer.raw.ResizeCrop = CN()
_C.loaders.train.transformer.raw.ResizeCrop.enabled = False
_C.loaders.train.transformer.raw.ResizeCrop.inter_size = [286, 286]
_C.loaders.train.transformer.raw.ResizeCrop.target_size = [256, 256]
_C.loaders.train.transformer.raw.ResizeCrop.is_label = False
_C.loaders.train.transformer.raw.RandomRotate90 = CN()
_C.loaders.train.transformer.raw.RandomRotate90.enabled = False
_C.loaders.train.transformer.raw.RandomRotate = CN()
_C.loaders.train.transformer.raw.RandomRotate.enabled = False
_C.loaders.train.transformer.raw.RandomRotate.axes = [[2, 1]]
_C.loaders.train.transformer.raw.RandomRotate.angle_spectrum = 45
_C.loaders.train.transformer.raw.RandomRotate.mode = 'reflect'
_C.loaders.train.transformer.raw.ElasticDeformation = CN()
_C.loaders.train.transformer.raw.ElasticDeformation.enabled = False
_C.loaders.train.transformer.raw.ElasticDeformation.spline_order = 3
_C.loaders.train.transformer.raw.GaussianBlur3D = CN()
_C.loaders.train.transformer.raw.GaussianBlur3D.enabled = False
_C.loaders.train.transformer.raw.GaussianBlur3D.execution_probability = 0.5
_C.loaders.train.transformer.raw.AdditiveGaussianNoise = CN()
_C.loaders.train.transformer.raw.AdditiveGaussianNoise.enabled = False
_C.loaders.train.transformer.raw.AdditiveGaussianNoise.execution_probability = 0.2
_C.loaders.train.transformer.raw.AdditivePoissonNoise = CN()
_C.loaders.train.transformer.raw.AdditivePoissonNoise.enabled = False
_C.loaders.train.transformer.raw.AdditivePoissonNoise.execution_probability = 0.2
_C.loaders.train.transformer.raw.ToTensor = CN()
_C.loaders.train.transformer.raw.ToTensor.enabled = True
_C.loaders.train.transformer.raw.ToTensor.expand_dims = False

_C.loaders.test = CN()
_C.loaders.test.ori_file_path = '../../data/cmsr_data/acpc_align_extra_cropped'
_C.loaders.test.file_paths = ['../../data/cmsr_data/acpc_align_extra_cropped_h5/test', ]
_C.loaders.test.ref_path = None

_C.loaders.test.slice_builder = CN()
_C.loaders.test.slice_builder.name = 'SliceBuilder'
_C.loaders.test.slice_builder.patch_shape = [1, 256, 256]
_C.loaders.test.slice_builder.stride_shape = [1, 16, 16]
_C.loaders.test.slice_builder.threshold = 0.6
_C.loaders.test.slice_builder.slack_acceptance = 0.01
_C.loaders.test.slice_builder.gray_threshold = 100
_C.loaders.test.slice_builder.gray_percentile = 0.2

_C.loaders.test.transformer = CN()
_C.loaders.test.transformer.raw = CN()
_C.loaders.test.transformer.raw.PercentileNormalizer = CN()
_C.loaders.test.transformer.raw.PercentileNormalizer.enabled = False
_C.loaders.test.transformer.raw.PercentileNormalizer.pmin = 0.0
_C.loaders.test.transformer.raw.PercentileNormalizer.pmax = 100.0
_C.loaders.test.transformer.raw.Standardize = CN()
_C.loaders.test.transformer.raw.Standardize.enabled = False
_C.loaders.test.transformer.raw.Normalize = CN()
_C.loaders.test.transformer.raw.Normalize.enabled = True
_C.loaders.test.transformer.raw.Normalize.min_value = 0.0
_C.loaders.test.transformer.raw.Normalize.max_value = 255.0
_C.loaders.test.transformer.raw.ToTensor = CN()
_C.loaders.test.transformer.raw.ToTensor.enabled = True
_C.loaders.test.transformer.raw.ToTensor.expand_dims = False

_C.model = CN()
_C.model.name = 'stylegan3'  # 'chooses which model to use. [stylegan3 | comodgan ]'
_C.model.combine_ab = True  # whether combine image A and image B before feeding to Discriminator
_C.model.G = CN()
_C.model.G.netG = 'stylegan3'
_C.model.D = CN()
_C.model.D.netD = 'comodgan'

_C.optimizer = CN()
_C.optimizer.lr_G = 0.0002
_C.optimizer.lr_D = 0.0002
_C.optimizer.beta1 = 0.5

_C.loss = CN()
_C.loss.gan_mode = 'vanilla'
#  the type of GAN objective. [vanilla| lsgan | wgangp].
#  vanilla GAN loss is the cross-entropy objective used in the original GAN paper.
_C.loss.lambda_L1 = 100.0  # weight for L1 loss
_C.loss.lambda_r1 = 10.0  # weight for R1 regularization

_C.scheduler = CN()
_C.scheduler.n_epochs = 100
_C.scheduler.epoch_count = 1
# the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...
_C.scheduler.n_epochs_decay = 100  # number of epochs to linearly decay learning rate to zero
_C.scheduler.lr_decay_iters = 50  # number of epochs to linearly decay learning rate to zero
_C.scheduler.lr_policy = 'linear'  # learning rate policy. [linear | step | plateau | cosine]
_C.scheduler.ema = CN()
_C.scheduler.ema.enabled = False
_C.scheduler.ema.ema_kimgs = 10
_C.scheduler.ema.ramp = None

_C.display = CN()
_C.display.display_id = 0
_C.display.use_html = False
_C.display.display_server = "http://localhost"
_C.display.display_port = 8097
_C.display.display_env = 'main'
#  visdom display environment name (default is "main")
_C.display.display_winsize = 256
_C.display.display_ncols = 4
_C.display.use_wandb = True
_C.display.wandb_project_name = 'cmsr'

_C.trainer = CN()
_C.trainer.print_freq = 100  # frequency of showing training results on console
_C.trainer.display_freq = 400  # frequency of showing training results on console
_C.trainer.update_html_freq = 1000  # frequency of showing training results on console
_C.trainer.save_latest_freq = 5000  # frequency of saving the latest results
_C.trainer.save_epoch_freq = 5000  # frequency of saving checkpoints at the end of epochs
_C.trainer.save_by_iter = False  # frequency of saving checkpoints at the end of epochs

_C.predictor = CN()
_C.predictor.patch_halo = [0, 8, 8]


