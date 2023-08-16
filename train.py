import time
import torch
import random
import numpy as np
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

from configs import default_argument_parser
from data.get_util import get_logger
from util.evaluation import evaluate_2D

if __name__ == '__main__':
    config = default_argument_parser()
    logger = get_logger('Config')
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        logger.warning('Using CuDNN deterministic setting. This may slow down the training!')
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True

    dataset = create_dataset(config.loaders, phase='train')  # create a dataset given opt.dataset_mode and other options
    val_dataset = create_dataset(config.loaders, phase='val')
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(config)      # create a model given opt.model and other options
    model.setup(config)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(config)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    ssim_max = 0

    for epoch in range(config.scheduler.epoch_count, config.scheduler.n_epochs + config.scheduler.n_epochs_decay + 1):
        # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % config.trainer.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += config.loaders.batch_size
            epoch_iter += config.loaders.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(cur_nimg=total_iters)   # calculate loss functions, get gradients, update network weights

            if total_iters % config.trainer.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % config.trainer.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % config.trainer.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / config.loaders.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if config.display.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if config.scheduler.ema.enabled:
                # Update G_ema.
                ema_nimg = config.scheduler.ema.ema_kimgs * 1000
                ema_rampup = config.scheduler.ema.ramp
                if ema_rampup is not None:
                    ema_nimg = min(ema_nimg, total_iters * ema_rampup)
                ema_beta = 0.5 ** (config.loaders.batch_size / max(ema_nimg, 1e-8))
                for p_ema, p in zip(model.netG_ema.parameters(), model.netG.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(model.netG_ema.buffers(), model.netG.buffers()):
                    b_ema.copy_(b)

            if total_iters % config.trainer.save_latest_freq == 0:   # evaluate our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if config.trainer.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                # evaluate
                c_psnr = []
                c_ssim = []
                model.isTrain = False
                for i, data in enumerate(val_dataset):
                    model.set_input(data)
                    model.test()
                    predictions = model.fake_B.unsqueeze(1)
                    real_B = model.real_B.unsqueeze(1)

                    predictions = (predictions.cpu().numpy() + 1) / 2
                    real_B = (real_B.cpu().numpy() + 1) / 2
                    predictions = np.clip(predictions, 0, 1)
                    real_B = np.clip(real_B, 0, 1)
                    oneBEva = evaluate_2D(predictions, real_B)
                    if oneBEva is None:
                        continue
                    else:
                        c_psnr.append(oneBEva[0])
                        c_ssim.append(oneBEva[1])

                val_results = {'psnr': np.mean(c_psnr),
                               'ssim': np.mean(c_ssim)}
                visualizer.plot_val_results(val_results)

                ssim_new = np.mean(c_ssim)
                if ssim_new > ssim_max:
                    ssim_max = ssim_new
                    model.save_networks('best')

                model.isTrain = True
                print('End of epoch %d / %d \t Time Taken: %d sec' % (
                epoch, config.scheduler.n_epochs + config.scheduler.n_epochs_decay, time.time() - epoch_start_time))
                print(" ^^^VALIDATION   psnr:{:.6}, ssim:{:.6}".format(np.mean(c_psnr), np.mean(c_ssim)))

            iter_data_time = time.time()

        if epoch % config.trainer.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)


