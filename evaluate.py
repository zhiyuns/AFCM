import os
import glob
import importlib
import nibabel as nib
from configs import default_argument_parser
from data.get_util import get_logger
from data.utils import get_test_loaders
from models import create_model
from util.evaluation import evaluate_3D, evaluate_slice
import numpy as np


def _get_predictor(model, output_dir, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('models.predictor')
    predictor_class = getattr(m, class_name)

    return predictor_class(model, output_dir, config, **predictor_config)


def rescale_intensity(volume, percentils=[0.5, 99.5], bins_num=256, norm=False):
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])

    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1

    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    if norm:
        volume = volume.astype(float) / (bins_num - 1)

    return volume


def main():
    config = default_argument_parser()
    logger = get_logger('Config')
    logger.info(config)

    test_loaders = get_test_loaders(config)
    model = create_model(config)
    model.isTrain = False
    model.setup(config)
    out_path = os.path.join(config.checkpoints_dir, config.name, 'evaluate')
    os.makedirs(out_path, exist_ok=True)
    fw = open(os.path.join(out_path, 'evaluate.txt'), 'a')
    predictor = _get_predictor(model, out_path, config)

    ori_path = config.loaders.test.ori_file_path
    prefix_img = '_predictions0.nii.gz'
    prefix_ori = '_ori.nii.gz'
    prefix_input = '_predictions2.nii.gz'
    c_psnr = []
    c_ssim = []
    c_psnr_slice = []
    c_ssim_silce = []
    for test_loader in test_loaders:
        # run the model prediction on the test_loader and save the results in the output_dir
        predictor(test_loader)
        subject = test_loader.dataset.file_path.split('/')[-1].split('.')[0]
        input = nib.load(os.path.join(out_path, subject+prefix_input))
        img = nib.load(os.path.join(out_path, subject+prefix_img))

        target_img = nib.load(glob.glob(os.path.join(ori_path, subject, subject+f'*{config.loaders.raw_internal_path_out[-1]}.nii.gz'))[0])
        img_data = img.get_fdata().transpose(2, 1, 0)
        target_data = target_img.get_fdata().transpose(2, 1, 0)
        input_data = input.get_fdata().transpose(2, 1, 0)
        target_data = np.clip(target_data, 0, 255) / 255
        img_data = (np.clip(img_data, -1, 1) + 1) / 2

        print(img_data.shape, target_data.shape)

        oneBEva = evaluate_3D(img_data, target_data)
        oneBEva_slice = evaluate_slice(img_data, target_data)

        c_psnr.append(oneBEva[0])
        c_ssim.append(oneBEva[1])
        c_psnr_slice.append(oneBEva_slice[0])
        c_ssim_silce.append(oneBEva_slice[1])
        img = nib.Nifti1Image(input_data.transpose(2, 1, 0), affine=target_img.affine, header=target_img.header)
        nib.save(img, os.path.join(out_path, subject + prefix_input))
        img = nib.Nifti1Image(img_data.transpose(2, 1, 0), affine=target_img.affine, header=target_img.header)
        nib.save(img, os.path.join(out_path, subject+prefix_img))
        img = nib.Nifti1Image(target_data.transpose(2, 1, 0), affine=target_img.affine, header=target_img.header)
        nib.save(img, os.path.join(out_path, subject + prefix_ori))

        metrics = " subject:{}   psnr:{:.6}, ssim:{:.6}, psnr_slice:{:.6}, ssim_slice:{:.6}\n".format(subject, oneBEva[0], oneBEva[1], oneBEva_slice[0], oneBEva_slice[1])
        fw.write(metrics)
        print(metrics)
        

    metrics = " ^^^VALIDATION mean psnr:{:.6}, ssim:{:.6}, psnr_slice:{:.6}, ssim_slice:{:.6}\n".format(np.mean(c_psnr), np.mean(c_ssim), np.mean(c_psnr_slice), np.mean(c_ssim_silce))
    metrics += " std   psnr:{:.6}, ssim:{:.6}, psnr_slice:{:.6}, ssim_slice:{:.6}\n".format(np.std(c_psnr), np.std(c_ssim), np.std(c_psnr_slice), np.std(c_ssim_silce))
    fw.write(metrics)
    print(metrics)
    


if __name__ == '__main__':
    main()
