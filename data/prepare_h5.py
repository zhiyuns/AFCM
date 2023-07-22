import  h5py
import os
import SimpleITK as sitk
# import nibabel as nib
from tqdm import tqdm
import numpy as np
from multiprocessing import Process

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

def convert_h5(subject, data_path, out_path):
    h5_file = os.path.join(out_path, subject+'.h5')
    h5_file = h5py.File(h5_file, 'w')
    modalities = ['T1_HR', 'T2_FLAIR_linear', ]
    for modality in modalities:
        img = os.path.join(data_path, subject, subject+f'_{modality}.nii.gz')
        if os.path.exists(img):
            img_nii = sitk.ReadImage(os.path.join(data_path, subject, subject+f'_{modality}.nii.gz'))
            img_data = sitk.GetArrayFromImage(img_nii)
            assert (img_data.shape[1] == 256 and img_data.shape[2] == 256)
            img_data = np.around(img_data)
            img_data = np.clip(img_data, 0, 255)
            img_data = img_data.astype('uint8')

            h5_file[modality] = img_data
    h5_file.close()

data_path = r'./alighed_data'
out_path = r'./data_h5'
os.makedirs(out_path, exist_ok=True)
all_subject = os.listdir(data_path)

num_processes = 1

for loop in tqdm(range(len(all_subject) // num_processes + 1)):
    processes = []
    for subject in all_subject[loop*num_processes:(loop+1)*num_processes]:
        p = Process(target=convert_h5, args=(subject,data_path, out_path))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
