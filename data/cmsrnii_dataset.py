import glob
import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from itertools import chain
from torch.utils.data import ConcatDataset
from .cmsr_dataset import AbstractHDF5Dataset
from scipy.ndimage import gaussian_filter
from .get_util import get_logger
logger = get_logger('TrainingSetup')


class StandardNIIDataset(AbstractHDF5Dataset):
    """
    Implementation of the HDF5 dataset which loads the data from all of the H5 files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, file_path, phase, slice_builder_config, transformer_config, 
                 raw_internal_path_in='raw', raw_internal_path_out='raw', rand_output=False, cat_inputs=False,
                 thickness=(), slice_num=3, global_normalization=True):
        super().__init__(file_path=file_path,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         raw_internal_path_in=raw_internal_path_in,
                         raw_internal_path_out=raw_internal_path_out,
                         rand_output=rand_output,
                         cat_inputs=cat_inputs,
                         thickness=thickness,
                         slice_num=slice_num,
                         global_normalization=global_normalization)

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config['train'] if phase == 'train' else dataset_config['test']

        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        prefix = phase_config.get('prefix', '_predictions0')
        file_paths = cls.traverse_nii_paths(file_paths, prefix)

        datasets = []
        for file_path in file_paths:
            try:
                logger.info(f'Loading {phase} set from: {file_path}...')
                dataset = cls(file_path=file_path,
                              phase=phase,
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              raw_internal_path_in=dataset_config.get('raw_internal_path_in', 'raw'),
                              raw_internal_path_out=dataset_config.get('raw_internal_path_out', 'raw'),
                              rand_output=dataset_config.get('rand_output', False),
                              cat_inputs=dataset_config.get('cat_inputs', False),
                              thickness=dataset_config.get('thickness', ()),
                              slice_num=dataset_config.get('slice_num', 4),
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets

    @staticmethod
    def traverse_nii_paths(file_paths, prefix=''):
        assert isinstance(file_paths, list)
        results = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                results.extend([os.path.join(file_path, x) for x in os.listdir(file_path)])
        return results
    
    def __percentile_clip(self, input_tensor, reference_tensor=None, p_min=0.5, p_max=99.5, strictlyPositive=True):
        """Normalizes a tensor based on percentiles. Clips values below and above the percentile.
        Percentiles for normalization can come from another tensor.

        Args:
            input_tensor (torch.Tensor): Tensor to be normalized based on the data from the reference_tensor.
                If reference_tensor is None, the percentiles from this tensor will be used.
            reference_tensor (torch.Tensor, optional): The tensor used for obtaining the percentiles.
            p_min (float, optional): Lower end percentile. Defaults to 0.5.
            p_max (float, optional): Upper end percentile. Defaults to 99.5.
            strictlyPositive (bool, optional): Ensures that really all values are above 0 before normalization. Defaults to True.

        Returns:
            torch.Tensor: The input_tensor normalized based on the percentiles of the reference tensor.
        """
        if(reference_tensor == None):
            reference_tensor = input_tensor
        v_min, v_max = np.percentile(reference_tensor, [p_min,p_max]) #get p_min percentile and p_max percentile

        if( v_min < 0 and strictlyPositive): #set lower bound to be 0 if it would be below
            v_min = 0
        output_tensor = np.clip(input_tensor,v_min,v_max) #clip values to percentiles from reference_tensor
        output_tensor = (output_tensor - v_min)/(v_max-v_min) #normalizes values to [0;1]

        return output_tensor
    
    def create_h5_file(self, file_path):
        out_dict = {}
        for raw_name in self.raw_internal_path:
            img_nii = sitk.ReadImage(glob.glob(os.path.join(file_path, file_path.split('/')[-1]+f'*{raw_name}.nii*'))[0])
            img_data = sitk.GetArrayFromImage(img_nii)
            img_data = self.__percentile_clip(img_data)
            img_data = (img_data * 255).astype('uint8')
            out_dict[raw_name] = img_data

        return out_dict


class CmsrNIIDataset(ConcatDataset):
    def __init__(self, opt, phase='test'):
        train_datasets = StandardNIIDataset.create_datasets(opt, phase=phase)
        super(CmsrNIIDataset, self).__init__(train_datasets)
