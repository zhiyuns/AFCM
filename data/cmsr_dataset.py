import glob
import os
import torch
import random
from itertools import chain
import h5py
import numpy as np
from .augment import transforms
from .utils import get_slice_builder, ConfigDataset, calculate_stats
from .get_util import get_logger
from torch.utils.data import ConcatDataset
logger = get_logger('TrainingSetup')

GLOBAL_RANDOM_STATE = np.random.RandomState(50)


def get_cls_label(shape, idx):
    onehot = np.zeros(shape, dtype=np.float32)
    onehot[idx] = 1
    label = onehot
    return label.copy()


class AbstractHDF5Dataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self, file_path,
                 phase,
                 slice_builder_config,
                 transformer_config,
                 raw_internal_path_in='raw',
                 raw_internal_path_out='raw',
                 cat_inputs=False,
                 all_hr=False,
                 thickness=(),
                 slice_num=4,
                 global_normalization=True):
        """
        :param file_path: path to H5 file containing raw data as well as labels and per pixel weights (optional)
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :para'/home/adrian/workspace/ilastik-datasets/VolkerDeconv/train'm slice_builder_config: configuration of the SliceBuilder
        :param transformer_config: data augmentation configuration
        :param raw_internal_path (str or list): H5 internal path to the raw dataset
        :param label_internal_path (str or list): H5 internal path to the label dataset
        :param weight_internal_path (str or list): H5 internal path to the per pixel weights
        """
        assert phase in ['train', 'val', 'test']
        self.cat_inputs = cat_inputs
        self.all_hr = all_hr
        self.phase = phase
        self.file_path = file_path
        self.raw_internal_path_in = raw_internal_path_in
        self.raw_internal_path_out = raw_internal_path_out
        self.raw_internal_path = raw_internal_path = self.raw_internal_path_in + self.raw_internal_path_out
        self.thickness = thickness
        self.slice_num = slice_num

        input_file = self.create_h5_file(file_path)

        self.raw = self.fetch_and_check(input_file, raw_internal_path)

        if global_normalization:
            stats = calculate_stats(self.raw)
        else:
            stats = {'pmin': None, 'pmax': None, 'mean': None, 'std': None}

        self.transformer = transforms.Transformer(transformer_config, stats)

        print(self.raw[raw_internal_path[-1]].shape)
        slice_builder = get_slice_builder(self.raw[raw_internal_path[-1]], None,
                                          self.weight_map, slice_builder_config)

        self.raw_slices = slice_builder.raw_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

    @staticmethod
    def fetch_and_check(input_file, internal_paths):
        ds_dict = {}
        for each_path in internal_paths:
            assert each_path in input_file.keys(), f'Image {each_path} not found!'
            ds = input_file[each_path][:]
            if ds.ndim == 2:
                # expand dims if 2d
                ds = np.expand_dims(ds, axis=0)
            ds_dict[each_path] = ds
        return ds_dict

    def __getitem__(self, idx):
        if len(self.thickness)>0:
            if self.phase == 'train':
                thickness = random.choice(self.thickness)
            else:
                thickness = self.thickness[0]
        else:
            thickness = -1

        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        data_dict = {}
        raw_transform = self.transformer.raw_transform()
        if self.slice_num == 1:
            idx_A = idx
            data_A = []
            data_A.append(raw_transform(self.raw[self.raw_internal_path_in[0]][raw_idx]))
            data_dict['A'] = torch.cat(data_A)
        elif self.slice_num == 4:
            idx_A = int((idx_B // thickness) * thickness)
            raw_idx_A_minus = self.raw_slices[idx_A - thickness] if idx_A - thickness >= 0 else None
            raw_idx_A = self.raw_slices[idx_A]
            raw_idx_A_plus = self.raw_slices[idx_A + thickness] if idx_A + thickness <= self.patch_count - 1 else None
            raw_idx_A_plus_plus = self.raw_slices[idx_A + thickness * 2] if idx_A + thickness * 2 <= self.patch_count - 1 else None
            raw_idx_As = [raw_idx_A_minus, raw_idx_A, raw_idx_A_plus, raw_idx_A_plus_plus]
            image_A = []
            for slice_idx in raw_idx_As:
                if slice_idx is not None:
                    raw_transform = self.transformer.raw_transform()
                    image_A.append(raw_transform(self.raw[self.raw_internal_path_in[0]][slice_idx]))
                else:
                    image_A.append(raw_transform(np.zeros(self.raw[self.raw_internal_path_in[0]][0:1].shape)))
            data_dict['A'] = torch.cat(image_A)
        else:
            raise NotImplementedError('slice number %s not suppoered' % self.slice_num)
            
        if self.phase != 'test':
            raw_transform = self.transformer.raw_transform()
            idx_B = idx
            data_dict['B'] = raw_transform(self.raw[self.raw_internal_path_out[-1]][self.raw_slices[idx_B]])
            data_dict['B_class'] = get_cls_label(len(self.raw_internal_path_out), len(self.raw_internal_path_out) - 1)
            data_dict['B_idx'] = torch.Tensor([idx_B])
            return data_dict
        else:
            return data_dict['A'], torch.Tensor(
                    np.array([idx_B - idx_A], dtype=np.float32) / thickness), raw_idx    
        
    def __len__(self):
        return self.patch_count

    @staticmethod
    def create_h5_file(file_path):
        raise NotImplementedError

    @staticmethod
    def _check_volume_sizes(raw, label):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        for each_raw in raw.values():
            for each_label in label.values():
                assert each_raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
                assert each_label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
                assert _volume_shape(each_raw) == _volume_shape(each_label), 'Raw and labels have to be of the same size'

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
        file_paths = cls.traverse_h5_paths(file_paths)

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
                              cat_inputs=dataset_config.get('cat_inputs', False),
                              all_hr=dataset_config.get('all_hr', False),
                              thickness=dataset_config.get('thickness', ()),
                              slice_num=dataset_config.get('slice_num', 4),
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets

    @staticmethod
    def traverse_h5_paths(file_paths):
        assert isinstance(file_paths, list)
        results = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # if file path is a directory take all H5 files in that directory
                iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
                for fp in chain(*iters):
                    results.append(fp)
            else:
                results.append(file_path)
        return results


class StandardHDF5Dataset(AbstractHDF5Dataset):
    """
    Implementation of the HDF5 dataset which loads the data from all of the H5 files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, file_path, phase, slice_builder_config, transformer_config, 
                 raw_internal_path_in='raw', raw_internal_path_out='raw', cat_inputs=False,
                 all_hr=False, thickness=(), slice_num=3, global_normalization=True):
        super().__init__(file_path=file_path,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         raw_internal_path_in=raw_internal_path_in,
                         raw_internal_path_out=raw_internal_path_out,
                         cat_inputs=cat_inputs,
                         all_hr=all_hr,
                         thickness=thickness,
                         slice_num=slice_num,
                         global_normalization=global_normalization)

    @staticmethod
    def create_h5_file(file_path):
        return h5py.File(file_path, 'r')


class CmsrDataset(ConcatDataset):
    def __init__(self, opt, phase='train'):
        train_datasets = StandardHDF5Dataset.create_datasets(opt, phase=phase)
        super(CmsrDataset, self).__init__(train_datasets)


