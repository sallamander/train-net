"""Class for running a specified tensorflow training job"""

import pandas as pd

from trainet.datasets.augmented_dataset import AugmentedDataset
from trainet.datasets.hdf5_dataset import HDF5Dataset
from trainet.training.base_training_job import BaseTrainingJob
from trainet.training.tf.data_loader import DataLoader
from trainet.training.tf.utils import set_global_random_seed
from trainet.utils.generic_utils import import_object


class TrainingJob(BaseTrainingJob):
    """Runs a training job as specified via a config

    See the parent class docstring for details on how to specify a config for
    training.
    """

    def _instantiate_dataset(self, set_name):
        """Return a dataset object to be used as an iterator during training

        The dataset that is returned should be able to be directly passed into
        the `train` method of whatever trainer class is specified in
        `self.config`, as either the `train_dataset` or `validation_dataset`
        argument.

        :param set_name: set to return the dataset for, one of
         {'train', 'validation'}
        :type set_name: str
        :return: two element tuple holding an iterable over the dataset for
         `set_name`, as well as the number of batches in a single pass over the
         dataset
        :rtype: tuple
        """

        assert set_name in {'train', 'validation'}
        dataset_spec = self.config['dataset']

        fpath_df_obs_key = 'fpath_df_{}'.format(set_name)
        if fpath_df_obs_key in dataset_spec:
            fpath_df_obs = dataset_spec[fpath_df_obs_key]
            df_obs = pd.read_csv(fpath_df_obs)
            dataset_spec['init_params']['df_obs'] = df_obs

        dataset_importpath = dataset_spec['importpath']
        DataSet = import_object(dataset_importpath)
        dataset = DataSet(**dataset_spec['init_params'])

        fpath_hdf5_key = 'fpath_hdf5_{}'.format(set_name)
        if fpath_hdf5_key in dataset_spec:
            fpath_hdf5 = dataset_spec[fpath_hdf5_key]
            dataset = HDF5Dataset(fpath_hdf5, dataset)

        albumentations_key = '{}_albumentations'.format(set_name)
        albumentations = dataset_spec.get(albumentations_key, {})
        if albumentations:
            albumentations = self._parse_albumentations(albumentations)
            dataset = AugmentedDataset(dataset, albumentations)

        transformations_key = '{}_transformations'.format(set_name)
        transformations = dataset_spec[transformations_key]
        transformations = self._parse_transformations(transformations)
        dataset = AugmentedDataset(dataset, transformations)

        loader = DataLoader(dataset)
        loading_params = dataset_spec['{}_loading_params'.format(set_name)]
        dataset_gen = loader.get_infinite_iter(**loading_params)

        return dataset_gen
