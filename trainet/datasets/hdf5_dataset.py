"""Dataset to support training from HDF5 files"""

import h5py

from trainet.datasets.base_dataset import NumPyDataset


# TODO: Can probably replace certain things pulled from the numpy_dataset with
# attributes of the hdf5 file, and could maybe just store some things in the
# file itself
class HDF5Dataset(NumPyDataset):
    """Dataset for applying transformations to a NumPyDataset"""

    def __init__(self, fpath_hdf5, numpy_dataset):
        """Init

        :param fpath_hdf5: filepath to an hdf5 dataset to read from
        :type fpath_hdf5: str
        :param numpy_dataset: numpy dataset that the hdf5 file is built from;
         used primarly as a means of getting sample keys, sample shapes, etc.
        :type numpy_dataset: datasets.base_dataset.NumPyDataset
        """
        
        self.fpath_hdf5 = fpath_hdf5
        self.hdf5_file = h5py.File(self.fpath_hdf5, 'r')

        self.numpy_dataset = numpy_dataset

    def __getitem__(self, idx):
        """Return the transformed sample at index `idx` of `self.numpy_dataset`

        :param idx: the index of the observation to return
        :type idx: int
        :return: sample returned from `self.numpy_dataset.__getitem__`
        :rtype: dict
        """

        sample = {}

        dataset_keys = self.input_keys + self.target_keys
        for dataset_key in dataset_keys:
            sample[dataset_key] = self.hdf5_file[dataset_key][idx, ...]
        
        sample['label'] = sample['label'][0]

        return sample

    def __len__(self):
        """Return the size of the dataset

        :return: size of the dataset
        :rtype: int
        """

        return len(self.numpy_dataset)

    @property
    def input_keys(self):
        """Return the sample keys that denote a learning algorithm's inputs

        These keys should be contained in the dictionary returned from
        __getitem__, and correspond to the keys that will be the inputs to a
        neural network.

        :return: input key names
        :rtype: list[str]
        """
        return self.numpy_dataset.input_keys

    @property
    def required_config_keys(self):
        """Return the keys required to be in the config passed to the __init__

        :return: required configuration keys
        :rtype: set{str}
        """
        return self.numpy_dataset.required_config_keys

    @property
    def sample_shapes(self):
        """Return shapes of the sample elements returned from __getitem__

        :return: dict holding tuples of the shapes for the elements of the
         sample returned from __getitem__
        :return: dict{str: tuple}
        """
        return self.numpy_dataset.sample_shapes

    @property
    def sample_types(self):
        """Return data types of the sample elements returned from __getitem__

        :return: element data types for each element in a sample returned from
         __getitem__
        :rtype: dict{str: str}
        """
        return self.numpy_dataset.sample_types

    @property
    def target_keys(self):
        """Return the sample keys that denote a learning algorithm's targets

        These keys should be contained in the dictionary returned from
        __getitem__, and correspond to the keys that will be the targets to a
        neural network.

        :return: target key names
        :rtype: list[str]
        """
        return self.numpy_dataset.target_keys
