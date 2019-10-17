"""Dataset utilities"""

import os

import h5py
from tqdm import tqdm


def save_to_hdf5(numpy_dataset, fpath_hdf5):
    """Save the examples from the `numpy_dataset` to the `fpath_h5`

    :param numpy_dataset: numpy backed dataset that will return back examples
     to save to the HDF5 file
    :type numpy_dataset: datasets.base_dataset.NumPyDataset
    :param fpath_hdf5: HDF5 filepath to save the examples to
    :type fpath_hdf5: str
    """

    hdf5_file = h5py.File(fpath_hdf5, mode='w')

    dataset_keys = (
        numpy_dataset.input_keys + numpy_dataset.target_keys
    )
    for dataset_key in dataset_keys:
        dataset_shape = (
            (len(numpy_dataset), ) +  numpy_dataset.sample_shapes[dataset_key]
        )
        hdf5_file.create_dataset(
            dataset_key, dataset_shape,
            numpy_dataset.sample_types[dataset_key]
        )


    n_observations = len(numpy_dataset)
    for idx_observation in tqdm(range(n_observations), total=n_observations):
        observation = numpy_dataset[idx_observation]
        
        for dataset_key in dataset_keys:
            hdf5_file[dataset_key][idx_observation, ...] = (
                observation[dataset_key]
            )

    hdf5_file.close()
