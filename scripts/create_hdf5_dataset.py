"""Create an HDF5 file holding a specified dataset"""

import argparse
import os

import pandas as pd
import yaml

from trainet.datasets.utils import save_to_hdf5
from trainet.utils.generic_utils import import_object


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--fpath_config', type=str, required=True,
        help=(
            'Absolute filepath to a YML training config that specifies '
            'a training and validation dataset to train with. HDF5 files '
            'will be created for both the training and validation datasets, '
            'saved in the --dirpath_output that must be specified.'
        )
    )
    parser.add_argument(
        '--dirpath_output', type=str, required=True,
        help=('Absolute directory path to save the HDF5 files in.')
    )

    # Optional
    parser.add_argument(
        '--fname_base', type=str,
        help=(
            'Holds the base filename to use when saving the HDF5 files; '
            'the set name will be prepended to the --fname_base before saving '
            ', e.g. \'--fname_base context05\' results in '
            '\'train_context05.h5\' and \'validation_context05.h5\' files '
            'being saved in --dirpath_output. If none, then there will '
            'simply be \'train.h5\' and \'validation.h5\' files saved.'
        )
    )

    args = parser.parse_args()
    return args


def main():
    """Main"""

    args = parse_args()

    fpath_config = args.fpath_config
    dirpath_output = args.dirpath_output
    fname_base = args.fname_base

    os.makedirs(dirpath_output, exist_ok=True)

    with open(fpath_config) as f:
        training_config = yaml.load(f, Loader=yaml.FullLoader)
    dataset_spec = training_config['dataset']

    # TODO: This won't generalize to non-dataframe stuffs
    for set_name in ('train', 'validation'):
        fpath_df_obs_key = 'fpath_df_{}'.format(set_name)
        fpath_df_obs = dataset_spec[fpath_df_obs_key]
        df_obs = pd.read_csv(fpath_df_obs)
        dataset_spec['init_params']['df_obs'] = df_obs

        dataset_importpath = dataset_spec['importpath']
        DataSet = import_object(dataset_importpath)
        train_dataset = DataSet(**dataset_spec['init_params'])

        if not fname_base:
            fname_hdf5 = '{}.h5'.format(set_name)
        else:
            fname_hdf5 = '{}_{}.h5'.format(set_name, fname_base)
        fpath_hdf5 = os.path.join(dirpath_output, fname_hdf5)

        save_to_hdf5(train_dataset, fpath_hdf5)


if __name__ == "__main__":
    main()
