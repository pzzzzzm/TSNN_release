import numpy as np

from basicts.data.simple_tsf_dataset import TimeSeriesForecastingDataset
from basicts.utils import get_regular_settings


def construct_data(dataset_name, with_val=False, suffix=''):

    regular_settings = get_regular_settings(dataset_name)
    INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
    OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence
    TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
    # NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
    # RESCALE = regular_settings['RESCALE'] # Whether to rescale the data
    # NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data

    # TRAIN_VAL_TEST_RATIO = (0.6, 0.2, 0.2)

    ds_tr = TimeSeriesForecastingDataset(dataset_name, TRAIN_VAL_TEST_RATIO, 'train', INPUT_LEN, OUTPUT_LEN)
    ds_val = TimeSeriesForecastingDataset(dataset_name, TRAIN_VAL_TEST_RATIO, 'valid', INPUT_LEN, OUTPUT_LEN)
    ds_te = TimeSeriesForecastingDataset(dataset_name, TRAIN_VAL_TEST_RATIO, 'test', INPUT_LEN, OUTPUT_LEN)


    def build_input_ds(ds):
        new_ds = np.empty([len(ds), INPUT_LEN, ds.data.shape[1], ds.data.shape[2]])
        new_tar = np.empty([len(ds), INPUT_LEN, ds.data.shape[1], ds.data.shape[2]])
        for i in range(len(ds)):
            new_ds[i] = ds[i]['inputs']
            new_tar[i] = ds[i]['target']
        return new_ds, new_tar

    nds, nta = build_input_ds(ds_tr)

    if with_val:
        ndsv, ntav = build_input_ds(ds_val)
        nds = np.concatenate((nds, ndsv))
        nta = np.concatenate((nta, ntav))

    np.save('datasets/constructed_data/train_inp_{}.npy'.format(dataset_name+suffix), nds)
    np.save('datasets/constructed_data/train_tar_{}.npy'.format(dataset_name+suffix), nta)

    nds, nta = build_input_ds(ds_val)
    np.save('datasets/constructed_data/valid_inp_{}.npy'.format(dataset_name), nds)
    np.save('datasets/constructed_data/valid_tar_{}.npy'.format(dataset_name), nta)

    nds, nta = build_input_ds(ds_te)
    np.save('datasets/constructed_data/test_inp_{}.npy'.format(dataset_name+suffix), nds)
    np.save('datasets/constructed_data/test_tar_{}.npy'.format(dataset_name+suffix), nta)
