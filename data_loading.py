import mne
import os
from glob import glob
import numpy as np
import random
import keras.utils
from matplotlib import pyplot as plt


def read_data(param):
    epochs_target = []
    epochs_non_target = []
    try:
        root, dirs, files = next(os.walk(param.path))

        for folder_name in dirs:
            path_file_vhdr = param.path + folder_name + '/Data/*.vhdr'
            file_name_vhdr = glob(path_file_vhdr)
            if len(file_name_vhdr) == 0:
                continue
            raw = mne.io.read_raw_brainvision(file_name_vhdr[0], preload=True)
            raw.filter(l_freq=param.l_freq, h_freq=param.h_freq)
            if (raw.info['nchan'] == 4):
                raw = raw.drop_channels(['EOG'])

            path_file_txt = param.path + folder_name + '/Data/*.txt'
            file_name_txt = glob(path_file_txt)
            if len(file_name_txt) == 0:
                continue
            loaded_txt = open(file_name_txt[0], "r")
            text = loaded_txt.readlines()
            line = text[2]
            event_id_target = int(line.split(": ")[1])
            events_loaded = mne.events_from_annotations(raw)
            epochs_t_subject = mne.Epochs(raw, events=events_loaded[0], event_id=event_id_target, tmin=param.t_min,
                                          tmax=param.t_max, baseline=param.baseline)
            epochs_t_subject = mne.epochs.combine_event_ids(epochs_t_subject, [str(event_id_target)], {'target': 0})

            non_target_random = random.randint(1, 9)
            while event_id_target == non_target_random:
                non_target_random = random.randint(1, 9)
            epochs_n_subject = mne.Epochs(raw, events=events_loaded[0], event_id=non_target_random, tmin=param.t_min,
                                          tmax=param.t_max,
                                          baseline=param.baseline)
            epochs_n_subject = mne.epochs.combine_event_ids(epochs_n_subject, [str(non_target_random)],
                                                            {'non_target': 11})
            epochs_target.append(epochs_t_subject)
            epochs_non_target.append(epochs_n_subject)
            loaded_txt.close()
    except StopIteration:
        pass
        print("Error ocurred:")
        print("Directory with dataset does not found!")
        print("Program will be terminated")
        exit(1)

    epochs_target = mne.concatenate_epochs(epochs_target)
    reject = dict(eeg=param.amplitude)
    epochs_target.drop_bad(reject=reject)

    epochs_non_target = mne.concatenate_epochs(epochs_non_target)
    epochs_non_target.drop_bad(reject=reject)
    epochs_all = mne.concatenate_epochs([epochs_target, epochs_non_target])

    scalings = dict(eeg=param.scaling)
    scaler = mne.decoding.Scaler(epochs_all.info, scalings=scalings)

    X = epochs_all.get_data()
    X = scaler.fit_transform(X)

    out_t_labels = keras.utils.to_categorical(epochs_target.events[:, 2], 2)
    out_n_labels = keras.utils.to_categorical(epochs_non_target.events[:, 2] - 10, 2)

    out_labels = np.vstack((out_t_labels, out_n_labels))

    return X, out_labels
