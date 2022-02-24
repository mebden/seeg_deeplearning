
'''

iEEG Machine Learning Dataset Creation Script. Currently loads in the data from .mat files for each trial
and then coverts things to MNE Epoch objects (ref: https://github.com/mne-tools/mne-python). We then use
an autorejection cross-validated algorithm (ref: https://autoreject.github.io/) for artifact detection and repair of affected epochs
Jas et al. Autoreject: Automated artifact rejection for MEG and EEG data (2016)

Nebras M. Warsi
Dec 2020

'''

import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import scipy
import mne
from nilearn import plotting
from autoreject import AutoReject
from tqdm import tqdm
from datafuncs import *
from PLI_time import calculate_PLV_WPLI_across_time
import gc

matplotlib.use('Agg')

path = "/d/gmi/1/nebraswarsi/Network Analysis/patients/"
plot = False
Fs = 2048
new_Fs = 256 # Downsamp freq
percentile = 50

n_bootstrap = 200
epochs = {
    '1': [64, 192],
    '2': [192, 320],
    '3': [320, 448]
}
freqbands = {
    'theta': [4, 8], 
    'alpha': [8, 12], 
    'beta': [12, 30] 
    }
n_freq_bands = len(freqbands)

if __name__ == "__main__":

    try:
        patients = [sys.argv[1]]
    except:
        # local debugging parameters
        patients = []

    for patient in patients:

        path_to_load = os.path.join(path, patient, 'raw')
        path_to_save = os.path.join(path, patient, 'processed')

        print(path_to_save)

        if not os.path.isdir(path_to_save):
            os.makedirs(path_to_save)

        # Loads data 
        shifted_respTimes, unshifted_respTimes, respTimes, shifts, contact_info = load_data(path_to_load)
        contacts = contact_info['contact'].values
        x, y, z = contact_info['x'].values, contact_info['y'].values, contact_info['z'].values
        coords = np.stack((x, y, z), axis=1)

        X_full = []
        Y = []

        for i in range(respTimes.shape[0]):
            trialindex = "%03d" % (i + 1)
            f = h5py.File(os.path.join(path_to_load, "trialdata_" + trialindex + ".mat"), 'r')
            # Holds the trial data for each contact over the 4096 timepoints (num_contacts x time)
            trialData_full = f['trialdata'][:]
            trialData_full = np.transpose(trialData_full)
            X_full.append(trialData_full)
            Y.append(respTimes[i])

        X_full = np.asarray(X_full)
        Y = np.asarray(Y)

        # Now we are ready to split shift and non shift trials
        X_shift = []
        Y_shift_RT = []
        X_nonshift = []
        Y_nonshift_RT = []

        shift_idx = 0
        for is_shift in shifts:
            if is_shift:
                X_shift.append(X_full[shift_idx,:,:])
                Y_shift_RT.append(Y[shift_idx])
            else:
                X_nonshift.append(X_full[shift_idx,:,:])
                Y_nonshift_RT.append(Y[shift_idx])
            shift_idx += 1

        # Now we will get the fast vs. slow dichotomization based on RT for each trial type
        Y_i, Y_shift, Y_nonshift = calculate_percentile(Y, Y_shift_RT, Y_nonshift_RT, shifts, percentile=percentile)

        X_shift = np.asarray(X_shift)
        Y_shift = np.asarray(Y_shift)
        X_nonshift = np.asarray(X_nonshift)
        Y_nonshift = np.asarray(Y_nonshift)

        # Resample broadband signal to 256Hz for saving
        X_full = mne.filter.resample(X_full, up=(new_Fs/Fs))

        # #######
        #
        # Next we will autoreject the "bad" epochs
        # For cleaner data
        
        try: # Some patients won't have localization data

            # MNE data
            ch_names = tuple(contact_info['contact'])
            ch_coords = contact_info[['x', 'y', 'z']].to_numpy(dtype=float)
            # Remove NaNs as these would stop autoreject from running
            NaN_idx = np.isnan(ch_coords)
            ch_coords[NaN_idx] = 1
            ch_coords = ch_coords / 1000  # coordinates are in mm, but MNE requires m
            ch_pos = dict(zip(ch_names, ch_coords))
            montage = mne.channels.make_dig_montage(ch_pos, coord_frame='mri')

            # Create MNE Epoch Object with relevent montage info for the sEEG
            info = mne.create_info(ch_names, Fs, 'eeg')
            X_ar = mne.EpochsArray(X_full, info, tmin=-1)  # No baseline of raw to avoid distortion of phase
            X_ar.set_montage(montage)

            # Autorejection Algo
            autoreject = AutoReject(n_jobs=-1)
            autoreject.fit(X_ar)
            reject_log = autoreject.get_reject_log(X_ar)

            # Find indices of trials that were dropped and verify them on screen
            dropped_trial_idx = [i for i, x in enumerate(reject_log.bad_epochs) if x]
            print(dropped_trial_idx)

            X_full = np.delete(X_full, dropped_trial_idx, axis=0)
            Y = np.delete(Y, dropped_trial_idx)
            shifts = np.delete(shifts, dropped_trial_idx)

        except: # For patients without localization data, we skip and print message

            print('Skipping autoreject... Patient localization data missing')

        # Compute the CWT for power data
        freqs = np.logspace(np.log10(4), np.log10(100), 50)
        cwt = mne.time_frequency.tfr_array_morlet(
            X_full, sfreq=new_Fs, freqs=freqs,
            n_cycles=np.logspace(np.log10(2), np.log10(7), 50))

        dataPower = np.abs(cwt)
        dataPower = 10 * np.log10(
            dataPower
            /
            np.mean(np.mean(dataPower[:, :, :, :], axis=-1), axis=0)[None, :, :, None]
        )

        # Now we will save our data to an H5PY file
        h5py_file = h5py.File(os.path.join(path_to_save, "processed_data.h5"), 'w')
        h5py_file.create_dataset('broadband', data=X_full)
        h5py_file.create_dataset('dataPower', data=dataPower)
        h5py_file.create_dataset('shifts', data=shifts)
        h5py_file.create_dataset('respTimes', data=respTimes)
        h5py_file.create_dataset('respSpeed', data = Y_i)

        h5py_file.attrs['contacts'] = contact_info['contact'].to_list()
        h5py_file.attrs['x'] = contact_info['x'].to_list()
        h5py_file.attrs['y'] = contact_info['y'].to_list()
        h5py_file.attrs['z'] = contact_info['z'].to_list()
        h5py_file.attrs['regions'] = contact_info['aal_label'].to_list()
        h5py_file.attrs['FIND_networks'] = contact_info['FIND_label'].to_list()
        h5py_file.attrs['YEO_networks'] = contact_info['yeo_label'].to_list()
        h5py_file.close()

        print('---------'*5)
        print('***Data Saved***')
        print('---------'*5)

        gc.collect()
