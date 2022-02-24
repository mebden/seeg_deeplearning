'''

iEEG CNN Heatmap / feature importance mapper from shap using shapley values
https://github.com/slundberg/shap

This script is to be run after the relevant final models have been produced to explain the features used by the best model

This script further uses a simple clustering algorithm to identify whether a given contact is predictive 
in the pre-, peri-, or post-stimulus phases (or some combination of all three). This can then be used to 
determine the relevance of a given contact in temporal relation to the stimulus.

'''

import tensorflow as tf
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from keras_adabound import AdaBound
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, Sequential, activations
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import math
import h5py
import csv
import gc
import shap
from shap.explainers._gradient import Gradient
# import cv2

path = '/d/gmi/1/nebraswarsi/ML/patients/'

try:
    patients = [sys.argv[1]]
except:
    # local debugging parameters

    patients = ['SEEG-SK-02', 'SEEG-SK-04','SEEG-SK-06','SEEG-SK-07','SEEG-SK-09','SEEG-SK-11','SEEG-SK-14'
    'SEEG-SK-15', 'SEEG-SK-18','SEEG-SK-19', 'SEEG-SK-20', 'SEEG-SK-21', 'SEEG-SK-24', 'SEEG-SK-26']

# Training Parameters
epochs = 200
seed = 42
auc = tf.keras.metrics.AUC()

# CWT Parameters
step = 128
Fs = 2048
t = np.linspace(-1, 1, int((2*Fs)/step))[1:-1] # Because we trimmed image edges in CNN script
pre_stim = np.argwhere(t < -0.25)
peri_stim = np.argwhere(((t >= -0.25) & (t <= 0.25)))
post_stim = np.argwhere((t > 0.25))
stim = np.argwhere(t == np.amin(np.abs(t)))

freqs = np.logspace(math.log10(4), math.log10(80), 25)
theta = np.argwhere(freqs <= 7)
alpha = np.argwhere((freqs > 7) & (freqs <= 12))
beta = np.argwhere((freqs > 12) & (freqs <= 30))
gamma = np.argwhere(freqs > 30)

# Unified Plotting Details
yticks = (int(theta[0]), int(theta[-1]), int(alpha[-1]), int(beta[-1]), int(gamma[-1]))
ylabels = ('$4$', '$7$', '$12$', '$30$', '$80$')
y_title = 'Freq (Hz)'
xticks = (int(pre_stim[0]), 7, int(stim), 22, int(post_stim[-1]))
xlabels = ('$-930$', '$-500$', '$0$', '$500$', '$930$')
x_title = 'Time (ms)'

###
# This script will first attempt to load the best saved test model from the runs
# However in case that file is lost and we are unable to load model weights, it will then 
# load the hp data from the CSV files and re-train/checkpoint the best model
###

def ieeg_cnn(hp_filt1, hp_filt2, hp_dense, hp_lr, hp_do1, hp_do2, hp_do_last):

    # Initialize model and the input shape required
    model = Sequential()

    # Layer I1 and C2: Input, Convultion, and Batch Normalization
    model.add(Conv2D(hp_filt1, kernel_size=(25, 1), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(hp_do1))
            
    # Layer C3:
    model.add(Conv2D(hp_filt2, kernel_size=(1, 3), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(hp_do2))

    # Layer F4:
    model.add(Flatten())
    model.add(Dropout(hp_do_last))

    # Layer D5:
    model.add(Dense(units = hp_dense))
    model.add(Activation('relu'))
    model.add(Dropout(hp_do_last))

    # Layer O6 (output layer):
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=AdaBound(lr=hp_lr, final_lr=1e-2, weight_decay=1e-1), metrics=['binary_accuracy', auc])
    return model

for patient in patients:
    # Loads data paths and then gets dataset from the H5PY file
    path_to_load = os.path.join(path, patient, 'processed')
    path_to_save = os.path.join(path, patient, 'CNN_results/')
    model_path = os.path.join(path_to_save, 'best_models/')

    if not os.path.isdir(path_to_save):
        print("patient %s data missing! Skipping..." % patient)
        continue

    # This means the best models have not been saved
    if not os.path.isdir(model_path):
        
        os.makedirs(model_path)
        reload_hps = True
        print('----'*5)
        print('Best models NOT saved, loading HP data from file...')
        print('----'*5)

        # Now load the best HP data gathered from main script
        # loads best HPs from the saved CSV file
        hp_file_shift = os.path.join(path_to_save + "shift_cnn_test_results.csv")
        hp_file_nonshift = os.path.join(path_to_save + "nonshift_cnn_test_results.csv")

        with open(hp_file_shift, newline='\n') as csvfile:
            hp_data_shift = pd.read_csv(csvfile)
            csvfile.close()

        with open(hp_file_nonshift, newline='\n') as csvfile:
            hp_data_non_shift = pd.read_csv(csvfile)
            csvfile.close()        

        # Now extract data from the dfs in numpy arrays
        # Shift
        hp_filt1_s = hp_data_shift['Layer One Filters'].values
        hp_filt2_s = hp_data_shift['Layer Two Filters'].values
        hp_dense_s = hp_data_shift['Dense Layer Neurons'].values
        hp_lr_s = hp_data_shift['Learning Rate'].values
        hp_bs_s = hp_data_shift['Batch Size'].values
        hp_dropout1_s = hp_data_shift['First Dropout'].values
        hp_dropout2_s = hp_data_shift['Second Dropout'].values
        hp_dropout_last_s = hp_data_shift['Final Dropout'].values

        # Non-shift
        hp_filt1_ns = hp_data_non_shift['Layer One Filters'].values
        hp_filt2_ns = hp_data_non_shift['Layer Two Filters'].values
        hp_dense_ns = hp_data_non_shift['Dense Layer Neurons'].values
        hp_lr_ns = hp_data_non_shift['Learning Rate'].values
        hp_bs_ns = hp_data_non_shift['Batch Size'].values
        hp_dropout1_ns = hp_data_non_shift['First Dropout'].values
        hp_dropout2_ns = hp_data_non_shift['Second Dropout'].values
        hp_dropout_last_ns = hp_data_non_shift['Final Dropout'].values
    
    else:
        reload_hps = False
    
    ### 
    # Now we will perform the calculations of SHAP values on a per-contact basis
    # and plot the relevant figures
    ###

    dataset = h5py.File(os.path.join(path_to_load, "processed_data.h5"), 'r')

    X_full, respTimes, respSpeed, shifts = dataset.get('data_full')[()], dataset.get('respTimes')[()], dataset.get('respSpeed')[()], dataset.get('shifts')[()]
    contacts, regions = dataset.attrs['contacts'], dataset.attrs['regions']
    dataset.close()

    # Confirm valid data shapes
    print('Dataset: %s' % str(X_full.shape))
    print('Reaction times: %s' % str(respTimes.shape))
    print('Reaction speeds: %s' % str(respSpeed.shape))
    print('Trials: %s' % str(shifts.shape))
    print('Contacts: %s' % str(contacts.shape))

    for contact_i in range(len(contacts)):   
        # Gets the CWT for given contact
        contact_cwt = X_full[contact_i]

        # Now we need to split shift / nonshift, don't need to flatten as we are using CNN
        cwt_shift = []
        cwt_nonshift = []
        Y_shift = []
        Y_nonshift = []

        for i in range(len(contact_cwt)):
            if shifts[i]:
                cwt_shift.append(contact_cwt[i])
                Y_shift.append(respSpeed[i])
            else:
                cwt_nonshift.append(contact_cwt[i])
                Y_nonshift.append(respSpeed[i])

        for tt in ['shift', 'nonshift']:
            
            # In this case, best model was NOT saved and we will re-train using best HPs
            if reload_hps:
                # Split into shift and non shift and trim the first and last timepoints to eliminate edge effects
                if(tt == 'shift'):
                    X_all, Y_all = np.asarray(cwt_shift)[:,:,1:-1], np.asarray(Y_shift)
                    hp_filt1 = hp_filt1_s[contact_i]
                    hp_filt2 = hp_filt2_s[contact_i]
                    hp_dense = hp_dense_s[contact_i]
                    hp_lr = hp_lr_s[contact_i]
                    hp_bs = hp_bs_s[contact_i]
                    hp_do1 = hp_dropout1_s[contact_i]
                    hp_do2 = hp_dropout2_s[contact_i]
                    hp_do_last = hp_dropout_last_s[contact_i]

                else:
                    X_all, Y_all = np.asarray(cwt_nonshift)[:,:,1:-1], np.asarray(Y_nonshift)
                    hp_filt1 = hp_filt1_ns[contact_i]
                    hp_filt2 = hp_filt2_ns[contact_i]
                    hp_dense = hp_dense_ns[contact_i]
                    hp_lr = hp_lr_ns[contact_i]
                    hp_bs = hp_bs_ns[contact_i]
                    hp_do1 = hp_dropout1_ns[contact_i]
                    hp_do2 = hp_dropout2_ns[contact_i]
                    hp_do_last = hp_dropout_last_ns[contact_i]
                
                X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, shuffle=True, test_size=0.2, random_state=seed)
                X_train, X_test, Y_train, Y_test = tf.convert_to_tensor(X_train, tf.float32), tf.convert_to_tensor(X_test, tf.float32), tf.convert_to_tensor(Y_train, tf.float32), tf.convert_to_tensor(Y_test, tf.float32)
                X_plot = X_test

                # Re-create and save best model
                model = ieeg_cnn(hp_filt1, hp_filt2, hp_dense, hp_lr, hp_do1, hp_do2, hp_do_last)
                best_model_path = (model_path + "%s_%s_final.h5" % (tt, str(contacts[contact_i])))
                checkpoint = ModelCheckpoint(best_model_path, monitor='val_auc', mode='max', save_best_only=True)
                X_train, X_test = np.expand_dims(X_train, axis=-1), np.expand_dims(X_test, axis=-1)
                model.fit(X_train, Y_train, epochs=epochs, batch_size=hp_bs, callbacks=[checkpoint], validation_data = (X_test, Y_test), verbose=0)

            # In this cases (likely updated scripts) we have saved the best test models and can simply load as follows:
            else:
                if(tt == 'shift'):
                    X_all, Y_all = np.asarray(cwt_shift)[:,:,1:-1], np.asarray(Y_shift)
                else:
                    X_all, Y_all = np.asarray(cwt_nonshift)[:,:,1:-1], np.asarray(Y_nonshift)

                X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, shuffle=True, test_size=0.2, random_state=seed)
                X_train, X_test, Y_train, Y_test = tf.convert_to_tensor(X_train, tf.float32), tf.convert_to_tensor(X_test, tf.float32), tf.convert_to_tensor(Y_train, tf.float32), tf.convert_to_tensor(Y_test, tf.float32)
                X_plot = X_test
                X_train, X_test = np.expand_dims(X_train, axis=-1), np.expand_dims(X_test, axis=-1)

                best_model_path = (model_path + "%s_%s_final.h5" % (tt, str(contacts[contact_i])))
            
            # Now we will explain the model's predicitions on the test set using SHAP values
            # We can do this both for individual predictions and to build and overall "heatmap"
            # of feature importance
            try:
                model = load_model(best_model_path, compile=False)
            except:
                ##
                # In this case, the model file is missing (likely on of the few runs that crashed on HPF)
                # so we skip this contact 
                ##
                continue

            model.compile(loss='binary_crossentropy', metrics=['binary_accuracy', auc])

            shap_path = os.path.join(path_to_save, 'SHAP_plots', contacts[contact_i])
            if not os.path.isdir(shap_path):
                os.makedirs(shap_path)
    
            e = Gradient(model, X_train)
            Y_pred = (model.predict(X_test) > 0.5).astype("int32")[:,0]
            prediction_accuracy = (Y_pred == Y_test)
            
            # Sort different trial classification types 
            fast_idx = []
            slow_idx = []
            true_fast_idx = []
            true_slow_idx = []
            false_fast_idx = []
            false_slow_idx = []

            for idx in range(len(Y_pred)):
                if (prediction_accuracy[idx]):
                    if (Y_pred[idx] == 0): 
                        true_fast_idx.append(idx)
                        fast_idx.append(idx)
                    else:
                        true_slow_idx.append(idx)
                        slow_idx.append(idx)
                else:
                    if (Y_pred[idx] == 0): 
                        false_fast_idx.append(idx)
                        fast_idx.append(idx)
                    else:
                        false_slow_idx.append(idx)
                        slow_idx.append(idx)

            fast_explain = X_test[(fast_idx)]
            slow_explain = X_test[(slow_idx)]
            true_fast_explain = X_test[(true_fast_idx)]
            true_slow_explain = X_test[(true_slow_idx)]
            false_fast_explain = X_test[false_fast_idx]
            false_slow_explain = X_test[false_slow_idx]

            # Now we can plot the shapley values for individual per-trial predictions
            # that can explain why the model made the given prediction on each trial example
            # note that it may not have predicted any within a given rung, so we will check this before plotting 

            if (len(true_fast_explain) != 0):

                shap_values = e.shap_values(true_fast_explain)
                shap.plots.image(shap_values, pixel_values=true_fast_explain, show=False)
                plt.suptitle(('%s:\nShapley value based explanation for correctly \npredicted fast %s trials' % (contacts[contact_i], tt)), fontsize=14)
                plt.savefig(((shap_path + '/%s_correct_fast_shapleys.png') % (tt)))
                plt.close()

            if (len(true_slow_explain) != 0):

                shap_values = e.shap_values(true_slow_explain)
                shap.plots.image(shap_values, pixel_values=true_slow_explain, width=100, aspect=3, show=False)
                plt.suptitle(('%s:\nShapley value based explanation for correctly \npredicted slow %s trials' % (contacts[contact_i], tt)), fontsize=14)
                plt.savefig(((shap_path + '/%s_correct_slow_shapleys.png') % (tt)))
                plt.close()

            if (len(false_fast_explain) != 0):

                shap_values = e.shap_values(false_fast_explain)
                shap.plots.image(shap_values, pixel_values=false_fast_explain, width=100, aspect=3, show=False)
                plt.suptitle(('%s:\nShapley value based explanation for %s trials\nmis-classified as fast' % (contacts[contact_i], tt)), fontsize=14)
                plt.savefig(((shap_path + '/%s_misclass_fast_shapleys.png') % (tt)))
                plt.close()

            if (len(false_slow_explain) != 0):

                shap_values = e.shap_values(false_slow_explain)
                shap.plots.image(shap_values, pixel_values=false_slow_explain, width=100, aspect=3, show=False)
                plt.suptitle(('%s:\nShapley value based explanation for %s trials\nmis-classified as slow' % (contacts[contact_i], tt)), fontsize=14)
                plt.savefig(((shap_path + '/%s_misclass_slow_shapleys.png') % (tt)))
                plt.close()

            '''         
            For publication figures and ease of interpretability, we will also look at plotting "difference" plots
            since we want our classifiers to be able to identify slow trials, we will make a difference plot 
            of slow vs. fast trials to indentify the features that aid in distinguishing a slow trial from a fast one

            Essentially, we are looking now at the average shapley values for the CORRECT fast and slow classifications and will compare with 
            averaged feature map
            '''

            shap_values_fast = np.mean(e.shap_values(true_fast_explain), axis=1)
            shap_values_slow = np.mean(e.shap_values(true_slow_explain), axis=1)
            slow_diff_shaps = np.subtract(shap_values_slow, shap_values_fast)
            shap_scale = np.amax(np.abs(slow_diff_shaps))/2

            explain_fast_avg = np.mean(true_fast_explain, axis=0)
            explain_slow_avg = np.mean(true_slow_explain, axis=0)
            slow_diff_morls = np.subtract(explain_slow_avg, explain_fast_avg)
            morl_scale = np.amax(np.abs(slow_diff_morls))/3
            
            # Formatting
            fig, (ax1, ax2) = plt.subplots(1, 2)        
            divider_morl, divider_shap = make_axes_locatable(ax1), make_axes_locatable(ax2)
            cax_morl = divider_morl.append_axes('bottom', size='10%', pad=0.75)
            cax_shap = divider_shap.append_axes('bottom', size='10%', pad=0.75)
            ax1.set_yticks(yticks); ax1.set_yticklabels(ylabels); ax1.set_xticks(xticks); ax1.set_xticklabels(xlabels)
            ax1.set_xlabel(x_title); ax1.set_ylabel(y_title)
            ax2.set_yticks(yticks); ax2.set_yticklabels(ylabels); ax2.set_xticks(xticks); ax2.set_xticklabels(xlabels)

            morl_im = ax1.imshow(slow_diff_morls, origin='lower', cmap='seismic', vmin=-morl_scale, vmax=morl_scale)
            ax1.set_title('Slow rel. Fast')
            cb_morl = fig.colorbar(morl_im, cax=cax_morl, label="Power (Z)", orientation="horizontal")
            cb_morl.outline.set_visible(False)

            shap_im = ax2.imshow(np.reshape(slow_diff_shaps, (X_all.shape[1], X_all.shape[2])), origin='lower', cmap='PiYG', vmin=-shap_scale, vmax=shap_scale)
            ax2.set_title('Pred. Slow')
            cb_shap = fig.colorbar(shap_im, cax=cax_shap, label="SHAP", orientation="horizontal")
            cb_shap.outline.set_visible(False)
            cb_shap.set_ticks([])

            fig.suptitle(('%s: SHAP explanation of RT\n(%s trials)' % (contacts[contact_i], tt)), fontsize=12)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(((shap_path + '/%s_AVG_SHAP.png') % (tt)), bbox_inches="tight")
            plt.clf()

            ###
            # Now that descriptive Shap values have been plotted, we will determine the epoch at which the contact
            # is predictive. For this we will need the exact timing and freq parameters:
            ###

            # Significance threshold (p < 0.05) for masking plots
            sig_thresh = 1.645
            shap_map = np.abs(np.reshape(slow_diff_shaps, (X_all.shape[1], X_all.shape[2])))
            mean_shap = np.mean(shap_map)
            std_shap = np.std(shap_map)

            # Now we convert the SHAP map into a Z-score map
            for freq in range(shap_map.shape[0]):
                shap_map[freq] = [(shap_map[freq, time] - mean_shap)/std_shap for time in range(shap_map.shape[1])]
            
            # Mask the non-sig values in power and SHAP graphs
            sig_shaps = np.ma.masked_where((shap_map < sig_thresh), shap_map)
            denoise_mask = np.where(shap_map >= sig_thresh, 1, 0)
            denoise_mask = cv2.filterSpeckles(denoise_mask.astype(np.uint8), 0, 1, 1)[0]  # Removes single pixel artefacts
            sig_shaps = np.ma.masked_where((denoise_mask == 0), sig_shaps)
            slow_diff_morls = np.ma.masked_where((sig_shaps == 0), np.reshape(slow_diff_morls, (X_all.shape[1], X_all.shape[2])))

            fig, (ax1) = plt.subplots(1, 1)            
            divider_morl = make_axes_locatable(ax1)
            cax_morl = divider_morl.append_axes('bottom', size='10%', pad=0.75)
            ax1.set_yticks(yticks); ax1.set_yticklabels(ylabels); ax1.set_xticks(xticks); ax1.set_xticklabels(xlabels)
            ax1.set_xlabel(x_title); ax1.set_ylabel(y_title)

            morl_im = ax1.imshow(slow_diff_morls, origin='lower', cmap='seismic', vmin=-morl_scale, vmax=morl_scale)
            cb_morl = fig.colorbar(morl_im, cax=cax_morl, label="Power (Z)", orientation="horizontal")
            cb_morl.outline.set_visible(False)
            
            fig.suptitle(('%s: Sig. Predictors of RT\n(%s trials)' % (contacts[contact_i], tt)), fontsize=12)
            fig.savefig(((shap_path + '/%s_SIG_SHAP.png') % (tt)), bbox_inches="tight")
            plt.clf()

            ##
            # Now we can analyze the SHAP maps to determine the relative contribution of each 
            # epoch and frequency band to the contact's predictions
            ##

            sig_shaps = np.ma.filled(sig_shaps, 0)
            # Epoch contributions
            prestim_contrib = np.sum(sig_shaps[:,pre_stim])
            peristim_contrib = np.sum(sig_shaps[:,peri_stim])
            poststim_contrib = np.sum(sig_shaps[:,post_stim])
            total_shaps = prestim_contrib + peristim_contrib + poststim_contrib
            
            prestim_contrib = prestim_contrib/total_shaps
            peristim_contrib = peristim_contrib/total_shaps
            poststim_contrib = poststim_contrib/total_shaps
            all_epochs = [prestim_contrib, peristim_contrib, poststim_contrib]
            epoch_labels = ['pre-stim', 'peri-stim', 'post-stim']
            best_epoch = epoch_labels[np.argmax(all_epochs)]
            
            # Frequency contributions
            theta_contrib = np.sum(sig_shaps[theta,:])/total_shaps
            alpha_contrib = np.sum(sig_shaps[alpha,:])/total_shaps
            beta_contrib = np.sum(sig_shaps[beta,:])/total_shaps
            gamma_contrib = np.sum(sig_shaps[gamma,:])/total_shaps

            # Temporal contributions by frequency
            prestim_theta = np.sum(sig_shaps[theta[:,None], pre_stim])/total_shaps
            prestim_alpha = np.sum(sig_shaps[alpha[:,None], pre_stim])/total_shaps
            prestim_beta = np.sum(sig_shaps[beta[:,None], pre_stim])/total_shaps
            prestim_gamma = np.sum(sig_shaps[gamma[:,None], pre_stim])/total_shaps

            peristim_theta = np.sum(sig_shaps[theta[:,None], peri_stim])/total_shaps
            peristim_alpha = np.sum(sig_shaps[alpha[:,None], peri_stim])/total_shaps
            peristim_beta = np.sum(sig_shaps[beta[:,None], peri_stim])/total_shaps
            peristim_gamma = np.sum(sig_shaps[gamma[:,None], peri_stim])/total_shaps

            poststim_theta = np.sum(sig_shaps[theta[:,None], post_stim])/total_shaps
            poststim_alpha = np.sum(sig_shaps[alpha[:,None], post_stim])/total_shaps
            poststim_beta = np.sum(sig_shaps[beta[:,None], post_stim])/total_shaps
            poststim_gamma = np.sum(sig_shaps[gamma[:,None], post_stim])/total_shaps

            # Try plotting heat map of epoch/frequency per contact:
            contribs = [[prestim_theta, prestim_alpha, prestim_beta, prestim_gamma], 
                    [peristim_theta, peristim_alpha, peristim_beta, peristim_gamma], 
                    [poststim_theta, poststim_alpha, poststim_beta, poststim_gamma]]

            contribs = np.transpose(contribs)

            fig, (ax1) = plt.subplots(1, 1)            
            divider_heatmap = make_axes_locatable(ax1)
            cax_heatmap = divider_heatmap.append_axes('bottom', size='10%', pad=0.75)
            ax1.set_yticks([0, 1, 2, 3]); ax1.set_yticklabels(['theta', 'alpha', 'beta', 'gamma']); ax1.set_xticks([0, 1, 2]); ax1.set_xticklabels(['pre-stim', 'peri-stim', 'post-stim'])
            ax1.set_xlabel('Epoch'); ax1.set_ylabel('Freq. Band')

            heatmap = ax1.imshow(contribs, origin='lower', cmap='hot')
            cb_morl = fig.colorbar(heatmap, cax=cax_heatmap, label="Relative contribution", orientation="horizontal")
            cb_morl.outline.set_visible(False)
            
            fig.suptitle(('%s: Prediction Contributors\n(%s trials)' % (contacts[contact_i], tt)), fontsize=12)
            fig.savefig(((shap_path + '/%s_SHAP_per_epoch.png') % (tt)), bbox_inches="tight")
            plt.clf()

            # Save test metrics of the current contact
            shap_save_path = os.path.join(path_to_save + "%s_SHAP_epoch_results.csv" % tt)
            with open(shap_save_path, 'a', newline='') as f:
                writer = csv.writer(f)

                if f.tell() == 0:
                    # First time writing to file. Write header row.
                    writer.writerow(['Index', 'Contact', 'Best Epoch', 'Pre-Stim', 'Peri-Stim', 'Post-Stim',
                        'Theta', 'Alpha', 'Beta', 'Gamma', 
                        'Theta Pre', 'Alpha Pre', 'Beta Pre', 'Gamma Pre',
                        'Theta Peri', 'Alpha Peri', 'Beta Peri', 'Gamma Peri',
                        'Theta Post', 'Alpha Post', 'Beta Post', 'Gamma Post'])

                data = [contact_i, contacts[contact_i], best_epoch, prestim_contrib, peristim_contrib, poststim_contrib,
                theta_contrib, alpha_contrib, beta_contrib, gamma_contrib,
                prestim_theta, prestim_alpha, prestim_beta, prestim_gamma,
                peristim_theta, peristim_alpha, peristim_beta, peristim_gamma,
                poststim_theta, poststim_alpha, poststim_beta, poststim_gamma
                ]

                writer.writerow(data)

        print('Plotted %s!' % (contacts[contact_i]))