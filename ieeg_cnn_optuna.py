## CNN Classifier Script for sEEG RT
##
##
## Nebras M. Warsi
## Ibrahim Lab
## Jan 2021

"""
In this script the model being utilized is a simplified CNN *without* MaxPooling layers, which helps maintain relevance of timing/location in timeseries data.
Ref: Feng et al. Applied Sciences. A Novel Simplified CNN Classification Algorithm of Motor Imagery EEG Signals based on Deep Learning (2020). 
DOI: https://doi.org/10.3390/app10051605

The Optimizer being used is AdaBound (Keras), which provides advantages of both Adam and SGD. Ref: https://github.com/Luolc/AdaBound

Optuna is being used for hyperparameter optimization. Ref: Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019. 
Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD (arXiv).


"""

import sys
import tensorflow as tf
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score, roc_curve, confusion_matrix
from keras_adabound import AdaBound
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, Sequential, activations
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd
import csv
import optuna
import itertools
import gc

path = '/d/gmi/1/nebraswarsi/ML/patients/'

try:
    patients = [sys.argv[1]]
except:
    # local debugging parameters
    patients = ['SEEG-SK-04','SEEG-SK-06','SEEG-SK-09','SEEG-SK-11','SEEG-SK-15','SEEG-SK-18','SEEG-SK-19', 'SEEG-SK-20', 'SEEG-SK-21']

# Training Parameters
k = 10
epochs = 200
seed = 42
auc = tf.keras.metrics.AUC()

def ieeg_cnn(trial):

    # Initialize model and the input shape required
    model = Sequential()

    # Layer I1 and C2: Input, Convultion, and Batch Normalization
    hp_filt1 = trial.suggest_categorical("filt_1", [8, 16, 32])
    model.add(Conv2D(hp_filt1, kernel_size=(25, 1), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    hp_dropout = trial.suggest_float('dropout', low = 0.3, high = 0.7, step = 0.2) # Tunes size of dropout layer
    model.add(Dropout(hp_dropout))
            
    # Layer C3:
    hp_filt2 = trial.suggest_categorical("filt_2", [4, 8, 16, 32])
    model.add(Conv2D(hp_filt2, kernel_size=(1, 3), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    hp_dropout2 = trial.suggest_float('dropout2', low = 0.3, high = 0.7, step = 0.2) # Tunes size of 2nd dropout layer
    model.add(Dropout(hp_dropout2))

    # Layer F4:
    model.add(Flatten())
    hp_dropout_final = trial.suggest_float('dropout_final', low = 0.3, high = 0.5, step = 0.2) # Tunes size of final dropout layer
    model.add(Dropout(hp_dropout_final))

    # Layer D5:
    hp_units = trial.suggest_int('units', low = 32, high = 352, step = 64) # Tunes size of dense layer
    model.add(Dense(units = hp_units))
    model.add(Activation('relu'))
    model.add(Dropout(hp_dropout_final))

    # Layer O6 (output layer):
    model.add(Dense(1, activation='sigmoid'))

    hp_learning_rate = trial.suggest_loguniform('learning_rate', low = 1e-5, high = 1e-2)
    model.compile(loss='binary_crossentropy', optimizer=AdaBound(lr=hp_learning_rate, final_lr=1e-2, weight_decay=1e-1), metrics=['binary_accuracy', auc])
    return model

# Optuna Optimizer 
def objective(trial, x, y):
    tf.keras.backend.clear_session()
    batch_size = trial.suggest_int('batch_size', low = 16, high = 48, step = 16)

    val_metrics = []
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)    
    fold = 1

    for fold, (train_indices, val_indices) in enumerate(cv.split(x,y)):
        x_train, x_val = x[train_indices], x[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
    
        x_train, x_val = np.expand_dims(x_train, axis=-1), np.expand_dims(x_val, axis=-1)
        # Convert np arrays to tensors for main operation
        x_train, x_val, y_train, y_val = tf.convert_to_tensor(x_train, tf.float32), tf.convert_to_tensor(x_val, tf.float32), tf.convert_to_tensor(y_train, tf.float32), tf.convert_to_tensor(y_val, tf.float32)

        # Tunes for Accuracy and AUC (favors AUC)
        model_path = (path_to_save_models + "%s_%s_%s_fold%i.h5" % (str(trial.number), tt, str(contacts[contact_i]), fold+1))
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True)
        model = ieeg_cnn(trial)
        model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val), epochs=epochs, callbacks=[checkpoint], verbose=0, use_multiprocessing=True, workers=12)

        # Gets the best AUC from across the model process
        model = load_model(model_path, compile=False)
        model.compile(loss='binary_crossentropy', optimizer=AdaBound())
        y_prob = model.predict(x_val)[:,0]
        y_pred = model.predict_classes(x_val)[:,0]

        trial_auc = roc_auc_score(y_val, y_prob)
        trial_acc = accuracy_score(y_val, y_pred)
        metric = trial_acc + 1.5*trial_auc

        val_metrics.append(metric)

        # For successive halving pruner:
        intermediate_value = np.mean(val_metrics)
        trial.report(intermediate_value, (fold))

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    metric = np.mean(val_metrics)
    return metric 

def plot_confusion_matrix(cm,
                          target_names,
                          path,
                          title='Confusion matrix',
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(path + '_test_confusion_matrix.png')
    plt.close()

for patient in patients:
    # Loads data paths and then gets dataset from the H5PY file
    path_to_load = os.path.join(path, patient, 'processed')
    path_to_save = os.path.join('./CNN_results/')
    
    if not os.path.isdir(path_to_save):
        os.makedirs(path_to_save)

    dataset = h5py.File(os.path.join(path_to_load, "processed_data.h5"), 'r')

    X_full, respTimes, respSpeed, shifts = dataset.get('data_full')[()], dataset.get('respTimes')[()], dataset.get('respSpeed')[()], dataset.get('shifts')[()]
    contacts, regions = dataset.attrs['contacts'], dataset.attrs['regions']
    dataset.close()

    # Confirm valid data shapes
    print('Dataset: %s' % str(X_full.shape))
    print('Reaction times: %s' % str(respTimes.shape))
    print('Reaction speeds: %s' % str(respSpeed.shape))
    print('Shifts: %s' % str(shifts.shape))
    print('Contacts: %s' % str(contacts.shape))

    # Now we begin running the CNN model for each contact:
    for contact_i in range(len(contacts)):
        # Define save paths
        path_to_save_graphs = os.path.join(path_to_save, 'graphs', contacts[contact_i])
        path_to_save_models = os.path.join(path_to_save, 'best_models/')
        
        if not os.path.isdir(path_to_save_graphs):
            os.makedirs(path_to_save_graphs)
        if not os.path.isdir(path_to_save_models):
            os.makedirs(path_to_save_models)
        
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

            # Split into shift and non shift and trim the first and last timepoints to eliminate edge effects
            if(tt == 'shift'):
                X_all, Y_all = np.asarray(cwt_shift)[:,:,1:-1], np.asarray(Y_shift)
                
            else:
                X_all, Y_all = np.asarray(cwt_nonshift)[:,:,1:-1], np.asarray(Y_nonshift)
            
            X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, shuffle=True, test_size=0.2, random_state=seed)

            # Hyperparameter tuning with Optuna
            study = optuna.create_study(direction = 'maximize', pruner=optuna.pruners.SuccessiveHalvingPruner(reduction_factor=3))
            study.optimize(lambda trial: objective(trial, X_train, Y_train), n_trials=40, gc_after_trial=True)

            # Get the optimal hyperparameters
            best_hps = study.best_params

            print(f"""
            The hyperparameter search is complete. In the first layer, optimal filter number is {best_hps['filt_1']} and {best_hps['filt_2']} in the second layer.
            There are {best_hps['units']} neurons in the densely-connected layer, the optimal dropout for the first conv layer is {best_hps['dropout']}, {best_hps['dropout2']} in the second,
            and final dropout rate is {best_hps['dropout_final']}, optimal learning rate is {best_hps['learning_rate']}, and batch size is {best_hps['batch_size']}.
            """)

            batch_size = best_hps['batch_size']

            # Now we re-train the best model and plot its progress
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

            # Save metrics across all folds
            fold_acc = []
            fold_prec = []
            fold_rec = []
            fold_f1 = []
            fold_auc = []
            fold_fpr = []
            fold_tpr = []

            for iters, (train_indices, val_indices) in enumerate(cv.split(X_train, Y_train)):
                
                fold = iters+1
                x_train, x_val = X_train[train_indices], X_train[val_indices]
                y_train, y_val = Y_train[train_indices], Y_train[val_indices]
        
                x_train, x_val = np.expand_dims(x_train, axis=-1), np.expand_dims(x_val, axis=-1)

                # Convert np arrays to tensors for main operation
                x_train, x_val, y_train, y_val = tf.convert_to_tensor(x_train, tf.float32), tf.convert_to_tensor(x_val, tf.float32), tf.convert_to_tensor(y_train, tf.float32), tf.convert_to_tensor(y_val, tf.float32)

                # Using the optimized hyperparameters to build the 'best' model
                model = ieeg_cnn(study.best_trial)
                model.compile(loss='binary_crossentropy', optimizer=AdaBound(lr=best_hps['learning_rate'], final_lr=1e-2, weight_decay=1e-1), metrics=['binary_accuracy', auc])
                model_path = (path_to_save_models + "best_%s_%s_fold%i.h5" % (tt, str(contacts[contact_i]), fold))
                checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True)
                history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint], validation_data = (x_val, y_val), use_multiprocessing=True, workers=12)

                # Training Metrics:
                # Plots Fold Loss
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                    
                loss_path = os.path.join(path_to_save_graphs, tt, 'loss/')
                if not os.path.isdir(loss_path):
                    os.makedirs(loss_path)
                plt.savefig(loss_path + 'loss_fold_%i.png' % (fold))
                plt.close()

                # Plots Fold Accuracy
                plt.plot(history.history['binary_accuracy'])
                plt.plot(history.history['val_binary_accuracy'])
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                    
                acc_path = os.path.join(path_to_save_graphs, tt, 'acc/')
                if not os.path.isdir(acc_path):
                    os.makedirs(acc_path)
                plt.savefig(acc_path + 'acc_fold_%i.png' % (fold))
                plt.close()

                # Plots Fold AUC
                plt.plot(history.history['auc'])
                plt.plot(history.history['val_auc'])
                plt.title('model auc')
                plt.ylabel('auc')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')

                auc_path = os.path.join(path_to_save_graphs, tt, 'auc/')
                if not os.path.isdir(auc_path):
                    os.makedirs(auc_path)
                plt.savefig(auc_path + 'auc_fold_%i.png' % (fold))
                plt.close()

                # Val Metrics
                model = load_model(model_path, compile=False)
                model.compile(loss='binary_crossentropy', optimizer=AdaBound(lr=best_hps['learning_rate'], final_lr=1e-2, weight_decay=1e-1), metrics=['binary_accuracy', auc])

                Y_prob = model.predict(x_val)
                Y_pred = model.predict_classes(x_val)
                Y_prob = Y_prob[:, 0]
                Y_pred = Y_pred[:, 0]

                fold_acc.append(accuracy_score(y_val, Y_pred))
                fold_prec.append(precision_score(y_val, Y_pred))
                fold_rec.append(recall_score(y_val, Y_pred))
                fold_f1.append(f1_score(y_val, Y_pred))
                fold_auc.append(roc_auc_score(y_val, Y_prob))

                # ROC Curve Metrics for Plotting
                fpr, tpr, thresholds = roc_curve(y_val, Y_prob)
                fold_fpr.append(fpr)
                fold_tpr.append(tpr)

                # Finally, save metrics of the current fold
                full_save_path = os.path.join(path_to_save + "%s_cnn_per_fold_val_results.csv" % tt)
                with open(full_save_path, 'a', newline='') as f:
                    writer = csv.writer(f)

                    if f.tell() == 0:
                        # First time writing to file. Write header row.
                        writer.writerow(['Contact', 'Region', 'Fold Accuracy', 'Fold AUC', 'Fold Precision', 'Fold Recall', 'Fold F1'])

                    data = [contacts[contact_i], regions[contact_i], fold_acc[fold-1], fold_auc[fold-1], fold_prec[fold-1], fold_rec[fold-1], fold_f1[fold-1]]
                    writer.writerow(data)
               
                # Clean memory
                gc.collect()

            avg_acc = np.mean(fold_acc)
            avg_auc = np.mean(fold_auc)
            avg_prec = np.mean(fold_prec)
            avg_rec = np.mean(fold_rec)
            avg_f1 = np.mean(fold_f1)

            std_acc = np.std(fold_acc)
            std_prec = np.std(fold_prec)
            std_rec = np.std(fold_rec)
            std_f1 = np.std(fold_f1)
            std_auc = np.std(fold_auc)

            # Plot the ROC/AUC curves of all folds together:
            for run in range(fold-1):
                plt.plot(fold_fpr[run], fold_tpr[run], label="%s, AUC=%03f" % (str('Fold ' + str(run+1)), fold_auc[run]))
                
            plt.axis([0,1,0,1])
            plt.plot([0,1], [0,1], color='orange', linestyle='--')
            plt.xlabel('False Positive Rate') 
            plt.ylabel('True Positive Rate') 
            plt.title("%s ROC Curve\n %s \nMean AUC = %03f" % (tt, str(contacts[contact_i]), avg_auc))
            plt.legend(prop={'size':13}, loc='lower right')
            plt.savefig((os.path.join(path_to_save_graphs, tt) + '_roc_auc.png'))
            plt.close()

            # Save val metrics of the current contact
            full_save_path = os.path.join(path_to_save + "%s_cnn_val_results.csv" % tt)
            with open(full_save_path, 'a', newline='') as f:
                writer = csv.writer(f)

                if f.tell() == 0:
                    # First time writing to file. Write header row.
                    writer.writerow(['Contact', 'Region', 'Mean Accuracy', 'STDDEV Acc', 'Mean AUC', 'STDDEV AUC', 'Mean Precision', 'STDDEV Prec', 'Mean Recall', 'STDDEV Rec', 'Mean F1', 'STDDEV F1'])

                data = [contacts[contact_i], regions[contact_i], avg_acc, std_acc, avg_auc, std_auc, avg_prec, std_prec, avg_rec, std_rec, avg_f1, std_f1]
                writer.writerow(data)

            # Now we will test on the held-out set
            model = ieeg_cnn(study.best_trial)
            model.compile(loss='binary_crossentropy', optimizer=AdaBound(lr=best_hps['learning_rate'], final_lr=1e-2, weight_decay=1e-1), metrics=['binary_accuracy', auc])

            test_model_path = (path_to_save_models + "%s_%s_final.h5" % (tt, str(contacts[contact_i])))
            checkpoint = ModelCheckpoint(test_model_path, monitor='val_auc', mode='max', save_best_only=True)

            X_train, X_test = np.expand_dims(X_train, axis=-1), np.expand_dims(X_test, axis=-1)
            test_history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint], validation_data = (X_test, Y_test))
            
            # Plot training curve of the final model on the held-out test set
            plt.plot(test_history.history['loss'])
            plt.plot(test_history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
                    
            plt.savefig(loss_path + 'TEST_loss.png')
            plt.close()

            # Acc
            plt.plot(test_history.history['binary_accuracy'])
            plt.plot(test_history.history['val_binary_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
                    
            plt.savefig(acc_path + 'TEST_acc.png')
            plt.close()

            # AUC
            plt.plot(test_history.history['auc'])
            plt.plot(test_history.history['val_auc'])
            plt.title('model auc')
            plt.ylabel('auc')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')

            plt.savefig(auc_path + 'TEST_auc.png')
            plt.close()

            # Now finalized metrics for the best overall model
            model = load_model(test_model_path, compile=False)
            model.compile(loss='binary_crossentropy', optimizer=AdaBound(lr=best_hps.get('learning_rate'), final_lr=1e-2, weight_decay=1e-1), metrics=['binary_accuracy', auc])

            Y_prob = model.predict(X_test)
            Y_pred = model.predict_classes(X_test)
            Y_prob = Y_prob[:, 0]
            Y_pred = Y_pred[:, 0]

            # Test (not val) Metrics
            test_acc = accuracy_score(Y_test, Y_pred)
            test_prec = precision_score(Y_test, Y_pred)
            test_rec = recall_score(Y_test, Y_pred)
            test_f1 = f1_score(Y_test, Y_pred)
            test_auc = roc_auc_score(Y_test, Y_prob)
            test_cm = confusion_matrix(Y_test, Y_pred)

            # Plot test CM
            plot_confusion_matrix(cm=test_cm, target_names = ['fast', 'slow'], path = (path_to_save_graphs + '/%s' % tt), title='Confusion matrix', normalize=True)

            # ROC Curve Metrics for Plotting
            fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
            fold_fpr.append(fpr)
            fold_tpr.append(tpr)

            # Save test metrics of the current contact
            full_save_path = os.path.join(path_to_save + "%s_cnn_test_results.csv" % tt)
            with open(full_save_path, 'a', newline='') as f:
                writer = csv.writer(f)

                if f.tell() == 0:
                    # First time writing to file. Write header row.
                    writer.writerow(['Contact', 'Region', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'Layer One Filters', 'Layer Two Filters', 'Dense Layer Neurons', 'Learning Rate', 'Batch Size', 'First Dropout', 'Second Dropout', 'Final Dropout'])

                data = [contacts[contact_i], regions[contact_i], test_acc, test_auc, test_prec, test_rec, test_f1, str(best_hps['filt_1']), 
                    str(best_hps['filt_2']), str(best_hps['units']), str(best_hps['learning_rate']), str(best_hps['batch_size']), 
                    str(best_hps['dropout']), str(best_hps['dropout2']), str(best_hps['dropout_final']),]
                writer.writerow(data)
