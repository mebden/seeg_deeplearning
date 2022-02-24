'''

iEEG Machine Learning Summary Statistics and Glass Brains for post-hoc analytics

Nebras M. Warsi
Jan 2021

'''


import numpy as np
import pandas as pd
import os
import csv
from nilearn import plotting
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
import statsmodels.api as sm
from statsmodels.formula.api import ols

path = "/d/gmi/1/nebraswarsi/ML/patients/"
path_to_save = '/d/gmi/1/nebraswarsi/ML/analysis/ml_summary/'
FIND_path = '/d/gmi/1/nebraswarsi/ML/analysis/ml_summary/FIND_graphs/'
YEO_path = '/d/gmi/1/nebraswarsi/ML/analysis/ml_summary/YEO_graphs/'

if not os.path.exists(path_to_save):
    os.mkdir(path_to_save)
if not os.path.exists(FIND_path):
    os.mkdir(FIND_path)
if not os.path.exists(YEO_path):
    os.mkdir(YEO_path)

patients = []

shift_data = []
non_shift_data = []

def plot_glassbrains(data, path, tt):

    if not os.path.exists(path):
        os.mkdir(path)

    contacts = np.asarray(data['Contact'])
    find_networks = np.array(data['FIND Network'])
    yeo_networks = np.asarray(data['YEO Network'])
    prestim = np.asarray(data['Pre Weight'])
    peristim = np.asarray(data['Peri Weight'])
    poststim = np.asarray(data['Post Weight'])
    x = np.asarray(data['x'])
    y = np.asarray(data['y'])
    z = np.asarray(data['z'])

    coords = np.stack((x,y,z), axis=1)
    yeo_marker_color = []
    find_marker_color = []
    pre_marker_size = []
    peri_marker_size = []
    post_marker_size = []

    for contact_idx in range(len(contacts)):
        # Yeo colors:
        if yeo_networks[contact_idx] == 'VAN':
            yeo_marker_color.append('gold')
        elif yeo_networks[contact_idx] == 'DAN':
            yeo_marker_color.append('lawngreen')
        elif yeo_networks[contact_idx] == 'DMN':
            yeo_marker_color.append('navy')
        elif yeo_networks[contact_idx] == 'Somatomotor':
            yeo_marker_color.append('salmon')
        elif yeo_networks[contact_idx] == 'Visuomotor':
            yeo_marker_color.append('orange')
        elif yeo_networks[contact_idx] == 'Limbic':
            yeo_marker_color.append('lavender')
        elif yeo_networks[contact_idx] == 'Frontoparietal':
            yeo_marker_color.append('forestgreen')
        else:
            yeo_marker_color.append('slategrey')
        
        # FIND colors:
        if find_networks[contact_idx] == 'Auditory':
            find_marker_color.append('black')
        elif find_networks[contact_idx] == 'anterior_Salience':
            find_marker_color.append('gold')
        elif find_networks[contact_idx] == 'post_Salience':
            find_marker_color.append('yellow')
        elif find_networks[contact_idx] == 'Basal_Ganglia':
            find_marker_color.append('lavender')
        elif find_networks[contact_idx] == 'L_ExecutiveControlNetwork':
            find_marker_color.append('lawngreen')
        elif find_networks[contact_idx] == 'R_ExecutiveControlNetwork':
            find_marker_color.append('forestgreen')
        elif find_networks[contact_idx] == 'Language':
            find_marker_color.append('pink')        
        elif find_networks[contact_idx] == 'Visuospatial':
            find_marker_color.append('violet')
        elif find_networks[contact_idx] == 'Precuneus':
            find_marker_color.append('navy')
        elif find_networks[contact_idx] == 'dorsalDefaultModeNetwork':
            find_marker_color.append('navy')
        elif find_networks[contact_idx] == 'ventralDefaultModeNetwork':
            find_marker_color.append('blue')
        elif find_networks[contact_idx] == 'Sensorimotor':
            find_marker_color.append('magenta')
        elif find_networks[contact_idx] == 'prim_Visual':
            find_marker_color.append('red')
        else:
            find_marker_color.append('white')

        pre_marker_size.append(prestim[contact_idx]*75-25)
        peri_marker_size.append(peristim[contact_idx]*75-25)
        post_marker_size.append(poststim[contact_idx]*75-25)
    
    plot = plotting.view_markers(coords, marker_color=yeo_marker_color, marker_size=pre_marker_size)
    plot.save_as_html(path + '/%s_YEO_prestim_glassbrain.html' % (tt))    
    plot = plotting.view_markers(coords, marker_color=yeo_marker_color, marker_size=peri_marker_size)
    plot.save_as_html(path + '/%s_YEO_peristim_glassbrain.html' % (tt))  
    plot = plotting.view_markers(coords, marker_color=yeo_marker_color, marker_size=post_marker_size)
    plot.save_as_html(path + '/%s_YEO_poststim_glassbrain.html' % (tt))  

    plot = plotting.view_markers(coords, marker_color=find_marker_color, marker_size=pre_marker_size)
    plot.save_as_html(path + '/%s_FIND_prestim_glassbrain.html' % (tt))    
    plot = plotting.view_markers(coords, marker_color=find_marker_color, marker_size=peri_marker_size)
    plot.save_as_html(path + '/%s_FIND_peristim_glassbrain.html' % (tt))  
    plot = plotting.view_markers(coords, marker_color=find_marker_color, marker_size=post_marker_size)
    plot.save_as_html(path + '/%s_FIND_poststim_glassbrain.html' % (tt))  

# Load individual patient contact data
for patient in patients:
    path_to_load = os.path.join(path, patient, 'CNN_results')
    network_path = os.path.join(path, patient, 'raw')

    if not os.path.exists(path_to_load):
        continue

    for tt in ['shift', 'nonshift']:

        # Loads contact, epoch, and network data
        try:

            with open((path_to_load + str('/%s_cnn_test_results.csv' % (tt))), newline='\n') as csvfile:
                data = pd.read_csv(csvfile)
                data['patient'] = patient

            with open((path_to_load + str('/%s_SHAP_epoch_results.csv' % (tt))), newline='\n') as csvfile:
                epoch_data = pd.read_csv(csvfile)

            # Join dataframes and remove outdated network info
            data = data.join(epoch_data.loc[:, 'Best Epoch':])  
            if 'Network' in data.columns.values:
                data.drop(columns=['Network'], inplace=True) 
            if 'FIND Network' in data.columns.values:
                data.drop(columns=['FIND Network'], inplace=True) 

            # Load latest network info
            with open((network_path + str('/contact_mapping.csv')), newline='\n') as csvfile:
                network_data = pd.read_csv(csvfile)

            # Remove missing contacts
            missing_contacts = [i for i in network_data['contact'] if i not in data['Contact'].values]
            for missing in missing_contacts:
                missing_idx = network_data[network_data['contact'] == missing].index
                network_data.drop(missing_idx, inplace=True)

            # Join data
            data = data.join(network_data.loc[:, 'FIND_label':'yeo_label'])
            data = data.join(network_data.loc[:, 'ho_label'])
            data.rename(columns={'FIND_label': 'FIND Network', 'yeo_label': 'YEO Network', 'ho_label': 'HO Region'}, inplace=True)
            data.to_csv(os.path.join(path, patient, 'summary_data.csv'))

        except:
            print('Error: Missing %s data file for patient %s' % (tt, patient))
            continue

        # Weighted Contributions
        data['Pre Weight'] = data['AUC']*data['Pre-Stim']
        data['Peri Weight'] = data['AUC']*data['Peri-Stim']
        data['Post Weight'] = data['AUC']*data['Post-Stim']
        plot_glassbrains(data, path_to_load, tt)
                
        if tt == 'shift':
            shift_data.append(data)
        else:
            non_shift_data.append(data)

shift_data = pd.concat(shift_data)
shift_data = shift_data.sort_values(by=['AUC', 'Accuracy'], ascending=False)
non_shift_data = pd.concat(non_shift_data)
non_shift_data = non_shift_data.sort_values(by=['AUC', 'Accuracy'], ascending=False)

# Print and save contacts by trial type
print('-'*20)
print('shift %i contacts:' % len(shift_data))
print(shift_data)
print('-'*20)

# Statistical Analysis by Region:
#
#
#
#
## Need to work on this more later
region_model = ols('AUC ~ Region', data=shift_data).fit()
table_shift = sm.stats.anova_lm(region_model, type=2)
print(table_shift)

shift_data.reset_index(drop=True).to_csv((path_to_save + '/shift_all_contacts.csv')) # save this output
shift_region_summ = shift_data.groupby(["HO Region"])[["AUC", "Accuracy"]].describe()
shift_region_summ.to_csv((path_to_save + '/shift_all_region_summary.csv'))
sns_plot = sns.boxplot(y=shift_data['Region'], x=shift_data['AUC'])
plt.xticks(rotation=60)
plt.savefig((path_to_save + '/shift_all_regions.png'), bbox_inches = "tight")
plt.close()

shift_network_summ = shift_data.groupby(["FIND Network"])[["AUC", "Accuracy"]].describe()
shift_network_summ.to_csv((path_to_save + '/shift_FIND_network_summary.csv'))
shift_network_summ = shift_data.groupby(["YEO Network"])[["AUC", "Accuracy"]].describe()
shift_network_summ.to_csv((path_to_save + '/shift_YEO_network_summary.csv'))

shift_top = shift_data[(shift_data['AUC'] >= 0.7) & (shift_data['Accuracy'] >= 0.7)]
shift_top = shift_top.sort_values(by=['AUC', 'Accuracy'], ascending=False)

print('-'*20)
print('shift top %i contacts:' % len(shift_top))
print(shift_top)
print('-'*20)

# Save top regions only
shift_top.reset_index(drop=True).to_csv((path_to_save + '/shift_TOP_contacts.csv')) # save this output
shift_region_summ = shift_top.groupby(["HO Region"])[["AUC", "Accuracy"]].describe()
shift_region_summ.to_csv((path_to_save + '/shift_TOP_region_summary.csv'))
sns_plot = sns.boxplot(y=shift_top['Region'], x=shift_top['AUC'])
plt.xticks(rotation=60)
plt.savefig((path_to_save + '/shift_top_regions.png'), bbox_inches = "tight")
plt.close()

# Plotting FIND network locations (with all contacts)
for network in shift_data.dropna(subset=['FIND Network'])['FIND Network'].unique():
    network_plot = pd.melt(shift_data[shift_data['FIND Network'] == network].loc[:,'Pre Weight':'Post Weight'])
    ax = sns.boxplot(x='variable', y='value', data=network_plot)
    test_results = add_stat_annotation(ax, data=network_plot, x='variable',
        y='value', box_pairs=[('Pre Weight', 'Peri Weight'), ('Pre Weight', 'Post Weight'), ('Peri Weight', 'Post Weight')],
        test='t-test_ind',text_format='star',loc='inside')
    plt.xticks(rotation=60)
    plt.title('%s Contibution per Epoch' % (network))
    plt.savefig((FIND_path + '/SHIFT_%s_epochs.png' % (network)), bbox_inches = "tight", size=(20,20))
    plt.close()

# Plotting YEO network locations (with all conctacts)
for network in shift_data.dropna(subset=['YEO Network'])['YEO Network'].unique():
    network_plot = pd.melt(shift_data[shift_data['YEO Network'] == network].loc[:,'Pre Weight':'Post Weight'])
    ax = sns.boxplot(x='variable', y='value', data=network_plot)
    test_results = add_stat_annotation(ax, data=network_plot, x='variable',
        y='value', box_pairs=[('Pre Weight', 'Peri Weight'), ('Pre Weight', 'Post Weight'), ('Peri Weight', 'Post Weight')],
        test='t-test_ind',text_format='star',loc='inside')    
    plt.xticks(rotation=60)
    plt.title('%s Contibution per Epoch' % (network))
    plt.savefig((YEO_path + '/SHIFT_%s_epochs.png' % (network)), bbox_inches = "tight")
    plt.close()

print('-'*20)
print('non shift %i contacts:' % len(non_shift_data))
print(non_shift_data)
print('-'*20)

non_shift_data.reset_index(drop=True).to_csv(path_to_save + '/non_shift_all_contacts.csv')
non_shift_region_summ = non_shift_data.groupby(["HO Region"])[["AUC", "Accuracy"]].describe()
non_shift_region_summ.to_csv((path_to_save + '/non_shift_region_summary.csv'))
sns_plot = sns.boxplot(y=non_shift_data['Region'], x=non_shift_data['AUC'])
plt.xticks(rotation=60)
plt.savefig((path_to_save + '/non_shift_all_regions.png'), bbox_inches = "tight")
plt.close()

non_shift_top = non_shift_data[(non_shift_data['AUC'] >= 0.7) & (non_shift_data['Accuracy'] >= 0.7)]
non_shift_top = non_shift_top.sort_values(by=['AUC', 'Accuracy'], ascending=False)

print('-'*20)
print('non shift best %i contacts:' % len(non_shift_top))
print(non_shift_top)
print('-'*20)

# Save top regions only
non_shift_top.reset_index(drop=True).to_csv((path_to_save + '/non_shift_TOP_contacts.csv')) # save this output
non_shift_region_summ = non_shift_top.groupby(["HO Region"])[["AUC", "Accuracy"]].describe()
non_shift_region_summ.to_csv((path_to_save + '/non_shift_TOP_region_summary.csv'))
sns_plot = sns.boxplot(y=non_shift_top['Region'], x=non_shift_top['AUC'])
plt.xticks(rotation=60)
plt.savefig((path_to_save + '/non_shift_top_regions.png'), bbox_inches = "tight")
plt.close()

non_shift_network_summ = non_shift_data.groupby(["FIND Network"])[["AUC", "Accuracy"]].describe()
non_shift_network_summ.to_csv((path_to_save + '/non_shift_FIND_network_summary.csv'))
non_shift_network_summ = non_shift_data.groupby(["YEO Network"])[["AUC", "Accuracy"]].describe()
non_shift_network_summ.to_csv((path_to_save + '/non_shift_YEO_network_summary.csv'))

# Plotting FIND network locations
for network in non_shift_data.dropna(subset=['FIND Network'])['FIND Network'].unique():
    network_plot = pd.melt(non_shift_data[non_shift_data['FIND Network'] == network].loc[:,'Pre Weight':'Post Weight'])
    ax = sns.boxplot(x='variable', y='value', data=network_plot)
    test_results = add_stat_annotation(ax, data=network_plot, x='variable',
        y='value', box_pairs=[('Pre Weight', 'Peri Weight'), ('Pre Weight', 'Post Weight'), ('Peri Weight', 'Post Weight')],
        test='t-test_ind',text_format='star',loc='inside')    
    plt.xticks(rotation=60)
    plt.title('%s Contibution per Epoch' % (network))
    plt.savefig((FIND_path + '/NON_SHIFT_%s_epochs.png' % (network)), bbox_inches = "tight")
    plt.close()

# Plotting YEO network locations
for network in non_shift_data.dropna(subset=['YEO Network'])['YEO Network'].unique():
    network_plot = pd.melt(non_shift_data[non_shift_data['YEO Network'] == network].loc[:,'Pre Weight':'Post Weight'])
    ax = sns.boxplot(x='variable', y='value', data=network_plot)
    test_results = add_stat_annotation(ax, data=network_plot, x='variable',
        y='value', box_pairs=[('Pre Weight', 'Peri Weight'), ('Pre Weight', 'Post Weight'), ('Peri Weight', 'Post Weight')],
        test='t-test_ind',text_format='star',loc='inside')
    plt.xticks(rotation=60)
    plt.title('%s Contibution per Epoch' % (network))
    plt.savefig((YEO_path + '/NON_SHIFT_%s_epochs.png' % (network)), bbox_inches = "tight")
    plt.close()

# Now that we have general summary data
# we can load the epoched TFD data from SHAP
# explanations

for tt in ['shift', 'nonshift']:

    if tt == 'shift':
        data = shift_data
    else:
        data = non_shift_data

    plot_glassbrains(data=data, path=(path_to_save + '/glassbrains'), tt=tt)
