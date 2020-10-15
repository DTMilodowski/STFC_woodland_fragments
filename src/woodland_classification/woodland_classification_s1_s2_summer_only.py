"""
sentinel_fusion_test.py
================================================================================
This is a function to combine preprocessed remote sensing data streams from
Sentinel 1 and Sentinel 2 and calibrate and validate a classification algorithm
to map tree cover based on four classes:
- low vegetation
- isolated trees (>8 contiguous pixels above two metres height)
- open woodland (>20% canopy cover at 2m)
- closed woodland (>50% canopy cover at 2m)

The S1 data streams are annual average and standard deviation (see Hansen et al,
2020) for HH, HV and HH-HV at 10m resolution.

The S2 data streams are 10m resolution visible bands (B3, B4, B5, B8) and 20m
resolution bands (B5, B6, B7, B8A, B11, B12) superresolved at 10m resolution
(using DSen2), and the following derivatives: EVI, BI, MSAVI2, SI (see Brandt &
Stolle, in review).

The classifications under consideration are Random Forest and SVM (see Hansen et
al, 2020). Future work could test the framework developed by Brandt & Stolle,
but this is beyond the scope of this work.

A buffered k-fold approach is used to cross-validate the algorithm in one site
as a test. An extension will be to carry out the same analysis using 7 sites.

Accuracy assessments will be carried out on two levels:
(1) Standard accuracy assessment based on the confusion matrix
(2) Adapted accuracy assessment the allows 10m (a single pixel) margin of error
    that removes the prevalence of geolocation errors, that are not of interest
    with regards to the sensitivity of sensor combinations to trees in the
    landscape
--------------------------------------------------------------------------------
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from skimage import morphology
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
sns.set()

sys.path.append('../data_io')
sys.path.append('../accuracy_assessment/')
import data_io as io
import accuracy_assessment as acc

"""
Defining some general variables and other project details
--------------------------------------------------------------------------------
"""
site = 'Arisaig'

# paths
path2s1 = '/disk/scratch/local.2/dmilodow/Sentinel1/'
path2s2 = '/exports/csce/datastore/geos/users/dmilodow/STFC/DATA/Sentinel2/processed_bands_and_derivatives/'
path2lidar = '/disk/scratch/local.2/dmilodow/SEOS/raster/OSGBgrid/'
path2output = '/exports/csce/datastore/geos/users/dmilodow/STFC/woodland_fragments/figures/'

# lidar files
if site == 'Auchteraw':
    dtm_file = '%s/%se__Digital_Terrain_Model_1m_OSGB.tif' % (path2lidar,site)
    dsm_file = '%s/%se__Digital_Surface_Model_1m_OSGB.tif' % (path2lidar,site)
elif site == 'Arisaig':
    dtm_file = '%s/%s_South__Digital_Terrain_Model_1m_OSGB.tif' % (path2lidar,site)
    dsm_file = '%s/%s_South__Digital_Surface_Model_1m_OSGB.tif' % (path2lidar,site)
else:
    dtm_file = '%s/%s__Digital_Terrain_Model_1m_OSGB.tif' % (path2lidar,site)
    dsm_file = '%s/%s__Digital_Surface_Model_1m_OSGB.tif' % (path2lidar,site)

# S1 files (temporal average and standard deviations)
s1vh_A_summer_file = '%s%s/processed_temporal/2019/S1A__IW__A_2019_VH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_mean_summer.tif' % (path2s1,site)
s1vv_A_summer_file = '%s%s/processed_temporal/2019/S1A__IW__A_2019_VV_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_mean_summer.tif' % (path2s1,site)
s1diff_A_summer_file = '%s%s/processed_temporal/2019/S1A__IW__A_2019_diffVVVH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_mean_summer.tif' % (path2s1,site)

s1vh_A_std_summer_file = '%s%s/processed_temporal/2019/S1A__IW__A_2019_VH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_stdev_summer.tif' % (path2s1,site)
s1vv_A_std_summer_file = '%s%s/processed_temporal/2019/S1A__IW__A_2019_VV_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_stdev_summer.tif' % (path2s1,site)
s1diff_A_std_summer_file = '%s%s/processed_temporal/2019/S1A__IW__A_2019_diffVVVH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_stdev_summer.tif' % (path2s1,site)

s1vh_D_summer_file = '%s%s/processed_temporal/2019/S1A__IW__D_2019_VH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_mean_summer.tif' % (path2s1,site)
s1vv_D_summer_file = '%s%s/processed_temporal/2019/S1A__IW__D_2019_VV_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_mean_summer.tif' % (path2s1,site)
s1diff_D_summer_file = '%s%s/processed_temporal/2019/S1A__IW__D_2019_diffVVVH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_mean_summer.tif' % (path2s1,site)

s1vh_D_std_summer_file = '%s%s/processed_temporal/2019/S1A__IW__D_2019_VH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_stdev_summer.tif' % (path2s1,site)
s1vv_D_std_summer_file = '%s%s/processed_temporal/2019/S1A__IW__D_2019_VV_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_stdev_summer.tif' % (path2s1,site)
s1diff_D_std_summer_file = '%s%s/processed_temporal/2019/S1A__IW__D_2019_diffVVVH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_stdev_summer.tif' % (path2s1,site)


# S2 files (stack of bands and derivative indices)
s2_file = '%s/%s_sentinel2_bands_10m.tif' % (path2s2,site)

# open one of the S1 files for template
template = xr.open_rasterio(s1vh_A_summer_file).sel(band=1)
dx = template.x.values[1]-template.x.values[0]
dy = template.y.values[1]-template.y.values[0]
E = template.x.values.max()+np.abs(dx)/2.
W = template.x.values.min()-np.abs(dx)/2.
N = template.y.values.max()+np.abs(dy)/2.
S = template.y.values.min()-np.abs(dy)/2.

"""
Generate reference land cover classes based on LiDAR TCH
- classify into four classes
    1) Bare/grass
    2) Low vegetation
    3) Isolated trees
    4) Open woodland
    5) Dense woodland
--------------------------------------------------------------------------------
"""
print('Loading reference maps')
lc_labels = ['ground', 'shrubs', 'isolated trees', 'open woodland', 'dense woodland']
lc = xr.open_rasterio('%s%s_lc_class_%.0fm.tif' % (path2output,site,dx))[0]
lc.values[lc.values==lc.nodatavals[0]]=np.nan

# Create a simplified version 1 = no trees, 2 = trees
lc_simple = lc.copy(deep=True)
lc_simple.values[lc.values<3]=1
lc_simple.values[lc.values>=3]=2

mask=np.isfinite(lc.values)

"""
Load the satellite data
- S1 temporal average and standard deviations for HH, HV and HH-HV over summer
- Store in variable stack for calibration of machine learning model
--------------------------------------------------------------------------------
"""
print('Loading the satellite data and preparing for ingestion into random forest')
# S1 Ascending Summer
s1vhA_summer = xr.open_rasterio(s1vh_A_summer_file)
s1vvA_summer = xr.open_rasterio(s1vv_A_summer_file)
s1diffA_summer = xr.open_rasterio(s1diff_A_summer_file)

s1vhA_std_summer = xr.open_rasterio(s1vh_A_std_summer_file)
s1vvA_std_summer = xr.open_rasterio(s1vv_A_std_summer_file)
s1diffA_std_summer = xr.open_rasterio(s1diff_A_std_summer_file)

# S1 Descending Summer
s1vhD_summer = xr.open_rasterio(s1vh_D_summer_file)
s1vvD_summer = xr.open_rasterio(s1vv_D_summer_file)
s1diffD_summer = xr.open_rasterio(s1diff_D_summer_file)

s1vhD_std_summer = xr.open_rasterio(s1vh_D_std_summer_file)
s1vvD_std_summer = xr.open_rasterio(s1vv_D_std_summer_file)
s1diffD_std_summer = xr.open_rasterio(s1diff_D_std_summer_file)

mask*(s1diffA_summer.values[0]!=s1diffA_summer.nodatavals[0])
mask*(s1diffD_summer.values[0]!=s1diffD_summer.nodatavals[0])

# S2 layers
s2_os_file = '%s/%s_sentinel2_bands_10m_osgb.tif' % (path2s2,site)
os.system('gdalwarp -overwrite -t_srs EPSG:27700 -r near -te %f %f %f %f -tr %f %f %s %s' % (W, S, E, N, dx, dy, s2_file, s2_os_file))
s2layers = xr.open_rasterio(s2_os_file)#.sel(x=slice(W,E),y=slice(N,S))

mask*=np.isfinite(np.sum(s2layers.values,axis=0))

# stack layers
ns1 = 12
ns2 = s2layers.shape[0] # check This
satellite_layers = np.concatenate((s1vhA_summer.values, s1vvA_summer.values, s1diffA_summer.values,
                            s1vhA_std_summer.values, s1vvA_std_summer.values, s1diffA_std_summer.values,
                            s1vhD_summer.values, s1vvD_summer.values, s1diffD_summer.values,
                            s1vhD_std_summer.values, s1vvD_std_summer.values, s1diffD_std_summer.values,
                            s2layers.values),axis=0)

X = np.zeros((mask.sum(),satellite_layers.shape[0]))
for ii, layer in enumerate(satellite_layers):
    X[:,ii] = layer[mask]

y = lc.values[mask]
y_simple = lc_simple.values[mask]

assert(np.isnan(y).sum()==0)
assert(np.isnan(X).sum()==0)

# summarise class distributions
classes,class_count = np.unique(lc.values[mask],return_counts=True)

"""
Calibrate machine learning model with Random Forest and Support Vector Machine
- K-fold cross validation using a buffered block strategy to reduce overfitting
  of spatial autocorrelations
--------------------------------------------------------------------------------
"""
print('Setting up k-fold template')
"""
Setting up the blocked k-fold template
"""
# k-folds
k = 10

# create a blocked sampling grid at ~1 degree resolution
raster_res = s1vvA_summer.attrs['res'][0]
block_res = 1000
block_width = int(np.ceil(block_res/raster_res))
buffer_width = 100
buffer = int(np.ceil(buffer_width/raster_res))

blocks_array = np.zeros(s1vvA_summer.values[0].shape)
block_label = 0
for rr,row in enumerate(np.arange(0,blocks_array.shape[0],block_width)):
    for cc, col in enumerate(np.arange(0,blocks_array.shape[1],block_width)):
        blocks_array[row:row+block_width,col:col+block_width]=block_label
        block_label+=1

# test blocks for training data presence
blocks = blocks_array[mask]
blocks_keep,blocks_count = np.unique(blocks,return_counts=True)
# remove blocks with no training data
blocks_array[~np.isin(blocks_array,blocks_keep)]=np.nan

# permute blocks randomly
val_blocks_array = blocks_array.copy()
blocks_kfold = np.random.permutation(blocks_keep)
blocks_in_fold = int(np.floor(blocks_kfold.size/k))
for ii in range(0,k):
    blocks_iter = blocks_kfold[ii*blocks_in_fold:(ii+1)*blocks_in_fold]
    # label calibration blocks with fold
    val_blocks_array[np.isin(blocks_array,blocks_iter)]=ii

blocks_to_be_allocated = blocks_kfold.size%k
blocks_allocated = blocks_kfold.size-blocks_to_be_allocated

for ii in range(0,k):
    if ii+blocks_allocated<blocks_kfold.size:
        blocks_iter = blocks_kfold[ii+blocks_allocated]
        val_blocks_array[np.isin(blocks_array,blocks_iter)]=ii

val_blocks = val_blocks_array[mask].astype('int')
# now filter the blocks based on proximity to validation data to avoid
# neighbouring pixels biasing the analysis
cal_blocks_array = val_blocks_array.copy()
for ii in range(0,k):
    val_data_mask = np.all((val_blocks_array==ii,mask),axis=0)
    # expand neighbourhood with buffer
    for bb in range(0,buffer):
        val_data_mask = morphology.binary_dilation(val_data_mask)
    cal_blocks_array[np.all((val_data_mask,val_blocks_array!=ii),axis=0)]=np.nan

cal_blocks = cal_blocks_array[mask].astype('int')

print('fitting random forest classifier')
# k-fold cross validation
rf = RandomForestClassifier(random_state = 706)
rf_trees = RandomForestClassifier(random_state = 808)
rf_no_trees = RandomForestClassifier(random_state = 2310)
"""
rf = RandomForestClassifier(class_weight='balanced_subsample',random_state = 706)
rf_trees = RandomForestClassifier(class_weight='balanced_subsample',random_state = 808)
rf_no_trees = RandomForestClassifier(class_weight='balanced_subsample', random_state = 2310)
"""
# vectors to host predicted values
y_rf = np.zeros(y.shape)*np.nan
y_rf_simple = np.zeros(y.shape)*np.nan

# cross validation loop
# - first tree/no tree
for kk in range(0,k):
    print('calibrating first level classifiers... iteration %i of %i' %(kk+1,k), end='\r')
    rf.fit(X[cal_blocks!=kk],y_simple[cal_blocks!=kk])
    y_rf_simple[val_blocks==kk] = rf.predict(X[val_blocks==kk])

# - second level classification
for kk in range(0,k):
    print('calibrating second level classifiers... iteration %i of %i' %(kk+1,k), end='\r')
    Xcal = X[cal_blocks!=kk]
    ycal = y[cal_blocks!=kk]
    y_rf_simple_cal = y_rf_simple[cal_blocks!=kk]

    Xval = X[val_blocks==kk]
    y_rf_simple_val = y_rf_simple[val_blocks==kk]
    y_rf_val = np.zeros(y_rf_simple_val.size)

    rf_trees.fit(Xcal[y_rf_simple_cal==2],ycal[y_rf_simple_cal==2])
    y_rf_val[y_rf_simple_val==2] = rf_trees.predict(Xval[y_rf_simple_val==2])

    rf_no_trees.fit(Xcal[y_rf_simple_cal==1],ycal[y_rf_simple_cal==1])
    y_rf_val[y_rf_simple_val==1] = rf_no_trees.predict(Xval[y_rf_simple_val==1])

    y_rf[val_blocks==kk] = y_rf_val.copy()



# predicted land cover maps
lc_rf = lc.copy(deep=True)
lc_rf.values*=np.nan
lc_rf.values[mask]=y_rf.copy()

lc_rf_simple = lc.copy(deep=True)
lc_rf_simple.values*=np.nan
lc_rf_simple.values[mask]=y_rf_simple.copy()

"""
Accuracy Assessment
- Plot variable distributions for the five land cover classes to provide visual
  assessment of class separability
- Overall accuracy, Predictors accuracy and Users Accuracy
- Two approaches to calculating (i) basic, and (ii) allowing 1 pixel margin of
  error
- Assessment of variable quality of "tree" detction depending on context:
                            **hypothesis**
  Improved performance from isolated trees -> open woodland -> dense woodland
--------------------------------------------------------------------------------
"""
print('Accuracy assessment')
CM_rf = acc.build_confusion_matrix(y,y_rf)
acc_stats_rf = acc.calculate_accuracy_stats(CM_rf)

lc_labels_simple = ['no trees', 'trees']
CM_rf_simple = acc.build_confusion_matrix(y_simple,y_rf_simple)
acc_stats_rf_simple = acc.calculate_accuracy_stats(CM_rf_simple)

fig1, axes = plt.subplots(nrows=1,ncols=2,figsize=(7, 5))
sns.heatmap(CM_rf.astype('int'), annot=True, fmt="d", linewidths=.5, ax=axes[0], cmap="YlGnBu",
            xticklabels=lc_labels, yticklabels=lc_labels, cbar=False, square=True)
sns.heatmap(CM_rf_simple.astype('int'), annot=True, fmt="d", linewidths=.5, ax=axes[1], cmap="YlGnBu",
            xticklabels=lc_labels_simple, yticklabels=lc_labels_simple, cbar=False, square=True)
axes[0].set_title('Confusion Matrix\nFull Classification')
axes[1].set_title('Confusion Matrix\nBinary Classification')
fig1.show()
fig1.tight_layout()
fig1.savefig('ConfusionMatrix_RF_%s_s1_s2_summer.png' % site)

# plot land cover maps
# two panels: a) LiDAR; b) RF
colours = np.asarray(['#ffcfe2', '#ff9dc8', '#00cba7', '#00735c', '#003d30'])
lc_cmap = ListedColormap(sns.color_palette(colours).as_hex())

plt.rcParams["axes.axisbelow"] = False
fig2,axes = plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,figsize=(9,8))
fig2.subplots_adjust(right=0.75)
lc.plot(ax=axes[0],cmap = lc_cmap, add_colorbar=False)
#lc_svm.plot(ax=axes[1],cmap = lc_cmap)
lc_rf.plot(ax=axes[1],cmap = lc_cmap, add_colorbar=False)

#for ii, title in enumerate(['Reference (LiDAR)','Support Vector Machine', 'Random Forest']):
for ii, title in enumerate(['Reference (LiDAR)', 'Random Forest']):
    axes[ii].set_title(title)
    axes[ii].set_xlabel('Easting / m')
    axes[ii].set_ylabel('Northing / m')
    axes[ii].set_aspect('equal')


for ax in axes:
    ax.grid(True,which='both')

# plot legend for color map
cax = fig2.add_axes([0.8,0.375,0.05,0.25])
bounds = np.arange(len(lc_labels)+1)
norm = mpl.colors.BoundaryNorm(bounds, lc_cmap.N)
cb = mpl.colorbar.ColorbarBase(cax,cmap=lc_cmap,norm=norm,orientation='vertical')
n_class = len(lc_labels)
loc = np.arange(0,n_class)+0.5
cb.set_ticks(loc)
cb.set_ticklabels(lc_labels)
cb.update_ticks()

fig2.show()
fig2.savefig('landcover_classifications_%s_s1_s2_summer.png' % site)

"""
Version two of the accuracy assessment with the more nuanced treatment of
errors described by Brandt and Stolle (in review)

Adapted accuracy assessment the allows 10m (a single pixel) margin of error that
removes the prevalence of geolocation errors, that are not of interest with
regards to the sensitivity of sensor combinations to trees in the landscape
"""
print('Improved accuracy assessment')
acc_stats_rf_tol0 = acc.calculate_accuracy_stats_with_margin_for_error(lc.values,lc_rf.values,tolerance=0)
acc_stats_rf_tol1 = acc.calculate_accuracy_stats_with_margin_for_error(lc.values,lc_rf.values)
acc_stats_rf_simple_tol1 = acc.calculate_accuracy_stats_with_margin_for_error(lc_simple.values,lc_rf_simple.values)
acc_stats_rf_simple_tol0 = acc.calculate_accuracy_stats_with_margin_for_error(lc_simple.values,lc_rf_simple.values,tolerance=0)


# assess the distribution of omission errors based on the land cover
# characteristics
OEmap = acc_stats_rf_simple_tol1['omission_error_map']
CEmap = acc_stats_rf_simple_tol1['commission_error_map']
omission_classes,omission_error_count = np.unique(lc.values[OEmap==1],return_counts=True)
commission_classes,commission_error_count = np.unique(lc_rf.values[CEmap==1],return_counts=True)
OErate=omission_error_count/class_count
CErate=commission_error_count/class_count
print('1 pixel tolerance')
print('labels',lc_labels)
print('OErate',OErate*100)
print('CErate',CErate*100)
print('PA    ',(1-OErate)*100)
print('UA    ',(1-CErate)*100)
print('OA    ',(acc_stats_rf_simple_tol1['OA'])*100)


OEmap = acc_stats_rf_simple_tol0['omission_error_map']
CEmap = acc_stats_rf_simple_tol0['commission_error_map']
omission_classes,omission_error_count = np.unique(lc.values[OEmap==1],return_counts=True)
commission_classes,commission_error_count = np.unique(lc_rf.values[CEmap==1],return_counts=True)
OErate=omission_error_count/class_count
CErate=commission_error_count/class_count
print('0 pixel tolerance')
print('labels',lc_labels)
print('OErate',OErate*100)
print('CErate',CErate*100)
print('PA    ',(1-OErate)*100)
print('UA    ',(1-CErate)*100)
print('OA    ',(acc_stats_rf_simple_tol0['OA'])*100)



OEmap = acc_stats_rf_tol1['omission_error_map']
CEmap = acc_stats_rf_tol1['commission_error_map']
omission_classes,omission_error_count = np.unique(lc.values[OEmap==1],return_counts=True)
commission_classes,commission_error_count = np.unique(lc_rf.values[CEmap==1],return_counts=True)
OErate=omission_error_count/class_count
CErate=commission_error_count/class_count
print('1 pixel tolerance')
print('labels',lc_labels)
print('OErate',OErate*100)
print('CErate',CErate*100)
print('PA    ',(1-OErate)*100)
print('UA    ',(1-CErate)*100)
print('OA    ',(acc_stats_rf_tol1['OA'])*100)


OEmap = acc_stats_rf_tol0['omission_error_map']
CEmap = acc_stats_rf_tol0['commission_error_map']
omission_classes,omission_error_count = np.unique(lc.values[OEmap==1],return_counts=True)
commission_classes,commission_error_count = np.unique(lc_rf.values[CEmap==1],return_counts=True)
OErate=omission_error_count/class_count
CErate=commission_error_count/class_count
print('0 pixel tolerance')
print('labels',lc_labels)
print('OErate',OErate*100)
print('CErate',CErate*100)
print('PA    ',(1-OErate)*100)
print('UA    ',(1-CErate)*100)
print('OA    ',(acc_stats_rf_tol0['OA'])*100)


"""
Plot up the error map
"""
error_colours = np.asarray(['white', 'black', '#cecece', '#ff9DC8', '#656565', '#00cba7'])[::-1]
error_labels = np.asarray(['no trees', 'trees',
                            'adjacent omission error', 'omission error',
                            'adjacent commission error', 'commission error'])[::-1]
error_cmap = ListedColormap(sns.color_palette(error_colours).as_hex())

error_map = lc.copy(deep=True)
error_map.values*=np.nan

error_map.values[(acc_stats_rf_simple_tol0['omission_error_map']==0)*(lc_simple.values==1)] = 6      # no tree
error_map.values[(acc_stats_rf_simple_tol0['omission_error_map']==0)*(lc_simple.values==2)] = 5      # tree
error_map.values[(acc_stats_rf_simple_tol0['omission_error_map']==1)*
                 (acc_stats_rf_simple_tol1['commission_error_map']==0)*(lc_simple.values==2)] = 4    # adjacent omission error
error_map.values[(acc_stats_rf_simple_tol0['commission_error_map']==1)*
                 (acc_stats_rf_simple_tol1['commission_error_map']==0)*(lc_rf_simple.values==2)] = 2 # adjacent commision error
error_map.values[(acc_stats_rf_simple_tol1['omission_error_map']==1)*(lc_simple.values==2)] = 3   #  omission error
error_map.values[(acc_stats_rf_simple_tol1['commission_error_map']==1)*(lc_rf_simple.values==2)] = 1 # commission error

plt.rcParams["axes.axisbelow"] = False
fig3,ax = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=(9,4))
fig3.subplots_adjust(right=0.75)
error_map.plot(ax=ax,cmap = error_cmap, add_colorbar=False)

#for ii, title in enumerate(['Reference (LiDAR)','Support Vector Machine', 'Random Forest']):
ax.set_xlabel('Easting / m')
ax.set_ylabel('Northing / m')
ax.set_aspect('equal')
ax.grid(True,which='both')

# plot legend for color map
cax = fig3.add_axes([0.8,0.25,0.05,0.5])
bounds = np.arange(len(error_labels)+1)
norm = mpl.colors.BoundaryNorm(bounds, error_cmap.N)
cb = mpl.colorbar.ColorbarBase(cax,cmap=error_cmap,norm=norm,orientation='vertical')
n_class = len(error_labels)
loc = np.arange(0,n_class)+0.5
cb.set_ticks(loc)
cb.set_ticklabels(error_labels)
cb.update_ticks()

fig3.show()
fig3.savefig('error_map_%s_s1_s2_summer.png' % site)

"""
save layers to file
"""
io.write_xarray_to_GeoTiff(lc_rf,'%s%s_lc_class_rf_s1s2_%.0fm' % (path2output,site,dx))
io.write_xarray_to_GeoTiff(error_map,'%s%s_error_map_rf_s1s2_%.0fm' % (path2output,site,dx))
