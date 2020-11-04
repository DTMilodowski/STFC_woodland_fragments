"""
single_tree_woodland_classification.py
================================================================================
This is a function to combine preprocessed remote sensing data streams from
Sentinel 1 and Sentinel 2 and calibrate and validate a classification algorithm
to map tree cover based on four classes:
- low vegetation
- isolated trees (>8 contiguous pixels above two metres height)
- open woodland (>20% canopy cover at 2m)
- closed woodland (>50% canopy cover at 2m)

The S1 data streams are annual average and standard deviation (see Hansen et al,
2020) for HH, HV and HH-HV at 10m resolution. Only April-September included as
full twelve months did not lead to an improvement in performance.

The S2 data streams are 10m resolution visible bands (B3, B4, B5, B8) and 20m
resolution bands (B5, B6, B7, B8A, B11, B12) superresolved at 10m resolution
(using DSen2), and the following derivatives: EVI, BI, MSAVI2, SI (see Brandt &
Stolle, in review). Currently the input is a single scene. Regional upscaling
will require a framework to take in multiple scenes and deal with cloud cover.

The classifications are undertaken with a Random Forest and Classifier. Future
work could test the neural network framework developed by Brandt & Stolle, but
this is beyond the scope of this work. Hyperparameter optimisation has not been
fully explored, but doing so will be straightforward in the future.

A buffered k-fold approach is used to cross-validate the algorithm in one site
as a test. An extension will be to carry out the same analysis using 7 sites
with a "leave-one-site-out" cross-validation.

Accuracy assessments will be carried out on two levels:
(1) Standard accuracy assessment based on the confusion matrix
(2) Adapted accuracy assessment that allows 10m (a single pixel) margin of error.
    This removes the prevalence of geolocation errors, that are not of interest
    with regards to the sensitivity of sensor combinations to trees in the
    landscape

Final note: the script below contains code to carry out spatially independent
cross-validation. This is as much as I needed for the purposes of the original
project. In the SingleTree workflow, I would envisage that there would be a
separate script for cross-validation and accuracy assessment, one to do
the regional upscaling, and any visualisation split into another.
--------------------------------------------------------------------------------
D. T. Milodowski
"""
# python libraries
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from skimage import morphology
from sklearn.ensemble import RandomForestClassifier

# plotting libraries
import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sns.set()

# additional libraries
sys.path.append('../data_io')
sys.path.append('../accuracy_assessment/')
import data_io as io # this is a set of functions to help write geotiff rasters
import accuracy_assessment as acc # this is a set of functios to carry out accuracy assessment

"""
Defining some general variables and other project details
--------------------------------------------------------------------------------
"""
site = 'Arisaig' # Site name, I have other sites set up with systematic filenames, so site name all that's needed to switch

# paths - these are currently hard coded to point to the required directories.
#         There are probably better ways to do this, but I have at least tried
#         to ensure that the paths are defined in one location!
path2s1 = '/disk/scratch/local.2/dmilodow/Sentinel1/processed_temporal/2019/' # path to the processed sentinel 1 data
path2s2 = '/exports/csce/datastore/geos/users/dmilodow/STFC/DATA/Sentinel2/processed_bands_and_derivatives/' # path to the processed sentinel 2 data
path2figures = '/exports/csce/datastore/geos/users/dmilodow/STFC/woodland_fragments/figures/' # output path for figures
path2output = '/exports/csce/datastore/geos/users/dmilodow/STFC/woodland_fragments/output/' # output path for rasters

# S1 files (temporal average and standard deviations)
s1vh_A_file = '%s%s/S1A__IW__A_2019_VH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_mean_summer.tif' % (path2s1,site)
s1vv_A_file = '%s%s/S1A__IW__A_2019_VV_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_mean_summer.tif' % (path2s1,site)
s1diff_A_file = '%s%s/S1A__IW__A_2019_diffVVVH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_mean_summer.tif' % (path2s1,site)

s1vh_A_std_file = '%s%s/S1A__IW__A_2019_VH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_stdev_summer.tif' % (path2s1,site)
s1vv_A_std_file = '%s%s/S1A__IW__A_2019_VV_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_stdev_summer.tif' % (path2s1,site)
s1diff_A_std_file = '%s%s/S1A__IW__A_2019_diffVVVH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_stdev_summer.tif' % (path2s1,site)

s1vh_D_file = '%s%s/S1A__IW__D_2019_VH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_mean_summer.tif' % (path2s1,site)
s1vv_D_file = '%s%s/S1A__IW__D_2019_VV_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_mean_summer.tif' % (path2s1,site)
s1diff_D_file = '%s%s/S1A__IW__D_2019_diffVVVH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_mean_summer.tif' % (path2s1,site)

s1vh_D_std_file = '%s%s/S1A__IW__D_2019_VH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_stdev_summer.tif' % (path2s1,site)
s1vv_D_std_file = '%s%s/S1A__IW__D_2019_VV_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_stdev_summer.tif' % (path2s1,site)
s1diff_D_std_file = '%s%s/S1A__IW__D_2019_diffVVVH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_stdev_summer.tif' % (path2s1,site)


# S2 files (stack of bands and derivative indices)
s2_file = '%s/%s_sentinel2_bands_10m.tif' % (path2s2,site)

# open one of the S1 files for template
template = xr.open_rasterio(s1vh_A_file).sel(band=1)
dx = template.x.values[1]-template.x.values[0] # x resolution
dy = template.y.values[1]-template.y.values[0] # y resolution
# bounding box of site
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
lc_binary = lc.copy(deep=True)
lc_binary.values[lc.values<3]=1
lc_binary.values[lc.values>=3]=2

mask=np.isfinite(lc.values)

"""
Load the satellite data
- S1 temporal average and standard deviations for HH, HV and HH-HV over summer
- Store in variable stack for calibration of machine learning model
--------------------------------------------------------------------------------
"""
print('Loading the satellite data and preparing for ingestion into random forest')
# S1 Ascending Summer
s1vhA = xr.open_rasterio(s1vh_A_file)
s1vvA = xr.open_rasterio(s1vv_A_file)
s1diffA = xr.open_rasterio(s1diff_A_file)

s1vhA_std = xr.open_rasterio(s1vh_A_std_file)
s1vvA_std = xr.open_rasterio(s1vv_A_std_file)
s1diffA_std = xr.open_rasterio(s1diff_A_std_file)

# S1 Descending Summer
s1vhD = xr.open_rasterio(s1vh_D_file)
s1vvD = xr.open_rasterio(s1vv_D_file)
s1diffD = xr.open_rasterio(s1diff_D_file)

s1vhD_std = xr.open_rasterio(s1vh_D_std_file)
s1vvD_std = xr.open_rasterio(s1vv_D_std_file)
s1diffD_std = xr.open_rasterio(s1diff_D_std_file)

mask*(s1diffA.values[0]!=s1diffA.nodatavals[0])
mask*(s1diffD.values[0]!=s1diffD.nodatavals[0])

# S2 layers
s2_os_file = '%s/%s_sentinel2_bands_10m_osgb.tif' % (path2s2,site)
os.system('gdalwarp -overwrite -t_srs EPSG:27700 -r near -te %f %f %f %f -tr %f %f %s %s' % (W, S, E, N, dx, dy, s2_file, s2_os_file))
s2layers = xr.open_rasterio(s2_os_file)#.sel(x=slice(W,E),y=slice(N,S))

mask*=np.isfinite(np.sum(s2layers.values,axis=0))

# stack layers
ns1 = 12
ns2 = s2layers.shape[0] # check This
satellite_layers = np.concatenate((s1vhA.values, s1vvA.values, s1diffA.values,
                            s1vhA_std.values, s1vvA_std.values, s1diffA_std.values,
                            s1vhD.values, s1vvD.values, s1diffD.values,
                            s1vhD_std.values, s1vvD_std.values, s1diffD_std.values,
                            s2layers.values),axis=0)

X = np.zeros((mask.sum(),satellite_layers.shape[0]))
for ii, layer in enumerate(satellite_layers):
    X[:,ii] = layer[mask]

y = lc.values[mask]
y_binary = lc_binary.values[mask]

assert(np.isnan(y).sum()==0)
assert(np.isnan(X).sum()==0)

# summarise class distributions
classes,class_count = np.unique(lc.values[mask],return_counts=True)

"""
Calibrate machine learning model with Random Forest and Support Vector Machine
- K-fold cross validation using a buffered block strategy to reduce overfitting
  of spatial autocorrelations
- Note that this process is only required for the k-fold cross validation. A
  final upscaling model would be carried out using single fit on all available
  data
--------------------------------------------------------------------------------
"""
print('Setting up k-fold template')
"""
Setting up the blocked k-fold template
"""
# k-folds
k = 10

# create a blocked sampling grid at ~1 degree resolution
raster_res = s1vvA.attrs['res'][0]
block_res = 1000
block_width = int(np.ceil(block_res/raster_res))
buffer_width = 100
buffer = int(np.ceil(buffer_width/raster_res))

blocks_array = np.zeros(s1vvA.values[0].shape)
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

# vectors to host predicted values
y_rf = np.zeros(y.shape)*np.nan
y_rf_binary = np.zeros(y.shape)*np.nan

# cross validation loop
# - first tree/no tree
for kk in range(0,k):
    print('calibrating first level classifiers... iteration %i of %i' %(kk+1,k), end='\r')
    rf.fit(X[cal_blocks!=kk],y_binary[cal_blocks!=kk])
    y_rf_binary[val_blocks==kk] = rf.predict(X[val_blocks==kk])

# - second level classification
for kk in range(0,k):
    print('calibrating second level classifiers... iteration %i of %i' %(kk+1,k), end='\r')
    Xcal = X[cal_blocks!=kk]
    ycal = y[cal_blocks!=kk]
    y_rf_binary_cal = y_rf_binary[cal_blocks!=kk]

    Xval = X[val_blocks==kk]
    y_rf_binary_val = y_rf_binary[val_blocks==kk]
    y_rf_val = np.zeros(y_rf_binary_val.size)

    rf_trees.fit(Xcal[y_rf_binary_cal==2],ycal[y_rf_binary_cal==2])
    y_rf_val[y_rf_binary_val==2] = rf_trees.predict(Xval[y_rf_binary_val==2])

    rf_no_trees.fit(Xcal[y_rf_binary_cal==1],ycal[y_rf_binary_cal==1])
    y_rf_val[y_rf_binary_val==1] = rf_no_trees.predict(Xval[y_rf_binary_val==1])

    y_rf[val_blocks==kk] = y_rf_val.copy()

# predicted land cover maps
lc_rf = lc.copy(deep=True)
lc_rf.values*=np.nan
lc_rf.values[mask]=y_rf.copy()

lc_rf_binary = lc.copy(deep=True)
lc_rf_binary.values*=np.nan
lc_rf_binary.values[mask]=y_rf_binary.copy()

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

lc_labels_binary = ['no trees', 'trees']
CM_rf_binary = acc.build_confusion_matrix(y_binary,y_rf_binary)
acc_stats_rf_binary = acc.calculate_accuracy_stats(CM_rf_binary)

fig1, axes = plt.subplots(nrows=1,ncols=2,figsize=(7, 5))
sns.heatmap(CM_rf.astype('int'), annot=True, fmt="d", linewidths=.5, ax=axes[0], cmap="YlGnBu",
            xticklabels=lc_labels, yticklabels=lc_labels, cbar=False, square=True)
sns.heatmap(CM_rf_binary.astype('int'), annot=True, fmt="d", linewidths=.5, ax=axes[1], cmap="YlGnBu",
            xticklabels=lc_labels_binary, yticklabels=lc_labels_binary, cbar=False, square=True)
axes[0].set_title('Confusion Matrix\nFull Classification')
axes[1].set_title('Confusion Matrix\nBinary Classification')
fig1.show()
fig1.tight_layout()
fig1.savefig('%sConfusionMatrix_RF_%s_s1_s2_summer.png' % (path2figures,site))

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
acc_stats_rf_binary_tol1 = acc.calculate_accuracy_stats_with_margin_for_error(lc_binary.values,lc_rf_binary.values)
acc_stats_rf_binary_tol0 = acc.calculate_accuracy_stats_with_margin_for_error(lc_binary.values,lc_rf_binary.values,tolerance=0)


# assess the distribution of omission errors based on the land cover
# characteristics
OEmap = acc_stats_rf_binary_tol1['omission_error_map']
CEmap = acc_stats_rf_binary_tol1['commission_error_map']
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
print('OA    ',(acc_stats_rf_binary_tol1['OA'])*100)


OEmap = acc_stats_rf_binary_tol0['omission_error_map']
CEmap = acc_stats_rf_binary_tol0['commission_error_map']
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
print('OA    ',(acc_stats_rf_binary_tol0['OA'])*100)



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
save layers to file
"""
io.write_xarray_to_GeoTiff(lc_rf,'%s%s_lc_class_rf_s1s2_%.0fm' % (path2output,site,dx))
io.write_xarray_to_GeoTiff(error_map,'%s%s_error_map_rf_s1s2_%.0fm' % (path2output,site,dx))
