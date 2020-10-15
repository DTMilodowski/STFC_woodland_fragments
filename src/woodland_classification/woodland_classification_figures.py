"""
woodland_classification_figures.py
================================================================================
Script to generate figures for paper on woodland mapping in NW Scotland
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


print('Loading LiDAR layers')
dtm=xr.open_rasterio(dtm_file).sel(band=1)
dsm=xr.open_rasterio(dsm_file).sel(band=1)
dtm.values[dtm.values==-9999.]=np.nan
dsm.values[dsm.values==-9999.]=np.nan
tch = dsm - dtm
tch.values[tch.values<0]=0

print('Loading reference maps')
lc_labels = ['ground', 'shrubs', 'isolated trees', 'open woodland', 'dense woodland']
lc = xr.open_rasterio('%s%s_lc_class_%.0fm.tif' % (path2output,site,dx))[0]
lc.values[lc.values==lc.nodatavals[0]]=np.nan

# Create a simplified version 1 = no trees, 2 = trees
lc_simple = lc.copy(deep=True)
lc_simple.values[lc.values<3]=1
lc_simple.values[lc.values>=3]=2

print('Loading predicted land cover')
lc_rf_s1s2 = xr.open_rasterio('%s%s_lc_class_rf_s1s2_%.0fm.tif' % (path2output,site,dx))[0]
lc_rf_s1s2.values[lc_rf_s1s2.values==lc_rf_s1s2.nodatavals[0]]=np.nan
lc_rf_s1__ = xr.open_rasterio('%s%s_lc_class_rf_s1___%.0fm.tif' % (path2output,site,dx))[0]
lc_rf_s1__.values[lc_rf_s1__.values==lc_rf_s1__.nodatavals[0]]=np.nan


print('Loading error maps')
emap_s1s2 = xr.open_rasterio('%s%s_error_map_rf_s1s2_%.0fm.tif' % (path2output,site,dx))[0]
emap_s1s2.values[emap_s1s2.values==emap_s1s2.nodatavals[0]]=np.nan
emap_s1__ = xr.open_rasterio('%s%s_error_map_rf_s1___%.0fm.tif' % (path2output,site,dx))[0]
emap_s1__.values[emap_s1__.values==emap_s1__.nodatavals[0]]=np.nan


"""
***Figure***
Location map and LiDAR canopy height map
"""


"""
***Figure***
Reference land cover map, prediction (S1 & S2), prediction (S1 only)
"""
colours = np.asarray(['#ffcfe2', '#ff9dc8', '#00cba7', '#00735c', '#003d30'])
lc_cmap = ListedColormap(sns.color_palette(colours).as_hex())

plt.rcParams["axes.axisbelow"] = False
fig2,axes = plt.subplots(nrows=3,ncols=1,sharex=True,sharey=True,figsize=(9,11))
fig2.subplots_adjust(right=0.75)
lc.plot(ax=axes[0],cmap = lc_cmap, add_colorbar=False)
lc_rf_s1s2.plot(ax=axes[1],cmap = lc_cmap, add_colorbar=False)
lc_rf_s1__.plot(ax=axes[2],cmap = lc_cmap, add_colorbar=False)

#for ii, title in enumerate(['Reference (LiDAR)','Support Vector Machine', 'Random Forest']):
for ii, title in enumerate(['Reference (LiDAR)',
                            'Classification with Sentinel 1 & Sentinel 2',
                            'Classification with Sentinel 1 only']):
    axes[ii].set_title(title)
    axes[ii].set_ylabel('Northing / m')
    axes[ii].set_xlabel('')
    axes[ii].set_aspect('equal')
    axes[ii].grid(True,which='both')

axes[-1].set_xlabel('Easting / m')

# plot legend for color map
cax = fig2.add_axes([0.75,0.375,0.05,0.25])
bounds = np.arange(len(lc_labels)+1)
norm = mpl.colors.BoundaryNorm(bounds, lc_cmap.N)
cb = mpl.colorbar.ColorbarBase(cax,cmap=lc_cmap,norm=norm,orientation='vertical')
n_class = len(lc_labels)
loc = np.arange(0,n_class)+0.5
cb.set_ticks(loc)
cb.set_ticklabels(lc_labels)
cb.update_ticks()

fig2.show()
fig2.savefig('landcover_classifications_comparison_%s.png' % site)


"""
***Figure***
Zoom in to an area with mixed land cover. 6 panels (2 cols, 3 rows)

LiDAR                               Reference
classification S1&S2                classification S1 only
Error map S1&S2                     classification S1 only
"""
error_colours = np.asarray(['white', 'black', '#cecece', '#ff9DC8', '#656565', '#00cba7'])[::-1]
error_labels = np.asarray(['no trees', 'trees',
                            'adjacent omission\nerror', 'omission error',
                            'adjacent commission\n error', 'commission error'])[::-1]
error_cmap = ListedColormap(sns.color_palette(error_colours).as_hex())

N=785050
S=784450
E=168450
W=167850
plt.rcParams["axes.axisbelow"] = False
fig3,axes = plt.subplots(nrows=3,ncols=2,sharex=True,sharey=True,figsize=(9,11))
fig3.subplots_adjust(right=0.66)
im_tch = tch.sel(x=slice(W,E),y=slice(N,S)).plot(ax=axes[0,0],vmin=0,vmax=5,cmap = 'viridis', add_colorbar=False)
lc.sel(x=slice(W,E),y=slice(N,S)).plot(ax=axes[0,1],cmap = lc_cmap, add_colorbar=False)
lc_rf_s1s2.sel(x=slice(W,E),y=slice(N,S)).plot(ax=axes[1,0],cmap = lc_cmap, add_colorbar=False)
lc_rf_s1__.sel(x=slice(W,E),y=slice(N,S)).plot(ax=axes[1,1],cmap = lc_cmap, add_colorbar=False)
emap_s1s2.sel(x=slice(W,E),y=slice(N,S)).plot(ax=axes[2,0],cmap = error_cmap, add_colorbar=False)
emap_s1__.sel(x=slice(W,E),y=slice(N,S)).plot(ax=axes[2,1],cmap = error_cmap, add_colorbar=False)

#for ii, title in enumerate(['Reference (LiDAR)','Support Vector Machine', 'Random Forest']):
for ii, title in enumerate(['LiDAR TCH', 'Reference Classification (LiDAR)',
                            'Classification (S1 & S2)','Classification (S1 only)',
                            'Error map (S1 & S2)','Error map (S1 only)']):
    axes.flatten()[ii].set_title(title)
    axes.flatten()[ii].set_ylabel('')
    axes.flatten()[ii].set_xlabel('')
    axes.flatten()[ii].set_aspect('equal')
    axes.flatten()[ii].grid(True,which='both')

axes[2,0].set_xlabel('Easting / m')
axes[2,1].set_xlabel('Easting / m')
axes[0,0].set_ylabel('Northing / m')
axes[1,0].set_ylabel('Northing / m')
axes[2,0].set_ylabel('Northing / m')

# plot legend for color map
cax1 = fig3.add_axes([0.70,0.675,0.05,0.2])
cb1 = fig3.colorbar(im_tch, cax=cax1,orientation='vertical',extend='max')
cb1.set_label('Top of canopy height / m')

cax2 = fig3.add_axes([0.70,0.4,0.05,0.2])
bounds = np.arange(len(lc_labels)+1)
norm = mpl.colors.BoundaryNorm(bounds, lc_cmap.N)
cb2 = mpl.colorbar.ColorbarBase(cax2,cmap=lc_cmap,norm=norm,orientation='vertical')
n_class = len(lc_labels)
loc = np.arange(0,n_class)+0.5
cb2.set_ticks(loc)
cb2.set_ticklabels(lc_labels)
cb2.update_ticks()

cax3 = fig3.add_axes([0.7,0.125,0.05,0.2])
bounds = np.arange(len(error_labels)+1)
norm = mpl.colors.BoundaryNorm(bounds, error_cmap.N)
cb3 = mpl.colorbar.ColorbarBase(cax3,cmap=error_cmap,norm=norm,orientation='vertical')
n_class = len(error_labels)
loc = np.arange(0,n_class)+0.5
cb3.set_ticks(loc)
cb3.set_ticklabels(error_labels)
cb3.update_ticks()

fig3.show()
fig3.savefig('classification_comparison_detail_%s.png' % site)


"""
***figure***
Comparison of the error maps S1&S2 vs S1 only
"""
fig4,axes = plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,figsize=(8,7))
fig4.subplots_adjust(right=0.7)
emap_s1s2.plot(ax=axes[0],cmap = error_cmap, add_colorbar=False)
#lc_svm.plot(ax=axes[1],cmap = lc_cmap)
emap_s1__.plot(ax=axes[1],cmap = error_cmap, add_colorbar=False)

#for ii, title in enumerate(['Reference (LiDAR)','Support Vector Machine', 'Random Forest']):
for ii, title in enumerate(['Classification (S1 & S2)','Classification (S1 only)']):
    axes[ii].set_title(title)
    axes[ii].set_xlabel('')
    axes[ii].set_ylabel('Northing / m')
    axes[ii].set_aspect('equal')
    axes[ii].grid(True,which='both')

axes[1].set_xlabel('Easting / m')

# plot legend for color map
cax = fig4.add_axes([0.725,0.35,0.05,0.3])
bounds = np.arange(len(error_labels)+1)
norm = mpl.colors.BoundaryNorm(bounds, error_cmap.N)
cb = mpl.colorbar.ColorbarBase(cax,cmap=error_cmap,norm=norm,orientation='vertical')
n_class = len(error_labels)
loc = np.arange(0,n_class)+0.5
cb.set_ticks(loc)
cb.set_ticklabels(error_labels)
cb.update_ticks()

fig4.show()
fig4.savefig('error_map_classifications_%s.png' % site)
