"""
lidar_reference_classification.py
================================================================================
This script creates a reference tree cover classification using gridded airborne
laser scanning data at 10m resolution
Classification used is:
- bare/ground vegetation
- shrubs
- isolated trees (>8 contiguous pixels above two metres height)
- open woodland (>20% canopy cover at 2m)
- closed woodland (>50% canopy cover at 2m)

The landcover map is produced on an identical grid to the remote sensing layers
used in the subsequent analysis. Regridding uses gdalwarp via a command line
call
--------------------------------------------------------------------------------
"""
import os
import sys

import numpy as np
import xarray as xr
from skimage import morphology

sys.path.append('../data_io')
import data_io as io

"""
Defining some general variables and other project details
--------------------------------------------------------------------------------
"""
site = 'Arisaig'

# paths
path2s1 = '/disk/scratch/local.2/dmilodow/Sentinel1/processed_temporal/2019/'
path2lidar = '/disk/scratch/local.2/dmilodow/SEOS/raster/OSGBgrid/'
path2output = '/exports/csce/datastore/geos/users/dmilodow/STFC/woodland_fragments/output/'

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

# open one of the S1 files for template
s1vh_A_file = '%s%s/S1A__IW__A_2019_VH_tnr_bnr_Orb_Cal_TF_TC_dB_temporal_mean_summer.tif' % (path2s1,site)
template = xr.open_rasterio(s1vh_A_file).sel(band=1)
dx = template.x.values[1]-template.x.values[0]
dy = template.y.values[1]-template.y.values[0]
E = template.x.values.max()+np.abs(dx)/2.
W = template.x.values.min()-np.abs(dx)/2.
N = template.y.values.max()+np.abs(dy)/2.
S = template.y.values.min()-np.abs(dy)/2.

"""
Generate reference land cover classes based on LiDAR TCH
- three phases
    - generate rasters at 1m resolution based on the CHM
    - regrid using gdal to 10m resolution
    - classify into four classes
        1) Bare/grass
        2) Low vegetation
        3) Isolated trees
        4) Open woodland
        5) Dense woodland
--------------------------------------------------------------------------------
"""
print('Generating reference maps from lidar data')
# load DSM and DTM
dtm=xr.open_rasterio(dtm_file).sel(band=1)
dsm=xr.open_rasterio(dsm_file).sel(band=1)
dtm.values[dtm.values==-9999.]=np.nan
dsm.values[dsm.values==-9999.]=np.nan

# TCH
tch = dsm - dtm
tch.values[tch.values<0]=0

# Cover at 2m height
cov_2m = tch.copy(deep=True)
cov_2m.values = (tch.values>=2).astype('float')
cov_2m.values[np.isnan(tch.values)]=np.nan
io.write_xarray_to_GeoTiff(cov_2m, '%s_cover2m_1m' % site)

# - Cover at 0.1m height
cov_10cm = tch.copy(deep=True)
cov_10cm.values = (tch.values>=0.1).astype('float')
cov_10cm.values[np.isnan(tch.values)]=np.nan
io.write_xarray_to_GeoTiff(cov_10cm, '%s_cover10cm_1m' % site)

# Identify "trees"
# - define as contiguous regions with canopy cover >2m comprising >8 pixels
#   (~3m diameter)
# - use two-step procedure
#       (i)  fill "holes"
#       (ii) remove objects based on connectivity direct connections in either
#            row or column (ignoring diagonals)
trees = cov_2m.copy(deep=True)
trees.values[trees.values<0]=0
trees.values = morphology.remove_small_holes(trees.values.astype('int'),area_threshold=1)
trees.values = morphology.remove_small_objects(trees.values,min_size=8).astype('float')
trees.values[np.isnan(tch.values)]=np.nan
io.write_xarray_to_GeoTiff(trees, '%s_trees2m_1m' % site)

trees = None; cov_2m = None; cov_10cm = None

# Use gdal to regrid
os.system('gdalwarp -overwrite -r average -te %f %f %f %f -tr %f %f %s_cover2m_1m.tif %s_cover2m_%.0fm.tif' % (W, S, E, N, dx, dy, site, site, dx))
os.system('gdalwarp -overwrite -r max -te %f %f %f %f -tr %f %f %s_trees2m_1m.tif %s_trees2m_%.0fm.tif' % (W, S, E, N, dx, dy, site, site, dx))
os.system('gdalwarp -overwrite -r average -te %f %f %f %f -tr %f %f %s_cover10cm_1m.tif %s_cover10cm_%.0fm.tif' % (W, S, E, N, dx, dy, site, site, dx))

# now load regridded layers
trees = xr.open_rasterio('%s_trees2m_%.0fm.tif' % (site, dx)).sel(band=1)
cov_2m = xr.open_rasterio('%s_cover2m_%.0fm.tif' % (site, dx)).sel(band=1)
cov_10cm = xr.open_rasterio('%s_cover10cm_%.0fm.tif' % (site, dx)).sel(band=1)

# Buffer edge of survey to acoid edge effects on regridding
nanmask = cov_2m.values==cov_2m.nodatavals[0]
nanmask = morphology.binary_dilation(nanmask,selem=np.ones((3,3)))
trees.values[nanmask]=np.nan
cov_2m.values[nanmask]=np.nan
cov_10cm.values[nanmask]=np.nan

# Generate land cover product
lc = cov_2m.copy(deep=True)
lc.values[(trees==0)*(cov_10cm<0.2)]=1 # ground vegetation
lc.values[(trees==0)*(cov_10cm>=0.2)*(cov_2m<0.2)]=2 # low vegetation
lc.values[(trees==1)*(cov_2m<0.2)]=3 # isolated trees
lc.values[cov_2m>=0.2]=4 # open woodland
lc.values[cov_2m>=0.5]=5 # closed woodland
lc.values[nanmask]=np.nan

io.write_xarray_to_GeoTiff(lc, '%s%s_lc_class_%.0fm.tif' % (path2output,site,dx))
