"""
Basicread/write functions
"""
import numpy as np
import glob
import xarray as xr #xarray to read all types of formats
import rasterio
from osgeo import gdal
import osr

"""
load_geotiff
A very simple function that reads a geotiff and returns it as an xarray. Nodata
values are converted to the numpy nodata value.
The input arguments are:
- filename (this should include the full path to the file)
Optional arguments are:
- band (default = 1)
- x_name (default = 'longitude')
- y_name (default = 'latitude')
- nodata_option (default = 0).
            0: use value in dataset metadata. This is usually fine, except if
                there is an issue with the precision, and is applied in all
                cases.
            1: arbitrary cutoff for nodata to account for precision problems
                with float32. Other similar options could be added if necessary.
            2: set all negative values as nodata


"""
def load_geotiff(filename, band = 1,x_name='longitude',y_name='latitude',option=0):
    xarr = xr.open_rasterio(filename).sel(band=band)
    if(option==0):
        xarr.values[xarr.values==xarr.nodatavals[0]]=np.nan
    if(option==1):
        xarr.values[xarr.values<-3*10**38]=np.nan
    if(option==2):
        xarr.values[xarr.values<0]=np.nan
    return xarr #return the xarray object

"""
copy_xarray_template
"""
def copy_xarray_template(xarr):
    xarr_new = xarr.copy()
    xarr_new.values = np.zeros(xarr.values.shape)*np.nan
    return xarr_new

"""
create a geotransformation object from an xarray object, needed for writing
geotiff with gdal
"""
def create_geoTrans(array,x_name='x',y_name='y'):
    lat = array.coords[y_name].values
    lon = array.coords[x_name].values
    dlat = lat[1]-lat[0]
    dlon = lon[1]-lon[0]
    geoTrans = [0,dlon,0,0,0,dlat]
    geoTrans[0] = np.min(lon)-dlon/2.
    if geoTrans[5]>0:
        geoTrans[3]=np.min(lat)-dlat/2.
    else:
        geoTrans[3]=np.max(lat)-dlat/2.
    return geoTrans

"""
check orientation of array
"""
def check_array_orientation(array,geoTrans,north_up=True):
    if north_up:
        # for north_up array, need the n-s resolution (element 5) to be negative
        if geoTrans[5]>0:
            geoTrans[5]*=-1
            geoTrans[3] = geoTrans[3]-(array.shape[0]+1.)*geoTrans[5]
        # Get array dimensions and flip so that it plots in the correct orientation on GIS platforms
        if len(array.shape) < 2:
            print('array has less than two dimensions! Unable to write to raster')
            sys.exit(1)
        elif len(array.shape) == 2:
            array = np.flipud(array)
        elif len(array.shape) == 3:
            (NRows,NCols,NBands) = array.shape
            for i in range(0,NBands):
                array[:,:,i] = np.flipud(array[:,:,i])
        else:
            print('array has too many dimensions! Unable to write to raster')
            sys.exit(1)

    else:
        # for north_up array, need the n-s resolution (element 5) to be positive
        if geoTrans[5]<0:
            geoTrans[5]*=-1
            geoTrans[3] = geoTrans[3]-(array.shape[0]+1.)*geoTrans[5]
        # Get array dimensions and flip so that it plots in the correct orientation on GIS platforms
        if len(array.shape) < 2:
            print('array has less than two dimensions! Unable to write to raster')
            sys.exit(1)
        elif len(array.shape) == 2:
            array = np.flipud(array)
        elif len(array.shape) == 3:
            (NRows,NCols,NBands) = array.shape
            for i in range(0,NBands):
                array[:,:,i] = np.flipud(array[:,:,i])
        else:
            print ('array has too many dimensions! Unable to write to raster')
            sys.exit(1)

    # Get array dimensions and flip so that it plots in the correct orientation on GIS platforms
    if len(array.shape) < 2:
        print ('array has less than two dimensions! Unable to write to raster')
        sys.exit(1)
    elif len(array.shape) == 2:
        array = np.flipud(array)
    elif len(array.shape) == 3:
        (NRows,NCols,NBands) = array.shape
        for i in range(0,NBands):
            array[:,:,i] = np.flipud(array[:,:,i])
    else:
        print ('array has too many dimensions! Unable to write to raster')
        sys.exit(1)

    return array,geoTrans

"""
write xarray to geotiff. input arguments:
- an xarray object to write to file. This assumes that the array has only two
  dimensions can update if you need to write multiband arrays)
- the filename
- (optional) orientation (north_up = True is the default). If things are upside
  down when you view file in a GIS package, try switching to False
"""
def write_xarray_to_GeoTiff(array, outfilename,north_up=True,EPSG_CODE = ''):

    # check filename suffix is .tif
    if outfilename[:-4]!='.tif':
        outfilename=outfilename+'.tif'

    # Some dimension info
    if len(array.values.shape) < 2:
        print ('array has less than two dimensions! Unable to write to raster')
        sys.exit(1)
    elif len(array.shape) == 2:
        NRows,NCols = array.values.shape
        NBands = 1
        array = array.expand_dims('band',axis=-1)
    elif len(array.values.shape) == 3:
        NRows,NCols,NBands = array.values.shape
    else:
        print ('array has too many dimensions! Unable to write to raster')
        sys.exit(1)

    # create geotrans object
    geoTrans = create_geoTrans(array)
    #EPSG_CODE = array.attrs['crs'].split(':')[-1]

    # check orientation
    array.values,geoTrans = check_array_orientation(array.values.copy(),geoTrans,north_up=north_up)

    # set nodatavalue
    array.values[np.isnan(array.values)] = -9999

    # Write GeoTiff
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    # set all the relevant geospatial information
    dataset = driver.Create( outfilename, NCols, NRows, NBands, gdal.GDT_Float32 )
    dataset.SetGeoTransform( geoTrans )
    if len(EPSG_CODE)>0:
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS( 'EPSG:'+EPSG_CODE )
        dataset.SetProjection( srs.ExportToWkt() )
    # write array
    for bb in range(0,NBands):
        dataset.GetRasterBand(bb+1).SetNoDataValue( -9999 )
        dataset.GetRasterBand(bb+1).WriteArray( array.values[:,:,bb] )
    dataset = None
    return 0
