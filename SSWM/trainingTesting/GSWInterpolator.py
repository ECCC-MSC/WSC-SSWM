import math
import numpy as np
import os
import shutil
import logging

from osgeo import gdal
from scipy.interpolate import RegularGridInterpolator as rgi

logger = logging.getLogger(__name__)

class GSWInterpolator:
    """ Handle Global Surface Water files
    
    *Parameters*
   
    sat_f_name : str
        File path to satellite imagery scene for which water mask is to be interpolated
    gsw_dir : str
        File path to location of global surface water *.tif files 
    output_dir : str
        File path to desired location of output files
    data : 
    use_cols_vector :     
    """
    THRESHOLD = 90 # GSW percent-water threshold to use as training data 
    interpolators={}
    
    def __init__(self,sat_f_name,gsw_dir,output_dir=None,data=None,use_cols_vector=None):
        self.gsw_dir=gsw_dir
        self.sat_f_name=sat_f_name
        self.output_dir=os.getcwd() if not output_dir else output_dir
        self.data = data
        self.use_cols_vector=use_cols_vector
          
    def get_covering_global_surface_water_file_names(self,min_lat,max_lat,min_lon,max_lon):
        """
        Assuming we're running in the path where there is a 'coverage' directory 
        containing all the RS2 BBOX and convex hulls.
        """
        def roundup(value):
            return int(math.ceil(abs(value)/10)*10)
        
        base_f_name = 'occurrence_{:}W_{:}N.tif'
        gsw_names = []
        limits_lat = roundup(min_lat),roundup(max_lat)
        limits_lon = roundup(min_lon),roundup(max_lon)
        logger.info(f'latitude limits: {limits_lat}')
        logger.info(f'longitude limits : {limits_lon}')
        
        if limits_lat[0] == limits_lat[1]:
            limits_lat = limits_lat[0],
        if limits_lon[0] == limits_lon[1]:
            limits_lon = limits_lon[0],
        for lat in limits_lat:
            for lon in limits_lon:
                gsw_name = os.path.join(self.gsw_dir,base_f_name.format(lon,lat))
                tmp_name = os.path.join('/tmp', os.path.basename(gsw_name))
                #shutil.copy(gsw_name,tmp_name)
                #gsw_names.append(tmp_name)
                gsw_names.append(gsw_name)
        logger.info('gsw files required:\n'+'\n'.join(str(os.path.basename(c)) for c in gsw_names))
 
        return gsw_names

    
    def get_gsw_information(self,rgi_file):
        logger.info(f'Get gsw information: {rgi_file}')
        
        ds = gdal.Open(rgi_file)
        logger.info(f'DS loaded with gdal: {ds}')
        upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size = ds.GetGeoTransform()
        band = ds.GetRasterBand(1)
        print('a')
        band_data = band.ReadAsArray().astype(np.uint8)
        print('z')
        x = (np.arange(band_data.shape[1],dtype=np.float32) * x_size + upper_left_x + (x_size/2.))
        y = (np.arange(band_data.shape[0],dtype=np.float32) * y_size + upper_left_y + (y_size / 2.))
        band_data[(band_data > 0) & (band_data < self.THRESHOLD)] = 255
        band_data[(band_data >= self.THRESHOLD) & (band_data <=100)] = 1
        band_data[(band_data > 100) | np.isnan(band_data)] = 255
        unique, counts = np.unique(band_data, return_counts=True)
        print('b')
        logger.info(np.asarray((unique, counts)).T)
        logger.info(f'Band data loaded. Shape is: {band_data.shape}')
        logger.info(f'number of water pixels: {np.sum(band_data == 1)}, number of non-water: {np.sum(band_data == 255)}')
        return x, y, band_data
        
    
    def get_regular_grid_interpolator(self,rgi_file):
        #SRC
        x, y, band_data = self.get_gsw_information(rgi_file)
        interpolator = rgi(points=(y[::-1], x), values=np.flip(band_data, 0), method='nearest', bounds_error=False, fill_value=255)
        del x, y, band_data
        return interpolator
          
    def set_least_used_interpolator_key(self):
        lowest=1000
        for k,v in self.interpolators.items():
            if v['uses'] < lowest:
                self.least_used_interpolator=k
                lowest=v['uses']

    def get_data_from_npy(self,f_path,percent_data=1.0):
        use_cols_vector=[]
        input_pos = {'lats':0,
                    'lons':1,
                    'HH':2,
                    'VV':3,
                    'HV':4,
                    'VH':5,
                    'incidence_angle':6,
                    'beam_mode_id':7,
                    'is_water':8 }
        
        cols_index=np.array([])
        cols=np.array([])
        if f_path.endswith('.npz'):
            data = np.load(f_path)
            final_data=data['data'][:]
            use_cols_vector=data['using_vector']
        else:
            final_data = np.load(f_path)
            
        return final_data,use_cols_vector

    def get_water_presence_for_points(self, min_lat, max_lat, min_lon, max_lon, pts):
        """ """
        logger.info('DESTINATION COORDINATES:')
        logger.info(pts)
        #SRC:
        rgi_files = self.get_covering_global_surface_water_file_names(min_lat, max_lat, min_lon, max_lon)
        
        interpolators=[]
        for rgi_file in rgi_files:
            interp_name = os.path.basename(rgi_file)
            lat_lon_items = os.path.splitext(interp_name)[0].split('_')
            lat = int(lat_lon_items[2][:-1])
            lon = int(lat_lon_items[1][:-1])
            interpolators.append((self.interpolators.setdefault(
                                                interp_name,{}).setdefault('interpolator',
                                                self.get_regular_grid_interpolator(rgi_file)
                                                ),  
                                  lon,
                                  lat
                                  )
                                )
            logger.info(f'Interpolators created => {interpolators}')
            uses = self.interpolators[interp_name].setdefault('uses',0)
            self.interpolators[interp_name]['uses']+= 1
        logger.info(f'Interpolators check Created: {self.interpolators}')
        water_presence = np.zeros(pts.shape[0])
        if len(rgi_files) > 1:
            for interpolator, lon, lat in interpolators:
                logger.info(f"assigning pixels for latitudes in ({lat - 10}, {lat}) and longitudes in {-lon}, {-lon+10})")
                logger.warning("This assumes surface water index files have negative longitudes (W)")
                idx_valid_points = np.where((pts[:,0] > lat - 10) & (pts[:,0] < lat) & (pts[:,1] > -lon) & (pts[:,1] < -lon + 10))
                logger.info(f"valid indices for pixels using this interpolator")
                logger.info(idx_valid_points)
                logger.info(pts[idx_valid_points[0],:])
                logger.info(interpolator(pts[idx_valid_points[0], :]))
                water_presence[idx_valid_points[0]]=interpolator(pts[idx_valid_points[0], :])
        else:
            interpolator, lon, lat = interpolators[0]
            #print(np.amin(pts[:,0]),np.amax(pts[:,0]))
            #print(np.amin(pts[:,1]),np.amax(pts[:,1]))
            idx_valid_pts=((pts[:,0] != 0.) & (pts[:,1] != 0.))
            water_presence[:] = interpolator(pts[idx_valid_pts])
        logger.info(f'Water presence size: {water_presence.shape[0]}')
        logger.info(f'Points Size : {pts.shape[0]}')
        logger.info(f'Number of water pixels : {np.sum(water_presence==1)}')
        return water_presence

