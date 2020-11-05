import os
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
import pyproj

from osgeo import osr, gdal
from SSWM.trainingTesting.SRIDConverter import SRIDConverter
from SSWM.trainingTesting.TrainingTestingutils import bin_ndarray
from SSWM.trainingTesting.Hdf5Writer import Hdf5Writer
from SSWM.trainingTesting.GSWInterpolator import GSWInterpolator

# Switch backend display to accomodate X-forwarding on CMC
# https://stackoverflow.com/questions/35737116/runtimeerror-invalid-display-variable
plt.switch_backend('agg')
logger = logging.getLogger(__name__)

class PixStats:
    """ 
    Get satellite images ready for neural net processing
    
    *Parameters*
    
        f_path : str
            imagery file to be converted
        output_dir : str
        
        gsw_path : str
            Path to directory containing global surface water data 
        images_output_dir : str
            
        fst_converter_path : str 
            File path with location to save the files in FST format
    
    """
    # <ID>, <(min,max) incidence angle>
    beam_modes = {'S1':(100,(19,53)),
                 'S2':(102,(19,53)),
                 'S3':(104,(19,53)),
                 'S4':(106,(19,53)),
                 'S5':(108,(19,53)),
                 'S6':(110,(19,53)),
                 'S7':(112,(19,53)),
                 'W1':(200,(19,45)),
                 'W2':(202,(19,45)),
                 'W3':(204,(19,45)),
                 'W4':(206,(19,45)),
                 'F0W1':(208,(19,45)),
                 'F0W2':(210,(19,45)),
                 'F0W3':(212,(19,45)) } 
    
    # These RCM incidence angles and codes are made up right now 
    RCM_modes = {'FSL'   :(1, (19,53)),
                 '3M'    :(2, (19,53)),
                 '5M'    :(3, (19,53)),
                 '16M'   :(4, (19,53)),
                 'SC30M' :(5, (19,53)),
                 'SC50M' :(6, (19,53)),
                 'SC100M':(7, (19,53)),
                 'SCLN'  :(8, (19,53)),
                 'SCSD'  :(9, (19,53)),
                 'QP'    :(10, (19,53))}
    
    polarization = {'HH': np.array([1,0,0,0]),
                    'VV': np.array([0,1,0,0]),
                    'HV': np.array([0,0,1,0]),
                    'VH': np.array([0,0,0,1])
                    }
    
    available_bands = None
                         
    valid_bands=[ 'incidence_angle', 'HH', 'HV', 'VV', 'VH',
                  'Valid Data Pixels', 'Unfiltered Seeds',
                  'Unfilt.Ext from energy HV',  'Final HV+TexEn Ext Mask BLO',
                  'Filtered Seeds', 'Filtered Extended']
    
    def __init__(self,f_path,output_dir=None,gsw_path=None,images_output_dir=None,fst_converter_path=None):
        self.f_path=f_path
        self.available_bands = self.get_valid_bands()
        self.output_dir=output_dir
        self.gsw_path = gsw_path
        self.images_output_dir=images_output_dir
        self.fst_converter_path=fst_converter_path
    
    def get_valid_bands(self):
        """ build dictionary to describe order of bands"""
        img = gdal.Open(self.f_path)
        logger.info(f'Opening file {self.f_path}')
        bnds = {img.GetRasterBand(i + 1).GetDescription(): i + 1  for i in range(img.RasterCount) 
                                        if img.GetRasterBand(i + 1).GetDescription() in self.valid_bands}
        img = None
        return bnds
    
    @classmethod
    def get_file_pol(cls, file):
        """ get valid polarizations for a file """
        img = gdal.Open(file)
        bands = [img.GetRasterBand(i + 1).GetDescription() for i in range(img.RasterCount)]
        bands = [b for b in bands if b in cls.polarization.keys()]
        img = None
        return bands
        
    def get_bands_infos(self):
        f_name=self.f_path
        self.base_name= os.path.basename(f_name)
        ds = gdal.Open(f_name)
        self.original_dataset = ds
        srs = osr.SpatialReference(gdal.Info(f_name, format='json')['coordinateSystem']['wkt'])
        src_srs=srs.ExportToProj4()
        invert_xy=True# srs.IsProjected()# TODO[NB]: what's going on here? It doesn't work for unprojected images if this is left as srs.IsProjected()
        return ds, src_srs, invert_xy
    
    def get_bbox_coords(self,coords):
        min_lat = np.amin(coords[:,0])
        min_lon = np.amin(coords[:,1])
        max_lat = np.amax(coords[:,0])
        max_lon = np.amax(coords[:,1])

        return min_lat,min_lon,max_lat,max_lon
    
    def get_geotransform(self,lat0,lon0,dlat,dlon):
        logger.info(f'Geotransform: \n lat0:{lat0},\nlon0:{lon0},\ndlat: {dlat},\n dlon{dlon}')
        geotransform=[]
        geotransform.append(lon0)           # top left x
        geotransform.append(dlon)           # w-e pixel resolution
        geotransform.append(0)              # 0
        geotransform.append(round(lat0,3))  # top left y
        geotransform.append(0)              # 0
        geotransform.append(-dlat)          # n-s pixel resolution (negative value)
        return geotransform
    
    def to_geotiff(self, array, mask=None, grid_dims=None, f_name=None, geotransform=None, srs=None, gdal_type=gdal.GDT_Byte):
        logger.info("input array".format())
        if not f_name:
            f_name = f'{os.path.splitext(self.base_name)[0]}_a_priori_classification.tiff'
        logger.info('saving {os.path.join(self.images_output_dir, f_name}')
        f_path = os.path.join(self.images_output_dir,f_name)
        logger.info(f'File path: {f_path}')
        driver = gdal.GetDriverByName("GTiff")
        if not grid_dims:
            grid_dims=self.grid_dims
        output_file = driver.Create(f_path, grid_dims[1], grid_dims[0], 1, gdal_type)
        #output_file = driver.Create(f_path, self.grid_dims[1], self.grid_dims[0], 1, gdal.GDT_Byte)
        
        if not geotransform:
            geotransform = self.original_dataset.GetGeoTransform()
        output_file.SetGeoTransform(geotransform)
        
        if not srs:
            srs = self.original_dataset.GetProjection()
        output_file.SetProjection(srs)
        
        output_file.GetRasterBand(1).SetNoDataValue(255)
        if mask is not None:
            logger.info('Setting invalid data to 255')
            logger.info(f'Shapes\nMask : {mask.shape}\nData : {array.shape}')
            array[~mask]=255
            logger.info('Done')
        logger.info('Writing array')
        output_file.GetRasterBand(1).WriteArray(array)
        logger.info('Flushing cache')
        output_file.FlushCache()
        output_file=None
        logger.info(f'Geotiff created successfully at {f_path}')
    
    def get_beam_mode(self):
        base_name= os.path.basename(self.f_path)
        beam_mode_str = base_name.split('_')[1]
        one_hot_beam_mode = np.array([0 if b != beam_mode_str else 1 for b in self.beam_modes])
        return one_hot_beam_mode
    
    def prepare_from_geotif(self, classified_img, convert_probabilities=False, **kwargs):
        """ Equivalent to prepare_fst_info but used when pol_fst_array doesn't exist
        
        *Parameters*
        
        classified_img : str
            path to gdal-supported raster 
        """
        logger.info(f"Preparing npz from {classified_img}")
        ds, src_srs, invert_xy = self.get_bands_infos()

        # get valid pix
        val_pix_band = self.available_bands['Valid Data Pixels']
        valid_pix = np.array(ds.GetRasterBand(val_pix_band).ReadAsArray(), dtype=bool)
        ds=None
        
        # get classification
        cls = gdal.Open(classified_img)
        clsarray = cls.GetRasterBand(1).ReadAsArray()
        if convert_probabilities:
            clsarray[clsarray <= 50] = 0
            clsarray[clsarray > 50] = 1
        cls=None
        
        # select only valid class pixels and flatten
        classified_1d = clsarray.ravel()[valid_pix.ravel()]
                
        
    def prepare_predict(self):
        beam_mode = self.get_beam_mode()
        ds, src_srs, invert_xy = self.get_bands_infos()
        coords, nb_pixels = self.get_coords_for_file(ds, invert_xy)
        wgs84_coords = np.zeros((coords.shape[0], 2), order='F')
        wgs84_coords[:, 1], wgs84_coords[:, 0] = SRIDConverter.convert_from_coordinates_check_geo(coords, src_srs)
        self.coords = wgs84_coords
        min_lat, min_lon, max_lat, max_lon, water_presence = self.get_water_pixels()
        mask = np.array(ds.GetRasterBand(self.available_bands['Valid Data Pixels']).ReadAsArray(),dtype=bool)
        
        #hdf_writer.add_beam_mode(self.f_path,beam_mode)
        water_presence=water_presence.reshape(self.grid_dims)
        
        # Set as ambiguous pixels (anything that is >0 and <90
        water_presence[water_presence==255]=2
        
        # Mask invalid pixels
        water_presence[~mask]=255
        
        # A priori water presence
        self.to_geotiff(water_presence.reshape(self.grid_dims), mask=mask)
        del mask
            
    def get_predict_data(self, index=0, num_procs=8, polarization='HH', water_weight=1.0):
        """
        Put image data into a dictionary that can be fed as features into an tf.estimator.inputs.numpy_input_fn object
            
        *Returns*
        
        A dictionary whose keys correspond to the names of image bands 
        """
        beam_mode = self.get_beam_mode()
        ds, src_srs, invert_xy = self.get_bands_infos()
        mask = np.array(ds.GetRasterBand(self.available_bands['Valid Data Pixels']).ReadAsArray(),dtype=bool)
        shape = mask.shape
        nb_pixels=mask[mask].size
        nb_pixels_per_task = math.ceil(nb_pixels/num_procs)
        start = index * nb_pixels_per_task
        end = start + nb_pixels_per_task
        dict_predict={}
        
        for band in set(['incidence_angle', polarization]).intersection(set(self.available_bands)):
            band_array = np.array(ds.GetRasterBand(self.available_bands[band]).ReadAsArray()[mask].ravel()[start:end],dtype=np.float32)
            dict_predict['beam_mode'] = np.tile(beam_mode,(band_array.shape[0],1))
            dict_predict['weight'] = np.tile(water_weight,(band_array.shape[0],))
            
            if band in self.polarization.keys():
                pol = np.tile(self.polarization[band], (band_array.shape[0],1))
                dict_predict['polarization'] = pol
                dict_predict['value'] = band_array
            
            else:
                dict_predict[band] = band_array
                
        del mask
        return dict_predict
            
    def get_stats(self, write_water_mask=True, write_hist=False):
        """ Create hdf5 file and a priori water mask for radar image 
        
        Creates a *.tif file corresponding to the 89-100% confidence interval for water in the
        GSW product. Also takes any pixels with water likelihood equal to zero or greater than 89 
        and writes them to an hdf5 file (these become the data on which the model will be trained) 
        
        Creates histograms of water/non-water pixel values
        
        """
        f_name = f'{os.path.splitext(os.path.basename(self.f_path))[0]}.h5'
        hdf5_path = os.path.join(self.output_dir,f_name)
        
        if os.path.isfile(hdf5_path):
            logger.warning(f"File {hdf5_path} already exists and will be overwritten")
            os.remove(hdf5_path)
        
        hdf_writer = Hdf5Writer(hdf5_path)
        hdf_writer.open()
        
        beam_mode = self.get_beam_mode()
        ds, src_srs, invert_xy = self.get_bands_infos()
        coords, nb_pixels = self.get_coords_for_file(ds,invert_xy)
        logger.info(f'Coordinates shape: {coords.shape}')
        wgs84_coords = np.zeros((coords.shape[0], 2), order='F')
        wgs84_coords[:, 1], wgs84_coords[:,0] = SRIDConverter.convert_from_coordinates_check_geo(coords, src_srs)
        self.coords = wgs84_coords
        min_lat, min_lon, max_lat, max_lon, water_presence = self.get_water_pixels()
        logger.info(min_lat, min_lon, max_lat, max_lon)
        
        logger.info(f'Water Presence vector (h5): {water_presence}')
        mask = np.array(ds.GetRasterBand(self.available_bands['Valid Data Pixels']).ReadAsArray(),dtype=bool)
        hdf_writer.add_beam_mode(self.f_path, beam_mode)
        water_presence = water_presence.reshape(self.grid_dims)
        
        logger.info(f'MASK: {mask.shape} (shape) \n {mask}')
        logger.info(f'Water Presence: {water_presence.shape}\n {water_presence}')

        water_presence[~mask] = 255
        logger.info(f'{np.sum(water_presence==1)} water pixels. {np.sum(water_presence==0)} land pixels. {np.sum(water_presence==255)} non-water')
        if write_water_mask:
            self.to_geotiff(water_presence.reshape(self.grid_dims), mask=mask)
        del mask
        
        water_presence_idx = water_presence==1
        idx_valid = water_presence != 255
        idx_water = water_presence[idx_valid] == 1
        coords_valid = water_presence.ravel() != 255
        logger.info(f'Adding valid coordinates:{self.coords[coords_valid].shape}')
        hdf_writer.add_rs2_coords(self.f_path, self.coords[coords_valid], min_lat, max_lat, min_lon, max_lon, shape=self.grid_dims)
        logger.info(f'Water Presence: {water_presence.shape}')
        hdf_writer.add_water(self.f_path,idx_water)
        del self.coords
        
        data_bands = set(self.available_bands.keys()) - set(['Valid Data Pixels'])
        for band in data_bands:
            band_array = np.array(ds.GetRasterBand(self.available_bands[band]).ReadAsArray()[idx_valid],dtype=np.float32)
            logger.info(f'Adding {band} values')
            logger.info(f'max:{np.max(band_array)}, min:{np.min(band_array)}')
            if band == 'incidence_angle':
                hdf_writer.add_incidence_angle(self.f_path, band_array)
            elif band in self.polarization.keys():
                hdf_writer.add_pol(self.f_path, band, band_array, self.polarization[band])
                if write_hist:
                    plt.figure()
                    self.plot_histogram_data(band, band_array, idx_water, ~idx_water)
                    plt.close()
            else:
                hdf_writer.add_generic(self.f_path, band, band_array)
                if write_hist:
                    plt.figure()
                    self.plot_histogram_data(band, band_array, idx_water, ~idx_water)
                    plt.close()
            del band_array
            logger.info('Done')
  
    def plot_histogram3d_data(self,ds,band='HH',water_presence=None,all_unambiguous_water=None):
        f_name = f'{os.path.splitext(self.base_name)[0]}_histogram_{band}.png'
        f_path = os.path.join(self.output_dir,f_name)
        incidence_angle = ds.GetRasterBand(self.available_bands['incidence_angle']).ReadAsArray()
        band_data = ds.GetRasterBand(self.available_bands[band]).ReadAsArray()
        #hist, bins = np.histogram(band_data.ravel()[all_unambiguous_water], bins=2048)
        hist, xedges, yedges = np.histogram2d(band_data.ravel()[all_unambiguous_water], incidence_angle.ravel()[all_unambiguous_water], bins=2048)
        hist_water, xedges, yedges = np.histogram2d(band_data.ravel()[water_presence],incidence_angle.ravel()[water_presence], bins=(xedges,yedges))
        #plt.zscale('log', nonposz='clip')
        plt.xlim([-30,20])
        self.plot_histogram3d(hist, xedges, yedges)
        self.plot_histogram3d(hist_water, xedges, yedges)
        plt.title(f"Histogram {band}")
        plt.savefig(f_path, bbox_inches='tight')
        
    def plot_histogram_data(self, band='HH', band_data=None, water_presence_idx=None, no_water_idx=None):
        f_name = f'{os.path.splitext(self.base_name)[0]}_histogram_{band}.png'
        f_path = os.path.join(self.images_output_dir,f_name)
        logger.info(f'plotting {band} data with shape {band_data.shape}')
        
        hist, bins = np.histogram(band_data[no_water_idx], bins=2048)
        hist_water, bins = np.histogram(band_data[water_presence_idx], bins=bins)
        plt.yscale('log', nonposy='clip')
        #plt.xlim([-30,20])
        self.plot_histogram(hist, bins, color=(1.0, 0.0, 0.0, 0.5))
        self.plot_histogram(hist_water, bins, color=(0.0, 0.0, 1.0, 0.5))
        plt.title(f"Histogram {band}")
        plt.savefig(f_path, bbox_inches='tight')
        
    def plot_histogram(self, hist, bins, **kwargs):
        logger.info(hist)
        logger.info(bins)
        width = np.diff(bins)
        plt.bar(bins[:-1], hist, width=width, **kwargs)
    
    def plot_histogram3d(self,hist,xedges,yedges):
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
        xpos = xpos.flatten('F')
        ypos = ypos.flatten('F')
        zpos = np.zeros_like(xpos)
        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()
        dz = hist.flatten()
        plt.bar3d(xpos, ypos, zpos, dx, dy, dz)

    def get_coords_for_file(self,ds,invert_xy=False):
        logger.info('Get coords for file called')
        logger.info('invert_xy is {}'.format(invert_xy))
        (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = ds.GetGeoTransform()
        if invert_xy:
            x = (np.arange(ds.RasterXSize,dtype=np.float32) * x_size + upper_left_x + (x_size / 2.))
            y = (np.arange(ds.RasterYSize,dtype=np.float32) * y_size + upper_left_y + (y_size / 2.))
        else:
            x = (np.arange(ds.RasterYSize,dtype=np.float32) * x_size + upper_left_x + (x_size / 2.))
            y = (np.arange(ds.RasterXSize,dtype=np.float32) * y_size + upper_left_y + (y_size / 2.))
        xs = np.empty((ds.RasterYSize, ds.RasterXSize), dtype='float32')
        ys = np.empty((ds.RasterYSize, ds.RasterXSize), dtype='float32')
        xs[:,:], ys[:,:] = np.meshgrid(x,y, copy=False)
        
        logger.info(xs)
        logger.info(ys)
        logger.info(xs.shape)
        self.grid_dims = xs.shape
        nb_pixels = ds.RasterYSize * ds.RasterXSize
        coords = np.empty((nb_pixels,2), dtype='float32')
        coords[:,0] = xs.ravel()
        coords[:,1] = ys.ravel()
        del x, y, xs, ys
        #coords = np.hstack((xs.ravel()[:, np.newaxis], ys.ravel()[:, np.newaxis]))
        
        return coords, nb_pixels
    
    def get_band_histogram(self,band_number,bins=2048,name=None):
        plt.figure(figsize=(1024, 768))
        b_name = f'{os.path.splitext(self.base_name)[0]}_histogram_{name}.png'
        band_data = ds.GetRasterBand(band_number).ReadAsArray()
        plt.hist(band_data, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram")
        plt.savefig(os.path.join(self.output_dir, b_name), bbox_inches='tight')
        
    def get_water_pixels(self):
        interpolator = GSWInterpolator(sat_f_name=self.f_path, gsw_dir=self.gsw_path, output_dir=self.output_dir)
        min_lat, min_lon, max_lat, max_lon = self.get_bbox_coords(self.coords)
        logger.info("Bounding box: lat ({}, {}), lon({},{})".format(min_lat, max_lat, min_lon, max_lon))
        return min_lat, min_lon, max_lat, max_lon, interpolator.get_water_presence_for_points(min_lat,max_lat,min_lon,max_lon,self.coords)
    
    def make_training_samples(self):
        pass
