import os
import math
import numpy as np
import logging
import pyproj

from osgeo import osr, gdal
from SSWM.trainingTesting.SRIDConverter import SRIDConverter
from SSWM.trainingTesting.TrainingTestingutils import bin_ndarray, bandnames
from SSWM.trainingTesting.GSWInterpolator import GSWInterpolator

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
    
    available_bands = None
                         
    valid_bands=bandnames.DATA_BANDS + bandnames.VALID_PIX_BAND
    
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

    def get_stats_and_sample(self, valseed, nwater, nland, max_L2W_ratio, write_water_mask=False):
        """ Sample an image, we no longer use H5 files

        """

        ds, src_srs, invert_xy = self.get_bands_infos()
        coords, nb_pixels = self.get_coords_for_file(ds, invert_xy)
        logger.info(f'Coordinates shape: {coords.shape}')
        wgs84_coords = np.zeros((coords.shape[0], 2), order='F')
        wgs84_coords[:, 1], wgs84_coords[:, 0] = SRIDConverter.convert_from_coordinates_check_geo(coords, src_srs)
        self.coords = wgs84_coords
        min_lat, min_lon, max_lat, max_lon, water_presence = self.get_water_pixels()

        logger.info("min_lat {}, min_lon {}, max_lat {}, max_lon {} ".format(min_lat, min_lon, max_lat, max_lon))

        logger.info(f'Water Presence vector (h5): {water_presence}')
        mask = np.array(ds.GetRasterBand(self.available_bands['Valid Data Pixels']).ReadAsArray(), dtype=bool)

        water_presence = water_presence.reshape(self.grid_dims)

        logger.info(f'MASK: {mask.shape} (shape) \n {mask}')
        logger.info(f'Water Presence: {water_presence.shape}\n {water_presence}')

        water_presence[~mask] = 255
        logger.info(
            f'{np.sum(water_presence == 1)} water pixels. {np.sum(water_presence == 0)} land pixels. {np.sum(water_presence == 255)} non-water')
        if write_water_mask:
            self.to_geotiff(water_presence.reshape(self.grid_dims), mask=mask)
        del mask

        # water_presence_idx = water_presence==1

        idx_valid = water_presence != 255
        idx_water = np.nonzero(water_presence[idx_valid] == 1)[0]
        idx_land = np.nonzero(water_presence[idx_valid] == 0)[0]

        valid_vals = water_presence[idx_valid] == 1
        # coords_valid = water_presence.ravel() != 255
        # logger.info(f'Adding valid coordinates:{self.coords[coords_valid].shape}')
        # hdf_writer.add_rs2_coords(self.f_path, self.coords[coords_valid], min_lat, max_lat, min_lon, max_lon, shape=self.grid_dims)
        # logger.info(f'Water Presence: {water_presence.shape}')
        # hdf_writer.add_water(self.f_path,idx_water)
        del self.coords

        if max_L2W_ratio:
            nwat = len(idx_water)
            nland = len(idx_land)
            nwater = min(nwater, int(nwat * 0.33))
            # ratio = min(max_L2W_ratio, nland // nwat)
            # nland = min(nwater * ratio, len(idx_water))

            ratio = max(nwat / nland, 0.05)
            nland = int((nwater / ratio)) - nwater

            logger.info("Num land after L2W ratio: {}".format(nland))
            logger.info("Num water: {}".format(nwater))

        if max_L2W_ratio:
            nwat = len(idx_water)
            nland = len(idx_land)
            ratio = min(max_L2W_ratio, nland // nwat)
            nland = min(nwater * ratio, len(idx_water))

            logger.info("Num land after L2W ratio: {}".format(nland))

            # take sample of water pix and land pix
        np.random.seed(valseed);
        wat_ix_sampl = np.random.choice(idx_water, nwater)
        np.random.seed(valseed);
        land_ix_sampl = np.random.choice(idx_land, nland)

        data_bands = [b for b in bandnames.DATA_BANDS if b in self.available_bands]

        struct = [(var, np.float32) for var in data_bands]
        struct.append(('water_mask', bool))
        water_sample = np.empty(shape=(nwater,), dtype=struct)
        land_sample = np.empty(shape=(nland,), dtype=struct)

        for band in data_bands:
            band_array = np.array(ds.GetRasterBand(self.available_bands[band]).ReadAsArray()[idx_valid],
                                  dtype=np.float32)
            logger.info(f'Sampling {band} values')
            logger.info(f'max:{np.max(band_array)}, min:{np.min(band_array)}')

            water_sample[band][:, ] = band_array[wat_ix_sampl][:, np.newaxis][:, 0]
            land_sample[band][:, ] = band_array[land_ix_sampl][:, np.newaxis][:, 0]

            logger.info('Done')

            del band_array

        water_sample['water_mask'][:, ] = valid_vals[wat_ix_sampl][:, np.newaxis][:, 0]
        land_sample['water_mask'][:, ] = valid_vals[land_ix_sampl][:, np.newaxis][:, 0]

        return water_sample, land_sample


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
        
    def get_water_pixels(self):
        interpolator = GSWInterpolator(sat_f_name=self.f_path, gsw_dir=self.gsw_path, output_dir=self.output_dir)
        min_lat, min_lon, max_lat, max_lon = self.get_bbox_coords(self.coords)
        logger.info("Bounding box: lat ({}, {}), lon({},{})".format(min_lat, max_lat, min_lon, max_lon))
        return min_lat, min_lon, max_lat, max_lon, interpolator.get_water_presence_for_points(min_lat,max_lat,min_lon,max_lon,self.coords)

