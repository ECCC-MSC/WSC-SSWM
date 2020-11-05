from pyproj import Proj, transform
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SRIDConverter:
    """ convert coordinates from one coordinate reference system to another using
         a spatial reference identifier (SRID) """
         
    @classmethod
    def convert_from_coordinates(cls, x, y, src_srid, dest_srid=4326):
        src = Proj(init='epsg:{:}'.format(src_srid))
        dest = Proj(init='epsg:{:}'.format(dest_srid))
        new_x, new_y = transform(src, dest, x, y)
        new_x = new_x[:, np.newaxis]
        new_y = new_y[:, np.newaxis]
        return new_x, new_y
    
    @classmethod
    def convert_from_coordinates_check_geo(cls, coordinates, src_srs, dest_srid=4326):
        """

        *Returns*
        
        A tuple (x, y) consisting of x and y coordinates
        """
        logger.info(f'converting from epsg:{src_srs} to epsg:{dest_srid}')
        src=None
        
        try:
            src = Proj(init='epsg:{:}'.format(src_srs))
        except:
            src = Proj(src_srs) 
        dest = Proj(init='epsg:{:}'.format(dest_srid))
        x,y = coordinates[:,0],coordinates[:,1]
        new_x, new_y = transform(src, dest, x, y)
        return new_x, new_y
