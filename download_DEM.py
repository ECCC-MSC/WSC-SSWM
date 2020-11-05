"""
Downloads DEM tiles for orthorectification. Can either download for just the prairies
or for the entire country. Tiles are downloaded at 1:250k scale to save on space.
"""

from SSWM.preprocess.DEM import NTS_tiles_from_extent, download_multiple_DEM
import argparse
 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='downloads a collection of DEM tiles from CDED.')
    
    parser.add_argument('dir', type=str, default=None, 
                              help='Path to DEM directory')
                              
    parser.add_argument('-a', '--all', action='store_true', 
                              help='Flag to download all of Canada. Otherwise, just the prairies is downloaded')
             
             
    args = parser.parse_args()   
              
    DEM_dir = args.dir
    
    canada = {'ymin': 40, 'ymax': 88, 'xmin' : -144, 'xmax' : -48}
    prairies = {'ymin': 40, 'ymax': 60, 'xmin' : -120, 'xmax' : -95}
    
    ext = canada if args.all else prairies
    print(ext)
    tiles = NTS_tiles_from_extent(ext, scale=1)
    
    download_multiple_DEM(tiles, DEM_dir, 'CDED')