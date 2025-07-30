""" 
This file contains functions to download and mosaic DEM tiles from a 
variety of providers
"""

from osgeo import gdal
import itertools
from osgeo import osr
import os
import re
import tarfile
import zipfile

import requests
import netrc

import numpy as np
import urllib.request

from os import path, remove, listdir, makedirs
from SSWM.preprocess.NTS import nts


def get_tile_path_CDED(NTS):
    """ Get FTP path for a CDED NTS tile 
    
    *Parameters*
    
    NTS : str
        Name of NTS sheet for which a DEM is desired
    
    *Returns*
    
    str
        FTP location for CDED DEM tile
    
    Example
    -------
    get_tile_path_CDED("079D01")
    get_tile_path_CDED("079D")
    """
    
    NTS = NTS.lower()
    
    # test resolution from NTS specification
    if len(NTS) == 6:
        resolution = "50k_dem" 
    elif len(NTS) == 4:
        resolution = "250k_dem" 
    else:
        raise Exception("Invalid NTS sheet!")
    
    # base path for all files
    basepath =  "http://ftp.geogratis.gc.ca/pub/nrcan_rncan/archive/elevation/geobase_cded_dnec"
    
    # build ftp path
    tile = NTS[0:3]
    ftp_path = "{}/{}/{}/{}".format(basepath, resolution, tile, NTS)
    ftp_path = ftp_path + ".zip"
    
    return(ftp_path)

        
def SRTM_tile_name(lon, lat):
    """ Build name of SRTM DEM file
    
    *Parameters*
    
    lon : int
        Longitude (whole number) for corner of DEM tile. Negative numbers correspond to W
    lat : int
        Lattiude (whole number) for corner of DEM tile. Negative numbers correspond to 

    *Returns*
    
    str
        File name of DEM tile for SRTM
        
    *Example*
    
    SRTM_tile_name(-110, 49)
    
    """
    
    EW = "w" if lon < 0 else "e"
    NS = "s" if lat < 0 else "n"
    lon = "{:03d}".format(np.abs(lon))
    lat = "{:02d}".format(np.abs(lat))
    name = "NASADEM_HGT_{}{}{}{}.zip".format(NS, lat, EW, lon)
 
    return(name)
    
def get_tile_path_SRTM(lon=None, lat=None, name=None):
    """
    get_tile_path_SRTM(-110, 49)
    """
    baseurl = "https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11/"
    if name:
        url = baseurl + name
    else:
        name = SRTM_tile_name(lon, lat)
        url = baseurl + name
    return url


class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'
    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

   # Overrides from the library to keep headers when redirected to or from
   # the NASA auth host.

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and \
                redirect_parsed.hostname != self.AUTH_HOST and \
                original_parsed.hostname != self.AUTH_HOST:

                del headers['Authorization']
        return
        
            
def downloadSRTM(url, destfile, username=None, password=None, retry=5):
    """
    Downloads an SRTM tile.  
    
    *Parameters*
    
    url : str
        path to SRTM tile
    username : str (optional)
        USGS Earthdata username. If missing, looks for the existence of a .netrc
        file in your home directory 
    password : str (optional)
        USGS Earthdata username. If missing, looks for the existence of a .netrc
        file in your home directory 
    retry : int
        how many times to retry downloading
    If username / password are not provided, the function requires a netrc
     file in your home directory (named either '_netrc' or '.netrc')
    with the following contents:
    machine <hostname>
    login <login>
    password <password>
    """
    
    if not (username and password):
        try:
            auth = netrc.netrc()
        except OSError:
            auth = netrc.netrc('_netrc')
        username, account, password = auth.authenticators("urs.earthdata.nasa.gov")

    session = SessionWithHeaderRedirection(username, password)

    while retry:
        try:
            # submit the request using the session
            response = session.get(url, stream=True)
            print(response.status_code)
            # raise an exception in case of http errors
            response.raise_for_status()  
        
            # save the file
            with open(destfile, 'wb') as fd:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    fd.write(chunk)
            break
            
        except requests.exceptions.HTTPError as e:
            retry -= 1
            print(e)
        
        except requests.exceptions.ConnectionError as e:
            retry -= 1
            print(e)
            
def download_and_unzip_SRTM(url, destfile, exdir, rmzip=True):
    downloadSRTM(url, destfile)
    with zipfile.ZipFile(destfile, "r") as zipf:
        zipf.extractall(exdir)
    if rmzip:
        os.remove(destfile)
    return(True)
    
def get_tile_path_CDEM():
    raise NotImplementedError()
    
def download_single_DEM(DEM_id, DEM_dir, replace=False, product="CDED"):
    """ Download a DEM tile 
    
    *Parameters*
    
    DEM_id : str
        Name or NTS sheet of tile to download. If product is "SRTM", a name should
        be specified, but if product is "CDED", then a NTS sheet should be.
    DEM_dir : str 
        Path to which files are downloaded
    replace : boolean
        Whether or not existing files should be re-downloaded and overwritten
    product : str
        Which DEM tile series should be downloaded: ('SRTM', 'CDED')
        
    *Returns*
    
    list
        List of file paths to DEM files. There may be more than one 
        file per single zipped tile.
        
    """
    output = True

    if product.upper() == "CDED":
        ftp_path = get_tile_path_CDED(NTS = DEM_id)
    elif product.upper() == "SRTM":
        ftp_path = get_tile_path_SRTM(name = DEM_id)
    else:
        raise NotImplementedError("DEM product not implemented")
    
    # create appropriate file name / directory structure based on ftp path
    file_paths = ftp_path.split('/')[-3:]
    
    save_dir = path.join(DEM_dir, *file_paths[0:2])
    if not path.isdir(save_dir):
        makedirs(save_dir)
    
    destfile = path.join(DEM_dir, *file_paths)
    
    if destfile.endswith("tar.gz"):
        dest_dir = re.sub("\\.tar\\.gz", "", destfile)
    else:
        dest_dir = os.path.splitext(destfile)[0]

    # Check to see if file already exists
    if not replace and path.isdir(dest_dir):
        print("{} exists locally and was not downloaded\n".format(dest_dir))
    else:
        if product.upper() == "SRTM":
            output = download_and_unzip_SRTM(url = ftp_path, destfile = destfile, exdir = dest_dir)
        else:
            output = download_and_unzip(url = ftp_path, destfile = destfile, exdir = dest_dir)
        
    # If an appropriate file was downloaded, return the corresponding file paths
    if output:
        pattern_dict = {"CDED" : "dem[ew_].*[td][ie][fm]$",
                        "CDEM" : "dem[ew_].*[td][ie][fm]$",
                        "CDSM" : ".*_cdsm_final_[ew]\\.tif",
                        "SRTM" : "hgt$"}
    
        pattern = pattern_dict[product]

        dem = [f for f in listdir(dest_dir) if re.search(pattern, f)]
        dem = [path.join(dest_dir, x) for x in dem]
        
        return(dem)
    
def download_and_unzip(url, destfile, exdir, rmzip=True):
    """ 
    Downloads and unzips a file

    *Parameters*
    
    url : str
        Url path
    destfile :  str
        Filepath of output zipfile
    exdir : str 
        The directory to which files are extracted
    rmzip : boolean
        Whether or not to remove zipfile after extraction. 
        
    *Returns*
    
    str
        path(s) to target tiles
    """
    
    try:
        print("Downloading file from {}".format(url))
        urllib.request.urlretrieve(url, destfile)
   
    # if the url doesn't exist    
    except Exception as e:
        print(url)
        print(e)
        return(False)
    
    if destfile.endswith("tar.gz"):
        with tarfile.open(destfile, "r:gz") as tarf:
            tarf.extractall(exdir)
        
    elif destfile.endswith("zip"):
        with zipfile.ZipFile(destfile, "r") as zipf:
            zipf.extractall(exdir)
            
    else:
        rmzip = False
        
    if rmzip:
        remove(destfile)
    
    return(True)
        


def download_multiple_DEM(DEM, DEM_dir, product="CDED"):
    """ Download a list of DEM URLs. If they exist already, they are not downloaded
    
    *Parameters*
   
    DEM : list
        List of DEM urls (SRTM) or NTS tiles (CDED)
    DEM_dir : str 
        Path to which files are downloaded
    product : str
        Which DEM tile series should be downloaded: ('SRTM', 'CDED')
        
    *Returns*
   
    list
        a list of file paths for target DEMs
    """
    # sanity check: make sure NTS names are well-formed
    if product.upper() in ["CDED", "CDEM", "CDSM"]:
        if not all([re.search("^\\d{3}\\w(\\d{2})?$", x) for x in DEM]):
            raise Exception("Bad format for one or more NTS strings")
            
    # download each DEM file using the map function
    get_single = lambda x: download_single_DEM(x, DEM_dir = DEM_dir, product=product)        
    files = map(get_single, DEM)
    
    # return list of files
    files = [f for f in files if f is not None]
    files = [dem for sublist in files for dem in sublist]
    return(files)
        
def create_DEM_mosaic(DEM, DEM_dir, dstfile, product="CDED",
                        vrt_only=False, format="GTiff"):
    """ Create a Mosaic from a list of DEM urls or NTS tiles. Missing tiles will
    be downloaded """
    
    
    files = download_multiple_DEM(DEM, DEM_dir, product)
    
    # build VRT
    VRT_path = path.join(path.dirname(dstfile), "tmp.VRT")
    VRT = gdal.BuildVRT(VRT_path, files, resampleAlg='cubic')
    VRT.FlushCache()
    VRT = None

    
    # return VRT-only if desired
    if vrt_only:
        return(VRT_path)
    
    # set warp parameters

    ds = gdal.Translate(dstfile, VRT_path, format=format, resampleAlg='cubic')

    ds.FlushCache()
    ds = None
    
    remove(VRT_path)
    return(dstfile)
    
def SRTM_tiles_from_extent(ext):
    return degree_tiles_from_extent(ext, SRTM_tile_name, yoff=-1)
    
def degree_tiles_from_extent(ext, tile_function, xoff=0, yoff=0):
    """ Get a list of raster tiles required to cover a spatial extent 
    
    *Parameters*
    
    ext : dict
        Dictionary with the following keys: {xmin, xmax, ymin, ymax} corresponding
        to the spatial extent in WGS84 decimal degrees 
    tile_function : function
        function that takes lon, lat as keyword arguments and returns tile name
    
    *Returns*
    
    list
        List of tile names required to cover specified spatial extent
        
    *Examples*
    
    E = {'xmin': -110 ,'xmax': -108,'ymin': 48 ,'ymax': 51 }
    """
    
    # unpack extent dictionary
    xmin = ext['xmin']
    xmax = ext['xmax']
    ymin = ext['ymin']
    ymax = ext['ymax']
    
    # get all corner coordinates to cover extent
            # +1 beacuse of 0 indexing 
    xrange = range(int(np.floor(xmin)), int(np.floor(xmax)) + 1) 
    yrange = range(int(np.ceil(ymin)), int(np.ceil(ymax)) + 1)
    
    xrange = [x + xoff for x in xrange]
    yrange = [y + yoff for y in yrange]
    
    pts = itertools.product(xrange, yrange)
  
    # get tile index for all 
    f = lambda x: tile_function(lon = x[0], lat = x[1])
    tiles =  [pth for pth in map(f, pts)]
    
    return(tiles)
    
def NTS_tiles_from_extent(ext, scale=1):
    ''' Determine which NTS tiles are required to cover a target spatial extent 
    
    *Parameters*
    
    ext : dict
        Dictionary with the following keys: {xmin, xmax, ymin, ymax} corresponding
        to the spatial extent in WGS84 decimal degrees 
    scale : int
        
        
    *Examples*
    
        ext = {'ymin': 52, 'ymax': 53, 'xmin' : -114, 'xmax' : -112}
        NTS_tiles_from_extent(ext)
    '''
    # unpack extent dictionary
    w = ext['xmin']
    e = ext['xmax']
    s = ext['ymin']
    n = ext['ymax']
    
    # find NTS tiles
    bbox = nts.makebbox(n=n, e=e, s=s, w=w)
    tiles = nts.bybbox(bbox, scale)
    
    # convert to list of strings
    tile_list = [''.join(tile) for tile in tiles]
    
    return(tile_list)

def get_spatial_extent(raster_path, target_EPSG = 4326, tol=0.5):
    """ Get the spatial extent of a raster file. 
    
    If the file is not georeferenced (e.g. for raw radarsat 2), this function 
    attempts to use the GCPs in the image . Howver, this doesn't always
    produce exact results, so it is advisable to use an extra buffer tolerance
    in your spatial extent (maybe ~0.1 decimal degrees) 
    
    *Parameters*
    
    raster_path : str
        Path to raster for which a spatial extent is desired
    target_EPSG : int
        EPSG code specifying coordinate system for output file
    tol : float 
        By how many decimal degrees to buffer spatial extent
        
    *Returns*
    
    dict
        Dictionary with the following keys: {xmin, xmax, ymin, ymax} corresponding
        to the spatial extent in WGS84 decimal degrees 
    
    """

    # open file
    src = gdal.Open(raster_path)
    prj = src.GetProjection()
    
    # check if georeferencing information is available
    if (sum(src.GetGeoTransform()) == 2) and src.GetGCPCount() > 3:
        gt = gdal.GCPsToGeoTransform(src.GetGCPs())
        prj = src.GetGCPProjection()
    else:
        gt = src.GetGeoTransform()
    
    # get untransformed upper left and lower right corner coordinates
    ulx, xres, xskew, uly, yskew, yres  = gt
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    
    
    # Setup the source projection - you can also import from epsg, proj4...
    source = osr.SpatialReference()
    source.ImportFromWkt(prj)

    # The target projection
    target = osr.SpatialReference()
    target.ImportFromEPSG(target_EPSG)
    
    # set up transform
    transform = osr.CoordinateTransformation(source, target)

    # transform coordinates
    upper = transform.TransformPoint(ulx, uly)
    lower = transform.TransformPoint(lrx, lry)
    
    ext = {'xmin' : min(upper[0], lower[0]) - tol,
           'xmax' : max(upper[0], lower[0]) + tol,
           'ymin' : min(upper[1], lower[1]) - tol,
           'ymax' : max(upper[1], lower[1]) + tol}
    

    return(ext)
    
def create_DEM_mosaic_from_extent(ext, dstfile, DEM_dir, product="CDED", 
                                    vrt_only=False):
    """ Generate DEM mosaic covering a extent
    
    *Parameters*
    
    ext : dict
        Dictionary with the following keys: {xmin, xmax, ymin, ymax} corresponding
        to the spatial extent in WGS84 decimal degrees 
    dstfile : str
        Path to file to create
    DEM_dir : str
        Directory where DEM files are saved
    product : str
        DEM source to use. One of ("SRTM", "CDED").
    vrt_only : boolean  
        Whether to create a VRT as an output file or 
  
        
    *Returns*
    
    str
        Path to mosaicked DEM
        
    *Examples*
    
    from os import path
    home = path.expanduser('~')
    ext = {'ymin': 52, 'ymax': 53, 'xmin' : -114, 'xmax' : -112}
    create_DEM_mosaic_from_extent(ext, 
                                  dstfile = path.join(home, 'mosaic.tif'),
                                  DEM_dir = path.join(home, 'DEM'),
                                  product = "CDED",
                                  vrt_only = False)
    """ 
    
    #
    if product.upper() == "SRTM":
        tiles = SRTM_tiles_from_extent(ext)
    elif product.upper() == "CDED":
        tiles = NTS_tiles_from_extent(ext)
    else:
        raise NotImplementedError
    
    # create mosaic
    mosaic = create_DEM_mosaic(DEM = tiles, DEM_dir=DEM_dir, dstfile=dstfile, 
                                product=product, vrt_only=vrt_only)
    
    return(mosaic)
    
    
def gdalslope(DEM, dst, latlon = True):
    if latlon:
        scale = 111120
    else:
        scale  = 1
    gdal.DEMProcessing(dst, DEM, "slope", scale=scale) 
    
def gdalTPI(DEM, dst, latlon = True):
    gdal.DEMProcessing(dst, DEM, "TPI") 

def egm96_to_wgs84_heights(dem, geoid):
    
    """
    Convert heights above the EGM96 geoid to heights above the WGS84 ellipsoid,
    for example, as required to use the RADARSAT-2 rational function model.        
    """
    
    dem_ds = gdal.Open(dem, gdal.GA_Update)
    # Load the EGM96 dataset (DEM ds is passed as an argument)
    #egm96_ds = gdal.Open(os.path.join(os.path.dirname(ag.__file__), "etc", "WW15MGH_copy.tif"))  # File is located in the software directory
    egm96_ds = gdal.Open(geoid)
    
    # Setup an in-memory raster using the DEM as a template
    driver = gdal.GetDriverByName("MEM")
    resamp_ds = driver.Create("", dem_ds.RasterXSize, dem_ds.RasterYSize, 1, gdal.GDT_Float32)
    resamp_ds.SetProjection(dem_ds.GetProjection())
    resamp_ds.SetGeoTransform(dem_ds.GetGeoTransform())

    # Reproject / resample the EGM96 data to the DEM resolution / space.
    gdal.ReprojectImage(egm96_ds, resamp_ds, egm96_ds.GetProjection(), dem_ds.GetProjection(), gdal.GRA_Cubic)

    # Add (subtract) the difference between EGM96 and WGS84 and return the resulting DEM.
    egm96 = resamp_ds.ReadAsArray()
    dem = dem_ds.ReadAsArray().astype(np.float32)
    dem[dem != -32768] += egm96[dem != -32768]

    # re-write DEM
    dem_ds.GetRasterBand(1).WriteArray(dem)
    del dem_ds
    
