from osgeo import gdal
import os
import subprocess
    
def orthorectify_dem_rpc(input, output, DEM, dtype=None):
    """ Orthorectify raster using rational polynomial coefficients and a DEM

    *Parameters*
    
    input : str
        Path to image to orthorectify
    output : str
        Path to output image
    DEM : str
        Path to DEM 
    dtype : int
        GDAL data type for output image (UInt16=2, Float32=6 etc.)
    
    *Returns*
    
    boolean
        True if it completes sucessfully
    """
    
    if dtype is None:
        dtype =  max([input.GetRasterBand(i + 1).DataType for i in range(input.RasterCount)])
      
    # set warp options
    optns = gdal.WarpOptions(
                transformerOptions = ["RPC_DEM={}".format(DEM)],
                creationOptions = ["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"],
                rpc = True,
                multithread=True,
                outputType=dtype,
                resampleAlg='cubic')
    
    # run warp
    gdal.Warp(output, input, options=optns)
    
    return(True)


def orthorectify_otb(input, output, DEMFolder, gridspacingx, ram=1000):
    """ Orthorectify raster using orfeotoolbox Ortho

    Parameters
    ----------
    input : str
        Path to image to orthorectify
    output : str
        Path to output image
    DEM : str
        Path to DEM
    gridspacing : float
        pixel size of deformation grid used for the ortho

    Returns
    -------
    """

    command = '''otbcli_OrthoRectification -io.in {} -io.out {} -map wgs -elev.dem {} -interpolator linear -opt.ram {} -opt.gridspacing {}'''.format(
        str(input), str(output), str(DEMFolder), str(ram), str(gridspacingx))

    print(command)
    '''
    print(command)

    p = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, shell=True)  # text=True decodes output to string

    while True:
        line = p.stdout.readline()
        if not line:
            break  # No more output, subprocess finished

        # Process the line to extract and print progress
        print(f"Subprocess output: {line.strip()}")

    p.wait()  # Wait for the subprocess to fully terminate
    '''
    ok = os.system(command)
    #print("Command result: {}".format(p.returncode))
    print(ok)
    

