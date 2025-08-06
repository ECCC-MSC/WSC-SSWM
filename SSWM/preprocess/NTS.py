# Determine NTS tile sheets from coordinates.  
# e.g. 
# ext = nts.getextent(62, -110, 60, -112)
# tiles = nts.bybbox(ext, 1)
#
# Ported from R using code from: github.com/paleolimbot/rcanvec (/R/ntstiles.R)

import numpy as np
import pandas as pd


class nts:
    MAP_SERIES_N_OF_80 = np.array(("910", "780", "560", "340", "120", 
                                    None, "781", "561", "341", "121")).reshape(2,5)
    
    
    MAP_250K = np.array(("D", "C", "B", "A", 
                        "E", "F", "G", "H", 
                        "L", "K", "J", "I",
                        "M", "N", "O", "P")).reshape(4,4)
    
    
    
    MAP_250K_N_OF_68 = np.array(("B", "A", "C", "D", "F", "E", "G", "H")).reshape(4,2)
    
    MAP_50K = np.array(("04", "03", "02", "01", 
                        "05", "06", "07", "08",
                        "12", "11", "10", "09",
                        "13", "14", "15", "16")).reshape(4,4)
                        
    #' A contstant denoting NTS Series scale (0)
    SCALESERIES = 0
    
    #'  A constant denoting NTS Map Area (1:250k) scale (1)
    SCALE250K = 1
    
    #'  A constant denoting NTS Map Sheet (1:50k) scale (2)
    SCALE50K = 2
                        
    def __init__(self):
        pass
    
    @staticmethod
    def indexxy(value, mapp): 
        ind = [x for (x,y) in enumerate(mapp) if y == value]

    @staticmethod
    def makebbox(n, e, s, w):
        '''  
           min max
        x -66 -64
        y  45  46
        '''
        bbox = np.array((w, e, s, n)).reshape((2,2))   
        return(bbox)
    
    
    @staticmethod
    def widthandoffset250(lat): 
        if (lat >= 80.0):
            wo = np.array((8.0, 8.0))
        elif (lat >= 68.0):
            wo = np.array((4.0, 0.0))
        else:
            wo = np.array((2.0, 0.0))
        return(wo)
        
    @staticmethod
    def widthandoffsetseries(lat):
        if (lat >= 80):
            wo = np.array((16.0, 8.0))
        else:
            wo = np.array((8.0, 0.0))
        return(wo)
     
    def mapsperseries(self, tile250ky):
        if (tile250ky >= 28):
            return(2)
        else:
            return(4)
            
    @staticmethod
    def tileseriesy(lat):
        series = int(np.floor((lat - 40.0) / 4.0))
        return(series)
    
    def tileseriesx(self, lon, lat):
        wo = self.widthandoffsetseries(lat)
        series = int(np.floor((lon + (144.0 - wo[1])) / wo[0]))
        return(series)
       
    def tileseries(self, lon, lat):
        series = (self.tileseriesx(lon, lat), self.tileseriesy(lat))
        return(series)
    
        
    def bboxseries(self, tile):
        minlat = tile[2] * 4.0 + 40.0
        wo = self.widthandoffsetseries(minlat)
        minlon = -144 + tile[0] * wo[0] + wo[1]
        series = self.makebbox(minlat + 4.0, minlon + wo[0], minlat, minlon) #n, e, s, w
        return(series)
        
    def validtileseries(self, tile):
        if (tile[1] == 0):
            if (7 <= tile[0] <= 11):
                return(True)
        
        elif (tile[1] == 1):
            if (6 <= tile[0] <= 11) or (tile[0] == 1) or (tile[0] == 2): 
                return(True)
            
        elif (2 <= tile[1] <= 4): 
            if (0 <= tile[0] <= 11): 
                return(True)
            
        elif (5 <= tile[1] <= 8):
            if(0 <= tile[0] <= 10): 
                return(True)
            
        elif (tile[1] == 9):
            if(0 <= tile[0] <= 9):
                return(True)
            
        elif (tile[1] == 10):
            if(0 <= tile[0] <= 4):
                return(True)
            
        elif(tile[1] == 11): 
            if(1 <= tile[0] <= 4): 
                return(True)
            
        return(False)
    
    
    def idseries(self, tile):
        if (not self.validtileseries(tile)):
            return(None)
        
        if (tile[1] >= 10):
            id = self.MAP_SERIES_N_OF_80[tile[1] - 10, tile[0]]
            return(id)
        else:
            seriesrow = str(tile[1])
            seriescol = str(11 - tile[0])
            if (len(seriescol) == 1):
                seriescol = "0" + seriescol
            id = seriescol + seriesrow
        
        return(id)


    def tileseriesbyid(self, series): 
        if (len(series) >= 2):
            result = self.indexxy(series, self.MAP_SERIES_N_OF_80) #row, col returned
            
            if (len(result) > 0):
                return((result[0] - 1, result[1] + 10 - 1))
            else:
                seriesy = int(series[-1]) #last character
                seriesx = 11 - int(series[0:-1])
                return((seriesx, seriesy))
            
        else:
            return(None)

    def tileseriesfromtile250(self, tile250):
        mapsperseries = self.mapsperseries(tile250[1])
        tilex = int(np.floor(tile250[0] / mapsperseries))
        tiley = int(np.floor(tile250[1] / 4))
        tiles = (tilex, tiley)
        return(tiles)
        
    @staticmethod    
    def tile250y(lat):
        tile = int(np.floor(lat - 40.0))
        return(tile)
        
    def tile250x(self, lon, lat):
        wo = self.widthandoffset250(lat)
        tile = int(np.floor((lon + (144 - wo[1])) / wo[0]))
        return(tile)
            
    def tile250(self, lon, lat):
        tile = [self.tile250x(lon, lat), self.tile250y(lat)]
        return(tile)
        
    def bbox250(self, tile):
        minlat = tile[1] + 40
        wo = self.widthandoffset250(minlat)
        minlon = -144 + wo[1] + (tile[0] * wo[0])
        maxlat = minlat + 1
        maxlon = minlon + wo[1]
        bbox = self.makebbox(maxlat, maxlon, minlat, minlon)
        return(bbox)
        
    def id250(self, tile250):
        tileS = self.tileseriesfromtile250(tile250)
        seriesid = self.idseries(tileS)
        if (seriesid is None):
            return(None)
        
        seriesMinYTile = tileS[1] * 4
        yTileInSeries = tile250[1] - seriesMinYTile
        
        mapsPerSeries = self.mapsperseries(tile250[1])
        seriesMinXTile = tileS[0] * mapsPerSeries
        xTileInSeries = tile250[0] - seriesMinXTile
        
        arealetter = None
        if (tileS[1] >= 7):
            arealetter = self.MAP_250K_N_OF_68[yTileInSeries, xTileInSeries]
        else:
            arealetter = self.MAP_250K[yTileInSeries, xTileInSeries]
        
        id = [seriesid, arealetter]
        return(id)
        
        
    def tile50byid(self, ntsid):
        tile250 = self.tile250byid(ntsid)
        if (len(ntsid) < 3):
            raise Exception("Invalid NTS for 50k tile: {}".format(ntsid))
        result = self.indexxy(ntsid[2], self.MAP_50K)
        
        if (len(result)==0):
            raise Exception("Invalid NTS: {} ({})".format(ntsid, ntsid[3]))
        tiley = tile250[1] * 4 + result[0] - 1
        tilex = tile250[0] * 4 + result[1] - 1
        tile = [tilex, tiley]
        return(tile)
    
    def tile50y(self, lat):
        tile = int(np.floor((lat - 40.0) / 0.25))
        return(tile)
    
    def tile50x(self, lon, lat):
        tile250x = self.tile250x(lon, lat)
        wo = self.widthandoffset250(lat)
        londiff = lon - (-144.0 + wo[1]+(tile250x*wo[0]))
        plustilesx = int(np.floor(4.0*  londiff / wo[0]))
        tile  = 4 * tile250x + plustilesx
        return(tile)
        
    def tile50(self, lon, lat) :
        tile = (self.tile50x(lon, lat), self.tile50y(lat))
        tile = np.array(tile)
        return(tile)
        
        
    def bbox50(self, tile50):
        minlat = 40 + tile50[1] * 0.25
        wo = self.widthandoffset250(minlat)
        wd = wo[0] / 4.0
        minlon = -144.0 + wo[1] + tile50[0] * wd
        bbox = self.makebbox(minlat + 0.25, minlon + wd, minlat, minlon)
        return(bbox)
            
        
    def id50(self, tile50):
        tile250 = (np.floor(tile50 / 4.0)).astype(int)
        id250 = self.id250(tile250)
        if (id250[0] is None):
            return(None)
        
        mintiles  = tile250 * 4
        plustiles = tile50 - mintiles
        sheet = nts.MAP_50K[plustiles[1], plustiles[0]]
        id = id250 + [sheet]
        return(id)
        
    
    def bybboxgeneric(self, bbox, tilefuncx, tilefuncy, idfunc):
        minx = max(bbox[0,0], -144.0)
        miny = max(bbox[1,0], 40.0)
        maxx = min(bbox[0,1], -48.0)
        maxy = min(bbox[1,1], 88.0)

        if ((maxx < minx) or (maxy < miny)):
            raise Exception("Bounds provided may be outside the NTS grid")
        
        containsabove80 = maxy > 80.0
        containsabove68 = containsabove80 or (miny >= 68.0) or (maxy>68.0)
        containsbelow68 = miny < 68.0
        containsbelow80 = miny < 80.0
        
        self.idlist = list()
        
        def ld(n=maxy, e=maxx, s=miny, w=minx):
            mint = (tilefuncx(w, s), tilefuncy(s))
            maxt = (tilefuncx(e, n), tilefuncy(n))
            
            for x in range(mint[0], maxt[0] + 1):
                for y in range(mint[1], maxt[1] + 1):
                    tileid = idfunc(np.array([x,y]))
                    if(np.atleast_1d(tileid)[0] is not None):
                        self.idlist.append(tileid)
                        

        if (containsabove80):
            if (containsbelow80):
                ld(s=80)
            else:
                ld()            
            return(np.array(self.idlist))

        
        tempn = maxy
        temps = miny
        
        if (containsabove68):
            if (containsabove80):
                tempn = 79.99

            if (containsbelow68):
                temps = 68.0

            ld(s=temps, n=tempn)

        
        if (containsbelow68):
            tempn = maxy
            if (containsabove68):
                tempn = 67.99
            
            ld(n=tempn)
            
        return(np.array(self.idlist))
            
            
    def _bybbox(self, bbox, atscale):
        if (atscale == self.SCALESERIES):
            tilexfunc = self.tileseriesx
            tileyfunc = self.tileseriesy
            idfunc    = self.idseries
       
        elif (atscale == self.SCALE250K):
            tilexfunc = self.tile250x
            tileyfunc = self.tile250y
            idfunc    = self.id250
        
        elif (atscale == self.SCALE50K):
            tilexfunc = self.tile50x
            tileyfunc = self.tile50y
            idfunc    = self.id50
        
        else:
            raise Exception("Invalid value for atscale: {}".format(atscale))

        sheets = self.bybboxgeneric(bbox, tilexfunc, tileyfunc, idfunc)
        
        return(sheets)
    
    @classmethod
    def bybbox(cls, bbox, atscale):
        N = cls()
        result = N._bybbox(bbox, atscale)
        return(result)
        
    def _bbox(self, ntsid):
        if len(np.array(ntsid)) > 1:
            out = list()
            for i in ntsid:
                out.append(self.bbox(i))
         
            return(out)
        
        else:
            if (len(ntsid)>=3):
                fun = nts.bbox50
                tilef = nts.tile50byid
            elif (len(ntsid)==2):
                fun = nts.bbox250
                tilef = nts.tile250byid
            elif (len(ntsid)==1):
                fun = nts.bboxseries
                tilef = nts.tileseriesbyid
            else:
                raise Exception("Invalid NTS: ".format(ntsid))
            
            tile = tilef(ntsid)
            bbox = fun(tile)
            
            return(bbox)

    @classmethod
    def bbox(cls, ntsid):
        N = cls()
        result = N._bbox(ntsid)
        return(result)
        
    
def valid_nts_tiles(tilesfile, return_50k = False):
    
    t50k = np.loadtxt(tilesfile, dtype='str')
    
    if return_50k:
        out = t50k
    else:
        t250k = np.unique([t[0:4] for t in t50k])
        out = t250k
    
    return(out)
    





































































