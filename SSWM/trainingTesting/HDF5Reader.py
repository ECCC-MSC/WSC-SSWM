import os
import numpy as np
import h5py
import random

class HDF5Reader:
    # set the data bands that are read by the HDF5Feeder so that all modification
    # data bands include anything that has its own layer in the input image
    # note that this does not include valid pixels (because that's handled elsewhere, but will 
    # include things like incidence angle, slope, windspeed, etc.
    # Basically anything that is returned by get_pol_data that isn't the classification.
    # The order of this list matters because it determines the tuple that is sent to the ANN
    data_bands = ['value']

    def __init__(self, file_path):
        self.file_path = file_path
        self.group_basename = os.path.splitext(os.path.basename(file_path))[0].replace('_eval','').replace('_test','')
        self.water_presence = None
        self.land_presence = None
        self.water_pixels = {}
        self.land_pixels_start_index = 0
        self.pol_one_hot = {}
    
    def close(self):
        self.hfh.flush()
        self.hfh.close()
        
    def open(self,file_path=None):
        if not file_path:
            file_path = self.file_path
        self.hfh = h5py.File(file_path,'r')
        
    def get_beam_mode(self,group=None):
        if not group:
            group = self.group_basename
        self.beam_mode = np.array(self.hfh[group].attrs['beam mode'],dtype=np.int8)
        return self.beam_mode
    
    def get_classified_pixels_idx(self,group=None):
        if not group:
            group = self.group_basename
        self.water_presence_mask = np.array(self.hfh['/{}/water_mask'.format(group)],dtype=bool)
        #Get the indexes and shuffle them:
        self.water_presence = np.nonzero(self.water_presence_mask)
        np.random.shuffle(self.water_presence[0])
        self.land_presence = np.nonzero(~self.water_presence_mask)
        np.random.shuffle(self.land_presence[0])
        self.nb_water_pixels = self.water_presence[0].shape[0]
        
    def get_h5_data(self,group,dataset,idx=None):
        return self.hfh[group][dataset][:][idx][:,np.newaxis]
        
        
    def get_pol_data(self,group,pol,nb_land_pixels):
        self.nb_array_rows = nb_land_pixels+self.nb_water_pixels
        # incidence_angle,value,classification
        # sar_array = np.zeros((self.nb_array_rows,1),dtype=[('incidence_angle', '<f4'), ('value', '<f4'), ('classification', 'i1')]) # old version. saved for reference. 
        sar_array = np.zeros((self.nb_array_rows,1),dtype=[('value', '<f4'), ('classification', 'i1')]) 
        #These will always be returned (the water pixels):
        # sar_array[:self.nb_water_pixels]['incidence_angle'] = self.water_pixels.setdefault('incidence_angle',self.get_h5_data(group,'incidence_angle',self.water_presence)) # old version
        sar_array[:self.nb_water_pixels]['value'] = self.water_pixels.setdefault(pol,self.get_h5_data(group,pol,self.water_presence))
        sar_array[:self.nb_water_pixels]['classification'] = True
        
        # Pick random land pixels (not all of them)
        land_pixels_idx = self.land_presence[0][self.land_pixels_start_index:self.land_pixels_start_index+nb_land_pixels]
        
        # sar_array[self.nb_water_pixels:]['incidence_angle'] = self.get_h5_data(group,'incidence_angle',land_pixels_idx) # old version
        sar_array[self.nb_water_pixels:]['value'] = self.get_h5_data(group,pol,land_pixels_idx)
        sar_array[self.nb_water_pixels:]['classification'] = False
        self.land_pixels_start_index+=nb_land_pixels
        
        return sar_array  
        
    def pick_random_data(self,group=None,min_water_ratio=15,max_water_ratio=60):
        """
        *Parameters* 
        	group: name of the hdf group where the data is
        	min_water_ratio: minimum water/total pixels ratio in percent
        	max_water_ratio: maximum water/total pixels ratio in percent 
        
        *Returns*
        
        
        """
        self.open()
        if not group:
            group = self.group_basename
        if self.water_presence is None:
            self.get_classified_pixels_idx(group)
        ratio = random.randint(min_water_ratio,max_water_ratio)/100.
        print('Ratio: {}'.format(ratio))
        total_pixels = int(self.nb_water_pixels/ratio)
        print('Total pixels in file random sample: {} for each polarizations available'.format(total_pixels))
        nb_land_pixels = total_pixels-self.nb_water_pixels

        sar_data=None
        pol_data=None
        for pol in ('HH','VV','HV','VH'):
            if pol in self.hfh[group].keys():
                #beam mode one hot, pol one hot , incidence angle , value , class
                pol_one_hot=np.array(self.hfh[group][pol].attrs['one_hot'],dtype=np.int8)
                if sar_data is not None:
                    s_data = self.get_pol_data(group,pol,nb_land_pixels)
                    p_data = np.tile(pol_one_hot,(s_data.shape[0],1))
                    pol_data = np.vstack((pol_data,p_data))
                    sar_data = np.vstack((sar_data,s_data))
                else:
                    sar_data = self.get_pol_data(group,pol,nb_land_pixels)
                    pol_data = np.tile(pol_one_hot,(sar_data.shape[0],1))
        return self.get_beam_mode(), pol_data, sar_data.view(np.recarray)
