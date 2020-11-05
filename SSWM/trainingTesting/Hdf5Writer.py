import os
import h5py
import numpy as np
import logging
import gc

logger = logging.getLogger(__name__)

class Hdf5Writer:
    hfh=None
    
    def __init__(self,file_path,group_name=None):
        self.file_path = file_path
        if not group_name:
            self.group_basename=os.path.splitext(os.path.basename(file_path))[0]
        else:
            self.group_basename=group_name
    
    def open(self):
        try:
            open(self.file_path)
            self.hfh = h5py.File(self.file_path,'r+')
        except:
            self.hfh = h5py.File(self.file_path, 'w')
            
    def create_group(self,parent=None,group_name=None,desc=''):
        if not parent:
            parent = self.hfh
        if not group_name:
            group_name = self.group_basename 
        return parent.require_group(group_name)
    
    def create_dataset(self,parent=None,dataset_name=None,compression="gzip", compression_opts=2, data=np.array([]), chunks=True):
        parent = self.hfh if not parent else parent
        return parent.create_dataset(dataset_name, compression=compression, compression_opts=compression_opts, data=data, chunks=chunks)
    
    def add_metadata(self,obj,**kwargs):
        for k,v in kwargs.items():
            obj.attrs[k] = v
    
    def add_rs2_coords(self, rs2_file_name, coords, min_lat, max_lat, min_lon, max_lon, shape=None):
        # Group is the file's (base) name
        group_name = os.path.splitext(os.path.basename(rs2_file_name))[0]
        group = self.create_group(group_name=group_name)
        
        num_coords = coords.shape[0]
        chunksize=(1000000, 2)
        logger.info(f'Writing coordinates with shape {coords.shape} to file')
        ds = self.create_dataset(parent=group, dataset_name='coordinates', data=coords, chunks=chunksize)
        logger.info('Done')
        attribs = {'min_lat': min_lat,
                   'max_lat': max_lat,
                   'min_lon': min_lon,
                   'max_lon': max_lon,
                   'grid_shape': shape}
                   
        self.add_metadata(ds, **attribs)
        logger.info('Metadata written')
        return group,ds
    
    def add_mask(self, rs2_file_name,mask):
        group_name=os.path.splitext(os.path.basename(rs2_file_name))[0]
        group = self.create_group(group_name=group_name)
        ds = self.create_dataset(parent=group, dataset_name='mask', data=mask)
        ds.attrs['nb valid pixels']=np.count_nonzero(mask)
        del group
    
    def add_water(self, rs2_file_name, water_mask):
        group_name=os.path.splitext(os.path.basename(rs2_file_name))[0]
        group = self.create_group(group_name=group_name)
        ds = self.create_dataset(parent=group, dataset_name='water_mask', data=water_mask)
        del group
        
    def add_incidence_angle(self,rs2_file_name,incidence_angle,group=None):
        self.open()
        if not group:
            group_name = os.path.splitext(os.path.basename(rs2_file_name))[0]
            group = self.hfh.require_group(group_name)
        self.create_dataset(parent=group, dataset_name='incidence_angle',data=incidence_angle)
        del group
        
    def add_beam_mode(self,rs2_file_name,beam_mode):
        group_name=os.path.splitext(os.path.basename(rs2_file_name))[0]
        group = self.create_group(group_name=group_name)
        group.attrs['beam mode']=beam_mode
        del group
        
    def add_pol(self, rs2_file_name, pol, pol_array, pol_one_hot):
        logger.info(f'writing data with shape {pol_array.shape} to file')
        group_name = os.path.splitext(os.path.basename(rs2_file_name))[0]
        group = self.create_group(group_name=group_name)
        ds = self.create_dataset(parent=group, dataset_name=pol, data=pol_array, chunks=(1000000,))
        ds.attrs['one_hot'] = pol_one_hot
        del group
        
    def add_generic(self, rs2_file_name, dataset_name, dataset):
        logger.info(f'writing data with shape {dataset.shape} to file')
        group_name = os.path.splitext(os.path.basename(rs2_file_name))[0]
        group = self.create_group(group_name=group_name)
        ds = self.create_dataset(parent=group, dataset_name=dataset_name, data=dataset, chunks=(1000000,))
        del group
        