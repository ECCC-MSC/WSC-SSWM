import numpy as np
import os
import random
import time
import logging

from SSWM.trainingTesting.HDF5Reader import HDF5Reader
logger = logging.getLogger()
class HDF5Feeder:
    
    def __init__(self,hdf_directory,global_time=None):
        self.files_path=hdf_directory
        available_files = [f.path for f in os.scandir(hdf_directory) if f.name.lower().endswith('.h5')]
        eval_files = [f.path for f in os.scandir(os.path.join(hdf_directory, 'eval')) if f.name.lower().endswith('.h5')]
        test_files = [f.path for f in os.scandir(os.path.join(hdf_directory, 'test')) if f.name.lower().endswith('.h5')]
        self.source_files = [f for f in available_files if 'eval' not in f.lower() and 'test' not in f.lower()]
        # self.eval_file = [f for f in available_files if 'eval' in f.lower()][0]
        # self.test_file = [f for f in available_files if 'test' in f.lower()][0]
        self.eval_file = [f for f in eval_files][0]
        self.test_file = [f for f in test_files][0]
        logging.info(f'using{self.eval_file} as evaluation data and {self.test_file} as test data')
        self.groups=None
        self.batch=None
        self.data_renew=0
        if global_time is None:
            self.global_time = time.time()
        else:
            self.global_time=global_time
        
    def pick_random_files(self):
        print('Getting data using random files,water/land ratios and pixels')
        print('This will take a while...')
        t=time.time()
        random_files = random.sample(self.source_files,random.randint(3,6))
        #random_files = random.sample(self.source_files,1)
        random_sar_data = []
        print('Data will come from following files:')
        print('\n'.join(random_files))
        for f in random_files:
            print(80*'-')
            print(f'Getting data from {f}')
            print(80*'-')
            reader = HDF5Reader(f)
            random_sar_data.append(reader.pick_random_data())
        print(f'Done getting the random training data [time to process: {time.time()-t}s ; global time = {time.time()-self.global_time}]')
        return random_sar_data
    
    def get_batch_generator(self,yielded=False,water_weight=1.0):
        print(80*'-')
        print('Sending train data')
        print(80*'-')
        random_sar_data = self.pick_random_files()
        while (time.time() - self.global_time) <= 11500:
            source = list(random.choice(random_sar_data))
            samples = random.sample(range(source[1].shape[0]),1000)
            # source[0]: beam_mode <scalar>
            # source[1]: pol one hot , <col>
            # source[2]: sar_data <recarray> 'value','classification'
            # int8,
            source[1]=source[1][samples]
            source[2]=source[2][samples]
            try:
                # We get the relevant data bands from the HDF5Reader class so that it isn't 
                # necessary to change the Feeder class when the neural network changes to accommodate
                # different input files with different bands (e.g. slope, wind, incidence angle)
                databands = HDF5Reader.data_bands
                yield (((source[0],
                         source[1][j]) +
                         tuple(source[2][i][j][0] for i in databands) +
                         (water_weight,), 
                             source[2].classification[j][0]) for j in range(1000))
                yielded = True
            except Exception as exc:
                if yielded and (time.time() - self.global_time) <= 10000:
                    # We have gone through the whole generator and still have time to load another loop
                    del random_sar_data
                    print('Generator exhausted, calling new batches of random train data')
                    random_sar_data = self.pick_random_files()
                else:
                    print('Unknown Exception')
                    print(type(exc))
                    print(exc)
                    return
            if ((time.time() - self.global_time) // 3600) > self.data_renew:
                print('Renewing data samples after 1 hour of training')
                self.data_renew+=1
                del random_sar_data
                random_sar_data = self.pick_random_files()
        return
                
    def get_test_or_eval_data(self,data_file=None,flag='eval'):
        print(80*'-')
        print(f'Sending random {flag} data')
        print(f'Eval file: {self.eval_file}')
        print(80*'-')
        # pick random data from eval file:
        reader = HDF5Reader(data_file)
        source = list(reader.pick_random_data())
        #self.get_beam_mode(),pol_data,sar_data.view(np.recarray)
        source[0] = np.tile(source[0],(source[1].shape[0],1))
        return source
                
    def get_random_eval_data(self):
        return self.get_test_or_eval_data(data_file=self.eval_file,flag='eval')
    
    def get_random_test_data(self):
        return self.get_test_or_eval_data(data_file=self.test_file,flag='test')
