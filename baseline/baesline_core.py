import os
from dataset import Dataset_core
import pandas as pd
class Baseline_core():
    def train(self, args):
        pass
    
    def sample(self, args):
        pass
    
    def distance(self, args):
        data : Dataset_core = args.dataset
        data_dir = args.data_dir
        df = self._sample(args, seed=123, nums=data.get_test_frame().shape[0])
        dcr, nndr = data.distance_evaluation(df)
        df['dcr'] = dcr
        df['nndr'] = nndr
        if not os.path.exists(os.path.join(data_dir, 'private')):
            os.makedirs(os.path.join(data_dir, 'private'))
        df.to_csv(os.path.join(data_dir, 'private/private.csv'), index=False)
        # Write the result to a CSV file
        result = {'dcr_min':df['dcr'].min(), 'dcr_mean':df['dcr'].mean(), 'nndr_mean':df['nndr'].mean(), 'method':self.__class__.__name__}
        result = pd.DataFrame(result, index=[0])
        result.to_csv(os.path.join(data_dir, 'private/private_result.csv'), index=False)
        print(result)
    
    def ks_tv(self, args):
        data : Dataset_core = args.dataset
        data_dir = args.data_dir
        df = self._sample(args, seed=0, nums=data.get_test_frame().shape[0])
        result = data.similarity_evaluation(df)
        result['method'] = data.__class__.__name__
        result = pd.DataFrame(result, index=[0])
        result.to_csv(os.path.join(data_dir, 'ks_tv_result.csv'), index=False)
        print(result)
    def _sample(self, args, seed, nums):
        pass