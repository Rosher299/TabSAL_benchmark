import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .scripts.train import tddpm_train
from .scripts.sample import tddpm_sample
from dataset import Dataset_core
from example.utils import set_seed, result_analysis
import pandas as pd
from baesline_core import Baseline_core


seeds = [47,929,5,19898,88]
d_layers={'Adult':[256, 1024, 1024, 1024, 1024, 256],
          "Insurance":[256, 1024, 1024, 1024, 1024, 256],
          "California":[256, 1024, 1024, 1024, 1024, 256],
          "Covertype":[256, 1024, 1024, 1024, 1024, 256],
          "Loan":[256, 1024, 1024, 1024, 1024, 256],
          "Buddy":[256,256],
          "Abalone":[256,128],
          "Diabetes":[256,256],}
          
class Tabddpm(Baseline_core):
    def train(self, args) -> None:
        """
        Train the Tabddpm model.

        Args:
            args (argparse.Namespace): Objects containing the required parameters.

        Returns:
            None
        """
        data : Dataset_core = args.dataset
        steps = args.epoch
        data_dir = args.data_dir

        # Ensure the data directory exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)  
        model_params = {'num_classes': data.num_classes, 'is_y_cond':True, 
                        'rtdl_params':{'d_layers': d_layers[data.__class__.__name__],
                                    'dropout' : 0.0}}
        tddpm_train(data_dir, data, model_params,steps=steps, seed=199)

    def sample(self, args) -> None:
        """
        Sample data using the specified arguments.

        Args:
            args (argparse.Namespace): Objects containing the required parameters.

        Returns:
            None
        """
        data : Dataset_core = args.dataset
        data_dir = args.data_dir

        # Create sample directory if it doesn't exist
        if not os.path.exists(os.path.join(data_dir, 'sample')):
            os.mkdir(os.path.join(data_dir, 'sample'))
        nums = args.sample_nums
        model_params = {'num_classes': data.num_classes, 'is_y_cond':True, 
                        'rtdl_params':{'d_layers': d_layers[data.__class__.__name__],
                                    'dropout' : 0.0}}
        result = []
        # Iterate over seeds for sampling
        for seed in seeds:
            df = self._sample(args, seed, nums)
            df.to_csv(os.path.join(data_dir, 'sample/sample{}_seed{}.csv'.format(nums, seed)), index=False)
           
            # Evaluate the sampled data using MLE evaluation method
            result_dict = data.mle_evluation(df)
            result_dict['seed'] = seed
            result_dict['num'] = nums
            result_dict['method'] = 'tabddpm'
            result.append(result_dict)

        result = pd.DataFrame(result)
        print(result)
        analysis = result_analysis(result, data_dir)
        print(analysis)
    
    def _sample(self, args, seed: int, nums: int) -> pd.DataFrame:
        """
        Generate a specified number of sample data from the Tabddpm model.

        Args:
            args (argparse.Namespace): Objects containing the required parameters.
            seed (int): random seed.
            nums (int): Number of generated samples.

        Returns:
            pd.DataFrame: Generated sample data.
        """
        data : Dataset_core = args.dataset
        data_dir = args.data_dir
        model_params = {'num_classes': data.num_classes, 'is_y_cond':True, 
                        'rtdl_params':{'d_layers': d_layers[data.__class__.__name__],
                                    'dropout' : 0.0}}
        df = tddpm_sample(data_dir, data, num_samples=nums, model_params=model_params, seed=seed)
        df = df.iloc[:nums, :]
        return df