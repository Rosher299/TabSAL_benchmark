import sys 
import os
sys.path.insert(0,os.getcwd())
from ctgan import TVAE
from dataset import Dataset_core
from example.utils import set_seed, result_analysis
import pandas as pd
from baesline_core import Baseline_core


seeds = [0, 1, 2, 3, 4]

class Tvae(Baseline_core):
    def train(self, args) -> None:
        """
        Train the Tave model.

        Args:
            args (argparse.Namespace): Objects containing the required parameters.

        Returns:
            None
        """
        data : Dataset_core = args.dataset
        epoch = args.epoch
        data_dir = args.data_dir

        # Ensure the data directory exists
        if  not os.path.exists(data_dir):
            os.makedirs(data_dir)

        model = TVAE(epochs=epoch)
        df = data.get_train_frame()
        model.fit(df, data.category_column)

        # Save the model
        model.save(os.path.join(data_dir, 'model.pkl'))

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
        nums = args.sample_nums

        # Create sample directory if it doesn't exist
        if not os.path.exists(os.path.join(data_dir, 'sample')):
            os.mkdir(os.path.join(data_dir, 'sample'))

        # Load Tave model
        model = TVAE()
        model = model.load(os.path.join(data_dir, 'model.pkl'))

        result = []
        # Iterate over seeds for sampling
        for seed in seeds:
            set_seed(seed)

            # Generate samples using Tave model
            df = model.sample(nums)
            df.to_csv(os.path.join(data_dir, 'sample/sample{}_seed{}.csv'.format(nums, seed)), index=False)
            
            # Evaluate the sampled data using MLE evaluation method
            result_dict = data.mle_evluation(df)
            result_dict['seed'] = seed
            result_dict['num'] = nums
            result_dict['method'] = 'tvae'
            result.append(result_dict)

        result = pd.DataFrame(result)
        print(result)
        analysis = result_analysis(result, data_dir)
        print(analysis)
    
    def _sample(self, args, seed: int, nums: int) -> pd.DataFrame:
        """
        Generate a specified number of sample data from the Tave model.

        Args:
            args (argparse.Namespace): Objects containing the required parameters.
            seed (int): random seed.
            nums (int): Number of generated samples.

        Returns:
            pd.DataFrame: Generated sample data.
        """
        data_dir = args.data_dir
        set_seed(seed)

        # Load the model
        model = TVAE()
        model = model.load(os.path.join(data_dir, 'model.pkl'))
        df = model.sample(nums)
        df = df.iloc[:nums, :]
        return df
