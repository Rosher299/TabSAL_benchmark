import sys 
import os
sys.path.insert(0,os.getcwd())
from ctgan import CTGAN
from dataset import Dataset_core
from example.utils import set_seed, result_analysis
import pandas as pd
from baesline_core import Baseline_core


seeds = [0, 1, 2, 3, 4]

class Ct(Baseline_core):
    def train(self, args) -> None:
        """
        Train the CTGAN model.

        Args:
            args (argparse.Namespace): Objects containing the required parameters.

        Returns:
            None
        """
        data: Dataset_core = args.dataset
        epoch: int = args.epoch
        data_dir: str = args.data_dir

        # Ensure the data directory exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        model = CTGAN(epochs=epoch)
        df = data.get_train_frame()
        model.fit(df, data.category_column)

        # Save the model
        model_path = os.path.join(data_dir, 'model.pkl')
        model.save(model_path)
    
    def sample(self, args) -> None:
        """
        Sample data using the specified arguments.

        Args:
            args (argparse.Namespace): Objects containing the required parameters.

        Returns:
            None
        """
        data: Dataset_core = args.dataset
        data_dir = args.data_dir
        
        # Create sample directory if it doesn't exist
        sample_dir = os.path.join(data_dir, 'sample')
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        
        # Set number of samples and load CTGAN model
        nums = args.sample_nums
        model = CTGAN()
        model_path = os.path.join(data_dir, 'model.pkl')
        model = model.load(model_path)
        
        results = []
        # Iterate over seeds for sampling
        for seed in seeds:
            set_seed(seed)
            
            # Generate samples using CTGAN model
            df = model.sample(nums)
            output_file = os.path.join(sample_dir, f'sample{nums}_seed{seed}.csv')
            df.to_csv(output_file, index=False)
            
            # Evaluate the sampled data using MLE evaluation method
            evaluation_result = data.mle_evluation(df)
            evaluation_result['seed'] = seed
            evaluation_result['num'] = nums
            evaluation_result['method'] = 'ctgan'
            results.append(evaluation_result)
        
        results_df = pd.DataFrame(results)
        print(results_df)
        analysis = result_analysis(results_df, data_dir)
        print(analysis)

    def _sample(self, args, seed: int, nums: int) -> pd.DataFrame:
        """
        Generate a specified number of sample data from the CTGAN model.

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
        model = CTGAN()
        model = model.load(os.path.join(data_dir, 'model.pkl'))
        df = model.sample(nums)
        df = df.iloc[:nums, :]
        return df