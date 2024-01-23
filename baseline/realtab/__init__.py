import sys 
import os
sys.path.append(os.getcwd())
from realtabformer import REaLTabFormer
from dataset import  Dataset_core
from example.utils import set_seed, result_analysis
import pandas as pd
from baesline_core import Baseline_core


seeds = [60464, 91, 2568, 5253, 12]

class Realtab(Baseline_core):
    def train(self, args) -> None:
        """
        Train the Realtab model.

        Args:
            args (argparse.Namespace): Objects containing the required parameters.

        Returns:
            None
        """
        data : Dataset_core = args.dataset
        epoch = args.epoch
        data_dir = args.data_dir

        # Ensure the data directory exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        set_seed(0)
        model = REaLTabFormer(
            model_type="tabular",
            gradient_accumulation_steps=4,
            logging_steps=100,
            epochs=epoch,
            checkpoints_dir=data_dir,)
        df = data.get_train_frame()
        model.fit(df, n_critic=-1)

        # Save the model
        model.save(data_dir)

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

        # Load the model
        model = REaLTabFormer.load_from_dir(data_dir)

        result = []
        # Iterate over seeds for sampling
        for seed in seeds:
            set_seed(seed)

            # Generate samples using Realtab model
            df = model.sample(nums)
            df.to_csv(os.path.join(data_dir, 'sample/sample{}_seed{}.csv'.format(nums, seed)), index=False)
            
            # Evaluate the sampled data using MLE evaluation method
            result_dict = data.mle_evluation(df)
            result_dict['seed'] = seed
            result_dict['num'] = nums
            result_dict['method'] = 'realtabformer'
            result.append(result_dict)

        result = pd.DataFrame(result)
        result.to_csv(os.path.join(data_dir, 'sample/sample{}_result.csv'.format(nums)), index=False)
        print(result)
        analysis = result_analysis(result, data_dir)
        print(analysis)

    def _sample(self, args, seed: int, nums: int) -> pd.DataFrame:
        """
        Generate a specified number of sample data from the Realtab model.

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
        model = REaLTabFormer.load_from_dir(data_dir)
        df = model.sample(nums)
        df = df.iloc[:nums, :]
        return df