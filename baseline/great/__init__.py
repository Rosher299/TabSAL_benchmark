import sys 
import os
sys.path.insert(0,os.getcwd())
from .be_great import GReaT
from dataset import Dataset_core
from example.utils import set_seed, result_analysis
import pandas as pd
from baesline_core import Baseline_core


seeds = [0, 1, 2, 3, 4]

class Great(Baseline_core):
    def train(self,args) -> None:
        """
        Train the Great model.

        Args:
            args (argparse.Namespace): Objects containing the required parameters.
            
        Returns:
            None
        """
        data : Dataset_core = args.dataset
        epoch = args.epoch
        data_dir = args.data_dir
        # Load the modal
        model = GReaT(llm=args.base_model, 
                     batch_size=32, epochs=epoch, experiment_dir=data_dir)
        df = data.get_train_frame()
        model.fit(df, column_names=data.column_names)
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

        # Create sample directory if it doesn't exist
        if not os.path.exists(os.path.join(data_dir, 'sample')):
            os.mkdir(os.path.join(data_dir, 'sample'))

        # Set number of samples and load the model
        nums = args.sample_nums
        model = GReaT(llm=args.base_model, batch_size=32, experiment_dir=data_dir)
        model = model.load_from_dir(data_dir)

        result = []
        # Iterate over seeds for sampling
        for seed in seeds:
            set_seed(seed)

            # Generate samples using model
            df = model.sample(nums, k=128)
            df.to_csv(os.path.join(data_dir, 'sample/sample{}_seed{}.csv'.format(nums, seed)), index=False)
            
            # Evaluate the sampled data using MLE evaluation method
            result_dict = data.mle_evluation(df)
            result_dict['seed'] = seed
            result_dict['num'] = nums
            result_dict['method'] = 'great'
            result.append(result_dict)

        result = pd.DataFrame(result)
        print(result)
        analysis = result_analysis(result, data_dir)
        print(analysis)

    def predict(self, args) -> None:
        """
        Predict the test set and output the prediction results.

        Args:
            args (argparse.Namespace): Objects containing the required parameters.
            
        Returns:
            None
        """
        data : Dataset_core = args.dataset
        df = data.get_test_frame()
        data_dir = args.data_dir

        # Ensure the data directory exists
        if not os.path.exists(os.path.join(data_dir, 'predict')):
            os.mkdir(os.path.join(data_dir, 'predict'))
        
        # Predict the test set
        model = GReaT.load_from_dir(data_dir)
        result = []
        for seed in seeds:
            y_pre = model.predict(df, data.target_name)
            result_dict = data.evaluate(y_pre)
            result_dict['seed'] = seed
            result.append(result_dict)
        result = pd.DataFrame(result)
        result.to_csv(os.path.join(data_dir, 'predict/predict_result.csv'), index=False)
        
        # Calculate mean and variance
        analysis = {}
        analysis['mean']={}
        analysis['std']={}
        for col in result.columns[:-1]:
            analysis['mean'][col] = result[col].mean()
            analysis['std'][col] = result[col].std()
        analysis = pd.DataFrame(analysis)
        analysis.to_csv(os.path.join(data_dir, 'predict_analysis.csv'))
        print(result)
        print(analysis)

    def _sample(self, args, seed: int, nums: int) -> pd.DataFrame:
        """
        Generate a specified number of sample data from the Great model.

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
        model = GReaT(llm=args.base_model, batch_size=32, experiment_dir=data_dir)
        model = model.load_from_dir(data_dir)
        df = model.sample(nums, k=128)
        df = df.iloc[:nums, :]
        return df
