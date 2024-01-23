import logging
import random
import numpy as np
import torch
import pandas as pd
import argparse
from dataset import Adult, Insurance, Covertype, Loan, California, Dataset_core, Buddy, Abalone, Diabetes
import os

def result_analysis(result:pd.DataFrame, data_dir:str, result_name:str='result', analysis_name:str='result_analysis'):
    # Save results to CSV
    result.to_csv(os.path.join(data_dir, f"{result_name}.csv"))
    # Save the results to CSV to calculate the mean and variance
    analysis = {}
    analysis['mean']={}
    analysis['std']={}
    for col in result.columns[:-3]: 
        analysis['mean'][col] = result[col].mean()
        analysis['std'][col] = result[col].std()
    analysis = pd.DataFrame(analysis)
    analysis.to_csv(os.path.join(data_dir, f"{analysis_name}.csv"))
    return analysis
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_model',
        type=str,
        default="/data/lijiale/.cache/huggingface/hub/models--distilgpt2/snapshots/38cc92ec43315abd5136313225e95acc5986876c",
    ),
    parser.add_argument(
        '--sample_nums',
        type=int,
        default=6000,
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=['Tabula', 'REaLTabFormer','GReaT', 'CTGAN', 'TVAE', 'TabDDPM']
    )
    parser.add_argument(
        '--train_or_sample',
        type=str,
        default='train',
        choices=['train', 'sample', 'distance', 'ks_tv', 'datasize', 'predict']
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='Adult',
        choices=['Adult', 'Insurance', 'Covertype', 'Loan', 'California', 'Buddy', 'Abalone', 'Diabetes']
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/data/lijiale/data/LTG_relabeler/iris_test',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
    )
    args = parser.parse_args()
    # Convert dataset to specific classes
    if args.dataset == 'Adult':
        args.dataset = Adult()
    elif args.dataset == 'Insurance':
        args.dataset = Insurance()
    elif args.dataset == 'Covertype':
        args.dataset = Covertype()
    elif args.dataset == 'Loan':
        args.dataset = Loan()
    elif args.dataset == 'California':
        args.dataset = California()
    elif args.dataset == 'Buddy':
        args.dataset = Buddy()
    elif args.dataset == 'Abalone':
        args.dataset = Abalone()
    elif args.dataset == 'Diabetes':
        args.dataset = Diabetes()

    if args.base_model == 'distilgpt2':
        args.base_model = "/data/lijiale/.cache/huggingface/hub/models--distilgpt2/snapshots/38cc92ec43315abd5136313225e95acc5986876c"
    elif args.base_model == 'gpt2':
        args.base_model = "/data/lijiale/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10"
    elif args.base_model == 'opt1.3b':
        args.base_model = "/data/lijiale/.cache/huggingface/hub/models--facebook--opt-1.3b/snapshots/8c7b10754972749675d22364c25c428b29face51"
    elif args.base_model == 'opt6.7b':
        args.base_model = "/data/lijiale/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0"
    elif args.base_model == 'llama2-7b':
        args.base_model = "/data/lijiale/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852"
    elif args.base_model == 'palmyra':
        args.base_model = "/data/lijiale/.cache/huggingface/hub/models--Writer--palmyra-base/snapshots/a87d3faa529765bd3de2b43b3ef01022388d2d0d"
    elif args.base_model == 'roberta':
        args.base_model = "/data/lijiale/.cache/huggingface/hub/models--deepset--roberta-base-squad2/snapshots/e09df911dd96d8b052d2665dfbb309e9398a9d70"
    elif args.base_model == 'opt350m':
        args.base_model ="/data/lijiale/.cache/huggingface/hub/models--facebook--opt-350m/snapshots/cb32f77e905cccbca1d970436fb0f5e6b58ee3c5"
    return args

def digitization(data:pd.DataFrame):
    """_summary_
    Digitization of dataset attributes
    Parameters
    ----------
    data : pd.DataFrame
        _description_
    """
    data  = data.copy()
    # Traverse the data types of each column in the data
    for col in data.columns:
        # If it is an object type
        if data[col].dtype == 'object':
            # Convert it to category type
            data[col] = pd.Categorical(data[col]).codes
    return data

# Set random number seeds
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CustomFormatter(logging.Formatter):
    grey = "\x1b[39;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def set_logging_level(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)

    return logger
