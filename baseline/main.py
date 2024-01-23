import sys 
import os
sys.path.insert(0,os.getcwd())
from example.utils import parse_args, set_seed
from baesline_core import Baseline_core
if __name__ == "__main__":
    args = parse_args()
    baseline_core : Baseline_core
    if args.model == 'Tabula':
        from baseline.tabula import Tabula
        baseline_core = Tabula()
    elif args.model == 'REaLTabFormer':
        from baseline.realtab import Realtab
        baseline_core = Realtab()
    elif args.model == 'GReaT':
        from baseline.great import Great
        baseline_core = Great()
    elif args.model == 'CTGAN':
        from baseline.ct import Ct
        baseline_core = Ct()
    elif args.model == 'TVAE':
        from baseline.tvae import Tvae
        baseline_core = Tvae()
    elif args.model == 'TabDDPM':
        from baseline.tabddpm import Tabddpm
        baseline_core = Tabddpm()

    if args.train_or_sample == 'train':
        baseline_core.train(args)
    elif args.train_or_sample == 'sample':
        baseline_core.sample(args)
    elif args.train_or_sample == 'distance':
        baseline_core.distance(args)
    elif args.train_or_sample == 'ks_tv':
        baseline_core.ks_tv(args)
    elif args.train_or_sample == 'predict':
        baseline_core.predict(args)