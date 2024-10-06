# Benchmark of TabSAL

 This is a benchmark for data generation tasks which integrates eight datasets(*"Adult" "Insurance" "Loan" "California" "Covertype" "Buddy" "Abalone" "Diabetes"*) and six data generation methods(*"CTGAN" "GReaT" "REaLTabFormer" "TabDDPM" "Tabula" "TVAE"*). This benchmark evaluates the performance of tabular data synthesis methods with respect to MLE, statistical similarity and privacy-preservation.



## Contents

- [The benchmark component](#The benchmark component)

- [Installation](#Installation)

- [Quick start](#Quick start)

  - [Dataset](#Dataset)
  - [Baseline](#Baseline)
    - [Train](#Train)
    - [Sample](#Sample)
    - [Predict](#Predict)
    - [KS_TV](#KS_TV)
    - [Distance](#Distance)

- [Citation](#Citation)

- [Contact](#Contact)

  

### The benchmark component

The following is a brief explanation of the entire benchmark directory structure and content.

```
filetree 
├── baseline (This directory integrates 6 methods)
│  ├── ct (CTGAN is a tabular data synthesizing method based on a generative adversarial network)
│  ├── great (GReaT is a language model-based method and is used as SOTA for synthesizing tabular data)
│  ├── realtab (REaLTabFormer offers a unified framework for synthesizing tabular data)
│  ├── tabddpm (TabDDPM is a diffusion model that can be universally applied to any tabular dataset)
│  ├── tabula (Tabula is a language model-based method and is used as SOTA for synthesizing tabular data)
│  └── tvae (CTGAN is a tabular data synthesizing method based on a variational autoencoder)
├── dataset (This directory integrates 8 datasets)
│  ├── abalone (Task Type: REGRESSION)
│  ├── adult (Task Type: CLASSIFICATION)
│  ├── buddy (Task Type: CLASSIFICATION)
│  ├── california_housing (Task Type: REGRESSION)
│  ├── covertype (Task Type: CLASSIFICATION)
│  ├── diabetes (Task Type: CLASSIFICATION)
│  ├── insurance (Task Type: REGRESSION)
│  └── loan (Task Type: CLASSIFICATION)
├── example
│  ├── utils.py (Contains some functions used)
├── readme.md
```



### Installation

Run the following commands to set up the environment.

1. Clone the repo and datasets.

```sh
git clone https://github.com/XXX/ltg_benchmark.git
wget -O https://drive.google.com/file/d/11-YItITDYGIB6bGuIxGebf2rS3FgtAml/view?usp=sharing
unzip datasets.zip
```

2. Create a new virtual environment and activate it.

```sh
conda create -n benchmark python=3.9.18
conda activate benchmark
```

3. Install extra dependency.

```sh
pip install -r requirements.txt
```



### Quick start

The following are specific usage instructions. 

#### Dataset

It is worth noting that before using, you need to fill in the directory path where your dataset is stored in. For example, for the Insurance dataset, write the training set path to  **get_train_frame**() in /dataset/abalone. py, write the test set path to **get_test_frame**(). The following is the usage of the Insurance dataset.

```
import os
from dataset import Insurance
import pandas as pd
from utils import set_seed, result_analysis


origin_seeds = [0,45,1245,64,1]
data = Insurance()
df = data.get_train_frame()  # Get the training set content
result = []
for seed in origin_seeds:
    set_seed(seed)
    # The default model is RandomForest, which can pass in different parameters to use other models in mle_evluation(). 
    result_dict = data.mle_evluation(df)
    result_dict['method'] = 'origin'
    result_dict['seed'] = seed
    result.append(result_dict)
result = pd.DataFrame(result)
result = result_analysis(result, "/your_file_path")  # Calculate mean and variance
```

#### Baseline

Taking CTGAN as an example, instantiate it.

```
from example.utils import parse_args
from baseline.baesline_core import Baseline_core
from baseline.ct import Ct

args = parse_args()
baseline_core : Baseline_core
baseline_core = Ct()
```

For different usage purposes, different parameters need to be entered on the command line. There are the following parameters in total：

|     parameter     |                         explanation                          |
| :---------------: | :----------------------------------------------------------: |
|      --model      |                      Method to be used                       |
|     --dataset     |                      Dataset to be used                      |
| --train_or_sample | What functions are implemented, including train, sample, predict, distance and KS_TV. |
|    --data_dir     |                  The path to save the model                  |
|      --epoch      |              The number of training iterations               |
|   --sample_nums   |                 Number of generated samples                  |
|   --base_model    | Large Language Model used when using the GReaT and Tabula methods |
|   --batch_size    |     The number of data selected for one training session     |

##### Train

```
baseline_core.train(args)
```

##### Sample

```
baseline_core.sample(args)
```

##### Predict

```
baseline_core.predict(args)
```

##### KS_TV

Statistical similarity between the original dataset and the synthesized dataset.

```
baseline_core.ks_tv(args)
```

##### Distance

Evaluate the privacy protection ability of the synthesis method by quantifying the distance between the synthesized dataset and the original dataset. 

```
baseline_core.distance(args)
```



### Citation

```
@article{LI2024112438,
title = {TabSAL: Synthesizing Tabular data with Small agent Assisted Language models},
journal = {Knowledge-Based Systems},
volume = {304},
pages = {112438},
year = {2024},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2024.112438},
url = {https://www.sciencedirect.com/science/article/pii/S0950705124010724},
author = {Jiale Li and Run Qian and Yandan Tan and Zhixin Li and Luyu Chen and Sen Liu and Jie Wu and Hongfeng Chai},
keywords = {Data synthesis, Natural language processing, Machine learning, Data generation},
abstract = {Tabular data are widely used in machine-learning tasks because of their prevalence in various fields; however, the potential risks of data breaches in tabular data and privacy protection regulations render such data almost unavailable. Tabular data generation methods alleviate data unavailability by synthesizing privacy-free data, and generating data using language models is a novel innovation. Language models can synthesize high-quality datasets by learning knowledge from nondestructive information and recognizing the semantics of table columns. However, when current language models function as generators, their encoding methods are hindered by complicated decoding processes, and the limited predictive ability of language models restricts their generative capability. To this end, we propose an encoding method based on interactive data structures such as JavaScript Object Notation for converting tabular data. We design TabSAL, which is a pluggable tabular data generation framework with small agent assisted language models, to boost the predictive capability, resulting in high-quality synthetic datasets with a much lower computational resource cost. In addition, a benchmark that integrates eight datasets, three methods, and three assessment directions has been issued, which indicates that TabSAL surpasses the state of the art by up to 60%.}
}
```


