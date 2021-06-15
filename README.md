# Data-to-Text-Hierarchical [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-hierarchical-model-for-data-to-text/data-to-text-generation-on-rotowire)](https://paperswithcode.com/sota/data-to-text-generation-on-rotowire?p=a-hierarchical-model-for-data-to-text)

Code for [A Hierarchical Model for Data-to-Text Generation](https://arxiv.org/abs/1912.10011) (Rebuffel, Soulier, Scoutheeten, Gallinari; ECIR 2020); most of this code is based on [OpenNMT](https://github.com/OpenNMT/OpenNMT-py).

UPDATE 11/03/2021: The original checkpoints used to produce results from the 
paper are officialy lost. However, I still have the actual model outputs, which 
are now included in this repo. Simply `unzip outputs.zip`.

Furthermore, [Radmil Raychev][1] and [Craig Thomson][2] from the University of Aberdeen 
are currently working with this repo, and have agreed to share their checkpoints,
namely [htransformer.tar.gz2][5].  
(Note that this file is not downloadable by command line, still looking for a better
alternative)

Once it's downloaded, simply `tar -xvf htransformer.tar.gz2`.  
You'll find the `data` used to train the model, as well as `*.cfg` files and
`*.pt` checkpoints. Note that the data is from [SportSett][3], which contains some 
additional info (such as day of the week for instance).  
(Also see [Thomson et al.][4] for more info regarding the impact of additional data
on system performances.)

[1]: https://github.com/radmilr
[2]: https://github.com/nlgcat
[3]: https://github.com/nlgcat/sport_sett_basketball
[4]: https://www.aclweb.org/anthology/2020.inlg-1.6/
[5]: https://dl.orangedox.com/hierarchical-transformer-checkpoint

## Requirements

You will need a recent python to use it as is, especially OpenNMT. However I guess files could be tweaked to work with older pythons. Please note that at the time of writting, torch==1.1.0 can be problematic with very recent version of python. I suggest running the code with python=3.6

Full requirements can be found in `requirements.txt`. Note that they are not really all required, it's the full pip freeze of a clean conda virtual env, you can probably make it work with less.

Beyond standard packages included in miniconda, usefull packages are torch==1.1.0 torchtext==0.4 and some others required to make onmt work (PyYAML and configargparse for example). I also use more_itertools to format the dataset.

# Dataset

The dataset used in the paper can be downloaded [here](https://github.com/harvardnlp/boxscore-data). More specifically, you just need to download the [RotoWire dataset](https://github.com/harvardnlp/boxscore-data/blob/master/rotowire.tar.bz2): 

```bash
cd data
wget https://github.com/harvardnlp/boxscore-data/raw/master/rotowire.tar.bz2
tar -xvjf rotowire.tar.bz2
cd ..
```

You'll need to format the dataset so that it can be preprocessed by OpenNMT.

`python data/make-dataset.py --folder data/`

At this stage, your repository should look like this:

```
.
├── onmt/                   	# Most of the heavy-lifting is done by onmt
├── data/   					# Dataset is here    
│	├── rotowire/				# Raw data stored here
├	├── make-dataset.py			# formating script
├	├── train_input.txt
├	├── train_output.txt
│	└── ...
└── ...
```

# Experiments

Before any code run, we build an experiment folder to keep things contained

`python create-experiment.py --name exp-1`

At this stage, your repository should look like this:

```
.
├── onmt		             	# Most of the heavy-lifting is done by onmt
├── experiments 	           	# Experiments are stored here
│	└── exp-1
│	│	├── data
├	│	├── gens
│	│	└── models
├── data						# Dataset is here
└── ...
```

# Preprocessing

Before training models via OpenNMT, you must preprocess the data. I've handled all useful parameters with a config file. Please check it out if you want to tweak things, I have tried to include comments on each command. For futher info you can always check out the OpenNMT [preprocessing doc](http://opennmt.net/OpenNMT-py/options/preprocess.html)

```
python preprocess.py --config preprocess.cfg
```

At this stage, your repository should look like this:

```
├── onmt		             	# Most of the heavy-lifting is done by onmt
├── experiments 	           	# Experiments are stored here
│	└── exp-1
│	│	├── data
│	│	│	├── data.train.0.pt
│	│	│	├── data.valid.0.pt
│	│	│	├── data.vocab.pt
│	│	│	├── preprocess-log.txt
├	│	├── gens
│	│	└── models
├── data						# Dataset is here
└── ...
```

# Training 

To train a hierarchical model on Rotowire you can run:

`python train.py --config train.cfg`

To train with different parameters than used in the paper, please refer to my comments in the config file, or check OpenNMT [train doc](http://opennmt.net/OpenNMT-py/options/train.html).

This config file runs the training for 100 000 steps, however we manually stopped the training at 30 000.

# Translating

Before translating, we average the last checkpoints. If you did anything different from previous commands, please change the first few line of `average-checkpoints.py`.

You can average checkpoints by running:

`python average-checkpoints.py --folder exp-1 --steps 31000 32000 33000 34000 35000`

Now you can simply translate the test input by running:

`python translate.py --config translate.cfg`

# Evaluation

RG, CS and CO metrics were originaly ([see here](https://github.com/harvardnlp/data2text)) coded in Lua and python2.
Because of compatibility issues with modern hardware, I have re-implemented RG
in PyTorch and python3:

Follow instructions at: [KaijuML/rotowire-rg-metric](https://github.com/KaijuML/rotowire-rg-metric). 

You can evaluate the BLEU score using [SacreBLEU](https://github.com/mjpost/sacreBLEU)
from [Post, 2018](aclweb.org/anthology/W18-6319). 
See the repo for installation, it should be a breeze with pip.

You can get the BLEU score by running:

`cat experiments/exp-1/gens/test/predictions.txt | sacrebleu --force data/test_output.txt`

(Note that --force is not required as it doesn't change the score computation, 
it just suppresses a warning because this dataset is already tokenized)

Alternatively you can use any prefered method for BLEU computation. 
I have also checked scoring models with [NLTK](aclweb.org/anthology/W18-6319) and scores were virtually the same.
