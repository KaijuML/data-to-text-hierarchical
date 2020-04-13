# Data-to-Text-Hierarchical [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-hierarchical-model-for-data-to-text/data-to-text-generation-on-rotowire)](https://paperswithcode.com/sota/data-to-text-generation-on-rotowire?p=a-hierarchical-model-for-data-to-text)

Code for [A Hierarchical Model for Data-to-Text Generation](https://arxiv.org/abs/1912.10011) (Rebuffel, Soulier, Scoutheeten, Gallinari; ECIR 2020); most of this code is based on [OpenNMT](https://github.com/OpenNMT/OpenNMT-py).

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

Evaluation metrics can be download at the [orignial repo](https://github.com/harvardnlp/data2text). You can follow instructions there to get the CS, RG and CO scores. Please note that I had issue running the code (mostly due to unmaintained torch repo). If it's not me, and you also run in some issues let me know and I'll try to help.

You can evaluate the BLEU score using [SacreBLEU](https://github.com/mjpost/sacreBLEU) from [Post, 2018](aclweb.org/anthology/W18-6319). See the repo for installation, it should be a breeze with pip.

You can get the BLEU score by running:

`cat experiments/exp-1/gens/test/predictions.txt | sacrebleu --force data/test_output.txt`

(Note that --force is not required as it doesn't change the score computation, it just suppresses a warning because this dataset is always tokenized)

Alternatively you can use any prefered method for BLEU computation. I have also checked scoring models with [NLTK](aclweb.org/anthology/W18-6319) and scores were virtually the same.
