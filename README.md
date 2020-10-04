# Modality-Transferable Emotion Embeddings for Low-Resource Multimodal Emotion Recognition

<img src="img/pytorch-logo-dark.png" width="10%"/> [![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/) [![CC BY 4.0][cc-by-shield]][cc-by]


<img align="right" src="img/HKUST.jpg" width="15%"/>

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

Paper accepted at the [AACL-IJCNLP 2020](http://www.aacl2020.org/):

[Modality-Transferable Emotion Embeddings for Low-Resource Multimodal Emotion Recognition](https://arxiv.org/abs/2009.09629) 
(Wenliang Dai, Zihan Liu, Tiezheng Yu, Pascale Fung)

## Abstract

Despite the recent achievements made in the multi-modal emotion recognition task, two problems still exist and have not been well investigated: 1) the relationship between different emotion categories are not utilized, which leads to sub-optimal performance; and 2) current models fail to cope well with low-resource emotions, especially for unseen emotions. In this paper, we propose a modality-transferable model with emotion embeddings to tackle the aforementioned issues. We use pre-trained word embeddings to represent emotion categories for textual data. Then, two mapping functions are learned to transfer these embeddings into visual and acoustic spaces. For each modality, the model calculates the representation distance between the input sequence and target emotions and makes predictions based on the distances. By doing so, our model can directly adapt to the unseen emotions in any modality since we have their pre-trained embeddings and modality mapping functions. Experiments show that our model achieves state-of-the-art performance on most of the emotion categories. In addition, our model also outperforms existing baselines in the zero-shot and few-shot scenarios for unseen emotions.

## Dataset

We use the pre-processed features from the [CMU-Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK).

Or you can directly download the data from [here](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/).

## Preparation for running
1. Create a new folder named ***data*** at the root of this project

2. Download Emotion Embeddings from [here](https://drive.google.com/file/d/1RXdPUv03DIA7vO_RXTfBQ4RLQ6el0MFf/view?usp=sharing), and then put it in the $data$ folder.

3. Download data
    - For a quick run
      - Just download our saved `torch.utils.data.dataset.Dataset` datasets from [here](https://drive.google.com/file/d/17561ChjBP3psNypLMNcX7AohZqcTd4wI/view?usp=sharing), unzip it at the root of this project.
    - For a normal run
      - Download the data from [here](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/)
      - Check the ***data_folder_structure.txt*** file, which shows the structure about how to organize data files
      - Put data files correspondingly

4. Good to go!

## Command examples

Training our model on the CMU-MOSEI dataset

```console
python main.py --cuda=0 -bs=64 -lr=1e-3 -ep=100 --model=eea --data-folder=./data/cmu-mosei/seq_length_20/data --data-seq-len=20 --dataset=mosei_emo --loss=ce --clip=1.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav --patience=5 --aligned -bi --num-layers=2 --dropout=0.15
```

Training our model on the IEMOCAP dataset

```console
python main.py --cuda=0 -bs=64 -lr=1e-3 -ep=100 --model=eea --data-folder=./data/iemocap/seq_length_20/data --data-seq-len=20 --dataset=iemocap --loss=ce --clip=1.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav --patience=5 --aligned -bi --num-layers=2 --dropout=0.15
```

Training a early fusion lstm baseline

```console
python main.py --cuda=0 -bs=64 -lr=1e-3 -ep=100 --model=rnn --fusion=ef --data-folder=./data/iemocap/seq_length_20/data --data-seq-len=20 --dataset=iemocap --loss=ce --clip=1.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav --patience=5 --aligned -bi --num-layers=2 --dropout=0.15
```
