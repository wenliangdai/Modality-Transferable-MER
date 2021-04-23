# Modality-Transferable Emotion Embeddings for Low-Resource Multimodal Emotion Recognition

<img src="img/pytorch-logo-dark.png" width="10%"/> [![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/) [![CC BY 4.0][cc-by-shield]][cc-by]


<img align="right" src="img/HKUST.jpg" width="15%"/>

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

Paper accepted at the [AACL-IJCNLP 2020](https://www.aclweb.org/anthology/events/aacl-2020/):

**Modality-Transferable Emotion Embeddings for Low-Resource Multimodal Emotion Recognition**, by **[Wenliang Dai](https://wenliangdai.github.io/)**, [Zihan Liu](https://zliucr.github.io/), Tiezheng Yu, Pascale Fung.

[[ACL Anthology](https://www.aclweb.org/anthology/2020.aacl-main.30/)][[ArXiv](https://arxiv.org/abs/2009.09629)][[Semantic Scholar](https://www.semanticscholar.org/paper/Modality-Transferable-Emotion-Embeddings-for-Dai-Liu/22201ea212a41fa5b1c24ba3e1f596d73fc13584)]

If your work is inspired by our paper, or you use any code snippets in this repo, please cite this paper, the BibTex is shown below:

<pre>
@inproceedings{dai-etal-2020-modality,
    title = "Modality-Transferable Emotion Embeddings for Low-Resource Multimodal Emotion Recognition",
    author = "Dai, Wenliang  and
      Liu, Zihan  and
      Yu, Tiezheng  and
      Fung, Pascale",
    booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.aacl-main.30",
    pages = "269--280",
    abstract = "Despite the recent achievements made in the multi-modal emotion recognition task, two problems still exist and have not been well investigated: 1) the relationship between different emotion categories are not utilized, which leads to sub-optimal performance; and 2) current models fail to cope well with low-resource emotions, especially for unseen emotions. In this paper, we propose a modality-transferable model with emotion embeddings to tackle the aforementioned issues. We use pre-trained word embeddings to represent emotion categories for textual data. Then, two mapping functions are learned to transfer these embeddings into visual and acoustic spaces. For each modality, the model calculates the representation distance between the input sequence and target emotions and makes predictions based on the distances. By doing so, our model can directly adapt to the unseen emotions in any modality since we have their pre-trained embeddings and modality mapping functions. Experiments show that our model achieves state-of-the-art performance on most of the emotion categories. Besides, our model also outperforms existing baselines in the zero-shot and few-shot scenarios for unseen emotions.",
}
</pre>

## Abstract

Despite the recent achievements made in the multi-modal emotion recognition task, two problems still exist and have not been well investigated: 1) the relationship between different emotion categories are not utilized, which leads to sub-optimal performance; and 2) current models fail to cope well with low-resource emotions, especially for unseen emotions. In this paper, we propose a modality-transferable model with emotion embeddings to tackle the aforementioned issues. We use pre-trained word embeddings to represent emotion categories for textual data. Then, two mapping functions are learned to transfer these embeddings into visual and acoustic spaces. For each modality, the model calculates the representation distance between the input sequence and target emotions and makes predictions based on the distances. By doing so, our model can directly adapt to the unseen emotions in any modality since we have their pre-trained embeddings and modality mapping functions. Experiments show that our model achieves state-of-the-art performance on most of the emotion categories. In addition, our model also outperforms existing baselines in the zero-shot and few-shot scenarios for unseen emotions.

## Dataset

We use the pre-processed features from the [CMU-Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK).

Or you can directly download the data from [here](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/).

## Preparation for running
1. Create a new folder named **data** at the root of this project

2. Download Emotion Embeddings from [here](https://drive.google.com/file/d/1RXdPUv03DIA7vO_RXTfBQ4RLQ6el0MFf/view?usp=sharing), and then put it in the $data$ folder.

3. Download data
    - For a quick run
      - Just download our saved `torch.utils.data.dataset.Dataset` datasets from [here](https://drive.google.com/file/d/17561ChjBP3psNypLMNcX7AohZqcTd4wI/view?usp=sharing), unzip it at the root of this project.
    - For a normal run
      - Download the data from [here](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/)
      - Check the **data_folder_structure.txt** file, which shows the structure about how to organize data files
      - Put data files correspondingly

4. Good to go!

## Command line arguments and examples

```
usage: main.py [-h] -bs BATCH_SIZE -lr LEARNING_RATE [-wd WEIGHT_DECAY] -ep
               EPOCHS [-es EARLY_STOP] [-cu CUDA] [-mo MODEL] [-fu FUSION]
               [-cl CLIP] [-sc] [-se SEED] [-pa PATIENCE] [-ez] [--loss LOSS]
               [--optim OPTIM] [--threshold THRESHOLD] [--verbose]
               [-mod MODALITIES] [--valid] [--test] [--dataset DATASET]
               [--aligned] [--data-seq-len DATA_SEQ_LEN]
               [--data-folder DATA_FOLDER] [--glove-emo-path GLOVE_EMO_PATH]
               [--cap] [--iemocap4] [--iemocap9] [--zsl ZSL]
               [--zsl-test ZSL_TEST] [--fsl FSL] [--ckpt CKPT] [-dr DROPOUT]
               [-nl NUM_LAYERS] [-hs HIDDEN_SIZE]
               [-hss HIDDEN_SIZES [HIDDEN_SIZES ...]] [-bi] [--gru]
               [--hidden-dim HIDDEN_DIM]

Multimodal Emotion Recognition

optional arguments:
  -h, --help            show this help message and exit
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate
  -wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        Weight decay
  -ep EPOCHS, --epochs EPOCHS
                        Number of epochs
  -es EARLY_STOP, --early-stop EARLY_STOP
                        Early stop
  -cu CUDA, --cuda CUDA
                        Cude device number
  -mo MODEL, --model MODEL
                        Model type: mult/rnn/transformer/eea
  -fu FUSION, --fusion FUSION
                        Modality fusion type: ef/lf
  -cl CLIP, --clip CLIP
                        Use clip to gradients
  -sc, --scheduler      Use scheduler to optimizer
  -se SEED, --seed SEED
                        Random seed
  -pa PATIENCE, --patience PATIENCE
                        Patience of the scheduler
  -ez, --exclude-zero   Exclude zero in evaluation
  --loss LOSS           loss function: l1/mse/ce/bce
  --optim OPTIM         optimizer function: adam/sgd
  --threshold THRESHOLD
                        Threshold of for multi-label emotion recognition
  --verbose             Verbose mode to print more logs
  -mod MODALITIES, --modalities MODALITIES
                        What modalities to use
  --valid               Valid mode
  --test                Test mode
  --dataset DATASET     Dataset to use
  --aligned             Aligned experiment or not
  --data-seq-len DATA_SEQ_LEN
                        Data sequence length
  --data-folder DATA_FOLDER
                        path for storing the dataset
  --glove-emo-path GLOVE_EMO_PATH
  --cap                 Capitalize the first letter of emotion words
  --iemocap4            Only use 4 emtions in IEMOCAP
  --iemocap9            Only use 9 emtions in IEMOCAP
  --zsl ZSL             Do zero shot learning on which emotion (index)
  --zsl-test ZSL_TEST   Notify which emotion was zsl before
  --fsl FSL             Do few shot learning on which emotion (index)
  --ckpt CKPT
  -dr DROPOUT, --dropout DROPOUT
                        dropout
  -nl NUM_LAYERS, --num-layers NUM_LAYERS
                        num of layers of LSTM
  -hs HIDDEN_SIZE, --hidden-size HIDDEN_SIZE
                        hidden vector size of LSTM
  -hss HIDDEN_SIZES [HIDDEN_SIZES ...], --hidden-sizes HIDDEN_SIZES [HIDDEN_SIZES ...]
                        hidden vector size of LSTM
  -bi, --bidirectional  Use Bi-LSTM
  --gru                 Use GRU rather than LSTM
  --hidden-dim HIDDEN_DIM
                        Transformers hidden unit size
```

## Run the code

`main.py` is the entry file of the whole project, use corresponding CLIs for different purposes.

### Training

Training the model on the CMU-MOSEI dataset

```console
python main.py --cuda=0 -bs=64 -lr=1e-3 -ep=100 --model=eea -bi --hidden-sizes 300 200 100 --num-layers=2 --dropout=0.15 --data-folder=./data/cmu-mosei/ --data-seq-len=20 --dataset=mosei_emo --aligned --loss=bce --clip=1.0 --early-stop=8 -mod=tav --patience=5   
```

Training the model on the IEMOCAP dataset

```console
python main.py --cuda=0 -bs=64 -lr=1e-3 -ep=100 --model=eea --data-folder=./data/iemocap/ --data-seq-len=50 --dataset=iemocap --loss=bce --clip=1.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav --patience=5 --aligned -bi --num-layers=2 --dropout=0.15
```

Training a early fusion lstm baseline

```console
python main.py --cuda=0 -bs=64 -lr=1e-3 -ep=100 --model=rnn --fusion=ef --data-folder=./data/iemocap/ --data-seq-len=50 --dataset=iemocap --loss=bce --clip=1.0 --early-stop=8 --hidden-sizes 300 200 100 -mod=tav --patience=5 --aligned -bi --num-layers=2 --dropout=0.15
```

### Validating and testing

If you only want to do a validation or testing on a trained model, you can add a `--valid` or `--test` flag to the original command, and also include `--ckpt=[PathToSavedCheckpoint]` to indicate the path of the trained model.

### Zero-shot learning (ZSL)

Add a `--zsl=[EmotionIndex]` cli to the original training command, in which the EmotionIndex is the index of the emotion category that you want to do zero-shot on. As mentioned in the paper, due to different strategies for CMU-MOSEI and IEMOCAP datasets, `--zsl=[EmotionIndex]` has slightly different meaning for them, we list the correct cli here:

For CMU-MOSEI (ZSL emotion data will be removed from the training data),

* `--zsl=0`, do ZSL on **anger**
* `--zsl=1`, do ZSL on **disgust**
* `--zsl=2`, do ZSL on **fear**
* `--zsl=3`, do ZSL on **happy**
* `--zsl=4`, do ZSL on **sad**
* `--zsl=5`, do ZSL on **surprise**

For IEMOCAP (the training data remains unchanged, as ZSL emotion is from extra low-resource data),
* `--zsl=1`, do ZSL on **excited**
* `--zsl=4`, do ZSL on **surprised**
* `--zsl=5`, do ZSL on **frustrated**


### Few-shot learning (FSL)

For few-shot learning, the logic is similar to ZSL, just use `--fsl=[EmotionIndex]`

## Requirements

1. Python 3.6 +
2. PyTorch 1.4 +
3. Nvidia GTX 1080Ti GPU (or more advanced)
