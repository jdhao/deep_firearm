# Introduction

This is the code our ICPR2018 paper "DeepFirearm: Learning Discriminative Feature
Representation for Fine-grained Firearm Retrieval".


# How to run the code

## Package info

The code is written using the PyTorch version 0.3.0. So In order to run this code, you may
install version 0.3.0 of PyTorch or adapt it to the newer version of PyTorch.

## Instructions for dataset download

You can download the dataset from [here](http://forensics.idealtest.org/Firearm14k/). Two separate data
are used for the experiment. One is for classification training, and the other is for the retrieva
training. After downloading this dataset, extract it under the folder `data` using the following command:

```
tar -zxvf firearm-train-val.tar.gz -C data/ # for the classification data
```

```
tar -zxvf firearm-dataset.tar.gz -C data/ # for the retrieval data
```

## Train the classification model

In order to train the classification model, run the following command:

```
python train_cls.py
```

## Train the retrieval model after classfication

To get better retrieval performance, we further train the model using retrieval task based on the classification
model. To train the model, use the following command:

```
python train_retr_from_cls.py
```

## Benchmark on test set

To check the model's performance on test set, run the following command:

```
python benchmark_on_test.py
```

It will show both the mAP and rank-k accuracy for different feature dimensions.

# Citation information

If you use this dataset or use our code, please cite the following work:

```
@INPROCEEDINGS{HJD2018DFLD,
author={J. Hao and J. Dong and W. Wang and T. Tan},
booktitle={2018 24th International Conference on Pattern Recognition (ICPR)},
title={DeepFirearm: Learning Discriminative Feature Representation for Fine-grained Firearm Retrieval},
year={2018},
volume={},
number={},
pages={3335-3340},
keywords={feature extraction;feedforward neural nets;image classification;image representation;image retrieval;learning (artificial intelligence);convolutional neural networks;single margin contrastive loss;firearm images;double margin contrastive loss;negative image pairs;positive image pairs;fine-grained recognition;Firearm 14k;image retrieval techniques;social media;fine-grained Firearm retrieval;discriminative feature representation;Training;Task analysis;Labeling;Correlation;Image retrieval;Forensics;Convolutional neural networks},
doi={10.1109/ICPR.2018.8545529},
ISSN={1051-4651},
month={Aug},}
```
